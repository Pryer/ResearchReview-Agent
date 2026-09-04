"""论文检索工具。

统一论文检索入口，协调多个客户端（arXiv / Semantic Scholar / OpenAlex / Crossref）。
支持双通道混合检索（Relevance + Recency），兼顾经典高引与最新前沿论文的覆盖率。
"""

from __future__ import annotations

import math
import re
from contextvars import ContextVar
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, List, Optional

import requests

_CLIENT_DIAGNOSTICS: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "search_client_diagnostics", default=None
)


def record_client_diagnostic(
    outcome: str,
    *,
    error_code: str | None = None,
    message: str | None = None,
) -> None:
    """供客户端在保留 ``[]`` 返回兼容性的同时传递失败原因。"""
    events = _CLIENT_DIAGNOSTICS.get()
    if events is not None:
        # 部分 client 只保留 ``requests.HTTPError`` 的字符串；在事件边界
        # 再识别 401/403/429，避免认证或限流被泛化成 API_ERROR。
        if outcome == "api_failed" and message:
            inferred, inferred_code = _classify_search_exception(RuntimeError(message))
            if inferred != "api_failed":
                outcome = inferred
                error_code = error_code if error_code and error_code != "API_ERROR" else inferred_code
        events.append({"outcome": outcome, "error_code": error_code, "message": message})

from app.core.logger import get_logger
from app.core.config import get_settings
from app.core.source_capabilities import compatible_sources, is_chinese_query
from app.schemas.paper_schema import PaperMetadata, SourceDiagnostic

logger = get_logger(__name__)

# 双通道检索的默认配额比例：70% relevance, 30% recency
_RELEVANCE_RATIO = 0.7
_RECENCY_RATIO = 0.3


def _is_chinese_keyword(keyword: str) -> bool:
    """兼容旧名称；实现位于来源能力模块。"""
    return is_chinese_query(keyword)


def select_sources_by_language(keyword: str, configured_sources: List[str]) -> List[str]:
    """兼容旧 API；只做来源能力过滤，不再编码来源优先级策略。"""
    return compatible_sources(keyword, configured_sources)


def _diagnostic_for_outcome(
    source: str,
    outcome: str,
    *,
    returned_count: int = 0,
    error_code: str | None = None,
    message: str | None = None,
) -> SourceDiagnostic:
    """创建来源诊断，并保留旧 ``status`` 字段供现有消费者使用。"""
    status = {
        "success_with_results": "success",
        "success_empty": "empty",
        "query_not_adapted": "skipped",
        "rate_limited": "failed",
        "timeout": "failed",
        "authentication_failed": "failed",
        "api_failed": "failed",
        "human_action_required": "human_action_required",
        "skipped": "skipped",
    }.get(outcome, "failed")
    return SourceDiagnostic(
        source=source,
        status=status,
        outcome=outcome,
        failure_category=outcome if outcome not in {"success_with_results", "success_empty"} else None,
        returned_count=returned_count,
        error_code=error_code,
        message=message,
    )


def _classify_search_exception(exc: BaseException) -> tuple[str, str]:
    """将客户端抛出的异常映射为稳定 outcome 与保留错误码。"""
    error_code = str(getattr(exc, "error_code", "") or "").upper()
    text = str(exc)
    status_code = getattr(getattr(exc, "response", None), "status_code", None)
    status_code = status_code or getattr(exc, "status_code", None)
    if status_code is None:
        match = re.search(r"(?:HTTP(?:Error)?|status(?:_code)?|status)[ =:_-]*(401|403|429)", text, re.I)
        status_code = int(match.group(1)) if match else None
    if error_code in {"RATE_LIMITED", "RATE_LIMIT", "TOO_MANY_REQUESTS"} or status_code == 429:
        return "rate_limited", error_code or "HTTP_429"
    if isinstance(exc, (requests.Timeout, TimeoutError)) or "timeout" in text.lower() or "超时" in text:
        return "timeout", error_code or "TIMEOUT"
    if status_code in {401, 403} or error_code in {"UNAUTHORIZED", "FORBIDDEN", "AUTHENTICATION_FAILED", "LOGIN_REQUIRED"}:
        return "authentication_failed", error_code or f"HTTP_{status_code}"
    return "api_failed", error_code or "API_ERROR"


def _append_language_diagnostic(diagnostics: Optional[List[SourceDiagnostic]], source: str) -> None:
    if diagnostics is not None:
        diagnostics.append(_diagnostic_for_outcome(
            source, "query_not_adapted", error_code="INCOMPATIBLE_QUERY_LANGUAGE",
            message="该数据源不支持当前检索语言，已跳过；可改用匹配语言的关键词或其他来源。",
        ))


def _invoke_client(
    fn: Callable[..., List[PaperMetadata]],
    query: str,
    start_year: int,
    end_year: int,
    quota: int,
    sort_by: str,
) -> tuple[List[PaperMetadata], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    token = _CLIENT_DIAGNOSTICS.set(events)
    try:
        return fn(query, start_year, end_year, quota, sort_by), events
    except BaseException as exc:
        # 客户端为保持旧返回类型可能吞掉异常；若仍抛出，则把已记录的
        # 细分事件挂到异常上，调度层保留 429/超时等原始分类。
        setattr(exc, "_client_diagnostics", events)
        raise
    finally:
        _CLIENT_DIAGNOSTICS.reset(token)


def search_papers(
    query: str,
    start_year: int,
    end_year: int,
    max_results: int,
    sources: List[str] | None = None,
    diagnostics: Optional[List[SourceDiagnostic]] = None,
    should_cancel: Callable[[], bool] | None = None,
    enable_dual_channel: bool = True,
) -> List[PaperMetadata]:
    """从多个开放数据源检索论文。

    Args:
        query: 检索关键词。
        start_year: 起始年份。
        end_year: 结束年份。
        max_results: 每个数据源的最大返回数。
        sources: 数据源列表。
        diagnostics: 诊断信息收集列表。
        should_cancel: 取消回调。
        enable_dual_channel: 是否开启双通道混合检索（默认 True）。
            开启后，每个数据源同时以 relevance + date 两种排序方式检索，
            合并去重，兼顾经典高引论文与最新前沿论文。

    Returns:
        合并后的论文元数据列表。
    """
    if sources is None:
        sources = ["arxiv", "semantic_scholar", "openalex", "crossref"]

    # 支持 sort_by 的数据源
    _DUAL_CHANNEL_SOURCES = {"arxiv", "semantic_scholar", "openalex", "crossref"}

    client_map = {
        "arxiv": _search_arxiv,
        "semantic_scholar": _search_semantic_scholar,
        "openalex": _search_openalex,
        "crossref": _search_crossref,
        "cnki": _search_cnki,
    }

    all_papers: List[PaperMetadata] = []
    valid_tasks: list[tuple[str, object, str, int]] = []  # (source, fn, sort_by, quota)

    for source in sources:
        client_fn = client_map.get(source)
        if not client_fn:
            logger.warning("Unknown source: %s", source)
            if diagnostics is not None:
                diagnostics.append(_diagnostic_for_outcome(
                    source, "api_failed", error_code="UNKNOWN_SOURCE",
                    message=f"未注册的数据源: {source}",
                ))
            continue
        # CNKI 只检索中文关键词；若调用方显式传入不兼容来源，记录诊断。
        if source == "cnki" and not is_chinese_query(query):
            logger.debug("CNKI skipped for non-Chinese query: %s", query[:60])
            _append_language_diagnostic(diagnostics, source)
            continue

        # arXiv 只检索英文关键词：中文查询只会返回 0/噪声，白白占用配额。
        if source == "arxiv" and is_chinese_query(query):
            logger.debug("arXiv skipped for Chinese query: %s", query[:60])
            _append_language_diagnostic(diagnostics, source)
            continue

        # Semantic Scholar 对中文检索式同样基本返回 0/噪声，一并拦截。
        if source == "semantic_scholar" and is_chinese_query(query):
            logger.debug("Semantic Scholar skipped for Chinese query: %s", query[:60])
            _append_language_diagnostic(diagnostics, source)
            continue

        # 双通道分配
        if enable_dual_channel and source in _DUAL_CHANNEL_SOURCES:
            relevance_quota = max(5, math.ceil(max_results * _RELEVANCE_RATIO))
            recency_quota = max(5, math.ceil(max_results * _RECENCY_RATIO))
            valid_tasks.append((source, client_fn, "relevance", relevance_quota))
            valid_tasks.append((source, client_fn, "date", recency_quota))
            logger.info(
                "[%s] Dual-channel: relevance=%d, recency=%d",
                source, relevance_quota, recency_quota,
            )
        else:
            # CNKI 或关闭双通道时，单通道 relevance
            valid_tasks.append((source, client_fn, "relevance", max_results))

    if should_cancel and should_cancel():
        raise InterruptedError("论文检索已取消")

    worker_count = max(
        1,
        min(get_settings().search_source_max_workers, len(valid_tasks) or 1),
    )
    results_by_key: dict[str, list[PaperMetadata]] = {}
    diagnostics_by_source: dict[str, SourceDiagnostic] = {}

    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="paper-search") as executor:
        future_to_meta = {}
        for source, fn, sort_by, quota in valid_tasks:
            task_key = f"{source}:{sort_by}"
            future = executor.submit(_invoke_client, fn, query, start_year, end_year, quota, sort_by)
            future_to_meta[future] = (source, sort_by, task_key)

        for future in as_completed(future_to_meta):
            if should_cancel and should_cancel():
                for pending in future_to_meta:
                    pending.cancel()
                raise InterruptedError("论文检索已取消")
            source, sort_by, task_key = future_to_meta[future]
            try:
                papers, client_events = future.result()
                results_by_key[task_key] = papers
                logger.info("[%s] (sort=%s) Found %d papers", source, sort_by, len(papers))
                if client_events and not papers:
                    event = client_events[-1]
                    outcome = str(event.get("outcome") or "api_failed")
                    diagnostics_by_source[source] = _diagnostic_for_outcome(
                        source, outcome, error_code=event.get("error_code"),
                        message=event.get("message"),
                    )

                # 诊断信息：合并同一数据源的双通道结果。空结果仍明确标为
                # success_empty，不能被后续流程误认为健康有结果。
                prev = diagnostics_by_source.get(source)
                new_count = len(papers) + (prev.returned_count if prev else 0)
                if new_count > 0:
                    # 即使本通道有结果，也保留同源另一通道报告的限流/API事件。
                    if prev and prev.outcome not in {"success_empty", "success_with_results"}:
                        diagnostics_by_source[source] = prev.model_copy(update={"returned_count": new_count})
                    else:
                        diagnostics_by_source[source] = _diagnostic_for_outcome(
                            source, "success_with_results", returned_count=new_count,
                        )
                elif prev is None or prev.outcome in {"success_empty", "success_with_results"}:
                    diagnostics_by_source[source] = _diagnostic_for_outcome(
                        source, "success_empty", returned_count=0,
                    )
            except Exception as e:
                logger.warning("[%s] Search failed (sort=%s): %s", source, sort_by, e)
                client_events = getattr(e, "_client_diagnostics", None) or []
                if client_events:
                    event = client_events[-1]
                    outcome = str(event.get("outcome") or "api_failed")
                    error_code = event.get("error_code") or "API_ERROR"
                    message = event.get("message") or f"{type(e).__name__}: {e}"
                else:
                    outcome, error_code = _classify_search_exception(e)
                    message = f"{type(e).__name__}: {e}"
                previous = diagnostics_by_source.get(source)
                # 保留已成功通道的结果，同时将故障细节留在诊断中；双通道
                # 某一排序失败不应覆盖同源另一排序已经得到的证据。
                if previous is None or not previous.returned_count:
                    diagnostics_by_source[source] = _diagnostic_for_outcome(
                        source, outcome, error_code=error_code,
                        message=message,
                    )
                elif previous.message is None:
                    diagnostics_by_source[source] = previous.model_copy(update={
                        "error_code": error_code,
                        "failure_category": outcome,
                        "message": f"{type(e).__name__}: {e}",
                    })

    # 按配置源顺序合并（先 relevance 后 date，保证去重优先经典论文）
    merged_order = []
    for source in sources:
        for sort_by in ("relevance", "date"):
            key = f"{source}:{sort_by}"
            merged_order.extend(results_by_key.get(key, []))
    if diagnostics is not None:
        for source in sources:
            if source in diagnostics_by_source:
                diagnostics.append(diagnostics_by_source[source])

    result = merge_search_results(merged_order)
    if enable_dual_channel:
        logger.info(
            "Dual-channel merge: %d total papers (from %d raw results across %d tasks)",
            len(result), len(merged_order), len(valid_tasks),
        )
    return result


def merge_search_results(papers: List[PaperMetadata]) -> List[PaperMetadata]:
    """合并不同数据源的结果（按 DOI / arXiv ID 去重）。

    同一篇论文保留信息更完整的版本。
    """
    # 简单的字典去重
    seen: dict[str, PaperMetadata] = {}

    def _key(p: PaperMetadata) -> str:
        return (
            (p.doi or "").lower()
            or (p.arxiv_id or "").lower()
            or p.paper_id
        )

    for source_paper in papers:
        p = source_paper.model_copy(deep=True)
        k = _key(p)
        if not k:
            continue
        if k in seen:
            existing = seen[k]
            # 合并信息
            if not existing.title and p.title:
                existing.title = p.title
            if not existing.authors and p.authors:
                existing.authors = p.authors
            if not existing.year and p.year:
                existing.year = p.year
            if not existing.venue and p.venue:
                existing.venue = p.venue
            if not existing.abstract and p.abstract:
                existing.abstract = p.abstract
            if not existing.pdf_url and p.pdf_url:
                existing.pdf_url = p.pdf_url
            if not existing.keywords and p.keywords:
                existing.keywords = p.keywords
            if (p.citation_count or 0) > (existing.citation_count or 0):
                existing.citation_count = p.citation_count
        else:
            seen[k] = p

    return list(seen.values())


def filter_by_year(
    papers: List[PaperMetadata],
    start_year: int,
    end_year: int,
    *,
    retain_unknown: bool = False,
) -> List[PaperMetadata]:
    """按年份过滤。

    默认严格排除年份未知的论文；检索召回阶段若希望保留待补全记录，必须显式
    传入 ``retain_unknown=True``。
    """
    return [
        p for p in papers
        if (p.year is None and retain_unknown)
        or (p.year is not None and start_year <= p.year <= end_year)
    ]


# ============================================================
# 内部：调用各客户端（统一增加 sort_by 参数传递）
# ============================================================
def _search_arxiv(
    query: str, start_year: int, end_year: int, max_results: int,
    sort_by: str = "relevance",
) -> List[PaperMetadata]:
    """调用 arXiv 客户端。"""
    from app.clients.arxiv_client import search_arxiv
    return search_arxiv(query, start_year, end_year, max_results, sort_by=sort_by)


def _search_semantic_scholar(
    query: str, start_year: int, end_year: int, max_results: int,
    sort_by: str = "relevance",
) -> List[PaperMetadata]:
    """调用 Semantic Scholar 客户端。"""
    from app.clients.semantic_scholar_client import search_semantic_scholar
    return search_semantic_scholar(query, start_year, end_year, max_results, sort_by=sort_by)


def _search_openalex(
    query: str, start_year: int, end_year: int, max_results: int,
    sort_by: str = "relevance",
) -> List[PaperMetadata]:
    """调用 OpenAlex 客户端。"""
    from app.clients.openalex_client import search_openalex
    return search_openalex(query, start_year, end_year, max_results, sort_by=sort_by)


def _search_crossref(
    query: str, start_year: int, end_year: int, max_results: int,
    sort_by: str = "relevance",
) -> List[PaperMetadata]:
    """调用 Crossref 客户端（中文期刊论文覆盖较好）。"""
    from app.clients.crossref_client import search_crossref
    return search_crossref(query, start_year, end_year, max_results, sort_by=sort_by)


# CNKI 站内检索对“综述”类泛化后缀非常敏感：
# “少样本动作识别综述”只能命中 2 条，去掉后缀后召回显著扩大。
_CNKI_GENERIC_SUFFIX_RE = re.compile(r"(?:研究综述|文献综述|综述|述评)$")


def _search_cnki(
    query: str, start_year: int, end_year: int, max_results: int,
    sort_by: str = "relevance",
) -> List[PaperMetadata]:
    """调用 CNKI（知网）Selenium 客户端（中文论文主力源）。"""
    from app.clients.cnki_client import search_cnki

    cleaned_query = _CNKI_GENERIC_SUFFIX_RE.sub("", str(query or "").strip()).strip()
    if not cleaned_query:
        cleaned_query = str(query or "").strip()
    if cleaned_query != str(query or "").strip():
        logger.info("[cnki] generic suffix stripped: %s -> %s", query, cleaned_query)
    return search_cnki(cleaned_query, start_year, end_year, max_results)

