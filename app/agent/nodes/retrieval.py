"""检索与排序相关节点。"""

from __future__ import annotations

import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from app.agent.decorators import node, optional, provides, requires
from app.agent.nodes.base import (
    _compact_debug_value,
    _latest_step,
    _needs_current_time_tool,
    _paper_debug_item,
    _paper_identity_key,
    _preview_text,
    _select_batch_first_keywords,
    _select_branch_diverse_keywords,
    _select_search_keywords,
    _summarize_papers,
    append_step,
)
from app.core.config import get_settings
from app.core.logger import get_logger
from app.core.source_capabilities import source_name
from app.schemas.paper_schema import SourceDiagnostic

if TYPE_CHECKING:
    from app.agent.state import ResearchAgentState

logger = get_logger(__name__)


def _keyword_tokens(keyword: str) -> set[str]:
    """检索词的内容词元集合：英文取 2+ 字母词，中文取整段连续汉字。"""
    return set(re.findall(r"[a-z]{2,}|[\u4e00-\u9fff]+", str(keyword or "").lower()))


def _drop_redundant_keywords(
    keywords: list[str], topic_anchor: str = ""
) -> tuple[list[str], list[str]]:
    """确定性剔除同批内近重复检索词，返回 (保留, 剔除)。

    若当前检索词的内容词元与已保留词重叠 >=80%，视为近重复并剔除；
    具体判断保持当前词作为分母，避免把仅增加限定表达的互补宽查询误判为重复。
    输入顺序由调用方按查询角色排序，topic_recall 会优先于 scope_precision。
    主题锚点不参与剔除。
    """
    kept: list[str] = []
    kept_tokens: list[set[str]] = []
    dropped: list[str] = []
    anchor = str(topic_anchor or "").strip()
    for keyword in keywords:
        text = str(keyword)
        toks = _keyword_tokens(text)
        if not toks or (anchor and anchor in text):
            kept.append(text)
            kept_tokens.append(toks)
            continue

        redundant = any(
            previous and len(toks & previous) / len(toks) >= 0.8
            for previous in kept_tokens
        )
        if redundant:
            dropped.append(text)
        else:
            kept.append(text)
            kept_tokens.append(toks)
    return kept, dropped


def _normalized_window_key(keyword: str, start_year: int, end_year: int) -> str:
    """词元排序后的窗口键：词序不同的等价查询共享同一键。"""
    tokens = sorted(_keyword_tokens(keyword))
    base = " ".join(tokens) if tokens else str(keyword or "").strip().lower()
    return f"~{base}|{start_year}|{end_year}"


def _count_by_source(papers: list) -> dict[str, int]:
    """按检索源统计论文数量（来源缺失记为 unknown）。

    用于观测各源的真实贡献：返回量大不等于贡献大，只有最终进入
    候选池/入选集合的论文才算数，为裁撤低效源与调整双语配额
    提供依据。
    """
    counts: dict[str, int] = {}
    for paper in papers:
        record = paper if isinstance(paper, dict) else paper.model_dump()
        name = source_name(record) or "unknown"
        counts[name] = counts.get(name, 0) + 1
    return counts


@node(name="search", category="retrieval", description="调用论文检索工具，获取候选论文")
@requires("topic", "keywords", "start_year", "end_year", "max_papers", "retrieval_target")
@provides(
    "candidate_papers", "searched_keywords", "searched_query_windows",
    "last_search_new_results", "source_diagnostics", "search_failed",
)
@optional("search_branches", "search_refinement_count")
def search_node(
    state: "ResearchAgentState",
    should_cancel=None,
) -> "ResearchAgentState":
    """调用论文检索工具，获取候选论文。"""
    t0 = time.time()
    # search_failed 由本节点唯一写入：进入时清零，失败时置位。
    # 调用方不再散落手动重置，标志始终反映最近一次检索的结果。
    state["search_failed"] = False
    try:
        from app.tools.search_papers import search_papers
        from app.tools.search_papers import merge_search_results
        from app.core.config import get_settings
        from app.utils.date_utils import default_year_range, current_year

        settings = get_settings()
        default_start, default_end = default_year_range(
            current_year(), settings.default_year_lookback
        )
        incremental_window = state.get("incremental_search_window") or {}
        search_start_year = int(
            incremental_window.get("start_year")
            or state.get("start_year")
            or default_start
        )
        search_end_year = int(
            incremental_window.get("end_year")
            or state.get("end_year")
            or default_end
        )
        raw_keywords = state.get("keywords") or [state.get("topic", "")]
        keywords = []
        keyword_roles = {
            str(query).strip().casefold(): str(role).strip()
            for query, role in (state.get("query_roles") or {}).items()
            if str(query).strip() and str(role).strip()
        }
        for item in raw_keywords:
            if isinstance(item, dict):
                keyword = str(item.get("keyword") or item.get("query") or "").strip()
                role = str(item.get("role") or item.get("query_role") or "").strip()
                if role and keyword:
                    keyword_roles[keyword.casefold()] = role
            else:
                keyword = str(item).strip()
            if keyword and keyword not in keywords:
                keywords.append(keyword)
        state["query_roles"] = keyword_roles
        scope_query_roles: dict[str, str] = {}
        for query in (state.get("scope_search_queries") or []):
            key = str(query).strip().casefold()
            if key:
                scope_query_roles[key] = "scope_precision"
        for query, role in (state.get("scope_query_roles") or {}).items():
            key = str(query).strip().casefold()
            role = str(role).strip()
            if not key or not role:
                continue
            # 主题召回是稳定的宽查询角色；不得被 scope 精确角色降级。
            if role == "topic_recall" or key not in scope_query_roles:
                scope_query_roles[key] = role
        for query, role in keyword_roles.items():
            if role == "topic_recall" or query not in scope_query_roles:
                scope_query_roles[query] = role
        core_keys = {
            str(item).strip().casefold()
            for item in (state.get("core_keywords") or [state.get("topic") or ""])
            if str(item).strip()
        }
        # 规划阶段会先把范围种子查询合并进 keywords；核心主题即使与
        # 范围查询文本相同，也必须保持 topic_recall 身份，不能被 scope
        # 精确角色覆盖，否则纯主题词不会获得首选排序和 CNKI 名额。
        for query in core_keys:
            scope_query_roles[query] = "topic_recall"
        state["scope_query_roles"] = scope_query_roles
        retrieval_target = state.get("retrieval_target") or state.get("max_papers") or settings.default_max_papers
        max_results_per_keyword = min(
            settings.max_results_per_keyword,
            max(settings.min_results_per_keyword, retrieval_target),
        )
        all_papers = []

        searched_windows = {
            str(item).lower() for item in (state.get("searched_query_windows") or [])
        }
        searched = {
            str(keyword).lower() for keyword in (state.get("searched_keywords") or [])
        } if not incremental_window and not searched_windows else set()
        unseen_keywords = [
            keyword for keyword in keywords
            if (
                str(keyword).lower() not in searched
                and f"{str(keyword).strip().lower()}|{search_start_year}|{search_end_year}"
                not in searched_windows
                # 词序不同的等价查询（词元排序键）也不重复检索
                and _normalized_window_key(
                    keyword, search_start_year, search_end_year
                ) not in searched_windows
            )
        ]
        keyword_batches = state.get("keyword_batches") or []
        # 核心词承担主题锚定，扩展词只用于补充召回；即使批次标记把扩展词
        # 排在前面，也不能让它们挤掉尚未检索的核心词。
        core_keys = {
            str(item).strip().casefold()
            for item in (state.get("core_keywords") or [state.get("topic") or ""])
            if str(item).strip()
        }
        core_unseen = [item for item in unseen_keywords if str(item).strip().casefold() in core_keys]
        expanded_unseen = [item for item in unseen_keywords if item not in core_unseen]
        ordered_unseen = [*core_unseen, *expanded_unseen]
        # scope 查询只是精确补充，不能在宽主题召回词之前占满本轮名额。
        ordered_unseen = sorted(
            enumerate(ordered_unseen),
            key=lambda pair: (
                0 if scope_query_roles.get(str(pair[1]).strip().casefold(), "topic_recall") == "topic_recall" else 1,
                pair[0],
            ),
        )
        ordered_unseen = [item for _, item in ordered_unseen]
        if keyword_batches:
            # 批次优先：exact 完整表达先检，broader/variant 外扩后检；
            # 批内仍用分支多样选择，未分批词排在批次之后兜底。
            search_keywords = _select_batch_first_keywords(
                ordered_unseen,
                keyword_batches,
                state.get("search_branches") or [],
                limit=settings.max_search_keywords,
            )
        else:
            search_keywords = _select_branch_diverse_keywords(
                ordered_unseen,
                state.get("search_branches") or [],
                limit=settings.max_search_keywords,
            )
        # 批次选择后再次稳定提升核心词，避免分支多样性覆盖主题核心。
        selected_core = [item for item in search_keywords if str(item).strip().casefold() in core_keys]
        selected_expanded = [item for item in search_keywords if item not in selected_core]
        search_keywords = [*selected_core, *selected_expanded][:settings.max_search_keywords]

        # 中英混杂检索词的确定性清洗：拆出中文段或丢弃，避免把
        # "survey 近三年…综述"这类词浪费在英文源上（英文源返回 0/噪声）。
        from app.core.source_capabilities import (
            is_generic_search_keyword,
            sanitize_search_keyword,
        )

        sanitized_keywords: list[str] = []
        dropped_malformed_keywords: list[str] = []
        topic_for_generic_check = str(state.get("topic") or "").strip()
        for keyword in search_keywords:
            cleaned = sanitize_search_keyword(keyword)
            if not cleaned:
                dropped_malformed_keywords.append(str(keyword))
                continue
            if cleaned != keyword:
                dropped_malformed_keywords.append(f"{keyword} -> {cleaned}")
            if is_generic_search_keyword(cleaned) and cleaned != topic_for_generic_check:
                dropped_malformed_keywords.append(f"{cleaned} (通用词，无区分力)")
                continue
            if cleaned not in sanitized_keywords:
                sanitized_keywords.append(cleaned)
        search_keywords = sanitized_keywords
        # CNKI 锚点判定依赖主题核心词：含主题词的纯中文词优先升级为
        # CNKI 检索词，避免拆分/扩展产生的近义中文词（如“课堂行为”）
        # 抢占主题核心词（如“课堂行为分析”）的检索位。
        topic_anchor = str(state.get("topic") or "").strip()

        # 同批近重复剔除：重叠 ≥80% 的检索词召回高度重叠，只浪费各源
        # 配额并产生大量重复论文；腾出的名额留给后续批次更多样的词。
        # topic_recall 必须先于 scope_precision，避免精确范围词成为唯一锚点。
        def _role_priority(query: str) -> tuple[int, int]:
            role = scope_query_roles.get(str(query).strip().casefold(), "topic_recall")
            return (0 if role == "topic_recall" else 1, search_keywords.index(query))

        search_keywords = sorted(search_keywords, key=_role_priority)
        search_keywords, dropped_redundant_keywords = _drop_redundant_keywords(
            search_keywords, topic_anchor
        )

        # 主题锚点必检：若锚点在本轮候选池却被分支词/近义词挤出
        # （每轮仅 max_search_keywords 个名额），强制替换末位词，
        # 避免最精准的主题表达一轮轮顺延、只能靠后缀变体碰运气。
        anchor_promotion = None
        if topic_anchor and not any(
            topic_anchor in str(kw) for kw in search_keywords
        ):
            anchor_in_pool = next(
                (kw for kw in unseen_keywords if topic_anchor in str(kw)),
                None,
            )
            if anchor_in_pool is not None:
                anchor_keyword = (
                    sanitize_search_keyword(str(anchor_in_pool))
                    or str(anchor_in_pool)
                )
                if search_keywords:
                    displaced = search_keywords[-1]
                    search_keywords[-1] = anchor_keyword
                    anchor_promotion = f"{displaced} -> {anchor_keyword}"
                else:
                    search_keywords = [anchor_keyword]
                    anchor_promotion = f"(空批) -> {anchor_keyword}"

        def _is_pure_chinese_query(query: str) -> bool:
            return bool(re.search(r"[\u4e00-\u9fff]", query)) and not bool(
                re.search(r"[A-Za-z]{3,}", query)
            )

        def _query_role(query: str) -> str:
            role = scope_query_roles.get(str(query).strip().casefold())
            if role:
                return role
            return "topic_recall"

        def _is_cnki_anchor(query: str) -> bool:
            return (
                _query_role(query) == "topic_recall"
                and bool(topic_anchor)
                and topic_anchor in query
                and _is_pure_chinese_query(query)
            )

        # 本批预选：含主题核心词的锚点词优先占用 CNKI 检索位；无锚点时
        # 首个纯中文词作初查。跨批次时锚点若首次出现可再升级一次。
        cnki_anchor_in_batch = next(
            (kw for kw in search_keywords if _is_cnki_anchor(kw)), None,
        )
        cnki_fallback_in_batch = next(
            (
                kw for kw in search_keywords
                if _query_role(kw) == "topic_recall" and _is_pure_chinese_query(kw)
            ),
            None,
        )

        # 按关键词语言路由检索源：中文 → Crossref/OpenAlex，英文 → arXiv/SS/OpenAlex
        from app.core.source_capabilities import compatible_sources

        configured_sources = settings.search_sources_list
        per_keyword_debug = []
        keyword_diagnostics: List[SourceDiagnostic] = []
        # CNKI 浏览器检索成本远高于 API 数据源。每个研究会话最多用两次
        # CNKI：首个纯中文词初查一次；若含主题核心词的锚点词出现在后续
        # 批次，允许再用锚点词升级检索一次（避免泛化词抢占唯一名额后，
        # 更精准的锚点词永远无法进入 CNKI）。
        cnki_searched = False if incremental_window else any(
            str(
                diag.get("source") if isinstance(diag, dict) else getattr(diag, "source", "")
            ).lower() == "cnki"
            for diag in (state.get("source_diagnostics") or [])
        )
        # 历史轮次已检索过 CNKI 时不再初查；锚点名额单独由状态追踪，
        # 保证跨 refine 轮次仍能补上更精准的锚点检索。
        cnki_anchor_done = bool(state.get("cnki_anchor_query_used"))
        branch_by_query = {
            str(query).strip().lower(): str(branch.get("branch_type") or "topic_core")
            for branch in (state.get("search_branches") or [])
            for query in (branch.get("queries") or [])
            if str(query).strip()
        }
        # 串行预选：CNKI 名额决策有顺序依赖（锚点 > 初查 > 让位），
        # 必须先按序定稿每个关键词的来源列表，再做并发派发，保证
        # CNKI 预算语义与串行版完全一致。
        keyword_sources: list[list[str]] = []
        for keyword in search_keywords:
            sources = compatible_sources(keyword, configured_sources)
            if "cnki" in sources:
                is_anchor = _is_cnki_anchor(keyword)
                if cnki_anchor_done:
                    # 锚点完成后不再启动浏览器。
                    sources = [source for source in sources if source != "cnki"]
                elif is_anchor:
                    # 锚点检索：初查或跨批次升级，本会话最多一次。
                    # 即使本次 CNKI 返回空或超时，也不重复启动浏览器。
                    cnki_searched = True
                    cnki_anchor_done = True
                    state["cnki_anchor_query_used"] = keyword
                elif cnki_searched or cnki_anchor_in_batch:
                    # 已初查过，或本批后面就有更精准的锚点词，让位。
                    sources = [source for source in sources if source != "cnki"]
                elif keyword == cnki_fallback_in_batch:
                    cnki_searched = True
                else:
                    sources = [source for source in sources if source != "cnki"]
            keyword_sources.append(sources)

        def _dispatch_keyword(index: int, keyword: str, sources: list[str]):
            logger.info("search_node: keyword=%s sources=%s", keyword[:40], sources)
            # 诊断写入各自独立列表，收集后按原顺序合并，避免并发乱序。
            local_diagnostics: List[SourceDiagnostic] = []
            found = search_papers(
                query=keyword,
                start_year=search_start_year,
                end_year=search_end_year,
                max_results=max_results_per_keyword,
                sources=sources,
                diagnostics=local_diagnostics,
                should_cancel=should_cancel,
            )
            return index, keyword, sources, found, local_diagnostics

        # 关键词并发派发：源级并发已有 search_source_max_workers 控制，
        # 词级并发单独限额，避免叠加放大对 API 限速的压力。各词结果
        # 先分桶收集，再按原关键词顺序归并，合并/去重/诊断顺序与
        # 串行版完全一致。
        keyword_workers = max(
            1, min(settings.search_keyword_max_workers, len(search_keywords))
        )
        dispatched: dict[int, tuple[int, str, list[str], list, List[SourceDiagnostic]]] = {}
        with ThreadPoolExecutor(
            max_workers=keyword_workers, thread_name_prefix="keyword-search"
        ) as executor:
            futures = [
                executor.submit(_dispatch_keyword, index, keyword, sources)
                for index, (keyword, sources) in enumerate(
                    zip(search_keywords, keyword_sources)
                )
            ]
            future_index = {future: index for index, future in enumerate(futures)}
            for future in as_completed(futures):
                # 单个关键词失败只作废该词：并发下其他已完成词的结果与
                # 诊断不能陪葬（原先 future.result() 直接抛出，整批丢弃）。
                # 全部词都失败时 all_papers 为空，仍由下方 RuntimeError
                # 走 search_failed 语义。
                index = future_index[future]
                try:
                    dispatched[index] = future.result()
                except Exception as exc:  # noqa: BLE001
                    keyword = search_keywords[index]
                    logger.warning(
                        "search_node: keyword=%r dispatch failed, keeping other keywords: %s",
                        str(keyword)[:40], exc,
                    )
                    dispatched[index] = (index, keyword, keyword_sources[index], [], [])
        for index, keyword in enumerate(search_keywords):
            _, _, sources, papers, local_diagnostics = dispatched[index]
            keyword_diagnostics.extend(local_diagnostics)
            paper_dicts = [p if isinstance(p, dict) else p.model_dump() for p in papers]
            branch_type = branch_by_query.get(str(keyword).strip().lower(), _query_role(keyword))
            for paper in paper_dicts:
                paper["_search_branches"] = list(dict.fromkeys([
                    *(paper.get("_search_branches") or []),
                    branch_type,
                ]))
            per_keyword_debug.append({
                "keyword": keyword,
                "branch_type": branch_type,
                "sources": sources,
                "returned": len(paper_dicts),
                "sample": _summarize_papers(paper_dicts, limit=5),
            })
            all_papers.extend(paper_dicts)
        # 把本轮各关键词各数据源的诊断写入 state（追加模式，保留历史）
        existing_diags = list(state.get("source_diagnostics") or [])
        state["source_diagnostics"] = existing_diags + keyword_diagnostics
        try:
            from app.core.metrics import get_metrics_collector

            collector = get_metrics_collector()
            for diag in keyword_diagnostics:
                collector.record_source(
                    diag.source,
                    success=diag.status in ("success", "empty"),
                    count=diag.returned_count,
                )
        except Exception:
            logger.debug("metrics recording skipped for source_diagnostics", exc_info=True)
        previous_candidates = list(state.get("candidate_papers") or [])
        combined_papers = previous_candidates + all_papers
        combined_dicts = [
            paper if isinstance(paper, dict) else paper.model_dump()
            for paper in combined_papers
        ]
        from app.utils.deduplicate import deduplicate_papers, normalize_title

        # 在搜索节点就执行 DOI/arXiv/标题联合去重。此前只按数据库 paper_id
        # 去重，同一论文来自 OpenAlex、Crossref、S2 时会被重复计数。
        previous_unique = deduplicate_papers(previous_candidates)
        papers = deduplicate_papers(combined_dicts)

        def candidate_key(paper: dict) -> str:
            return (
                str(paper.get("doi") or "").strip().lower()
                or str(paper.get("arxiv_id") or "").strip().lower()
                or normalize_title(str(paper.get("title") or ""))
                or str(paper.get("paper_id") or "")
            )

        previous_keys = {candidate_key(paper) for paper in previous_unique}
        new_papers = [
            paper for paper in papers if candidate_key(paper) not in previous_keys
        ]
        newly_added_unique = len(new_papers)
        # 本轮新增论文的身份键集合（跨轮累积）：供增量轮末尾 LLM 重排
        # 区分新论文与旧榜单，只对新增部分做重排归并。
        if state.get("incremental_retrieval"):
            state["incremental_new_paper_keys"] = list(dict.fromkeys([
                *(state.get("incremental_new_paper_keys") or []),
                *(_paper_identity_key(paper) for paper in new_papers),
            ]))
        # 各源真实贡献观测：本轮新增唯一论文与累计候选池按来源统计，
        # 与“各源返回量”诊断对照即可识别高返回低贡献的源。
        new_by_source = _count_by_source(new_papers)
        pool_by_source = _count_by_source(papers)
        raw_returned = len(all_papers)
        duplicate_count = len(combined_dicts) - len(papers)
        state["candidate_papers"] = papers
        searched_keywords = list(state.get("searched_keywords") or [])
        for keyword in search_keywords:
            if keyword not in searched_keywords:
                searched_keywords.append(keyword)
        state["searched_keywords"] = searched_keywords
        query_windows = list(state.get("searched_query_windows") or [])
        for keyword in search_keywords:
            window_key = f"{str(keyword).strip().lower()}|{search_start_year}|{search_end_year}"
            if window_key not in query_windows:
                query_windows.append(window_key)
            normalized_key = _normalized_window_key(
                keyword, search_start_year, search_end_year
            )
            if normalized_key not in query_windows:
                query_windows.append(normalized_key)
        state["searched_query_windows"] = query_windows
        # 精化循环应依据真正新增的唯一论文，而不是各数据源返回量之和。
        state["last_search_new_results"] = newly_added_unique
        if incremental_window:
            state["incremental_search_new_candidates"] = (
                int(state.get("incremental_search_new_candidates") or 0)
                + newly_added_unique
            )
        if not papers:
            raise RuntimeError(
                "所有论文数据源均未返回结果，可能发生接口限流或网络异常，请稍后重试"
            )
        append_step(
            state, "search", "success",
            tool_name="search_papers",
            input_data={
                "keywords": search_keywords,
                "configured_sources": configured_sources,
                "start_year": search_start_year,
                "end_year": search_end_year,
                "incremental": bool(incremental_window),
                "max_results_per_keyword": max_results_per_keyword,
                "previous_candidates": len(previous_candidates),
            },
            output_data={
                "count": len(papers),
                "raw_returned_count": raw_returned,
                "new_unique_count": newly_added_unique,
                "duplicate_removed_count": duplicate_count,
                # 兼容旧日志消费者；语义调整为“本轮新增唯一论文数”。
                "new_count": newly_added_unique,
                "dropped_malformed_keywords": dropped_malformed_keywords,
                "dropped_redundant_keywords": dropped_redundant_keywords,
                "anchor_promotion": anchor_promotion,
                "new_by_source": new_by_source,
                "pool_by_source": pool_by_source,
                "per_keyword": per_keyword_debug,
                "deduplicated_sample": _summarize_papers(state["candidate_papers"], limit=10),
            },
            duration_ms=int((time.time() - t0) * 1000),
        )
    except InterruptedError as exc:
        from app.agent.graph import AgentCancelledError

        raise AgentCancelledError(str(exc)) from exc
    except Exception as e:
        # 用 ServiceUnavailableError 包装底层异常，保留结构化错误信息；
        # 当前仍保持写入 errors 为字符串，不改变 graph.py 的 search_failed 分支行为。
        from app.agent.exceptions import ServiceUnavailableError

        error = ServiceUnavailableError(
            str(e), step="search", original_error=e,
        )
        logger.error("search_node failed: %s", e)
        state["search_failed"] = True
        state.setdefault("errors", []).append(f"search: {e}")
        append_step(state, "search", "failed", error=str(e), output_data=error.to_dict())
    return state


# ============================================================
# Rank 节点
# ============================================================
def _recovery_branch_minimums(
    state: "ResearchAgentState",
    search_branches: list[dict[str, Any]],
) -> dict[str, int]:
    """为本轮定向恢复分支推导软配额。

    配额 = 该轮受影响路线的核心证据缺口之和，只影响 top_k 名额分配，
    不放宽任何硬过滤或语义筛选：候选仍须先通过完整评分链。
    """
    if not state.get("incremental_retrieval"):
        return {}
    decision = state.get("recovery_decision") or {}
    targets = dict(decision.get("route_targets") or {})
    if not targets:
        return {}
    core_counts = {
        str(route.get("route_id") or ""): len(route.get("core_paper_ids") or [])
        for route in state.get("validated_routes") or []
        if route.get("route_id")
    }
    affected = [
        str(route_id)
        for route_id in decision.get("affected_route_ids") or []
        if str(route_id)
    ]
    deficit = sum(
        max(0, int(targets.get(route_id) or 0) - int(core_counts.get(route_id) or 0))
        for route_id in affected
    )
    if deficit <= 0:
        return {}
    recovery_branches = [
        str(branch.get("branch_type") or "")
        for branch in search_branches
        if str(branch.get("constraint_level") or "") == "targeted_recovery"
        and str(branch.get("branch_type") or "")
    ]
    if not recovery_branches:
        return {}
    # 只给最新一轮恢复分支配额；历史轮次的召回已在池中。
    return {recovery_branches[-1]: deficit}


# 双分支规则筛选诊断中可按分支求和的计数字段。
_BRANCH_FILTER_COUNTERS = (
    "pre_dedup_count", "deduplicated_count", "duplicate_removed_count",
    "filtered_count", "passed_hard_filters", "selected_count",
    "reserve_selected_count", "truncated_by_top_k",
)
# 被过滤样例的展示上限，与 rank_papers._record_filter_diagnostic 保持一致。
_BRANCH_FILTER_EXAMPLE_LIMIT = 12


def _merge_branch_filter_diagnostics(
    branch_diagnostics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """把中英分支各自的规则筛选诊断合并成 rank 步骤的一级字段。

    WHY: 双分支模式此前只把 ``passed_hard_filters`` 拷进 ``branch_stats``，
    逐阶段过滤计数、被过滤样例、top_k 截断量、去重损耗与分支软配额全部随
    局部变量丢弃，排查筛选退化时只能靠 grep 日志行还原。计数按分支求和，
    阶段/原因两个映射按键累加，样例截断到与单分支相同的上限。
    """
    merged: dict[str, Any] = {}
    # 两个"键 → 计数"映射：阶段粒度与原因粒度，均按分支逐键累加。
    counters_by_key: dict[str, dict[str, int]] = {"filtered_by_stage": {}, "filter_reasons": {}}
    examples: list[dict[str, Any]] = []
    for diagnostics in branch_diagnostics.values():
        for key in _BRANCH_FILTER_COUNTERS:
            if key in diagnostics:
                merged[key] = int(merged.get(key) or 0) + int(diagnostics.get(key) or 0)
        for map_name, accumulator in counters_by_key.items():
            for name, count in (diagnostics.get(map_name) or {}).items():
                accumulator[str(name)] = (
                    int(accumulator.get(str(name)) or 0) + int(count or 0)
                )
        for example in diagnostics.get("filtered_examples") or []:
            if len(examples) >= _BRANCH_FILTER_EXAMPLE_LIMIT:
                break
            examples.append(example)
        # 两个分支收到同一份软配额，取任一非空值即可，不应求和。
        for key in ("source_minimums", "branch_minimums"):
            if not merged.get(key) and diagnostics.get(key):
                merged[key] = dict(diagnostics[key])
    for map_name, accumulator in counters_by_key.items():
        if accumulator:
            merged[map_name] = accumulator
    if examples:
        merged["filtered_examples"] = examples
    return merged


# rank 步骤一级字段 ← llm_rerank_papers 诊断键。只收录可跨分支求和的计数，
# reserve_target 这类阈值留在嵌套诊断里，求和会得到误导性的数字。
_RERANK_STEP_FIELDS = {
    "rerank_candidate_count": "candidate_count",
    "rerank_retained_count": "retained_count",
    "rerank_selected_count": "selected_count",
    "rerank_excluded_count": "excluded_count",
    "rerank_screening_degraded_count": "screening_degraded_count",
    "rerank_reserve_backfilled_count": "reserve_backfilled_count",
    "rerank_deepened_batch_count": "deepened_batch_count",
}


def _promote_rerank_diagnostics(rerank_diagnostics: dict[str, Any]) -> dict[str, Any]:
    """把 LLM 语义重排的计数提升为 rank 步骤的一级字段。

    WHY: 这些计数此前只存在于嵌套的 ``llm_rerank`` 子字典，双分支模式下更是
    随局部变量整个丢弃，排查"排除率过高导致引用缺口"时只能靠 grep INFO 日志
    行还原。单分支与双分支的诊断形状互斥（前者计数在顶层，后者只在
    ``branches`` 下），因此两处一起求和不会重复计数。
    """
    sources = [rerank_diagnostics]
    sources.extend((rerank_diagnostics.get("branches") or {}).values())
    return {
        step_key: sum(int(source.get(source_key) or 0) for source in sources)
        for step_key, source_key in _RERANK_STEP_FIELDS.items()
    }


def _screening_reserve_k(
    required_reference_count: int,
    max_papers: int,
    candidate_max: int,
) -> int:
    """规则粗排在 ``max_papers`` 之外额外保留的筛选后备窗口大小。

    WHY: LLM 语义重排只对规则粗排的输出打分，尾部此前被 ``top_k`` 直接丢弃。
    实测 1101 篇候选被截到 64 篇后排除 34 篇，重排的回填池结构性为空，
    "不少于 40 篇引用"的缺口再也补不回来。保留一段规则合格的尾部，重排
    阶段才能继续向深处筛选，而不是拿未经语义确认的论文凑数。
    """
    required = max(0, int(required_reference_count or 0))
    window = max(0, int(max_papers or 0))
    if not required or not window:
        return 0
    return max(0, int(candidate_max or 0) - window)


@node(name="rank", category="retrieval", description="候选论文去重、排序、筛选")
@requires("candidate_papers", "topic", "keywords", "max_papers", "retrieval_target")
@provides("ranked_papers", "screening_report", "compiled_scope")
@optional(
    "required_concepts", "excluded_title_terms", "selected_scope",
    "required_reference_count", "search_branches", "research_semantic_frame",
    "screening_protocol", "topic_anchors"
)
def rank_node(state: "ResearchAgentState", llm=None) -> "ResearchAgentState":
    """候选论文去重、排序、筛选。

    中英文分支独立处理：先按语言拆分 → 各自去重/硬过滤/评分/LLM重排 →
    百分位归一化 → 配额合并 → 跨语言去重 → 最终论文池。
    旧会话无协议或 language_branch 被禁用时保留原统一过滤链。
    """
    t0 = time.time()
    try:
        from app.tools.rank_papers import (
            deduplicate_and_rank,
            llm_rerank_papers,
        )
        from app.tools.paper_matching import compile_scope
        from app.tools.language_router import split_papers_by_language
        from app.tools.branch_merge import (
            build_language_coverage_contract,
            merge_language_branches,
        )

        candidates = state.get("candidate_papers") or []
        topic = state.get("topic", "")
        from app.core.config import get_settings

        settings = get_settings()
        max_papers = (
            state.get("retrieval_target")
            or state.get("max_papers")
            or settings.default_max_papers
        )
        selected_scope = state.get("selected_scope") or {}
        search_branches = state.get("search_branches") or []
        research_mode = str(
            (state.get("research_semantic_frame") or {}).get("research_mode") or ""
        )
        required_concepts = state.get("required_concepts") or []
        excluded_title_terms = state.get("excluded_title_terms") or []
        screening_protocol = state.get("screening_protocol") or {}
        compiled_scope = compile_scope(
            selected_scope=selected_scope,
            semantic_frame=state.get("research_semantic_frame") or {},
            screening_protocol=screening_protocol,
            required_concepts=required_concepts,
            topic_anchors=state.get("topic_anchors") or [],
            search_branches=search_branches,
            topic=topic,
            research_mode=research_mode,
            excluded_title_terms=excluded_title_terms,
        )
        state["compiled_scope"] = compiled_scope
        rule_diagnostics: dict[str, Any] = {"compiled_scope_fingerprint": compiled_scope["fingerprint"]}
        rerank_diagnostics: dict[str, Any] = {}
        branch_stats: dict[str, Any] = {}
        # 分支诊断必须在分支块之外声明：块内局部变量在合并阶段就取不到了，
        # 双分支模式因此只剩 branch_stats 里手抄的一个 passed_hard_filters。
        zh_diag: dict[str, Any] = {}
        en_diag: dict[str, Any] = {}
        zh_rerank_diag: dict[str, Any] = {}
        en_rerank_diag: dict[str, Any] = {}
        # 定向恢复轮的软配额：为本轮 targeted_recovery 分支保留名额，避免为
        # 某条路线补来的论文因全局分数偏低而在 top_k 截断处消失。
        branch_minimums = _recovery_branch_minimums(state, search_branches)
        # 规则粗排在主窗口之外保留的筛选后备段：LLM 重排据此在高排除率下
        # 继续向尾部加深筛选，而不是拿未筛论文回填。
        screening_reserve_k = _screening_reserve_k(
            state.get("required_reference_count") or 0,
            max_papers,
            settings.rerank_candidate_max,
        )

        # 判断是否启用双分支模式
        use_branched = (
            settings.language_branch_enabled
            and bool(screening_protocol)
            and len(candidates) > 0
        )

        if use_branched:
            # ============================================================
            # 双分支模式：中英文独立处理 + 配额合并
            # ============================================================
            zh_papers, en_papers = split_papers_by_language(candidates)
            branch_stats["zh_initial"] = len(zh_papers)
            branch_stats["en_initial"] = len(en_papers)

            # --- 中文分支 ---
            zh_ranked: list[dict] = []
            if zh_papers:
                zh_deduped = deduplicate_and_rank(
                    zh_papers, topic,
                    top_k=min(max_papers, len(zh_papers)),
                    keywords=state.get("keywords") or [],
                    required_concepts=required_concepts,
                    excluded_title_terms=excluded_title_terms,
                    scope=selected_scope,
                    search_branches=search_branches,
                    research_mode=research_mode,
                    screening_protocol=screening_protocol,
                    filter_diagnostics=zh_diag,
                    branch_minimums=branch_minimums,
                    language_branch="zh",
                    start_year=state.get("start_year"),
                    end_year=state.get("end_year"),
                    # 与单分支同理用 reserve_k 而非抬高 top_k：抬高会把
                    # branch_minimums 的配额基准从主窗口挪走，定向恢复论文
                    # 可能落到 72 名之外，再被下游全局重排截掉。
                    reserve_k=screening_reserve_k,
                    compiled_scope=compiled_scope,
                )
                branch_stats["zh_after_hard_filter"] = zh_diag.get("passed_hard_filters", 0)
                zh_ranked = zh_deduped

            # --- 英文分支 ---
            en_ranked: list[dict] = []
            if en_papers:
                en_deduped = deduplicate_and_rank(
                    en_papers, topic,
                    top_k=min(max_papers, len(en_papers)),
                    keywords=state.get("keywords") or [],
                    required_concepts=required_concepts,
                    excluded_title_terms=excluded_title_terms,
                    scope=selected_scope,
                    search_branches=search_branches,
                    research_mode=research_mode,
                    screening_protocol=screening_protocol,
                    filter_diagnostics=en_diag,
                    branch_minimums=branch_minimums,
                    language_branch="en",
                    start_year=state.get("start_year"),
                    end_year=state.get("end_year"),
                    # 同中文分支：配额基准保持在主窗口。
                    reserve_k=screening_reserve_k,
                    compiled_scope=compiled_scope,
                )
                branch_stats["en_after_hard_filter"] = en_diag.get("passed_hard_filters", 0)
                en_ranked = en_deduped

            # --- 硬过滤后轻量跨语言去重（仅 DOI/ID 精确匹配，不做标题语义）---
            from app.tools.branch_merge import identifier_level_cross_dedup

            if zh_ranked and en_ranked:
                zh_ranked, en_ranked = identifier_level_cross_dedup(zh_ranked, en_ranked)
                branch_stats["zh_after_identifier_dedup"] = len(zh_ranked)
                branch_stats["en_after_identifier_dedup"] = len(en_ranked)

            # --- 分支 LLM 重排 ---
            # WHY: 分支重排只负责语义筛选与分支内排序，minimum_required 必须
            # 保持 0——引用缺口的安全网属于合并后那次全局重排。若在每个语言
            # 分支各自套用全局引用要求，两支会分别去凑同一个目标，筛选成本
            # 翻倍且各自虚高保留量。
            if llm is not None:
                if zh_ranked:
                    try:
                        zh_ranked = llm_rerank_papers(
                            zh_ranked, topic=topic, scope=selected_scope,
                            llm=llm, top_k=max_papers + screening_reserve_k,
                            research_mode=research_mode,
                            screening_protocol=screening_protocol,
                            rerank_diagnostics=zh_rerank_diag,
                            minimum_required=0,
                        )
                    except Exception as exc:
                        logger.warning("ZH branch LLM rerank failed: %s", exc)
                        zh_rerank_diag.update({
                            "mode": "rule_order_fallback",
                            "error": str(exc),
                        })
                if en_ranked:
                    try:
                        en_ranked = llm_rerank_papers(
                            en_ranked, topic=topic, scope=selected_scope,
                            llm=llm, top_k=max_papers + screening_reserve_k,
                            research_mode=research_mode,
                            screening_protocol=screening_protocol,
                            rerank_diagnostics=en_rerank_diag,
                            minimum_required=0,
                        )
                    except Exception as exc:
                        logger.warning("EN branch LLM rerank failed: %s", exc)
                        en_rerank_diag.update({
                            "mode": "rule_order_fallback",
                            "error": str(exc),
                        })

            # 统一使用 _branch_final_score 供后续合并
            for p in zh_ranked:
                p["_branch_final_score"] = float(p.get("_rank_score", 0))
            for p in en_ranked:
                p["_branch_final_score"] = float(p.get("_rank_score", 0))
            branch_stats["zh_after_rank"] = len(zh_ranked)
            branch_stats["en_after_rank"] = len(en_ranked)

            state["language_coverage_target"] = build_language_coverage_contract(
                int(state.get("required_reference_count") or state.get("max_papers") or 0),
                float(
                    state.get("language_branch_zh_ratio")
                    or settings.language_branch_zh_ratio
                ),
                min_zh=settings.language_branch_min_zh,
                min_en=settings.language_branch_min_en,
                eligible_zh=len(zh_ranked),
                eligible_en=len(en_ranked),
                affinity=str(
                    (state.get("research_semantic_frame") or {}).get("language_affinity")
                    or "balanced"
                ),
            )

            # --- 异常告警与配额保护 ---
            en_pass_rate = None
            if branch_stats.get("en_initial", 0) >= 30:
                en_pass_rate = (
                    branch_stats.get("en_after_hard_filter", 0)
                    / max(1, branch_stats["en_initial"])
                )
                if en_pass_rate < 0.10:
                    logger.warning(
                        "英文分支硬过滤通过率低于 10%% (%.1f%%)，"
                        "暂停英文配额转移并保留语言缺口供定向恢复",
                        en_pass_rate * 100,
                    )
            state["screening_report_low_pass_protection"] = {
                "english_initial": int(branch_stats.get("en_initial") or 0),
                "english_after_hard_filter": int(branch_stats.get("en_after_hard_filter") or 0),
                "english_pass_rate": en_pass_rate,
                "quota_transfer_blocked": bool(en_pass_rate is not None and en_pass_rate < 0.10),
            }
            if branch_stats.get("zh_initial", 0) >= 20:
                zh_pass_rate = (
                    branch_stats.get("zh_after_hard_filter", 0)
                    / max(1, branch_stats["zh_initial"])
                )
                if zh_pass_rate < 0.10:
                    logger.warning(
                        "中文分支硬过滤通过率低于 10%% (%.1f%%)，"
                        "可能存在中文别名缺失或协议过严",
                        zh_pass_rate * 100,
                    )

            # --- 分支合并 ---
            # 中文配额优先取规划阶段按主题语言倾向解析的值；缺省回落到全局配置。
            effective_zh_ratio = float(
                state.get("language_branch_zh_ratio")
                or settings.language_branch_zh_ratio
            )
            ranked = merge_language_branches(
                zh_ranked=zh_ranked,
                en_ranked=en_ranked,
                # 合并窗口同样要带上筛选后备段：否则这里截回主窗口，下游全局
                # 重排就没有可加深的材料（实测 2026-09-01 走的正是双分支模式）。
                # calculate_branch_targets 按 top_k 等比推导两侧配额，比例不变。
                top_k=max_papers + screening_reserve_k,
                zh_ratio=effective_zh_ratio,
                min_zh=settings.language_branch_min_zh,
                min_en=settings.language_branch_min_en,
                allow_quota_transfer=not bool(
                    state.get("screening_report_low_pass_protection", {}).get(
                        "quota_transfer_blocked"
                    )
                ),
            )
            rule_diagnostics = {
                "mode": "branched",
                "input_count": len(candidates),
                "branch_stats": branch_stats,
                "zh_ratio": effective_zh_ratio,
                "zh_ratio_reason": state.get("language_branch_zh_ratio_reason") or "",
                "zh_filter": zh_diag,
                "en_filter": en_diag,
                **_merge_branch_filter_diagnostics({"zh": zh_diag, "en": en_diag}),
            }
            if zh_rerank_diag or en_rerank_diag:
                rerank_diagnostics["branches"] = {
                    "zh": zh_rerank_diag,
                    "en": en_rerank_diag,
                }
            rerank_triggered = True
        else:
            # ============================================================
            # 单分支模式（向后兼容）
            # ============================================================
            source_minimums: dict[str, int] = {}
            cnki_count = sum(
                1 for paper in candidates
                if str(paper.get("source") or "").strip().lower() == "cnki"
            )
            if cnki_count:
                configured_source_count = max(1, len(set(settings.search_sources_list)))
                requested_evidence = int(
                    state.get("required_reference_count")
                    or state.get("max_papers")
                    or 1
                )
                source_quota = max(
                    1,
                    math.ceil(requested_evidence / configured_source_count),
                )
                source_minimums["cnki"] = min(source_quota, cnki_count)

            ranked = deduplicate_and_rank(
                candidates,
                topic,
                max_papers,
                keywords=state.get("keywords") or [],
                required_concepts=required_concepts,
                excluded_title_terms=excluded_title_terms,
                scope=selected_scope,
                search_branches=search_branches,
                research_mode=research_mode,
                screening_protocol=screening_protocol,
                filter_diagnostics=rule_diagnostics,
                source_minimums=source_minimums,
                branch_minimums=branch_minimums,
                start_year=state.get("start_year"),
                end_year=state.get("end_year"),
                # 用 reserve_k 而非抬高 top_k：CNKI 来源配额与恢复分支配额都
                # 必须继续按主窗口计算，后备尾部在配额逻辑之后才追加。
                reserve_k=screening_reserve_k,
                compiled_scope=compiled_scope,
            )

            rerank_triggered = False
            state.pop("language_coverage_target", None)
            if llm is not None and ranked:
                try:
                    ranked = llm_rerank_papers(
                        ranked,
                        topic=topic,
                        scope=selected_scope,
                        llm=llm,
                        top_k=max_papers,
                        research_mode=research_mode,
                        screening_protocol=screening_protocol,
                        rerank_diagnostics=rerank_diagnostics,
                        minimum_required=int(
                            state.get("required_reference_count") or 0
                        ),
                    )
                    rerank_triggered = True
                except Exception as rerank_exc:
                    logger.warning(
                        "LLM rerank failed, falling back to rule scores: %s",
                        rerank_exc,
                    )

        state["ranked_papers"] = ranked
        # 入选集合的来源贡献观测：后续裁撤低效源/调整双语配额以此为准。
        selected_by_source = _count_by_source(ranked)
        state["screening_report"] = {
            "protocol": screening_protocol,
            "compiled_scope": {
                "version": compiled_scope.get("version"),
                "fingerprint": compiled_scope.get("fingerprint"),
            },
            "rule_filter": rule_diagnostics,
            "llm_rerank": rerank_diagnostics,
        }
        append_step(
            state, "rank", "success",
            tool_name="rank_papers",
            input_data={
                "topic": topic,
                "candidates": len(candidates),
                "max_papers": max_papers,
                "required_reference_count": state.get("required_reference_count"),
                "retrieval_target": state.get("retrieval_target"),
                "keywords": state.get("keywords") or [],
                "required_concepts": required_concepts,
                "excluded_title_terms": excluded_title_terms,
                "selected_scope": selected_scope,
                "compiled_scope": {
                    "version": compiled_scope.get("version"),
                    "fingerprint": compiled_scope.get("fingerprint"),
                },
                "search_branches": search_branches,
                "research_mode": research_mode,
                "screening_protocol": screening_protocol,
                "candidate_sample": _summarize_papers(candidates, limit=10),
            },
            output_data={
                "ranked": len(ranked),
                "screening_reserve_k": screening_reserve_k,
                "screening_reserve_count": sum(
                    1 for paper in ranked if paper.get("_rule_screened_reserve")
                ),
                "rerank_triggered": rerank_triggered,
                "filter_mode": rule_diagnostics.get("mode"),
                "filtered_count": rule_diagnostics.get("filtered_count", 0),
                "filter_reasons": rule_diagnostics.get("filter_reasons", {}),
                "filtered_by_stage": rule_diagnostics.get("filtered_by_stage", {}),
                "filtered_examples": rule_diagnostics.get("filtered_examples", []),
                "passed_hard_filters": rule_diagnostics.get("passed_hard_filters", 0),
                "rule_selected_count": rule_diagnostics.get("selected_count", 0),
                "rule_reserve_selected_count": rule_diagnostics.get(
                    "reserve_selected_count", 0
                ),
                "truncated_by_top_k": rule_diagnostics.get("truncated_by_top_k", 0),
                "source_minimums": rule_diagnostics.get("source_minimums", {}),
                "branch_minimums": rule_diagnostics.get("branch_minimums", {}),
                "selected_by_source": selected_by_source,
                "llm_rerank": rerank_diagnostics,
                **_promote_rerank_diagnostics(rerank_diagnostics),
                "ranked_sample": _summarize_papers(ranked, limit=10),
            },
            duration_ms=int((time.time() - t0) * 1000),
        )
    except InterruptedError as exc:
        from app.agent.graph import AgentCancelledError

        raise AgentCancelledError(str(exc)) from exc
    except Exception as e:
        from app.agent.exceptions import DegradableAgentError

        error = DegradableAgentError(str(e), step="rank", original_error=e)
        logger.error("rank_node failed: %s", error.message)
        state.setdefault("errors", []).append(f"rank: {e}")
        append_step(state, "rank", "failed", error=str(e))
    return state


@node(name="refine_search", category="retrieval", description="根据检索和筛选反馈，让 LLM 生成下一轮关键词")
@requires("topic", "keywords", "ranked_papers")
@provides("keywords", "search_refinement_count", "focus_coverage")
def refine_search_node(state: "ResearchAgentState", llm=None) -> "ResearchAgentState":
    """根据检索和筛选反馈，让 LLM 生成下一轮关键词。"""
    t0 = time.time()
    try:
        from app.agent.planner import (
            _normalize_topic_anchor_groups,
            refine_search_strategy,
        )
        from app.agent.focus_coverage import (
            required_focus_coverage,
            supplemental_focus_queries,
        )

        current_keywords = list(state.get("keywords") or [])
        core_keywords = list(state.get("core_keywords") or [state.get("topic") or ""])
        state.setdefault("expanded_keywords", [])
        state.setdefault("search_refinement_count", 0)
        semantic_frame = state.get("research_semantic_frame") or {}
        if not semantic_frame.get("evidence_requirements"):
            # 旧会话按原始请求与已保存重点重新规范化，避免维护方法专属补丁表。
            from app.agent.research_semantic_parser import parse_research_semantics

            legacy_focuses = [
                str(item) for item in semantic_frame.get("required_focuses") or []
                if str(item).strip()
            ]
            semantic_query = "\n".join(filter(None, [
                str(state.get("user_query") or ""),
                "用户明确重点：" + "；".join(legacy_focuses) if legacy_focuses else "",
            ]))
            parsed = parse_research_semantics(
                semantic_query,
                str(state.get("canonical_topic") or state.get("topic") or ""),
                llm=llm,
            ).model_dump(mode="json")
            if legacy_focuses:
                selected_requirements: list[dict[str, Any]] = []
                for focus in legacy_focuses:
                    focus_frame = parse_research_semantics(
                        focus,
                        str(state.get("canonical_topic") or state.get("topic") or ""),
                        llm=llm,
                    ).model_dump(mode="json")
                    focus_requirements = focus_frame.get("evidence_requirements") or []
                    primary = [
                        item for item in focus_requirements
                        if item.get("evidence_role") != "interpretation"
                    ] or focus_requirements
                    selected_requirements.extend(primary)
                parsed["evidence_requirements"] = list({
                    str(item.get("requirement_id") or ""): item
                    for item in selected_requirements
                    if item.get("requirement_id")
                }.values())
                parsed["required_focuses"] = legacy_focuses
            semantic_frame = parsed
            state["research_semantic_frame"] = semantic_frame
        focus_coverage = required_focus_coverage(
            semantic_frame,
            state.get("ranked_papers") or [],
        )
        state["focus_coverage"] = focus_coverage
        search_step = _latest_step(state, "search")
        rank_step = _latest_step(state, "rank")
        feedback = {
            "target": state.get("max_papers"),
            "candidate_count": len(state.get("candidate_papers") or []),
            "ranked_count": len(state.get("ranked_papers") or []),
            "searched_keywords": state.get("searched_keywords") or [],
            "search_output": search_step.get("output_data") or {},
            "rank_output": rank_step.get("output_data") or {},
        }
        # 成文后实际引用数低于用户硬性要求时，把缺口数量交给 refine，
        # 让下一轮关键词定向扩召回，而不是只靠质量横幅提示兜底。
        if int(state.get("citation_shortfall_count") or 0) > 0:
            feedback["citation_shortfall"] = {
                "required_reference_count": state.get("required_reference_count"),
                "actual_cited": state.get("_citation_gap_repair_previous_cited"),
                "shortfall": int(state.get("citation_shortfall_count") or 0),
                "guidance": (
                    "成文后实际引用数低于用户硬性要求，请给出与已有查询互补、"
                    "能扩大召回面的检索词（子方向、同义变体、跨语言变体），"
                    "优先召回尚未纳入的新文献。"
                ),
            }
        strategy = (
            refine_search_strategy(
                topic=state.get("topic", ""),
                user_query=state.get("user_query", ""),
                current_keywords=core_keywords,
                feedback=feedback,
                llm=llm,
                existing_batches=state.get("keyword_batches") or [],
            )
            if llm is not None
            else {"keywords": current_keywords}
        )
        focus_queries = supplemental_focus_queries(
            focus_coverage.get("missing_requirement_ids")
            or focus_coverage.get("missing_focuses") or [],
            str(state.get("canonical_topic") or state.get("topic") or ""),
            semantic_frame,
        )
        new_keywords = list(dict.fromkeys([
            *(strategy.get("keywords") or current_keywords),
            *focus_queries,
        ]))
        # 规划/精化产生的中英混杂词在入池前清洗，避免污染 searched_keywords。
        from app.core.source_capabilities import sanitize_search_keyword

        new_keywords = list(dict.fromkeys(
            cleaned for keyword in new_keywords
            if (cleaned := sanitize_search_keyword(keyword))
        ))
        if new_keywords == current_keywords:
            append_step(
                state,
                "refine_search",
                "failed",
                tool_name="llm_refine_search_strategy",
                input_data={
                    "keywords": current_keywords,
                    "feedback": feedback,
                },
                output_data={"keywords": new_keywords, "focus_coverage": focus_coverage},
                error="LLM 未生成新的关键词",
                duration_ms=int((time.time() - t0) * 1000),
            )
            return state

        state["keywords"] = new_keywords
        core_keys = {str(item).strip().casefold() for item in core_keywords if str(item).strip()}
        state["expanded_keywords"] = list(dict.fromkeys([
            *(state.get("expanded_keywords") or []),
            *[item for item in new_keywords if str(item).strip().casefold() not in core_keys],
        ]))
        # 批次随词池一起更新：refine 新增词已在 strategy 内并入对应批次
        refined_batches = strategy.get("keyword_batches") or []
        if refined_batches:
            state["keyword_batches"] = refined_batches
        state["search_refinement_count"] = int(state.get("search_refinement_count") or 0) + 1
        # 规划阶段锚点为空时（如概念组被双语门禁全部丢弃），采纳 refine
        # 通过门禁的双语概念组，避免 rank 阶段长期退化为裸主题串匹配。
        refined_anchors = _normalize_topic_anchor_groups(
            strategy.get("topic_anchors") or strategy.get("required_concepts") or []
        )
        if refined_anchors and not (state.get("required_concepts") or []):
            state["required_concepts"] = refined_anchors
            logger.info(
                "Adopted %d refined topic anchor group(s) into required_concepts",
                len(refined_anchors),
            )
        append_step(
            state,
            "refine_search",
            "success",
            tool_name="llm_refine_search_strategy",
            input_data={
                "keywords": current_keywords,
                "feedback": feedback,
            },
            output_data={
                "keywords": state["keywords"],
                "required_concepts": state.get("required_concepts") or [],
                "excluded_title_terms": state.get("excluded_title_terms") or [],
                "ignored_refined_required_concepts": strategy.get("topic_anchors") or strategy.get("required_concepts") or [],
                "dropped_refined_anchor_groups": strategy.get("dropped_monolingual_groups") or [],
                "refinement_count": state["search_refinement_count"],
                "focus_coverage": focus_coverage,
                "supplemental_focus_queries": focus_queries,
            },
            duration_ms=int((time.time() - t0) * 1000),
        )
    except Exception as e:
        # P1 集成：关键词精化失败可降级（继续用旧关键词），非致命。
        from app.agent.exceptions import LLMGenerationError

        error = LLMGenerationError(
            str(e), step="refine_search", fallback_available=True, original_error=e,
        )
        logger.error("refine_search_node failed: %s", error.to_dict())
        append_step(
            state,
            "refine_search",
            "skipped",
            tool_name="llm_refine_search_strategy",
            input_data={"keywords": state.get("keywords") or []},
            error=str(e),
            duration_ms=int((time.time() - t0) * 1000),
        )
    return state


# ============================================================
# 默认年份范围扩展节点
# ============================================================
@node(name="expand_search_year", category="retrieval", description="近三年结果不足目标篇数时，逐年扩展到最多近五年")
@requires("ranked_papers", "start_year", "end_year")
@provides(
    "ranked_papers", "candidate_papers", "start_year", "search_expanded",
    "retrieval_requirement_met",
)
def expand_search_year_node(
    state: "ResearchAgentState",
    should_cancel=None,
) -> "ResearchAgentState":
    """近三年结果不足目标篇数时，逐年扩展到最多近五年。

    用户使用“仅限”等严格约束词时不扩展。
    """
    t0 = time.time()
    target = int(state.get("retrieval_target") or state.get("max_papers") or 0)
    ranked_count = len(state.get("ranked_papers") or [])

    if state.get("year_range_explicit"):
        state["search_expanded"] = False
        state["retrieval_requirement_met"] = ranked_count >= target
        append_step(
            state,
            "expand_year",
            "success",
            input_data={
                "start_year": state.get("start_year"),
                "end_year": state.get("end_year"),
                "ranked_count": ranked_count,
                "target": target,
            },
            output_data={
                "expanded": False,
                "reason": "explicit_user_defined_year_range",
                "ranked": ranked_count,
                "target": target,
                "requirement_met": state["retrieval_requirement_met"],
            },
            duration_ms=0,
        )
        return state

    if ranked_count >= target:
        state["search_expanded"] = False
        state["retrieval_requirement_met"] = True
        append_step(
            state,
            "expand_year",
            "success",
            input_data={
                "start_year": state.get("start_year"),
                "end_year": state.get("end_year"),
                "ranked_count": ranked_count,
                "target": target,
            },
            output_data={
                "expanded": False,
                "reason": "target_reached",
                "ranked": ranked_count,
                "target": target,
                "requirement_met": True,
            },
            duration_ms=0,
        )
        return state

    try:
        from app.core.config import get_settings
        from app.tools.rank_papers import deduplicate_and_rank, evaluate_scope_filter
        from app.tools.paper_matching import compile_scope
        from app.core.source_capabilities import compatible_sources
        from app.tools.search_papers import search_papers

        settings = get_settings()
        original_start = int(state["start_year"])
        end_year = int(state["end_year"])
        current_span = end_year - original_start + 1
        can_expand = current_span == settings.default_year_lookback
        if not can_expand:
            state["search_expanded"] = False
            state["retrieval_requirement_met"] = ranked_count >= target
            append_step(
                state,
                "expand_year",
                "success",
                input_data={
                    "start_year": state.get("start_year"),
                    "end_year": state.get("end_year"),
                    "ranked_count": ranked_count,
                    "target": target,
                },
                output_data={
                    "expanded": False,
                    "reason": "user_defined_non_default_year_range",
                    "ranked": ranked_count,
                    "target": target,
                    "requirement_met": state["retrieval_requirement_met"],
                },
                duration_ms=int((time.time() - t0) * 1000),
            )
            return state

        extension = max(1, settings.search_year_extension)
        keywords = _select_search_keywords(
            state.get("keywords") or [state.get("topic", "")],
            limit=settings.max_search_keywords,
        )
        combined = list(state.get("candidate_papers") or [])
        ranked = list(state.get("ranked_papers") or [])
        expanded_start = original_start
        added_candidates = 0
        searched_years: list[int] = []

        while len(ranked) < target and current_span < settings.max_year_lookback:
            expanded_end = expanded_start - 1
            next_start = max(
                expanded_end - extension + 1,
                end_year - settings.max_year_lookback + 1,
            )
            extra_papers: list[dict] = []
            expand_diagnostics: list[SourceDiagnostic] = []
            for keyword in keywords:
                expand_sources = compatible_sources(keyword, settings.search_sources_list)
                results = search_papers(
                    query=keyword,
                    start_year=next_start,
                    end_year=expanded_end,
                    max_results=max(5, target),
                    sources=expand_sources,
                    diagnostics=expand_diagnostics,
                    should_cancel=should_cancel,
                )
                extra_papers.extend(
                    paper if isinstance(paper, dict) else paper.model_dump()
                    for paper in results
                )

            combined.extend(extra_papers)
            added_candidates += len(extra_papers)
            searched_years.extend(range(next_start, expanded_end + 1))
            expanded_start = next_start
            current_span = end_year - expanded_start + 1
            selected_scope = state.get("selected_scope") or {}
            search_branches = state.get("search_branches") or []
            research_mode = str(
                (state.get("research_semantic_frame") or {}).get("research_mode") or ""
            )
            compiled_scope = compile_scope(
                selected_scope=selected_scope,
                semantic_frame=state.get("research_semantic_frame") or {},
                screening_protocol=state.get("screening_protocol") or {},
                required_concepts=state.get("required_concepts") or [],
                topic_anchors=state.get("topic_anchors") or [],
                search_branches=search_branches,
                excluded_title_terms=state.get("excluded_title_terms") or [],
                topic=state.get("topic") or "",
                research_mode=research_mode,
            )
            state["compiled_scope"] = compiled_scope
            ranked = deduplicate_and_rank(
                combined,
                state.get("topic", ""),
                target,
                keywords=state.get("keywords") or [],
                required_concepts=state.get("required_concepts") or [],
                excluded_title_terms=state.get("excluded_title_terms") or [],
                scope=selected_scope,
                search_branches=search_branches,
                research_mode=research_mode,
                screening_protocol=state.get("screening_protocol") or {},
                start_year=expanded_start,
                end_year=end_year,
                compiled_scope=compiled_scope,
            )

        state["candidate_papers"] = combined
        state["ranked_papers"] = ranked
        state["start_year"] = expanded_start
        state["search_expanded"] = expanded_start < original_start
        state["retrieval_requirement_met"] = len(ranked) >= target

        append_step(
            state,
            "expand_year",
            "success",
            tool_name="search_papers",
            input_data={
                "keywords": keywords,
                "searched_years": searched_years,
                "previous_candidates": max(0, len(combined) - added_candidates),
                "previous_ranked": ranked_count,
            },
            output_data={
                # 与 state["search_expanded"] 一致：循环可能一次都没执行
                # （已到最大回溯年限），此时并未真正扩展年份。
                "expanded": state["search_expanded"],
                "previous_start_year": original_start,
                "effective_start_year": expanded_start,
                "added_candidates": added_candidates,
                "ranked": len(ranked),
                "target": target,
                "requirement_met": state["retrieval_requirement_met"],
                "ranked_sample": _summarize_papers(ranked, limit=10),
            },
            duration_ms=int((time.time() - t0) * 1000),
        )
    except InterruptedError as exc:
        from app.agent.graph import AgentCancelledError

        raise AgentCancelledError(str(exc)) from exc
    except Exception as e:
        from app.agent.exceptions import DegradableAgentError

        error = DegradableAgentError(str(e), step="expand_year", original_error=e)
        logger.error("expand_search_year_node failed: %s", error.message)
        state.setdefault("errors", []).append(f"expand_year: {e}")
        append_step(state, "expand_year", "failed", error=str(e))
    return state


def evidence_yield_report(state: "ResearchAgentState") -> dict[str, Any]:
    """统计证据成品率；观测不完整时返回空 dict。

    两个率分开落盘：``evidence_availability_rate``（可用/卡片）反映形态复核、
    元数据核验与角色判定这一段损耗，``citation_realization_rate``（引用/可用）
    反映路线归属与章节分配这一段损耗。池目标只按端到端率倒推，但排查时必须
    知道损耗发生在哪一段——2026-09-01 实测 45 卡片 → 32 可用 → 25 引用。
    """
    card_count = len(state.get("paper_cards") or [])
    usable = int(
        (state.get("generation_readiness") or {}).get("usable_reference_count") or 0
    )
    cited = int(state.get("unique_cited_paper_count") or 0)
    if card_count <= 0 or usable <= 0 or cited <= 0:
        return {}
    return {
        "paper_card_count": card_count,
        "usable_reference_count": usable,
        "unique_cited_paper_count": cited,
        "evidence_availability_rate": round(min(usable / card_count, 1.0), 4),
        "citation_realization_rate": round(min(cited / usable, 1.0), 4),
        "end_to_end_rate": round(min(cited / card_count, 1.0), 4),
    }


def absolute_evidence_pool_target(
    required_reference_count: int,
    observed_yield: dict[str, Any] | None,
    reserve_ratio: float,
) -> int:
    """把引用要求按端到端成品率倒推成证据池绝对目标。

    WHY: 用户要求的是最终综述使用的唯一参考文献数（AGENTS.md 规则 10），而池
    里的论文要经两段损耗才变成引用。静态 ``evidence_pool_reserve_ratio=0.5``
    等价于假设成品率 1/1.5≈0.667（40 篇 → 60），实测端到端只有 0.556
    （40 篇 → 需约 72），系统性偏小，显式引用要求因此长期不达标。首轮无观测
    时仍退回该默认假设。
    """
    if required_reference_count <= 0:
        return 0
    end_to_end = float((observed_yield or {}).get("end_to_end_rate") or 0.0)
    if end_to_end > 0:
        # 下限 0.20：一次异常观测不得把池目标推到无节制规模。
        yield_rate = max(end_to_end, 0.20)
    else:
        yield_rate = 1.0 / (1.0 + max(0.0, float(reserve_ratio)))
    return int(math.ceil(required_reference_count / min(yield_rate, 1.0)))


# ============================================================
# Fetch Detail 节点
# ============================================================
@node(name="fetch_detail", category="retrieval", description="补全论文详情")
@requires("ranked_papers")
@provides("paper_details", "retrieval_requirement_met", "compiled_scope")
def fetch_detail_node(
    state: "ResearchAgentState",
    should_cancel=None,
) -> "ResearchAgentState":
    """补全论文详情。"""
    t0 = time.time()
    from app.agent.graph import AgentCancelledError

    try:
        from app.tools.fetch_metadata import fetch_batch_details

        from app.tools.rank_papers import (
            evaluate_paper_hard_filters,
        )
        from app.tools.paper_matching import compile_scope
        from app.tools.language_router import detect_paper_language
        from app.core.config import get_settings

        ranked_all = state.get("ranked_papers") or []
        incremental = bool(state.get("incremental_retrieval"))
        existing_details = list(state.get("paper_details") or []) if incremental else []
        existing_keys = {
            _paper_identity_key(paper) for paper in existing_details
            if _paper_identity_key(paper)
        }
        ranked = (
            [paper for paper in ranked_all if _paper_identity_key(paper) not in existing_keys]
            if incremental else ranked_all
        )
        required_count = int(
            state.get("required_reference_count") or state.get("max_papers") or 0
        )
        generation_limit = int(state.get("generation_limit") or 0)
        settings = get_settings()
        compiled_scope = compile_scope(
            selected_scope=state.get("selected_scope") or {},
            semantic_frame=state.get("research_semantic_frame") or {},
            screening_protocol=state.get("screening_protocol") or {},
            required_concepts=state.get("required_concepts") or [],
            topic_anchors=state.get("topic_anchors") or [],
            search_branches=state.get("search_branches") or [],
            excluded_title_terms=state.get("excluded_title_terms") or [],
            topic=state.get("topic") or "",
            research_mode=str((state.get("research_semantic_frame") or {}).get("research_mode") or ""),
        )
        state["compiled_scope"] = compiled_scope
        usable_before = max(
            len(existing_details),
            int(
                (state.get("generation_readiness") or {}).get("usable_reference_count")
                or 0
            ),
        ) if incremental else 0
        incremental_minimum = int(
            state.get("incremental_required_new_evidence") or 0
        ) if incremental else 0
        required_to_fetch = (
            max(1, incremental_minimum, required_count - usable_before)
            if incremental and required_count else required_count
        )
        if required_to_fetch:
            increment_target = int(
                math.ceil(required_to_fetch * (1.0 + settings.evidence_pool_reserve_ratio))
            )
            observed_yield = state.get("evidence_yield") or {}
            absolute_target = absolute_evidence_pool_target(
                required_count, observed_yield, settings.evidence_pool_reserve_ratio
            )
            # WHY: 预留余量必须加在"绝对池目标"上，不能只加在本轮增量上。旧实现
            # 增量轮 required_to_fetch = max(1, 40-30) = 10 → ceil(10×1.5) = 15，
            # 池目标从首轮的 60 缩到 15（与日志逐字吻合），补检索再也补不回缺口。
            # 取三者最大值：绝对目标、按增量算的目标、以及前几轮已持久化的目标。
            evidence_pool_target = max(
                required_to_fetch,
                increment_target,
                absolute_target,
                int(state.get("evidence_pool_target") or 0),
            )
            if generation_limit:
                evidence_pool_target = min(evidence_pool_target, generation_limit)
        else:
            observed_yield = state.get("evidence_yield") or {}
            absolute_target = 0
            evidence_pool_target = generation_limit or len(ranked)
        state["evidence_pool_target"] = evidence_pool_target
        initial_limit = min(len(ranked), evidence_pool_target)

        papers: list[dict] = []
        validated: list[dict] = []
        thesis_excluded = 0
        cursor = 0

        # 先补全生成阶段会实际使用的论文。只有主题复核后不足用户要求时，
        # 才继续处理后续候选，避免为 3 倍召回池无条件发送外部请求。
        while cursor < len(ranked):
            if should_cancel and should_cancel():
                raise AgentCancelledError("任务已在论文详情补全批次间取消")
            if cursor == 0:
                batch_end = initial_limit
            else:
                missing = max(1, required_to_fetch - len(validated))
                batch_end = min(len(ranked), cursor + max(10, missing))

            batch = fetch_batch_details(ranked[cursor:batch_end])
            if should_cancel and should_cancel():
                raise AgentCancelledError("任务已在论文详情补全后取消")
            papers.extend(batch)
            research_mode = str(
                (state.get("research_semantic_frame") or {}).get("research_mode") or ""
            )
            screening_protocol = state.get("screening_protocol") or {}
            for paper in batch:
                # WHY: 详情补全会新增摘要、venue 和 DOI；这里必须重放排序阶段
                # 的完整硬规则。旧分支一旦存在 protocol 就只验 protocol，导致
                # scope/topic/branch 被绕过，越界论文可重新进入证据池。
                language_branch = str(
                    paper.get("_language_branch") or detect_paper_language(paper)
                )
                passed, stage, _ = evaluate_paper_hard_filters(
                    paper,
                    topic=state.get("topic", ""),
                    keywords=state.get("keywords") or [],
                    required_concepts=state.get("required_concepts") or [],
                    excluded_title_terms=state.get("excluded_title_terms") or [],
                    scope=state.get("selected_scope") or {},
                    search_branches=state.get("search_branches") or [],
                    research_mode=research_mode,
                    screening_protocol=screening_protocol,
                    language_branch=language_branch,
                    compiled_scope=compiled_scope,
                )
                if passed:
                    validated.append(paper)
                elif stage == "document_type_filter":
                    thesis_excluded += 1
            cursor = batch_end
            if not required_to_fetch or len(validated) >= required_to_fetch:
                break

        if incremental:
            from app.utils.deduplicate import deduplicate_papers

            state["paper_details"] = deduplicate_papers([*existing_details, *validated])
            state["incremental_new_paper_ids"] = [
                str(paper.get("paper_id") or "")
                for paper in validated if paper.get("paper_id")
            ]
        else:
            state["paper_details"] = validated
        state["retrieval_requirement_met"] = (
            len(state["paper_details"])
            >= int(state.get("required_reference_count") or state.get("max_papers") or 0)
        )
        append_step(
            state, "fetch_detail", "success",
            tool_name="fetch_metadata",
            input_data={
                "ranked": len(ranked_all),
                "new_ranked": len(ranked),
                "reused_details": len(existing_details),
                "incremental": incremental,
                "initial_detail_limit": initial_limit,
                "required_reference_count": required_count,
                "usable_before": usable_before,
                "required_new_evidence": required_to_fetch,
                "generation_limit": generation_limit,
                "evidence_pool_target": evidence_pool_target,
                "absolute_pool_target": absolute_target,
                "evidence_yield": observed_yield,
                "compiled_scope": {
                    "version": compiled_scope.get("version"),
                    "fingerprint": compiled_scope.get("fingerprint"),
                },
                "ranked_sample": _summarize_papers(ranked, limit=10),
            },
            output_data={
                "fetched": len(papers),
                "skipped_unneeded": len(ranked) - len(papers),
                "retained": len(state["paper_details"]),
                "newly_retained": len(validated),
                "discarded_after_validation": len(papers) - len(validated),
                "thesis_excluded": thesis_excluded,
                "fetched_sample": _summarize_papers(papers, limit=10),
                "retained_sample": _summarize_papers(validated, limit=10),
            },
            duration_ms=int((time.time() - t0) * 1000),
        )
    except AgentCancelledError:
        # 协作式取消必须继续向上传播，不能被当作普通失败吞掉。
        raise
    except Exception as e:
        from app.agent.exceptions import ServiceUnavailableError

        error = ServiceUnavailableError(str(e), step="fetch_detail", original_error=e)
        logger.error("fetch_detail_node failed: %s", error.message)
        state.setdefault("errors", []).append(f"fetch_detail: {e}")
        append_step(state, "fetch_detail", "failed", error=str(e))
    return state




@node(name="retrieval_shortfall", category="generation", description="未获得任何可用论文时，无法生成相关工作")
@provides("review", "generation_blocked")
def retrieval_shortfall_node(state: "ResearchAgentState") -> "ResearchAgentState":
    """未获得任何可用论文时，按用户实际请求的交付物给出说明。"""
    target = int(state.get("required_reference_count") or state.get("max_papers") or 0)
    details = state.get("paper_details")
    actual = len(details if details is not None else (state.get("ranked_papers") or []))
    start_year = state.get("start_year")
    end_year = state.get("end_year")
    requirement_source = (
        "用户要求" if state.get("max_papers_explicit", False) else "系统默认要求"
    )
    requested_sections = set(state.get("requested_sections") or [])
    if {"background", "research_status"}.issubset(requested_sections):
        deliverable_label = "研究背景和研究现状"
    elif "background" in requested_sections:
        deliverable_label = "研究背景"
    elif "research_status" in requested_sections:
        deliverable_label = "研究现状"
    elif "related_work" in requested_sections:
        deliverable_label = "相关工作"
    else:
        deliverable_label = "叙述性综述"
    # 契约声明了 generation_blocked 就必须落实：否则 derive_result_status
    # 五个分支无一命中，检索零结果会被误报为 success。
    state["generation_blocked"] = True
    state["quality_gate"] = {
        "passed": False,
        "phase": "pre_generation",
        "blocking_issues": [{
            "code": "retrieval_shortfall_no_usable_papers",
            "message": f"未获得可用论文，无法生成{deliverable_label}",
        }],
        "recovery_options": ["扩大年份范围", "调整关键词后重新提交研究请求"],
    }
    state["review"] = (
        f"## 未生成{deliverable_label}\n\n"
        f"{requirement_source}期望引用约 {target} 篇论文，但在 {start_year}-{end_year} 年范围内，"
        f"经多关键词检索、去重和主题相关性筛选后未获得可用论文。"
        f"系统无法生成可靠{deliverable_label}。请扩大年份范围、调整关键词或稍后重试。"
    )
    append_step(
        state,
        "retrieval_shortfall",
        "failed",
        tool_name="generate_review",
        input_data={
            "target": target,
            "actual": actual,
            "start_year": start_year,
            "end_year": end_year,
            "paper_sample": _summarize_papers(details or state.get("ranked_papers") or [], limit=10),
        },
        output_data={
            "reason": "minimum_paper_count_not_met",
            "target": target,
            "actual": actual,
            "review_preview": _preview_text(state.get("review"), limit=1200),
        },
        error=f"最低论文数量未满足：要求 {target}，实际 {actual}",
        duration_ms=0,
    )
    return state

