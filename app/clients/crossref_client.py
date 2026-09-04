"""Crossref API 客户端。

Crossref 提供 DOI 元数据查询，常用于补全论文详情。
文档：https://api.crossref.org/
"""

from __future__ import annotations

from typing import List, Optional

import requests

from app.core.config import get_settings
from app.core.logger import get_logger
from app.schemas.paper_schema import PaperMetadata
from app.utils.title_cleaner import clean_abstract, clean_title

logger = get_logger(__name__)
settings = get_settings()


def _record_search_diagnostic(outcome: str, *, error_code: str | None = None, message: str | None = None) -> None:
    try:
        from app.tools.search_papers import record_client_diagnostic
        record_client_diagnostic(outcome, error_code=error_code, message=message)
    except Exception:
        pass


CROSSREF_API_URL = "https://api.crossref.org/works"


def search_crossref(
    query: str,
    start_year: int,
    end_year: int,
    max_results: int = 20,
    sort_by: str = "relevance",
) -> List[PaperMetadata]:
    """调用 Crossref API 检索论文。

    Args:
        query: 检索关键词。
        start_year: 起始年份。
        end_year: 结束年份。
        max_results: 最大返回数。
        sort_by: 排序方式，"relevance"（默认）或 "date"（按发表时间降序）。

    Returns:
        论文元数据列表。
    """
    from app.core.circuit_breaker import get_circuit_breaker
    from app.core.rate_limiter import get_rate_limiter

    cb = get_circuit_breaker("crossref", failure_threshold=5, recovery_timeout=30.0)
    if not cb.allow_request():
        logger.warning("Crossref search skipped due to active circuit breaker")
        return []

    limiter = get_rate_limiter("crossref")
    if not limiter.acquire(1.0, timeout=10.0):
        # 令牌等待超时说明本域已严重超速：跳过本次请求而不是照发不误。
        logger.warning("Crossref search skipped: rate limit token wait timed out")
        return []

    # 排序映射
    cr_sort = "published" if sort_by == "date" else "relevance"

    # 在按时间排序时，若为多词英文短语则加引号以提升领域精准度
    cr_query = query.strip()
    if sort_by == "date" and " " in cr_query and '"' not in cr_query:
        cr_query = f'"{cr_query}"'

    params = {
        "query": cr_query,
        "rows": min(max_results, 50),
        "filter": f"from-pub-date:{start_year}-01-01,until-pub-date:{end_year}-12-31",
        "sort": cr_sort,
        "order": "desc",
        "mailto": settings.crossref_mailto or "research-review-agent@example.com",
    }

    try:
        resp = requests.get(
            CROSSREF_API_URL,
            params=params,
            timeout=settings.agent_request_timeout,
        )
        resp.raise_for_status()
        cb.record_success()
        return parse_crossref_response(resp.json())
    except requests.RequestException as e:
        cb.record_failure(e)
        logger.warning("Crossref search failed (sort=%s): %s", sort_by, e)
        status_code = getattr(getattr(e, "response", None), "status_code", None)
        outcome = "timeout" if isinstance(e, requests.Timeout) else (
            "authentication_failed" if status_code in {401, 403} else "api_failed"
        )
        error_code = "TIMEOUT" if outcome == "timeout" else (
            f"HTTP_{status_code}" if status_code in {401, 403} else "API_ERROR"
        )
        _record_search_diagnostic(outcome, error_code=error_code, message=str(e))
        return []


def get_crossref_detail(doi: str) -> Optional[dict]:
    """根据 DOI 获取详情。"""
    from app.core.circuit_breaker import get_circuit_breaker
    from app.core.rate_limiter import get_rate_limiter

    # detail 接口与 search 共用同一熔断器与限流器：此前完全绕过。
    cb = get_circuit_breaker("crossref", failure_threshold=5, recovery_timeout=30.0)
    if not cb.allow_request():
        logger.warning("Crossref detail skipped due to active circuit breaker")
        return None
    limiter = get_rate_limiter("crossref")
    if not limiter.acquire(1.0, timeout=10.0):
        logger.warning("Crossref detail skipped: rate limit token wait timed out")
        return None

    url = f"{CROSSREF_API_URL}/{doi}"
    headers = {
        "User-Agent": "ResearchReview-Agent/0.1 (mailto:research-review-agent@example.com)"
    }

    try:
        resp = requests.get(url, headers=headers, timeout=settings.agent_request_timeout)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        cb.record_success()
        items = resp.json().get("message", {})
        return _crossref_item_to_dict(items)
    except requests.RequestException as e:
        cb.record_failure(e)
        logger.debug("Crossref detail failed: %s", e)
        return None


def parse_crossref_response(response_json: dict) -> List[PaperMetadata]:
    """解析 Crossref JSON 响应。"""
    papers: List[PaperMetadata] = []
    items = response_json.get("message", {}).get("items", [])

    for item in items:
        try:
            detail = _crossref_item_to_dict(item)
            if detail:
                papers.append(PaperMetadata(**detail))
        except Exception as e:
            logger.debug("Failed to parse Crossref item: %s", e)
            continue

    logger.info("Crossref parsed %d papers", len(papers))
    return papers


def _crossref_item_to_dict(item: dict) -> Optional[dict]:
    """将 Crossref item 转为 PaperMetadata 兼容字典。"""
    title_list = item.get("title") or []
    raw_title = title_list[0] if title_list else ""
    clean_t = clean_title(raw_title)
    if not clean_t:
        return None

    authors = [
        f"{a.get('given', '')} {a.get('family', '')}".strip()
        for a in (item.get("author") or [])
    ]

    # 年份
    date_parts = item.get("published", {}).get("date-parts", [[]])
    year = date_parts[0][0] if date_parts and date_parts[0] else None

    doi = item.get("DOI", "")

    license_urls = [
        str(license_item.get("URL") or "").lower()
        for license_item in item.get("license") or []
        if isinstance(license_item, dict)
    ]
    has_open_license = any(
        "creativecommons.org/licenses/" in url
        or "creativecommons.org/publicdomain/" in url
        for url in license_urls
    )

    # PDF 链接
    pdf_url = None
    for link in item.get("link") or []:
        if link.get("content-type") == "application/pdf":
            pdf_url = link.get("URL")
            break

    return {
        "paper_id": f"doi:{doi}",
        "title": clean_t,
        "authors": authors,
        "year": year,
        "venue": item.get("container-title", [None])[0] if item.get("container-title") else None,
        "abstract": clean_abstract(item.get("abstract", "")),
        "doi": doi,
        "url": item.get("URL"),
        "pdf_url": pdf_url,
        "citation_count": item.get("is-referenced-by-count"),
        "source": "crossref",
        # Crossref 的 PDF link 只表示存在落地链接，不代表可公开下载。
        "is_open_access": bool(pdf_url and has_open_license),
    }
