"""Crossref API 客户端。

Crossref 提供 DOI 元数据查询，常用于补全论文详情。
文档：https://api.crossref.org/
"""

from __future__ import annotations

from typing import List, Optional
import random
import threading
import time

import requests

from app.core.config import get_settings
from app.core.logger import get_logger
from app.core.retry_policy import (
    bounded_request_timeout,
    interruptible_sleep,
    retry_after_seconds,
    check_retrieval_execution,
)
from app.schemas.paper_schema import PaperMetadata
from app.utils.title_cleaner import clean_abstract, clean_title

logger = get_logger(__name__)
settings = get_settings()


def _record_search_diagnostic(
    outcome: str, *, error_code: str | None = None, message: str | None = None
) -> None:
    try:
        from app.tools.search_papers import record_client_diagnostic

        record_client_diagnostic(outcome, error_code=error_code, message=message)
    except Exception:
        pass


CROSSREF_API_URL = "https://api.crossref.org/works"


# A shared source-level cooldown protects search and DOI-detail requests.
# HTTP 429 is quota pressure, not evidence that Crossref is unavailable.
_COOLDOWN_LOCK = threading.Lock()
_COOLDOWN_UNTIL = 0.0


def _cooldown_remaining() -> float:
    with _COOLDOWN_LOCK:
        return max(0.0, _COOLDOWN_UNTIL - time.monotonic())


def _enter_cooldown(seconds: float, *, retry_after: float = 0.0) -> None:
    global _COOLDOWN_UNTIL
    # WHY: 只钳制本地兜底冷却；服务端 Retry-After 是最早重试时刻，不能提前。
    ceiling = max(1.0, float(settings.crossref_max_cooldown_seconds))
    with _COOLDOWN_LOCK:
        _COOLDOWN_UNTIL = max(
            _COOLDOWN_UNTIL,
            time.monotonic() + max(retry_after, min(max(0.0, seconds), ceiling)),
        )


def _reset_cooldown() -> None:
    """Test and explicit recovery hook."""
    global _COOLDOWN_UNTIL
    with _COOLDOWN_LOCK:
        _COOLDOWN_UNTIL = 0.0


def _request_ready(cb) -> bool:
    check_retrieval_execution()
    if not cb.allow_request():
        _record_search_diagnostic(
            "api_failed", error_code="CIRCUIT_OPEN", message="Crossref circuit breaker is open"
        )
        return False
    if (remaining := _cooldown_remaining()) > 0:
        _record_search_diagnostic(
            "rate_limited",
            error_code="HTTP_429",
            message=f"Crossref cooldown: {remaining:.1f}s remaining",
        )
        return False
    return True


def _request_crossref(
    url: str, *, params: dict | None = None, headers: dict | None = None
) -> requests.Response | None:
    """Bounded retrieval with distinct 429, transient and auth failure paths."""
    from app.core.circuit_breaker import get_circuit_breaker
    from app.core.rate_limiter import get_rate_limiter

    cb = get_circuit_breaker("crossref", failure_threshold=5, recovery_timeout=30.0)
    limiter = get_rate_limiter("crossref")
    retries = max(0, int(settings.crossref_max_retries))
    max_wait = max(0.0, float(settings.crossref_max_retry_wait_seconds))
    for attempt in range(retries + 1):
        if not _request_ready(cb):
            return None
        if not limiter.acquire(1.0, timeout=10.0):
            _record_search_diagnostic(
                "rate_limited",
                error_code="RATE_LIMIT_WAIT_TIMEOUT",
                message="Crossref token acquisition timed out",
            )
            return None
        if not _request_ready(cb):
            return None
        try:
            resp = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=bounded_request_timeout(settings.agent_request_timeout),
            )
        except (requests.Timeout, requests.ConnectionError) as exc:
            cb.record_failure(exc)
            if attempt < retries:
                interruptible_sleep(min(max_wait, 2**attempt))
                continue
            outcome = "timeout" if isinstance(exc, requests.Timeout) else "api_failed"
            _record_search_diagnostic(
                outcome,
                error_code="TIMEOUT" if outcome == "timeout" else "NETWORK_ERROR",
                message=f"Crossref {outcome}",
            )
            return None
        except requests.RequestException as exc:
            cb.record_failure(exc)
            _record_search_diagnostic(
                "api_failed", error_code="API_ERROR", message=f"Crossref {type(exc).__name__}"
            )
            return None

        if resp.status_code == 429:
            retry_after = retry_after_seconds(resp.headers)
            # Respect Retry-After; only attempt in-line retries when the bounded
            # wait can fit. Otherwise, enter cooldown and let other sources run.
            wait = (
                retry_after
                if retry_after > 0
                else min(max_wait, 2**attempt) + random.uniform(0.0, 0.25)
            )
            if attempt < retries and wait <= max_wait:
                _enter_cooldown(wait)
                interruptible_sleep(wait)
                continue
            _enter_cooldown(
                max(retry_after, float(settings.crossref_cooldown_seconds)), retry_after=retry_after
            )
            _record_search_diagnostic(
                "rate_limited", error_code="HTTP_429", message="Crossref HTTP 429; cooldown started"
            )
            return None
        if resp.status_code in {401, 403}:
            _record_search_diagnostic(
                "authentication_failed",
                error_code=f"HTTP_{resp.status_code}",
                message="Crossref authentication/authorization failed",
            )
            return None
        if resp.status_code == 404:
            return None
        if resp.status_code in {500, 502, 503, 504}:
            cb.record_failure()
            if attempt < retries:
                interruptible_sleep(min(max_wait, 2**attempt))
                continue
            _record_search_diagnostic(
                "api_failed",
                error_code=f"HTTP_{resp.status_code}",
                message="Crossref transient server failure",
            )
            return None
        if resp.status_code >= 400:
            _record_search_diagnostic(
                "api_failed",
                error_code=f"HTTP_{resp.status_code}",
                message="Crossref HTTP request failed",
            )
            return None
        cb.record_success()
        return resp
    return None


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

    resp = _request_crossref(CROSSREF_API_URL, params=params)
    if resp is None:
        return []
    return parse_crossref_response(resp.json())


def get_crossref_detail(doi: str) -> Optional[dict]:
    """根据 DOI 获取详情。"""
    url = f"{CROSSREF_API_URL}/{doi}"
    headers = {"User-Agent": "ResearchReview-Agent/0.1 (mailto:research-review-agent@example.com)"}
    resp = _request_crossref(url, headers=headers)
    if resp is None:
        return None
    return _crossref_item_to_dict(resp.json().get("message", {}))


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
        f"{a.get('given', '')} {a.get('family', '')}".strip() for a in (item.get("author") or [])
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
        "creativecommons.org/licenses/" in url or "creativecommons.org/publicdomain/" in url
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
