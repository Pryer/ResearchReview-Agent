"""Semantic Scholar API 客户端。

Semantic Scholar 提供免费的学术搜索 API，无需 key（有速率限制）。
文档：https://api.semanticscholar.org/api-docs/
"""

from __future__ import annotations

import threading
import time
from typing import List, Optional

import requests

from app.core.config import get_settings
from app.core.logger import get_logger
from app.schemas.paper_schema import PaperMetadata

logger = get_logger(__name__)
settings = get_settings()


def _record_search_diagnostic(outcome: str, *, error_code: str | None = None, message: str | None = None) -> None:
    try:
        from app.tools.search_papers import record_client_diagnostic
        record_client_diagnostic(outcome, error_code=error_code, message=message)
    except Exception:
        pass

S2_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_S2_COOLDOWN_UNTIL = 0.0
_S2_NEXT_ALLOWED_AT = 0.0
_S2_RATE_LOCK = threading.Lock()


class SemanticScholarRateLimitError(RuntimeError):
    """Semantic Scholar 明确拒绝请求；用于区别“确实没有搜索结果”。"""

    error_code = "RATE_LIMITED"


def _headers() -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if settings.semantic_scholar_api_key:
        headers["x-api-key"] = settings.semantic_scholar_api_key
    return headers


def _retry_after_seconds(resp: requests.Response) -> float:
    value = resp.headers.get("Retry-After")
    if not value:
        return 0.0
    try:
        return max(0.0, float(value))
    except ValueError:
        return 0.0


def _respect_rate_limit() -> None:
    """Semantic Scholar key is limited to 1 request/sec across endpoints."""
    global _S2_NEXT_ALLOWED_AT

    interval = max(0.0, float(settings.semantic_scholar_min_interval_seconds or 0))
    # 无专用 key 时使用共享配额，主动降低请求频率。
    if not settings.semantic_scholar_api_key:
        interval = max(interval, 3.0)
    if interval <= 0:
        return

    with _S2_RATE_LOCK:
        now = time.monotonic()
        wait_seconds = _S2_NEXT_ALLOWED_AT - now
        if wait_seconds > 0:
            time.sleep(wait_seconds)
            now = time.monotonic()
        _S2_NEXT_ALLOWED_AT = now + interval


def _request_json(url: str, params: dict) -> Optional[dict]:
    """请求 Semantic Scholar，显式处理 429，避免同一轮任务持续撞限流。"""
    global _S2_COOLDOWN_UNTIL
    from app.core.circuit_breaker import get_circuit_breaker

    cb = get_circuit_breaker("semantic_scholar")
    if not cb.allow_request():
        logger.warning("Semantic Scholar request skipped due to active circuit breaker")
        return None

    from app.core.rate_limiter import get_rate_limiter

    limiter = get_rate_limiter("semanticscholar")

    with _S2_RATE_LOCK:
        now = time.monotonic()
        cooldown_remaining = _S2_COOLDOWN_UNTIL - now
    if cooldown_remaining > 0:
        _record_search_diagnostic("rate_limited", error_code="RATE_LIMITED", message=f"Semantic Scholar 正在限流冷却，剩余 {cooldown_remaining:.1f} 秒")
        raise SemanticScholarRateLimitError(
            f"Semantic Scholar 正在限流冷却，剩余 {cooldown_remaining:.1f} 秒"
        )

    max_retries = max(0, int(settings.semantic_scholar_max_retries or 0))
    max_wait = max(0.0, float(settings.semantic_scholar_max_retry_wait_seconds or 0))
    for attempt in range(max_retries + 1):
        try:
            if not limiter.acquire(1.0, timeout=10.0):
                # 令牌等待超时说明本域已严重超速：跳过本次请求而不是照发
                # 不误，否则限流器形同虚设，只会换来 429 封禁。
                logger.warning("Semantic Scholar request skipped: rate limit token wait timed out")
                return None
            _respect_rate_limit()
            resp = requests.get(
                url,
                params=params,
                headers=_headers(),
                timeout=settings.agent_request_timeout,
            )
            if resp.status_code == 429:
                retry_after = _retry_after_seconds(resp)
                import random
                # 优先遵循 Retry-After；服务端未提供时采用指数退避（带 Jitter）。
                base_sleep = retry_after or min(max_wait, (1.5 ** (attempt + 1)))
                sleep_seconds = base_sleep + random.uniform(0.1, 0.8)
                logger.warning(
                    "Semantic Scholar rate limited: status=429 retry_after=%s attempt=%d/%d url=%s",
                    resp.headers.get("Retry-After"),
                    attempt + 1,
                    max_retries + 1,
                    resp.url,
                )
                if attempt < max_retries and sleep_seconds <= max_wait:
                    time.sleep(sleep_seconds)
                    continue
                cooldown_dur = float(settings.semantic_scholar_cooldown_seconds or 0)
                if settings.semantic_scholar_api_key:
                    cooldown_dur = min(cooldown_dur, 15.0)
                with _S2_RATE_LOCK:
                    _S2_COOLDOWN_UNTIL = time.monotonic() + max(retry_after, cooldown_dur)
                cb.record_failure()
                _record_search_diagnostic("rate_limited", error_code="HTTP_429", message="Semantic Scholar 连续返回 HTTP 429；已停止本轮后续请求")
                raise SemanticScholarRateLimitError(
                    "Semantic Scholar 连续返回 HTTP 429；已停止本轮后续请求"
                )
            if resp.status_code >= 400:
                logger.warning(
                    "Semantic Scholar request failed: status=%s body=%s url=%s",
                    resp.status_code,
                    resp.text[:300],
                    resp.url,
                )
                cb.record_failure()
                outcome = "authentication_failed" if resp.status_code in {401, 403} else "api_failed"
                _record_search_diagnostic(outcome, error_code=f"HTTP_{resp.status_code}", message=f"Semantic Scholar HTTP {resp.status_code}")
                return None
            cb.record_success()
            return resp.json()
        except requests.RequestException as e:
            cb.record_failure(e)
            logger.warning("Semantic Scholar request failed: %s", e)
            outcome = "timeout" if isinstance(e, requests.Timeout) else "api_failed"
            _record_search_diagnostic(outcome, error_code="TIMEOUT" if outcome == "timeout" else "API_ERROR", message=str(e))
            return None
    return None


def search_semantic_scholar(
    query: str,
    start_year: int,
    end_year: int,
    max_results: int = 20,
    sort_by: str = "relevance",
) -> List[PaperMetadata]:
    """调用 Semantic Scholar 检索论文。

    Args:
        sort_by: 排序方式，"relevance"（默认）或 "date"。
                 Semantic Scholar 无原生日期排序，date 模式通过缩窄年份范围
                 （仅查最近 2 年）来偏重最新论文。
    """
    # date 模式：将年份范围缩窄至最近 2 年，提高前沿覆盖率
    effective_start = start_year
    if sort_by == "date":
        effective_start = max(start_year, end_year - 1)

    params = {
        "query": query,
        "limit": min(max_results, 50),
        "fields": "title,authors,year,venue,abstract,externalIds,citationCount,openAccessPdf,url",
        "year": f"{effective_start}-{end_year}",
    }

    data = _request_json(S2_API_URL, params)
    if data:
        return parse_semantic_scholar_response(data)
    if not settings.semantic_scholar_api_key:
        logger.info(
            "Semantic Scholar returned no data without API key; set SEMANTIC_SCHOLAR_API_KEY for a dedicated quota"
        )
    return []


def parse_semantic_scholar_response(response_json: dict) -> List[PaperMetadata]:
    """解析 Semantic Scholar JSON 响应。"""
    papers: List[PaperMetadata] = []
    data = response_json.get("data", [])

    for item in data:
        try:
            title = item.get("title", "")
            if not title:
                continue

            authors = [
                a.get("name", "") for a in (item.get("authors") or [])
            ]

            external_ids = item.get("externalIds") or {}
            doi = external_ids.get("DOI")
            arxiv_id = external_ids.get("ArXiv")

            open_access_pdf = item.get("openAccessPdf") or {}
            pdf_url = open_access_pdf.get("url") if open_access_pdf.get("status") == "GREEN" else None

            # 引用量：同时设置 citation_count 和 citation_count_by_source
            citation_count = item.get("citationCount")
            citation_count_by_source = (
                {"semantic_scholar": citation_count} if citation_count is not None else None
            )

            paper = PaperMetadata(
                paper_id=f"s2:{item.get('paperId', '')}",
                title=title,
                authors=authors,
                year=item.get("year"),
                venue=item.get("venue"),
                abstract=item.get("abstract"),
                doi=doi,
                arxiv_id=arxiv_id,
                url=item.get("url"),
                pdf_url=pdf_url,
                citation_count=citation_count,
                citation_count_by_source=citation_count_by_source,
                source="semantic_scholar",
                is_open_access=bool(pdf_url),
            )
            papers.append(paper)

        except Exception as e:
            logger.debug("Failed to parse S2 entry: %s", e)
            continue

    logger.info("Semantic Scholar parsed %d papers", len(papers))
    return papers


def get_semantic_scholar_detail(identifier: str) -> Optional[dict]:
    """获取 Semantic Scholar 论文详情。

    Args:
        identifier: paper_id 或 DOI（加 DOI: 前缀）。
    """
    fields = "title,authors,year,venue,abstract,externalIds,citationCount,openAccessPdf,url"
    url = f"https://api.semanticscholar.org/graph/v1/paper/{identifier}"
    params = {"fields": fields}

    data = _request_json(url, params)
    if not data:
        return None
    return {
        "paper_id": f"s2:{data.get('paperId', '')}",
        "title": data.get("title", ""),
        "authors": [a.get("name", "") for a in (data.get("authors") or [])],
        "year": data.get("year"),
        "venue": data.get("venue"),
        "abstract": data.get("abstract"),
        "citation_count": data.get("citationCount"),
        "open_access_pdf": (
            (data.get("openAccessPdf") or {}).get("url")
        ),
    }
