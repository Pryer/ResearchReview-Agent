"""OpenAlex API 客户端。

OpenAlex 提供完全开放的学术元数据 API，无需 key。
文档：https://docs.openalex.org/
"""

from __future__ import annotations

import random
import threading
import time
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


OPENALEX_API_URL = "https://api.openalex.org/works"

# 未配置真实 mailto 时 OpenAlex 只按 common pool 计费，polite pool 的 10 req/s
# 会直接换来 429。限流器按 polite 速率初始化，故此处显式降速。
_COMMON_POOL_RATE = 2.0
_PLACEHOLDER_MAILTO = "research-review-agent@example.com"

# 429 惩罚窗口长于熔断器 30s 冷却，仅靠熔断探针会反复撞墙。用进程级冷却门
# 挡住整轮并发任务，避免 HALF_OPEN 探针一放出就再次跳闸。
_RATE_LOCK = threading.Lock()
_COOLDOWN_UNTIL = 0.0


def _effective_mailto() -> str:
    """返回可用的 mailto；未配置时回落到占位地址（不进 polite pool）。"""
    return settings.crossref_mailto.strip() or _PLACEHOLDER_MAILTO


def _get_limiter():
    """按 mailto 是否真实配置选择限流档位。"""
    from app.core.rate_limiter import get_rate_limiter

    limiter = get_rate_limiter("openalex")
    if not settings.crossref_mailto.strip() and limiter.rate > _COMMON_POOL_RATE:
        # 桶容量同步收敛到 1，抑制同一秒内多关键词并发涌出的突发流量。
        limiter.rate = _COMMON_POOL_RATE
        limiter.capacity = 1.0
        limiter.tokens = min(limiter.tokens, limiter.capacity)
        logger.info(
            "OpenAlex mailto not configured; rate limited to %.1f req/s (common pool)",
            _COMMON_POOL_RATE,
        )
    return limiter


def _retry_after_seconds(resp: requests.Response) -> float:
    value = resp.headers.get("Retry-After")
    if not value:
        return 0.0
    try:
        return max(0.0, float(value))
    except ValueError:
        return 0.0


def _cooldown_remaining() -> float:
    with _RATE_LOCK:
        return _COOLDOWN_UNTIL - time.monotonic()


def _enter_cooldown(seconds: float, retry_after: float = 0.0) -> None:
    """进入 429 冷却，时长钳制在 ``openalex_max_cooldown_seconds`` 以内。

    OpenAlex 在日配额耗尽时会返回指向"次日零点"的 Retry-After（实测达 41559
    秒 / 11.5 小时）。无条件采纳会让本进程整天不再访问 OpenAlex，代价远大于
    定期探测——429 立即返回且不传数据，探测成本极低。因此钳制上限，并对被
    钳制的情形单独记日志，使"日配额耗尽"在日志中可辨识。
    """
    global _COOLDOWN_UNTIL
    ceiling = max(1.0, float(settings.openalex_max_cooldown_seconds or 1.0))
    requested = max(0.0, seconds)
    effective = min(requested, ceiling)
    if requested > ceiling:
        logger.warning(
            "OpenAlex Retry-After %.0fs exceeds cooldown ceiling; capped to %.0fs "
            "(likely daily quota exhausted on the shared common pool — "
            "configure CROSSREF_MAILTO to use the polite pool)",
            retry_after or requested,
            ceiling,
        )
    with _RATE_LOCK:
        _COOLDOWN_UNTIL = max(_COOLDOWN_UNTIL, time.monotonic() + effective)


def _reset_cooldown() -> None:
    """仅供测试与显式恢复使用。"""
    global _COOLDOWN_UNTIL
    with _RATE_LOCK:
        _COOLDOWN_UNTIL = 0.0


def search_openalex(
    query: str,
    start_year: int,
    end_year: int,
    max_results: int = 20,
    sort_by: str = "relevance",
) -> List[PaperMetadata]:
    """调用 OpenAlex 检索论文。

    Args:
        sort_by: 排序方式，"relevance"（默认）或 "date"（按发表日期降序）。
    """
    from app.core.circuit_breaker import get_circuit_breaker

    cb = get_circuit_breaker("openalex")
    if not cb.allow_request():
        logger.warning("OpenAlex search skipped due to active circuit breaker")
        return []

    remaining = _cooldown_remaining()
    if remaining > 0:
        logger.warning(
            "OpenAlex search skipped: rate limit cooldown for another %.1fs", remaining,
        )
        return []

    limiter = _get_limiter()

    # 排序映射
    oa_sort = "publication_date:desc" if sort_by == "date" else "relevance_score:desc"

    oa_query = query.strip()
    if sort_by == "date" and " " in oa_query and '"' not in oa_query:
        oa_query = f'"{oa_query}"'

    params = {
        "search": oa_query,
        "per-page": min(max_results, 200),
        "filter": f"publication_year:{start_year}-{end_year}",
        "sort": oa_sort,
        "mailto": _effective_mailto(),
    }

    max_retries = max(0, int(settings.openalex_max_retries or 0))
    max_wait = max(0.0, float(settings.openalex_max_retry_wait_seconds or 0))
    for attempt in range(max_retries + 1):
        if not limiter.acquire(1.0, timeout=10.0):
            # 令牌等待超时说明本域已严重超速：跳过本次请求而不是照发不误，
            # 否则限流器形同虚设，只会换来 429 封禁。
            logger.warning("OpenAlex search skipped: rate limit token wait timed out")
            _record_search_diagnostic("rate_limited", error_code="RATE_LIMIT_WAIT_TIMEOUT", message="OpenAlex 限流等待超时")
            return []
        try:
            resp = requests.get(
                OPENALEX_API_URL,
                params=params,
                timeout=settings.agent_request_timeout,
            )
        except requests.RequestException as e:
            cb.record_failure(e)
            logger.warning("OpenAlex search failed (sort=%s): %s", sort_by, e)
            _record_search_diagnostic("timeout" if isinstance(e, requests.Timeout) else "api_failed", error_code="TIMEOUT" if isinstance(e, requests.Timeout) else "API_ERROR", message=str(e))
            return []

        if resp.status_code == 429:
            retry_after = _retry_after_seconds(resp)
            # 优先遵循 Retry-After；服务端未提供时指数退避并加 Jitter，
            # 避免多个并发任务在同一时刻齐步重试。
            base_sleep = retry_after or min(max_wait, 1.5 ** (attempt + 1))
            sleep_seconds = base_sleep + random.uniform(0.1, 0.8)
            logger.warning(
                "OpenAlex rate limited: status=429 retry_after=%s attempt=%d/%d sort=%s",
                resp.headers.get("Retry-After"),
                attempt + 1,
                max_retries + 1,
                sort_by,
            )
            if attempt < max_retries and sleep_seconds <= max_wait:
                time.sleep(sleep_seconds)
                continue
            # 重试用尽：进入进程级冷却，并且不把 429 计入熔断失败——
            # 限流是本端速率问题，不是 OpenAlex 不可用。
            _enter_cooldown(
                max(retry_after, float(settings.openalex_cooldown_seconds or 0)),
                retry_after=retry_after,
            )
            _record_search_diagnostic("rate_limited", error_code="HTTP_429", message="OpenAlex 返回 HTTP 429；已进入限流冷却")
            return []

        try:
            resp.raise_for_status()
        except requests.RequestException as e:
            cb.record_failure(e)
            logger.warning("OpenAlex search failed (sort=%s): %s", sort_by, e)
            _record_search_diagnostic("timeout" if isinstance(e, requests.Timeout) else "api_failed", error_code="TIMEOUT" if isinstance(e, requests.Timeout) else "API_ERROR", message=str(e))
            return []

        cb.record_success()
        return parse_openalex_response(resp.json())
    return []


def parse_openalex_response(response_json: dict) -> List[PaperMetadata]:
    """解析 OpenAlex JSON 响应。"""
    papers: List[PaperMetadata] = []
    results = response_json.get("results", [])

    for item in results:
        try:
            raw_title = item.get("title", "") or ""
            title = clean_title(raw_title)
            if not title:
                continue

            # 作者
            authorships = item.get("authorships") or []
            authors = [
                a.get("author", {}).get("display_name", "")
                for a in authorships
                if a.get("author")
            ]

            # 年份
            year = item.get("publication_year")

            # DOI
            doi = item.get("doi", "")
            if doi:
                doi = doi.replace("https://doi.org/", "")

            # 开放获取 PDF
            open_access = item.get("open_access") or {}
            pdf_url = open_access.get("oa_url") if open_access.get("is_oa") else None

            # 引用量
            citation_count = item.get("cited_by_count")
            citation_count_by_source = (
                {"openalex": citation_count} if citation_count is not None else None
            )

            # venue
            host_venue = item.get("primary_location", {}).get("source", {}) or {}
            venue = host_venue.get("display_name")

            raw_abstract = _reconstruct_abstract(item.get("abstract_inverted_index"))

            paper = PaperMetadata(
                paper_id=f"openalex:{item.get('id', '').split('/')[-1]}",
                title=title,
                authors=authors,
                year=year,
                venue=venue,
                abstract=clean_abstract(raw_abstract),
                doi=doi or None,
                url=item.get("doi") or item.get("id"),
                pdf_url=pdf_url,
                citation_count=citation_count,
                citation_count_by_source=citation_count_by_source,
                source="openalex",
                is_open_access=bool(pdf_url) or open_access.get("is_oa", False),
            )
            papers.append(paper)

        except Exception as e:
            logger.debug("Failed to parse OpenAlex entry: %s", e)
            continue

    logger.info("OpenAlex parsed %d papers", len(papers))
    return papers


def get_openalex_detail(work_id: str) -> Optional[dict]:
    """获取 OpenAlex work 详情。"""
    from app.core.circuit_breaker import get_circuit_breaker

    # detail 接口与 search 共用同一熔断器、限流器与 429 冷却门。
    cb = get_circuit_breaker("openalex")
    if not cb.allow_request():
        logger.warning("OpenAlex detail skipped due to active circuit breaker")
        return None
    remaining = _cooldown_remaining()
    if remaining > 0:
        logger.warning(
            "OpenAlex detail skipped: rate limit cooldown for another %.1fs", remaining,
        )
        return None
    limiter = _get_limiter()
    if not limiter.acquire(1.0, timeout=10.0):
        logger.warning("OpenAlex detail skipped: rate limit token wait timed out")
        return None

    url = f"{OPENALEX_API_URL}/{work_id}"
    params = {"mailto": _effective_mailto()}

    try:
        resp = requests.get(url, params=params, timeout=settings.agent_request_timeout)
        if resp.status_code == 404:
            return None
        if resp.status_code == 429:
            # 与 search 一致：429 不计入熔断失败，改为进入冷却门。
            retry_after = _retry_after_seconds(resp)
            _enter_cooldown(
                max(retry_after, float(settings.openalex_cooldown_seconds or 0)),
                retry_after=retry_after,
            )
            logger.warning(
                "OpenAlex detail rate limited: status=429 retry_after=%s",
                resp.headers.get("Retry-After"),
            )
            return None
        resp.raise_for_status()
        cb.record_success()
        return resp.json()
    except requests.RequestException as e:
        cb.record_failure(e)
        logger.debug("OpenAlex detail fetch failed: %s", e)
        return None


def _reconstruct_abstract(inverted_index: Optional[dict]) -> Optional[str]:
    """从 OpenAlex 的 inverted index 重建摘要文本。"""
    if not inverted_index:
        return None
    # inverted_index: {"word": [pos1, pos2, ...]}
    word_positions: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort(key=lambda x: x[0])
    return " ".join(w for _, w in word_positions)
