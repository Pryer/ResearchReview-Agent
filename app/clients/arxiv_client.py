"""arXiv API 客户端。

arXiv 提供免费的 Atom XML API，无需 API key。
文档：https://arxiv.org/help/api/user-manual
"""

from __future__ import annotations

import time
import threading
import xml.etree.ElementTree as ET
from typing import List, Optional
from urllib.parse import urlencode

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


ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"

# arXiv 建议单个客户端每 3 秒一次请求；高频会被 429 或连接超时
_ARXIV_REQUEST_INTERVAL = 3.0
_last_request_time: float = 0.0
_request_lock = threading.Lock()


def _reserve_request_slot() -> None:
    """原子预留请求时间，保证多检索线程之间仍满足全局间隔。"""
    global _last_request_time
    with _request_lock:
        now = time.monotonic()
        scheduled = max(now, _last_request_time + _ARXIV_REQUEST_INTERVAL)
        _last_request_time = scheduled
    wait = scheduled - now
    if wait > 0:
        time.sleep(wait)


def _arxiv_get(url: str, timeout: int, max_retries: int = 3) -> requests.Response:
    """带间隔控制和指数退避的 arXiv GET 请求。"""
    from app.core.rate_limiter import get_rate_limiter

    limiter = get_rate_limiter("arxiv")
    for attempt in range(1, max_retries + 1):
        try:
            if not limiter.acquire(1.0, timeout=10.0):
                # 令牌等待超时说明本域已严重超速：直接放弃本次请求，
                # 与串行间隔控制共同生效，避免 429 封禁。
                logger.warning("arXiv request skipped: rate limit token wait timed out")
                raise requests.RequestException("arXiv rate limit token wait timed out")
            _reserve_request_slot()
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 429:
                wait = min(10 * attempt, 30)
                logger.warning("arXiv rate limited (429), waiting %ds before retry %d/%d", wait, attempt, max_retries)
                _record_search_diagnostic("rate_limited", error_code="HTTP_429", message="arXiv 返回 HTTP 429，正在退避重试")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except requests.Timeout:
            if attempt < max_retries:
                wait = 5 * attempt
                logger.warning("arXiv timeout on attempt %d/%d, retrying in %ds", attempt, max_retries, wait)
                time.sleep(wait)
            else:
                raise
        except requests.RequestException:
            raise
    raise requests.RequestException("arXiv max retries exceeded")


def _format_arxiv_query(query: str, force_phrase: bool = False) -> str:
    """构造适用于 arXiv API 的检索表达式。

    处理规则：
    1. 若已有字段前缀（如 ti:, abs:, all:）或已包含引号，直接返回。
    2. 若为多词英文短语（包含空格且无显式布尔操作符），加上双引号构造精确短语 all:"phrase"。
    3. 其他情况返回 all:query。
    """
    clean_q = query.strip()
    if not clean_q:
        return "all:"

    if clean_q.startswith(('all:', 'ti:', 'abs:', 'cat:', 'au:')) or '"' in clean_q:
        return clean_q

    # 检查是否为多词短语
    has_spaces = " " in clean_q
    has_bool_ops = any(op in clean_q.upper().split() for op in ("AND", "OR", "ANDNOT"))

    if (has_spaces and not has_bool_ops) or force_phrase:
        return f'all:"{clean_q}"'

    return f"all:{clean_q}"


def search_arxiv(
    query: str,
    start_year: int,
    end_year: int,
    max_results: int = 20,
    sort_by: str = "relevance",
) -> List[PaperMetadata]:
    """调用 arXiv API 检索论文。

    Args:
        query: 检索关键词。
        start_year: 起始年份。
        end_year: 结束年份。
        max_results: 最大返回数。
        sort_by: 排序方式，"relevance"（默认）或 "date"（按提交时间降序）。

    Returns:
        论文元数据列表。
    """
    # 排序映射
    if sort_by == "date":
        arxiv_sort_by = "submittedDate"
    else:
        arxiv_sort_by = "relevance"

    formatted_query = _format_arxiv_query(query, force_phrase=(sort_by == "date"))

    def _execute_query(search_q: str) -> List[PaperMetadata]:
        params = {
            "search_query": search_q,
            "start": 0,
            "max_results": min(max_results, 50),
            "sortBy": arxiv_sort_by,
            "sortOrder": "descending",
        }
        url = f"{ARXIV_API_URL}?{urlencode(params)}"
        try:
            resp = _arxiv_get(url, timeout=max(settings.agent_request_timeout, 30))
            return parse_arxiv_response(resp.text, start_year, end_year)
        except requests.RequestException as e:
            logger.warning("arXiv search failed (sort=%s, query=%s): %s", sort_by, search_q, e)
            status_code = getattr(getattr(e, "response", None), "status_code", None)
            outcome = "timeout" if isinstance(e, requests.Timeout) else (
                "authentication_failed" if status_code in {401, 403} else "api_failed"
            )
            error_code = "TIMEOUT" if outcome == "timeout" else (
                f"HTTP_{status_code}" if status_code in {401, 403} else "API_ERROR"
            )
            _record_search_diagnostic(outcome, error_code=error_code, message=str(e))
            return []

    # 1. 执行精准检索
    papers = _execute_query(formatted_query)

    # 2. 如果精准短语返回 0 条且原始 query 未加引号，平滑回退到松散分词检索
    if not papers and formatted_query != f"all:{query.strip()}" and '"' not in query:
        fallback_q = f"all:{query.strip()}"
        logger.info("arXiv phrase query returned 0, falling back to loose query: %s", fallback_q)
        papers = _execute_query(fallback_q)

    return papers


def parse_arxiv_response(
    response_text: str,
    start_year: int,
    end_year: int,
    ) -> List[PaperMetadata]:
    """解析 arXiv Atom XML 响应。"""
    papers: List[PaperMetadata] = []

    try:
        root = ET.fromstring(response_text)
    except ET.ParseError as e:
        logger.warning("Failed to parse arXiv XML: %s", e)
        return []

    for entry in root.findall(f"{ATOM_NS}entry"):
        try:
            title = clean_title(entry.findtext(f"{ATOM_NS}title", ""))
            summary = clean_abstract(entry.findtext(f"{ATOM_NS}summary", ""))
            published = entry.findtext(f"{ATOM_NS}published", "")

            # 解析年份
            year = None
            if published:
                try:
                    year = int(published[:4])
                except ValueError:
                    pass

            # 年份过滤
            if year and (year < start_year or year > end_year):
                continue

            # arXiv ID
            arxiv_id_url = entry.findtext(f"{ATOM_NS}id", "")
            arxiv_id = arxiv_id_url.split("/abs/")[-1] if "/abs/" in arxiv_id_url else arxiv_id_url

            # 版本号去除
            if "v" in arxiv_id:
                arxiv_id = arxiv_id.split("v")[0]

            # 作者
            authors = [
                author.findtext(f"{ATOM_NS}name", "").strip()
                for author in entry.findall(f"{ATOM_NS}author")
            ]

            # PDF URL
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

            # DOI
            doi = None
            for link in entry.findall(f"{ATOM_NS}link"):
                if link.get("title") == "doi":
                    doi = link.get("href", "").replace("https://doi.org/", "")
                    break

            paper = PaperMetadata(
                paper_id=f"arxiv:{arxiv_id}",
                title=title,
                authors=authors,
                year=year,
                venue="arXiv",
                abstract=summary,
                doi=doi,
                arxiv_id=arxiv_id,
                url=f"https://arxiv.org/abs/{arxiv_id}",
                pdf_url=pdf_url,
                citation_count=None,
                source="arxiv",
                is_open_access=True,
            )
            papers.append(paper)

        except Exception as e:
            logger.debug("Failed to parse arXiv entry: %s", e)
            continue

    logger.info("arXiv parsed %d papers", len(papers))
    return papers


def get_arxiv_pdf_url(arxiv_id: str) -> str:
    """生成 arXiv PDF URL。"""
    return f"https://arxiv.org/pdf/{arxiv_id}"
