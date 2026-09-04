"""PDF 下载工具。

只下载开放获取 PDF，失败不中断整体流程。
"""

from __future__ import annotations

import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from app.core.config import get_settings
from app.core.exceptions import PDFDownloadError
from app.core.logger import get_logger
from app.utils.pdf_utils import (
    pdf_page_count as _pdf_page_count,
    validate_pdf_file as _validate_pdf_magic,
)
import requests

logger = get_logger(__name__)

_RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}


def is_cnki_paper(paper: Dict[str, Any]) -> bool:
    """兼容旧调用方；来源识别规则统一由 core 能力策略维护。"""
    from app.core.source_capabilities import source_name

    return source_name(paper) == "cnki"


def allows_pdf_download(paper: Dict[str, Any]) -> bool:
    """返回该来源是否允许进入 PDF 下载与解析流程。"""
    from app.core.source_capabilities import source_allows_full_text

    return source_allows_full_text(paper)


def is_open_access(paper: Dict[str, Any]) -> bool:
    """判断论文是否有明确授权的开放获取 PDF。

    ``pdf_url`` 只表示存在一个候选下载地址，不等价于开放获取授权。除 arXiv
    这类来源策略明确的仓储外，调用方必须提供可信的 ``is_open_access`` 元数据。
    """
    if not allows_pdf_download(paper):
        return False
    pdf_url = paper.get("pdf_url")
    if not pdf_url:
        return False
    if paper.get("is_open_access") is True:
        return True
    source = str(paper.get("source") or "").strip().lower()
    paper_id = str(paper.get("paper_id") or "").strip().lower()
    # arXiv 是开放仓储，可以由稳定的来源身份直接判定。
    if source == "arxiv" or paper_id.startswith("arxiv:") or "arxiv.org" in str(pdf_url).lower():
        return True
    return False


def _pdf_path_for(paper: Dict[str, Any], save_dir: str) -> Path:
    paper_id = str(paper.get("paper_id") or "unknown")
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in paper_id)[:64]
    digest = hashlib.sha256(paper_id.encode("utf-8")).hexdigest()[:10]
    return Path(save_dir) / f"{safe_name}_{digest}.pdf"


def _is_parseable_pdf(path: Path) -> bool:
    """验证魔数、最小大小，并确认 PyMuPDF 至少能读取一页。"""
    return _validate_pdf_magic(path) and _pdf_page_count(path) > 0


def _retry_delay(attempt: int, config=None) -> float:
    config = config or get_settings()
    return max(0.0, float(config.pdf_download_backoff_seconds)) * (2 ** (attempt - 1))


def download_open_access_pdf(
    paper: Dict[str, Any],
    save_dir: Optional[str] = None,
) -> Optional[str]:
    """下载单篇开放获取 PDF。

    Args:
        paper: 含 pdf_url 的论文字典。
        save_dir: 保存目录。

    Returns:
        下载后的本地路径，失败返回 None。
    """
    if not allows_pdf_download(paper):
        logger.info(
            "PDF download skipped by source policy: paper_id=%s source=cnki",
            paper.get("paper_id") or "unknown",
        )
        return None

    if not is_open_access(paper):
        logger.info(
            "PDF download skipped because open-access permission is unverified: paper_id=%s",
            paper.get("paper_id") or "unknown",
        )
        return None

    config = get_settings()

    pdf_url = paper.get("pdf_url")
    if not pdf_url:
        return None

    save_dir_str = str(save_dir or config.pdf_save_dir)
    Path(save_dir_str).mkdir(parents=True, exist_ok=True)

    paper_id = str(paper.get("paper_id") or "unknown")
    pdf_path = _pdf_path_for(paper, save_dir_str)
    part_path = pdf_path.with_suffix(".pdf.part")

    # 只有通过完整校验的缓存文件才复用；损坏缓存会被清理并重新下载。
    if pdf_path.exists() and _is_parseable_pdf(pdf_path):
        logger.debug("PDF already exists: %s", pdf_path)
        return str(pdf_path)
    pdf_path.unlink(missing_ok=True)
    part_path.unlink(missing_ok=True)

    max_attempts = max(1, int(config.pdf_download_retries))
    max_bytes = max(1, int(config.pdf_download_max_mb)) * 1024 * 1024
    timeout = (
        max(1, int(config.pdf_download_connect_timeout)),
        max(1, int(config.agent_request_timeout)),
    )

    for attempt in range(1, max_attempts + 1):
        response = None
        retryable = False
        try:
            response = requests.get(
                str(pdf_url),
                timeout=timeout,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 ResearchReview-Agent/0.1"
                    ),
                    "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.1",
                },
                stream=True,
                allow_redirects=True,
            )
            if response.status_code in _RETRYABLE_STATUS_CODES:
                retryable = True
            response.raise_for_status()

            content_type = str(response.headers.get("Content-Type") or "").lower()
            if content_type and not any(t in content_type for t in ("pdf", "octet-stream", "binary")) and ("html" in content_type or "text/" in content_type or "json" in content_type):
                logger.info(
                    "PDF download skipped: response is non-PDF content-type=%s (paper_id=%s)",
                    content_type,
                    paper_id,
                )
                retryable = False
                break

            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > max_bytes:
                raise PDFDownloadError(
                    f"PDF exceeds {config.pdf_download_max_mb} MB limit"
                )

            total = 0
            with open(part_path, "wb") as file_obj:
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > max_bytes:
                        raise PDFDownloadError(
                            f"PDF exceeds {config.pdf_download_max_mb} MB limit"
                        )
                    file_obj.write(chunk)

            content_type = str(response.headers.get("Content-Type") or "")
            if not _is_parseable_pdf(part_path):
                raise PDFDownloadError(
                    f"response is not a parseable PDF (content-type={content_type or 'unknown'})"
                )

            os.replace(part_path, pdf_path)
            logger.info(
                "Downloaded PDF: paper_id=%s bytes=%d attempts=%d path=%s",
                paper_id,
                total,
                attempt,
                pdf_path,
            )
            return str(pdf_path)
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            retryable = status_code in _RETRYABLE_STATUS_CODES
            logger.warning(
                "PDF HTTP failure: paper_id=%s status=%s attempt=%d/%d",
                paper_id,
                status_code,
                attempt,
                max_attempts,
            )
        except requests.RequestException as exc:
            retryable = True
            logger.warning(
                "PDF network failure: paper_id=%s attempt=%d/%d error=%s",
                paper_id,
                attempt,
                max_attempts,
                exc,
            )
        except (PDFDownloadError, OSError, ValueError) as exc:
            logger.warning(
                "PDF validation failure: paper_id=%s attempt=%d/%d error=%s",
                paper_id,
                attempt,
                max_attempts,
                exc,
            )
        finally:
            part_path.unlink(missing_ok=True)
            if response is not None:
                response.close()

        if not retryable or attempt >= max_attempts:
            break
        time.sleep(_retry_delay(attempt, config))

    pdf_path.unlink(missing_ok=True)
    return None


def batch_download_pdfs(
    papers: List[Dict[str, Any]],
    existing: Optional[Dict[str, str]] = None,
    save_dir: Optional[str] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> Dict[str, Optional[str]]:
    """批量下载开放论文 PDF。

    Args:
        papers: 论文列表。
        existing: 已有的 paper_id → pdf_path 映射（跳过）。
        save_dir: 保存目录。
        should_cancel: 协作式取消探针；为真时抛 InterruptedError，
            未启动的下载任务被移出队列。

    Returns:
        paper_id → pdf_path（或 None）的映射字典。
    """
    paths: Dict[str, Optional[str]] = dict(existing or {})
    # 清理不在本批 papers 中但仍残留于 state 的 CNKI 全文路径，避免下游误解析。
    for existing_id in list(paths):
        if is_cnki_paper({"paper_id": existing_id}):
            paths[existing_id] = None
    pending: list[Dict[str, Any]] = []
    policy_skipped_ids: list[str] = []
    for paper in papers:
        paper_id = str(paper.get("paper_id") or "")
        if not paper_id:
            continue
        if not allows_pdf_download(paper):
            # 即使历史映射中已有 CNKI PDF，也不把它交给后续全文解析。
            paths[paper_id] = None
            policy_skipped_ids.append(paper_id)
            continue
        existing_path = paths.get(paper_id)
        if existing_path and _is_parseable_pdf(Path(existing_path)):
            continue
        if not is_open_access(paper):
            paths[paper_id] = None
            continue
        pending.append(paper)

    config = get_settings()
    if should_cancel and should_cancel():
        raise InterruptedError("PDF 下载已取消")
    workers = min(max(1, int(config.pdf_download_max_workers)), len(pending) or 1)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="pdf-download") as executor:
        futures = {
            executor.submit(download_open_access_pdf, paper, save_dir): str(paper.get("paper_id") or "")
            for paper in pending
        }
        for future in as_completed(futures):
            if should_cancel and should_cancel():
                # 未启动的任务移出队列；已启动的随 with 退出自然结束。
                for pending_future in futures:
                    pending_future.cancel()
                raise InterruptedError("PDF 下载已取消")
            paper_id = futures[future]
            try:
                paths[paper_id] = future.result()
            except Exception as exc:  # 单篇失败不得中断整批
                logger.warning("Unexpected PDF download failure for %s: %s", paper_id, exc)
                paths[paper_id] = None

    eligible_ids = {
        str(paper.get("paper_id") or "")
        for paper in papers
        if paper.get("paper_id") and is_open_access(paper)
    }
    downloaded = sum(1 for paper_id in eligible_ids if paths.get(paper_id))
    eligible = len(eligible_ids)
    logger.info(
        "Batch download: papers=%d policy_skipped=%d eligible=%d attempted=%d "
        "succeeded=%d failed=%d",
        len(papers),
        len(policy_skipped_ids),
        eligible,
        len(pending),
        downloaded,
        max(0, eligible - downloaded),
    )
    return paths


def validate_pdf_file(pdf_path: str) -> bool:
    """检查 PDF 文件是否有效。"""
    return _is_parseable_pdf(Path(pdf_path))
