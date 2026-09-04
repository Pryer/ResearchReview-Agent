"""PDF 文件辅助工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from app.core.logger import get_logger

logger = get_logger(__name__)


def validate_pdf_file(pdf_path: Union[str, Path]) -> bool:
    """检查 PDF 文件是否有效。

    判断依据：
    - 文件存在
    - 大小 > 1KB
    - 以 ``%PDF`` 魔数开头

    Args:
        pdf_path: PDF 文件路径。

    Returns:
        是否有效。
    """
    p = Path(pdf_path)
    if not p.exists() or not p.is_file():
        return False
    if p.stat().st_size < 1024:
        logger.warning("PDF too small: %s", p)
        return False
    try:
        with open(p, "rb") as f:
            header = f.read(5)
        return header.startswith(b"%PDF-")
    except OSError as e:
        logger.warning("Failed to read PDF header: %s — %s", p, e)
        return False


def pdf_page_count(pdf_path: Union[str, Path]) -> int:
    """获取 PDF 页数。

    若解析失败则返回 0，不会抛出异常。
    """
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(str(pdf_path))
        count = len(doc)
        doc.close()
        return count
    except Exception as e:
        logger.warning("Failed to get page count for %s: %s", pdf_path, e)
        return 0


def is_open_access_arxiv(paper: dict) -> bool:
    """判断 arXiv 论文是否可直接下载 PDF。

    arXiv 论文几乎都是开放获取。
    """
    return bool(paper.get("arxiv_id") or paper.get("pdf_url"))
