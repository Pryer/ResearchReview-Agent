"""PDF 解析工具。

使用 PyMuPDF 提取全文并粗略分段。
后续可用 GROBID 提升结构化解析质量。
"""

from __future__ import annotations

import re
import hashlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from app.core.exceptions import PDFParseError
from app.core.logger import get_logger
from app.utils.text_cleaner import clean_parsed_text

logger = get_logger(__name__)


def parse_pdf(pdf_path: str, *, paper_id: str | None = None) -> Dict[str, Any]:
    """解析 PDF 返回结构化文本。

    Returns:
        包含 sections 和 full_text 的字典。
    """
    diagnostics: Dict[str, Any] = {}
    pages = extract_pdf_pages(pdf_path, diagnostics=diagnostics)
    full_text = clean_parsed_text("\n".join(page["text"] for page in pages))
    sections = split_paper_sections(full_text)
    references = extract_references(full_text)

    digest = hashlib.sha256()
    with open(pdf_path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    from app.services.document_ingestion import PARSER_PROFILE, source_fingerprint
    result = {
        "full_text": full_text,
        "sections": sections,
        "abstract": sections.get("abstract", ""),
        "references": references,
        "paper_id": paper_id or Path(pdf_path).stem,
        "pages": pages,
        "source_pdf_hash": digest.hexdigest(),
        "parser_profile": PARSER_PROFILE,
        "parse_diagnostics": {**diagnostics, "recognized_sections": list(sections),
                              "table_count": None, "reading_order_verified": False},
    }
    result["parsed_source_hash"] = source_fingerprint(result)
    return result


def extract_pdf_text(pdf_path: str) -> str:
    """提取 PDF 全文文本。

    解析失败会抛出 :class:`PDFParseError`；批处理入口负责逐篇隔离失败。
    """
    pages = extract_pdf_pages(pdf_path)
    return clean_parsed_text("\n".join(page["text"] for page in pages))


def extract_pdf_pages(pdf_path: str, *, diagnostics: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """逐页提取文本，保留 Evidence Card 所需的页码信息。"""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        pages = [
            {"page": index + 1, "text": clean_parsed_text(page.get_text())}
            for index, page in enumerate(doc)
        ]
        doc.close()
        if diagnostics is not None:
            empty = [page["page"] for page in pages if not page["text"].strip()]
            diagnostics.update(
                total_pages=len(pages), empty_pages=empty,
                text_characters=sum(len(page["text"]) for page in pages),
                # WHY: 低文本只表示 OCR 候选；空白封面等也会命中，不等同于解析失败。
                ocr_candidate_pages=[page["page"] for page in pages if len(page["text"].strip()) < 40],
            )
        return [page for page in pages if page["text"]]
    except ImportError:
        raise PDFParseError("PyMuPDF is required to parse PDF files")
    except Exception as e:
        raise PDFParseError(f"Failed to parse PDF {pdf_path}: {e}")


def split_paper_sections(text: str) -> Dict[str, str]:
    """粗略划分论文段落。

    基于常用标题关键词做正则分段。
    返回 abstract / introduction / method / experiment / conclusion / references。
    """
    if not text:
        return {}

    sections: Dict[str, str] = {}

    from app.tools.chunk_pdf import section_headings

    # WHY: 解析保留历史 experiment 名称；与 Chunk 使用同一标题边界，避免准入错位。
    names = {"results": "experiment", "discussion": "conclusion"}
    splits = [(names.get(name, name), start) for start, name in section_headings(text)]

    # 按位置排序
    splits.sort(key=lambda x: x[1])

    # 提取各段
    for i, (name, start) in enumerate(splits):
        end = splits[i + 1][1] if i + 1 < len(splits) else len(text)
        content = text[start:end].strip()
        # 取最长的同名段落
        if name not in sections or len(content) > len(sections[name]):
            sections[name] = content

    return sections


def extract_abstract(text: str) -> str:
    """从全文提取摘要段。"""
    if not text:
        return ""
    # 找 "abstract" 到下一个大标题之间的内容
    m = re.search(
        r"abstract\s*[:\n]+\s*(.+?)(?:\n\s*\d+\.?\s|\n\s*(?:introduction|1\.))",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return ""
    return m.group(1).strip()


def extract_references(text: str) -> List[str]:
    """提取参考文献列表。

    粗略实现：定位 references / bibliography / 参考文献标题段，
    再匹配 ``[1]`` / ``1.``（含全角变体）开头的条目行。
    """
    if not text:
        return []
    ref_section_match = re.search(
        r"(?:references|bibliography|参考文献)\s*\n(.+)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if not ref_section_match:
        return []

    ref_text = ref_section_match.group(1)
    refs: list[str] = []
    # 匹配 [N] / ［N］ 开头的引用（GB/T 7714 与英文编号风格）
    entry_re = re.compile(r"^(?:[\[［]\d+[\]］]|\d+[\.．])")
    strip_re = re.compile(r"^(?:[\[［]\d+[\]］]|\d+[\.．])\s*")
    for line in ref_text.split("\n"):
        line = line.strip()
        if entry_re.match(line):
            # 只剥离开头的编号，避免误删正文中的年份/DOI 数字
            line = strip_re.sub("", line).strip()
            if len(line) > 10:
                refs.append(line)
    return refs


def batch_parse_pdfs(
    pdf_paths: Dict[str, str],
    diagnostics: Optional[List[Dict[str, Any]]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> Dict[str, Dict[str, Any]]:
    """批量解析 PDF。

    Args:
        pdf_paths: paper_id → 本地 pdf_path 映射。
        should_cancel: 协作式取消探针；为真时抛 InterruptedError。

    Returns:
        paper_id → 解析结果 的映射。仅包含成功解析的。
    """
    results: Dict[str, Dict[str, Any]] = {}
    for paper_id, path in pdf_paths.items():
        if should_cancel and should_cancel():
            raise InterruptedError("PDF 解析已取消")
        if not path:
            continue
        try:
            result = parse_pdf(path, paper_id=paper_id)
            if result.get("full_text"):
                results[paper_id] = result
                if diagnostics is not None:
                    diagnostics.append({"paper_id": paper_id, "status": "success"})
            elif diagnostics is not None:
                diagnostics.append({
                    "paper_id": paper_id,
                    "status": "empty",
                    "message": "PDF parsed without usable text",
                })
        except PDFParseError as e:
            logger.warning("Parse failed for %s: %s", paper_id, e)
            if diagnostics is not None:
                diagnostics.append({
                    "paper_id": paper_id,
                    "status": "failed",
                    "message": str(e),
                })
    logger.info("Batch parse: %d / %d succeeded", len(results), len(pdf_paths))
    return results
