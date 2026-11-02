"""PDF 文本分段与参考文献抽取测试（纯函数部分，不依赖 PyMuPDF）。"""

from __future__ import annotations

from app.tools.parse_pdf import extract_references, split_paper_sections
from app.tools.parse_pdf import build_section_spans


def test_extract_references_from_english_section():
    text = (
        "Introduction\n\nSome claims.\n\n"
        "References\n"
        "[1] Smith J. Deep learning survey. 2023.\n"
        "[2] Doe A. Vision transformers. 2024.\n"
    )
    refs = extract_references(text)
    assert len(refs) == 2
    assert refs[0] == "Smith J. Deep learning survey. 2023."


def test_extract_references_recognizes_chinese_heading():
    """中文论文的“参考文献”标题段同样要能定位（此前只认英文标题）。"""
    text = (
        "引言\n\n本文研究课堂行为分析。\n\n"
        "参考文献\n"
        "[1] 张三. 课堂行为识别研究. 计算机学报, 2022.\n"
        "[2] 李四. 深度学习综述. 软件学报, 2023.\n"
    )
    refs = extract_references(text)
    assert len(refs) == 2
    assert refs[0].startswith("张三")


def test_extract_references_keeps_year_and_doi_intact():
    """编号剥离必须锚定行首，不得吞掉正文中的年份或 DOI 数字。"""
    text = (
        "References\n"
        "[1] Wang X. Study 2020. https://doi.org/10.1000/xyz.\n"
    )
    refs = extract_references(text)
    assert refs == ["Wang X. Study 2020. https://doi.org/10.1000/xyz."]


def test_extract_references_accepts_fullwidth_variants():
    text = (
        "参考文献\n"
        "［1］王五. 智能教育研究. 2021.\n"
        "2．赵六. 学习分析进展. 2022.\n"
    )
    refs = extract_references(text)
    assert len(refs) == 2


def test_split_sections_recognizes_chinese_headings():
    text = (
        "摘要\n\n本文提出……\n\n"
        "引言\n\n近年来……\n\n"
        "结论\n\n总结……\n\n"
        "参考文献\n"
        "[1] 张三. 2022.\n"
    )
    sections = split_paper_sections(text)
    assert "abstract" in sections
    assert "introduction" in sections
    assert "conclusion" in sections
    assert "references" in sections


def test_ordered_sections_preserve_repeats_continuations_and_source_offsets():
    from app.tools.chunk_pdf import chunk_pdf
    from app.schemas.paper_schema import PaperDetail

    pages = [{"page": 1, "text": "Title\nMethods\nFirst method.\nMethods\nSecond method."},
             {"page": 3, "text": "Continuing second method.\nResults\nFirst result.\nDiscussion\nA caveat."}]
    spans = build_section_spans(pages)
    assert [s.section for s in spans] == ["full_text", "method", "method", "method", "results", "discussion"]
    assert [s.page for s in spans] == [1, 1, 1, 3, 3, 3]
    detail = PaperDetail(paper_id="p", section_spans=[s.model_dump() for s in spans])
    assert PaperDetail.model_validate_json(detail.model_dump_json()).section_spans == spans
    by_page = {page["page"]: page["text"] for page in pages}
    chunks = chunk_pdf({"pages": pages})
    assert len(chunks) == len(spans)
    for span, chunk in zip(spans, chunks):
        assert by_page[span.page][span.char_start:span.char_end] == span.text
        assert span.model_dump() == {key: chunk[key] for key in span.model_fields}
    assert split_paper_sections("Methods\nLong first method.\nMethods\nShort.")["method"] == "Methods\nLong first method."


def test_section_spans_handle_no_heading_and_full_text_basis():
    text = "前言没有标准章节标题。\n仍然保留全文。"
    spans = build_section_spans([{"page": None, "text": text}, {"page": 2, "text": ""}])
    assert len(spans) == 1
    assert spans[0].text == text
    assert spans[0].offset_basis == "full_text"
    assert split_paper_sections(text) == {}
