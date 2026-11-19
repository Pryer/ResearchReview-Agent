"""入库指纹与证据锚点基线，不调用模型。"""
import pytest

from app.services.document_ingestion import source_fingerprint, validate_chunk_sources
from app.schemas.vector_schema import VectorIntegrityError
from app.tools.chunk_pdf import chunk_pdf
from app.tools.parse_pdf import parse_pdf


def test_pdf_diagnostics_keep_empty_page_identity_and_cleaned_offsets(tmp_path):
    import fitz
    path = tmp_path / "source.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Methods\nWe propose a dual encoder retrieval method.")
    doc.new_page()
    doc.new_page().insert_text((72, 72), "Results\nResults show an accuracy of 91%.")
    doc.save(path)
    doc.close()
    parsed = parse_pdf(str(path), paper_id="p")
    assert parsed["parse_diagnostics"]["total_pages"] == 3
    assert parsed["parse_diagnostics"]["empty_pages"] == [2]
    assert [p["page"] for p in parsed["pages"]] == [1, 3]
    assert len(parsed["source_pdf_hash"]) == 64
    assert parsed["parsed_source_hash"] == source_fingerprint(parsed)
    assert parsed["parse_diagnostics"]["table_count"] is None
    by_page = {p["page"]: p["text"] for p in parsed["pages"]}
    assert [s["section"] for s in parsed["section_spans"]] == ["method", "results"]
    for span in parsed["section_spans"]:
        assert by_page[span["page"]][span["char_start"]:span["char_end"]] == span["text"]
    validate_chunk_sources("p", chunk_pdf(parsed, paper_id="p"), parsed)


def test_page_order_and_document_identity_affect_fingerprint():
    pages = [{"page": 1, "text": "One"}, {"page": 2, "text": "Two"}]
    first = source_fingerprint({"pages": pages})
    assert source_fingerprint({"pages": list(reversed(pages))}) != first
    assert source_fingerprint({"pages": pages, "source_pdf_hash": "different"}) != first


def test_layout_reorders_two_columns_without_merging_page_offsets(tmp_path, monkeypatch):
    import fitz
    from app.core.config import get_settings
    from app.services.source_alignment import locate_quote

    path = tmp_path / "columns.pdf"
    with fitz.open() as doc:
        page = doc.new_page()
        left = "Methods\nLeft column explains the setup."
        right = "Right column introduces the result\nand the paragraph continues"
        # The PDF content stream stores the right column first.
        page.insert_text((350, 90), right)
        page.insert_text((52, 90), left)
        doc.new_page().insert_text((52, 90), "on the next page without a heading.")
        doc.save(path)
    settings = get_settings()
    monkeypatch.setattr(settings, "pdf_layout_enabled", False)
    native = parse_pdf(str(path))
    assert native["pages"][0]["text"] == f"{right}\n{left}"
    monkeypatch.setattr(settings, "pdf_layout_enabled", True)
    parsed = parse_pdf(str(path))
    assert parsed["pages"][0]["text"] == f"{left}\n{right}"
    assert parsed["parse_diagnostics"]["page_layouts"][0]["reading_order"] == "two_columns"
    assert parsed["parsed_source_hash"] != native["parsed_source_hash"]
    assert parsed["section_spans"][-1]["section"] == "method"
    for page, quote in zip(parsed["pages"], [right, "on the next page without a heading."]):
        match = locate_quote(page["text"], quote)
        assert match and page["text"][match.start:match.end] == quote
    validate_chunk_sources(parsed["paper_id"], chunk_pdf(parsed, paper_id=parsed["paper_id"]), parsed)


def test_layout_table_retains_row_and_column_identity(tmp_path, monkeypatch):
    import fitz
    from app.core.config import get_settings

    rows = [["Method", "F1", "Accuracy"], ["First", "86%", "91%"], ["Second", "91%", "86%"]]
    path = tmp_path / "table.pdf"
    with fitz.open() as doc:
        page = doc.new_page()
        for x in (50, 150, 250, 350):
            page.draw_line((x, 50), (x, 140))
        for y in (50, 80, 110, 140):
            page.draw_line((50, y), (350, y))
        for row in range(len(rows)):
            for column, value in enumerate(rows[row]):
                page.insert_text((60 + column * 100, 68 + row * 30), value)
        doc.save(path)
    monkeypatch.setattr(get_settings(), "pdf_layout_enabled", True)
    parsed = parse_pdf(str(path))
    layout = parsed["parse_diagnostics"]["page_layouts"][0]
    assert layout["reading_order"] == "native"
    assert parsed["parse_diagnostics"]["table_count"] == 1
    cells = layout["tables"][0]["rows"]
    assert [[cell["text"] for cell in row] for row in cells] == rows
    assert cells[1][1]["anchor"] is None and cells[2][2]["anchor"] is None
    anchor = cells[1][0]["anchor"]
    assert parsed["pages"][0]["text"][anchor["char_start"]:anchor["char_end"]] == "First"


def test_exclusion_changes_invalidate_processing_profile():
    from app.core.config import Settings
    from app.services.document_ingestion import processing_profile
    parsed={'full_text':'A sufficiently long original sentence.'}
    chunks=chunk_pdf(parsed,paper_id='p')
    settings=Settings(_env_file=None)
    before=processing_profile(parsed,chunks,settings)
    changed=[{**chunks[0],'excluded':True}]
    assert processing_profile(parsed,changed,settings)!=before


@pytest.mark.parametrize("change", ["text", "offset", "paper", "duplicate"])
def test_bad_source_anchors_rejected(change):
    parsed = {"full_text": "A sufficiently long original sentence."}
    chunks = chunk_pdf(parsed, paper_id="p")
    if change == "text":
        chunks[0]["text"] = "Invented text."
    elif change == "offset":
        chunks[0]["char_start"] = 1
    elif change == "paper":
        chunks[0]["paper_id"] = "another"
    else:
        chunks.append(dict(chunks[0]))
    with pytest.raises(VectorIntegrityError):
        validate_chunk_sources("p", chunks, parsed)
