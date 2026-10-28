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
    validate_chunk_sources("p", chunk_pdf(parsed, paper_id="p"), parsed)


def test_page_order_and_document_identity_affect_fingerprint():
    pages = [{"page": 1, "text": "One"}, {"page": 2, "text": "Two"}]
    first = source_fingerprint({"pages": pages})
    assert source_fingerprint({"pages": list(reversed(pages))}) != first
    assert source_fingerprint({"pages": pages, "source_pdf_hash": "different"}) != first


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
