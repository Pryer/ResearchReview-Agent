"""证据基准指纹与无外部调用的入库校验。"""
from __future__ import annotations

from app.schemas.retrieval_schema import fingerprint
from app.schemas.vector_schema import VectorIntegrityError

PARSER_PROFILE = "pymupdf.cleaned-pages.v1"
CHUNKER_PROFILE = "page-section-chars.v2"


def source_fingerprint(parsed):
    pages = parsed.get("pages") or []
    if any(str(page.get("text") or "").strip() for page in pages):
        source = [(page.get("page"), str(page.get("text") or "")) for page in pages]
    else:
        source = [(None, str(parsed.get("full_text") or ""))]
    return fingerprint((parsed.get("source_pdf_hash"), source))


def validate_chunk_sources(paper_id, chunks, parsed):
    ids = set()
    pages = {page.get("page"): str(page.get("text") or "") for page in parsed.get("pages", [])}
    if len(pages) != len(parsed.get("pages", [])):
        raise VectorIntegrityError("重复页面身份")
    for chunk in chunks:
        start, end = chunk["char_start"], chunk["char_end"]
        source = pages.get(chunk["page"], "") if chunk["offset_basis"] == "page" else str(parsed.get("full_text") or "")
        if (chunk.get("paper_id") != paper_id or chunk["chunk_id"] in ids
                or not 0 <= start < end <= len(source) or source[start:end] != chunk["text"]):
            raise VectorIntegrityError("Chunk 身份或原文位置校验失败")
        ids.add(chunk["chunk_id"])


def embedding_profile(settings):
    return fingerprint((settings.retrieval_provider, settings.retrieval_embedding_url,
                        settings.retrieval_embedding_model, settings.retrieval_embedding_dimension,
                        settings.retrieval_cache_version, "cosine-normalized-float32", "document/query"))


def processing_profile(parsed, chunks, settings):
    # WHY: 次级排序和被索引的块集合也属于快照语义，不能只比较正文字符串。
    return fingerprint((parsed.get("parser_profile", PARSER_PROFILE), CHUNKER_PROFILE,
                        settings.card_extraction_chunk_chars, 400,
                        [(c["chunk_id"], c["section"], c["page"], c["char_start"], c["char_end"],
                          c["offset_basis"], c["text"]) for c in chunks]))
