from app.tools.chunk_pdf import chunk_pdf
from app.tools.parse_pdf import build_section_spans
from app.services.document_ingestion import validate_chunk_sources
from app.tools.chunk_retriever import coverage_order


def test_bibliographic_model_line_does_not_reopen_method_on_next_page():
    parsed={'pages':[
        {'page':1,'text':'Methods\nWe train a retrieval model.\nReferences\nA. Author. 2020. A paper.'},
        {'page':2,'text':'model\npre-training. arXiv preprint.\nOther Author. 2021. Another paper.'},
        {'page':3,'text':'A\nAppendix\nAdditional experimental examples and their limitations.'}]}
    chunks=chunk_pdf(parsed,paper_id='p')
    assert all(c['section']=='references' and c['excluded'] for c in chunks if c['page']==2)
    assert any(c['section']=='appendix' and not c['excluded'] for c in chunks if c['page']==3)
    assert all(c['excluded'] for c in chunks if c['text'].strip()=='A')
    spans=build_section_spans(parsed['pages'])
    assert all(s.section=='references' for s in spans if s.page==2)
    assert not {c['chunk_id'] for c in chunks if c['excluded']} & set(coverage_order(chunks))
    validate_chunk_sources('p',chunks,parsed)


def test_single_character_is_preserved_for_provenance_but_excluded():
    chunks=chunk_pdf({'full_text':'A'},paper_id='p')
    assert chunks[0]['text']=='A' and chunks[0]['element_type']=='noise' and chunks[0]['excluded']


def test_explicit_supplementary_heading_exits_references():
    chunks=chunk_pdf({'full_text':'References\nAuthor. Model\nSupplementary Materials\nAdditional analysis.'},paper_id='p')
    assert chunks[-1]['section']=='appendix' and not chunks[-1]['excluded']
