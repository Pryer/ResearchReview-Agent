import pytest
from app.services.source_alignment import locate_quote


def test_whitespace_mapping_returns_original_text_and_offsets():
    source = 'Intro. We do not\n  improve accuracy\ton this dataset. End.'
    hit = locate_quote(source, 'We do not improve accuracy on this dataset.')
    assert hit.method == 'whitespace'
    assert source[hit.start:hit.end] == hit.text == 'We do not\n  improve accuracy\ton this dataset.'


@pytest.mark.parametrize('source,quote', [
    ('We do not use retrieval.', 'We do use retrieval.'),
    ('The score is 91.0%.', 'The score is 91%.'),
    ('We use ﬁne-tuning.', 'We use fine-tuning.'),
    ('We use re-\ntrieval.', 'We use retrieval.'),
    ('The therapist responds.', 'The the rapist responds.'),
    ('A matching sentence. A matching sentence.', 'A matching sentence.'),
    ('We use\na method. We use a\tmethod.', 'We use a method.'),
])
def test_semantic_changes_and_ambiguous_matches_are_rejected(source, quote):
    assert locate_quote(source, quote) is None


def test_exact_quote_keeps_exact_coordinates():
    hit = locate_quote('Before. Exact source text. After.', 'Exact source text.')
    assert hit.method == 'exact' and hit.start == 8


def test_aligned_card_survives_readback_serialization_and_claim_verification(monkeypatch):
    import json
    from app.core.config import Settings
    from app.schemas.chunk_schema import ChunkRetrievalReport
    from app.schemas.paper_schema import PaperCard
    from app.tools.extract_paper_card import extract_paper_card
    from app.tools.verify_claims import verify_review_claims
    from app.services.evidence_support import authorized_field_claim
    source='Methods\nWe propose a dual encoder method for\n  document retrieval.'
    quote='We propose a dual encoder method for document retrieval.'
    monkeypatch.setattr('app.core.config.get_settings',lambda:Settings(_env_file=None,card_extraction_max_chunks=1))
    monkeypatch.setattr('app.tools.chunk_retriever.retrieve_chunks',lambda *a,**kw:
        ChunkRetrievalReport(mode='hybrid',indexed_chunk_ids=[0],selected_chunk_ids=[0]))
    class Extractor:
        def complete(self,prompt,**kwargs):return json.dumps({'method':[quote]})
    card=extract_paper_card({'paper_id':'p','title':'Study'},
        {'full_text':source,'pages':[{'page':1,'text':source}],'sections':{'method':source}},Extractor())
    restored=PaperCard.model_validate_json(card.model_dump_json()).model_dump(mode='json')
    claim=restored['field_claims']['method'][0]
    span=next(s for s in restored['evidence_spans'] if s['evidence_id']==claim['evidence_id'])
    assert source[span['char_start']:span['char_end']]==span['text']==claim['claim']
    assert span['source_status']=='source_verified' and authorized_field_claim(restored,claim)
    class Verifier:
        def complete(self,prompt,**kwargs):
            return '{"results":[{"claim_id":"c001u01","label":"entailed","confidence":0.99}]}'
    report=verify_review_claims(quote+' [p]',[restored],llm=Verifier())
    assert report['claims'][0]['support_status']=='supported'
