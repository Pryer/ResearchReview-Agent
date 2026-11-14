"""Count actual work: final query dedupe, per-provider completion and dependency caches."""
import copy
import pytest
from app.core.config import get_settings,Settings
from app.core.source_capabilities import sanitize_search_keyword


@pytest.mark.parametrize('query',['检索增强生成','研究方法','RAG 检索增强生成','retrieval augmented generation'])
def test_terms_survive_sanitization(query):
    result=sanitize_search_keyword(query)
    assert result and ('检索增强生成' in result if '检索增强生成' in query else result==query)


@pytest.mark.parametrize('query,term',[('检索增强生成','检索增强生成'),('并行计算','并行计算'),
    ('请撰写检索增强生成研究背景','检索增强生成')])
def test_topic_cleaning_preserves_method_names(query,term):
    from app.agent.slot_extractor import extract_topic
    assert term in (extract_topic(query) or '')


def test_provider_channels_cached_separately_and_windows_not_conflated(monkeypatch):
    import app.tools.search_papers as tool
    calls=[]
    monkeypatch.setattr(tool,'_search_arxiv',lambda *args:(calls.append(args) or []))
    ledger={}
    for _ in range(2):tool.search_papers('retrieval augmented generation',2020,2023,10,sources=['arxiv'],query_ledger=ledger)
    assert len(calls)==2 and {a[-1] for a in calls}=={'date','relevance'}
    tool.search_papers('retrieval augmented generation',2021,2023,10,sources=['arxiv'],query_ledger=ledger)
    assert len(calls)==4


def test_provider_failure_is_retryable_not_cached_as_empty(monkeypatch):
    import app.tools.search_papers as tool
    monkeypatch.setattr(get_settings(),'search_query_retry_seconds',0)
    calls=[]
    def source(*args):
        calls.append(args)
        if len(calls)==1:raise TimeoutError('timeout')
        return []
    monkeypatch.setattr(tool,'_search_arxiv',source)
    ledger={}
    for _ in range(3):tool.search_papers('RAG',2020,2023,10,sources=['arxiv'],query_ledger=ledger,enable_dual_channel=False)
    assert len(calls)==2 and next(iter(ledger.values()))['status']=='empty'


def test_gap_query_keeps_a_slot_when_many_branches_exist():
    from app.services.retrieval_ranking_service import build_queries
    s={'topic':'RAG','search_branches':[{'queries':[f'route {i}']} for i in range(10)],
        'recovery_decision':{'queries':['RAG uncertainty benchmark']}}
    queries=build_queries(s,6)
    assert any(q.role=='gap' and 'uncertainty' in q.text for q in queries)
    assert queries[0].role=='overall'


def test_chunk_rerank_uses_query_content_cache(tmp_path):
    from tests.test_chunk_retriever import Encoder,Reranker,chunks,TARGET
    from app.tools.chunk_retriever import retrieve_chunks
    cfg=Settings(_env_file=None,card_vector_backend='numpy',retrieval_cache_dir=str(tmp_path),
        retrieval_embedding_dimension=2,card_chunk_enable_rerank=True)
    source=chunks(TARGET);reranker=Reranker()
    for _ in range(2):retrieve_chunks('p',source,settings=cfg,embedding_client=Encoder(),reranker_client=reranker)
    assert len(reranker.calls)==7


def test_card_cache_reuses_raw_evidence_but_refreshes_eligibility(monkeypatch):
    from tests.test_draft_delivery_policy import draft_state
    from app.agent.nodes.extraction import extract_card_node
    from app.agent.extraction_reuse import extraction_fingerprint
    s=draft_state();s['paper_details'][0].update(abstract='unchanged source',_topic_relation='direct',_eligible_deliverables=['research_background'])
    paper=s['paper_details'][0]
    key=extraction_fingerprint(paper,None,s['topic'],get_settings())
    s['extraction_cache']={key:copy.deepcopy(s['paper_cards'][0])}
    paper['_rank_score']=.01;paper['_eligible_deliverables']=['research_background','narrative_review']
    monkeypatch.setattr('app.tools.extract_paper_card.batch_extract_paper_cards',lambda papers,*a,**kw:([] if not papers else pytest.fail('unchanged raw source re-extracted')))
    extract_card_node(s)
    assert s['paper_cards'][0]['eligible_deliverables']==['research_background','narrative_review']
    paper['abstract']='new source'
    assert extraction_fingerprint(paper,None,s['topic'],get_settings())!=key


def test_full_text_chunk_responses_are_reused_and_offsets_rechecked():
    from tests.test_full_text_extraction import PAPER,Model,parsed,METHOD,RESULT
    from app.tools.extract_paper_card import extract_paper_card
    source=parsed(METHOD,RESULT);model=Model()
    first=extract_paper_card(PAPER,source,model)
    count=len(model.calls)
    second=extract_paper_card(PAPER,source,model)
    assert count and len(model.calls)==count
    assert second.extraction_coverage.cache_hits==count
    assert second.extraction_coverage.remote_requests==0
    assert first.method==second.method and first.results==second.results
    full_text_spans=[span for span in second.evidence_spans if span.source_type=='full_text']
    assert full_text_spans
    for span in full_text_spans:
        text=source['pages'][span.page-1]['text']
        assert text[span.char_start:span.char_end]==span.text


def test_failed_chunk_cache_never_becomes_successful_evidence():
    from tests.test_full_text_extraction import PAPER,Model,parsed,METHOD
    from app.tools.extract_paper_card import extract_paper_card
    first=extract_paper_card(PAPER,parsed(METHOD),Model(TimeoutError('temporary')))
    assert first.extraction_coverage.chunks[0].status=='failed'
    second=extract_paper_card(PAPER,parsed(METHOD),Model())
    assert second.method==METHOD and second.extraction_coverage.remote_requests==1
