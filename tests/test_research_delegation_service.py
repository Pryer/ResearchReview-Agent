"""任务入口复用真实Handler组合根；仅外部检索/详情响应使用替身。"""
import pytest
from app.agent.execution_budget import budget_scope
from app.services.durable_execution_service import durable_scope, active_runtime
from app.services.research_delegation_service import run_search_delegation
from tests.test_v2_delegation_runtime import db, state


def configure(monkeypatch, calls):
    from app.agent import graph
    from app.core.config import get_settings
    from app.services.retrieval_ranking_service import admission_scope_fingerprint, _document
    monkeypatch.setattr(graph, 'require_rag_runtime', lambda settings: {})
    monkeypatch.setattr(get_settings(), 'enable_pdf_pipeline', False)
    paper = {'paper_id': 'p1', 'title': 'Test research evidence', 'year': 2024, 'authors': ['Author'],
             'abstract': 'Test research evidence for academic literature review.', 'venue': 'Journal',
             'source': 'openalex', '_screening_decision': 'include', '_topic_relation': 'direct',
             '_eligible_deliverables': ['research_status'], '_semantic_conditions_confirmed': True,
             '_semantic_contract_complete': True, '_screening_confidence': 0.99}
    def search(current, **kwargs):
        calls.append('search')
        p = dict(paper)
        p['_screened_content_fingerprint'] = _document(p).content_fingerprint
        p['_screened_scope_fingerprint'] = admission_scope_fingerprint(current)
        current.update(candidate_papers=[p], ranked_papers=[p])
    def details(papers):
        calls.append('metadata_api')
        return [dict(p) for p in papers]
    monkeypatch.setattr(graph, '_search_rank_with_refinement', search)
    monkeypatch.setattr('app.tools.fetch_metadata.fetch_batch_details', details)


def test_task_entry_calls_shared_fetch_handler_and_real_node(db, monkeypatch):
    current, calls = state(), []
    current.update(max_papers=1, required_reference_count=1, generation_limit=1)
    configure(monkeypatch, calls)
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        result = run_search_delegation(current, {'agent': 'search', 'profile': 'initial_research', 'goal': 'test research'}, llm=object())
        assert calls == ['search', 'metadata_api']
        assert result.outcome == 'satisfied' and len(result.action_result_refs) == 2
        assert current['paper_details'][0]['paper_id'] == 'p1'
        assert 'active_delegation_id' not in current


def test_completed_delegation_survives_failure_before_final_marker_cleanup(db, monkeypatch):
    current, calls = state(), []
    current.update(max_papers=1, required_reference_count=1, generation_limit=1)
    configure(monkeypatch, calls)
    request = {'agent': 'search', 'profile': 'initial_research', 'goal': 'test research'}
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        runtime = active_runtime()
        original = runtime.snapshot
        def fail_final(candidate, **kwargs):
            if 'active_delegation_id' in current and 'active_delegation_id' not in candidate:
                raise RuntimeError('crash after result committed')
            return original(candidate, **kwargs)
        monkeypatch.setattr(runtime, 'snapshot', fail_final)
        with pytest.raises(RuntimeError, match='after result'):
            run_search_delegation(current, request, llm=object())
        assert current.get('active_delegation_id')
        monkeypatch.setattr(runtime, 'snapshot', original)
        result = run_search_delegation(current, request, llm=object())
        assert result.outcome == 'satisfied'
        assert calls == ['search', 'metadata_api']


def test_wrong_role_rejected_before_handler_initialization(db, monkeypatch):
    from app.agent import graph
    current = state()
    monkeypatch.setattr(graph, 'build_research_actions', lambda *a, **k: pytest.fail('invalid request caused initialization'))
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        with pytest.raises(ValueError, match='role'):
            run_search_delegation(current, {'agent': 'writing', 'profile': 'initial_research', 'goal': 'invalid'}, llm=object())


def test_private_pipeline_requires_durable_execution(monkeypatch):
    from app.agent import graph
    from tests.test_subagent_react import ScriptedLLM
    current = state()
    current['architecture_version'] = 2
    llm = ScriptedLLM({})
    monkeypatch.setattr(graph, '_get_llm', lambda: llm)
    result = graph._run_autonomous_pipeline(current)
    assert result['status'] == 'failed' and not llm.calls
    assert current['errors'][-1]['code'] == 'RuntimeConflict'


def test_default_pipeline_resumes_orphan_active_record_without_state_marker(db, monkeypatch):
    from app.agent import graph
    from app.agent.policy.authorization import AuthorizationPolicy
    from app.database.delegation_repository import DelegationRepository
    from app.schemas.delegation_schema import DelegationRequest
    from tests.test_subagent_react import ScriptedLLM
    current = state()
    current.update(intent='search_papers', max_papers=1, required_reference_count=1)
    calls = []
    configure(monkeypatch, calls)
    monkeypatch.setattr(graph, '_get_llm', lambda: ScriptedLLM({'main_agent_decision': [('request_finish', {})]}))
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        repo = DelegationRepository(active_runtime())
        repo.create(AuthorizationPolicy().issue(DelegationRequest(agent='search', profile='initial_research', goal='test'), current))
        assert 'architecture_version' not in current
        result = graph._run_autonomous_pipeline(current)
        assert current['result_status'] == 'completed' and result['status'] == 'success'
        assert result['research_state']['architecture_version'] == 2
        assert repo.active() is None and calls == ['search']
