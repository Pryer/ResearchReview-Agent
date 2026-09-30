"""写作停止与失败重建的生产边界回归，禁止外部模型请求。"""
import copy
from types import SimpleNamespace

import pytest

from app.agent import graph
from app.agent.execution import AgentCancelledError
from app.agent.execution_budget import AgentBudgetExceeded, AgentExecutionStale
from app.database.runtime_repository import RuntimeConflict


@pytest.fixture(params=[AgentBudgetExceeded, AgentCancelledError, AgentExecutionStale, RuntimeConflict])
def stop(request):
    return request.param("token reservation exceeds remaining budget")


def test_outline_propagates_control_stop_without_deterministic_fallback(stop):
    from app.agent.writing_plan import _induce_background_outline
    calls = []

    def complete(*args, **kwargs):
        calls.append(kwargs['operation'])
        raise stop

    with pytest.raises(type(stop)) as caught:
        _induce_background_outline({}, [], SimpleNamespace(complete=complete))
    assert caught.value is stop
    assert calls == ['induce_background_outline']


def test_generation_node_does_not_wrap_control_stop_as_llm_failure(monkeypatch, stop):
    from app.agent.nodes.synthesis import generate_deliverables_node

    def fail(*args, **kwargs):
        raise stop

    monkeypatch.setattr('app.tools.synthesize_themes.build_search_report', fail)
    state = {'steps': [], 'errors': []}
    with pytest.raises(type(stop)) as caught:
        generate_deliverables_node(state)
    assert caught.value is stop
    assert not state['errors']
    assert not state['steps']


def test_graph_writer_boundary_preserves_control_stop(monkeypatch, stop):
    def fail(*args, **kwargs):
        raise stop

    monkeypatch.setattr(graph, 'generate_deliverables_node', fail)
    monkeypatch.setattr('app.agent.state_invariants.validate_research_state_invariants', lambda _: {})
    with pytest.raises(type(stop)) as caught:
        graph._generate_deliverables_or_block({}, llm=object())
    assert caught.value is stop


@pytest.mark.parametrize('gate_code', ['required_focus_evidence_not_met', 'minimum_planned_references_not_met', 'minimum_cited_references_not_met'])
def test_budget_stop_is_visible_even_with_quality_failure(gate_code):
    gate = {'passed': False, 'blocking_issues': [{'code': gate_code, 'requested': 40, 'actual': 0}]}
    state = {'agent_orchestration_mode': 'autonomous', 'result_status': 'blocked',
             'generation_blocked': True, 'quality_gate': copy.deepcopy(gate),
             'errors': [{'code': 'AgentBudgetExceeded', 'message': 'token reservation exceeds remaining budget'}]}
    output = graph._build_output(state)
    assert '剩余 token 预算不足' in output['answer']
    assert output['quality_gate']['blocking_issues'] == gate['blocking_issues']
    assert not output['quality_gate']['draft_released']
    assert output['body'] == ''


def test_failed_full_rebuild_keeps_private_candidate_across_restarts(monkeypatch):
    from app.agent.main_loop import MainAgentLoop
    old_review = '旧证据正文 [old-paper]。'
    old = {'review': old_review, 'references': ['旧文献'], 'claim_plans': [{'route_id': 'old'}],
           'quality_gate': {'passed': False, 'draft_released': False},
           'unique_valid_cited_paper_count': 38, 'claim_verification': {'unsupported': 21}}
    current = {**copy.deepcopy(old), 'intent': 'generate_review', 'topic': '测试主题',
               'required_reference_count': 40, 'start_year': 2024, 'end_year': 2026,
               'paper_cards': [{'paper_id': 'new-paper'}], 'paper_details': [{'paper_id': 'new-paper'}],
               'evidence_snapshot_version': 3, 'evidence_snapshot_fingerprint': 'old-evidence',
               'best_effort_generation': True}

    def run(loop, state, **kwargs):
        assert not state.get('review')
        assert not state.get('claim_plans')
        state['errors'].append({'code': 'AgentBudgetExceeded', 'message': 'token reservation exceeds remaining budget'})
        state['generation_blocked'] = True
        return 'blocked'

    monkeypatch.setattr(graph, '_get_llm', lambda: object())
    monkeypatch.setattr(MainAgentLoop, 'run', run)
    first = graph.regenerate_research_agent(current)
    second = graph.regenerate_research_agent(first['research_state'])
    for result in (first, second):
        assert result['status'] == 'blocked'
        assert result['body'] == ''
        assert old_review not in result['answer']
        assert result['previous_draft_available'] is True
        private = result['research_state']['quarantined_generation_snapshot']
        assert private['generation_products'] == {
            **old, 'section_checkpoints': {}, 'section_candidate_checkpoints': {},
        }
        assert private['evidence_snapshot_version'] == 3
        assert private['evidence_snapshot_fingerprint'] == 'old-evidence'
        assert not result['research_state'].get('claim_plans')
        assert result['research_state']['required_reference_count'] == 40
        assert (result['research_state']['start_year'], result['research_state']['end_year']) == (2024, 2026)
        assert result['paper_cards'] == [{'paper_id': 'new-paper'}]


def test_quarantined_candidate_is_fragmented_and_not_readable_by_agents():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from app.database.models import Base
    from app.services.research_artifact_service import ResearchArtifactService

    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    snapshot = {'generation_products': {'review': '隔离正文' * 70000, 'claim_plans': [{'route_id': 'old'}]},
                'evidence_snapshot_fingerprint': 'old-evidence', 'draft_disposition': 'quarantined'}
    with Session(engine) as db:
        service = ResearchArtifactService(db)
        stored = service.externalize_state('candidate', {'quarantined_generation_snapshot': snapshot})
        assert 'quarantined_generation_snapshot' not in stored
        entries = stored['artifact_manifest']['quarantined_generation_snapshot']['items']
        assert len(entries) > 1
        assert service.hydrate_state('candidate', stored)['quarantined_generation_snapshot'] == snapshot
        for role in ('main', 'search', 'analysis', 'writing'):
            with pytest.raises(PermissionError):
                service.resolve('candidate', entries[0]['ref'], role=role)
        with pytest.raises(PermissionError):
            service.resolve('another-session', entries[0]['ref'])
    engine.dispose()


def test_pre_generation_message_cannot_replace_previous_candidate():
    saved = {'generation_products': {'review': '先前的学术草稿。'}}
    state = {'review': '正文生成已阻止：证据不足',
             'quality_gate': {'phase': 'pre_generation', 'draft_available': False},
             'quarantined_generation_snapshot': copy.deepcopy(saved)}
    graph._preserve_generation_candidate(state)
    assert state['quarantined_generation_snapshot'] == saved


def test_real_writing_pipeline_rejects_outline_reservation_without_sending_request(monkeypatch):
    from unittest.mock import Mock
    from app.agent.execution_budget import budget_scope, budgeted_create
    from app.schemas.deliverable_schema import GenerationReadinessResult, DeliverableReadinessResult
    from app.tools.extract_paper_card import extract_paper_card

    paper = {'paper_id': 'p1', 'title': '课堂行为识别', 'authors': ['测试作者'],
             'year': 2024, 'venue': '测试期刊', 'abstract': '该研究通过课堂行为自动识别分析学生参与度。'}
    card = extract_paper_card(paper, llm=None).model_dump(mode='json')
    state = {'topic': '课堂行为分析', 'intent': 'generate_review',
             'core_deliverables': ['research_background'], 'paper_details': [paper],
             'paper_cards': [card], 'required_reference_count': 40, 'steps': [], 'errors': [],
             'agent_execution_budget': {'token_limit': 1000000, 'llm_tokens': 970019}}
    monkeypatch.setattr('app.agent.deliverable_router.check_generation_readiness',
                        lambda _, **kwargs: GenerationReadinessResult(ready=True, usable_reference_count=1))
    monkeypatch.setattr('app.agent.deliverable_router.check_deliverable_readiness',
                        lambda kind, _, **kwargs: DeliverableReadinessResult(requested_type=kind, effective_type=kind, ready=True))
    monkeypatch.setattr('app.agent.state_invariants.validate_research_state_invariants', lambda _: {})
    create = Mock()
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    calls = []

    class BudgetLLM:
        def complete(self, prompt, **kwargs):
            calls.append(kwargs['operation'])
            return budgeted_create(client, operation=kwargs['operation'],
                                   messages=[{'role': 'user', 'content': prompt}], max_tokens=56000)

    with budget_scope(state), pytest.raises(AgentBudgetExceeded, match='reservation exceeds'):
        graph._generate_deliverables_or_block(state, llm=BudgetLLM())
    assert calls == ['induce_background_outline']
    create.assert_not_called()
    assert not state['errors']
    assert not state.get('review')
    assert not state.get('quality_gate')
    assert state['agent_execution_budget']['llm_tokens'] == 970019
    assert not state['agent_execution_budget'].get('llm_requests')


def test_explicit_ten_million_budget_preserves_retrieval_and_llm_usage():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from app.database.models import Base
    from app.database.runtime_repository import ResearchRuntimeRepository

    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    ledger = {'token_limit': 1000000, 'llm_tokens': 899764, 'retrieval_model_tokens': 70255,
              'llm_requests': 184, 'retrieval_model_requests': 23, 'tokens_reserved': 0}
    with Session(engine) as db:
        runtime = ResearchRuntimeRepository(db, 'budget-raise')
        runtime.acquire({}, ledger)
        runtime.snapshot({'required_reference_count': 40}, ledger=ledger)
        runtime.release(ledger)
        updated = runtime.update_token_limit(expected_limit=1000000, new_limit=10000000)
        assert updated == {**ledger, 'token_limit': 10000000}
        restored = runtime.restore()
        assert restored['agent_execution_budget'] == updated
        assert restored['required_reference_count'] == 40
        assert updated['token_limit'] - updated['llm_tokens'] - updated['retrieval_model_tokens'] == 9029981
    engine.dispose()
