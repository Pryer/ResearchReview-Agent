"""v2业务控制流与既有发布/服务持久化契约一致，外部模型使用脚本替身。"""
import pytest
from app.agent import graph
from app.agent.controller import AgentController
from app.agent.execution_budget import budget_scope
from app.agent.task_main import TaskMainLoop
from app.core.config import get_settings
from app.services.durable_execution_service import durable_scope
from tests.test_v2_delegation_runtime import db, state
from tests.test_subagent_react import ScriptedLLM


def test_v2_clarification_is_persisted_as_a_resumable_user_question(db, monkeypatch):
    from app.database.repositories import ResearchSessionRepository
    from app.services.research_conversation_service import ResearchConversationService
    current = state()
    current.update(start_year=2020, end_year=2024, year_range_explicit=True, required_reference_count=40,
        max_papers_explicit=True, intent='generate_review', core_deliverables=[])
    monkeypatch.setattr(get_settings(), 'agent_architecture_version', 'v2')
    llm = ScriptedLLM({'main_agent_decision': [('request_clarification', {'question': '需要哪个研究方向？'})]})
    monkeypatch.setattr(graph, '_get_llm', lambda: llm)
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        output = graph._run_autonomous_pipeline(current)
    assert output['status'] == 'needs_clarification'
    assert output['clarification'] == {'needed': True, 'kind': 'main_agent', 'question': '需要哪个研究方向？'}
    saved = ResearchConversationService(db, llm=None)._persist_or_pause_result('v2', current['user_query'], current, output)
    assert saved['status'] == 'needs_clarification'
    session = ResearchSessionRepository(db).get('v2')
    assert session['clarification']['needed']
    editable = session['state']['editable_research_state']
    assert editable['required_reference_count'] == 40 and editable['end_year'] == 2024
    import json
    from types import SimpleNamespace as NS
    from app.schemas.agent_schema import AgentRequest
    followup = ScriptedLLM({'main_agent_decision': [('report_blocked', {'reason': '当前范围证据仍不足'})]})
    monkeypatch.setattr(graph, '_get_llm', lambda: followup)
    monkeypatch.setattr('app.agent.research_semantic_parser.parse_research_semantics', lambda *a, **kw: NS(model_dump=lambda **kw: {}))
    result = ResearchConversationService(db, llm=None).handle(AgentRequest(session_id='v2',
        user_query=current['user_query'], clarification_answer='仅限2021-2023年，仍需40篇'))
    assert result['status'] == 'blocked'
    goal = json.loads(followup.calls[0][1][-1]['content'])['goal']
    assert goal['constraints']['start_year'] == 2021 and goal['constraints']['end_year'] == 2023
    assert goal['constraints']['required_reference_count'] == 40
    assert ResearchSessionRepository(db).get('v2')['clarification'] is None


def test_v2_finalization_reserve_is_applied_before_exposing_search(db):
    current = state()
    current['delivery_mode'] = 'draft_first'
    llm = ScriptedLLM({'main_agent_decision': [('report_blocked', {'reason': '收尾额度内没有可交付材料'})]})
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        from app.agent.execution_budget import finalization_reserve
        reserve = finalization_reserve(current)['requests']
        current['agent_execution_budget'].update(model_request_limit=reserve + 10, llm_requests=10)
        assert TaskMainLoop(llm).run(current, handlers={'search_and_rank': lambda *a: pytest.fail('reserve bypassed')},
            finish_validator=lambda s: False) == 'blocked'
        assert current['finalization_phase'] == 'finalizing'
        assert 'delegate_agent' not in current['allowed_agent_actions']
        assert current['errors'][-1] == {'code': 'main_agent_blocked', 'message': '收尾额度内没有可交付材料'}


def test_search_only_does_not_reserve_nonexistent_writing_phase(db):
    from tests.test_v2_search_runner import handlers
    current, calls = state(), []
    current.update(intent='search_papers', core_deliverables=[], delivery_mode='draft_first',
                   finalization_phase='finalizing', finalization_reason='requests,tokens')
    llm = ScriptedLLM({'main_agent_decision': [('delegate_agent', {
        'agent': 'search', 'profile': 'initial_research', 'goal': 'search'}), ('request_finish', {})]})
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        current['agent_execution_budget'].update(model_request_limit=40, token_limit=120000)
        status = TaskMainLoop(llm).run(current, handlers=handlers(calls), finish_validator=lambda s: bool(s.get('candidate_papers')))
    assert status == 'completed' and calls == ['search']
    assert not current.get('finalization_phase') and not current.get('finalization_reason')
    assert current['agent_execution_budget']['model_request_limit'] == 40
    assert current['agent_execution_budget']['token_limit'] == 120000


def test_v2_publication_refusal_returns_observation_and_can_retry(db, monkeypatch):
    from tests.test_draft_delivery_policy import draft_state
    from app.agent.delivery import published_delivery
    current = draft_state()
    current['session_id'] = 'v2'
    llm = ScriptedLLM({'main_agent_decision': [('publish_draft', {}), ('publish_draft', {}), ('request_finish', {})]})
    publish = AgentController.publish_draft
    calls = []
    def reject_once(self, current, **kwargs):
        calls.append('publish')
        if len(calls) == 1:
            raise ValueError('candidate version changed before publication')
        return publish(self, current, **kwargs)
    monkeypatch.setattr(AgentController, 'publish_draft', reject_once)
    _, finish = graph.build_research_actions(current, llm=llm)
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        status = TaskMainLoop(llm).run(current, handlers={}, finish_validator=finish)
        assert status == 'partial'
        assert len(current['published_versions']) == 1
        assert published_delivery(current)['grade'] == 'draft'
        assert current['required_reference_count'] == 5
        assert any(item.get('status') == 'rejected' for item in current['main_agent_trajectory']['memory'])


def test_cancelled_check_keeps_previously_published_draft_and_hides_new_candidate(db):
    from tests.test_draft_delivery_policy import draft_state
    from app.agent.execution_budget import AgentExecutionCancelled
    current = draft_state()
    current['session_id'] = 'v2'
    llm = ScriptedLLM({'main_agent_decision': [('delegate_agent', {
        'agent': 'check', 'profile': 'independent_check', 'goal': 'check revised candidate'})]})
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        published = AgentController().publish_draft(current)
        current.update(review='UNVERIFIED REPLACEMENT', writing_version=2, pending_result_validation=True)
        def cancel(w, args):
            raise AgentExecutionCancelled('cancel during independent Check')
        assert TaskMainLoop(llm).run(current, handlers={'validate_result': cancel}, finish_validator=lambda s: False) == 'cancelled'
        output = graph._build_output(current)
        assert output['status'] == 'cancelled' and output['body'] == published['body']
        assert 'UNVERIFIED REPLACEMENT' not in output['body']
        assert output['delivery_status'] == 'draft' and not output['requirements_satisfied']


@pytest.mark.parametrize('after_commit', [False, True], ids=['before_publication_commit', 'after_publication_commit'])
def test_publication_and_main_observation_recover_as_one_checkpoint(db, monkeypatch, after_commit):
    from tests.test_draft_delivery_policy import draft_state
    from app.services.durable_execution_service import active_runtime
    current = draft_state()
    current['session_id'] = 'v2'
    llm = ScriptedLLM({'main_agent_decision': [('publish_draft', {})]})
    _, finish = graph.build_research_actions(current, llm=llm)
    class Crash(SystemExit):
        pass
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        runtime = active_runtime()
        runtime.snapshot(current, ledger=current['agent_execution_budget'])
        snapshot = runtime.snapshot
        def crash(candidate, **kwargs):
            if candidate.get('published_versions'):
                if after_commit:
                    snapshot(candidate, **kwargs)
                raise Crash()
            return snapshot(candidate, **kwargs)
        monkeypatch.setattr(runtime, 'snapshot', crash)
        with pytest.raises(Crash):
            TaskMainLoop(llm).run(current, handlers={}, finish_validator=finish)
        monkeypatch.setattr(runtime, 'snapshot', snapshot)
        restored = runtime.restore()
        assert bool(restored.get('published_versions')) == after_commit
        assert not (restored.get('main_agent_trajectory') or {}).get('pending')
    script = [('request_finish', {})] if after_commit else [('publish_draft', {}), ('request_finish', {})]
    llm = ScriptedLLM({'main_agent_decision': script})
    with budget_scope(restored), durable_scope(db, 'v2', restored, restored['agent_execution_budget']):
        assert TaskMainLoop(llm).run(restored, handlers={}, finish_validator=finish) == 'partial'
        assert len(restored['published_versions']) == 1


def test_publication_observation_has_no_separate_precommit_archive(db, monkeypatch):
    from tests.test_draft_delivery_policy import draft_state
    from app.agent.react_context import ReActContext
    from app.schemas.agent_decision_schema import AgentDecision
    from app.services.research_execution_service import _STORE
    from app.services.research_artifact_service import ResearchArtifactService
    from app.services.durable_execution_service import active_runtime
    current = draft_state()
    current['session_id'] = 'v2'
    token = _STORE.set(ResearchArtifactService(db))
    monkeypatch.setattr('app.agent.react_context.archive_payload', lambda *a, **kw: pytest.fail('precommit archive'))
    try:
        with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
            ReActContext(current).proposed(AgentDecision(action='publish_draft'))
            AgentController().publish_draft(current)
            restored = active_runtime().restore()
            assert restored['published_versions'][0]['grade'] == 'draft'
            assert not restored['main_agent_trajectory'].get('pending')
            assert restored['main_agent_trajectory']['memory'][-1]['status'] == 'completed'
    finally:
        _STORE.reset(token)
