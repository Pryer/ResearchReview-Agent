"""主控只选择任务Profile；专业动作、核验与最终结束由代码执行。"""
import pytest
from app.agent.task_main import TaskMainLoop
from app.agent.execution_budget import budget_scope
from app.services.durable_execution_service import durable_scope
from tests.test_v2_delegation_runtime import db, state
from tests.test_subagent_react import ScriptedLLM
from tests.test_v2_search_runner import handlers


def test_task_tool_scope_and_profiles_describe_actual_authorization_boundary():
    from app.agent.task_main import TaskMainPolicy
    from app.agent.policy.capability_profiles import PROFILES
    policy = TaskMainPolicy(object())
    properties = policy.tools[0]['function']['parameters']['properties']
    assert set(properties['profile']['enum']) == set(PROFILES)
    assert properties['scope']['additionalProperties'] is False
    assert set(properties['scope']['properties']) == {'section_ids', 'gap_ids', 'paper_ids'}
    assert 'single_action' not in properties['profile']['enum']
    assert 'legacy_single_v1' not in properties['profile']['enum']


@pytest.mark.parametrize('native', [False, True])
def test_task_policy_exposes_only_currently_admitted_tools_and_profiles(native):
    from app.agent.task_main import TaskMainPolicy
    from app.agent.context_builder import build_main_agent_context
    from app.agent.react_context import ReActContext
    import copy
    class RecordingLLM(ScriptedLLM):
        def complete_tool_call(self, messages, *, tools, operation):
            self.tools = copy.deepcopy(tools)
            return super().complete_tool_call(messages, tools=tools, operation=operation)
    llm = RecordingLLM({'main_agent_decision': [('delegate_agent', {
        'profile': 'independent_check', 'goal': 'check'})]}, native=native)
    current = state()
    policy = TaskMainPolicy(llm)
    decision = policy.decide(build_main_agent_context(current), ['delegate_agent', 'report_blocked'],
                             ReActContext(current), profiles=['independent_check'])
    assert decision.arguments['agent'] == 'check'
    if native:
        assert {tool['function']['name'] for tool in llm.tools} == {'delegate_agent', 'report_blocked'}
        field = llm.tools[0]['function']['parameters']['properties']['profile']
        assert 'agent' not in llm.tools[0]['function']['parameters']['properties']
        assert field['enum'] == ['independent_check'] and 'draft_generation' not in field['description']
    else:
        assert '"enum": ["independent_check"]' in llm.calls[0][1][0]['content']
        assert 'draft_generation' not in llm.calls[0][1][0]['content']
    assert len(policy.tools[0]['function']['parameters']['properties']['profile']['enum']) == 8


@pytest.mark.parametrize('native', [False, True])
def test_main_repairs_real_scope_misuse_before_authorizing_handler(db, native):
    current, calls = state(), []
    current.update(start_year=2020, end_year=2023, required_reference_count=3)
    good = {'agent': 'search', 'profile': 'initial_research', 'goal': 'research', 'scope': {}}
    bad = {**good, 'scope': {'topic': 'RAG', 'year_start': 2020, 'year_end': 2023, 'target_unique_papers': 3}}
    llm = ScriptedLLM({'main_agent_decision': [('delegate_agent', bad), ('delegate_agent', good), ('request_finish', {})]}, native=native)
    def within_year_range(handler):
        def run(working, arguments):
            handler(working, arguments)
            for field in ('candidate_papers', 'ranked_papers', 'paper_details'):
                for paper in working.get(field) or []:
                    paper['year'] = 2022
        return run
    operations = {name: within_year_range(handler) for name, handler in handlers(calls).items()}
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        status = TaskMainLoop(llm).run(current, handlers=operations, finish_validator=lambda s: bool(s.get('paper_details')))
    assert status == 'completed' and calls == ['search', 'fetch']
    assert current['start_year'] == 2020 and current['end_year'] == 2023 and current['required_reference_count'] == 3
    assert len(llm.calls) == 3


@pytest.mark.parametrize('scope', [{'year': 2022}, {'paper_ids': 'p1'}, {'section_ids': ['']}, {'gap_ids': ['g', 'g']}])
def test_delegation_scope_rejects_unknown_keys_and_non_id_values(scope):
    from app.schemas.delegation_schema import DelegationRequest
    with pytest.raises(ValueError, match='scope'):
        DelegationRequest(agent='search', profile='initial_research', goal='research', scope=scope)


@pytest.mark.parametrize('native', [False, True])
def test_main_delegates_search_task_then_finishes(db, native):
    from app.services.execution_report_capture import Capture, capture_scope
    current, calls = state(), []
    llm = ScriptedLLM({'main_agent_decision': [
        ('delegate_agent', {'agent': 'search', 'profile': 'initial_research', 'goal': 'research'}),
        ('request_finish', {})]}, native=native)
    capture = Capture('v2', 'research')
    with capture_scope(capture), budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        result = TaskMainLoop(llm).run(current, handlers=handlers(calls),
            finish_validator=lambda s: bool(s.get('paper_details')))
        assert result == 'completed'
        assert calls == ['search', 'fetch']
        assert not current.get('active_delegation_id')
        assert [r['decision']['action'] for r in capture.rounds] == ['delegate_agent', 'request_finish']
        assert [r['action'] for r in current['main_agent_decisions']] == ['delegate_agent', 'request_finish']
        assert all(set(r['context']) == {'goal', 'state', 'key_evidence', 'decisions', 'open_questions'} for r in capture.rounds)


def test_main_cannot_finish_without_quality_gate(db):
    current = state()
    llm = ScriptedLLM({'main_agent_decision': [('request_finish', {}), ('report_blocked', {'reason': 'missing evidence'})]})
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        assert TaskMainLoop(llm).run(current, handlers={}, finish_validator=lambda s: False) == 'blocked'


def test_handler_value_error_is_not_misclassified_as_rejected_model_request(db):
    current, calls = state(), []
    llm = ScriptedLLM({'main_agent_decision': [
        ('delegate_agent', {'agent': 'search', 'profile': 'initial_research', 'goal': 'research'})]})
    def broken(w, args):
        calls.append('handler')
        raise ValueError('invalid provider result')
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        with pytest.raises(ValueError, match='invalid provider result'):
            TaskMainLoop(llm).run(current, handlers={'search_and_rank': broken}, finish_validator=lambda s: False)
        assert calls == ['handler'] and len(llm.calls) == 1


def test_resume_refreshes_main_rejection_counter_after_state_replacement(db):
    from app.agent.policy.authorization import AuthorizationPolicy
    from app.database.delegation_repository import DelegationRepository
    from app.schemas.delegation_schema import DelegationRequest
    from app.services.durable_execution_service import active_runtime
    current, calls = state(), []
    current['task_main_control'] = {'stalled': 0, 'previous': '', 'attempts': {}}
    llm = ScriptedLLM({'main_agent_decision': [('delegate_agent', {
        'agent': 'writing', 'profile': 'draft_generation', 'goal': 'not yet available'})] * 3})
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        repo = DelegationRepository(active_runtime())
        grant = AuthorizationPolicy().issue(DelegationRequest(
            agent='search', profile='initial_research', goal='research'), current)
        repo.create(grant)
        current['active_delegation_id'] = grant.delegation_id
        result = TaskMainLoop(llm).run(current, handlers=handlers(calls), finish_validator=lambda s: False)
        assert result == 'blocked' and calls == ['search', 'fetch']
        assert current['task_main_control']['rejections'] == 3


def test_main_mandatory_task_uses_exact_single_operation(db):
    current, calls = state(), []
    current['agent_mandatory_actions'] = [{'action': 'search_and_rank', 'arguments': {}}]
    llm = ScriptedLLM({'main_agent_decision': [('request_finish', {})]})
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        result = TaskMainLoop(llm).run(current, handlers=handlers(calls),
            finish_validator=lambda s: bool(s.get('ranked_papers')))
        assert result == 'completed' and calls == ['search']
        assert not current.get('agent_mandatory_actions')
        from sqlalchemy import select
        from app.database.models import ResearchDelegationModel
        from app.schemas.delegation_schema import DelegationGrant
        grants = [DelegationGrant.model_validate_json(row.grant_json) for row in db.scalars(select(ResearchDelegationModel))]
        assert len(grants) == 1 and grants[0].profile == 'single_action'


def test_graph_can_select_v2_main_with_existing_search_finish_gate(db, monkeypatch):
    from app.agent import graph
    from app.core.config import get_settings
    from tests.test_research_delegation_service import configure
    current, calls = state(), []
    current.update(intent='search_papers', max_papers=1, required_reference_count=1)
    configure(monkeypatch, calls)
    monkeypatch.setattr(get_settings(), 'agent_architecture_version', 'v2')
    llm = ScriptedLLM({'main_agent_decision': [
        ('delegate_agent', {'agent': 'search', 'profile': 'initial_research', 'goal': 'research'}),
        ('request_finish', {})]})
    monkeypatch.setattr(graph, '_get_llm', lambda: llm)
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        graph._run_autonomous_pipeline(current)
        assert current['result_status'] == 'completed'
        assert current['search_result_quality']['passed']
        assert calls == ['search']


def test_task_main_hands_writer_to_independent_check_and_rejects_analysis(db):
    from tests.test_main_quality_loop import prepared
    from tests.test_check_agent import checked
    from app.agent.checking import check_is_current
    current, calls = prepared(), []
    current['session_id'] = 'v2'
    llm = ScriptedLLM({'main_agent_decision': [
        ('delegate_agent', {'agent': 'writing', 'profile': 'draft_generation', 'goal': 'write'}),
        ('delegate_agent', {'agent': 'analysis', 'profile': 'evidence_build', 'goal': 'skip check'}),
        ('delegate_agent', {'agent': 'check', 'profile': 'independent_check', 'goal': 'check'}),
        ('request_finish', {})]})
    def writer(w, args):
        calls.append('write')
        w.update(review='Candidate text [p0]', pending_result_validation=True)
    def checker(w, args):
        calls.append('check')
        checked(w)
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        result = TaskMainLoop(llm).run(current, handlers={
            'generate_deliverables': writer, 'validate_result': checker,
            'extract_paper_cards': lambda *a: pytest.fail('analysis ran before Check')},
            finish_validator=lambda s: check_is_current(s) and s['check_report']['verdict'] == 'passed')
        assert result == 'completed' and calls == ['write', 'check']


def test_old_task_configuration_upgrades_only_architecture_after_engine_retirement(monkeypatch):
    from app.core.config import get_settings
    from app.agent.runtime_config import task_settings_scope
    monkeypatch.setattr(get_settings(), 'agent_architecture_version', 'v2')
    current = {'effective_config': {'version': 'old', 'values': {'agent_main_max_rounds': 12}}}
    with task_settings_scope(current) as settings:
        assert settings.agent_architecture_version == 'v2'
        assert settings.agent_main_max_rounds == 12
    assert current['effective_config']['values']['agent_architecture_version'] == 'v2'
    assert current['effective_config']['architecture_migration'] == {'from': 'v1', 'to': 'v2'}


@pytest.mark.parametrize('after_commit', [False, True])
def test_delegation_observation_and_marker_cleanup_recover_atomically(db, monkeypatch, after_commit):
    from app.services.durable_execution_service import active_runtime
    current, calls = state(), []
    llm = ScriptedLLM({'main_agent_decision': [
        ('delegate_agent', {'agent': 'search', 'profile': 'initial_research', 'goal': 'research'})]})
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        runtime = active_runtime()
        original = runtime.snapshot
        def crash(candidate, **kwargs):
            if 'active_delegation_id' in current and 'active_delegation_id' not in candidate:
                if after_commit:
                    original(candidate, **kwargs)
                raise RuntimeError('at final observation checkpoint')
            return original(candidate, **kwargs)
        monkeypatch.setattr(runtime, 'snapshot', crash)
        with pytest.raises(RuntimeError, match='observation checkpoint'):
            TaskMainLoop(llm).run(current, handlers=handlers(calls), finish_validator=lambda s: True)
        monkeypatch.setattr(runtime, 'snapshot', original)
        restored = runtime.restore()
        assert bool(restored.get('active_delegation_id')) is not after_commit
        assert bool(restored['main_agent_trajectory'].get('pending')) is not after_commit
    llm = ScriptedLLM({'main_agent_decision': [('request_finish', {})]})
    with budget_scope(restored), durable_scope(db, 'v2', restored, restored['agent_execution_budget']):
        assert TaskMainLoop(llm).run(restored, handlers=handlers(calls),
            finish_validator=lambda s: bool(s.get('paper_details'))) == 'completed'
        assert calls == ['search', 'fetch']
        assert not restored['main_agent_trajectory'].get('pending')
        assert restored['main_agent_trajectory']['observation_count'] == 2
