"""角色模型实际路由、任务冻结、核验身份和三层上下文回归。"""
import json
import pytest
from pydantic import ValidationError

from app.core.config import Settings, get_settings
from app.services.llm_service import LLMService
from app.agent.runtime_config import task_settings_scope
from app.agent.react_context import ContextBudgetExceeded
from app.services.verifier_profile import verifier_profile
from app.services.role_execution_profile import execution_profile
from tests.test_llm_service import _fake_client, _resp, _tool_resp, _timeout_exc
from tests.test_v2_delegation_runtime import db, state


def service(monkeypatch, profiles):
    monkeypatch.setattr(get_settings(), 'agent_role_model_profiles', json.dumps(profiles))
    result = LLMService()
    result.api_key, result.base_url, result.model, result.provider = 'primary-secret', 'https://primary.test', 'global-model', 'primary-provider'
    result.backup_api_key, result.backup_base_url = 'backup-secret', 'https://backup.test'
    result.backup_enabled, result.backup_model, result.backup_provider = True, 'global-backup', 'backup-provider'
    result.thinking_enabled = False
    return result


@pytest.mark.parametrize('profiles', [
    {'unknown': {}}, {'check': {'api_key': 'forbidden'}}, {'writing': {'temperature': 3}},
    {'analysis': {'max_tokens': 0}}, {'search': {'fallback': 'primary'}},
    {'check': {'fallback': 'disabled', 'fallback_model': 'unreachable'}},
])
def test_invalid_or_credential_bearing_profile_is_rejected(profiles):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, agent_role_model_profiles=json.dumps(profiles))


def test_roles_use_independent_parameters_and_rebinding_uses_original_transport(monkeypatch):
    root = service(monkeypatch, {'analysis': {'model': 'analysis-model', 'temperature': 0.1},
        'writing': {'provider_slot': 'backup', 'model': 'writer-model', 'fallback': 'disabled'}})
    requests = []
    root._client = _fake_client(lambda **kw: requests.append(('primary', kw)) or _resp('ok'))
    root._backup_client = _fake_client(lambda **kw: requests.append(('backup', kw)) or _resp('ok'))
    analysis = root.for_agent('analysis')
    writer = analysis.for_agent('writing')
    assert analysis.for_agent('analysis') is analysis
    assert analysis.complete('extract', retry_empty=False) == 'ok'
    assert writer.complete('write', retry_empty=False) == 'ok'
    assert writer.for_agent('search').complete('search', retry_empty=False) == 'ok'
    assert [(slot, kw['model']) for slot, kw in requests] == [
        ('primary', 'analysis-model'), ('backup', 'writer-model'), ('primary', 'global-model')]
    assert requests[0][1]['temperature'] == 0.1
    assert root.model == 'global-model' and root.provider == 'primary-provider'
    assert root.for_agent('analysis')._service is analysis._service


def test_fallback_uses_actual_role_model_and_receipt(monkeypatch):
    from app.services.verifier_profile import verifier_receipt
    root = service(monkeypatch, {'check': {'model': 'check-model', 'fallback_model': 'check-fallback'}})
    calls = []
    def primary(**kw):
        calls.append(kw['model'])
        raise _timeout_exc()
    def backup(**kw):
        calls.append(kw['model'])
        response = _resp('supported')
        response.model, response.id = 'actual-check-fallback', 'response'
        return response
    root._client, root._backup_client = _fake_client(primary), _fake_client(backup)
    assert root.for_agent('check').complete('verify', operation='verify_claim_entailment', retry_empty=False) == 'supported'
    assert calls == ['check-model', 'check-fallback']
    assert verifier_receipt()['provider'] == 'backup-provider'
    assert verifier_receipt()['requested_model'] == 'check-fallback'
    assert verifier_receipt()['response_model'] == 'actual-check-fallback'


def test_disabled_or_unconfigured_role_fallback_never_calls_an_unapproved_provider(monkeypatch):
    from app.core.exceptions import LLMInvocationError
    root = service(monkeypatch, {'check': {'fallback': 'disabled'}})
    root._client = _fake_client(lambda **kw: (_ for _ in ()).throw(_timeout_exc()))
    root._backup_client = _fake_client(lambda **kw: pytest.fail('fallback disabled'))
    with pytest.raises(LLMInvocationError):
        root.for_agent('check').complete('verify', retry_empty=False)
    monkeypatch.setattr(get_settings(), 'agent_role_model_profiles', '{"writing":{"provider_slot":"backup"}}')
    root.backup_enabled = False
    with pytest.raises(ValueError, match='configured backup'):
        root.for_agent('writing')


def test_profiles_are_frozen_and_old_tasks_do_not_adopt_new_process_profiles(monkeypatch):
    base = get_settings()
    monkeypatch.setattr(base, 'agent_role_model_profiles', '{"analysis":{"model":"frozen"}}')
    current = {}
    with task_settings_scope(current):
        assert LLMService().for_agent('analysis').model == 'frozen'
    monkeypatch.setattr(base, 'agent_role_model_profiles', '{"analysis":{"model":"changed"}}')
    with task_settings_scope(current):
        assert LLMService().for_agent('analysis').model == 'frozen'
    old = {'effective_config': {'values': {'llm_model': 'old-global'}}}
    with task_settings_scope(old):
        assert LLMService().for_agent('analysis').model == 'old-global'
    persisted = json.dumps(current['effective_config'])
    assert 'api_key' not in persisted and 'base_url' not in persisted


def test_check_profile_invalidates_reports_without_unrelated_role_churn(monkeypatch):
    from app.agent.checking import check_input_fingerprint
    monkeypatch.setattr(get_settings(), 'agent_role_model_profiles', '{}')
    before, fingerprint = verifier_profile(), check_input_fingerprint({'review': 'body'})
    monkeypatch.setattr(get_settings(), 'agent_role_model_profiles', '{"search":{"model":"new-search"}}')
    assert verifier_profile() == before
    assert check_input_fingerprint({'review': 'body'}) == fingerprint
    monkeypatch.setattr(get_settings(), 'agent_role_model_profiles', '{"check":{"model":"new-check"}}')
    assert verifier_profile() != before
    assert check_input_fingerprint({'review': 'body'}) != fingerprint


@pytest.mark.parametrize('native', [False, True])
def test_role_window_counts_prefix_brief_and_tools_before_network(monkeypatch, native):
    root = service(monkeypatch, {'analysis': {'max_tokens': 64, 'control_plane_max_tokens': 64,
        'context_window_tokens': 12000}})
    root._client = _fake_client(lambda **kw: pytest.fail('over-budget request sent'))
    bound = root.for_agent('analysis')
    messages = [{'role': 'user', 'content': '证据' * 4000}]
    with pytest.raises(ContextBudgetExceeded, match='role_context_window'):
        if native:
            bound.complete_tool_call(messages, tools=[{'function': {'name': 'x'}}])
        else:
            bound.complete_messages(messages, retry_empty=False)
    assert len(messages) == 1


def test_execution_profile_fingerprints_address_without_persisting_it(monkeypatch):
    root = service(monkeypatch, {'analysis': {'model': 'analysis-model'}})
    bound = root.for_agent('analysis')
    first = execution_profile(bound, 'analysis')
    assert first['model'] == 'analysis-model'
    assert not any(value in json.dumps(first) for value in ('primary-secret', 'backup-secret', 'https://'))
    bound._service.base_url = 'https://changed.test'
    assert execution_profile(bound, 'analysis')['fingerprint'] != first['fingerprint']


def test_action_brief_keeps_v2_identity_and_scope_lifetimes_distinct():
    from app.agent.task_context import execution_task_brief, current_task_brief
    from app.services.invocation_scope import invocation_scope
    from app.schemas.delegation_schema import ActionInvocation
    from app.schemas.agent_task_schema import AgentTask, AgentRole
    task = AgentTask(task_id='task', role=AgentRole.ANALYSIS, operation='extract_paper_cards', objective='extract',
        source_state_version='v', source_state_fingerprint='source', input_artifact_refs=['artifact'], paper_scope=['p'])
    invocation = ActionInvocation(delegation_id='delegation', step_index=0, action=task.operation,
        expected_version=4, source_fingerprint='source')
    with invocation_scope(invocation), execution_task_brief(task):
        brief = current_task_brief('analysis')
        assert brief['task_id'] == 'task' and brief['delegation_id'] == 'delegation'
        assert brief['invocation_id'] == invocation.invocation_id and brief['attempt_id']
        assert brief['expected_version'] == 4 and brief['paper_scope'] == ['p']
        assert brief['input_artifact_refs'] == ['artifact']
        brief['input_artifact_refs'].clear()
        assert current_task_brief('analysis')['input_artifact_refs'] == ['artifact']
        assert current_task_brief('writing') is None
    assert current_task_brief('analysis') is None


def test_v2_planning_uses_role_model_and_current_business_version(db, monkeypatch):
    from app.agent.execution_budget import budget_scope
    from app.services.durable_execution_service import durable_scope
    from app.services.research_delegation_service import run_delegation
    from tests.test_v2_search_runner import handlers
    from tests.test_tool_execution import events
    root = service(monkeypatch, {'search': {'model': 'search-planner', 'control_plane_max_tokens': 128}})
    requests, calls = [], []
    choices = iter(['search_and_rank', 'fetch_metadata'])
    root._client = _fake_client(lambda **kw: requests.append(kw) or _tool_resp(next(choices), {}))
    current = state()
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        result = run_delegation(current, {'agent': 'search', 'profile': 'initial_research', 'goal': 'research'},
            llm=root, strategy='react', handlers=handlers(calls))
        metadata = events(db, 'delegation_execution')[0]['execution_profile']
        assert metadata['model'] == 'search-planner'
        assert current['delegation_control'][result.delegation_id]['execution_profile'] == metadata
    assert calls == ['search', 'fetch']
    assert [request['model'] for request in requests] == ['search-planner', 'search-planner']
    versions = []
    for request in requests:
        brief = next(message['content'] for message in request['messages'] if '"stage": "planning"' in message.get('content', ''))
        assert result.delegation_id in brief
        payload = json.loads(request['messages'][-1]['content'])
        assert 'paper_cards' not in payload['context']
        versions.append(payload['context']['state_version'])
    assert versions == [0, 1]


def test_resume_rejects_changed_role_profile_before_replaying_committed_action(db, monkeypatch):
    from app.agent.execution_budget import budget_scope
    from app.services.durable_execution_service import durable_scope
    from app.services.research_delegation_service import run_delegation
    from app.database.runtime_repository import RuntimeConflict
    from tests.test_v2_search_runner import handlers
    root = service(monkeypatch, {'search': {'model': 'original'}})
    current, calls = state(), []
    request = {'agent': 'search', 'profile': 'initial_research', 'goal': 'research'}
    operations = handlers(calls)
    def interrupted(w, args):
        raise RuntimeError('interrupted before metadata commit')
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        with pytest.raises(RuntimeError, match='interrupted'):
            run_delegation(current, request, llm=root, handlers={**operations, 'fetch_metadata': interrupted})
        monkeypatch.setattr(get_settings(), 'agent_role_model_profiles', '{"search":{"model":"changed"}}')
        with pytest.raises(RuntimeConflict, match='retain.*profile'):
            run_delegation(current, request, llm=root, handlers=operations)
        assert calls == ['search']
        monkeypatch.setattr(get_settings(), 'agent_role_model_profiles', '{"search":{"model":"original"}}')
        assert run_delegation(current, request, llm=root, handlers=operations).outcome == 'satisfied'
        assert calls == ['search', 'fetch']


def test_task_main_and_writer_use_bound_models_from_supplied_service(monkeypatch):
    from app.agent.task_main import TaskMainPolicy
    from app.agent.graph import build_research_actions
    root = service(monkeypatch, {'main': {'model': 'main-model'}, 'writing': {'model': 'writer-model'}})
    assert TaskMainPolicy(root).llm.model == 'main-model'
    seen = []
    def generate(current, **kwargs):
        seen.append(kwargs['llm'].model)
        current['review'] = 'Candidate body'
    monkeypatch.setattr('app.agent.graph._generate_deliverables_or_block', generate)
    actions, _ = build_research_actions({}, llm=root)
    actions['generate_deliverables']({}, {})
    assert seen == ['writer-model']


def test_output_expansion_rechecks_window_and_preserves_first_paid_usage(monkeypatch):
    from types import SimpleNamespace as NS
    from app.agent.react_context import estimate_tokens
    from app.agent.execution_budget import budget_scope, is_control_exception
    root = service(monkeypatch, {'analysis': {'max_tokens': 64, 'control_plane_max_tokens': 64}})
    requests = []
    response = _resp('', finish_reason='length')
    response.choices[0].message.reasoning_content = 'reasoning'
    response.usage = NS(prompt_tokens=5, completion_tokens=3)
    root._client = _fake_client(lambda **kw: requests.append(kw) or response)
    root._backup_client = _fake_client(lambda **kw: pytest.fail('context exhaustion cannot trigger provider fallback'))
    role = root.for_agent('analysis')
    messages = [{'role': 'system', 'content': role._system_prompt}, {'role': 'user', 'content': 'extract'}]
    role._service.context_window_tokens = estimate_tokens(messages, []) + 200
    current = state()
    with budget_scope(current), pytest.raises(ContextBudgetExceeded) as caught:
        role.complete('extract', retry_empty=True)
    assert is_control_exception(caught.value)
    assert len(requests) == 1 and requests[0]['max_tokens'] == 64
    assert current['agent_execution_budget']['llm_requests'] == 1
    assert current['agent_execution_budget']['llm_tokens'] == 8


@pytest.mark.parametrize('policy', ['local', 'main', 'v1'])
def test_window_exhaustion_is_never_repaired_as_model_output(policy):
    from app.agent.context_builder import build_main_agent_context
    from app.agent.react_context import ReActContext
    from tests.reference_v1.main_policy import MainAgentPolicy
    from app.agent.task_main import TaskMainPolicy
    from app.agent.runtime.search_strategy import RoleReActStrategy
    from app.agent.policy.authorization import AuthorizationPolicy
    from app.schemas.delegation_schema import DelegationRequest
    class Exhausted:
        native_tools_enabled = True
        calls = 0
        def complete_tool_call(self, *args, **kwargs):
            self.calls += 1
            raise ContextBudgetExceeded('role_context_window_exceeded')
    llm, current = Exhausted(), state()
    with pytest.raises(ContextBudgetExceeded):
        if policy == 'local':
            grant = AuthorizationPolicy().issue(DelegationRequest(agent='search', profile='initial_research', goal='research'), current)
            RoleReActStrategy(llm)(grant, {}, ['search_and_rank'])
        elif policy == 'main':
            TaskMainPolicy(llm).decide(build_main_agent_context(current), ['report_blocked'], ReActContext(current))
        else:
            MainAgentPolicy(llm).decide(build_main_agent_context(current), allowed_actions=['report_blocked'], trajectory=ReActContext(current))
    assert llm.calls == 1
