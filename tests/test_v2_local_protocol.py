"""局部模型双协议、有限纠错及真实任务入口的计划恢复边界。"""
import json
import pytest

from app.agent.execution_budget import AgentExecutionCancelled, budget_scope
from tests.reference_v1.main_policy import AgentDecisionError
from app.agent.policy.authorization import AuthorizationPolicy
from app.agent.runtime.search_strategy import RoleReActStrategy, RolePlanAndActStrategy
from app.agent.task_main import TaskMainLoop
from app.core.config import get_settings
from app.core.exceptions import NativeToolsUnsupportedError
from app.database.delegation_repository import DelegationRepository
from app.schemas.delegation_schema import DelegationRequest
from app.services.durable_execution_service import durable_scope, active_runtime
from app.services.research_delegation_service import run_delegation
from tests.test_subagent_react import ScriptedLLM
from tests.test_v2_delegation_runtime import db, state
from tests.test_v2_professional_roles import analysis_handlers


def grant():
    return AuthorizationPolicy().issue(DelegationRequest(
        agent='analysis', profile='evidence_build', goal='analyze'), state())


def test_local_native_unsupported_falls_back_even_with_one_attempt(monkeypatch):
    monkeypatch.setattr(get_settings(), 'agent_decision_repair_attempts', 1)
    class LLM:
        native_tools_enabled = True
        calls = []
        def complete_tool_call(self, *args, **kwargs):
            self.calls.append('native')
            raise NativeToolsUnsupportedError('unsupported')
        def complete_messages(self, *args, **kwargs):
            self.calls.append('json')
            return json.dumps({'action': 'extract_paper_cards', 'arguments': {}})
    llm = LLM()
    strategy = RoleReActStrategy(llm)
    assert strategy(grant(), {}, ('extract_paper_cards',)) == ('extract_paper_cards', {})
    strategy(grant(), {}, ('extract_paper_cards',))
    assert llm.calls == ['native', 'json', 'json']


def test_local_repairs_unauthorized_action_before_return(monkeypatch):
    monkeypatch.setattr(get_settings(), 'agent_decision_repair_attempts', 2)
    llm = ScriptedLLM({'analysis_agent_decision': [
        ('generate_deliverables', {}), ('extract_paper_cards', {})]})
    assert RoleReActStrategy(llm)(grant(), {}, ('extract_paper_cards',)) == ('extract_paper_cards', {})
    assert len(llm.calls) == 2


def test_local_repair_exhaustion_is_bounded(monkeypatch):
    monkeypatch.setattr(get_settings(), 'agent_decision_repair_attempts', 2)
    llm = ScriptedLLM({'analysis_agent_decision': [('plan_claims', {'invented': True})] * 2})
    with pytest.raises(AgentDecisionError, match='repair exhausted'):
        RoleReActStrategy(llm)(grant(), {}, ('plan_claims',))
    assert len(llm.calls) == 2


def test_local_cancel_is_not_converted_to_protocol_repair():
    error = AgentExecutionCancelled('cancelled')
    llm = ScriptedLLM({'analysis_agent_decision': [error]})
    with pytest.raises(AgentExecutionCancelled) as caught:
        RoleReActStrategy(llm)(grant(), {}, ('extract_paper_cards',))
    assert caught.value is error and len(llm.calls) == 1


@pytest.mark.parametrize('native', [False, True])
def test_task_main_uses_configured_model_plan_for_analysis(db, monkeypatch, native):
    monkeypatch.setattr(get_settings(), 'agent_v2_analysis_strategy', 'plan_and_act')
    current, calls = state(), []
    current['paper_details'] = [{'paper_id': 'p', 'title': 'Study'}]
    plan = {'steps': [{'action': name, 'arguments': {}} for name in
        ('extract_paper_cards', 'validate_routes', 'plan_claims')]}
    class LLM(ScriptedLLM):
        plans = 0
        def complete_tool_call(self, messages, *, tools, operation):
            if operation == 'analysis_agent_plan':
                self.plans += 1
                return {'name': 'submit_local_plan', 'arguments': plan}
            return super().complete_tool_call(messages, tools=tools, operation=operation)
        def complete_messages(self, messages, *, operation, **kwargs):
            if operation == 'analysis_agent_plan':
                self.plans += 1
                return json.dumps(plan)
            return super().complete_messages(messages, operation=operation, **kwargs)
    llm = LLM({'main_agent_decision': [
        ('delegate_agent', {'agent': 'analysis', 'profile': 'evidence_build', 'goal': 'analyze'}),
        ('request_finish', {})]}, native=native)
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        result = TaskMainLoop(llm).run(current, handlers=analysis_handlers(calls),
            finish_validator=lambda s: bool(s.get('claim_plans')))
        assert result == 'completed' and calls == ['extract', 'routes', 'claims']
        assert llm.plans == 1
        assert list(current['delegation_control'].values())[0]['strategy_mode'] == 'plan_and_act'


def test_planner_validates_future_steps_before_any_execution(monkeypatch):
    monkeypatch.setattr(get_settings(), 'agent_decision_repair_attempts', 2)
    llm = ScriptedLLM({'analysis_agent_plan': [('submit_local_plan', {'steps': [
        {'action': 'extract_paper_cards', 'arguments': {}},
        {'action': 'plan_claims', 'arguments': {'invented': 'bad'}}]})] * 2})
    with pytest.raises(AgentDecisionError):
        RolePlanAndActStrategy(llm).propose_plan(grant(), {}, ('extract_paper_cards',))
    assert len(llm.calls) == 2


def test_invalid_strategy_does_not_create_delegation(db):
    current = state()
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        request = DelegationRequest(agent='search', profile='initial_research', goal='search')
        with pytest.raises(ValueError, match='limited to evidence_build'):
            run_delegation(current, request, llm=object(), strategy='plan_and_act', handlers={})
        assert DelegationRepository(active_runtime()).active() is None
        assert 'active_delegation_id' not in current
