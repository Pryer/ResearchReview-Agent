"""Search任务级运行与恢复；模型与研究工具均为明确的受控替身。"""
import pytest
from app.agent.runtime.delegation_runner import DelegationRunner
from app.agent.runtime.action_executor import ActionExecutor
from app.agent.controller import AgentController
from app.agent.policy.authorization import AuthorizationPolicy
from app.schemas.delegation_schema import DelegationRequest
from app.database.delegation_repository import DelegationRepository
from app.services.durable_execution_service import durable_scope, active_runtime
from app.agent.execution_budget import budget_scope
from tests.test_v2_delegation_runtime import db, state


def handlers(calls):
    paper = {'paper_id': 'p', 'title': 'Study', 'year': 2024, 'authors': ['A']}
    def search(w, args):
        calls.append('search')
        w.update(candidate_papers=[paper.copy()], ranked_papers=[paper.copy()])
    def fetch(w, args):
        calls.append('fetch')
        w['paper_details'] = [paper.copy()]
    return {'search_and_rank': search, 'fetch_metadata': fetch}


def test_single_task_runs_two_actions_and_reports_all_results(db):
    current, calls = state(), []
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        repo = DelegationRepository(active_runtime())
        grant = AuthorizationPolicy().issue(DelegationRequest(agent='search', profile='initial_research', goal='research'), current)
        repo.create(grant)
        runner = DelegationRunner(ActionExecutor(AgentController()))
        result = runner.run(grant.delegation_id, current, handlers(calls))
        assert calls == ['search', 'fetch']
        assert result.outcome == 'satisfied' and len(result.action_result_refs) == 2
        assert runner.run(grant.delegation_id, current, handlers(calls)) == result
        assert calls == ['search', 'fetch']


def test_resume_after_second_handler_failure_does_not_replay_search(db):
    current, calls = state(), []
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        repo = DelegationRepository(active_runtime())
        grant = AuthorizationPolicy().issue(DelegationRequest(agent='search', profile='initial_research', goal='research'), current)
        repo.create(grant)
        normal = handlers(calls)
        def fail(w, a):
            raise ValueError('fetch failed before submit')
        runner = DelegationRunner(ActionExecutor(AgentController()))
        with pytest.raises(ValueError):
            runner.run(grant.delegation_id, current, {**normal, 'fetch_metadata': fail})
        first_ids = [item.invocation_id for item in repo.invocations(grant.delegation_id)]
        result = runner.run(grant.delegation_id, current, normal)
        assert calls == ['search', 'fetch'] and result.outcome == 'satisfied'
        assert first_ids == [item.invocation_id for item in repo.invocations(grant.delegation_id)]


def test_strategy_gets_safe_planning_view_and_can_ask_for_help(db):
    current = state()
    current['review'] = 'private draft not needed by Search planner'
    def strategy(grant, context, available):
        assert 'review' not in context and 'paper_cards' not in context
        return 'report_to_main', {}
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        repo = DelegationRepository(active_runtime())
        grant = AuthorizationPolicy().issue(DelegationRequest(agent='search', profile='initial_research', goal='research'), current)
        repo.create(grant)
        result = DelegationRunner(ActionExecutor(AgentController()), strategy=strategy).run(
            grant.delegation_id, current, handlers([]))
        assert result.outcome == 'needs_help' and not result.action_result_refs


def test_empty_search_is_not_accepted_as_completed(db):
    current = state()
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        repo = DelegationRepository(active_runtime())
        grant = AuthorizationPolicy().issue(DelegationRequest(agent='search', profile='initial_research', goal='research'), current)
        repo.create(grant)
        result = DelegationRunner(ActionExecutor(AgentController())).run(grant.delegation_id, current,
            {'search_and_rank': lambda w, a: None})
        assert result.outcome == 'needs_help'
        assert 'no_business_progress' in result.unresolved or 'no_eligible_candidates' in result.unresolved


@pytest.mark.parametrize('native', [False, True])
def test_local_model_strategy_native_and_json_choose_same_actions(db, native):
    from app.agent.runtime.search_strategy import SearchReActStrategy
    import json
    class LLM:
        native_tools_enabled = native
        def decision(self, messages):
            context = json.loads(messages[-1]['content'])['context']
            return 'fetch_metadata' if context['candidates'] else 'search_and_rank'
        def complete_messages(self, messages, **kwargs):
            return json.dumps({'action': self.decision(messages), 'arguments': {}})
        def complete_tool_call(self, messages, **kwargs):
            return {'name': self.decision(messages), 'arguments': {}}
    current, calls = state(), []
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        repo = DelegationRepository(active_runtime())
        grant = AuthorizationPolicy().issue(DelegationRequest(agent='search', profile='initial_research', goal='research'), current)
        repo.create(grant)
        result = DelegationRunner(ActionExecutor(AgentController()), strategy=SearchReActStrategy(LLM())).run(
            grant.delegation_id, current, handlers(calls))
        assert calls == ['search', 'fetch'] and result.outcome == 'satisfied'


def test_progress_requires_persisted_success_or_empty_exploration():
    from app.agent.runtime.delegation_runner import search_progress
    baseline = search_progress({})
    assert search_progress({'search_query_ledger': {'q': {'status': 'running', 'attempts': 1}}}) == baseline
    assert search_progress({'search_query_ledger': {'q': {'status': 'failed', 'attempts': 2}}}) == baseline
    explored = search_progress({'search_query_ledger': {'q': {'status': 'empty', 'attempts': 1}}})
    assert explored != baseline
    assert search_progress({'search_query_ledger': {'q': {'status': 'empty', 'attempts': 20,
        'updated_at': 'new timestamp'}}}) == explored
