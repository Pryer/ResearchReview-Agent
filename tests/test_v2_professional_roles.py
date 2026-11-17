"""四角色共用任务运行器；独立Check及字段授权保持硬边界。"""
import pytest
from app.agent.runtime.delegation_runner import DelegationRunner
from app.agent.runtime.strategies import PlanAndActStrategy
from app.agent.runtime.action_executor import ActionExecutor
from app.agent.controller import AgentController
from app.agent.policy.authorization import AuthorizationPolicy
from app.database.delegation_repository import DelegationRepository
from app.schemas.delegation_schema import DelegationRequest
from app.services.durable_execution_service import durable_scope, active_runtime
from app.agent.execution_budget import budget_scope
from tests.test_v2_delegation_runtime import db, state


def evidence():
    quote = 'The method uses retrieval.'
    return {'paper_id': 'p', 'title': 'Study', 'year': 2024, 'quality_status': 'partial',
        'evidence_source': 'abstract', 'evidence_state': {'access_level': 'abstract'},
        'evidence_spans': [{'evidence_id': 'p:e1', 'text': quote, 'source_type': 'abstract', 'page': 1}],
        'field_claims': {'method': [{'claim': quote, 'source_text': quote, 'evidence_id': 'p:e1',
            'explicitly_reported': True, 'source_section': 'Abstract', 'evidence_level': 'abstract'}]}}


def analysis_handlers(calls):
    def extract(w, a):
        calls.append('extract')
        w['paper_cards'] = [evidence()]
    def routes(w, a):
        calls.append('routes')
        w['validated_routes'] = [{'route_id': 'r', 'paper_ids': ['p']}]
    def claims(w, a):
        calls.append('claims')
        w['claim_plans'] = [{'route_id': 'r', 'claims': [{'claim_id': 'c', 'claim_text': 'The method uses retrieval.',
            'evidence_ids': ['p:e1'], 'support_level': 'single', 'claim_type': 'method'}]}]
    return {'extract_paper_cards': extract, 'validate_routes': routes, 'plan_claims': claims}


@pytest.mark.parametrize('planned', [False, True])
def test_analysis_builds_authorized_claims_in_one_task(db, planned):
    current, calls = state(), []
    current['paper_details'] = [{'paper_id': 'p', 'title': 'Study'}]
    def planner(grant, context, available):
        assert 'paper_cards' not in context
        return [{'action': name, 'arguments': {}} for name in
                ['extract_paper_cards', 'validate_routes', 'plan_claims']]
    strategy = PlanAndActStrategy(planner) if planned else None
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        repo = DelegationRepository(active_runtime())
        grant = AuthorizationPolicy().issue(DelegationRequest(agent='analysis', profile='evidence_build', goal='analyze'), current)
        repo.create(grant)
        result = DelegationRunner(ActionExecutor(AgentController()), strategy=strategy).run(
            grant.delegation_id, current, analysis_handlers(calls))
        assert result.outcome == 'satisfied' and len(result.action_result_refs) == 3
        assert calls == ['extract', 'routes', 'claims']


def test_writer_submits_then_separate_check_task_validates(db):
    from tests.test_main_quality_loop import prepared
    from tests.test_check_agent import checked
    current = prepared()
    current['session_id'] = 'v2'
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        repo = DelegationRepository(active_runtime())
        writer = AuthorizationPolicy().issue(DelegationRequest(agent='writing', profile='draft_generation', goal='write'), current)
        repo.create(writer)
        def write(w, a):
            w.update(review='Candidate text [p0]', pending_result_validation=True)
        runner = DelegationRunner(ActionExecutor(AgentController()), strategy=lambda *a: pytest.fail('direct used model'))
        result = runner.run(writer.delegation_id, current, {'generate_deliverables': write})
        assert result.outcome == 'satisfied' and current['pending_result_validation']
        check = AuthorizationPolicy().issue(DelegationRequest(agent='check', profile='independent_check', goal='check'), current)
        repo.create(check)
        result = runner.run(check.delegation_id, current, {'validate_result': lambda w, a: checked(w)})
        assert result.outcome == 'satisfied'
        assert current['check_report']['execution_status'] == 'completed' and not current['pending_result_validation']


def test_unsupported_claim_plan_is_not_accepted():
    from app.agent.policy.acceptance import accept_delegation
    current = state()
    current.update(paper_cards=[evidence()], claim_plans=[{'claims': [{'evidence_ids': ['invented:e1']}]}])
    grant = AuthorizationPolicy().issue(DelegationRequest(agent='analysis', profile='evidence_build', goal='analyze'), current)
    result = accept_delegation(grant, current, [{'operation': 'plan_claims', 'status': 'completed'}])
    assert not result.satisfied


def test_plan_cursor_recovers_after_action_commit_before_runtime_ack(db, monkeypatch):
    current, calls, plans = state(), [], []
    current['paper_details'] = [{'paper_id': 'p', 'title': 'Study'}]
    def planner(grant, context, available):
        plans.append('planned')
        return [{'action': name, 'arguments': {}} for name in
                ['extract_paper_cards', 'validate_routes', 'plan_claims']]
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        repo = DelegationRepository(active_runtime())
        grant = AuthorizationPolicy().issue(DelegationRequest(agent='analysis', profile='evidence_build', goal='analyze'), current)
        repo.create(grant)
        executor = ActionExecutor(AgentController())
        original = executor.execute
        def crash(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError('after commit before plan acknowledgement')
        monkeypatch.setattr(executor, 'execute', crash)
        runner = DelegationRunner(executor, strategy=PlanAndActStrategy(planner))
        with pytest.raises(RuntimeError, match='acknowledgement'):
            runner.run(grant.delegation_id, current, analysis_handlers(calls))
        monkeypatch.setattr(executor, 'execute', original)
        result = runner.run(grant.delegation_id, current, analysis_handlers(calls))
        assert result.outcome == 'satisfied' and plans == ['planned']
        assert calls == ['extract', 'routes', 'claims']
