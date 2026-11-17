"""v1活动委派通过同一CAS快照迁移，已提交动作不重放。"""
import copy
import pytest
from sqlalchemy import select

from app.agent.context_builder import context_source_fingerprint
from app.agent.controller import AgentController
from app.agent.execution_budget import budget_scope
from app.agent.policy.authorization import goal_fingerprint
from app.agent.react_context import ReActContext
from app.agent.task_main import TaskMainLoop
from app.core.config import get_settings
from app.database.delegation_repository import DelegationRepository
from app.database.models import ResearchDelegationModel
from app.database.runtime_repository import RuntimeConflict
from app.schemas.agent_decision_schema import AgentDecision
from app.schemas.agent_task_schema import AgentRole
from app.services.durable_execution_service import durable_scope, active_runtime
from app.services.legacy_checkpoint_migration import migrate_active
from app.services.research_artifact_service import ResearchArtifactService
from tests.test_v2_delegation_runtime import db, state
from tests.test_subagent_react import ScriptedLLM
from tests.test_tool_execution import events


def legacy(current, role='search', operation='search_and_rank', rounds=2):
    current['active_subagent'] = role
    current['subagent_sessions'] = {role: {'active': {
        'id': 'legacy-delegation', 'operation': operation, 'objective': 'original task', 'arguments': {},
        'source_fingerprint': context_source_fingerprint(current), 'goal_fingerprint': goal_fingerprint(current),
        'rounds': rounds, 'completed_operations': [], 'no_progress': 0,
        'progress': 'before', 'recent_progress': [], 'consume_mandatory_action': False}}}
    ReActContext(current).proposed(AgentDecision(action=operation))
    ReActContext(current, namespace=role).proposed(AgentDecision(action=operation))
    return current


def test_uncommitted_legacy_action_restores_exact_single_and_costs(db):
    current, calls = legacy(state()), []
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        current['agent_execution_budget']['llm_tokens'] = 17
        current['agent_execution_budget']['decision_rounds'] = 3
        def search(w, args):
            calls.append(args)
            w['ranked_papers'] = [{'paper_id': 'p', 'title': 'Study'}]
        llm = ScriptedLLM({'main_agent_decision': [('request_finish', {})]})
        result = TaskMainLoop(llm).run(current, handlers={'search_and_rank': search}, finish_validator=lambda s: bool(s.get('ranked_papers')))
        assert result == 'completed' and calls == [{}]
        assert current['agent_execution_budget']['llm_tokens'] == 17
        assert current['agent_execution_budget']['decision_rounds'] == 4
        assert current['subagent_sessions']['search']['last_delegation']['rounds'] == 2
        assert not current.get('active_subagent') and not current['main_agent_trajectory'].get('pending')


def test_committed_legacy_action_is_imported_without_reexecution(db):
    current = legacy(state())
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        AgentController().execute(role=AgentRole.SEARCH, operation='search_and_rank', objective='original task', state=current,
            handler=lambda w: w.update(ranked_papers=[{'paper_id': 'p', 'title': 'Study'}]))
        llm = ScriptedLLM({'main_agent_decision': [('request_finish', {})]})
        result = TaskMainLoop(llm).run(current, handlers={'search_and_rank': lambda *a: pytest.fail('committed action replayed')},
                                      finish_validator=lambda s: bool(s.get('ranked_papers')))
        assert result == 'completed'
        assert events(db, 'legacy_checkpoint_migrated')[0]['imported_committed_task']
        assert len(events(db, 'task_started')) == 1


def test_old_analysis_verification_moves_to_check_and_revalidates(db):
    from tests.test_main_quality_loop import prepared
    from tests.test_check_agent import checked
    current = prepared()
    current.update(session_id='v2', review='Original candidate', check_report={'verdict': 'passed'})
    legacy(current, role='analysis', operation='validate_result')
    current['subagent_sessions']['analysis']['active']['completed_operations'] = ['validate_result']
    calls = []
    def checker(w, args):
        calls.append(w['active_agent_role'])
        assert w['review'] == 'Original candidate' and not w.get('check_report')
        checked(w)
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        result = TaskMainLoop(ScriptedLLM({'main_agent_decision': [('request_finish', {})]})).run(
            current, handlers={'validate_result': checker}, finish_validator=lambda s: not s.get('pending_result_validation'))
        assert result == 'completed' and calls == ['check']


@pytest.mark.parametrize('problem', ['goal', 'role', 'fake_completed'])
def test_unknown_or_unproven_history_fails_without_mutating_checkpoint(db, problem):
    current = legacy(state())
    old = current['subagent_sessions']['search']['active']
    if problem == 'goal': current['required_reference_count'] = 8
    elif problem == 'role': current['active_subagent'] = 'unknown'
    else: old.update(completed_operations=['search_and_rank'], last_result={'task_id': 'invented'})
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        before = copy.deepcopy(current)
        with pytest.raises(RuntimeConflict): migrate_active(current)
        before['agent_execution_budget']['elapsed_seconds'] = current['agent_execution_budget']['elapsed_seconds']
        assert current == before
        assert db.scalar(select(ResearchDelegationModel)) is None


def test_migration_failure_rolls_back_grant_and_marker_atomically(db, monkeypatch):
    current = legacy(state())
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        original = ResearchArtifactService.validate_payload
        def fail(payload, *, max_bytes=262144):
            if max_bytes == 8 * 1024 * 1024: raise ValueError('migration snapshot failure')
            return original(payload, max_bytes=max_bytes)
        monkeypatch.setattr(ResearchArtifactService, 'validate_payload', staticmethod(fail))
        with pytest.raises(ValueError): migrate_active(current)
        assert current['active_subagent'] == 'search' and not current.get('active_delegation_id')
        assert db.scalar(select(ResearchDelegationModel)) is None
        monkeypatch.setattr(ResearchArtifactService, 'validate_payload', staticmethod(original))
        assert migrate_active(current)


def test_exhausted_old_rounds_do_not_gain_new_execution_allowance(db, monkeypatch):
    monkeypatch.setattr(get_settings(), 'agent_subagent_max_rounds', 2)
    current = legacy(state(), rounds=2)
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        assert migrate_active(current)
        row, grant = DelegationRepository(active_runtime()).read(current['active_delegation_id'])
        from app.services.research_delegation_service import run_delegation
        result = run_delegation(current, {'agent': 'search', 'profile': 'legacy_single_v1', 'goal': grant.goal}, llm=object(),
            forced_operation='search_and_rank', forced_arguments={}, handlers={'search_and_rank': lambda *a: pytest.fail('budget reset')})
        assert result.outcome == 'needs_help' and not result.action_result_refs


def test_resume_prefers_runtime_checkpoint_and_explicit_architecture_config(db, monkeypatch):
    from app.database.repositories import ResearchSessionRepository
    from app.services.research_conversation_service import ResearchConversationService
    from app.schemas.agent_schema import AgentRequest
    current = legacy(state())
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        current['effective_config'] = {'revision': 2, 'values': {'agent_architecture_version': 'v2'}}
        active_runtime().snapshot(current, ledger=current['agent_execution_budget'])
        older = copy.deepcopy(current)
        older['effective_config'] = {'revision': 1, 'values': {'agent_architecture_version': 'v1'}}
        ResearchSessionRepository(db).save('v2', 'failed', 'original', {
            'runtime_checkpoint_version': active_runtime().version, 'editable_research_state': older})
        db.commit()
        seen = {}
        def run(editable, **kwargs):
            seen.update(editable)
            return {'status': 'blocked', 'research_state': editable}
        monkeypatch.setattr('app.agent.graph._run_autonomous_pipeline', run)
        service = ResearchConversationService(db, llm=None)
        monkeypatch.setattr(service, '_persist_or_pause_result', lambda *args: args[-1])
        service._resume_checkpoint(AgentRequest(session_id='v2', user_query='继续', resume_from_checkpoint=True))
        assert seen['effective_config']['revision'] == 2
        assert seen['effective_config']['values']['agent_architecture_version'] == 'v2'
        assert seen['subagent_sessions']['search']['active']['id'] == 'legacy-delegation'


def test_committed_history_does_not_silently_consume_ambiguous_mandatory_queue(db):
    current = legacy(state())
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        AgentController().execute(role=AgentRole.SEARCH, operation='search_and_rank', objective='original task', state=current,
            handler=lambda w: w.update(ranked_papers=[{'paper_id': 'p', 'title': 'Study'}]))
        current['subagent_sessions']['search']['active']['consume_mandatory_action'] = True
        current['agent_mandatory_actions'] = [{'action': 'search_and_rank', 'arguments': {}}]
        with pytest.raises(RuntimeConflict, match='mandatory queue'):
            migrate_active(current)
        assert current['agent_mandatory_actions'] == [{'action': 'search_and_rank', 'arguments': {}}]
        assert db.scalar(select(ResearchDelegationModel)) is None


def test_migration_does_not_overwrite_a_concurrent_business_version(db, monkeypatch):
    from app.database.models import ResearchWorkVersionModel
    from sqlalchemy import update
    from sqlalchemy.orm import Session
    current = legacy(state())
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        cached = ResearchWorkVersionModel(session_id='v2', version=0)
        db.add(cached)
        db.commit()
        assert cached.version == 0
        runtime = active_runtime()
        original = runtime.snapshot
        def conflicting(candidate, **kwargs):
            with Session(db.bind) as concurrent:
                concurrent.execute(update(ResearchWorkVersionModel).where(
                    ResearchWorkVersionModel.session_id == 'v2').values(version=1))
                concurrent.commit()
            assert cached.version == 0
            return original(candidate, **kwargs)
        monkeypatch.setattr(runtime, 'snapshot', conflicting)
        with pytest.raises(RuntimeConflict, match='business version changed'):
            migrate_active(current)
        assert db.get(ResearchWorkVersionModel, 'v2', populate_existing=True).version == 1
        assert current.get('active_subagent') == 'search' and not current.get('active_delegation_id')
        assert db.scalar(select(ResearchDelegationModel)) is None
