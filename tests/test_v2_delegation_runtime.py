"""新委派协议通过真实 SQLite/Controller 的独立提交与恢复。"""
import copy
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from app.database.models import Base, ResearchDelegationModel, ResearchInvocationModel
from app.database.delegation_repository import DelegationRepository
from app.database.runtime_repository import RuntimeConflict
from app.services.durable_execution_service import durable_scope, active_runtime
from app.agent.execution_budget import budget_scope
from app.agent.controller import AgentController
from app.agent.runtime.action_executor import ActionExecutor
from app.agent.context_builder import context_source_fingerprint
from app.agent.policy.authorization import AuthorizationPolicy
from app.schemas.delegation_schema import DelegationRequest, ActionInvocation


def state():
    return {"session_id": "v2", "user_query": "test", "topic": "test", "steps": [], "errors": [],
            "research_plan": {"task_graph": []}, "agent_orchestration_mode": "autonomous"}


@pytest.fixture
def db(tmp_path):
    engine = create_engine("sqlite:///" + str(tmp_path / "v2.db"), connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        yield db
    engine.dispose()


def test_independent_action_cursor_observation_and_replay(db):
    current = state()
    calls = []
    handlers = {
        "search_and_rank": lambda w, args: (calls.append('search'), w.update(
            candidate_papers=[{"paper_id": "p", "title": "T"}], ranked_papers=[{"paper_id": "p", "title": "T"}])),
        "fetch_metadata": lambda w, args: (calls.append('fetch'), w.update(paper_details=[{"paper_id": "p", "title": "T"}])),
    }
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        repo = DelegationRepository(active_runtime())
        grant = AuthorizationPolicy().issue(DelegationRequest(agent='search', profile='initial_research', goal='search'), current)
        repo.create(grant)
        executor = ActionExecutor(AgentController())
        first = ActionInvocation(delegation_id=grant.delegation_id, step_index=0,
            action='search_and_rank', expected_version=repo.version(), source_fingerprint=context_source_fingerprint(current))
        result = executor.execute(first, current, handlers)
        assert db.get(ResearchInvocationModel, first.invocation_id).status == 'committed'
        row = db.get(ResearchDelegationModel, grant.delegation_id)
        assert row.next_step == 1 and row.observation_cursor == 0
        before = copy.deepcopy(current)
        assert executor.execute(first, current, handlers).task_id == result.task_id
        assert calls == ['search'] and current == before
        repo.consume(grant.delegation_id, 1)
        assert row.observation_cursor == 1
        second = ActionInvocation(delegation_id=grant.delegation_id, step_index=1,
            action='fetch_metadata', expected_version=repo.version(), source_fingerprint=context_source_fingerprint(current))
        executor.execute(second, current, handlers)
        assert calls == ['search', 'fetch'] and row.next_step == 2
        assert repo.version() == 2


def test_snapshot_and_budget_do_not_increment_business_version(db):
    current = state()
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        runtime = active_runtime()
        repo = DelegationRepository(runtime)
        grant = AuthorizationPolicy().issue(DelegationRequest(agent='check', profile='independent_check', goal='check'), current)
        repo.create(grant)
        runtime.snapshot(current)
        runtime.persist_budget(current['agent_execution_budget'])
        assert repo.version() == 0


def test_scope_and_role_are_server_enforced():
    auth = AuthorizationPolicy()
    with pytest.raises(ValueError, match='role'):
        auth.issue(DelegationRequest(agent='search', profile='section_revision', goal='bad'), state())
    with pytest.raises(ValueError, match='scope'):
        auth.issue(DelegationRequest(agent='search', profile='initial_research', goal='bad', scope={'start_year': 1900}), state())


def test_failed_commit_rolls_back_cursor_result_and_version(db, monkeypatch):
    current = state()
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        repo = DelegationRepository(active_runtime())
        grant = AuthorizationPolicy().issue(DelegationRequest(agent='search', profile='initial_research', goal='search'), current)
        repo.create(grant)
        invocation = ActionInvocation(delegation_id=grant.delegation_id, step_index=0,
            action='search_and_rank', expected_version=0, source_fingerprint=context_source_fingerprint(current))
        original = DelegationRepository.commit
        def crash(self, *args):
            original(self, *args)
            raise RuntimeError('crash before transaction commit')
        monkeypatch.setattr(DelegationRepository, 'commit', crash)
        with pytest.raises(RuntimeError, match='crash'):
            ActionExecutor(AgentController()).execute(invocation, current,
                {'search_and_rank': lambda w, a: w.update(candidate_papers=[{'paper_id': 'p'}])})
        assert repo.version() == 0
        assert db.get(ResearchDelegationModel, grant.delegation_id).next_step == 0
        assert db.get(ResearchInvocationModel, invocation.invocation_id).status != 'committed'
        assert not current.get('candidate_papers')


def test_reopen_recovers_committed_result_without_replaying_handler(db):
    from app.database.runtime_repository import ResearchRuntimeRepository
    current = state()
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        repo = DelegationRepository(active_runtime())
        grant = AuthorizationPolicy().issue(DelegationRequest(agent='search', profile='initial_research', goal='search'), current)
        repo.create(grant)
        invocation = ActionInvocation(delegation_id=grant.delegation_id, step_index=0, action='search_and_rank',
            expected_version=0, source_fingerprint=context_source_fingerprint(current))
        result = ActionExecutor(AgentController()).execute(invocation, current,
            {'search_and_rank': lambda w, a: w.update(candidate_papers=[{'paper_id': 'p'}])})
    with Session(db.bind) as reopened:
        restored = ResearchRuntimeRepository(reopened, 'v2').restore()
        with budget_scope(restored), durable_scope(reopened, 'v2', restored, restored['agent_execution_budget']):
            handlers = {'search_and_rank': lambda *args: pytest.fail('committed action replayed')}
            again = ActionExecutor(AgentController()).execute(invocation, restored, handlers)
            assert again.task_id == result.task_id
            assert DelegationRepository(active_runtime()).read(grant.delegation_id)[0].next_step == 1


def test_failed_attempt_retries_same_invocation_with_new_task(db):
    current = state()
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        repo = DelegationRepository(active_runtime())
        grant = AuthorizationPolicy().issue(DelegationRequest(agent='search', profile='initial_research', goal='search'), current)
        repo.create(grant)
        invocation = ActionInvocation(delegation_id=grant.delegation_id, step_index=0, action='search_and_rank',
            expected_version=0, source_fingerprint=context_source_fingerprint(current))
        executor = ActionExecutor(AgentController())
        def fail(w, a):
            raise ValueError('temporary failure')
        with pytest.raises(ValueError, match='temporary'):
            executor.execute(invocation, current, {'search_and_rank': fail})
        original = db.get(ResearchInvocationModel, invocation.invocation_id).task_id
        result = executor.execute(invocation, current, {'search_and_rank': lambda w, a: None})
        assert result.task_id != original
        assert repo.read(grant.delegation_id)[0].next_step == 1


def test_revoke_blocks_execution_and_replay(db):
    current = state()
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        repo = DelegationRepository(active_runtime())
        grant = AuthorizationPolicy().issue(DelegationRequest(agent='search', profile='initial_research', goal='search'), current)
        repo.create(grant)
        invocation = ActionInvocation(delegation_id=grant.delegation_id, step_index=0, action='search_and_rank',
            expected_version=0, source_fingerprint=context_source_fingerprint(current))
        repo.revoke(grant.delegation_id)
        with pytest.raises(RuntimeConflict, match='active'):
            ActionExecutor(AgentController()).execute(invocation, current, {'search_and_rank': lambda *a: pytest.fail()})


def test_mandatory_action_cannot_be_skipped_and_is_atomically_consumed(db):
    current = state()
    current['agent_mandatory_actions'] = [{'action': 'search_and_rank', 'arguments': {}}]
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        repo = DelegationRepository(active_runtime())
        grant = AuthorizationPolicy().issue(DelegationRequest(agent='search', profile='initial_research', goal='search'), current)
        repo.create(grant)
        invocation = ActionInvocation(delegation_id=grant.delegation_id, step_index=0, action='search_and_rank',
            expected_version=0, source_fingerprint=context_source_fingerprint(current))
        ActionExecutor(AgentController()).execute(invocation, current, {'search_and_rank': lambda w, a: None})
        assert not current.get('agent_mandatory_actions')
        assert repo.read(grant.delegation_id)[0].next_step == 1


@pytest.mark.parametrize('fault', ['cancel', 'revoke', 'business_version', 'grant_revision'])
def test_commit_rechecks_control_and_authority(db, fault):
    from sqlalchemy import update
    from app.database.models import ResearchWorkVersionModel
    from app.agent.execution_budget import AgentExecutionCancelled
    current = state()
    cancelled = [False]
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        repo = DelegationRepository(active_runtime())
        grant = AuthorizationPolicy().issue(DelegationRequest(agent='search', profile='initial_research', goal='search'), current)
        repo.create(grant)
        invocation = ActionInvocation(delegation_id=grant.delegation_id, step_index=0, action='search_and_rank',
            expected_version=0, source_fingerprint=context_source_fingerprint(current))
        def handler(working, arguments):
            working['candidate_papers'] = [{'paper_id': 'must-not-commit'}]
            if fault == 'cancel':
                cancelled[0] = True
            else:
                with Session(db.bind) as other:
                    if fault == 'revoke':
                        other.execute(update(ResearchDelegationModel).where(
                            ResearchDelegationModel.delegation_id == grant.delegation_id).values(status='cancelled'))
                    elif fault == 'grant_revision':
                        updated = grant.model_copy(update={'revision': 2})
                        other.execute(update(ResearchDelegationModel).where(
                            ResearchDelegationModel.delegation_id == grant.delegation_id).values(grant_json=updated.model_dump_json()))
                    else:
                        other.execute(update(ResearchWorkVersionModel).where(
                            ResearchWorkVersionModel.session_id == 'v2').values(version=1))
                    other.commit()
        with pytest.raises((RuntimeConflict, AgentExecutionCancelled)):
            ActionExecutor(AgentController()).execute(invocation, current, {'search_and_rank': handler},
                should_cancel=lambda: cancelled[0])
        assert not current.get('candidate_papers')
        assert repo.read(grant.delegation_id)[0].next_step == 0
        assert db.get(ResearchInvocationModel, invocation.invocation_id).status != 'committed'


def test_legacy_single_keeps_exact_action_and_arguments():
    from app.agent.policy.direct_arguments import direct_arguments
    auth = AuthorizationPolicy()
    original = {'section_ids': ['intro']}
    grant = auth.migrate_single(state(), role='writing', operation='rewrite_sections', arguments=original, goal='revise')
    original['section_ids'].append('forbidden')
    action, arguments = direct_arguments(grant)
    assert action == 'rewrite_sections' and arguments == {'section_ids': ['intro']}
    assert grant.allowed_actions == ('rewrite_sections',) and grant.max_steps == 1
    with pytest.raises(ValueError, match='role'):
        auth.migrate_single(state(), role='search', operation='rewrite_sections', arguments={}, goal='bad')


def test_direct_revision_builder_uses_only_granted_sections():
    from app.agent.policy.direct_arguments import direct_arguments
    grant = AuthorizationPolicy().issue(DelegationRequest(agent='writing', profile='section_revision',
        goal='revise', scope={'section_ids': ['intro', 'related_work']}),
        {**state(), 'writing_plans': [{'sections': [{'id': 'intro'}, {'id': 'related_work'}]}]})
    assert direct_arguments(grant) == ('rewrite_sections', {'section_ids': ['intro', 'related_work']})
    with pytest.raises(ValueError, match='requires explicit scope'):
        AuthorizationPolicy().issue(DelegationRequest(agent='writing', profile='section_revision', goal='bad'), state())


def test_committed_result_can_be_read_after_next_mandatory_action_changes(db):
    current = state()
    current['agent_mandatory_actions'] = [{'action': 'search_and_rank', 'arguments': {}},
                                          {'action': 'fetch_metadata', 'arguments': {}}]
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        repo = DelegationRepository(active_runtime())
        grant = AuthorizationPolicy().issue(DelegationRequest(agent='search', profile='initial_research', goal='search'), current)
        repo.create(grant)
        invocation = ActionInvocation(delegation_id=grant.delegation_id, step_index=0, action='search_and_rank',
            expected_version=0, source_fingerprint=context_source_fingerprint(current))
        executor = ActionExecutor(AgentController())
        result = executor.execute(invocation, current, {'search_and_rank': lambda w, a: None})
        again = executor.execute(invocation, current, {'search_and_rank': lambda *a: pytest.fail('replay')})
        assert again.task_id == result.task_id
        assert current['agent_mandatory_actions'][0]['action'] == 'fetch_metadata'


@pytest.mark.parametrize('verdict', ['deny', 'escalate'])
def test_v2_sidecar_cannot_execute_denied_action(db, monkeypatch, verdict):
    from types import SimpleNamespace
    from app.core.config import get_settings
    monkeypatch.setattr(get_settings(), 'agent_sidecar_enabled', True)
    monkeypatch.setattr(get_settings(), 'agent_sidecar_review_actions', 'search_and_rank')
    current = state()
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        repo = DelegationRepository(active_runtime())
        grant = AuthorizationPolicy().issue(DelegationRequest(agent='search', profile='initial_research', goal='search'), current)
        repo.create(grant)
        invocation = ActionInvocation(delegation_id=grant.delegation_id, step_index=0, action='search_and_rank',
            expected_version=0, source_fingerprint=context_source_fingerprint(current))
        review = SimpleNamespace(review=lambda **kw: SimpleNamespace(verdict=verdict, reason='review result'))
        with pytest.raises(ValueError, match='Main intervention'):
            ActionExecutor(AgentController(), sidecar=review).execute(invocation, current,
                {'search_and_rank': lambda *a: pytest.fail('denied action executed')})
        assert repo.read(grant.delegation_id)[0].next_step == 0
