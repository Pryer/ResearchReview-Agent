"""单步/两步 SQLite 故障窗口；持久身份跨Session恢复，已提交动作不重放。"""
import pytest
from sqlalchemy import update
from sqlalchemy.orm import Session
from app.database.models import ResearchRuntimeModel, ResearchInvocationModel
from app.database.runtime_repository import ResearchRuntimeRepository, cancel_runtime
from app.database.delegation_repository import DelegationRepository
from app.agent.controller import AgentController
from app.agent.runtime.action_executor import ActionExecutor
from app.agent.execution_budget import budget_scope, AgentExecutionCancelled, AgentExecutionStale
from app.agent.context_builder import context_source_fingerprint
from app.agent.policy.authorization import AuthorizationPolicy
from app.schemas.delegation_schema import DelegationRequest, ActionInvocation
from app.services.durable_execution_service import durable_scope, active_runtime
from tests.test_v2_delegation_runtime import db, state


WINDOWS = ('before_prepare', 'after_prepare', 'handler_started', 'before_commit',
           'after_commit', 'after_consumption', 'lease_takeover', 'cancel')


class Crash(SystemExit):
    pass


@pytest.mark.parametrize('multi', [False, True], ids=['single', 'multi'])
@pytest.mark.parametrize('window', WINDOWS)
def test_persisted_action_crash_matrix(db, monkeypatch, window, multi):
    current, calls = state(), []
    def search(w, a):
        calls.append('search')
        w.update(candidate_papers=[{'paper_id': 'p', 'title': 'T'}], ranked_papers=[{'paper_id': 'p', 'title': 'T'}])
    def fetch(w, a):
        calls.append('fetch')
        w['paper_details'] = [{'paper_id': 'p', 'title': 'T'}]
    handlers = {'search_and_rank': search, 'fetch_metadata': fetch}
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        repo = DelegationRepository(active_runtime())
        grant = AuthorizationPolicy().issue(DelegationRequest(agent='search', profile='initial_research', goal='test'), current)
        repo.create(grant)
        active_runtime().snapshot(current, ledger=current['agent_execution_budget'])
        executor = ActionExecutor(AgentController())
        if multi:
            first = ActionInvocation(delegation_id=grant.delegation_id, step_index=0, action='search_and_rank',
                expected_version=0, source_fingerprint=context_source_fingerprint(current))
            executor.execute(first, current, handlers)
        target = 'fetch_metadata' if multi else 'search_and_rank'
        invocation = ActionInvocation(delegation_id=grant.delegation_id, step_index=int(multi), action=target,
            expected_version=repo.version(), source_fingerprint=context_source_fingerprint(current))
        try:
            if window == 'before_prepare':
                raise Crash()
            repo.prepare(invocation)
            if window == 'after_prepare':
                raise Crash()
            def fault_handler(w, a):
                if window == 'handler_started':
                    raise Crash()
                handlers[target](w, a)
                if window in {'cancel', 'lease_takeover'}:
                    with Session(db.bind) as other:
                        if window == 'cancel':
                            cancel_runtime(other, 'v2')
                        else:
                            other.execute(update(ResearchRuntimeModel).where(ResearchRuntimeModel.session_id == 'v2')
                                          .values(owner='new-owner', lease_until=9999999999))
                        other.commit()
            if window == 'before_commit':
                original = DelegationRepository.commit
                def fault_commit(self, *args):
                    original(self, *args)
                    raise Crash()
                monkeypatch.setattr(DelegationRepository, 'commit', fault_commit)
            executor.execute(invocation, current, {**handlers, target: fault_handler})
            if window == 'after_consumption':
                repo.consume(grant.delegation_id, int(multi) + 1)
            raise Crash()
        except (Crash, AgentExecutionCancelled, AgentExecutionStale):
            pass
    if window == 'before_commit':
        monkeypatch.setattr(DelegationRepository, 'commit', original)
    if window == 'lease_takeover':
        # 测试模拟接管者随后退出；恢复只能在其租约过期后领取，不能偷取有效租约。
        with Session(db.bind) as owner:
            owner.execute(update(ResearchRuntimeModel).where(ResearchRuntimeModel.session_id == 'v2').values(lease_until=0))
            owner.commit()
    committed = window in {'after_commit', 'after_consumption'}
    with Session(db.bind) as reopened:
        restored = ResearchRuntimeRepository(reopened, 'v2').restore()
        with budget_scope(restored), durable_scope(reopened, 'v2', restored, restored['agent_execution_budget']):
            before = calls.count(target.replace('_and_rank', '').replace('_metadata', ''))
            result = ActionExecutor(AgentController()).execute(invocation, restored, handlers)
            assert result.status.value == 'completed'
            repo = DelegationRepository(active_runtime())
            assert repo.read(grant.delegation_id)[0].next_step == int(multi) + 1
            assert reopened.get(ResearchInvocationModel, invocation.invocation_id).status == 'committed'
            if multi:
                assert calls.count('search') == 1
            if committed:
                assert calls.count('fetch' if multi else 'search') == before


def test_goal_change_between_steps_rejects_original_grant(db):
    current = state()
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        repo = DelegationRepository(active_runtime())
        grant = AuthorizationPolicy().issue(DelegationRequest(agent='search', profile='initial_research', goal='test'), current)
        repo.create(grant)
        first = ActionInvocation(delegation_id=grant.delegation_id, step_index=0, action='search_and_rank',
            expected_version=0, source_fingerprint=context_source_fingerprint(current))
        ActionExecutor(AgentController()).execute(first, current, {'search_and_rank': lambda w, a: w.update(
            candidate_papers=[{'paper_id': 'p'}], ranked_papers=[{'paper_id': 'p'}])})
        current['topic'] = 'user changed the research scope'
        second = ActionInvocation(delegation_id=grant.delegation_id, step_index=1, action='fetch_metadata',
            expected_version=repo.version(), source_fingerprint=context_source_fingerprint(current))
        with pytest.raises(ValueError, match='target changed'):
            ActionExecutor(AgentController()).execute(second, current,
                {'fetch_metadata': lambda *a: pytest.fail('old grant executed new research goal')})
        assert repo.read(grant.delegation_id)[0].next_step == 1
