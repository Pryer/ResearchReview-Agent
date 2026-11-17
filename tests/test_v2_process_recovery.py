"""真实子进程强退：确认Provider请求前已持久预留，finally不运行也能恢复。"""
import subprocess
import sys
from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import Session
import pytest
from app.database.models import ResearchRuntimeModel, ResearchInvocationModel, ResearchDelegationModel, ResearchAttemptModel, ResearchEventModel
from app.database.runtime_repository import ResearchRuntimeRepository
from app.schemas.delegation_schema import ActionInvocation
from app.services.durable_execution_service import durable_scope
from app.agent.execution_budget import budget_scope
from app.agent.runtime.action_executor import ActionExecutor
from app.agent.controller import AgentController


CHILD = r'''
import os,sys
from types import SimpleNamespace as NS
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from app.database.models import Base
from app.database.delegation_repository import DelegationRepository
from app.services.durable_execution_service import durable_scope, active_runtime
from app.agent.execution_budget import budget_scope, budgeted_create
from app.agent.policy.authorization import AuthorizationPolicy
from app.agent.context_builder import context_source_fingerprint
from app.schemas.delegation_schema import DelegationRequest, ActionInvocation
from app.agent.runtime.action_executor import ActionExecutor
from app.agent.controller import AgentController
engine=create_engine('sqlite:///'+sys.argv[1],connect_args={'check_same_thread':False})
Base.metadata.create_all(engine)
state={'session_id':'process','user_query':'test','topic':'test','steps':[],'errors':[],
       'research_plan':{'task_graph':[]},'agent_orchestration_mode':'autonomous'}
with Session(engine) as db, budget_scope(state), durable_scope(db,'process',state,state['agent_execution_budget']):
    repo=DelegationRepository(active_runtime())
    grant=AuthorizationPolicy().issue(DelegationRequest(agent='search',profile='initial_research',goal='test'),state)
    repo.create(grant)
    active_runtime().snapshot(state,ledger=state['agent_execution_budget'])
    executor=ActionExecutor(AgentController())
    def search(w,a):
        w.update(candidate_papers=[{'paper_id':'p','title':'T'}],ranked_papers=[{'paper_id':'p','title':'T'}])
    if sys.argv[2]=='multi':
        first=ActionInvocation(delegation_id=grant.delegation_id,step_index=0,action='search_and_rank',
            expected_version=repo.version(),source_fingerprint=context_source_fingerprint(state))
        executor.execute(first,state,{'search_and_rank':search})
    def crash(w,a):
        client=NS(chat=NS(completions=NS(create=lambda **kwargs:os._exit(17))))
        budgeted_create(client,messages=[],max_tokens=8)
    action='fetch_metadata' if sys.argv[2]=='multi' else 'search_and_rank'
    invocation=ActionInvocation(delegation_id=grant.delegation_id,step_index=int(sys.argv[2]=='multi'),action=action,
        expected_version=repo.version(),source_fingerprint=context_source_fingerprint(state))
    executor.execute(invocation,state,{action:crash})
'''


@pytest.mark.parametrize('mode', ['single', 'multi'])
def test_process_exit_retains_reservation_and_committed_predecessor(tmp_path, mode):
    path = tmp_path / 'crashed.db'
    child = subprocess.run([sys.executable, '-c', CHILD, str(path), mode], capture_output=True, text=True, timeout=30)
    assert child.returncode == 17, child.stderr
    engine = create_engine('sqlite:///' + str(path), connect_args={'check_same_thread': False})
    with Session(engine) as db:
        assert db.scalar(select(ResearchEventModel).where(ResearchEventModel.event_type == 'tool_started')) is not None
        attempt = db.scalar(select(ResearchAttemptModel))
        assert attempt.status == 'running' and attempt.reserved_tokens > 0
        reserved = attempt.reserved_tokens
        invocation_row = db.scalar(select(ResearchInvocationModel).where(ResearchInvocationModel.status == 'running'))
        invocation = ActionInvocation.model_validate_json(invocation_row.invocation_json)
        previous_task = invocation_row.task_id
        # 子进程已确认退出；推进测试租约至过期，避免等待生产TTL，不冒用有效owner。
        db.execute(update(ResearchRuntimeModel).values(lease_until=0))
        db.commit()
        restored = ResearchRuntimeRepository(db, 'process').restore()
        with budget_scope(restored), durable_scope(db, 'process', restored, restored['agent_execution_budget']):
            assert db.scalar(select(ResearchEventModel).where(ResearchEventModel.event_type == 'tool_interrupted')) is not None
            assert restored['agent_execution_budget']['llm_tokens'] == reserved
            assert restored['agent_execution_budget']['tokens_reserved'] == 0
            result = ActionExecutor(AgentController()).execute(invocation, restored, {invocation.action: lambda w, a: None})
            assert result.task_id != previous_task
            assert restored['agent_execution_budget']['llm_tokens'] == reserved
            delegation = db.scalar(select(ResearchDelegationModel))
            assert delegation.next_step == (2 if mode == 'multi' else 1)
            records = db.scalars(select(ResearchInvocationModel)).all()
            assert len(records) == (2 if mode == 'multi' else 1)
    engine.dispose()
