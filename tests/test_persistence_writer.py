"""真实SQLite并发记账：Session归属、预留确认、故障结算与线程退出。"""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS
from functools import partial
import json
import threading
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from app.database.models import Base, ResearchAttemptModel, ResearchRuntimeModel
from app.services.durable_execution_service import durable_scope, active_runtime
from app.agent.execution_budget import budget_scope, budgeted_create, submit_with_context


@pytest.fixture
def db(tmp_path):
    engine = create_engine('sqlite:///' + str(tmp_path / 'writer.db'), connect_args={'check_same_thread': False})
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        yield db
    engine.dispose()


@pytest.mark.parametrize('workers', [1, 4, 8])
def test_workers_have_own_sessions_and_durable_reservation_before_call(db, monkeypatch, workers):
    state = {'session_id': 's'}
    count = 100
    main_thread = threading.get_ident()
    original_get, original_execute = db.get, db.execute
    def guarded(fn):
        def invoke(*args, **kwargs):
            assert threading.get_ident() == main_thread, 'parent Session used by worker'
            return fn(*args, **kwargs)
        return invoke
    monkeypatch.setattr(db, 'get', guarded(original_get))
    monkeypatch.setattr(db, 'execute', guarded(original_execute))
    def create(**kwargs):
        with Session(db.bind) as observer:
            assert observer.scalar(select(ResearchAttemptModel).where(ResearchAttemptModel.status == 'running')) is not None
        return NS(usage=NS(total_tokens=7, prompt_tokens=4, completion_tokens=3))
    client = NS(chat=NS(completions=NS(create=create)))
    with budget_scope(state), durable_scope(db, 's', state, state['agent_execution_budget']):
        runtime = active_runtime()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [submit_with_context(pool, partial(budgeted_create, client, messages=[], max_tokens=8)) for _ in range(count)]
            for future in futures:
                future.result(timeout=20)
        assert active_runtime() is runtime
        assert state['agent_execution_budget']['llm_tokens'] == count * 7
        assert state['agent_execution_budget']['tokens_reserved'] == 0
        attempts = db.scalars(select(ResearchAttemptModel)).all()
        assert len(attempts) == count and all(a.status == 'completed' for a in attempts)
        persisted = json.loads(db.get(ResearchRuntimeModel, 's', populate_existing=True).budget_json)
        assert persisted['llm_tokens'] == count * 7 and persisted['tokens_reserved'] == 0


def test_worker_provider_failure_retains_estimated_consumption(db):
    state = {'session_id': 's'}
    def fail(**kwargs):
        raise TimeoutError('provider did not return')
    client = NS(chat=NS(completions=NS(create=fail)))
    with budget_scope(state), durable_scope(db, 's', state, state['agent_execution_budget']):
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = submit_with_context(pool, partial(budgeted_create, client, messages=[], max_tokens=8))
            with pytest.raises(TimeoutError):
                future.result(timeout=10)
        attempt = db.scalar(select(ResearchAttemptModel))
        assert attempt.status == 'uncertain'
        assert state['agent_execution_budget']['llm_tokens'] == attempt.reserved_tokens


def test_worker_cannot_access_parent_db_or_commit_research(db):
    state = {'session_id': 's'}
    with budget_scope(state), durable_scope(db, 's', state, state['agent_execution_budget']):
        def work():
            runtime = active_runtime()
            for name in ('db', 'snapshot', 'begin_task', 'release'):
                with pytest.raises(AttributeError):
                    getattr(runtime, name)
            runtime.check()
        with ThreadPoolExecutor(max_workers=1) as pool:
            submit_with_context(pool, work).result(timeout=10)
