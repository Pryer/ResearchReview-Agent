"""执行作用域通过 ContextVar 传播，不把数据库句柄放进模型/任务输入。"""
from contextlib import contextmanager
from contextvars import ContextVar
from app.database.runtime_repository import ResearchRuntimeRepository, RuntimeConflict

_RUNTIME = ContextVar("durable_research_runtime", default=None)
_TASK = ContextVar("durable_research_task", default=None)


def active_runtime():
    return _RUNTIME.get()


def active_task_id():
    return _TASK.get()


@contextmanager
def task_scope(task_id):
    token = _TASK.set(task_id)
    try:
        yield
    finally:
        _TASK.reset(token)


@contextmanager
def durable_scope(db, session_id, state, ledger):
    from app.core.config import get_settings
    runtime = ResearchRuntimeRepository(db, session_id, ttl=get_settings().agent_execution_deadline_seconds)
    runtime.acquire(state, ledger)
    token = _RUNTIME.set(runtime)
    try:
        yield runtime
    finally:
        _RUNTIME.reset(token)
        runtime.release(ledger)
