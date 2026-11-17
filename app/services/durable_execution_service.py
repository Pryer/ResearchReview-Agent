"""执行作用域通过 ContextVar 传播，不把数据库句柄放进模型/任务输入。"""
from contextlib import contextmanager
from contextvars import ContextVar
from app.database.runtime_repository import ResearchRuntimeRepository, RuntimeConflict

_RUNTIME = ContextVar("durable_research_runtime", default=None)
_TASK = ContextVar("durable_research_task", default=None)


def active_runtime():
    runtime = _RUNTIME.get()
    return runtime.for_current_thread() if runtime is not None else None


def active_task_id():
    return _TASK.get()


@contextmanager
def task_scope(task_id):
    token = _TASK.set(task_id)
    try:
        yield
    finally:
        _TASK.reset(token)


def standalone_session():
    from app.database.db import init_db, SessionLocal
    init_db()
    return SessionLocal()


@contextmanager
def standalone_execution_scope(state):
    """直接Python入口与后台Job共用持久化内核；已有服务作用域时只复用。"""
    if active_runtime() is not None:
        yield None
        return
    from uuid import uuid4
    from app.services.research_artifact_service import ResearchArtifactService
    from app.services.research_execution_service import _STORE
    from app.services.execution_report_capture import _CAPTURE
    from app.services.execution_report_service import freeze_source
    if state.get('session_id') is None:
        state['session_id'] = 'standalone-' + uuid4().hex
    if not isinstance(state['session_id'], str) or not state['session_id'] or len(state['session_id']) > 128:
        raise ValueError('invalid standalone research session')
    with standalone_session() as db:
        token = _STORE.set(ResearchArtifactService(db))
        try:
            with durable_scope(db, state['session_id'], state, state['agent_execution_budget']) as runtime:
                cap = _CAPTURE.get()
                if cap is not None:
                    cap.session_id = state['session_id']
                    cap.start_version, cap.owner = runtime.version, runtime.owner
                try:
                    yield runtime
                finally:
                    if cap is not None:
                        # WHY: 释放租约/关闭Session前冻结；导出失败不覆盖原始研究异常。
                        try:
                            cap.end_version = runtime.version
                            cap.source = freeze_source(db, cap)
                        except Exception as exc:
                            cap.capture_error_type = type(exc).__name__
        finally:
            _STORE.reset(token)


@contextmanager
def durable_scope(db, session_id, state, ledger, *, new_research=False):
    from app.core.config import get_settings
    runtime = ResearchRuntimeRepository(db, session_id, ttl=get_settings().agent_execution_deadline_seconds)
    runtime.acquire(state, ledger, new_research=new_research)
    token = _RUNTIME.set(runtime)
    try:
        yield runtime
    finally:
        _RUNTIME.reset(token)
        runtime.release(ledger)
