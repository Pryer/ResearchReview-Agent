"""同步持久化接口：共享执行锁，Session 始终在线程内创建与关闭。"""
from contextlib import contextmanager
from types import SimpleNamespace
import time
from sqlalchemy.orm import Session


class PersistenceWriter:
    """短事务返回即 durable ack，不持有队列或常驻工作线程。"""
    METHODS = frozenset({"check", "reserve_attempt", "settle_attempt", "persist_budget",
                         "begin_tool_call", "finish_tool_call", "persist_tool_artifact", "read_tool_artifact",
                         "validate_tool_resources", "record_tool_resources", "section_cache"})

    def __init__(self, runtime):
        self.runtime = runtime
        self.engine = runtime._persistence_engine

    @contextmanager
    def _locked(self):
        from app.agent.execution_budget import active_budget, AgentBudgetExceeded
        current = active_budget()
        deadline = min(current['deadline'], time.monotonic() + 30.0) if current else time.monotonic() + 30.0
        while not self.runtime.lock.acquire(timeout=min(0.1, max(0, deadline - time.monotonic()))):
            if time.monotonic() >= deadline:
                raise AgentBudgetExceeded('persistence acknowledgement deadline exceeded')
        try:
            yield
        finally:
            self.runtime.lock.release()

    def call(self, method, *args, **kwargs):
        if method not in self.METHODS:
            raise ValueError('worker persistence operation is not permitted')
        from app.database.runtime_repository import ResearchRuntimeRepository
        # WHY: 同步短事务沿用预算锁→执行锁顺序；锁内不排队、不等Worker/Provider，
        # 也不回调预算逻辑，避免与预算预留的全局临界区构成反向等待。
        with self._locked():
            # WHY: Worker不接触父Session；新连接仍受相同owner、版本与租约约束。
            with Session(self.engine) as db:
                worker = ResearchRuntimeRepository(db, self.runtime.session_id, ttl=self.runtime.ttl)
                worker.owner = self.runtime.owner
                worker.version = self.runtime.version
                result = getattr(worker, method)(*args, **kwargs)
                if method == 'check':
                    return SimpleNamespace(owner=result.owner, version=result.version,
                        cancelled=result.cancelled, lease_until=result.lease_until)
                return result


class WorkerRuntime:
    """只向并发工具暴露必要账本接口，不暴露ORM Session或状态提交能力。"""
    def __init__(self, writer):
        self._writer = writer

    def __getattr__(self, name):
        if name not in PersistenceWriter.METHODS:
            raise AttributeError(name)
        return lambda *args, **kwargs: self._writer.call(name, *args, **kwargs)
