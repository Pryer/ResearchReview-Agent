"""短事务执行账本：外部请求不占事务，过期结果不回滚真实消耗。"""
import logging
from contextlib import contextmanager
import json
import threading
import time
from uuid import uuid4
from sqlalchemy import select, update, or_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session as OrmSession
from app.database.models import (
    ResearchRuntimeModel as Runtime, ResearchTaskModel as Task,
    ResearchAttemptModel as Attempt, ResearchCheckpointModel as Checkpoint,
    ResearchEventModel as Event,
)

logger = logging.getLogger(__name__)


class RuntimeConflict(RuntimeError):
    pass


def dumps(value):
    return json.dumps(value, ensure_ascii=False)


class ResearchRuntimeRepository:
    def __init__(self, db, session_id, *, ttl=1800):
        self.db, self.session_id = db, session_id
        self.owner = uuid4().hex
        self.ttl = ttl
        self.version = 0
        self.lock = threading.RLock()
        # WHY: 租约是崩溃探测机制而不是执行时长上限——进程健康但单步很久
        # （hybrid 对上千候选做 embedding/rerank、CNKI 逐页抓取）时必须续约，
        # 否则 30 分钟后所有闸门误判本执行已失效。
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None

    @contextmanager
    def transaction(self):
        # WHY: 同一执行的章节线程共用数据库 Session，所有账本访问串行且及时提交。
        with self.lock:
            try:
                yield self.db
                self.db.commit()
            except BaseException:
                self.db.rollback()
                raise

    def _row(self):
        return self.db.get(Runtime, self.session_id, populate_existing=True)

    def event(self, key, kind, payload):
        self.db.add(Event(session_id=self.session_id, event_key=key,
                          event_type=kind, payload_json=dumps(payload)))

    def acquire(self, initial_state, ledger):
        with self.transaction():
            row = self._row()
            if row is None:
                try:
                    with self.db.begin_nested():
                        self.db.add(Runtime(session_id=self.session_id, budget_json=dumps(ledger)))
                        self.db.flush()
                except IntegrityError:
                    # WHY: 两个进程首次创建同一会话时，唯一键只负责建行，租约 CAS 决定所有权。
                    if self._row() is None:
                        raise
            now = time.time()
            changed = self.db.execute(update(Runtime).where(
                Runtime.session_id == self.session_id,
                or_(Runtime.owner.is_(None), Runtime.lease_until <= now),
            ).values(owner=self.owner, lease_until=now + self.ttl, cancelled=False))
            if changed.rowcount != 1:
                raise RuntimeConflict("research session already has an active execution")
            self.db.flush()
            row = self._row()
            self.version = row.version
            saved = json.loads(row.budget_json)
            # WHY: 请求已离开进程而结果未知时，保留预留费用，禁止假设免费并自动重试。
            for attempt in self.db.scalars(select(Attempt).where(
                Attempt.session_id == self.session_id, Attempt.status == "running"
            )):
                attempt.status = "uncertain"
                saved["tokens_reserved"] = max(0, int(saved.get("tokens_reserved", 0)) - attempt.reserved_tokens)
                saved["llm_tokens"] = int(saved.get("llm_tokens", 0)) + attempt.reserved_tokens
                saved["estimated_tokens"] = int(saved.get("estimated_tokens", 0)) + attempt.reserved_tokens
                saved["usage_estimated"] = True
                self.event("interrupted:" + attempt.attempt_id, "attempt_uncertain", {"attempt_id": attempt.attempt_id})
            self.db.execute(update(Task).where(Task.session_id == self.session_id, Task.status == "running").values(status="interrupted"))
            ledger.update(saved)
            row.budget_json = dumps(ledger)
            self.event("acquire:" + self.owner, "execution_started", {"version": self.version})
        self._start_heartbeat()

    def _start_heartbeat(self) -> None:
        """启动守护线程按 ttl/4 周期续约；只续约自己拥有且未取消的租约。"""
        interval = max(0.05, self.ttl / 4)
        # WHY: 常规续约每 ttl/4 一次（一个 TTL 内 4 次机会）；单次续约遇到
        # 锁竞争/瞬时 IO 错误后按短间隔重试，而不是再等一个完整周期，否则
        # 两次连续瞬时失败就会耗尽整个 TTL 把健康执行误判为崩溃。
        retry_interval = min(interval, max(0.05, self.ttl / 20))

        def run() -> None:
            while not self._heartbeat_stop.wait(interval):
                try:
                    if not self._renew_lease():
                        # CAS 未命中：owner 已被接管（只能在租约过期后发生）
                        # 或会话被请求取消；心跳必须立即停止，绝不能再续。
                        logger.warning(
                            "lease heartbeat stopped: owner lost or cancelled "
                            "(session=%s)", self.session_id)
                        return
                except Exception:  # noqa: BLE001 - 心跳异常不能杀死执行线程
                    logger.warning(
                        "lease heartbeat renewal failed; will retry in %.1fs "
                        "(session=%s)", retry_interval, self.session_id, exc_info=True)
                    if self._heartbeat_stop.wait(retry_interval):
                        return

        thread = threading.Thread(
            target=run,
            name=f"runtime-lease-{self.session_id[:8]}",
            daemon=True,
        )
        self._heartbeat_thread = thread
        thread.start()

    def _renew_lease(self) -> bool:
        """CAS 续约；仅当 owner 仍是自己且未被取消时成功。

        WHY: 必须在独立短连接上执行。``self.db`` 是工作线程的请求级 Session，
        SQLAlchemy Session 不支持跨线程并发共享；心跳若复用它，与主线程频繁
        的 checkpoint/commit（含 SQLite WAL 写锁）交叉时会双双报错，主线程的
        待提交状态还可能被心跳的 commit/rollback 打断。2026-09-27 事故中
        续约因此静默失败，租约到期后健康长任务被误判失效、各节点降级空转。
        """
        with OrmSession(bind=self.db.bind) as heartbeat_db:
            changed = heartbeat_db.execute(update(Runtime).where(
                Runtime.session_id == self.session_id,
                Runtime.owner == self.owner,
                Runtime.cancelled.is_(False),
            ).values(lease_until=time.time() + self.ttl))
            heartbeat_db.commit()
            return changed.rowcount == 1

    def check(self):
        with self.lock:
            row = self._row()
            if row is None or row.owner != self.owner or row.lease_until <= time.time():
                raise RuntimeConflict("execution lease expired or changed")
            if row.cancelled:
                from app.agent.execution_budget import AgentExecutionCancelled
                raise AgentExecutionCancelled("persisted cancellation requested")

    def snapshot(self, state, *, task=None, result=None, ledger=None):
        """提交检查点、任务结果、资料引用与事件；CAS 失败则全部回滚。"""
        from app.services.research_artifact_service import ResearchArtifactService
        with self.transaction():
            changed = self.db.execute(update(Runtime).where(
                Runtime.session_id == self.session_id, Runtime.version == self.version,
                Runtime.owner == self.owner, Runtime.cancelled.is_(False),
                Runtime.lease_until > time.time(),
            ).values(version=Runtime.version + 1))
            if changed.rowcount != 1:
                raise RuntimeConflict("snapshot CAS rejected stale/cancelled execution")
            if task is not None:
                changed = self.db.execute(update(Task).where(
                    Task.task_id == task.task_id, Task.session_id == self.session_id,
                    Task.status == "running", Task.expected_version == self.version,
                ).values(status="completed", result_json=dumps(result.model_dump(mode="json"))))
                if changed.rowcount != 1:
                    raise RuntimeConflict("task is no longer active")
            # 研究快照外置重数据；任务补丁及提示词不进入运行日志。
            private = {k: v for k, v in state.items() if k not in {
                "main_agent_context", "main_context_snapshot", "_runtime_version",
            }}
            artifacts = ResearchArtifactService(self.db)
            stored = artifacts.externalize_state(self.session_id, private)
            artifacts.validate_payload(stored, max_bytes=8 * 1024 * 1024)
            encoded = dumps(stored)
            version = self.version + 1
            row = self._row()
            row.state_json = encoded
            if ledger is not None:
                row.budget_json = dumps(ledger)
            self.db.add(Checkpoint(session_id=self.session_id, version=version,
                                   state_json=encoded, task_id=task.task_id if task else None))
            self.event("snapshot:" + str(version), "state_committed", {"version": version,
                "task_id": task.task_id if task else None})
        self.version = version
        state["_runtime_version"] = version

    def begin_task(self, task, ledger):
        with self.transaction():
            self.check()
            row = self._row()
            if row.version != self.version:
                raise RuntimeConflict("task source version changed")
            existing = self.db.scalar(select(Task).where(Task.session_id == self.session_id,
                Task.idempotency_key == task.idempotency_key))
            if existing is not None:
                # 完成结果通过已提交快照恢复；未决请求必须由新操作明确重做。
                raise RuntimeConflict("persisted task already exists: " + existing.status)
            self.db.add(Task(task_id=task.task_id, session_id=self.session_id,
                idempotency_key=task.idempotency_key, operation=task.operation,
                expected_version=self.version, status="running"))
            row.budget_json = dumps(ledger)
            self.event("task:" + task.task_id, "task_started", {"operation": task.operation})

    def fail_task(self, task_id, status, ledger):
        with self.transaction():
            row = self._row()
            if row.owner != self.owner:
                return
            changed = self.db.execute(update(Task).where(Task.task_id == task_id,
                Task.session_id == self.session_id, Task.status == "running").values(status=status))
            if changed.rowcount != 1:
                return
            row.budget_json = dumps(ledger)
            self.event("terminal:" + task_id, "task_" + status, {"task_id": task_id})

    def reserve_attempt(self, task_id, reservation, ledger):
        attempt_id = uuid4().hex
        with self.transaction():
            self.check()
            self.db.add(Attempt(attempt_id=attempt_id, session_id=self.session_id,
                task_id=task_id, reserved_tokens=reservation, status="running"))
            self._row().budget_json = dumps(ledger)
            self.event("attempt:" + attempt_id, "attempt_started", {"task_id": task_id})
        return attempt_id

    def persist_budget(self, ledger):
        with self.transaction():
            self.check()
            self._row().budget_json = dumps(ledger)

    def update_token_limit(self, *, expected_limit, new_limit):
        """显式调整空闲会话额度，按原账本及版本 CAS；保留累计用量和审计事件。"""
        from app.database.models import ResearchJobModel
        if isinstance(new_limit, bool) or not isinstance(new_limit, int) or new_limit <= 0:
            raise ValueError("token limit must be a positive integer")
        with self.transaction():
            row = self._row()
            if row is None:
                raise ValueError("session has no runtime budget")
            ledger = json.loads(row.budget_json)
            if int(ledger["token_limit"]) != expected_limit:
                raise RuntimeConflict("token limit changed; reload before updating")
            if new_limit < int(ledger.get("llm_tokens") or 0) + int(ledger.get("tokens_reserved") or 0):
                raise ValueError("new limit is below consumed and reserved usage")
            active_jobs = select(ResearchJobModel.job_id).where(
                ResearchJobModel.session_id == self.session_id,
                ResearchJobModel.status.in_(["queued", "running", "cancel_requested"]),
            ).exists()
            active_attempts = select(Attempt.attempt_id).where(
                Attempt.session_id == self.session_id, Attempt.status == "running",
            ).exists()
            updated = {**ledger, "token_limit": new_limit}
            # WHY: 不能与领取任务/真实请求结算竞争，也不能通过改额度隐式接管崩溃租约。
            changed = self.db.execute(update(Runtime).where(
                Runtime.session_id == self.session_id, Runtime.version == row.version,
                Runtime.owner.is_(None), Runtime.budget_json == row.budget_json,
                ~active_jobs, ~active_attempts,
            ).values(budget_json=dumps(updated)))
            if changed.rowcount != 1:
                raise RuntimeConflict("budget is busy or changed; update rejected")
            self.event("budget-limit:" + uuid4().hex, "budget_limit_updated", {
                "old_limit": expected_limit, "new_limit": new_limit,
                "consumed_tokens": int(ledger.get("llm_tokens") or 0),
            })
        return updated

    def authorize_session_save(self, status):
        conditions = [Runtime.session_id == self.session_id, Runtime.owner == self.owner,
                      Runtime.version == self.version, Runtime.lease_until > time.time()]
        if status not in {"cancelled", "failed", "blocked"}:
            conditions.append(Runtime.cancelled.is_(False))
        changed = self.db.execute(update(Runtime).where(*conditions).values(version=Runtime.version))
        if changed.rowcount != 1:
            raise RuntimeConflict("public session save rejected stale/cancelled execution")

    def settle_attempt(self, attempt_id, usage, ledger, *, estimated):
        with self.transaction():
            row = self._row()
            if row.owner != self.owner:
                # 迟到响应不能覆盖新执行的账本；原预留已由恢复逻辑计为 uncertain。
                return
            changed = self.db.execute(update(Attempt).where(Attempt.attempt_id == attempt_id,
                Attempt.status == "running").values(status="uncertain" if estimated else "completed",
                    usage_json=dumps(usage)))
            if changed.rowcount != 1:
                return
            row.budget_json = dumps(ledger)
            self.event("settle:" + attempt_id, "attempt_settled", {"estimated": estimated, "usage": usage})

    def restore(self):
        from app.services.research_artifact_service import ResearchArtifactService
        with self.lock:
            row = self._row()
            state = ResearchArtifactService(self.db).hydrate_state(self.session_id, json.loads(row.state_json))
            state["agent_execution_budget"] = json.loads(row.budget_json)
            state["_runtime_version"] = row.version
            return state

    def release(self, ledger):
        # WHY: 先置停止标志，再在锁内释放；即使有续约正在等锁，它拿到锁后
        # CAS 也会因 owner 已清空而 0 行命中，不会把释放后的租约复活。
        self._heartbeat_stop.set()
        with self.transaction():
            self.db.execute(update(Runtime).where(Runtime.session_id == self.session_id,
                Runtime.owner == self.owner).values(owner=None, lease_until=0, budget_json=dumps(ledger)))
        # WHY: 等心跳线程退出后再返回，避免极端情况下独立连接的续约落在
        # release 之后（两个写事务串行，通常它先在 wait 上被立即唤醒）。
        thread = self._heartbeat_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)


def cancel_runtime(db, session_id):
    """与 job 的取消状态在同一事务写入；提交 CAS 必须检查本字段。"""
    db.execute(update(Runtime).where(Runtime.session_id == session_id,
        Runtime.owner.is_not(None)).values(cancelled=True))
