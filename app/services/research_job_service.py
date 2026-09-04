"""可查询、可取消的后台研究任务服务。"""

from __future__ import annotations

import uuid
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from sqlalchemy.orm import Session

from app.agent.graph import AgentCancelledError
from app.core.logger import get_logger
from app.core.config import get_settings
from app.database.db import SessionLocal
from app.database.repositories import ResearchJobRepository, ResearchSessionRepository
from app.schemas.agent_schema import AgentRequest, ResearchRevisionRequest
from app.services.research_conversation_service import ResearchConversationService

logger = get_logger(__name__)
_JOB_SETTINGS = get_settings()
_EXECUTOR = ThreadPoolExecutor(
    max_workers=_JOB_SETTINGS.research_job_max_workers,
    thread_name_prefix="research-job",
)
_CAPACITY = threading.BoundedSemaphore(
    _JOB_SETTINGS.research_job_max_workers + _JOB_SETTINGS.research_job_max_pending
)
_SUBMIT_LOCK = threading.Lock()


class ResearchJobCapacityError(RuntimeError):
    """后台任务池和等待队列均已满。"""


class ResearchSessionBusyError(RuntimeError):
    """同一研究会话已有活动任务。"""


class ResearchJobService:
    """提交后台任务并通过 SQLite 协调取消状态。"""

    def __init__(self, db: Session) -> None:
        self.db = db
        self.repo = ResearchJobRepository(db)

    def submit(self, request: AgentRequest) -> dict[str, Any]:
        session_id = request.session_id or uuid.uuid4().hex
        request.session_id = session_id
        job_id = self._create_and_enqueue(
            session_id,
            request.model_dump(mode="json"),
            operation="run",
        )
        return self.repo.get(job_id) or {"job_id": job_id, "session_id": session_id}

    def submit_revision(self, request: ResearchRevisionRequest) -> dict[str, Any]:
        job_id = self._create_and_enqueue(
            request.session_id,
            request.model_dump(mode="json"),
            operation="revise",
        )
        return self.repo.get(job_id) or {"job_id": job_id, "session_id": request.session_id}

    def _create_and_enqueue(self, session_id: str, payload: dict, operation: str) -> str:
        with _SUBMIT_LOCK:
            active = self.repo.find_active_for_session(session_id)
            if active:
                raise ResearchSessionBusyError(
                    f"会话 {session_id} 已有活动任务 {active['job_id']}"
                )
            if not _CAPACITY.acquire(blocking=False):
                raise ResearchJobCapacityError("后台研究任务队列已满，请稍后重试")
            job_id = uuid.uuid4().hex
            try:
                self.repo.create(job_id, session_id, payload, operation=operation)
                self.db.commit()
            except Exception:
                self.db.rollback()
                _CAPACITY.release()
                raise
            try:
                ResearchJobService._enqueue(job_id, capacity_reserved=True)
            except Exception as exc:
                self.repo.update(
                    job_id,
                    status="failed",
                    current_step="queue_submission_failed",
                    error=str(exc),
                )
                self.db.commit()
                _CAPACITY.release()
                raise
            return job_id

    @staticmethod
    def _enqueue(job_id: str, *, capacity_reserved: bool = False) -> bool:
        if not capacity_reserved and not _CAPACITY.acquire(blocking=False):
            return False
        future = _EXECUTOR.submit(ResearchJobService._run_job_guarded, job_id)
        # 测试替身可能不返回 Future；此时没有后台任务会消费该配额。
        if future is None:
            _CAPACITY.release()
        return True

    @staticmethod
    def _run_job_guarded(job_id: str) -> None:
        try:
            ResearchJobService._run_job(job_id)
        finally:
            _CAPACITY.release()

    def get(self, job_id: str) -> dict[str, Any] | None:
        return self.repo.get(job_id)

    def cancel(self, job_id: str) -> dict[str, Any] | None:
        job = self.repo.get(job_id)
        if not job:
            return None
        if job["status"] in {"completed", "partial", "blocked", "failed", "cancelled", "needs_clarification"}:
            return job
        # CAS 迁移：若 worker 在读取与写入之间恰好落了终态，这里影响行数为
        # 0，条件更新自动放弃覆盖，避免任务永久卡在 cancel_requested。
        self.repo.update_status_if_in(
            job_id,
            {"queued", "running", "cancel_requested"},
            status="cancel_requested",
        )
        self.db.commit()
        return self.repo.get(job_id)

    @staticmethod
    def recover_after_restart() -> dict[str, int]:
        """恢复持久化队列，并明确终止崩溃时正在运行的任务。

        queued 任务从保存的请求重新入队；running 任务不能安全地从任意节点
        续跑，因此标记 failed，避免永久卡在运行中；cancel_requested 直接取消。
        """
        summary = {"requeued": 0, "interrupted": 0, "cancelled": 0}
        requeue_ids: list[str] = []
        with SessionLocal() as db:
            repo = ResearchJobRepository(db)
            jobs = repo.list_by_statuses({"queued", "running", "cancel_requested"})
            for job in jobs:
                if job["status"] == "queued":
                    # 先记录、事务提交后再入队：worker 一启动就会写库，
                    # 在恢复事务提交前启动会与未提交的写锁竞争。
                    requeue_ids.append(job["job_id"])
                elif job["status"] == "cancel_requested":
                    repo.update(
                        job["job_id"],
                        status="cancelled",
                        current_step="cancelled",
                        error="应用重启时完成取消",
                    )
                    summary["cancelled"] += 1
                else:
                    repo.update(
                        job["job_id"],
                        status="failed",
                        current_step="interrupted_by_restart",
                        error="任务因应用重启中断，请重新提交",
                    )
                    summary["interrupted"] += 1
            db.commit()
        for job_id in requeue_ids:
            if ResearchJobService._enqueue(job_id):
                summary["requeued"] += 1
            else:
                # 容量满时任务保持 queued 落库状态，但当前没有消费者；
                # 静默丢弃会让它永远无人认领，必须留下告警线索。
                logger.warning(
                    "recover_after_restart: 任务 %s 因容量已满暂未入队，"
                    "保持 queued 状态等待下次恢复", job_id,
                )
        return summary

    @staticmethod
    def shutdown_executor(wait: bool = False) -> None:
        """应用关闭时排空执行器：取消排队任务，不阻塞进程退出。

        运行中的线程无法被强杀（协作式取消依赖 should_cancel 轮询），
        wait=False 让关闭流程不必等 LLM 调用自然结束；被取消的排队任务
        保持 queued 状态落库，由下次启动的 recover_after_restart 修复。
        """
        _EXECUTOR.shutdown(wait=wait, cancel_futures=True)

    @staticmethod
    def _is_cancel_requested(job_id: str) -> bool:
        with SessionLocal() as db:
            job = ResearchJobRepository(db).get(job_id)
            return bool(job and job["status"] in {"cancel_requested", "cancelled"})

    @staticmethod
    def _run_job(job_id: str) -> None:
        with SessionLocal() as db:
            repo = ResearchJobRepository(db)
            job = repo.get(job_id)
            if not job:
                return
            if job["status"] == "cancel_requested":
                repo.update(job_id, status="cancelled", current_step="cancelled")
                db.commit()
                return
            repo.update(job_id, status="running", current_step="preflight")
            db.commit()

            def progress(step: str, current: int, total: int) -> None:
                repo.update(
                    job_id,
                    current_step=step,
                    progress_current=current,
                    progress_total=total,
                )
                db.commit()

            try:
                service = ResearchConversationService(
                    db,
                    should_cancel=lambda: ResearchJobService._is_cancel_requested(job_id),
                    progress_callback=progress,
                )
                if job.get("operation") == "revise":
                    revision = ResearchRevisionRequest.model_validate(job["request"])
                    result = service.revise(revision)
                else:
                    request = AgentRequest.model_validate(job["request"])
                    result = service.handle(request)
                if ResearchJobService._is_cancel_requested(job_id):
                    raise AgentCancelledError("任务已取消")
                final_status = (
                    "needs_clarification"
                    if result.get("status") == "needs_clarification"
                    else "partial"
                    if result.get("status") == "partial"
                    else "blocked"
                    if result.get("status") == "blocked"
                    else "failed"
                    if result.get("status") == "failed"
                    else "completed"
                )
                latest_job = repo.get(job_id) or job
                # 终态写入同样是 CAS：只允许从 running 迁出。若 cancel() 已
                # 先行写入 cancel_requested，这里让位并落 cancelled 终态，
                # 否则 completed 覆盖 cancel_requested 后再被取消方回写，
                # 任务会永久卡死、会话 409。
                moved = repo.update_status_if_in(
                    job_id,
                    {"running"},
                    status=final_status,
                    result=result,
                    current_step=final_status,
                    progress_current=(
                        int(latest_job.get("progress_total") or 14)
                        if final_status in {"completed", "partial", "blocked"}
                        else 0
                    ),
                )
                if not moved:
                    repo.update_status_if_in(
                        job_id,
                        {"cancel_requested"},
                        status="cancelled",
                        current_step="cancelled",
                        error="任务完成前收到取消请求",
                    )
                db.commit()
            except AgentCancelledError as exc:
                db.rollback()
                session = ResearchSessionRepository(db).get(job["session_id"])
                if session:
                    ResearchSessionRepository(db).save(
                        session_id=job["session_id"],
                        status="cancelled",
                        original_query=session.get("original_query") or "",
                        state=session.get("state") or {},
                        clarification=session.get("clarification"),
                    )
                repo.update_status_if_in(
                    job_id,
                    {"running", "cancel_requested"},
                    status="cancelled",
                    current_step="cancelled",
                    error=str(exc),
                )
                db.commit()
            except Exception as exc:  # 后台线程必须把错误写回任务状态
                db.rollback()
                logger.exception("Research job failed: %s", job_id)
                repo.update_status_if_in(
                    job_id,
                    {"running", "cancel_requested"},
                    status="failed",
                    current_step="failed",
                    error=str(exc),
                )
                db.commit()
