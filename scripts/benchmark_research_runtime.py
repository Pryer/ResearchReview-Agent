"""独立 SQLite 文件上的真实 Job/Runtime 合成负载验收，不调用 LLM 或学术来源。

固定生产线程池为 2，提交 1/2/4/8 个任务，另测高频研究检查点和长读事务。
例：python scripts/benchmark_research_runtime.py --output-dir data/runtime_eval/20261020
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def distribution(values):
    if not values:
        return {"count": 0, "mean_ms": None, "max_ms": None}
    return {"count": len(values), "mean_ms": round(statistics.mean(values) * 1000, 3),
            "max_ms": round(max(values) * 1000, 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    if not output.is_relative_to(ROOT / "data") or args.repeats < 1:
        parser.error("use a local data directory and positive repeat count")
    output.mkdir(parents=True, exist_ok=False)
    database = output / "runtime.db"
    os.environ.update(DATABASE_URL="sqlite:///" + database.as_posix(),
                     APP_LOG_DIR=str(output / "logs"), RESEARCH_JOB_MAX_WORKERS="2",
                     RESEARCH_JOB_MAX_PENDING="20", CNKI_ENABLED="false")
    from sqlalchemy import event, select
    from app.database.db import engine, SessionLocal
    from app.database.models import Base, ResearchJobModel, ResearchRuntimeModel
    from app.database.runtime_repository import ResearchRuntimeRepository
    from app.schemas.agent_schema import AgentRequest
    from app.services.research_conversation_service import ResearchConversationService
    from app.services.research_job_service import ResearchJobService
    Base.metadata.create_all(engine)
    with engine.connect() as con:
        pragmas = {name: con.exec_driver_sql("PRAGMA " + name).scalar()
                   for name in ("journal_mode", "busy_timeout", "wal_autocheckpoint", "synchronous")}
    lock = threading.Lock()
    state = {"active": 0, "started": {}, "finished": set(), "snapshots": [],
             "heartbeat": [], "sql": [], "errors": [], "transactions": []}
    original_guard = ResearchJobService._run_job_guarded
    original_snapshot = ResearchRuntimeRepository.snapshot
    original_renew = ResearchRuntimeRepository._renew_lease
    def guarded(job_id):
        with lock:
            state["active"] += 1
            state["started"][job_id] = time.perf_counter()
        try:
            return original_guard(job_id)
        finally:
            with lock:
                state["active"] -= 1
                state["finished"].add(job_id)
    def snapshot(self, current, **kw):
        started = time.perf_counter()
        try:
            return original_snapshot(self, current, **kw)
        finally:
            with lock:
                state["snapshots"].append(time.perf_counter() - started)
    def renew(self):
        started = time.perf_counter()
        result = False
        try:
            result = original_renew(self)
            return result
        finally:
            with lock:
                state["heartbeat"].append({"seconds": time.perf_counter() - started, "success": result,
                                          "stop_requested": self._heartbeat_stop.is_set()})
    def before_cursor(conn, cursor, statement, params, context, executemany):
        context.benchmark_started = time.perf_counter()
    def after_cursor(conn, cursor, statement, params, context, executemany):
        if statement.lstrip().upper().startswith(("UPDATE", "INSERT", "DELETE")):
            with lock:
                state["sql"].append(time.perf_counter() - context.benchmark_started)
    def on_error(context):
        with lock:
            state["errors"].append(type(context.original_exception).__name__)
    def transaction_started(session, transaction, connection):
        session.info["benchmark_started"] = time.perf_counter()
    def transaction_finished(session, transaction):
        if transaction.parent is None:
            started = session.info.pop("benchmark_started", None)
            if started is not None:
                with lock:
                    state["transactions"].append(time.perf_counter() - started)
    from sqlalchemy.orm import Session
    event.listen(engine, "before_cursor_execute", before_cursor)
    event.listen(engine, "after_cursor_execute", after_cursor)
    event.listen(engine, "handle_error", on_error)
    event.listen(Session, "after_begin", transaction_started)
    event.listen(Session, "after_transaction_end", transaction_finished)
    ResearchJobService._run_job_guarded = staticmethod(guarded)
    ResearchRuntimeRepository.snapshot = snapshot
    ResearchRuntimeRepository._renew_lease = renew
    workload = {"steps": 10}
    payload = "x" * 16384
    def handle(self, request):
        # WHY: 只替换外部研究工作负载，队列、领取、SQLite、心跳与 Snapshot 仍走生产实现。
        current = {"session_id": request.session_id, "benchmark_payload": payload}
        runtime = ResearchRuntimeRepository(self.db, request.session_id, ttl=0.8)
        runtime.acquire(current, {})
        try:
            for index in range(workload["steps"]):
                time.sleep(0.8 / workload["steps"])
                current["iteration"] = index
                runtime.snapshot(current)
        finally:
            runtime.release({})
        return {"status": "completed", "answer": "synthetic runtime workload"}
    ResearchConversationService.handle = handle
    reports = []
    scenarios = [("A", 1, 10, False), ("B", 2, 10, False), ("C", 4, 10, False),
                 ("D", 8, 10, False), ("E", 2, 40, False), ("F", 2, 10, True)]
    for repeat in range(args.repeats):
        for name, jobs, steps, long_reader in scenarios:
            workload["steps"] = steps
            offsets = {key: len(state[key]) for key in ("snapshots", "heartbeat", "sql", "errors", "transactions")}
            reader = None
            if long_reader:
                reader = engine.connect()
                reader.exec_driver_sql("BEGIN")
                reader.exec_driver_sql("SELECT * FROM research_runtime").fetchall()
            submitted = {}
            session_ids = []
            started = time.perf_counter()
            for index in range(jobs):
                submitted_at = time.perf_counter()
                session_id = f"bench-{repeat}-{name}-{index}"
                session_ids.append(session_id)
                with SessionLocal() as db:
                    job = ResearchJobService(db).submit(AgentRequest(
                        user_query="synthetic runtime workload",
                        session_id=session_id))
                submitted[job["job_id"]] = submitted_at
            timeline = []
            while True:
                with lock:
                    active = state["active"]
                    queued = sum(key not in state["started"] for key in submitted)
                    finished = all(key in state["finished"] for key in submitted)
                timeline.append({"seconds": round(time.perf_counter() - started, 4),
                    "active_workers": active, "queued_jobs": queued,
                    "wal_bytes": Path(str(database) + "-wal").stat().st_size if Path(str(database) + "-wal").exists() else 0})
                if finished:
                    break
                if time.perf_counter() - started > 45:
                    raise RuntimeError("benchmark deadline exceeded")
                time.sleep(0.05)
            wal_checkpoint = None
            if reader is not None:
                with engine.connect() as con:
                    before = list(con.exec_driver_sql("PRAGMA wal_checkpoint(PASSIVE)").one())
                reader.rollback()
                reader.close()
                with engine.connect() as con:
                    after = list(con.exec_driver_sql("PRAGMA wal_checkpoint(PASSIVE)").one())
                wal_checkpoint = {"reader_active": before, "reader_released": after}
            with SessionLocal() as db:
                statuses = Counter(db.scalars(select(ResearchJobModel.status).where(
                    ResearchJobModel.job_id.in_(submitted))))
                state_sizes = [len(value.encode("utf-8")) for value in db.scalars(
                    select(ResearchRuntimeModel.state_json).where(ResearchRuntimeModel.session_id.in_(session_ids)))]
            heartbeats = state["heartbeat"][offsets["heartbeat"]:]
            reports.append({"scenario": name, "repeat": repeat + 1, "submitted_jobs": jobs,
                "max_workers": 2, "checkpoints_per_job": steps, "long_reader": long_reader,
                "elapsed_seconds": round(time.perf_counter() - started, 3), "statuses": dict(statuses),
                "submit_to_worker": distribution([state["started"][key] - value for key, value in submitted.items()]),
                "transaction": distribution(state["transactions"][offsets["transactions"]:]),
                "snapshot": distribution(state["snapshots"][offsets["snapshots"]:]),
                "write_statement": distribution(state["sql"][offsets["sql"]:]),
                "heartbeat": distribution([row["seconds"] for row in heartbeats]),
                "heartbeat_failures": sum(not row["success"] and not row["stop_requested"] for row in heartbeats),
                "heartbeat_stopped_during_release": sum(not row["success"] and row["stop_requested"] for row in heartbeats),
                "database_errors": state["errors"][offsets["errors"]:],
                "wal_checkpoint": wal_checkpoint, "timeline": timeline,
                "persisted_state_bytes": state_sizes})
    ResearchJobService.shutdown_executor(wait=True)
    report = {"at": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(), "pragmas": pragmas,
        "commit_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        "tracked_diff_sha256": sha256(subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT)).hexdigest(),
        "runner_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "workload": "synthetic I/O; real production Job service, thread pool, SQLite and Runtime repository",
        "limitations": ["single process, two workers", "not LLM throughput or production tail latency",
                         "16 KiB synthetic state, each task 0.8 seconds I/O plus persistence"], "runs": reports}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    engine.dispose()
    print(json.dumps({"runs": len(reports), "statuses": dict(sum((Counter(item["statuses"]) for item in reports), Counter()))}))


if __name__ == "__main__":
    main()
