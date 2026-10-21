"""在独立数据库运行一个真实研究任务；不自动回答澄清，不修改证据门禁。

例：python scripts/run_real_e2e.py --task T01 --output-dir data/real_e2e/20261020
使用当前 .env 的模型凭据。产物含研究数据，仅写本地 data/；报告不包含凭据或提示词。
"""
from __future__ import annotations

import argparse
from datetime import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def now():
    return datetime.now(ZoneInfo("Asia/Singapore")).isoformat(timespec="seconds")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="examples/real_e2e_tasks.jsonl")
    parser.add_argument("--task", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--deadline-seconds", type=int, default=480)
    parser.add_argument("--token-budget", type=int, default=180000)
    parser.add_argument("--sources", default="arxiv,semantic_scholar,openalex,crossref")
    args = parser.parse_args()
    if args.deadline_seconds <= 0 or args.token_budget <= 0:
        parser.error("deadline and token budget must be positive")
    if not args.sources.split(",") or not set(args.sources.split(",")) <= {"arxiv", "semantic_scholar", "openalex", "crossref"}:
        parser.error("only registered open sources are allowed")
    tasks = [json.loads(line) for line in Path(args.tasks).read_text(encoding="utf-8").splitlines() if line.strip()]
    task = next((item for item in tasks if item["task_id"] == args.task), None)
    if task is None:
        parser.error("unknown task")
    run_id = args.task + "_" + datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y%m%d_%H%M%S_%f")
    output = Path(args.output_dir).resolve() / run_id
    # WHY: 实验数据库和原始记录只允许落在被 Git 忽略的本地数据目录。
    if not output.is_relative_to(ROOT / "data"):
        parser.error("output must be under the repository data directory")
    output.mkdir(parents=True, exist_ok=False)
    os.environ.update({
        "DATABASE_URL": "sqlite:///" + (output / "research.db").as_posix(),
        "APP_LOG_DIR": str(output / "logs"),
        "CNKI_ENABLED": "false",
        "DEFAULT_SEARCH_SOURCES": args.sources,
        "RETRIEVAL_RANKING_MODE": "rules",
        "AGENT_MAIN_TOKEN_BUDGET": str(args.token_budget),
        "AGENT_EXECUTION_DEADLINE_SECONDS": str(args.deadline_seconds),
        "LLM_REQUEST_TIMEOUT": "45",
        "LLM_CONTROL_PLANE_TIMEOUT": "45",
        "LLM_FAILOVER_TOTAL_TIMEOUT": "75",
    })
    # 应用模块必须在隔离环境设置后加载，不能连接日常研究库。
    from sqlalchemy import func, select
    from app.core.config import get_settings
    from app.core.metrics import get_metrics_collector
    from app.database.db import engine, SessionLocal
    from app.database.models import Base, ResearchRuntimeModel, ResearchTaskModel, ResearchCheckpointModel
    from app.schemas.agent_schema import AgentRequest
    from app.services.llm_service import LLMService
    from app.services.research_conversation_service import ResearchConversationService

    settings = get_settings()
    Base.metadata.create_all(engine)
    metrics = get_metrics_collector()
    metrics.reset()
    responses = []
    original_record = LLMService._record_usage
    def record_usage(self, usage, **kw):
        tokens = original_record(self, usage, **kw)
        responses.append({"provider": kw["provider"], "model": kw["model"],
                          "operation": kw["operation"], "usage_reported": usage is not None,
                          "usage": tokens})
        return tokens
    LLMService._record_usage = record_usage
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    diff = subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT)
    manifest = {
        "run_id": run_id, "task_id": args.task, "started_at": now(),
        "commit_sha": head, "tracked_diff_sha256": sha256(diff).hexdigest(),
        "runner_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "tasks_sha256": sha256(Path(args.tasks).read_bytes()).hexdigest(),
        "task": task, "configuration": {
            "sources": settings.search_sources_list, "cnki_enabled": settings.cnki_enabled,
            "ranking_mode": settings.retrieval_ranking_mode,
            "token_limit": settings.agent_main_token_budget,
            "deadline_seconds": settings.agent_execution_deadline_seconds,
            "primary": {"provider": settings.llm_provider, "model": settings.llm_model},
            "backup": {"enabled": settings.llm_backup_enabled, "provider": settings.llm_backup_provider,
                       "model": settings.llm_backup_model},
            "request_timeout": settings.llm_request_timeout,
            "failover_total_timeout": settings.llm_failover_total_timeout,
        },
        "manual_review": "pending_human_review",
        "task_status": "running", "requirement_satisfied": None,
    }
    write_json(output / "run.json", manifest)
    started = time.monotonic()
    def progress(step, current, total):
        with (output / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"at": now(), "step": step, "current": current, "total": total}, ensure_ascii=False) + "\n")
    result = {}
    try:
        with SessionLocal() as db:
            service = ResearchConversationService(db, progress_callback=progress)
            result = service.handle(AgentRequest(session_id=run_id, **task["request"]))
            manifest["task_status"] = result.get("status") or "unknown"
    except Exception as exc:
        # 技术异常留在本地应用日志；导出报告只保留类型，不泄露服务地址或响应体。
        manifest.update(task_status="failed", error_type=type(exc).__name__)
        from app.core.logger import get_logger
        get_logger(__name__).exception("Real E2E task stopped with an exception")
    finally:
        LLMService._record_usage = original_record
        with SessionLocal() as db:
            runtime = db.get(ResearchRuntimeModel, run_id)
            manifest["budget_ledger"] = json.loads(runtime.budget_json) if runtime else None
            persisted_state = json.loads(runtime.state_json) if runtime else {}
            if runtime:
                from app.services.research_artifact_service import ResearchArtifactService
                try:
                    persisted_state = ResearchArtifactService(db).hydrate_state(run_id, persisted_state)
                except Exception as exc:
                    manifest["state_hydration_error"] = type(exc).__name__
            manifest["checkpoint_count"] = db.scalar(select(func.count()).select_from(ResearchCheckpointModel))
            manifest["task_operations"] = [{"operation": item.operation, "status": item.status}
                for item in db.scalars(select(ResearchTaskModel))]
        state = result.get("research_state") or persisted_state
        manifest.update(
            finished_at=now(), latency_seconds=round(time.monotonic() - started, 3),
            metrics=metrics.get_report(), llm_responses=responses,
            candidate_count=len(state["candidate_papers"]) if isinstance(state.get("candidate_papers"), list) else None,
            paper_card_count=len(result.get("paper_cards") or []),
            reference_count=len(result.get("references") or []),
            estimated_tokens=(manifest.get("budget_ledger") or {}).get("estimated_tokens", 0),
            source_diagnostics=state.get("source_diagnostics"),
            resolved_years={key: state.get(key) for key in ("start_year", "end_year", "year_range_explicit")},
            stop_error_codes=[item.get("code") for item in result.get("errors") or [] if isinstance(item, dict)],
        )
        # 只导出审查需要的业务字段；完整内部状态和原始 provenance 保存在本地数据库。
        keep = ("status", "answer", "references", "paper_cards", "review", "clarification",
                "quality_gate", "generation_readiness", "claim_verification", "generation_blocked",
                "deliverable_validation", "final_review_integrity", "core_deliverables")
        write_json(output / "result.json", {key: result[key] for key in keep if key in result})
        write_json(output / "run.json", manifest)
        engine.dispose()
    print(json.dumps({"task": args.task, "status": manifest["task_status"],
                      "seconds": manifest["latency_seconds"],
                      "reported_tokens": manifest["metrics"]["tokens"]["total_tokens"],
                      "estimated_tokens": manifest["estimated_tokens"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
