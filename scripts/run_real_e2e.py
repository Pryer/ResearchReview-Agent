"""隔离的分层 RAG 验收入口。默认只预检，不调用学术来源或模型。

真实运行须提供 --execute、--token-budget 和 --deadline-seconds。
产物仅写本地 data/；不自动回答澄清或放宽质量门禁。
"""
from __future__ import annotations

import argparse
from datetime import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
import time
from uuid import uuid4
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.e2e_reporting import REPORT_VERSION, comparison_key, configuration, digest, export_reports, source_identity, write_json
from scripts.e2e_observation import RunObserver, paper_identity


def now():
    return datetime.now(ZoneInfo("Asia/Singapore")).isoformat(timespec="seconds")


class RagPrecheckFailed(RuntimeError):
    pass


def _runtime_precheck(settings):
    from app.services.rag_runtime import check_rag_runtime
    report = check_rag_runtime(settings)
    report["checks"]["pgvector_required"] = settings.card_vector_backend == "pgvector"
    report["ready"] = all(report["checks"].values())
    if report["ready"]:
        from app.storage.pgvector_store import PgVectorStore
        report["database_versions"] = PgVectorStore(settings).health()
    return report


def _perform_task(run_id, task, progress):
    from app.database.db import SessionLocal
    from app.schemas.agent_schema import AgentRequest
    from app.services.research_conversation_service import ResearchConversationService
    with SessionLocal() as db:
        service = ResearchConversationService(db, progress_callback=progress)
        return service.handle(AgentRequest(**{**task["request"], "session_id": run_id}))


def _collect_state(run_id, manifest):
    from sqlalchemy import func, select
    from app.database.db import SessionLocal
    from app.database.models import ResearchRuntimeModel, ResearchTaskModel, ResearchCheckpointModel
    from app.services.research_artifact_service import ResearchArtifactService
    with SessionLocal() as db:
        runtime = db.get(ResearchRuntimeModel, run_id)
        manifest["budget_ledger"] = json.loads(runtime.budget_json) if runtime else None
        state = json.loads(runtime.state_json) if runtime else {}
        if runtime:
            state = ResearchArtifactService(db).hydrate_state(run_id, state)
        manifest["checkpoint_count"] = db.scalar(select(func.count()).select_from(ResearchCheckpointModel))
        manifest["task_operations"] = [{"operation": item.operation, "status": item.status}
                                       for item in db.scalars(select(ResearchTaskModel))]
    return state


def _local_manifest(value):
    path = Path(value).resolve()
    if path.is_dir():
        path /= "run.json"
    if not path.is_relative_to((ROOT / "data").resolve()):
        raise ValueError("related runs must be under the repository data directory")
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default=str(ROOT / "examples/real_e2e_tasks.jsonl"))
    parser.add_argument("--task", required=True)
    parser.add_argument("--output-dir", required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--precheck-only", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--deadline-seconds", type=int)
    parser.add_argument("--token-budget", type=int)
    parser.add_argument("--sources", default="arxiv,semantic_scholar,openalex,crossref")
    parser.add_argument("--rerun-of", help="前次运行目录；原任务约束必须一致")
    parser.add_argument("--baseline-run", help="用于比较资格审查的历史运行目录")
    args = parser.parse_args(argv)
    if args.execute and (args.deadline_seconds is None or args.token_budget is None):
        parser.error("--execute requires explicit --token-budget and --deadline-seconds")
    if any(value is not None and value <= 0 for value in (args.deadline_seconds, args.token_budget)):
        parser.error("deadline and token budget must be positive")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.task):
        parser.error("invalid task identifier")
    sources = [source.strip() for source in args.sources.split(",") if source.strip()]
    if not sources or not set(sources) <= {"arxiv", "semantic_scholar", "openalex", "crossref"}:
        parser.error("only registered open sources are allowed")
    try:
        tasks_bytes = Path(args.tasks).read_bytes()
        tasks = [json.loads(line) for line in tasks_bytes.decode("utf-8").splitlines() if line.strip()]
        matches = [task for task in tasks if task["task_id"] == args.task]
        if len(matches) != 1:
            raise ValueError("task identifier must match exactly one task")
        task = matches[0]
        prior = _local_manifest(args.rerun_of) if args.rerun_of else None
        baseline = _local_manifest(args.baseline_run) if args.baseline_run else None
        if prior and digest(prior["task"]) != digest(task):
            raise ValueError("reruns must preserve the frozen task")
        if prior and prior.get("run_mode") != "execute":
            raise ValueError("prechecks are not primary research runs")
    except (OSError, ValueError, KeyError) as exc:
        parser.error(f"invalid task or related-run input: {type(exc).__name__}")
    run_id = args.task + "_" + datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:10]
    output = Path(args.output_dir).resolve() / run_id
    if not output.is_relative_to((ROOT / "data").resolve()):
        parser.error("output must be under the repository data directory")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "report_version": REPORT_VERSION, "run_id": run_id, "task_id": args.task, "started_at": now(),
        "run_mode": "execute" if args.execute else "precheck", "task": task, "targets": task.get("target_papers") or [],
        "tasks_sha256": sha256(tasks_bytes).hexdigest(), "task_sha256": digest(task),
        "runner_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "primary_pre_registered_run": bool(args.execute and not prior), "rerun_of": prior.get("run_id") if prior else None,
        "experiment_id": prior.get("experiment_id", prior["run_id"]) if prior else run_id,
        "parent_manifest_sha256": digest(prior) if prior else None,
        "task_status": "initializing", "manual_review": "pending_human_review", "requirement_satisfied": None,
        "rag_runtime_precheck": None, "artifact_errors": [],
    }
    if baseline:
        manifest["baseline_identity"] = {**comparison_key(baseline), **{key: baseline.get(key) for key in
            ("run_id", "run_mode", "source_tree_unchanged", "observed_candidate_pool_sha256", "artifact_errors")}}
    # WHY: 必须先留下最小记录；Settings 校验和应用导入失败也属于可审计结果。
    write_json(output / "run.json", manifest)
    os.environ.update({
        "DATABASE_URL": "sqlite:///" + (output / "research.db").as_posix(),
        "APP_LOG_DIR": str(output / "logs"), "CNKI_ENABLED": "false",
        "DEFAULT_SEARCH_SOURCES": ",".join(sources), "RETRIEVAL_RANKING_MODE": "hybrid",
        "CARD_VECTOR_TENANT": "e2e_" + run_id, "RETRIEVAL_CACHE_DIR": str(output / "retrieval_cache"),
        "PDF_SAVE_DIR": str(output / "pdfs"), "PARSED_SAVE_DIR": str(output / "parsed"),
        "REVIEW_SAVE_DIR": str(output / "reviews"),
        "LLM_REQUEST_TIMEOUT": "45", "LLM_CONTROL_PLANE_TIMEOUT": "45", "LLM_FAILOVER_TOTAL_TIMEOUT": "75",
    })
    if args.token_budget is not None:
        os.environ["AGENT_MAIN_TOKEN_BUDGET"] = str(args.token_budget)
    if args.deadline_seconds is not None:
        os.environ["AGENT_EXECUTION_DEADLINE_SECONDS"] = str(args.deadline_seconds)
    stage, result, state, engine, metrics = "source_identity", {}, {}, None, None
    observer = RunObserver(output / "observations.jsonl")
    observer.path.touch()
    started = time.monotonic()
    exit_code = 0
    try:
        manifest.update(source_identity(ROOT))
        stage = "configuration"
        from app.core.config import get_settings
        get_settings.cache_clear()
        settings = get_settings()
        manifest["configuration"] = configuration(settings)
        from app.services.rag_runtime import RAG_PIPELINE_VERSION
        manifest["pipeline_version"] = RAG_PIPELINE_VERSION
        stage = "rag_precheck"
        manifest["rag_runtime_precheck"] = _runtime_precheck(settings)
        manifest["configuration"]["vector_database_versions"] = manifest["rag_runtime_precheck"].get("database_versions")
        write_json(output / "run.json", manifest)
        if not manifest["rag_runtime_precheck"]["ready"]:
            raise RagPrecheckFailed()
        if not args.execute:
            manifest["task_status"] = "precheck_passed"
        else:
            stage = "database_initialization"
            from app.core.metrics import get_metrics_collector
            from app.database.db import engine
            from app.database.models import Base
            Base.metadata.create_all(engine)
            metrics = get_metrics_collector()
            metrics.reset()
            stage = "research_execution"
            manifest["task_status"] = "running"
            write_json(output / "run.json", manifest)
            def progress(step, current, total):
                with (output / "progress.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"at": now(), "step": step, "current": current, "total": total}, ensure_ascii=False) + "\n")
            with observer.installed():
                result = _perform_task(run_id, task, progress)
            manifest["task_status"] = result.get("status") or "unknown"
    except (Exception, KeyboardInterrupt) as exc:
        # WHY: 只保留异常类型和阶段，禁止 ValidationError/连接异常回显凭据。
        manifest.update(task_status="failed", error_type=type(exc).__name__, error_stage=stage)
        exit_code = 130 if isinstance(exc, KeyboardInterrupt) else 1
    finally:
        if engine is not None:
            try:
                state = _collect_state(run_id, manifest)
            except Exception as exc:
                manifest["artifact_errors"].append({"stage": "state_hydration", "error_type": type(exc).__name__})
        state.update(result.get("research_state") or {})
        for name in ("paper_cards", "claim_verification", "citation_map", "citation_registry"):
            if name in result:
                state[name] = result[name]
        if isinstance(state.get("candidate_papers"), list):
            # WHY: 排序前停止时仍保留已发现论文；检查点命中不能冒充排序或准入完成。
            observer.emit("paper_candidates_checkpoint", candidates=[paper_identity(p) for p in state["candidate_papers"]])
        manifest.update(finished_at=now(), latency_seconds=round(time.monotonic() - started, 3),
                        metrics=metrics.get_report() if metrics else None,
                        llm_responses=[e for e in observer.events if e["kind"] == "llm_usage"],
                        candidate_count=len(state.get("candidate_papers") or []),
                        paper_card_count=len(state.get("paper_cards") or []),
                        reference_count=len(result.get("references") or []),
                        unique_valid_cited_paper_count=state.get("unique_valid_cited_paper_count"),
                        estimated_tokens=(manifest.get("budget_ledger") or {}).get("estimated_tokens"),
                        source_diagnostics=state.get("source_diagnostics"),
                        resolved_years={key: state.get(key) for key in ("start_year", "end_year", "year_range_explicit")},
                        stop_error_codes=[e.get("code") for e in result.get("errors") or [] if isinstance(e, dict)])
        keep = ("status", "answer", "references", "paper_cards", "review", "body", "related_work", "introduction",
                "citation_map", "citation_registry", "clarification", "errors",
                "quality_gate", "generation_readiness", "claim_verification", "generation_blocked",
                "deliverable_validation", "final_review_integrity", "core_deliverables")
        write_json(output / "result.json", {key: result[key] for key in keep if key in result})
        try:
            manifest["source_tree_unchanged"] = source_identity(ROOT)["source_tree_sha256"] == manifest.get("source_tree_sha256")
            if not manifest["source_tree_unchanged"]:
                manifest["artifact_errors"].append({"stage": "source_identity", "error_type": "SourceChangedOrUnrecorded"})
        except Exception as exc:
            manifest["artifact_errors"].append({"stage": "source_identity", "error_type": type(exc).__name__})
        try:
            export_reports(output, manifest, state, result, observer.events, baseline)
        except Exception as exc:
            manifest["artifact_errors"].append({"stage": "audit_export", "error_type": type(exc).__name__})
        if manifest["artifact_errors"]:
            exit_code = 1
        write_json(output / "run.json", manifest)
        if engine is not None:
            engine.dispose()
    print(json.dumps({"run_id": run_id, "task": args.task, "status": manifest["task_status"],
                      "seconds": manifest["latency_seconds"], "audit_complete": not manifest["artifact_errors"]}, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
