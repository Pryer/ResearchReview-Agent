"""冻结已有标注集，使用真实 Provider 有界复评论文检索与 Chunk 检索。

所有产物仅写 data/；旧标签不是新增人工验收，缺金标准时不计算准确率。
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timedelta
import copy
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
import time
from unittest.mock import patch
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class ProviderEvaluationStopped(RuntimeError):
    pass


def original_round_deadline(record, root=ROOT, seen=None):
    """恢复不能把每个子运行的开始时间当作新一轮截止时间。"""
    if record.get("round_deadline_at"):
        return datetime.fromisoformat(record["round_deadline_at"])
    if record.get("parent_run"):
        from scripts.e2e_reporting import digest
        parent = (root / record["parent_run"]).resolve()
        seen = set() if seen is None else set(seen)
        if not parent.is_relative_to((root / "data").resolve()) or parent in seen:
            raise ValueError("invalid parent run chain")
        seen.add(parent)
        prior = json.loads((parent / "run.json").read_text(encoding="utf-8"))
        if record.get("parent_manifest_sha256") != digest(prior):
            raise ValueError("parent manifest changed")
        return original_round_deadline(prior, root, seen)
    return datetime.fromisoformat(record["started_at"]) + timedelta(seconds=record["limits"]["deadline_seconds"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--token-budget", required=True, type=int)
    parser.add_argument("--request-limit", required=True, type=int)
    parser.add_argument("--deadline-seconds", required=True, type=int)
    parser.add_argument("--skip-semantic", action="store_true")
    parser.add_argument("--chunk-paper-limit", type=int, default=3)
    parser.add_argument("--resume-from", help="同一轮的先前产物目录，不重置累计用量或截止时间")
    parser.add_argument("--budget-checkpoint", help="可选的后续已结算运行账本 JSON")
    args = parser.parse_args()
    if min(args.token_budget, args.request_limit, args.deadline_seconds) <= 0 or not 0 <= args.chunk_paper_limit <= 10:
        parser.error("invalid budget or chunk limit")
    output = Path(args.output_dir).resolve()
    if not output.is_relative_to((ROOT / "data").resolve()):
        parser.error("output must be under repository data")
    output.mkdir(parents=True, exist_ok=False)
    previous, resume_root, starting_ledger = None, None, {}
    execution_seconds = args.deadline_seconds
    if args.resume_from:
        resume_root = Path(args.resume_from).resolve()
        if not resume_root.is_relative_to((ROOT / "data").resolve()):
            parser.error("resume source must be under repository data")
        previous = json.loads((resume_root / "run.json").read_text(encoding="utf-8"))
        if previous.get("status") == "running":
            parser.error("cannot resume a still-running evaluation")
        round_deadline = original_round_deadline(previous)
        execution_seconds = int((round_deadline - datetime.now(round_deadline.tzinfo)).total_seconds())
        if execution_seconds <= 0:
            parser.error("original evaluation deadline reached")
        starting_ledger = copy.deepcopy(previous["budget_ledger"])
        if args.budget_checkpoint:
            checkpoint = Path(args.budget_checkpoint).resolve()
            if not checkpoint.is_relative_to(resume_root):
                parser.error("budget checkpoint must belong to the original run")
            newer = json.loads(checkpoint.read_text(encoding="utf-8"))["final_cumulative_budget"]
            if any(int(newer.get(k) or 0) < int(starting_ledger.get(k) or 0) for k in
                   ("llm_tokens", "retrieval_model_tokens", "retrieval_model_requests", "llm_requests")):
                parser.error("budget checkpoint cannot lower recorded usage")
            starting_ledger = copy.deepcopy(newer)
        if starting_ledger.get("tokens_reserved"):
            parser.error("cannot resume an unsettled active request")
        starting_ledger["token_limit"] = args.token_budget
    os.environ.update({"APP_LOG_DIR": str(output / "logs"), "RETRIEVAL_CACHE_DIR": str(output / "cache"),
                       "AGENT_MAIN_TOKEN_BUDGET": str(args.token_budget),
                       "RETRIEVAL_MODEL_REQUEST_LIMIT": str(args.request_limit),
                       "AGENT_EXECUTION_DEADLINE_SECONDS": str(execution_seconds),
                       "RETRIEVAL_MAX_RETRIES": "0", "CARD_VECTOR_TENANT": "eval_" + output.name})
    if resume_root:
        os.environ["RETRIEVAL_CACHE_DIR"] = str(resume_root / "cache")
    from scripts.e2e_reporting import configuration, digest, source_identity, write_csv, write_json
    from scripts.e2e_observation import RunObserver
    from app.core.config import get_settings
    from app.agent.execution_budget import active_budget, budget_scope, check_execution, AgentBudgetExceeded
    from app.clients import embedding_client, reranker_client
    from app.core.metrics import get_metrics_collector
    from app.services.rag_runtime import RAG_PIPELINE_VERSION, check_rag_runtime
    from scripts.evaluate_hybrid_retrieval import evaluate, _metrics
    from scripts.evaluate_miskill_recovery import evaluate as recovery_evaluate

    get_settings.cache_clear()
    settings = get_settings()
    manifest = {"kind": "provider_retrieval_evaluation", "started_at": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
                "status": "initializing", "limits": vars(args), "pipeline_version": RAG_PIPELINE_VERSION,
                "configuration": configuration(settings), "artifacts": [], "completed_stages": [],
                "ground_truth_status": "historical_labels_not_independently_reverified", "errors": []}
    manifest.update(source_identity(ROOT))
    manifest["round_deadline_at"] = (round_deadline if previous else
        datetime.fromisoformat(manifest["started_at"]) + timedelta(seconds=args.deadline_seconds)).isoformat()
    if previous:
        manifest["parent_run"] = str(resume_root.relative_to(ROOT))
        manifest["parent_manifest_sha256"] = digest(previous)
        # WHY: 初始账本是审计快照，不能与随后持续结算的可变账本共享对象。
        manifest["initial_cumulative_budget"] = copy.deepcopy(starting_ledger)
        manifest["original_deadline_seconds_remaining"] = execution_seconds
        manifest["configuration"]["cache_policy"] = "shared_with_parent_same_round"
        manifest["reused_artifacts"] = {}
    write_json(output / "run.json", manifest)
    dataset_root = resume_root / "inputs" if resume_root else ROOT / "data/retrieval_eval"
    inputs = output / "inputs"
    inputs.mkdir()
    for path in sorted(dataset_root.glob("*qrels*.jsonl")):
        (inputs / path.name).write_bytes(path.read_bytes())
    specs = [("enriched", "real_merged_snapshot_enriched.json", "human_all_qrels.jsonl", 120),
             ("original", "real_merged_snapshot.json", "human_all_qrels.jsonl", 120),
             ("closed274", "human_test_snapshot.json", "human_test_qrels.jsonl", 72)]
    for _, snapshot, _, _ in specs:
        (inputs / snapshot).write_bytes((dataset_root / snapshot).read_bytes())
    round2 = [json.loads(line) for line in (inputs / "human_test_qrels_round2.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    wanted = {(r["task_id"], r["paper_id"]) for r in round2}
    enriched = json.loads((inputs / "real_merged_snapshot_enriched.json").read_text(encoding="utf-8"))
    subset = {"tasks": [{**t, "candidate_papers": [p for p in t["candidate_papers"] if (t["task_id"], p["paper_id"]) in wanted]}
                         for t in enriched["tasks"]]}
    write_json(inputs / "round2_closed_snapshot.json", subset)
    specs.append(("closed206", "round2_closed_snapshot.json", "human_test_qrels_round2.jsonl", 72))
    manifest["input_sha256"] = {p.name: sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir() if p.is_file()}
    write_json(output / "run.json", manifest)
    state, reports, usage_events = {"agent_execution_budget": starting_ledger} if starting_ledger else {}, {}, []
    observer = RunObserver(output / "observations.jsonl")
    observer.path.touch()
    metrics = get_metrics_collector()
    metrics.reset()
    original_post = embedding_client.post_json

    def bounded_post(url, key, body, **kwargs):
        check_execution()
        active = active_budget()
        ledger = active["ledger"]
        if int(ledger.get("retrieval_model_requests") or 0) >= settings.retrieval_model_request_limit:
            raise AgentBudgetExceeded("evaluation retrieval request limit reached")
        source = body.get("input") or {}
        count = len(source.get("texts") or source.get("documents") or [1])
        bound = len(json.dumps(source, ensure_ascii=False).encode()) + 512 * count
        bound += len(str(source.get("query") or "").encode()) * max(0, count - 1)
        used = int(ledger.get("llm_tokens") or 0) + int(ledger.get("retrieval_model_tokens") or 0)
        if used + int(ledger.get("tokens_reserved") or 0) + bound > ledger["token_limit"]:
            active["exhausted"] = "retrieval input reservation exceeds remaining evaluation budget"
            raise AgentBudgetExceeded(active["exhausted"])
        ledger["tokens_reserved"] = int(ledger.get("tokens_reserved") or 0) + bound
        try:
            payload = original_post(url, key, body, **kwargs)
        except BaseException as exc:
            # WHY: 请求结果未知时保守占用本地预算，不把它宣称为 Provider 实际账单。
            ledger["retrieval_model_tokens"] = int(ledger.get("retrieval_model_tokens") or 0) + bound
            ledger["estimated_retrieval_tokens"] = int(ledger.get("estimated_retrieval_tokens") or 0) + bound
            event = {"kind": kwargs.get("kind"), "status": "failed", "error_type": type(exc).__name__, "budget_estimate": bound}
            status = re.search(r"HTTP (\d{3})", str(exc))
            if status:
                event["http_status"] = int(status.group(1))
            usage_events.append(event)
            with (output / "provider_usage.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event) + "\n")
            if isinstance(exc, (AgentBudgetExceeded, KeyboardInterrupt)):
                raise
            raise ProviderEvaluationStopped(type(exc).__name__) from None
        finally:
            ledger["tokens_reserved"] -= bound
        usage = payload.get("usage")
        known = isinstance(usage, dict) and type(usage.get("total_tokens")) in {int, float}
        if not known:
            ledger["retrieval_model_tokens"] = int(ledger.get("retrieval_model_tokens") or 0) + bound
            ledger["estimated_retrieval_tokens"] = int(ledger.get("estimated_retrieval_tokens") or 0) + bound
        event = {"kind": kwargs.get("kind"), "model": body.get("model"), "status": "responded",
                 "usage": {k: v for k, v in (usage or {}).items() if k.endswith("tokens")} if isinstance(usage, dict) else None,
                 "budget_estimate": 0 if known else bound}
        usage_events.append(event)
        with (output / "provider_usage.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event) + "\n")
        return payload

    def save_stage(name, report):
        write_json(output / (name + ".json"), report)
        manifest["artifacts"].append(name + ".json")
        manifest["completed_stages"].append(name)
        manifest["budget_ledger"] = state.get("agent_execution_budget", {})
        write_json(output / "run.json", manifest)
        print(json.dumps({"stage": name, "status": "completed", "retrieval_requests": manifest["budget_ledger"].get("retrieval_model_requests", 0)}, ensure_ascii=False), flush=True)

    def label_views(name, snapshot, report):
        tasks = {t["task_id"]: t for t in snapshot["tasks"]}
        views = {}
        for path in sorted(inputs.glob("*qrels*.jsonl")):
            groups = {}
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    groups.setdefault(row["task_id"], {})[row["paper_id"]] = row["relevance"]
            valid = bool(groups) and set(groups) <= set(tasks)
            for tid, labels in groups.items():
                ids = {p["paper_id"] for p in tasks.get(tid, {}).get("candidate_papers", [])}
                missing = ids - set(labels)
                # WHY: 仅历史明确剔除学位论文的 eligible 视图允许缺4/2项；其他未标注不充当负例。
                excluded_ids = set()
                if missing and "eligible" in path.name:
                    from app.tools.paper_matching import compile_scope
                    from app.tools.rank_papers import evaluate_paper_hard_filters
                    task = tasks[tid]
                    compiled = compile_scope(selected_scope=task.get("selected_scope") or {},
                        semantic_frame=task.get("research_semantic_frame") or {}, screening_protocol=task.get("screening_protocol") or {},
                        required_concepts=task.get("required_concepts") or [], topic_anchors=task.get("topic_anchors") or [],
                        search_branches=task.get("search_branches") or [], topic=task.get("topic") or "")
                    for paper in task["candidate_papers"]:
                        if paper["paper_id"] not in missing:
                            continue
                        passed, reason, _ = evaluate_paper_hard_filters(paper, topic=task.get("topic") or "",
                            keywords=task.get("keywords") or [], required_concepts=task.get("required_concepts") or [],
                            excluded_title_terms=task.get("excluded_title_terms") or [], scope=task.get("selected_scope") or {},
                            search_branches=task.get("search_branches") or [], screening_protocol=task.get("screening_protocol") or {},
                            compiled_scope=compiled, ranking_mode="hybrid")
                        if not passed and reason == "document_type_filter":
                            excluded_ids.add(paper["paper_id"])
                eligible = missing <= excluded_ids
                if not set(labels) <= ids or (missing and not eligible):
                    valid = False
            if not valid:
                continue
            view = {"ground_truth": path.name, "label_status": "historical_not_new_heldout", "tasks": []}
            for task in report["tasks"]:
                if task["task_id"] not in groups:
                    continue
                grades = groups[task["task_id"]]
                view["tasks"].append({"task_id": task["task_id"], "labeled_count": len(grades),
                    "rankings": {mode: {str(threshold): {str(k): _metrics(ranking, grades, k, threshold)
                                                        for k in report["profile"]["metric_ks"]}
                                        for threshold in (1, 2)}
                                 for mode, ranking in task["raw_rankings"].items()}})
            views[path.name] = view
        save_stage(name + "_label_views", views)

    exit_code = 0
    stage = "precheck"
    try:
        manifest["precheck"] = check_rag_runtime(settings)
        if not manifest["precheck"]["ready"]:
            raise ProviderEvaluationStopped("RAG precheck failed")
        manifest["status"] = "running"
        write_json(output / "run.json", manifest)
        with budget_scope(state), ExitStack() as stack:
            stack.enter_context(patch.object(embedding_client, "post_json", bounded_post))
            stack.enter_context(patch.object(reranker_client, "post_json", bounded_post))
            stack.enter_context(observer.installed())
            for name, snapshot_file, qrels_file, k in specs:
                stage = name
                profile = {"k": k, "metric_ks": [x for x in (5, 10, 20, 50, 72, 120) if x <= k],
                           "split": "dev", "dataset_kind": "historical_label_reassessment", "annotation_version": qrels_file}
                write_json(inputs / (name + "_profile.json"), profile)
                reused = resume_root / (name + "_retrieval.json") if resume_root else None
                if reused and reused.exists():
                    report = json.loads(reused.read_text(encoding="utf-8"))
                    manifest["reused_artifacts"][reused.name] = sha256(reused.read_bytes()).hexdigest()
                else:
                    report = evaluate(inputs / snapshot_file, inputs / qrels_file, inputs / (name + "_profile.json"), relevance_threshold=2)
                if any(task["degraded"] for task in report["tasks"]):
                    write_json(output / (name + "_degraded.json"), report)
                    raise ProviderEvaluationStopped("degraded retrieval is not a valid Provider comparison")
                reports[name] = report
                save_stage(name + "_retrieval", report)
                label_views(name, json.loads((inputs / snapshot_file).read_text(encoding="utf-8")), report)
                recovery_path = resume_root / (name + "_miskill_recovery.json") if resume_root else None
                if recovery_path and recovery_path.exists():
                    recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
                    manifest["reused_artifacts"][recovery_path.name] = sha256(recovery_path.read_bytes()).hexdigest()
                else:
                    recovery = recovery_evaluate(str(inputs / snapshot_file), str(inputs / qrels_file), k)
                save_stage(name + "_miskill_recovery", recovery)
            if not args.skip_semantic:
                stage = "semantic_admission"
                from app.services.llm_service import LLMService
                from app.services.retrieval_ranking_service import rank_candidates, screen_candidates
                llm = LLMService()
                admission = []
                for task in enriched["tasks"]:
                    check_execution()
                    ranked, _ = rank_candidates(task, settings=settings, include_rerank=True)
                    accepted, diagnostics = screen_candidates(dict(task), ranked, llm, target=len(ranked), strict_provider_errors=True)
                    labels = {json.loads(l)["paper_id"]: json.loads(l)["relevance"]
                              for l in (inputs / "human_all_qrels.jsonl").read_text(encoding="utf-8").splitlines()
                              if l.strip() and json.loads(l)["task_id"] == task["task_id"]}
                    admission.append({"task_id": task["task_id"], "candidate_count": len(ranked),
                                      "admitted_ids": [p["paper_id"] for p in accepted], "diagnostics": diagnostics,
                                      "metrics": _metrics([p["paper_id"] for p in accepted], labels, max(1, len(accepted)), 2),
                                      "precision": sum(labels.get(p["paper_id"], 0) >= 2 for p in accepted) / len(accepted) if accepted else None})
                    save_stage("semantic_admission_partial", admission)
                save_stage("semantic_admission", admission)
            if args.chunk_paper_limit:
                stage = "chunk_retrieval"
                from app.tools.parse_pdf import parse_pdf
                from app.tools.chunk_pdf import chunk_pdf
                from app.tools.chunk_retriever import retrieve_chunks
                files = sorted((ROOT / "data/pdfs").glob("arxiv_*.pdf"))[:args.chunk_paper_limit]
                chunk_reports, review_rows = [], []
                for path in files:
                    check_execution()
                    paper_id = "arxiv:" + ".".join(path.stem.split("_")[1:3])
                    parsed = parse_pdf(str(path), paper_id=paper_id)
                    chunks = chunk_pdf(parsed, paper_id=paper_id, max_chars=settings.card_extraction_chunk_chars)
                    report = retrieve_chunks(paper_id, chunks, settings=settings, parsed_text=parsed, topic="Research problem, methodology, results and limitations")
                    chunk_reports.append({"paper_id": paper_id, "source_pdf_sha256": parsed["source_pdf_hash"], "report": report.model_dump(mode="json")})
                    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
                    for field, hits in report.field_hits.items():
                        for rank, hit in enumerate(hits, 1):
                            chunk = by_id[hit.chunk_id]
                            review_rows.append({"paper_id": paper_id, "field": field, "rank": rank, "chunk_id": hit.chunk_id,
                                                "page": chunk["page"], "section": chunk["section"], "text": chunk["text"],
                                                "human_relevance": "", "gold_evidence": "", "reviewer": "", "notes": ""})
                    write_csv(output / "chunk_human_review.csv", ["paper_id", "field", "rank", "chunk_id", "page", "section", "text", "human_relevance", "gold_evidence", "reviewer", "notes"], review_rows)
                    write_json(output / "chunk_retrieval_pending_gold.json", {"papers": chunk_reports, "evidence_recall": None, "reason": "requires independent human gold evidence"})
                save_stage("chunk_retrieval", {"papers": chunk_reports, "evidence_recall": None, "precision": None,
                                               "human_review_rows": len(review_rows), "status": "pending_human_gold"})
        manifest["status"] = "completed_automated_stages_pending_human"
    except (Exception, KeyboardInterrupt) as exc:
        manifest["status"] = "stopped"
        manifest["errors"].append({"stage": stage, "error_type": type(exc).__name__})
        exit_code = 1
    finally:
        manifest["budget_ledger"] = state.get("agent_execution_budget", {})
        manifest["metrics"] = metrics.get_report()
        manifest["provider_responses"] = usage_events
        manifest["finished_at"] = datetime.now(ZoneInfo("Asia/Singapore")).isoformat()
        manifest["source_tree_unchanged"] = source_identity(ROOT)["source_tree_sha256"] == manifest["source_tree_sha256"]
        manifest["manual_actions"] = ["确认历史480条相关性标签的标注来源；不能作为新的独立留出集。",
                                      "为Chunk评测独立标注每篇/字段的全部相关证据位置；仅审Top-K不足以计算Recall。",
                                      "检查高排名负例和漏召回正例，不能直接把LLM准入结果当人工标签。"]
        write_json(output / "run.json", manifest)
        print(json.dumps({"status": manifest["status"], "completed_stages": manifest["completed_stages"], "budget": manifest["budget_ledger"], "errors": manifest["errors"]}, ensure_ascii=False), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
