"""真实验收的本地导出契约；未知值不冒充人工判定或执行成功。"""
from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import platform
from importlib.metadata import PackageNotFoundError, version


REPORT_VERSION = "real-e2e-audit.v1"
MANUAL_VERDICTS = {"supported", "partially_supported", "unsupported", "contradicted", "uncertain", "not_reviewed"}


def digest(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def write_json(path, value):
    # WHY: 中断时保留上一份完整记录；临时文件与目标在同一目录。
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def source_identity(root):
    files = sorted([p for folder in ("app", "scripts") for p in (root / folder).rglob("*.py")
                    if "__pycache__" not in p.parts and not p.is_symlink()] +
                   [p for pattern in ("requirements*.txt", "environment.yml", "pyproject.toml", ".env.example")
                    for p in root.glob(pattern) if p.is_file() and not p.is_symlink()])
    inventory = {p.relative_to(root).as_posix(): sha256(p.read_bytes()).hexdigest() for p in files}
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.DEVNULL)
    return {"commit_sha": git("rev-parse", "HEAD").decode().strip(),
            "tracked_diff_sha256": sha256(git("diff", "--binary", "HEAD", "--", "app", "scripts", "*.txt", "*.yml", "pyproject.toml", ".env.example")).hexdigest(),
            "source_files": inventory, "source_tree_sha256": digest(inventory)}


def configuration(settings):
    # WHY: 只导出允许公开的配置值，连接串、端点和密钥均不进入报告。
    names = ("retrieval_ranking_mode", "enable_pdf_pipeline", "card_chunk_retrieval_mode",
             "card_vector_backend", "card_vector_db_failure_policy", "card_vector_tenant",
             "agent_main_token_budget", "agent_execution_deadline_seconds", "llm_provider", "llm_model",
             "llm_backup_enabled", "llm_backup_provider", "llm_backup_model", "llm_request_timeout",
             "llm_control_plane_timeout", "llm_failover_total_timeout", "retrieval_provider",
             "retrieval_embedding_model", "retrieval_embedding_dimension", "retrieval_embedding_batch_size",
             "retrieval_rerank_model", "retrieval_rerank_batch_size", "retrieval_request_timeout",
             "retrieval_max_retries", "retrieval_model_request_limit", "retrieval_query_max",
             "retrieval_bm25_top_k", "retrieval_dense_top_k", "retrieval_rrf_k",
             "retrieval_cross_encoder_initial_k", "retrieval_cross_encoder_max_k", "retrieval_cache_version",
             "card_extraction_chunk_chars", "card_extraction_max_chunks", "card_chunk_top_k",
             "card_chunk_rerank_window", "card_chunk_enable_rerank", "card_vector_index_max_chunks",
             "paper_screening_abstract_chars", "paper_screening_batch_abstract_chars",
             "enable_claim_verification", "claim_verification_max_context_chars", "cnki_enabled")
    values = {name: getattr(settings, name) for name in names}
    values["sources"] = settings.search_sources_list
    values["cache_policy"] = "fresh_per_run"
    safe_settings = {key: value for key, value in settings.model_dump(mode="json").items()
                     if not re.search(r"api_key|password|secret|credential|auth_token", key)
                     and not key.endswith(("_url", "_dir", "_path")) and key != "card_vector_tenant"}
    values["effective_behavior_sha256"] = digest(safe_settings)
    values["endpoint_sha256"] = {key: sha256(str(getattr(settings, key)).encode()).hexdigest()
                                 for key in ("llm_base_url", "llm_backup_base_url", "retrieval_embedding_url", "retrieval_rerank_url")}
    from app.core.config import get_review_threshold_policy
    values["quality_thresholds"] = get_review_threshold_policy().snapshot()
    values["python_version"] = platform.python_version()
    values["dependency_versions"] = {}
    for name in ("pydantic", "SQLAlchemy", "psycopg", "pgvector", "numpy", "PyMuPDF", "requests", "openai", "rank-bm25", "jieba"):
        try:
            values["dependency_versions"][name] = version(name)
        except PackageNotFoundError:
            values["dependency_versions"][name] = None
    return values


def comparison_key(manifest):
    config = {k: v for k, v in manifest.get("configuration", {}).items() if k != "card_vector_tenant"}
    return {"report_version": manifest.get("report_version"), "task_sha256": manifest.get("task_sha256"),
            "pipeline_version": manifest.get("pipeline_version"), "configuration": config,
            "source_tree_sha256": manifest.get("source_tree_sha256"),
            "targets_sha256": digest(manifest.get("targets", []))}


def _matches(target, paper):
    """强身份冲突时不能用相同题名兜底；缺强身份时允许题名+年份命中。"""
    def norm(key, value):
        value = str(value or "").strip().lower()
        if key == "doi":
            return re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:)", "", value)
        return re.sub(r"v\d+$", "", re.sub(r"^(?:arxiv:|https?://arxiv.org/abs/)", "", value))
    compared = [(norm(k, target.get(k)), norm(k, paper.get(k))) for k in ("doi", "arxiv_id")
                if target.get(k) and paper.get(k)]
    if compared:
        return "strong_id" if all(a == b for a, b in compared) else None
    title = lambda value: re.sub(r"[^\w]+", " ", str(value or "").lower()).strip()
    if target.get("title") and title(target["title"]) == title(paper.get("title")):
        if target.get("year") is not None and target["year"] == paper.get("year"):
            return "title_year"
    return None


def paper_report(targets, events):
    rounds = []
    for event in events:
        if event["kind"] not in {"paper_candidates_checkpoint", "paper_ranking", "paper_admission"}:
            continue
        rows = []
        for target in targets:
            ranked = event.get("ranked")
            positions = [i for i, p in enumerate(ranked or [], 1) if _matches(target, p)]
            candidate_matches = [{"paper": p, "match_basis": _matches(target, p)}
                                 for p in event.get("candidates", []) if _matches(target, p)]
            admitted = event.get("admitted")
            rows.append({"target": target, "candidate_matches": candidate_matches,
                         "candidate_hit": bool(candidate_matches),
                         "rank_in_observed_window": min(positions) if positions else None,
                         "hit_at_5": bool(positions and min(positions) <= 5) if ranked is not None else None,
                         "reciprocal_rank": 1 / min(positions) if positions else (0 if ranked is not None else None),
                         "admitted": any(_matches(target, p) for p in admitted) if admitted is not None else None})
        rounds.append({"sequence": event["sequence"], "kind": event["kind"], "targets": rows,
                       "error_type": event.get("error_type")})
    pool = [event.get("candidates", []) for event in events if event["kind"] == "paper_ranking"]
    return {"targets": targets, "rounds": rounds, "observed_candidate_pool_sha256": digest(pool) if pool else None,
            "metric_scope": "known_items_only_not_topic_recall", "target_count": len(targets)}


def _readback(span, paper_id, state):
    from app.schemas.retrieval_schema import fingerprint
    from app.tools.extract_paper_card import _clean_abstract
    if span.get("source_status") == "source_unverified":
        return {"readable": False, "reason": "source_unverified"}
    parsed = (state.get("parsed_papers") or {}).get(paper_id) or {}
    if span.get("source_type") == "abstract":
        paper = next((p for p in state.get("paper_details") or [] if p.get("paper_id") == paper_id), {})
        source = _clean_abstract(paper.get("abstract") or "")
    elif span.get("page") is not None:
        pages = [p for p in parsed.get("pages", []) if p.get("page") == span["page"]]
        source = pages[0].get("text", "") if len(pages) == 1 else ""
    elif span.get("offset_basis") == "full_text" or span.get("section") == "full_text":
        source = parsed.get("full_text") or ""
    else:
        source = (parsed.get("sections") or {}).get(span.get("section")) or ""
    if not source:
        return {"readable": None, "reason": "source_unavailable"}
    start, end = span.get("char_start"), span.get("char_end")
    if type(start) is not int or type(end) is not int:
        return {"readable": None, "reason": "missing_offsets"}
    if not 0 <= start < end <= len(source) or source[start:end] != span.get("text"):
        return {"readable": False, "reason": "offset_mismatch"}
    if span.get("source_document_hash") and fingerprint(source) != span["source_document_hash"]:
        return {"readable": False, "reason": "source_version_mismatch"}
    if span.get("source_text_hash") and sha256(span["text"].encode()).hexdigest() != span["source_text_hash"]:
        return {"readable": False, "reason": "text_hash_mismatch"}
    return {"readable": True, "reason": "exact_readback"}


CLAIM_COLUMNS = ["claim_id", "claim", "text_scope", "citation", "paper_id", "paper_identity",
                 "evidence_ids", "unresolved_evidence_ids", "source_spans", "span_readable", "readback_reason", "automatic_support",
                 "automatic_verification_status", "evidence_support", "final_judgement", "reviewer", "reviewed_at", "notes"]
CONSTRAINT_COLUMNS = ["constraint_id", "requirement", "judgement", "reviewer", "reviewed_at", "evidence"]


def claim_rows(state, result):
    cards = {p["paper_id"]: p for p in state.get("paper_cards") or []}
    all_evidence_ids = {span.get("evidence_id") for card in cards.values() for span in card.get("evidence_spans") or []}
    citation_map = {str(number): pid for pid, number in (state.get("citation_map") or {}).items()}
    report = state.get("claim_verification") or {}
    rows = []
    gate = result.get("quality_gate") or {}
    published = (result.get("status") in {"completed", "partial"}
                 and (gate.get("passed") is True or gate.get("draft_released") is True)
                 and any(str(result.get(key) or "").strip() for key in ("body", "review", "related_work", "introduction")))
    for sentence_index, sentence in enumerate(report.get("claims") or [], 1):
        if not sentence.get("factual", True):
            continue
        for atom_index, atom in enumerate(sentence.get("atomic_claims") or [sentence], 1):
            claim_id = f"{sentence.get('claim_id') or sentence_index}:a{atom_index}"
            for citation in dict.fromkeys(atom.get("citations") or [""]):
                citation = str(citation)
                pid = citation if citation in cards else citation_map.get(citation.strip("[]"), citation)
                card = cards.get(pid, {})
                spans = [s for s in card.get("evidence_spans") or [] if s.get("evidence_id") in (atom.get("evidence_ids") or [])]
                checks = [_readback(span, pid, state) for span in spans]
                readable = (False if any(c["readable"] is False for c in checks) else
                            True if checks and all(c["readable"] is True for c in checks) else None)
                rows.append({"claim_id": claim_id, "claim": atom.get("text") or atom.get("sentence") or sentence.get("sentence"),
                             "text_scope": "published" if published else "quarantined_or_unpublished",
                             "citation": citation, "paper_id": pid,
                             "paper_identity": json.dumps({k: card.get(k) for k in ("title", "doi", "year", "url")}, ensure_ascii=False),
                             "evidence_ids": json.dumps(atom.get("evidence_ids") or []),
                             "unresolved_evidence_ids": json.dumps([eid for eid in atom.get("evidence_ids") or [] if eid not in all_evidence_ids]),
                             "source_spans": json.dumps([{**s, "readback": c} for s, c in zip(spans, checks)], ensure_ascii=False),
                             "span_readable": "true" if readable is True else "false" if readable is False else "unknown",
                             "readback_reason": ";".join(c["reason"] for c in checks) if checks else "no_linked_source_span",
                             "automatic_support": atom.get("support_status", "unknown"),
                             "automatic_verification_status": atom.get("verification_status", "unknown"),
                             "evidence_support": "not_reviewed", "final_judgement": "not_reviewed",
                             "reviewer": "", "reviewed_at": "", "notes": ""})
    return rows


def write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _ratio(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "value": numerator / denominator if denominator else None}


def evaluation_summary(manifest, rows, constraints, baseline=None, paper=None):
    groups = {}
    for row in rows:
        groups.setdefault(row["claim_id"], []).append(row)
    judged, supported, uncertain, errors = 0, 0, 0, []
    for claim_id, links in groups.items():
        if any(row["evidence_support"] not in MANUAL_VERDICTS for row in links):
            errors.append({"claim_id": claim_id, "reason": "invalid_link_support_verdict"})
            continue
        verdicts = {row["final_judgement"] for row in links}
        if len(verdicts) != 1 or not verdicts <= MANUAL_VERDICTS:
            errors.append({"claim_id": claim_id, "reason": "inconsistent_or_invalid_verdict"})
            continue
        verdict = next(iter(verdicts))
        if verdict == "not_reviewed":
            continue
        if any(not row["reviewer"].strip() or not row["reviewed_at"].strip() for row in links):
            errors.append({"claim_id": claim_id, "reason": "missing_reviewer_or_time"})
            continue
        judged += 1
        supported += verdict == "supported"
        uncertain += verdict == "uncertain"
    applicable, satisfied, reviewed_constraints = len(constraints), 0, 0
    for row in constraints:
        verdict = row["judgement"]
        if verdict == "not_reviewed":
            continue
        if (verdict not in {"passed", "correctly_blocked", "failed", "uncertain", "not_applicable"}
                or not all(row.get(k, "").strip() for k in ("reviewer", "reviewed_at", "evidence"))):
            errors.append({"constraint_id": row["constraint_id"], "reason": "invalid_or_unsubstantiated_review"})
            continue
        reviewed_constraints += 1
        if verdict != "not_applicable":
            satisfied += verdict in {"passed", "correctly_blocked"}
        else:
            applicable -= 1
    reasons = []
    if baseline is None:
        reasons.append("no_baseline_provided")
    else:
        left, right = comparison_key(manifest), comparison_key(baseline)
        reasons.extend("different_or_missing:" + k for k in left if not left[k] or left[k] != right.get(k))
        if not paper or not paper.get("observed_candidate_pool_sha256") or paper.get("observed_candidate_pool_sha256") != baseline.get("observed_candidate_pool_sha256"):
            reasons.append("candidate_pool_not_frozen_or_different")
        if manifest.get("run_mode") != "execute" or baseline.get("run_mode") != "execute":
            reasons.append("not_executed_research_runs")
        if manifest.get("source_tree_unchanged") is not True or baseline.get("source_tree_unchanged") is not True:
            reasons.append("source_identity_unstable_or_unknown")
        if manifest.get("artifact_errors") or baseline.get("artifact_errors"):
            reasons.append("audit_incomplete")
    links = [row for row in rows if row["citation"]]
    constraint_rate = _ratio(satisfied, applicable)
    if reviewed_constraints < len(constraints):
        constraint_rate["value"] = None
    return {"report_version": REPORT_VERSION, "task_status": manifest["task_status"],
            "citation_readback_rate": _ratio(sum(r["span_readable"] == "true" for r in links), len(links)),
            "readback_unknown_links": sum(r["span_readable"] == "unknown" for r in links),
            "unresolved_evidence_ids": sorted({eid for row in rows for eid in json.loads(row["unresolved_evidence_ids"])}),
            "uncited_atomic_claims": sum(all(not row["citation"] for row in group) for group in groups.values()),
            "human_support_rate": _ratio(supported, judged),
            "human_review_coverage": _ratio(judged, len(groups)), "human_uncertain_claims": uncertain,
            "human_not_judged_claims": len(groups) - judged, "manual_review_errors": errors,
            "claim_text_scopes": {scope: sum(row["text_scope"] == scope for row in rows)
                                  for scope in {row["text_scope"] for row in rows}},
            "hard_constraint_adherence": constraint_rate,
            "constraint_review_coverage": _ratio(reviewed_constraints, len(constraints)),
            "independent_acceptance": "pending_human_review",
            "baseline_comparison": {"comparable": not reasons, "reasons": reasons,
                                    "scope": "strict_same_conditions_not_causal_effect"}}


def export_reports(output, manifest, state, result, events, baseline=None):
    rows = claim_rows(state, result)
    write_csv(output / "claim_evidence.csv", CLAIM_COLUMNS, rows)
    frozen_constraints = manifest["task"].get("acceptance_constraints") or [
        {"id": "original_request", "requirement": manifest["task"]["request"]["user_query"]}]
    constraints = [{"constraint_id": c["id"], "requirement": c["requirement"], "judgement": "not_reviewed",
                    "reviewer": "", "reviewed_at": "", "evidence": ""} for c in frozen_constraints]
    write_csv(output / "manual_constraints.csv", CONSTRAINT_COLUMNS, constraints)
    frozen = {"claims": rows, "constraints": constraints}
    write_json(output / "review_template.json", frozen)
    manifest["review_template_sha256"] = digest(frozen)
    paper = paper_report(manifest["targets"], events)
    write_json(output / "paper_retrieval.json", paper)
    manifest["observed_candidate_pool_sha256"] = paper["observed_candidate_pool_sha256"]
    chunk = {"observations": [e for e in events if e["kind"] == "chunk_retrieval"],
             "papers": [{"paper_id": c.get("paper_id"), "quality_status": c.get("quality_status"),
                          "extraction_coverage": c.get("extraction_coverage")}
                         for c in state.get("paper_cards") or []]}
    write_json(output / "chunk_retrieval.json", chunk)
    vectors = [e for e in events if e["kind"] == "vector_query"]
    models = [e for e in events if e["kind"] == "model"]
    manifest["rag_runtime_observed"] = {
        "vector_backend_used": "pgvector" if vectors else None,
        "vector_query_attempts": len(vectors), "vector_query_successes": sum(e["outcome"] == "succeeded" for e in vectors),
        "snapshot_ids": sorted({e["snapshot_id"] for e in vectors}),
        "indexes": [{"paper_id": e.get("paper_id"), **e["report"]["index_coverage"]}
                    for e in events if e["kind"] == "chunk_retrieval" and (e.get("report") or {}).get("index_coverage")],
        "model_operation_counts": {kind: {"attempted": sum(e["model_kind"] == kind for e in models),
                                             "succeeded": sum(e["model_kind"] == kind and e["outcome"] == "succeeded" for e in models),
                                             "http_responses": sum(e["http_responses"] for e in models if e["model_kind"] == kind)}
                                   for kind in ("embedding", "reranker")},
        "retrieval_http_attempts": {kind: (manifest.get("budget_ledger") or {}).get(f"retrieval_{kind}_requests")
                                    for kind in ("embedding", "rerank")},
        "model_operations": models,
        "retrieval_degraded": [e for e in events if (e.get("report") or {}).get("degraded")],
        "scope": "observed_calls_not_precheck_or_config"}
    conditions = []
    state_evidence = {key: result.get(key, state.get(key)) for key in
                      ("quality_gate", "generation_readiness", "generation_blocked", "deliverable_validation")}
    state_evidence["errors"] = [{"code": error.get("code")} for error in result.get("errors") or state.get("errors") or [] if isinstance(error, dict)]
    write_json(output / "state_evidence.json", state_evidence)
    if manifest.get("error_type"):
        conditions.append({"code": manifest["error_type"], "stage": manifest.get("error_stage"), "evidence": "run.json#/error_type"})
    for key in ("quality_gate", "generation_readiness", "generation_blocked", "deliverable_validation"):
        value = result.get(key, state.get(key))
        if value:
            conditions.append({"stage": key, "state": value, "evidence": f"state_evidence.json#/{key}"})
    for index, error in enumerate(result.get("errors") or state.get("errors") or []):
        if isinstance(error, dict):
            conditions.append({"stage": "reported_error", "code": error.get("code"), "evidence": f"state_evidence.json#/errors/{index}"})
    write_json(output / "root_cause_chain.json", {
        "primary_blocker": manifest.get("error_type"), "observed_conditions": conditions,
        "causal_edges": [], "inferred_causes": [], "note": "状态关联不等于因果关系；未独立定位时不推断首要根因。"})
    write_json(output / "evaluation_summary.json", evaluation_summary(manifest, rows, constraints, baseline, paper))


def refresh_summary(output):
    manifest = json.loads((output / "run.json").read_text(encoding="utf-8"))
    def read(name):
        with (output / name).open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))
    rows, constraints = read("claim_evidence.csv"), read("manual_constraints.csv")
    frozen = json.loads((output / "review_template.json").read_text(encoding="utf-8"))
    if digest(frozen) != manifest["review_template_sha256"]:
        raise ValueError("frozen review template changed")
    for actual, original, editable in (
        (rows, frozen["claims"], {"evidence_support", "final_judgement", "reviewer", "reviewed_at", "notes"}),
        (constraints, frozen["constraints"], {"judgement", "reviewer", "reviewed_at", "evidence"}),
    ):
        strip = lambda values: sorted(json.dumps({k: v for k, v in row.items() if k not in editable},
                                                  ensure_ascii=False, sort_keys=True) for row in values)
        if strip(actual) != strip(original):
            raise ValueError("review rows or automatic fields changed; annotations must preserve frozen samples")
    paper = json.loads((output / "paper_retrieval.json").read_text(encoding="utf-8"))
    baseline = manifest.get("baseline_identity")
    summary = evaluation_summary(manifest, rows, constraints, baseline, paper)
    summary["manual_files_sha256"] = {name: sha256((output / name).read_bytes()).hexdigest()
                                      for name in ("claim_evidence.csv", "manual_constraints.csv")}
    write_json(output / "evaluation_summary.json", summary)
    return summary
