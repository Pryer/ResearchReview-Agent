"""隔离 Runner 的失败留档与审查契约；禁止真实 Provider 调用。"""
import copy
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from uuid import uuid4

import pytest

from scripts.e2e_reporting import (
    claim_rows, evaluation_summary, export_reports, paper_report, refresh_summary,
    source_identity, write_csv, CLAIM_COLUMNS,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runs():
    path = ROOT / "data" / "test_runs" / ("e2e_" + uuid4().hex)
    path.mkdir(parents=True)
    yield path
    assert path.resolve().is_relative_to((ROOT / "data" / "test_runs").resolve())
    shutil.rmtree(path)


def run_script(runs, code="", args=(), env=None):
    environment = os.environ.copy()
    environment.update({"CARD_VECTOR_BACKEND": "pgvector", "CARD_VECTOR_DATABASE_URL": "",
                        "LLM_API_KEY": "secret-test-value", "LLM_BASE_URL": "https://example.invalid",
                        "LLM_MODEL": "offline", "DASHSCOPE_API_KEY": "secret-test-value",
                        "RETRIEVAL_EMBEDDING_URL": "https://example.invalid/embedding",
                        "RETRIEVAL_RERANK_URL": "https://example.invalid/rerank", **(env or {})})
    program = """
import sys
import requests
def network_forbidden(*args, **kwargs):
    raise AssertionError('network must not be called')
requests.sessions.Session.request = network_forbidden
from scripts import run_real_e2e as runner
""" + code + "\nraise SystemExit(runner.main(sys.argv[1:]))"
    return subprocess.run([sys.executable, "-X", "utf8", "-c", program,
                           "--task", "T01", "--output-dir", str(runs), *args],
                          cwd=ROOT, env=environment, capture_output=True, text=True, encoding="utf-8", timeout=35)


def load_run(runs):
    files = list(runs.glob("*/run.json"))
    assert len(files) == 1
    return files[0].parent, json.loads(files[0].read_text(encoding="utf-8"))


def test_default_precheck_uses_hybrid_without_connecting_business_database_or_models(runs):
    code = """
def check(settings):
    assert settings.retrieval_ranking_mode == 'hybrid'
    assert settings.card_vector_tenant.startswith('e2e_T01_')
    assert settings.database_url.endswith('/research.db')
    assert settings.retrieval_cache_dir.endswith('/retrieval_cache') or settings.retrieval_cache_dir.endswith('\\\\retrieval_cache')
    assert 'app.database.db' not in sys.modules
    return {'ready': True, 'database_checked': True, 'checks': {'vector_schema': True}}
runner._runtime_precheck = check
"""
    process = run_script(runs, code)
    assert process.returncode == 0, process.stdout + process.stderr
    output, manifest = load_run(runs)
    assert manifest["task_status"] == "precheck_passed"
    assert manifest["primary_pre_registered_run"] is False
    assert manifest["source_tree_unchanged"] is True
    assert manifest["rag_runtime_observed"]["vector_query_successes"] == 0
    assert manifest["rag_runtime_observed"]["vector_backend_used"] is None
    assert "scripts/e2e_reporting.py" in manifest["source_files"]
    assert not (output / "research.db").exists()
    for name in ("result.json", "paper_retrieval.json", "chunk_retrieval.json", "claim_evidence.csv",
                 "manual_constraints.csv", "root_cause_chain.json", "evaluation_summary.json"):
        assert (output / name).exists()
    summary = json.loads((output / "evaluation_summary.json").read_text())
    assert summary["human_support_rate"]["value"] is None
    assert summary["hard_constraint_adherence"]["value"] is None
    assert "secret-test-value" not in (output / "run.json").read_text()


@pytest.mark.parametrize("failure", ["configuration", "precheck"])
def test_early_failure_keeps_reports_and_never_dumps_exception_credentials(runs, failure):
    code = ""
    env = {}
    if failure == "configuration":
        env["CARD_EXTRACTION_MAX_CHUNKS"] = "invalid-secret-test-value"
    else:
        code = "runner._runtime_precheck = lambda settings: {'ready': False, 'checks': {'vector_schema': False}, 'database_checked': False}"
    process = run_script(runs, code, env=env)
    assert process.returncode == 1
    output, manifest = load_run(runs)
    assert manifest["error_stage"] == ("configuration" if failure == "configuration" else "rag_precheck")
    assert manifest["task_status"] == "failed"
    assert manifest["artifact_errors"] == []
    assert "secret-test-value" not in "".join(p.read_text(encoding="utf-8-sig") for p in output.glob("*.json"))
    assert "secret-test-value" not in process.stdout + process.stderr


def test_execute_requires_new_explicit_budget_and_deadline(runs):
    process = run_script(runs, args=("--execute",))
    assert process.returncode == 2
    assert not list(runs.glob("*/run.json"))


def test_execute_freezes_manifest_before_task_and_retains_failure(runs):
    code = """
runner._runtime_precheck = lambda settings: {'ready': True, 'checks': {'vector_schema': True}, 'database_checked': True}
def task(run_id, task, progress):
    import json
    from pathlib import Path
    from app.core.config import get_settings
    from app.database.db import engine
    settings = get_settings()
    record = json.loads((Path(settings.retrieval_cache_dir).parent / 'run.json').read_text(encoding='utf-8'))
    assert record['task_status'] == 'running'
    assert record['configuration']['agent_main_token_budget'] == 1000
    assert record['rag_runtime_precheck']['ready'] is True
    assert str(engine.url).endswith('/research.db')
    raise RuntimeError('postgresql://secret-test-value@example.invalid')
runner._perform_task = task
"""
    process = run_script(runs, code, args=("--execute", "--token-budget", "1000", "--deadline-seconds", "10"))
    assert process.returncode == 1, process.stdout + process.stderr
    output, manifest = load_run(runs)
    assert manifest["error_stage"] == "research_execution"
    assert manifest["error_type"] == "RuntimeError"
    assert manifest["primary_pre_registered_run"] is True
    assert manifest["artifact_errors"] == []
    assert "secret-test-value" not in (output / "run.json").read_text()
    assert "secret-test-value" not in process.stdout + process.stderr


def fixture_state():
    from app.schemas.retrieval_schema import fingerprint
    text = "We do not observe a significant improvement."
    span = {"evidence_id": "p1:e1", "source_type": "full_text", "page": 2, "section": "results",
            "char_start": 0, "char_end": len(text), "offset_basis": "page", "text": text,
            "source_status": "source_verified", "source_text_hash": sha256(text.encode()).hexdigest(),
            "source_document_hash": fingerprint(text)}
    atom = {"text": "The method significantly improves performance.", "citations": ["p1"],
            "support_status": "unsupported", "verification_status": "verified", "evidence_ids": ["p1:e1"]}
    return {"paper_cards": [{"paper_id": "p1", "title": "Study", "evidence_spans": [span]}],
            "parsed_papers": {"p1": {"pages": [{"page": 2, "text": text}]}},
            "claim_verification": {"claims": [{"claim_id": "c1", "factual": True, "atomic_claims": [atom]}]}}


def test_readable_span_does_not_turn_contradicted_claim_into_human_support():
    rows = claim_rows(fixture_state(), {"status": "blocked", "core_deliverables": ["research_background"]})
    assert rows[0]["span_readable"] == "true"
    assert rows[0]["automatic_support"] == "unsupported"
    assert rows[0]["final_judgement"] == "not_reviewed"
    assert rows[0]["text_scope"] == "quarantined_or_unpublished"
    summary = evaluation_summary({"task_status": "blocked"}, rows, [])
    assert summary["human_support_rate"]["value"] is None
    assert summary["citation_readback_rate"]["value"] == 1


@pytest.mark.parametrize("change, expected", [("unavailable", "unknown"), ("offset", "false"), ("version", "false"), ("uncited", "unknown")])
def test_readback_keeps_missing_corrupt_and_unlinked_sources_distinct(change, expected):
    state = fixture_state()
    if change == "unavailable":
        state["parsed_papers"] = {}
    elif change == "offset":
        state["paper_cards"][0]["evidence_spans"][0]["char_start"] = 1
    elif change == "version":
        state["paper_cards"][0]["evidence_spans"][0]["source_document_hash"] = "old"
    else:
        state["claim_verification"]["claims"][0]["atomic_claims"][0]["citations"] = ["missing"]
    row = claim_rows(state, {})[0]
    assert row["span_readable"] == expected


def test_joint_citations_count_one_atomic_claim_and_keep_all_links():
    state = fixture_state()
    state["claim_verification"]["claims"][0]["atomic_claims"][0]["citations"].append("missing")
    rows = claim_rows(state, {})
    assert len(rows) == 2
    for row in rows:
        row.update(final_judgement="supported", reviewer="human", reviewed_at="2026-10-29")
    summary = evaluation_summary({"task_status": "blocked"}, rows, [])
    assert summary["human_support_rate"] == {"numerator": 1, "denominator": 1, "value": 1.0}
    assert summary["citation_readback_rate"]["denominator"] == 2
    rows[1]["final_judgement"] = "contradicted"
    summary = evaluation_summary({"task_status": "blocked"}, rows, [])
    assert summary["manual_review_errors"]
    assert summary["human_support_rate"]["value"] is None


def test_uncertain_is_reviewed_but_not_supported_and_blocked_is_not_automatically_correct():
    rows = claim_rows(fixture_state(), {})
    rows[0].update(final_judgement="uncertain", reviewer="human", reviewed_at="2026-10-29")
    constraint = {"constraint_id": "c", "judgement": "correctly_blocked", "reviewer": "", "reviewed_at": "", "evidence": ""}
    summary = evaluation_summary({"task_status": "blocked"}, rows, [constraint])
    assert summary["human_review_coverage"]["value"] == 1
    assert summary["human_support_rate"]["value"] == 0
    assert summary["human_uncertain_claims"] == 1
    assert summary["hard_constraint_adherence"]["value"] is None


def test_known_item_report_separates_candidate_ranking_and_admission():
    target = {"doi": "10.1/a", "title": "Study", "year": 2020}
    correct = {**target, "paper_id": "p1"}
    conflict = {**correct, "doi": "10.1/b"}
    events = [{"sequence": 1, "kind": "paper_ranking", "candidates": [correct, conflict], "ranked": [conflict, correct]},
              {"sequence": 2, "kind": "paper_admission", "candidates": [correct], "admitted": []}]
    report = paper_report([target], events)
    first, second = [r["targets"][0] for r in report["rounds"]]
    assert first["candidate_hit"] is True
    assert len(first["candidate_matches"]) == 1
    assert first["rank_in_observed_window"] == 2
    assert first["reciprocal_rank"] == 0.5
    assert first["admitted"] is None
    assert second["admitted"] is False
    assert second["reciprocal_rank"] is None
    assert report["metric_scope"] == "known_items_only_not_topic_recall"


def test_candidate_checkpoint_before_ranking_does_not_invent_rank_or_admission():
    target = {"arxiv_id": "2005.11401", "title": "Known item", "year": 2020}
    report = paper_report([target], [{"sequence": 1, "kind": "paper_candidates_checkpoint", "candidates": [target]}])
    row = report["rounds"][0]["targets"][0]
    assert row["candidate_hit"] is True
    assert row["rank_in_observed_window"] is None
    assert row["reciprocal_rank"] is None
    assert row["admitted"] is None


def test_source_inventory_includes_staged_and_untracked_code_but_excludes_private_env(tmp_path):
    def git(*args):
        return subprocess.run(["git", *args], cwd=tmp_path, capture_output=True, check=True)
    (tmp_path / "app").mkdir()
    (tmp_path / "scripts").mkdir()
    tracked = tmp_path / "app" / "tracked.py"
    tracked.write_text("x = 1")
    git("init")
    git("add", "app")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "fixture")
    before = source_identity(tmp_path)
    tracked.write_text("x = 2")
    git("add", "app")
    (tmp_path / "scripts" / "new.py").write_text("y = 3")
    (tmp_path / ".env").write_text("SECRET=private")
    after = source_identity(tmp_path)
    assert before["tracked_diff_sha256"] != after["tracked_diff_sha256"]
    assert before["source_tree_sha256"] != after["source_tree_sha256"]
    assert "scripts/new.py" in after["source_files"]
    assert ".env" not in after["source_files"]


def test_export_has_no_successful_query_claim_without_observed_query_and_protects_review_samples(tmp_path):
    manifest = {"task_status": "blocked", "targets": [], "task": {"request": {"user_query": "strict task"}}}
    state = fixture_state()
    export_reports(tmp_path, manifest, state, {"status": "blocked"}, [])
    assert manifest["rag_runtime_observed"]["vector_query_successes"] == 0
    assert manifest["rag_runtime_observed"]["vector_backend_used"] is None
    (tmp_path / "run.json").write_text(json.dumps(manifest))
    assert refresh_summary(tmp_path)["human_support_rate"]["value"] is None
    with (tmp_path / "claim_evidence.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["span_readable"] = "false"
    write_csv(tmp_path / "claim_evidence.csv", CLAIM_COLUMNS, rows)
    with pytest.raises(ValueError, match="frozen samples"):
        refresh_summary(tmp_path)


def test_historical_baseline_without_frozen_config_is_not_comparable():
    summary = evaluation_summary({"task_status": "blocked"}, [], [], {"commit_sha": "old"})
    assert summary["baseline_comparison"]["comparable"] is False
    assert "candidate_pool_not_frozen_or_different" in summary["baseline_comparison"]["reasons"]


def test_observer_records_actual_queries_and_restores_methods_even_after_failure(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts.e2e_observation import RunObserver
    from app.storage.pgvector_store import PgVectorStore
    from app.services import retrieval_ranking_service as ranking
    def search(*args, **kwargs):
        return [(1, 0.9)]
    def rank(state, **kwargs):
        return state["candidate_papers"], {}
    monkeypatch.setattr(PgVectorStore, "search", search)
    monkeypatch.setattr(ranking, "rank_candidates", rank)
    observer = RunObserver(tmp_path / "events.jsonl")
    with pytest.raises(RuntimeError):
        with observer.installed():
            ranking.rank_candidates({"candidate_papers": [{"paper_id": "p1"}]})
            PgVectorStore.search(None, SimpleNamespace(paper_id="p1"), "snapshot", 1, [], 5)
            raise RuntimeError("stop")
    assert PgVectorStore.search is search
    assert ranking.rank_candidates is rank
    assert any(e["kind"] == "vector_query" and e["outcome"] == "succeeded" for e in observer.events)


def test_uncited_factual_claim_is_not_counted_as_a_citation_link():
    state = fixture_state()
    state["claim_verification"]["claims"][0]["atomic_claims"][0]["citations"] = []
    summary = evaluation_summary({"task_status": "blocked"}, claim_rows(state, {}), [])
    assert summary["uncited_atomic_claims"] == 1
    assert summary["citation_readback_rate"]["value"] is None
    assert summary["human_not_judged_claims"] == 1


def test_rerun_retains_primary_record_and_cannot_relax_original_task(runs):
    code = """
runner._runtime_precheck = lambda settings: {'ready': True, 'checks': {}, 'database_checked': True}
runner._perform_task = lambda *args: {'status': 'blocked', 'errors': [{'code': 'test_control'}]}
"""
    options = ("--execute", "--token-budget", "1000", "--deadline-seconds", "10")
    first = run_script(runs, code, options)
    assert first.returncode == 0, first.stdout + first.stderr
    output, primary = load_run(runs)
    original = (output / "run.json").read_bytes()
    second = run_script(runs, code, (*options, "--rerun-of", str(output)))
    assert second.returncode == 0, second.stdout + second.stderr
    manifests = [json.loads(p.read_text(encoding="utf-8")) for p in runs.glob("*/run.json")]
    rerun = next(m for m in manifests if m["run_id"] != primary["run_id"])
    assert rerun["rerun_of"] == primary["run_id"]
    assert rerun["experiment_id"] == primary["experiment_id"]
    assert rerun["primary_pre_registered_run"] is False
    assert (output / "run.json").read_bytes() == original
    task = copy.deepcopy(primary["task"])
    task["request"]["user_query"] = "Relax the original constraints"
    changed = runs / "changed.jsonl"
    changed.write_text(json.dumps(task), encoding="utf-8")
    rejected = run_script(runs, code, (*options, "--rerun-of", str(output), "--tasks", str(changed)))
    assert rejected.returncode == 2
    assert len(list(runs.glob("*/run.json"))) == 2


def test_manual_summary_accepts_annotations_but_never_automatically_signs_acceptance(tmp_path):
    manifest = {"task_status": "blocked", "targets": [], "task": {"request": {"user_query": "strict"}}}
    export_reports(tmp_path, manifest, fixture_state(), {}, [])
    (tmp_path / "run.json").write_text(json.dumps(manifest))
    with (tmp_path / "claim_evidence.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0].update(final_judgement="contradicted", evidence_support="contradicted", reviewer="human", reviewed_at="2026-10-29")
    write_csv(tmp_path / "claim_evidence.csv", CLAIM_COLUMNS, rows)
    summary = refresh_summary(tmp_path)
    assert summary["human_support_rate"] == {"numerator": 0, "denominator": 1, "value": 0.0}
    assert summary["human_review_coverage"]["value"] == 1
    assert summary["independent_acceptance"] == "pending_human_review"
    assert not summary["manual_review_errors"]


def test_published_fixture_exports_claim_chain_but_not_human_acceptance(runs):
    frozen = json.dumps(fixture_state())
    code = """
import json
runner._runtime_precheck = lambda settings: {'ready': True, 'checks': {}, 'database_checked': True}
runner._perform_task = lambda *args: {'status': 'completed', 'body': 'Synthetic contract fixture.',
    'quality_gate': {'passed': True}, 'research_state': json.loads(FIXTURE)}
""".replace("FIXTURE", repr(frozen))
    process = run_script(runs, code, ("--execute", "--token-budget", "1000", "--deadline-seconds", "10"))
    assert process.returncode == 0, process.stdout + process.stderr
    output, manifest = load_run(runs)
    with (output / "claim_evidence.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["text_scope"] == "published"
    assert rows[0]["span_readable"] == "true"
    assert rows[0]["automatic_support"] == "unsupported"
    summary = json.loads((output / "evaluation_summary.json").read_text())
    assert summary["human_support_rate"]["value"] is None
    assert summary["independent_acceptance"] == "pending_human_review"
    assert manifest["requirement_satisfied"] is None


def test_incomplete_constraint_review_never_uses_a_smaller_denominator():
    constraints = [{"constraint_id": "year", "judgement": "passed", "reviewer": "human", "reviewed_at": "date", "evidence": "checked years"},
                   {"constraint_id": "count", "judgement": "not_reviewed", "reviewer": "", "reviewed_at": "", "evidence": ""}]
    summary = evaluation_summary({"task_status": "completed"}, [], constraints)
    assert summary["hard_constraint_adherence"] == {"numerator": 1, "denominator": 2, "value": None}
