"""路线证据恢复闭环与 Claim Evidence Gate 测试。"""

from __future__ import annotations

import json

from app.agent.claim_plan import apply_claim_budget, enforce_claim_evidence_gate
from app.agent.evidence_recovery import (
    _source_health,
    decide_recovery,
    diagnose_evidence_gaps,
    query_novelty,
)
from app.agent.graph import _run_route_evidence_recovery
from app.agent.nodes import validate_routes_node
from app.core.config import Settings
from app.schemas.recovery_schema import RecoveryAction


def _route_state() -> dict:
    return {
        "user_query": "调研少样本视频动作识别",
        "topic": "少样本视频动作识别",
        "canonical_topic": "few-shot video action recognition",
        "start_year": 2024,
        "end_year": 2026,
        "keywords": ["few-shot video action recognition"],
        "searched_keywords": ["few-shot video action recognition"],
        "provisional_framework": {
            "provisional_routes": [{
                "route_id": "route_temporal",
                "name": "时序对齐",
                "research_question": "少样本条件下如何进行时序对齐？",
                "core_concepts": ["few-shot", "temporal alignment", "video action"],
                "search_queries": ["few-shot temporal alignment action recognition"],
            }],
        },
        "route_decisions": [{
            "route_id": "route_temporal",
            "route_name": "时序对齐",
            "diagnosis": "INSUFFICIENT_EVIDENCE",
            "action": "DROP",
            "scores": {
                "paper_count": 0,
                "core_paper_count": 0,
                "mean_route_fit": 0.0,
                "supporting_threshold": 0.08,
            },
        }],
        "route_validation_report": {
            "coverage": {"evidence_understood_rate": 0.1},
            "assignment_map": {},
        },
        "source_diagnostics": [{"source": "openalex", "status": "success"}],
        "paper_cards": [],
        "paper_details": [],
        "candidate_papers": [],
        "ranked_papers": [],
        "required_reference_count": 2,
        "max_papers": 2,
        "retrieval_target": 6,
        "generation_limit": 4,
        "steps": [],
        "errors": [],
    }


class DiagnosisLLM:
    def complete(self, prompt: str, **kwargs) -> str:
        assert kwargs.get("operation") == "diagnose_evidence_gaps"
        return json.dumps({
            "route_diagnoses": [{
                "route_id": "route_temporal",
                "gap_type": "SEARCH_COVERAGE_GAP",
                "reason": "缺少 temporal matching 相关术语的覆盖",
                "suggested_queries": ["temporal matching few-shot video recognition"],
                "missing_constraints": ["few-shot"],
                "exclusion_candidates": ["fully supervised"],
            }],
            "scope_revision_recommended": False,
            "notes": [],
        })


def test_query_novelty_rejects_near_duplicate_variants():
    novelty = query_novelty(
        "few shot video action recognition",
        ["few-shot action recognition"],
    )
    assert novelty < 0.35


def test_source_health_does_not_treat_empty_results_as_healthy():
    assert _source_health({"source_diagnostics": [
        {"source": "cnki", "status": "success"},
        {"source": "openalex", "status": "empty"},
    ]}) == "partial"
    assert _source_health({"source_diagnostics": [
        {"source": "openalex", "status": "empty"},
        {"source": "semantic_scholar", "status": "empty"},
    ]}) == "empty"
    assert _source_health({"source_diagnostics": [
        {"source": "openalex", "status": "success"},
    ]}) == "healthy"


def test_claim_budget_prefers_stronger_evidence_and_preserves_each_route():
    plans = []
    for route_index in range(3):
        claims = [
            {
                "claim_id": f"r{route_index}:c{claim_index}",
                "support_level": "strong" if claim_index == 5 else "single",
            }
            for claim_index in range(10)
        ]
        plans.append({
            "route_id": f"r{route_index}",
            "route_name": f"路线{route_index}",
            "claims": claims,
            "total_claims": len(claims),
            "single_evidence_claims": 9,
            "strong_plus_claims": 1,
        })

    limited, report = apply_claim_budget(
        plans, required_reference_count=10, minimum_per_route=2,
    )

    assert report == {
        "total_before": 30, "total_after": 10, "budget": 10, "dropped": 20,
    }
    assert all(plan["claims"] for plan in limited)
    assert all(
        any(claim["support_level"] == "strong" for claim in plan["claims"])
        for plan in limited
    )


def test_recovery_control_limits_are_configurable_and_clamped():
    settings = Settings(
        evidence_recovery_max_rounds=-1,
        evidence_recovery_min_query_novelty=1.5,
        evidence_recovery_scope_gap_ratio=-0.2,
    )

    assert settings.evidence_recovery_max_rounds == 0
    assert settings.evidence_recovery_min_query_novelty == 1.0
    assert settings.evidence_recovery_scope_gap_ratio == 0.0


def test_gap_diagnosis_uses_metrics_as_base_and_llm_for_terms():
    report = diagnose_evidence_gaps(_route_state(), llm=DiagnosisLLM())

    assert report.needs_recovery is True
    assert report.diagnosis_source == "hybrid"
    assert report.affected_route_ids == ["route_temporal"]
    assert report.gaps[0].suggested_queries == [
        "temporal matching few-shot video recognition"
    ]


def test_recovery_controller_stops_at_round_budget():
    state = _route_state()
    state["recovery_round"] = 2
    report = diagnose_evidence_gaps(state, llm=DiagnosisLLM())

    decision = decide_recovery(
        state,
        report,
        max_rounds=2,
        max_route_attempts=2,
        min_query_novelty=0.35,
        max_scope_revisions=1,
        max_queries=4,
    )

    assert decision.action == RecoveryAction.DEGRADE
    assert decision.status.value == "EXHAUSTED"


def test_recovery_controller_does_not_rewrite_queries_when_sources_failed():
    state = _route_state()
    state["source_diagnostics"] = [{"source": "openalex", "status": "failed"}]
    report = diagnose_evidence_gaps(state, llm=DiagnosisLLM())

    decision = decide_recovery(
        state,
        report,
        max_rounds=2,
        max_route_attempts=2,
        min_query_novelty=0.35,
        max_scope_revisions=1,
        max_queries=4,
    )

    assert decision.action == RecoveryAction.DEGRADE
    assert "数据源不可用" in decision.reason


def test_claim_gate_weakens_overclaim_and_drops_unbound_optional_claim():
    cards = [{
        "paper_id": "p1",
        "source": "openalex",
        "field_claims": {
            "results": [{
                "claim": "方法在该数据集上提升了准确率",
                "evidence_id": "p1:results:1",
                "explicitly_reported": True,
            }],
        },
    }]
    plans = [{
        "route_id": "r1",
        "route_name": "方法",
        "claims": [
            {
                "claim_id": "c1",
                "claim_text": "方法在该数据集上提升了准确率",
                "claim_type": "finding",
                "evidence_ids": ["p1:results:1"],
                "evidence_count": 7,
                "support_level": "established",
                "allowed_language": "已成为共识",
            },
            {
                "claim_id": "c2",
                "claim_text": "该方向已经形成统一趋势",
                "claim_type": "trend",
                "evidence_ids": ["missing:evidence"],
                "evidence_count": 4,
                "support_level": "strong",
                "allowed_language": "多项研究表明",
            },
        ],
        "total_claims": 2,
        "single_evidence_claims": 0,
        "strong_plus_claims": 2,
    }]

    repaired, report = enforce_claim_evidence_gate(plans, cards)

    assert report["weakened_claims"] == 1
    assert report["dropped_claims"] == 1
    assert repaired[0]["total_claims"] == 1
    assert repaired[0]["claims"][0]["support_level"] == "single"


def test_validate_routes_node_preserves_full_validation_report(monkeypatch):
    expected = {
        "validated_routes": [],
        "decisions": [],
        "coverage": {"evidence_understood_rate": 0.5},
        "assignment_map": {"p1": {"type": "single_route"}},
        "unassigned_paper_ids": [],
    }
    monkeypatch.setattr(
        "app.agent.provisional_routes.validate_routes_against_evidence",
        lambda *args, **kwargs: expected,
    )
    state = {
        "provisional_framework": {"provisional_routes": [{"route_id": "r1"}]},
        "paper_cards": [{"paper_id": "p1"}],
        "steps": [],
        "errors": [],
    }

    validate_routes_node(state, llm=None)

    assert state["route_validation_report"] == expected


def test_recovery_loop_runs_incrementally_and_stops_when_route_recovers(monkeypatch):
    state = _route_state()
    llm = DiagnosisLLM()
    monkeypatch.setattr("app.agent.recovery_loop._get_llm", lambda: llm)

    def fake_search(current, should_cancel=None):
        current["last_search_new_results"] = 1
        current["candidate_papers"] = [{"paper_id": "p-new", "title": "Temporal Matching"}]
        return current

    def fake_fetch(current, should_cancel=None):
        current["paper_details"] = [{"paper_id": "p-new", "title": "Temporal Matching"}]
        current["incremental_new_paper_ids"] = ["p-new"]
        return current

    def fake_extract(current, llm=None):
        current["paper_cards"] = [{"paper_id": "p-new", "title": "Temporal Matching"}]
        return current

    def fake_validate(current, llm=None):
        current["validated_routes"] = [{
            "route_id": "route_temporal",
            "name": "时序对齐",
            "paper_ids": ["p-new"],
            "core_paper_ids": ["p-new"],
        }]
        current["route_decisions"] = [{
            "route_id": "route_temporal",
            "diagnosis": "NICHE_ROUTE",
            "action": "KEEP",
            "scores": {"core_paper_count": 1},
        }]
        current["route_validation_report"] = {
            "coverage": {"evidence_understood_rate": 1.0},
            "assignment_map": {"p-new": {"type": "single_route"}},
        }
        return current

    monkeypatch.setattr("app.agent.recovery_loop.search_node", fake_search)
    def fake_rank(current, llm=None):
        assert llm is None
        return current

    monkeypatch.setattr("app.agent.recovery_loop.rank_node", fake_rank)
    monkeypatch.setattr("app.agent.recovery_loop.fetch_detail_node", fake_fetch)
    monkeypatch.setattr("app.agent.recovery_loop.download_pdf_node", lambda current: current)
    monkeypatch.setattr("app.agent.recovery_loop.should_parse_pdf", lambda current: False)
    monkeypatch.setattr("app.agent.recovery_loop.extract_card_node", fake_extract)
    monkeypatch.setattr("app.agent.recovery_loop.validate_routes_node", fake_validate)

    _run_route_evidence_recovery(state)

    assert state["recovery_round"] == 1
    assert state["evidence_recovery_status"] == "NOT_REQUIRED"
    assert state["recovery_history"][0]["new_relevant_evidence"] == 1
    assert state["recovery_history"][0]["stop_reason"] == "route evidence gaps resolved"
    assert "incremental_retrieval" not in state
    assert "temporal matching few-shot video recognition" in state["keywords"]
    assert state["evidence_gap_report"]["needs_recovery"] is False
    assert state["evidence_gap_report"]["evidence_snapshot_version"] == state["evidence_snapshot_version"]
    assert state["evidence_gap_report"]["evidence_snapshot_fingerprint"] == state["evidence_snapshot_fingerprint"]


def test_recovery_does_not_treat_off_target_evidence_as_progress(monkeypatch):
    """补来的证据全落在已达标路线时，缺证路线仍未达标，不能判为已解决。"""
    state = _route_state()
    state["core_deliverables"] = ["research_status"]
    state["required_reference_count"] = 12
    state["provisional_framework"]["provisional_routes"].append({
        "route_id": "route_other",
        "name": "已达标方向",
        "research_question": "另一方向？",
        "core_concepts": ["other concept"],
        "search_queries": ["other direction query"],
    })
    llm = DiagnosisLLM()
    monkeypatch.setattr("app.agent.recovery_loop._get_llm", lambda: llm)

    def fake_search(current, should_cancel=None):
        current["last_search_new_results"] = 1
        return current

    def fake_validate(current, llm=None):
        # 新证据全部落到 route_other，缺证的 route_temporal 仍是 0 篇。
        current["validated_routes"] = [
            {
                "route_id": "route_temporal",
                "name": "时序对齐",
                "paper_ids": [],
                "core_paper_ids": [],
            },
            {
                "route_id": "route_other",
                "name": "已达标方向",
                "paper_ids": ["p-off"],
                "core_paper_ids": ["p-off"],
            },
        ]
        current["route_decisions"] = [{
            "route_id": "route_temporal",
            "status": "WEAK",
            "action": "TARGETED_SEARCH",
            "diagnosis": "INSUFFICIENT_EVIDENCE",
            "scores": {"core_paper_count": 0},
        }]
        current["route_validation_report"] = {
            "coverage": {"evidence_understood_rate": 1.0},
            "assignment_map": {"p-off": {"type": "single_route"}},
            "validated_routes": current["validated_routes"],
        }
        return current

    monkeypatch.setattr("app.agent.recovery_loop.search_node", fake_search)
    monkeypatch.setattr("app.agent.recovery_loop.rank_node", lambda current, llm=None: current)
    monkeypatch.setattr(
        "app.agent.recovery_loop.fetch_detail_node",
        lambda current, should_cancel=None: current,
    )
    monkeypatch.setattr("app.agent.recovery_loop.download_pdf_node", lambda current: current)
    monkeypatch.setattr("app.agent.recovery_loop.should_parse_pdf", lambda current: False)
    monkeypatch.setattr(
        "app.agent.recovery_loop.extract_card_node",
        lambda current, llm=None: current.update(
            {"paper_cards": [{"paper_id": "p-off", "title": "Off target"}]}
        ) or current,
    )
    monkeypatch.setattr("app.agent.recovery_loop.validate_routes_node", fake_validate)

    _run_route_evidence_recovery(state)

    assert state["evidence_recovery_status"] != "NOT_REQUIRED"
    progress = state["route_recovery_progress"]["route_temporal"]
    assert progress["core_after"] == 0
    assert progress["new_relevant"] == 0
    assert progress["target"] > 0
    # 未达标路线如实落盘，供质量门禁 warning 使用。
    deficits = {item["route_id"] for item in state["route_evidence_deficits"]}
    assert "route_temporal" in deficits
