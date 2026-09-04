"""Frozen Route–Recovery gold cases.

The cases in this file are the acceptance contract for Route Validator v2.
Algorithms and thresholds may change; these inputs and expected behaviours must
not be rewritten merely to accommodate a new implementation.
"""

from __future__ import annotations

from copy import deepcopy

from app.agent.claim_plan import enforce_claim_evidence_gate
from app.agent.evidence_recovery import decide_recovery
from app.agent.provisional_routes import validate_routes_against_evidence
from app.agent.route_validator import guard_anchor_expansions
from app.schemas.recovery_schema import (
    EvidenceGapReport,
    RecoveryAction,
    RecoveryStatus,
    RouteEvidenceGap,
    RouteGapType,
)

GOLD_CASE_VERSION = "RR-GOLD-1.0.0"


def _route(route_id: str, name: str, anchor: str) -> dict:
    return {
        "route_id": route_id,
        "name": name,
        "research_question": f"如何通过{anchor}完成少样本视频识别？",
        "route_role": "interpretation",
        "core_concepts": [anchor, "few-shot video recognition", "support-query learning"],
        "semantic_anchors": [anchor, "few-shot video recognition", "support-query matching"],
        "method_concepts": [anchor, "metric learning"],
        "task_anchors": ["few-shot video recognition", "support-query classification"],
        "negative_anchors": ["fully supervised action localization"],
        "search_queries": [f"{anchor} few-shot video recognition metric learning"],
        "inclusion_criteria": [f"论文明确使用{anchor}解决少样本视频识别"],
        "exclusion_criteria": ["仅研究 fully supervised action localization"],
        "boundary_note": "与纯动作定位方法区分，必须包含 support-query 学习。",
    }


def _card(paper_id: str, text: str, method: str = "") -> dict:
    return {
        "paper_id": paper_id,
        "title": text,
        "abstract": f"We study few-shot video recognition with {text}. Support-query classification is evaluated.",
        "research_problem": "few-shot video recognition",
        "method": method or text,
        "quality_status": "partial",
        "evidence_source": "abstract",
        "field_claims": {
            "method": [{"claim": f"The method uses {method or text} for support-query matching."}]
        },
    }


def _decision(result: dict, route_id: str) -> dict:
    return next(item for item in result["decisions"] if item["route_id"] == route_id)


def test_rr01_chinese_route_matches_english_evidence_through_bilingual_anchors():
    route = _route("RR01", "基于时序对齐的度量学习方法", "temporal alignment")
    cards = [
        _card("p1", "Temporal matching for few-shot action recognition", "temporal alignment metric learning"),
        _card("p2", "Sequence alignment of support and query clips", "sequence alignment metric learning"),
        _card("p3", "Support-query temporal correspondence", "temporal alignment prototype matching"),
    ]

    result = validate_routes_against_evidence([route], cards, llm=None)
    decision = _decision(result, "RR01")

    assert decision["status"] == "KEEP"
    assert decision["route_validity"]["structurally_valid"] is True
    assert decision["evidence_sufficiency"]["sufficient"] is True
    assert all(
        "semantic_similarity" in result["feature_matrix"]["RR01"][paper_id]
        for paper_id in ("p1", "p2", "p3")
    )


def test_rr02_synonymous_expression_matches_anchors_without_route_title_overlap():
    route = _route("RR02", "基于类别中心比较的识别路线", "episodic nearest-centroid classification")
    route["semantic_anchors"].extend(["class prototype comparison", "prototype-based matching"])
    cards = [
        _card("p1", "Class prototype comparison for novel actions", "prototype-based matching"),
        _card("p2", "Episodic nearest-centroid video classification"),
        _card("p3", "Prototype matching between support and query videos"),
    ]

    result = validate_routes_against_evidence([route], cards, llm=None)
    decision = _decision(result, "RR02")

    assert decision["status"] == "KEEP"
    assert decision["scores"]["core_paper_count"] >= 3
    assert max(
        row["lexical_anchor_score"]
        for row in result["feature_matrix"]["RR02"].values()
    ) > 0


def test_rr03_valid_route_with_sparse_evidence_is_weak_not_dropped():
    route = _route("RR03", "基于时序对齐的度量学习方法", "temporal alignment")
    result = validate_routes_against_evidence(
        [route], [_card("p1", "Temporal alignment for few-shot video recognition")], llm=None
    )
    decision = _decision(result, "RR03")

    assert decision["route_validity"]["structurally_valid"] is True
    assert decision["evidence_sufficiency"]["sufficient"] is False
    assert decision["status"] == "WEAK"
    assert decision["action"] == "TARGETED_SEARCH"
    assert result["validated_routes"][0]["route_id"] == "RR03"


def test_rr04_evidence_rich_but_invalid_definition_requests_route_revision():
    route = {
        "route_id": "RR04",
        "name": "按年份分类",
        "research_question": "",
        "route_role": "",
        "core_concepts": ["prototype matching"],
        "semantic_anchors": ["prototype matching", "class prototype", "nearest centroid"],
        "method_concepts": ["prototype matching"],
        "task_anchors": [],
        "negative_anchors": [],
        "search_queries": ["prototype matching few-shot action recognition"],
        "inclusion_criteria": [],
        "exclusion_criteria": [],
        "boundary_note": "",
    }
    cards = [
        _card(f"p{i}", f"Prototype matching for novel action class {i}", "prototype matching")
        for i in range(1, 4)
    ]

    decision = _decision(validate_routes_against_evidence([route], cards, llm=None), "RR04")

    assert decision["route_validity"]["structurally_valid"] is False
    assert decision["evidence_sufficiency"]["sufficient"] is True
    assert decision["status"] == "WEAK"
    assert decision["action"] == "ROUTE_REVISION"


def test_rr05_five_coherent_routes_with_no_evidence_do_not_all_drop():
    routes = [
        _route(f"RR05-{index}", f"候选路线{index}", f"mechanism anchor {index}")
        for index in range(1, 6)
    ]

    result = validate_routes_against_evidence(routes, [], llm=None)

    assert all(item["status"] == "WEAK" for item in result["decisions"])
    assert result["coverage"]["provisional_route_survival_rate"] == 1.0
    assert result["coverage"]["route_validator_recheck"] is True
    assert len(result["validated_routes"]) == 5


def test_rr06_incremental_evidence_moves_weak_route_to_keep():
    route = _route("RR06", "基于时序对齐的度量学习方法", "temporal alignment")
    first = validate_routes_against_evidence(
        [route], [_card("p1", "Temporal alignment for few-shot video recognition")], llm=None
    )
    second = validate_routes_against_evidence(
        [route], [
            _card("p1", "Temporal alignment for few-shot video recognition"),
            _card("p2", "Temporal matching of support-query videos", "temporal alignment metric learning"),
            _card("p3", "Sequence alignment for novel action recognition", "temporal alignment metric learning"),
        ], llm=None,
    )

    assert _decision(first, "RR06")["status"] == "WEAK"
    assert _decision(second, "RR06")["status"] == "KEEP"
    assert (
        _decision(first, "RR06")["route_validity"]
        == _decision(second, "RR06")["route_validity"]
    )


def test_rr07_recovery_without_gain_degrades_at_budget_boundary():
    report = EvidenceGapReport(
        needs_recovery=True,
        affected_route_ids=["RR07"],
        gaps=[RouteEvidenceGap(
            route_id="RR07",
            gap_type=RouteGapType.SEARCH_COVERAGE_GAP,
            reason="valid route lacks evidence",
            suggested_queries=["novel recovery query"],
        )],
    )
    decision = decide_recovery(
        {"recovery_round": 2}, report,
        max_rounds=2, max_route_attempts=2, min_query_novelty=0.2,
        max_scope_revisions=1, max_queries=4,
    )

    assert decision.action == RecoveryAction.DEGRADE
    assert decision.status == RecoveryStatus.EXHAUSTED


def test_rr08_claim_stronger_than_evidence_is_weakened():
    plans = [{
        "route_id": "RR08",
        "claims": [{
            "claim_id": "c1",
            "claim": "All studies conclusively prove the method always outperforms every baseline.",
            "claim_type": "finding",
            "support_level": "strong",
            "evidence_ids": ["p1:findings:0"],
            "is_optional": False,
        }],
    }]
    cards = [{
        "paper_id": "p1",
            "field_claims": {"findings": [{
            "evidence_id": "p1:findings:0",
            "claim_id": "p1:findings:0",
            "claim": "The method improved accuracy on one benchmark.",
        }]},
    }]

    repaired, report = enforce_claim_evidence_gate(deepcopy(plans), cards)

    assert report["weakened_claims"] == 1
    assert repaired[0]["claims"][0]["support_level"] == "single"


def test_rr09_route_below_derived_target_enters_gap_report():
    """路线篇数低于交付物派生目标时必须进入缺口报告并带上目标与缺口。"""
    from app.agent.evidence_recovery import _deterministic_route_gaps

    state = {
        "core_deliverables": ["research_status"],
        "required_reference_count": 12,
        "provisional_framework": {"provisional_routes": [_route("RR09", "时序对齐", "temporal alignment")]},
        "validated_routes": [{
            "route_id": "RR09",
            "name": "时序对齐",
            "core_paper_ids": ["p1", "p2"],
            "supporting_paper_ids": [],
        }],
        "route_decisions": [{
            "route_id": "RR09",
            "route_name": "时序对齐",
            "status": "KEEP",
            "action": "KEEP",
            "diagnosis": "STRONG_ROUTE",
            "scores": {"core_paper_count": 2, "paper_count": 2, "mean_route_fit": 0.4},
        }],
        "searched_keywords": ["few-shot video recognition"],
    }

    gaps = _deterministic_route_gaps(state)

    assert len(gaps) == 1
    gap = gaps[0]
    # 目标由 required_reference_count 派生，高于 route_min_core_evidence
    assert gap.target_core_evidence > 2
    assert gap.core_evidence_count == 2
    assert gap.core_evidence_deficit == gap.target_core_evidence - 2


def test_rr10_split_sub_route_gap_is_not_missed():
    """SPLIT 产出的子路线不在 provisional_routes 中，此前必然漏诊。"""
    from app.agent.evidence_recovery import _deterministic_route_gaps

    state = {
        "core_deliverables": ["research_status"],
        "required_reference_count": 12,
        "provisional_framework": {"provisional_routes": [_route("RR10", "父路线", "temporal alignment")]},
        "validated_routes": [{
            "route_id": "RR10_S1",
            "name": "子路线一",
            "core_paper_ids": [],
            "supporting_paper_ids": [],
        }],
        "route_decisions": [{
            "route_id": "RR10_S1",
            "route_name": "子路线一",
            "status": "WEAK",
            "action": "TARGETED_SEARCH",
            "diagnosis": "INSUFFICIENT_EVIDENCE",
            "scores": {"core_paper_count": 0, "paper_count": 0, "mean_route_fit": 0.0},
        }],
        "searched_keywords": [],
    }

    gaps = _deterministic_route_gaps(state)

    assert [gap.route_id for gap in gaps] == ["RR10_S1"]
    assert gaps[0].core_evidence_deficit > 0


def test_rr11_every_gap_route_receives_at_least_one_query():
    """三条缺口路线时查询名额按缺口轮询分配，不能有路线零查询。"""
    report = EvidenceGapReport(
        needs_recovery=True,
        affected_route_ids=["A", "B", "C"],
        gaps=[
            RouteEvidenceGap(
                route_id="A", gap_type=RouteGapType.SEARCH_COVERAGE_GAP,
                reason="gap", core_evidence_deficit=5,
                suggested_queries=["query alpha one", "query alpha two", "query alpha three"],
            ),
            RouteEvidenceGap(
                route_id="B", gap_type=RouteGapType.SEARCH_COVERAGE_GAP,
                reason="gap", core_evidence_deficit=3,
                suggested_queries=["query beta one", "query beta two"],
            ),
            RouteEvidenceGap(
                route_id="C", gap_type=RouteGapType.SEARCH_COVERAGE_GAP,
                reason="gap", core_evidence_deficit=1,
                suggested_queries=["query gamma one"],
            ),
        ],
    )

    decision = decide_recovery(
        {}, report,
        max_rounds=2, max_route_attempts=2, min_query_novelty=0.2,
        max_scope_revisions=1, max_queries=3,
    )

    assert decision.action == RecoveryAction.TARGETED_SEARCH
    assert set(decision.route_query_allocation) == {"A", "B", "C"}
    assert all(queries for queries in decision.route_query_allocation.values())


def test_rr14_route_query_regeneration_survives_novelty_gate_without_llm():
    """无 LLM 时路线概念子集组合必须能产生通过新颖度门槛的补检索查询。"""
    from app.agent.evidence_recovery import _regenerated_route_queries

    route = _route("RR14", "时序对齐", "temporal alignment")
    # 首轮查询已在历史中，novelty=0；核心概念子集是更窄的检索式，仍然新颖。
    historical = list(route["search_queries"])

    regenerated = _regenerated_route_queries(route, historical, min_novelty=0.35)

    assert regenerated
    assert any(
        query not in set(route["search_queries"]) for query in regenerated
    )


def test_anchor_guard_rejects_unbound_and_negative_expansions():
    route = _route("DRIFT", "基于时序对齐的度量学习方法", "temporal alignment")
    route["anchor_expansions"] = [
        {"text": "temporal correspondence", "anchor_type": "semantic", "supports": "temporal alignment"},
        {"text": "medical diagnosis", "anchor_type": "task", "supports": "unknown task"},
        {"text": "fully supervised action localization", "anchor_type": "task", "supports": "few-shot video recognition"},
    ]

    guarded, report = guard_anchor_expansions(route)

    assert "temporal correspondence" in guarded["semantic_anchors"]
    assert "medical diagnosis" not in guarded["task_anchors"]
    assert "fully supervised action localization" not in guarded["task_anchors"]
    assert {item["reason"] for item in report["rejected"]} == {
        "unsupported_route_expansion", "negative_boundary_conflict",
    }


def test_exclusion_sentence_does_not_become_a_negative_anchor():
    route = _route("BOUNDARY", "基于时序对齐的度量学习方法", "temporal alignment")
    route["exclusion_criteria"] = [
        "排除仅把 few-shot video recognition 作为附属实验的 fully supervised localization 研究"
    ]

    result = validate_routes_against_evidence(
        [route], [
            _card("p1", "Temporal alignment for few-shot video recognition"),
            _card("p2", "Temporal matching for support-query videos", "temporal alignment"),
            _card("p3", "Sequence alignment for novel actions", "temporal alignment"),
        ], llm=None,
    )

    decision = _decision(result, "BOUNDARY")
    assert decision["route_validity"]["internal_consistency"] == 1.0
    assert decision["status"] == "KEEP"
