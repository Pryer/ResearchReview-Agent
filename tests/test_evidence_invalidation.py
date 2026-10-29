"""证据身份/内容与排序噪声分离，并验证失效后的交接。"""
import copy

import pytest

from app.agent.evidence_invalidation import evidence_snapshot, invalidate_changed_evidence
from app.agent.action_registry import recovery_handoff_action
from app.agent.deliverable_router import check_generation_readiness


def evidence_state():
    return {
        "topic": "RAG", "intent": "generate_introduction",
        "paper_details": [{"paper_id": "p1", "title": "Paper", "abstract": "Evidence",
                           "source": "crossref", "_screening_confidence": 0.95,
                           "_eligible_deliverables": ["research_background"]}],
        "paper_cards": [{"paper_id": "p1", "title": "Paper", "evidence_source": "abstract"}],
        "claim_plans": [{"claim_id": "old"}], "quality_gate": {"passed": True},
        "validated_routes": [{"route_id": "old"}],
    }


def test_ranking_noise_and_equivalent_source_aggregation_preserve_current_evidence():
    state = evidence_state()
    before = evidence_snapshot(state, "fetch_metadata")
    state["paper_details"][0].update(_sources=["crossref"], _rank_score=0.4, _llm_semantic_score=0.9)
    invalidate_changed_evidence(state, "fetch_metadata", before)
    assert state["paper_cards"] and state["claim_plans"] and state["quality_gate"]["passed"]
    assert not state.get("pending_card_rebuild_ids")
    assert state["paper_details"][0]["_sources"] == ["crossref"]


@pytest.mark.parametrize("change", [
    {"abstract": "Changed evidence"}, {"year": 2025}, {"doi": "10.1/new"},
    {"source": "arxiv"}, {"_sources": ["crossref", "arxiv"]},
    {"_screening_confidence": 0.2}, {"_eligible_deliverables": []},
    {"_topic_relation": "unrelated"}, {"unknown_provenance": "new"},
])
def test_content_admission_and_provenance_changes_still_invalidate_authorization(change):
    state = evidence_state()
    before = evidence_snapshot(state, "fetch_metadata")
    state["paper_details"][0].update(change)
    invalidate_changed_evidence(state, "fetch_metadata", before)
    assert state["paper_cards"] == []
    assert not state.get("claim_plans") and not state.get("quality_gate")
    assert state["pending_card_rebuild_ids"] == ["p1"]
    assert recovery_handoff_action(state) == "extract_paper_cards"


def test_rebuild_only_clears_after_current_cards_are_committed_and_never_restores_claims():
    state = evidence_state()
    before = evidence_snapshot(state, "fetch_metadata")
    state["paper_details"][0]["abstract"] = "Changed"
    invalidate_changed_evidence(state, "fetch_metadata", before)
    before = evidence_snapshot(state, "extract_paper_cards")
    invalidate_changed_evidence(state, "extract_paper_cards", before)
    assert state["pending_card_rebuild_ids"] == ["p1"]
    state["paper_cards"] = [{"paper_id": "p1", "evidence_source": "metadata"}]
    invalidate_changed_evidence(state, "extract_paper_cards", before)
    assert state["pending_card_rebuild_ids"] == []
    assert not state.get("claim_plans")
    assert not recovery_handoff_action(state)


def test_removed_paper_is_not_resurrected_and_unmodified_paper_is_retained():
    state = evidence_state()
    state["paper_details"].append({"paper_id": "removed", "abstract": "old"})
    state["paper_cards"].append({"paper_id": "removed"})
    before = evidence_snapshot(state, "fetch_metadata")
    state["paper_details"].pop()
    invalidate_changed_evidence(state, "fetch_metadata", before)
    assert [c["paper_id"] for c in state["paper_cards"]] == ["p1"]
    assert not state["pending_card_rebuild_ids"]
    assert not state.get("claim_plans")


def test_readiness_structured_options_obey_strict_constraints_but_keep_ordinary_choices():
    state = {"user_query": "不能扩大题名、年份或降低篇数", "required_reference_count": 3,
             "max_papers_explicit": True, "paper_cards": []}
    strict = check_generation_readiness(copy.deepcopy(state))
    assert not strict.ready and strict.requested_minimum_references == 3
    assert not any("扩大检索年份" in x or "接受少于" in x for x in strict.recovery_options)
    assert any("原范围" in x for x in strict.recovery_options)
    state["user_query"] = "需要至少3篇引用"
    ordinary = check_generation_readiness(state)
    assert any("扩大检索年份" in x for x in ordinary.recovery_options)
    assert any("接受少于" in x for x in ordinary.recovery_options)


def test_pending_rebuild_only_extracts_missing_current_papers(monkeypatch):
    from app.agent.nodes.extraction import extract_card_node
    state = evidence_state()
    state["paper_cards"][0]["eligible_deliverables"] = ["research_background"]
    state["paper_details"].append({"paper_id": "p2", "title": "new evidence"})
    state["pending_card_rebuild_ids"] = ["p2"]
    kept = copy.deepcopy(state["paper_cards"][0])
    calls = []
    def extract(papers, parsed, **kwargs):
        calls.append([p["paper_id"] for p in papers])
        return [{"paper_id": "p2", "evidence_source": "abstract"}]
    monkeypatch.setattr("app.tools.extract_paper_card.batch_extract_paper_cards", extract)
    extract_card_node(state)
    assert calls == [["p2"]]
    assert state["paper_cards"] == [kept, {"paper_id": "p2", "evidence_source": "abstract"}]


def test_explicit_constraint_changes_reach_writer_readiness_contract():
    from app.agent.action_contracts import CONTRACTS
    state = {"quality_original_query": "不能扩大年份或降低篇数", "required_reference_count": 3,
             "max_papers_explicit": True, "paper_cards": [],
             "quality_constraint_changes": [{"action": "expand_time_range"}]}
    projected = {k: v for k, v in state.items() if k in CONTRACTS["generate_deliverables"].inputs}
    readiness = check_generation_readiness(projected)
    assert any("扩大检索年份" in x for x in readiness.recovery_options)
    assert not any("接受少于" in x for x in readiness.recovery_options)
