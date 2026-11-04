"""分段恢复的权限、预算、检查点和候选发布边界。"""
import copy

import pytest

from app.agent.controller import AgentController
from app.agent.delegation_recovery import search_once, settle_after_analysis
from app.agent.execution_budget import budget_scope
from app.agent.action_registry import allowed_actions, recovery_handoff_action
from app.agent.subagents.base import AgentBoundaryViolation
from app.schemas.agent_task_schema import AgentRole
from app.core.config import get_settings
from tests.test_main_delegation_context import agent_artifacts


def source():
    return {"session_id": "recovery", "topic": "主题", "start_year": 2024, "end_year": 2026,
            "required_reference_count": 40, "max_papers_explicit": True,
            "paper_cards": [{"paper_id": "p1", "title": "已有论文"}],
            "paper_details": [{"paper_id": "p1", "title": "已有论文"}],
            "validated_routes": [{"route_id": "r1", "paper_ids": ["p1"], "core_paper_ids": ["p1"]}],
            "recovery_decision": {"action": "TARGETED_SEARCH", "queries": ["主题 新方法"],
                                  "affected_route_ids": ["r1"], "route_targets": {"r1": 3}},
            "evidence_gap_report": {"needs_recovery": True, "coverage_score": 0.3},
            "steps": [], "errors": []}


def test_search_commit_keeps_pending_phase_across_checkpoint_and_does_not_change_evidence(agent_artifacts):
    current = source()
    original = copy.deepcopy(current)
    def run_search(s, **kwargs):
        s["candidate_papers"] = [{"paper_id": "p2"}]
        s["last_search_new_results"] = 1
    with budget_scope(current):
        AgentController().execute(role=AgentRole.SEARCH, operation="targeted_search", objective="补充证据",
            state=current, handler=lambda s: search_once(s, search=run_search,
                rank=lambda value: value.update(ranked_papers=[{"paper_id": "p2"}])))
    assert current["paper_cards"] == original["paper_cards"]
    assert current["validated_routes"] == original["validated_routes"]
    assert current["recovery_round"] == 1
    assert current["recovery_action_count"] == 1
    assert current["agent_execution_budget"]["recovery_used"] == 1
    saved = agent_artifacts.externalize_state("recovery", current)
    restored = agent_artifacts.hydrate_state("recovery", saved)
    assert restored["pending_evidence_recovery"]["kind"] == "route"
    assert "targeted_search" not in allowed_actions(restored)
    assert recovery_handoff_action(restored) == "fetch_metadata"


@pytest.mark.parametrize("field,value", [("review", "越权正文"), ("claim_plans", [{"claim_id": "invented"}])])
def test_targeted_search_cannot_commit_analysis_or_writing_products(field, value):
    current = source()
    with pytest.raises(AgentBoundaryViolation):
        AgentController().execute(role=AgentRole.SEARCH, operation="targeted_search", objective="补搜",
            state=current, handler=lambda s: s.update({field: value}))
    assert field not in current


def test_new_hits_in_other_routes_do_not_count_as_progress_for_the_missing_route(monkeypatch):
    from app.agent import nodes
    current = source()
    current.update(pending_evidence_recovery={"kind": "route", "decision": current["recovery_decision"],
                   "before_card_ids": ["p1"], "coverage_before": 0.3, "core_before": {"r1": 1}},
                   recovery_round=1, route_recovery_attempts={"r1": 1},
                   incremental_retrieval=True,
                   paper_cards=[{"paper_id": "p1"}, {"paper_id": "p2"}],
                   validated_routes=[{"route_id": "r1", "core_paper_ids": ["p1"]},
                                     {"route_id": "r2", "core_paper_ids": ["p2"]}],
                   route_validation_report={"coverage": {"evidence_understood_rate": 0.8},
                       "assignment_map": {"p2": {"type": "single_route"}},
                       "validated_routes": [{"route_id": "r2", "core_paper_ids": ["p2"]}]})
    monkeypatch.setattr(nodes, "diagnose_evidence_gaps_node", lambda s, **k: s.update(
        evidence_gap_report={"needs_recovery": True, "gaps": [{"gap_type": "SEARCH_COVERAGE_GAP"}]}))
    monkeypatch.setattr(nodes, "global_evidence_gate_node", lambda s: None)
    settle_after_analysis(current)
    assert current["route_recovery_progress"]["r1"]["new_relevant"] == 0
    assert current["evidence_recovery_status"] == "EXHAUSTED"
    assert "pending_evidence_recovery" not in current
    assert "incremental_retrieval" not in current


def test_pending_candidate_cannot_be_published_or_rewritten_before_independent_validation():
    from app.agent.graph import _build_output
    current = source()
    current.update(review="尚未独立验证的候选正文", quality_gate={"passed": True, "draft_released": True},
                   pending_result_validation=True)
    output = _build_output(current)
    assert output["status"] == "blocked"
    assert output["body"] == ""
    assert not output["draft_released"]
    assert "尚未独立验证的候选正文" not in output["answer"]
    assert not {"generate_deliverables", "rewrite_sections"} & set(allowed_actions(current))
    with pytest.raises(ValueError, match="awaiting independent validation"):
        AgentController().execute(role=AgentRole.WRITING, operation="generate_deliverables",
            objective="不能绕过验证", state=current, handler=lambda s: pytest.fail("must not write"))


def test_scope_revision_cap_is_respected_in_the_action_catalog():
    current = source()
    current.update(provisional_framework={"provisional_routes": [{"route_id": "r1"}]},
                   recovery_decision={"action": "SCOPE_REVISION"},
                   scope_revision_count=get_settings().evidence_recovery_max_scope_revisions)
    assert "revise_scope" not in allowed_actions(current)
