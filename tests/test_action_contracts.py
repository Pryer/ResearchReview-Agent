"""动作字段权限：未知输入默认拒绝，越权输出在提交端再次验证。"""
import pytest
from app.agent.task_context import build_task_state
from app.agent.action_contracts import CONTRACTS, validate_patch
from app.agent.action_registry import ACTION_REGISTRY
from app.agent.controller import AgentController
from app.agent.subagents.base import AgentBoundaryViolation
from app.schemas.agent_task_schema import AgentRole


def test_all_executable_actions_have_explicit_contracts():
    assert set(CONTRACTS) == {name for name, spec in ACTION_REGISTRY.items() if spec.role is not None}


def test_new_private_fields_are_not_projected_and_writing_has_no_raw_documents():
    current = {"topic": "test", "billing": "private", "other_agent_private": "private",
        "parsed_papers": {"p1": "full document"}, "paper_cards": [{"paper_id": "p1"}],
        "agent_execution_budget": {"llm_tokens": 100}}
    result = build_task_state(current, AgentRole.WRITING, "rewrite_sections")
    assert set(result) == {"topic", "paper_cards", "active_agent_role"}
    result["paper_cards"][0]["paper_id"] = "changed"
    assert current["paper_cards"][0]["paper_id"] == "p1"


@pytest.mark.parametrize("patch,removed", [({"review": "injected"}, []), ({}, ["required_reference_count"]),
    ({"billing": 1}, []), ({"quality_gate": {"passed": True}}, [])])
def test_search_cannot_patch_or_remove_unapproved_fields(patch, removed):
    with pytest.raises(ValueError):
        validate_patch("search_and_rank", patch, removed)


def test_handler_cannot_add_unknown_output():
    current = {"user_query": "test", "topic": "test"}
    with pytest.raises(AgentBoundaryViolation):
        AgentController().execute(role=AgentRole.SEARCH, operation="search_and_rank",
            objective="search", state=current, handler=lambda working: working.update(billing=10))
    assert "billing" not in current


def test_output_shape_is_validated():
    with pytest.raises(ValueError):
        validate_patch("search_and_rank", {"candidate_papers": "not a list"}, [])


def test_actual_route_diagnostics_and_writer_verification_dependencies():
    validate_patch("validate_routes", {"coverage": {"total_papers": 3}}, [])
    inputs = build_task_state({"required_concepts": [["behavior"]], "generation_limit": 40},
                             AgentRole.WRITING, "generate_deliverables")
    assert inputs["required_concepts"] == [["behavior"]]
    assert inputs["generation_limit"] == 40


def test_search_task_receives_focus_recovery_decision_without_write_access():
    recovery = {"action": "TARGETED_SEARCH", "issues": [{
        "code": "required_focus_evidence_not_met",
        "details": {"missing_focuses": ["重点甲"]},
    }]}
    inputs = build_task_state(
        {"active_quality_recovery": recovery}, AgentRole.SEARCH, "search_and_rank"
    )
    assert inputs["active_quality_recovery"] == recovery
    with pytest.raises(ValueError):
        validate_patch("search_and_rank", {"active_quality_recovery": {}}, [])


def test_private_repair_fields_are_typed_and_conflicting_patch_is_rejected():
    with pytest.raises(ValueError):
        validate_patch("targeted_search", {"_citation_gap_repair_previous_cited": "wrong"}, [])
    with pytest.raises(ValueError, match="set and remove"):
        validate_patch("search_and_rank", {"candidate_papers": []}, ["candidate_papers"])


def test_adding_an_unknown_none_value_is_not_invisible():
    with pytest.raises(AgentBoundaryViolation):
        AgentController().execute(role=AgentRole.SEARCH, operation="search_and_rank",
            objective="search", state={},
            handler=lambda working: working.update(billing=None))
