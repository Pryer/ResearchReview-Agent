"""专业任务的隔离、提交复核、预算与结果语义。"""

from __future__ import annotations

import pytest

from app.agent.controller import AgentController
from app.agent.execution_budget import AgentBudgetExceeded
from app.schemas.agent_task_schema import AgentRole, AgentTaskOutcome, AgentTaskStatus


def _state() -> dict:
    return {
        "state_schema_version": "2",
        "state_revision": 0,
        "user_query": "主题",
        "topic": "主题",
        "core_deliverables": [],
        "steps": [],
        "errors": [],
        "research_plan": {"task_graph": []},
    }


def test_handler_failure_does_not_pollute_authoritative_state():
    state = _state()

    def fail(working):
        working["candidate_papers"] = [{"paper_id": "partial"}]
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        AgentController().execute(
            role=AgentRole.SEARCH,
            operation="search_and_rank",
            objective="search",
            state=state,
            handler=fail,
        )

    assert "candidate_papers" not in state


def test_source_change_during_execution_rejects_task_patch():
    state = _state()

    def race(working):
        working["candidate_papers"] = [{"paper_id": "old"}]
        state["required_reference_count"] = 99
        state["max_papers_explicit"] = True

    result = AgentController().execute(
        role=AgentRole.SEARCH,
        operation="search_and_rank",
        objective="search",
        state=state,
        handler=race,
    )

    assert result.status == AgentTaskStatus.STALE
    assert result.error == "source_state_changed_before_commit"
    assert "candidate_papers" not in state
    assert state["required_reference_count"] == 99


def test_zero_action_budget_prevents_handler_call():
    state = _state()
    called = False

    def handler(working):
        nonlocal called
        called = True

    with pytest.raises(AgentBudgetExceeded):
        AgentController().execute(
            role=AgentRole.SEARCH,
            operation="search_and_rank",
            objective="search",
            state=state,
            handler=handler,
            budget={"action_limit": 0},
        )

    assert called is False


def test_completed_execution_can_have_blocked_research_outcome():
    state = _state()

    result = AgentController().execute(
        role=AgentRole.WRITING,
        operation="generate_deliverables",
        objective="write",
        state=state,
        handler=lambda working: working.update({"generation_blocked": True}),
    )

    assert result.status == AgentTaskStatus.COMPLETED
    assert result.outcome == AgentTaskOutcome.BLOCKED
    assert state["generation_blocked"] is True


def test_repeated_no_change_task_reuses_idempotent_result_without_budget_charge():
    state = _state()
    calls = 0

    def handler(working):
        nonlocal calls
        calls += 1

    controller = AgentController()
    first = controller.execute(
        role=AgentRole.SEARCH, operation="search_and_rank", objective="search",
        state=state, handler=handler,
    )
    second = controller.execute(
        role=AgentRole.SEARCH, operation="search_and_rank", objective="search",
        state=state, handler=handler,
    )

    assert first.idempotency_key == second.idempotency_key
    assert calls == 1
    assert state["agent_execution_budget"]["actions_reserved"] == 1


def test_search_task_does_not_receive_existing_writing_products():
    state = _state()
    state.update({"review": "private draft", "writing_plans": [{"section_id": "s1"}]})

    def handler(working):
        assert "review" not in working
        assert "writing_plans" not in working
        working["candidate_papers"] = []

    AgentController().execute(
        role=AgentRole.SEARCH, operation="search_and_rank", objective="search",
        state=state, handler=handler,
    )

    assert state["review"] == "private draft"
