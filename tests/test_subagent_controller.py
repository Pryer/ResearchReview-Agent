"""专业 Agent 任务权限、版本和状态合并测试。"""

from __future__ import annotations

import pytest

from app.agent.context_builder import context_source_fingerprint, source_state_version
from app.agent.controller import AgentController
from app.agent.subagents.base import AgentBoundaryViolation
from app.agent.subagents.search_agent import SearchAgent
from app.schemas.agent_task_schema import AgentRole, AgentTask, AgentTaskStatus


def _state() -> dict:
    return {
        "state_schema_version": "2",
        "user_query": "检索主题",
        "topic": "主题",
        "core_deliverables": [],
        "steps": [],
        "errors": [],
        "research_plan": {
            "task_graph": [{"id": "search", "operation": "search", "status": "pending"}],
        },
    }


def test_controller_runs_search_task_and_updates_plan_status():
    state = _state()

    result = AgentController().execute(
        role=AgentRole.SEARCH,
        operation="search_and_rank",
        objective="检索",
        state=state,
        handler=lambda current: current.update({
            "candidate_papers": [{"paper_id": "p1"}],
            "ranked_papers": [{"paper_id": "p1"}],
            "retrieval_stop_reason": "target_met",
        }),
    )

    assert result.status == AgentTaskStatus.COMPLETED
    assert state["research_plan"]["task_graph"][0]["status"] == "completed"
    assert state["agent_task_results"][0]["role"] == "search"
    assert set(state["main_agent_context"]) == {
        "goal", "state", "key_evidence", "decisions", "open_questions",
    }


def test_subagent_rejects_stale_source_state():
    state = _state()
    task = AgentTask(
        role=AgentRole.SEARCH,
        operation="search_and_rank",
        objective="检索",
        source_state_version=source_state_version(state),
        source_state_fingerprint="stale",
    )

    result = SearchAgent().execute(task, state, lambda current: None)

    assert result.status == AgentTaskStatus.STALE


def test_search_agent_cannot_change_user_goal():
    state = _state()
    task = AgentTask(
        role=AgentRole.SEARCH,
        operation="search_and_rank",
        objective="检索",
        source_state_version=source_state_version(state),
        source_state_fingerprint=context_source_fingerprint(state),
    )

    with pytest.raises(AgentBoundaryViolation):
        SearchAgent().execute(
            task,
            state,
            lambda current: current.update({"user_query": "被篡改"}),
        )

    assert state["user_query"] == "检索主题"
