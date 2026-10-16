"""Book chapter-4 Sidecar: isolated inputs and fail-closed decisions."""

from __future__ import annotations

import json

import pytest

from app.agent.execution import AgentCancelledError
from app.agent.execution_budget import AgentBudgetExceeded, AgentExecutionStale
from app.agent.main_loop import MainAgentLoop
from app.agent.sidecar import ActionSidecar
from app.core.config import get_settings
from app.core.exceptions import LLMProviderUnavailableError
from app.database.runtime_repository import RuntimeConflict


class StubLLM:
    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error
        self.messages = None

    def complete_messages(self, messages, **kwargs):
        self.messages = messages
        if self.error:
            raise self.error
        return json.dumps(self.payload)


def test_sidecar_sees_only_structured_action_and_arguments():
    llm = StubLLM({"verdict": "allow", "reason": "bounded research action"})
    decision = ActionSidecar(llm).review(
        operation="generate_deliverables", arguments={"mode": "draft"})
    assert decision.verdict == "allow"
    assert json.loads(llm.messages[1]["content"]) == {
        "operation": "generate_deliverables",
        "arguments": {"mode": "draft"},
    }
    assert len(llm.messages) == 2


def test_sidecar_denies_and_escalates():
    deny = StubLLM({"verdict": "deny", "reason": "untrusted target"})
    assert ActionSidecar(deny).review(operation="rewrite_sections", arguments={}).verdict == "deny"
    outage = StubLLM(error=RuntimeError("model offline"))
    assert ActionSidecar(outage).review(operation="rewrite_sections", arguments={}).verdict == "escalate"


def test_sidecar_invalid_classifier_response_fails_closed():
    llm = StubLLM({"verdict": "maybe", "reason": "unknown"})
    assert ActionSidecar(llm).review(operation="generate_deliverables", arguments={}).verdict == "escalate"


STOP_CASES = [
    (AgentBudgetExceeded, "blocked"),
    (AgentCancelledError, "cancelled"),
    (AgentExecutionStale, "failed"),
    (RuntimeConflict, "failed"),
    (LLMProviderUnavailableError, "failed"),
]


@pytest.mark.parametrize("error_type,status", STOP_CASES)
def test_sidecar_preserves_execution_control_exception(error_type, status):
    error = error_type("execution stopped")
    with pytest.raises(error_type) as caught:
        ActionSidecar(StubLLM(error=error)).review(operation="search_and_rank", arguments={})
    assert caught.value is error


@pytest.mark.parametrize("error_type,status", STOP_CASES)
def test_sidecar_stop_reaches_loop_without_clarification_or_execution(monkeypatch, error_type, status):
    monkeypatch.setattr(get_settings(), "agent_sidecar_enabled", True)
    monkeypatch.setattr(get_settings(), "agent_sidecar_review_actions", "search_and_rank")
    current = {
        "user_query": "研究主题", "topic": "主题", "steps": [], "errors": [],
        "agent_mandatory_actions": [{"action": "search_and_rank"}],
    }
    executed = []
    result = MainAgentLoop(StubLLM(error=error_type("execution stopped"))).run(
        current,
        handlers={"search_and_rank": lambda *args: executed.append(True)},
        finish_validator=lambda state: False,
    )
    assert result == status
    assert current["result_status"] == status
    assert current["errors"][-1]["code"] == error_type.__name__
    assert "clarification" not in current
    assert not executed
    assert not current["main_agent_trajectory"].get("pending")
