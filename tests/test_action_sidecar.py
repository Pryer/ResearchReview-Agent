"""Book chapter-4 Sidecar: isolated inputs and fail-closed decisions."""

from __future__ import annotations

import json

from app.agent.sidecar import ActionSidecar


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
