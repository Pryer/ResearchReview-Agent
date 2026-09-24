"""主 Agent 只消费五字段并输出单个受校验动作。"""

from __future__ import annotations

import json

import pytest

from app.agent.context_builder import build_main_agent_context
from app.agent.main_policy import AgentDecisionError, MainAgentPolicy


class JsonLLM:
    native_tools_enabled = False

    def __init__(self, response: dict):
        self.response = response
        self.messages = None

    def complete_messages(self, messages, **kwargs):
        self.messages = messages
        return json.dumps(self.response, ensure_ascii=False)


def _context():
    return build_main_agent_context({
        "user_query": "主题",
        "topic": "主题",
        "conversation_history": [{"content": "RAW_SENTINEL_SHOULD_NOT_LEAK"}],
        "core_deliverables": [],
    })


def test_json_policy_request_contains_only_five_field_dynamic_context():
    llm = JsonLLM({
        "action": "search_and_rank", "arguments": {}, "reason": "缺少论文",
        "evidence_refs": [],
    })

    decision = MainAgentPolicy(llm).decide(
        _context(), allowed_actions=["search_and_rank", "report_blocked"]
    )

    dynamic_json = llm.messages[0]["content"].split("\n", 1)[0]
    assert set(json.loads(dynamic_json)) == {
        "goal", "state", "key_evidence", "decisions", "open_questions",
    }
    assert "RAW_SENTINEL_SHOULD_NOT_LEAK" not in llm.messages[0]["content"]
    assert decision.action == "search_and_rank"


def test_policy_rejects_unknown_action_after_bounded_correction():
    llm = JsonLLM({
        "action": "delete_database", "arguments": {}, "reason": "", "evidence_refs": [],
    })

    with pytest.raises(AgentDecisionError):
        MainAgentPolicy(llm, correction_attempts=0).decide(
            _context(), allowed_actions=["search_and_rank"]
        )


def test_policy_rejects_extra_tool_arguments():
    llm = JsonLLM({
        "action": "search_and_rank", "arguments": {"expand_years": 10},
        "reason": "", "evidence_refs": [],
    })

    with pytest.raises(AgentDecisionError):
        MainAgentPolicy(llm, correction_attempts=0).decide(
            _context(), allowed_actions=["search_and_rank"]
        )


def test_native_tool_catalog_only_offers_currently_allowed_actions():
    class NativeLLM:
        native_tools_enabled = True

        def __init__(self):
            self.offered = []

        def complete_tool_call(self, messages, *, tools, operation):
            self.offered = [item["function"]["name"] for item in tools]
            return {"name": "search_and_rank", "arguments": {}}

    llm = NativeLLM()
    decision = MainAgentPolicy(llm).decide(
        _context(), allowed_actions=["search_and_rank", "report_blocked"]
    )

    assert llm.offered == ["search_and_rank", "report_blocked"]
    assert decision.action == "search_and_rank"
