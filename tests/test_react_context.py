"""Cache prefix, bounded feedback, epoch compaction and durable recovery contracts."""

import copy
import json
from types import SimpleNamespace as NS

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.agent.context_builder import build_main_agent_context
from app.agent.main_loop import MainAgentLoop
from app.agent.main_policy import AgentDecisionError, MainAgentPolicy
from app.agent.react_context import ContextBudgetExceeded, ReActContext, encode
from app.core.config import get_settings
from app.database.models import Base
from app.schemas.agent_decision_schema import AgentDecision
from app.services.llm_service import LLMService
from app.services.research_artifact_service import ResearchArtifactService
from app.services.research_execution_service import _STORE


@pytest.fixture
def store():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        service = ResearchArtifactService(db)
        token = _STORE.set(service)
        try:
            yield service
        finally:
            _STORE.reset(token)
    engine.dispose()


def state():
    return {
        "session_id": "s",
        "user_query": "研究主题，禁止扩大时间范围",
        "topic": "主题",
        "year_range_explicit": True,
        "start_year": 2022,
        "end_year": 2025,
        "user_clarifications": ["只使用有出处的证据"],
        "steps": [],
        "errors": [],
    }


class Decisions:
    def __init__(self, native):
        self.native_tools_enabled = native
        self.requests = []
        self.tools = []
        self.action = "search_and_rank"

    def complete_tool_call(self, messages, *, tools, operation):
        LLMService._validate_messages(messages)
        self.requests.append(copy.deepcopy(messages))
        self.tools.append(copy.deepcopy(tools))
        return {
            "name": self.action,
            "arguments": {},
            "tool_call_id": "call-" + str(len(self.requests)),
        }

    def complete_messages(self, messages, **kwargs):
        LLMService._validate_messages(messages)
        self.requests.append(copy.deepcopy(messages))
        return encode({"action": self.action, "arguments": {}})


@pytest.mark.parametrize("native", [False, True])
def test_requests_append_without_rewriting_prefix_and_survive_restore(store, native):
    current = state()
    llm = Decisions(native)
    policy = MainAgentPolicy(llm)
    trajectory = ReActContext(current)
    context = build_main_agent_context(current)
    policy.decide(context, allowed_actions=["search_and_rank"], trajectory=trajectory)
    trajectory.observe(
        {
            "status": "completed",
            "findings": ["检索到四篇论文"],
            "evidence_refs": ["artifact://verified-evidence"],
        }
    )
    stored = store.externalize_state("s", current)
    assert "main_agent_trajectory" not in stored
    recovered = store.hydrate_state("s", stored)
    assert recovered["main_agent_trajectory"] == current["main_agent_trajectory"]
    llm.action = "request_finish"
    policy.decide(context, allowed_actions=["request_finish"], trajectory=ReActContext(recovered))
    assert llm.requests[1][: len(llm.requests[0])] == llm.requests[0]
    assert "检索到四篇论文" in encode(llm.requests[1])
    assert "artifact://verified-evidence" in encode(llm.requests[1])
    if native:
        assert llm.tools[0] == llm.tools[1]
        tool = next(item for item in llm.requests[1] if item["role"] == "tool")
        assert tool["tool_call_id"] == "call-1"
    with pytest.raises(PermissionError):
        store.hydrate_state("other-session", stored)
    with pytest.raises(PermissionError):
        store.hydrate_state("s", stored, role="main")


def read_archive(store, refs):
    chunks = []
    for ref in refs:
        index = store.resolve("s", ref, role="service")["payload"]
        chunks.extend(
            store.resolve("s", part, role="service")["payload"]["value"]
            for part in index["fragments"]
        )
    return json.loads("".join(chunks))


def test_visible_feedback_is_bounded_and_raw_output_archived(store):
    current = state()
    trajectory = ReActContext(current)
    trajectory.proposed(
        AgentDecision(action="search_and_rank", transport="native_tool", tool_call_id="c")
    )
    payload = {
        "status": "completed",
        "findings": ["verified finding"],
        "evidence_refs": ["artifact://evidence"],
        "raw_document": "RAW_DOCUMENT" * 5000,
    }
    trajectory.observe(payload)
    message = trajectory.data["messages"][-1]
    assert len(message["content"]) <= get_settings().agent_context_tool_result_chars
    assert "RAW_DOCUMENT" not in message["content"]
    feedback = json.loads(message["content"])
    assert feedback["findings"] == ["verified finding"]
    assert feedback["evidence_refs"] == ["artifact://evidence"]
    restored = read_archive(store, feedback["artifact_refs"])
    assert restored == payload


def populated_trajectory():
    current = state()
    trajectory = ReActContext(current)
    goal = build_main_agent_context(current).goal.model_dump(mode="json")
    trajectory.data["goals"] = [goal]
    trajectory.data["messages"] = [{"role": "user", "content": encode({"goal": goal})}]
    for index in range(6):
        trajectory.data["messages"].append({"role": "user", "content": "old status " * 800})
        trajectory.proposed(
            AgentDecision(
                action="search_and_rank",
                transport="native_tool",
                tool_call_id=f"c{index}",
                reason="补检索证据缺口",
            )
        )
        trajectory.observe(
            {
                "status": "completed",
                "findings": [f"finding-{index}"],
                "evidence_refs": [f"artifact://evidence-{index}"],
            }
        )
    return current, trajectory


def test_threshold_compaction_archives_history_and_keeps_constraints_refs_and_pairs(
    store, monkeypatch
):
    current, trajectory = populated_trajectory()
    old_messages = copy.deepcopy(trajectory.data["messages"])
    monkeypatch.setattr(get_settings(), "agent_context_window_tokens", 24000)
    monkeypatch.setattr(get_settings(), "agent_context_compact_ratio", 0.8)
    messages = trajectory.request(
        build_main_agent_context(current),
        system="fixed",
        tools=[],
        allowed_actions=["search_and_rank"],
    )
    assert trajectory.data["epoch"] == 1
    assert messages[0] == {"role": "system", "content": "fixed"}
    LLMService._validate_messages(messages)
    summary = next(
        json.loads(item["content"])["context_checkpoint"]
        for item in messages
        if item["role"] == "user" and "context_checkpoint" in item["content"]
    )
    assert len(summary["observations"]) == 6
    for index in range(6):
        assert f"artifact://evidence-{index}" in encode(summary)
    assert "只使用有出处的证据" in encode(messages)
    assert "禁止扩大时间范围" in encode(messages)
    archive = read_archive(store, trajectory.data["archive_refs"])
    assert archive["messages"] == old_messages
    # Epoch 切换后的历史只保留结构化语义，不重放旧 Provider 的工具/思考协议。
    assert not any(item["role"] in {"assistant", "tool"} for item in messages)
    assert summary["observations"][-1]["decision"]["action"] == "search_and_rank"
    usage = current["agent_context_usage"]
    assert usage["estimated_input_tokens"] < usage["before_compaction_tokens"]
    # No repeated compaction merely because one new status bar was added.
    trajectory.request(
        build_main_agent_context(current), system="fixed", tools=[], allowed_actions=[]
    )
    assert trajectory.data["epoch"] == 1


def test_compaction_cannot_drop_hard_requirements_to_fit(store, monkeypatch):
    current, trajectory = populated_trajectory()
    context = build_main_agent_context(current)
    context.goal.request += "必须保留的要求" * 5000
    monkeypatch.setattr(get_settings(), "agent_context_window_tokens", 12000)
    with pytest.raises(ContextBudgetExceeded, match="exceeded_after_compaction"):
        trajectory.request(context, system="fixed", tools=[], allowed_actions=[])
    assert trajectory.data["goals"][-1]["request"] == context.goal.request
    assert trajectory.data["archive_refs"]


def test_loop_blocks_before_model_call_for_invalid_window(monkeypatch):
    monkeypatch.setattr(get_settings(), "agent_context_window_tokens", 10)
    current = state()
    llm = Decisions(True)
    assert (
        MainAgentLoop(llm).run(current, handlers={}, finish_validator=lambda s: False) == "blocked"
    )
    assert not llm.requests
    assert current["generation_blocked"]
    assert current["errors"][-1]["code"] == "ContextBudgetExceeded"


def test_unavailable_archive_store_blocks_instead_of_dropping_history(monkeypatch):
    current, trajectory = populated_trajectory()
    original = copy.deepcopy(trajectory.data["messages"])
    monkeypatch.setattr(get_settings(), "agent_context_window_tokens", 24000)
    with pytest.raises(ContextBudgetExceeded, match="archive_store_unavailable"):
        trajectory.request(
            build_main_agent_context(current), system="fixed", tools=[], allowed_actions=[]
        )
    assert trajectory.data["messages"] == original


def test_disallowed_native_call_is_paired_with_rejection_and_never_authorized():
    current = state()
    trajectory = ReActContext(current)
    llm = Decisions(True)
    llm.action = "generate_deliverables"
    with pytest.raises(AgentDecisionError, match="not allowed"):
        MainAgentPolicy(llm, correction_attempts=0).decide(
            build_main_agent_context(current),
            allowed_actions=["search_and_rank"],
            trajectory=trajectory,
        )
    assert "pending" not in trajectory.data
    LLMService._validate_messages(trajectory.data["messages"])
    assert json.loads(trajectory.data["messages"][-2]["content"])["status"] == "rejected"


def test_llm_transport_preserves_native_fields_and_rejects_unpaired_results():
    messages = [
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "provider-protocol-state",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "search_and_rank", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "verified result"},
    ]
    assert LLMService._validate_messages(messages) == messages
    with pytest.raises(ValueError, match="missing"):
        LLMService._validate_messages(messages[:1])
    with pytest.raises(ValueError, match="pending"):
        LLMService._validate_messages([messages[1]])
    with pytest.raises(ValueError, match="followed"):
        LLMService._validate_messages([messages[0], {"role": "user", "content": "skip result"}])


def test_openai_cached_tokens_are_recorded_as_actual_usage():
    service = LLMService()
    usage = service._record_usage(
        NS(prompt_tokens=100, completion_tokens=4, prompt_tokens_details=NS(cached_tokens=80)),
        model="mock",
        operation="test",
        provider="openai",
        started=0,
    )
    assert usage["prompt_cache_hit_tokens"] == 80
    assert service.last_usage == usage


def test_large_reference_lists_are_preserved_behind_one_bounded_archive_index(store):
    current = state()
    trajectory = ReActContext(current)
    trajectory.proposed(AgentDecision(action="search_and_rank"))
    payload = {
        "status": "completed",
        "evidence_refs": [f"artifact://source-{i}" for i in range(2000)],
    }
    trajectory.observe(payload)
    feedback = json.loads(trajectory.data["messages"][-1]["content"])["observation"]
    assert feedback["references_archived"] is True
    assert feedback["evidence_ref_count"] == 2000
    assert len(feedback["artifact_refs"]) == 1
    assert read_archive(store, feedback["artifact_refs"]) == payload
    assert len(encode(feedback)) <= get_settings().agent_context_tool_result_chars


def test_archive_failure_preserves_history_and_stops_compaction(store, monkeypatch):
    current, trajectory = populated_trajectory()
    original = copy.deepcopy(trajectory.data["messages"])
    monkeypatch.setattr(get_settings(), "agent_context_window_tokens", 24000)

    def fail(**kwargs):
        raise OSError("store unavailable")

    monkeypatch.setattr(store, "persist_payload", fail)
    with pytest.raises(ContextBudgetExceeded, match="archive_failed:OSError"):
        trajectory.request(
            build_main_agent_context(current), system="fixed", tools=[], allowed_actions=[]
        )
    assert trajectory.data["messages"] == original


def test_provider_switch_preserves_calls_but_strips_deepseek_only_extension():
    messages = [{"role": "assistant", "content": "answer", "reasoning_content": "protocol-state"}]
    assert LLMService._provider_messages(messages, "deepseek") == messages
    assert LLMService._provider_messages(messages, "openai") == [
        {"role": "assistant", "content": "answer"}
    ]
    assert messages[0]["reasoning_content"] == "protocol-state"


def test_unsupported_native_protocol_stays_json_across_rounds():
    from app.core.exceptions import NativeToolsUnsupportedError

    class Unsupported(Decisions):
        attempts = 0

        def complete_tool_call(self, messages, *, tools, operation):
            self.attempts += 1
            raise NativeToolsUnsupportedError("unsupported tools")

    current = state()
    llm = Unsupported(True)
    policy = MainAgentPolicy(llm)
    trajectory = ReActContext(current)
    context = build_main_agent_context(current)
    policy.decide(context, allowed_actions=["search_and_rank"], trajectory=trajectory)
    trajectory.observe({"status": "completed"})
    policy.decide(context, allowed_actions=["search_and_rank"], trajectory=trajectory)
    assert llm.attempts == 1
    assert llm.requests[1][: len(llm.requests[0])] == llm.requests[0]


def test_irreducible_status_view_budget_blocks_without_dropping_original_request():
    current = state()
    current["user_query"] = "不可删除的硬要求" * 10000
    llm = Decisions(True)
    assert (
        MainAgentLoop(llm).run(current, handlers={}, finish_validator=lambda s: False) == "blocked"
    )
    assert not llm.requests
    assert current["user_query"].endswith("不可删除的硬要求")
