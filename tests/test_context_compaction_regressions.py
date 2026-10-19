"""长历史有界压缩、归档回查、协议重建与子任务结果消费的回归。"""
import copy

import pytest

from app.agent.context_builder import build_main_agent_context
from app.agent.main_loop import MainAgentLoop
from app.agent.react_context import ReActContext, ContextBudgetExceeded, encode, execute_context_recall
from app.agent.subagents.loop import SubagentLoop
from app.agent.controller import AgentController
from app.core.config import get_settings
from app.schemas.agent_decision_schema import AgentDecision
from app.services.context_history_service import recall_history
from app.services.llm_service import LLMService
from tests.test_react_context import state, read_archive
from tests.test_subagent_react import ScriptedLLM, report, search


@pytest.fixture
def store():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from app.database.models import Base
    from app.services.research_artifact_service import ResearchArtifactService
    from app.services.research_execution_service import _STORE
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


def observe(context, i, *, finding=None, evidence=None):
    context.proposed(AgentDecision(action="search_and_rank", reason="retrieve evidence"))
    context.observe({"status": "completed", "findings": [finding or ("observation-" + str(i)) * 10],
                     "evidence_refs": [evidence or f"state://paper-card/p{i}"]})


@pytest.mark.parametrize("count", [100, 300])
def test_long_history_compaction_has_bounded_memory_and_message_size(store, count):
    current = state()
    context = ReActContext(current)
    for i in range(count):
        observe(context, i)
    context._compact(build_main_agent_context(current).model_dump(mode="json"))
    assert len(context.data["memory"]) <= get_settings().agent_context_memory_items
    assert len(encode(context.data["memory"])) <= get_settings().agent_context_memory_chars
    assert len(encode(context.data["messages"]).encode("utf-8")) < 20000
    assert context.data["observation_count"] == count
    assert context.data["last_compacted_turn"] == count
    archived = read_archive(store, context.data["archive_refs"])
    assert len(archived["memory"]) == count
    assert archived["memory"][0]["evidence_refs"] == ["state://paper-card/p0"]


def test_repeated_epochs_continue_compacting_using_monotonic_count(store, monkeypatch):
    monkeypatch.setattr(get_settings(), "agent_context_memory_items", 3)
    current = state()
    context = ReActContext(current)
    for batch in range(4):
        for i in range(5):
            observe(context, batch * 5 + i)
        context.request(build_main_agent_context(current), system="stable", tools=[], allowed_actions=[])
        assert len(context.data["memory"]) <= 3
        assert context.data["observation_count"] == (batch + 1) * 5
    assert context.data["epoch"] == 4
    restored = store.hydrate_state("s", store.externalize_state("s", current))
    assert ReActContext(restored).data["observation_count"] == 20
    assert recall_history(restored, ReActContext(restored), "state://paper-card/p0")["history"]["matches"]


def test_evicted_observation_is_recalled_by_evidence_id_and_raw_payload_survives(store, monkeypatch):
    monkeypatch.setattr(get_settings(), "agent_context_memory_items", 2)
    current = state()
    context = ReActContext(current)
    context.proposed(AgentDecision(action="plan_claims"))
    original = {"status": "completed", "evidence_ids": ["p1:e001"], "findings": ["early verified finding"],
                "small_original_field": "must remain archived"}
    context.observe(original)
    raw_ref = context.data["memory"][0]["artifact_refs"]
    for i in range(10):
        observe(context, i)
    context._compact(build_main_agent_context(current).model_dump(mode="json"))
    assert "early verified finding" not in encode(context.data["messages"])
    assert read_archive(store, raw_ref) == original
    result = recall_history(current, context, "p1:e001")
    assert len(result["history"]["matches"]) == 1
    assert result["history"]["matches"][0]["finding"] == "early verified finding"
    assert "small_original_field" not in encode(result)


def test_recall_is_scoped_to_own_trajectory_and_session(store):
    current = state()
    main = ReActContext(current)
    other = ReActContext(current, namespace="analysis")
    observe(other, 1)
    other._compact(build_main_agent_context(current).model_dump(mode="json"))
    assert not recall_history(current, main, "state://paper-card/p1")["history"]["matches"]
    main.data["archive_refs"] = other.data["archive_refs"]
    with pytest.raises(PermissionError, match="another trajectory"):
        recall_history(current, main, "state://paper-card/p1")
    foreign = {**current, "session_id": "other"}
    with pytest.raises(PermissionError, match="another session"):
        recall_history(foreign, other, "state://paper-card/p1")


def test_compaction_rebuilds_protocol_neutral_history_but_preserves_raw_archive(store):
    current = state()
    context = ReActContext(current)
    raw = {"role": "assistant", "content": None, "reasoning_content": "private-provider-protocol-state",
           "tool_calls": [{"id": "old", "type": "function", "function": {"name": "search_and_rank", "arguments": "{}"}}]}
    context.proposed(AgentDecision(action="search_and_rank", transport="native_tool", tool_call_id="old"),
                     assistant_message=raw)
    context.observe({"status": "completed", "findings": ["verified fact"]})
    assert context.data["messages"][0] == raw
    context._compact(build_main_agent_context(current).model_dump(mode="json"))
    assert "private-provider-protocol-state" not in encode(context.data["messages"])
    assert read_archive(store, context.data["archive_refs"])["messages"][0] == raw
    assert "verified fact" in encode(context.data["messages"])
    context.proposed(AgentDecision(action="search_and_rank", transport="native_tool", tool_call_id="new"),
                     assistant_message={**raw, "tool_calls": [{**raw["tool_calls"][0], "id": "new"}]})
    context.observe({"status": "completed"})
    messages = [{"role": "system", "content": "stable"}, *context.data["messages"]]
    LLMService._validate_messages(messages)
    assert any("reasoning_content" in m for m in LLMService._provider_messages(messages, "deepseek"))
    assert not any("reasoning_content" in m for m in LLMService._provider_messages(messages, "openai"))


def test_previous_goals_are_archived_and_latest_hard_constraints_retained(store):
    current = state()
    context = ReActContext(current)
    old = build_main_agent_context(current).goal.model_dump(mode="json")
    context.data["goals"] = [old, {**old, "topic": "revised"}]
    observe(context, 1)
    status = build_main_agent_context(current).model_dump(mode="json")
    status["goal"]["constraints"]["required_reference_count"] = 40
    context._compact(status)
    assert context.data["goals"] == [status["goal"]]
    assert len(read_archive(store, context.data["archive_refs"])["goals"]) == 2
    assert context.data["goals"][0]["constraints"]["required_reference_count"] == 40


def test_pending_native_call_cannot_be_compacted(store):
    current = state()
    context = ReActContext(current)
    context.proposed(AgentDecision(action="search_and_rank", transport="native_tool", tool_call_id="pending"))
    before = copy.deepcopy(context.data)
    with pytest.raises(ContextBudgetExceeded, match="pending_tool_call"):
        context._compact(build_main_agent_context(current).model_dump(mode="json"))
    assert context.data == before


def test_recall_control_errors_propagate_and_corruption_is_explicit(store, monkeypatch):
    from app.agent.execution_budget import AgentBudgetExceeded
    current = state()
    context = ReActContext(current)
    observe(context, 1)
    context._compact(build_main_agent_context(current).model_dump(mode="json"))
    context.proposed(AgentDecision(action="recall_context", arguments={"evidence_id": "old"}))
    def stop(*args, **kwargs):
        raise AgentBudgetExceeded("stop-now")
    monkeypatch.setattr(store, "resolve", stop)
    with pytest.raises(AgentBudgetExceeded, match="stop-now"):
        execute_context_recall(context, {"evidence_id": "old"})
    def corrupt(*args, **kwargs):
        raise ValueError("corrupt")
    monkeypatch.setattr(store, "resolve", corrupt)
    result = execute_context_recall(context, {"evidence_id": "old"})
    assert result["status"] == "failed"
    assert current["errors"][-1]["code"] == "context_recall_failed"


@pytest.mark.parametrize("status", ["needs_help", "blocked"])
def test_main_consumes_business_block_despite_completed_lifecycle(status):
    current = state()
    llm = ScriptedLLM({"main_agent_decision": ["search_and_rank", "request_finish",
                            ("report_blocked", {"reason": "missing evidence"})],
                       "search_agent_decision": [report(status, "need more evidence")]})
    assert MainAgentLoop(llm).run(current, handlers={"search_and_rank": search},
                                  finish_validator=lambda s: False) == "blocked"
    assert current["main_agent_rejections"][0]["reason"] == "subagent_" + status
    next_main = [m for op, m in llm.calls if op == "main_agent_decision"][1]
    assert "subagent_" + status in encode(next_main)
    assert "finish_rejected_by_quality_gates" in encode(current["main_agent_rejections"])


def test_new_delegation_cannot_reenter_an_active_role_context():
    current = state()
    current.update(active_subagent="search", subagent_sessions={"search": {"active": {"id": "original"}}})
    before = copy.deepcopy(current)
    from app.agent.main_policy import AgentDecisionError
    with pytest.raises(AgentDecisionError, match="another delegation"):
        SubagentLoop(ScriptedLLM({}), AgentController()).run(current,
            decision=AgentDecision(action="search_and_rank"), handlers={"search_and_rank": search})
    assert current == before


def test_main_and_subagent_can_recall_without_executing_research_tools(store):
    current = state()
    context = ReActContext(current)
    observe(context, 0, finding="archived fact")
    llm = ScriptedLLM({"main_agent_decision": [("recall_context", {"evidence_id": "state://paper-card/p0"}),
                                             ("report_blocked", {"reason": "stop"})]})
    assert MainAgentLoop(llm).run(current, handlers={}, finish_validator=lambda s: False) == "blocked"
    assert "archived fact" in encode(llm.calls[1][1])
    assert not current.get("agent_task_results")
    child = ReActContext(current, namespace="search")
    observe(child, 1, finding="child fact")
    llm = ScriptedLLM({"search_agent_decision": [("recall_context", {"evidence_id": "state://paper-card/p1"}),
                                                report("needs_help", "request help")]})
    result = SubagentLoop(llm, AgentController()).run(current, decision=AgentDecision(action="search_and_rank"),
                                                    handlers={"search_and_rank": search})
    assert result.delegation_status == "needs_help"
    assert result.outcome.value == "blocked" and result.status.value == "completed"
    assert "child fact" in encode(llm.calls[1][1])


def test_small_window_trims_optional_memory_to_fit_but_keeps_latest_status(store, monkeypatch):
    current = state()
    current["quality_gate"] = {"passed": False, "blocking_issues": [{"code": "unsupported_claim"}]}
    context = ReActContext(current)
    for i in range(20):
        observe(context, i, finding="中文证据观察" * 40)
    monkeypatch.setattr(get_settings(), "agent_context_window_tokens", 8500)
    messages = context.request(build_main_agent_context(current), system="stable", tools=[], allowed_actions=[])
    assert current["agent_context_usage"]["estimated_input_tokens"] + get_settings().llm_control_plane_max_tokens <= 8500
    assert "unsupported_claim" in encode(messages)
    assert "禁止扩大时间范围" in encode(messages)
    assert len(context.data["memory"]) < 8
    assert recall_history(current, context, "state://paper-card/p0")["history"]["matches"]


def test_memory_character_limit_and_legacy_count_migrate_safely(store, monkeypatch):
    current = state()
    context = ReActContext(current)
    for i in range(10):
        observe(context, i, finding="large finding" * 40)
    context.data.pop("observation_count")
    context.data["last_compacted_turn"] = 5
    restored = store.hydrate_state("s", store.externalize_state("s", current))
    context = ReActContext(restored)
    assert context.data["observation_count"] == 10
    monkeypatch.setattr(get_settings(), "agent_context_memory_chars", 512)
    context.request(build_main_agent_context(restored), system="stable", tools=[], allowed_actions=[])
    assert len(encode(context.data["memory"])) <= 512
    assert context.data["last_compacted_turn"] == 10
    observe(context, 11)
    assert context.data["memory"][-1]["sequence"] == 11


def test_completed_report_with_blocked_outcome_is_not_reported_as_success():
    from app.schemas.agent_task_schema import AgentTaskResult, AgentRole, AgentTaskStatus, AgentTaskOutcome
    current = state()
    result = AgentTaskResult(task_id="d", role=AgentRole.SEARCH, operation="search_and_rank",
        status=AgentTaskStatus.COMPLETED, outcome=AgentTaskOutcome.BLOCKED, delegation_status="completed",
        input_state_fingerprint="fp", findings=["operation ended with insufficient evidence"])
    MainAgentLoop._consume_delegation_result(current, result)
    feedback = current["main_agent_rejections"][-1]
    assert feedback["reason"] == "subagent_blocked"
    assert feedback["delegation_status"] == "completed"


def test_idempotent_blocked_result_keeps_task_graph_blocked():
    from app.schemas.agent_task_schema import AgentRole
    current = state()
    current.update(candidate_papers=[], research_plan={"task_graph": [{"operation": "search", "status": "pending"}]})
    def failed_search(working):
        working.update(search_failed=True, candidate_papers=[])
    controller = AgentController()
    first = controller.execute(role=AgentRole.SEARCH, operation="search_and_rank", objective="search",
                               state=current, handler=failed_search)
    assert first.outcome.value == "blocked"
    second = controller.execute(role=AgentRole.SEARCH, operation="search_and_rank", objective="search",
                                state=current, handler=lambda s: pytest.fail("idempotent result should be reused"))
    assert second.task_id == first.task_id
    assert current["research_plan"]["task_graph"][0]["status"] == "blocked"


def test_recall_scan_limit_reports_incomplete_instead_of_missing_evidence(store, monkeypatch):
    import app.services.context_history_service as service
    context = ReActContext(state())
    context.data["archive_refs"] = ["root-0"]
    calls = []
    def load(store, session_id, ref, scope_id, remaining_bytes):
        calls.append(ref)
        n = int(ref.split("-")[1])
        return {"memory": [], "previous_archive_refs": [f"root-{n+1}"]}, remaining_bytes
    monkeypatch.setattr(service, "_load_archive", load)
    result = recall_history(context.state, context, "unknown")
    assert len(calls) == 128
    assert result["history"]["matches"] == []
    assert result["history"]["search_complete"] is False


def test_history_recall_does_not_treat_earlier_recall_as_new_evidence(store):
    current = state()
    context = ReActContext(current)
    observe(context, 1, finding="actual observation")
    for _ in range(2):
        context.proposed(AgentDecision(action="recall_context", arguments={"evidence_id": "state://paper-card/p1"}))
        payload = execute_context_recall(context, {"evidence_id": "state://paper-card/p1"})
        assert [item["finding"] for item in payload["history"]["matches"]] == ["actual observation"]
