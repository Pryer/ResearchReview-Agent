"""任务驱动的历史记忆召回、反馈与证据权限边界。"""
import copy
import json

import pytest
from sqlalchemy import select

from app.agent.context_builder import build_main_agent_context
from app.agent.context_compaction import compact_main_context
from app.agent.main_loop import MainAgentLoop
from app.agent.main_policy import AgentDecisionError, MainAgentPolicy
from app.agent.react_context import ReActContext, encode
from app.core.config import get_settings
from app.database.models import ResearchEventModel
from app.schemas.agent_decision_schema import AgentDecision
from app.services.research_memory_retrieval import (
    ResearchMemoryRetriever, execute_memory_search, prepare_memory_context,
    record_memory_control_outcome, record_memory_feedback,
)
from tests.test_react_context import state


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


def events(store, kind):
    return [json.loads(row.payload_json)["payload"] for row in store.repo.db.scalars(
        select(ResearchEventModel).where(ResearchEventModel.event_type == kind)
        .order_by(ResearchEventModel.event_id)).all()]


def seed(store):
    service = ResearchMemoryRetriever(store)
    service.memory.append_event(session_id="s", event_type="analysis", payload={
        "findings": [{"finding": "跨模态学习缺少外部验证", "source_refs": ["artifact://original"]},
                     {"finding": "unrelated astronomy result"}],
        "decisions": [{"decision_id": "d1", "decision": "跨模态范围采用多中心研究"}],
        "open_questions": [{"question_id": "q1", "question": "跨模态外部验证是否足够"}],
    })
    return service


def search(current, **arguments):
    trajectory = ReActContext(current)
    trajectory.proposed(AgentDecision(action="search_research_memory", arguments=arguments))
    return execute_memory_search(current, trajectory, arguments)


def test_task_query_uses_persisted_summary_and_chinese_relevance(store):
    service = seed(store)
    ref = service.memory.rebuild_summary("s")
    result = service.retrieve("s", "跨模态外部验证", kind="finding", requested_ref=ref)
    assert result["summary_ref"] == ref
    assert len(result["matches"]) == 1
    item = result["matches"][0]
    assert item["text"] == "跨模态学习缺少外部验证"
    assert item["source_ref"] == ref
    assert item["requires_revalidation"] and not item["authorizes_evidence"]
    assert service.retrieve("s", "QUANTUM", kind="finding")["matches"] == []


def test_old_findings_survive_snapshot_replacement_but_invalid_decisions_do_not(store):
    service = seed(store)
    old = service.memory.rebuild_summary("s")
    service.memory.append_event(session_id="s", event_type="research_result", payload={"findings": []})
    result = service.retrieve("s", "跨模态", requested_ref=old)
    assert result["summary_ref"] != old
    assert [item["kind"] for item in result["matches"]] == ["finding"]
    assert result["matches"][0]["validity"] == "historical_summary"
    assert service.retrieve("s", "跨模态", kind="decision")["matches"] == []
    assert service.retrieve("s", "跨模态", kind="question")["matches"] == []


def test_cached_questions_are_evicted_after_resolution_and_scope_change(store):
    service = seed(store)
    current = state()
    search(current, query="跨模态", kind="question")
    assert current["research_memory_context"]["matches"]
    searched_cursor = current["research_memory_context"]["cursor"]
    service.memory.append_event(session_id="s", event_type="resolution", payload={"resolved_question_ids": ["q1"]})
    prepare_memory_context(current)
    assert current["research_memory_context"]["matches"] == []
    assert current["research_memory_context"]["cursor"] == searched_cursor
    assert current["research_memory_context"]["current_cursor"] > searched_cursor
    search(current, query="跨模态", kind="finding")
    current["topic"] = "changed scope"
    prepare_memory_context(current)
    assert "research_memory_context" not in current


def test_summary_cannot_be_read_by_another_session(store):
    service = seed(store)
    ref = service.memory.rebuild_summary("s")
    with pytest.raises(PermissionError):
        service.retrieve("foreign", "跨模态", requested_ref=ref)


def test_recency_breaks_ties_and_context_results_are_bounded(store, monkeypatch):
    service = ResearchMemoryRetriever(store)
    service.memory.append_event(session_id="s", event_type="analysis", payload={
        "findings": [{"finding": f"validation {i} " + "x" * 450} for i in range(12)]})
    monkeypatch.setattr(get_settings(), "research_memory_context_chars", 800)
    result = service.retrieve("s", "validation")
    assert len(json.dumps(result["matches"], ensure_ascii=False)) <= 800
    assert len(result["matches"]) == 1
    assert result["matches"][0]["text"].startswith("validation 11 ")
    assert result["result_limited"]


def test_failure_scan_horizon_is_explicit_and_keeps_latest_match(store):
    service = ResearchMemoryRetriever(store)
    for i in range(201):
        service.memory.append_event(session_id="s", event_type="agent_failure", payload={
            "error": f"retrieval timeout {i}", "operation": "search_and_rank"})
    result = service.retrieve("s", "retrieval timeout", kind="failure", limit=1)
    assert not result["search_complete"]
    assert result["matches"][0]["text"] == "retrieval timeout 200"


def test_summary_scan_horizon_does_not_claim_absence(store):
    service = seed(store)
    for i in range(18):
        service.memory.append_event(session_id="s", event_type="research_result", payload={
            "findings": [{"finding": f"snapshot {i}"}]})
        service.memory.rebuild_summary("s")
    result = service.retrieve("s", "跨模态")
    assert result["matches"] == [] and not result["search_complete"]
    current = state()
    search(current, query="跨模态")
    assert build_main_agent_context(current).state.memory_retrieval["search_complete"] is False


def test_memory_cannot_override_finish_gate(store):
    seed(store)
    current = state()
    search(current, query="跨模态", kind="finding")
    class LLM:
        native_tools_enabled = True
        def complete_tool_call(self, messages, **kwargs):
            assert json.loads(messages[-1]["content"])["state"]["research_memory"]
            return {"name": "request_finish", "arguments": {}}
    assert MainAgentLoop(LLM()).run(current, handlers={}, finish_validator=lambda s: False) == "blocked"
    assert all(row["status"] == "rejected" for row in events(store, "memory_action_outcome"))
    assert current["main_agent_rejections"][0]["reason"] == "finish_rejected_by_quality_gates"


def test_graph_without_store_does_not_inject_cached_research_memory(store):
    from app.services.research_execution_service import _STORE
    seed(store)
    current = state()
    search(current, query="跨模态")
    token = _STORE.set(None)
    try:
        prepare_memory_context(current)
        assert "research_memory_context" not in current
        assert current["research_memory_status"]["available"] is False
    finally:
        _STORE.reset(token)


@pytest.mark.parametrize("native", [True, False])
def test_main_loop_search_inject_feedback_action_outcome_and_finish_gate(store, native):
    seed(store)
    current = state()
    class LLM:
        native_tools_enabled = native
        turns = 0

        def response(self, messages, operation):
            from tests.agent_decision_fakes import delegated_response
            child = delegated_response(messages, operation)
            if child:
                return child
            self.turns += 1
            context = json.loads(messages[-1]["content"])
            if self.turns == 1:
                assert context["state"]["memory_retrieval"]["available"]
                return {"name": "search_research_memory", "arguments": {"query": "跨模态验证", "kind": "finding"}}
            item = context["state"]["research_memory"][0]
            assert item["text"] == "跨模态学习缺少外部验证"
            if self.turns == 2:
                return {"name": "search_and_rank", "arguments": {"memory_feedback": [{
                    "memory_id": item["memory_id"], "disposition": "used", "reason": "补充外部验证材料"}]}}
            return {"name": "request_finish", "arguments": {}}

        def complete_tool_call(self, messages, *, tools, operation):
            return self.response(messages, operation)

        def complete_messages(self, messages, **kwargs):
            raw = self.response(messages, kwargs["operation"])
            args = dict(raw["arguments"])
            return json.dumps({"action": raw["name"], "arguments": args,
                               "memory_feedback": args.pop("memory_feedback", [])})

    def handler(working, arguments):
        assert arguments == {}  # 通信/反馈元数据不能扩张 handler 契约。
        working["candidate_papers"] = [{"paper_id": "p1"}]

    gates = []
    def finish(working):
        gates.append(bool(working.get("candidate_papers")))
        return gates[-1]

    assert MainAgentLoop(LLM()).run(current, handlers={"search_and_rank": handler}, finish_validator=finish) == "completed"
    assert gates == [True]
    feedback = events(store, "memory_feedback")
    assert feedback[0]["feedback"][0]["disposition"] == "used"
    assert feedback[1]["usage_reported"] is False
    outcomes = events(store, "memory_action_outcome")
    assert outcomes[0]["operation"] == "search_and_rank" and outcomes[0]["memory_ids"]
    assert outcomes[-1]["operation"] == "request_finish"
    assert events(store, "memory_retrieved")


@pytest.mark.parametrize("field,value", [("memory_feedback", [{"memory_id": "forged", "disposition": "used", "reason": "test"}]),
                                         ("evidence_refs", ["artifact://historical-evidence"])])
def test_memory_does_not_authorize_forged_feedback_or_current_evidence(store, field, value):
    seed(store)
    current = state()
    search(current, query="跨模态", kind="finding")
    class LLM:
        native_tools_enabled = False
        def complete_messages(self, *args, **kwargs):
            return json.dumps({"action": "request_finish", "arguments": {}, field: value})
    with pytest.raises(AgentDecisionError):
        MainAgentPolicy(LLM(), correction_attempts=0).decide(build_main_agent_context(current), allowed_actions=["request_finish"])


def test_feedback_and_retrieved_context_survive_snapshot_and_audit_is_idempotent(store):
    seed(store)
    current = state()
    search(current, query="跨模态", kind="finding")
    item = current["research_memory_context"]["matches"][0]
    decision = AgentDecision(action="request_finish", memory_feedback=[{
        "memory_id": item["memory_id"], "disposition": "used", "reason": "recheck"}])
    record_memory_feedback(current, decision)
    saved = store.externalize_state("s", copy.deepcopy(current))
    restored = store.hydrate_state("s", saved)
    assert restored["research_memory_context"] == current["research_memory_context"]
    record_memory_control_outcome(restored, "request_finish", "rejected", "blocked")
    record_memory_feedback(current, decision)
    record_memory_control_outcome(current, "request_finish", "rejected", "blocked")
    assert len(events(store, "memory_feedback")) == len(events(store, "memory_action_outcome")) == 1


def test_compacted_status_does_not_falsely_report_dropped_memory_as_presented(store):
    seed(store)
    current = state()
    search(current, query="跨模态", kind="finding")
    record_memory_feedback(current, AgentDecision(action="request_finish"), presented_matches=[])
    assert events(store, "memory_feedback") == []
    context = build_main_agent_context(current)
    context.state.research_memory *= 100
    compacted = compact_main_context(context, max_chars=2500)
    assert len(compacted.state.research_memory) < len(context.state.research_memory)
    assert compacted.goal == context.goal


def test_unresolved_failures_are_pinned_and_success_unpins_but_archive_remains(store, monkeypatch):
    monkeypatch.setattr(get_settings(), "agent_context_memory_items", 3)
    current = state()
    trajectory = ReActContext(current)
    trajectory.proposed(AgentDecision(action="search_and_rank"))
    trajectory.observe({"status": "failed", "error": "跨模态 source timeout"})
    for _ in range(6):
        trajectory.proposed(AgentDecision(action="fetch_metadata"))
        trajectory.observe({"status": "completed", "findings": ["ok"]})
    trajectory._compact(build_main_agent_context(current).model_dump(mode="json"))
    assert "source timeout" in encode(trajectory.data["memory"])
    trajectory.proposed(AgentDecision(action="search_and_rank"))
    trajectory.observe({"status": "completed", "findings": ["source recovered"]})
    for _ in range(3):
        trajectory.proposed(AgentDecision(action="fetch_metadata"))
        trajectory.observe({"status": "completed"})
    trajectory._compact(build_main_agent_context(current).model_dump(mode="json"))
    assert "source timeout" not in encode(trajectory.data["memory"])
    matches = ResearchMemoryRetriever(store).retrieve("s", "跨模态 timeout", kind="failure")["matches"]
    assert matches[0]["operation"] == "search_and_rank"


def test_memory_read_failure_clears_cached_result_and_reports_diagnostic(store, monkeypatch):
    seed(store)
    current = state()
    search(current, query="跨模态")
    def fail(*args, **kwargs):
        raise OSError("unavailable")
    monkeypatch.setattr(ResearchMemoryRetriever, "retrieve", fail)
    assert search(current, query="跨模态")["status"] == "failed"
    assert "research_memory_context" not in current
    assert current["errors"][-1]["code"] == "research_memory_search_failed"


def test_audit_write_failure_is_diagnosable_and_cancellation_stops_writes(store, monkeypatch):
    from app.agent.execution_budget import AgentExecutionCancelled
    from app.services.research_memory_service import ResearchMemoryService
    import app.services.research_memory_retrieval as module
    seed(store)
    current = state()
    search(current, query="跨模态")
    def fail(*args, **kwargs):
        raise OSError("unavailable")
    monkeypatch.setattr(ResearchMemoryService, "append_event", fail)
    record_memory_feedback(current, AgentDecision(action="request_finish"))
    assert current["errors"][-1]["code"] == "research_memory_write_failed"
    def stop():
        raise AgentExecutionCancelled("stop")
    monkeypatch.setattr(module, "check_execution", stop)
    with pytest.raises(AgentExecutionCancelled):
        search(current, query="跨模态")
    with pytest.raises(AgentExecutionCancelled):
        record_memory_feedback(current, AgentDecision(action="request_finish"))
