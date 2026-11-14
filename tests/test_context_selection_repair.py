"""真实E2E中论文ID误作摘要ID的回归；模型与外部服务均为替身。"""
import copy
import json

import pytest

from app.agent.context_builder import build_main_agent_context
from app.agent.delegation_context import ContextSelectionError, validate_planning
from app.agent.main_policy import AgentDecisionError, MainAgentPolicy
from app.schemas.agent_decision_schema import AgentDecision, ContextSelection
from app.schemas.context_schema import KeyEvidence
from scripts.e2e_observation import RunObserver
from tests.test_subagent_react import ScriptedLLM, state, store
from tests.test_main_delegation_context import agent_artifacts


PAPER_ID = "s2:659bf9ce7175e1ec266ff54359e2bd76e0b7ff31"


def test_latest_selection_catalog_changes_without_mutating_tool_prefix():
    class Capture(ScriptedLLM):
        def complete_tool_call(self, messages, *, tools, operation):
            self.offered.append(copy.deepcopy(tools))
            return super().complete_tool_call(messages, tools=tools, operation=operation)

    llm = Capture({"main_agent_decision": ["search_and_rank", "search_and_rank"]})
    llm.offered = []
    policy = MainAgentPolicy(llm)
    context = build_main_agent_context(state())
    policy.decide(context, allowed_actions=["search_and_rank"])
    first = json.loads(llm.calls[0][1][-1]["content"])
    assert first["state"]["context_selection_ids"]["evidence_ids"] == []
    context.key_evidence = [KeyEvidence(evidence_id="current-summary", finding="真实摘录", paper_ids=[PAPER_ID])]
    policy.decide(context, allowed_actions=["search_and_rank"])
    second = json.loads(llm.calls[1][1][-1]["content"])
    assert second["state"]["context_selection_ids"]["evidence_ids"] == ["current-summary"]
    assert llm.offered[0] == llm.offered[1] == policy.tools
    assert llm.calls[0][1][0] == llm.calls[1][1][0]
    assert set(first) == {"goal", "state", "key_evidence", "decisions", "open_questions"}


@pytest.mark.parametrize("native", [True, False])
def test_wrong_paper_id_is_rejected_then_model_repairs_with_empty_selection(native):
    current = state()
    current.update(ranked_papers=[{"paper_id": PAPER_ID, "title": "真实候选论文"}])
    context = build_main_agent_context(current)
    llm = ScriptedLLM({"main_agent_decision": [
        ("fetch_metadata", {"context_selection": {"evidence_ids": [PAPER_ID]}}),
        ("fetch_metadata", {"context_selection": {"evidence_ids": []}}),
    ]}, native=native)
    decision = MainAgentPolicy(llm).decide(context, allowed_actions=["fetch_metadata"])
    assert len(llm.calls) == 2
    assert decision.action == "fetch_metadata" and decision.context_selection.evidence_ids == []
    assert decision.arguments == {}
    decoded = []
    for message in llm.calls[1][1]:
        if message.get("role") != "user":
            continue
        value = json.loads(message["content"])
        if "context_selection_error" in value:
            decoded.append(value["context_selection_error"])
    assert decoded and decoded[-1]["invalid_ids"] == {"evidence_ids": [PAPER_ID]}
    assert decoded[-1]["allowed_ids"]["evidence_ids"] == []
    assert "paper_id" in decoded[-1]["repair"]
    assert current["ranked_papers"][0]["paper_id"] == PAPER_ID


@pytest.mark.parametrize("native", [True, False])
def test_repeated_wrong_ids_still_fail_with_original_bounded_retry(native):
    wrong = ("fetch_metadata", {"context_selection": {"evidence_ids": [PAPER_ID]}})
    llm = ScriptedLLM({"main_agent_decision": [wrong, wrong]}, native=native)
    with pytest.raises(AgentDecisionError, match="outside the current status bar"):
        MainAgentPolicy(llm).decide(build_main_agent_context(state()), allowed_actions=["fetch_metadata"])
    assert len(llm.calls) == 2


@pytest.mark.parametrize("field", ["memory_ids", "question_ids"])
def test_ids_from_another_namespace_are_not_accepted(field):
    context = build_main_agent_context(state())
    context.key_evidence = [KeyEvidence(evidence_id="e1", finding="证据摘要")]
    decision = AgentDecision(action="fetch_metadata", context_selection=ContextSelection(**{field: ["e1"]}))
    with pytest.raises(ContextSelectionError) as caught:
        validate_planning(decision, context)
    assert caught.value.invalid_ids == {field: ["e1"]}
    assert caught.value.allowed_ids["evidence_ids"] == ["e1"]
    assert "e1" not in caught.value.allowed_ids[field]
    assert str(caught.value) == "selected context ID is outside the current status bar"


def test_e2e_observer_keeps_rejected_selection_without_changing_exception(tmp_path):
    observer = RunObserver(tmp_path / "events.jsonl")
    decision = AgentDecision(action="fetch_metadata", context_selection=ContextSelection(evidence_ids=[PAPER_ID]))
    context = build_main_agent_context(state())
    # 从模块取函数，覆盖真实调用方的晚导入路径。
    from app.agent import delegation_context
    with observer.installed():
        with pytest.raises(ContextSelectionError):
            delegation_context.validate_planning(decision, context)
    event = observer.events[-1]
    assert event["kind"] == "decision_context_rejection"
    assert event["selection"]["evidence_ids"] == [PAPER_ID]
    assert event["invalid_ids"] == {"evidence_ids": [PAPER_ID]}
    assert event["allowed_ids"]["evidence_ids"] == []
    assert set(event) == {"sequence", "kind", "stage", "action", "selection", "invalid_ids", "allowed_ids"}


def _previous_plan(current):
    old = AgentDecision(action="search_and_rank", plan=[{"action": "fetch_metadata", "objective": "旧计划核验资料"}])
    current["main_agent_decisions"] = [old.model_dump(mode="json")]
    return "plan-" + old.decision_id + "-0"


def test_main_new_plan_does_not_invalidate_its_own_selected_previous_summary():
    from app.agent.main_loop import MainAgentLoop
    from tests.test_subagent_react import fetch, report
    current = state()
    current["ranked_papers"] = [{"paper_id": "p1", "title": "待获取论文"}]
    selected_id = _previous_plan(current)
    llm = ScriptedLLM({
        "main_agent_decision": [("fetch_metadata", {
            "plan": [{"action": "extract_paper_cards", "objective": "新的后续计划"}],
            "context_selection": {"decision_ids": [selected_id]},
        }), "request_finish"],
        "search_agent_decision": ["fetch_metadata", report()],
    })
    assert MainAgentLoop(llm).run(current, handlers={"fetch_metadata": fetch},
                                 finish_validator=lambda s: bool(s.get("paper_details"))) == "completed"
    received = json.loads(next(messages for op, messages in llm.calls if op == "search_agent_decision")[-1]["content"])
    assert received["decisions"][0]["decision_id"] == selected_id
    assert received["decisions"][0]["reason"] == "旧计划核验资料"
    assert current["paper_details"]


def test_selected_snapshot_survives_restart_after_main_plan_replacement(agent_artifacts):
    from app.agent.controller import AgentController
    from app.agent.subagents.loop import SubagentLoop
    from tests.test_subagent_react import fetch, report
    class Crash(BaseException):
        pass
    current = state()
    current["ranked_papers"] = [{"paper_id": "p1", "title": "待获取论文"}]
    selected_id = _previous_plan(current)
    snapshot = build_main_agent_context(current)
    decision = AgentDecision(action="fetch_metadata", context_selection=ContextSelection(decision_ids=[selected_id]),
                             plan=[{"action": "extract_paper_cards", "objective": "新计划"}])
    current["main_agent_decisions"].append(decision.model_dump(mode="json"))
    with pytest.raises(Crash):
        SubagentLoop(ScriptedLLM({"search_agent_decision": [Crash()]}), AgentController()).run(
            current, decision=decision, decision_context=snapshot, handlers={"fetch_metadata": fetch})
    snapshot.decisions[0].reason = "调用方后来修改快照"
    restored = agent_artifacts.hydrate_state("s", agent_artifacts.externalize_state("s", current))
    child = ScriptedLLM({"search_agent_decision": ["fetch_metadata", report()]})
    SubagentLoop(child, AgentController()).run(restored, handlers={"fetch_metadata": fetch})
    received = json.loads(child.calls[0][1][-1]["content"])
    assert received["decisions"][0]["decision_id"] == selected_id
    assert received["decisions"][0]["reason"] == "旧计划核验资料"


def test_prepared_snapshot_does_not_replace_latest_gate_or_goal():
    from app.agent.controller import AgentController
    from app.agent.subagents.loop import SubagentLoop
    from tests.test_subagent_react import search, report
    current = state()
    selected_id = _previous_plan(current)
    snapshot = build_main_agent_context(current)
    decision = AgentDecision(action="search_and_rank", context_selection=ContextSelection(decision_ids=[selected_id]))
    current["quality_gate"] = {"passed": False, "blocking_issues": [{"code": "latest_gate"}]}
    child = ScriptedLLM({"search_agent_decision": ["search_and_rank", report()]})
    SubagentLoop(child, AgentController()).run(current, decision=decision, decision_context=snapshot,
                                              handlers={"search_and_rank": search})
    received = json.loads(child.calls[0][1][-1]["content"])
    assert received["state"]["gate_status"]["quality"] is False
    assert snapshot.state.gate_status["quality"] is None
    assert received["goal"]["explicit_constraints"]
    changed = state()
    changed["user_query"] = "另一个用户目标"
    with pytest.raises(AgentDecisionError, match="goal changed"):
        SubagentLoop(ScriptedLLM({}), AgentController()).run(changed, decision=decision,
            decision_context=snapshot, handlers={"search_and_rank": search})


@pytest.mark.parametrize("native", [True, False])
def test_oversized_report_repairs_without_reexecuting_tool(native):
    from app.agent.controller import AgentController
    from app.agent.subagents.loop import SubagentLoop
    from tests.test_subagent_react import search, report
    current = state()
    calls = []
    def handler(working, args):
        calls.append("search")
        search(working, args)
    child = ScriptedLLM({"search_agent_decision": ["search_and_rank",
        report(message="X" * 1201), report(message="检索完成，候选1篇，请主控安排资料获取。") ]}, native=native)
    result = SubagentLoop(child, AgentController()).run(current,
        decision=AgentDecision(action="search_and_rank"), handlers={"search_and_rank": handler})
    assert result.delegation_status == "completed"
    assert calls == ["search"] and len(child.calls) == 3
    correction = next(json.loads(m["content"])["decision_error"] for m in child.calls[-1][1]
                      if m.get("role") == "user" and "decision_error" in json.loads(m["content"]))
    assert "1201" in correction and "1200" in correction and "300" in correction
    assert "X" * 20 not in correction


def test_repeated_oversized_report_returns_main_at_original_retry_limit():
    from app.agent.controller import AgentController
    from app.agent.subagents.loop import SubagentLoop
    from tests.test_subagent_react import search, report
    child = ScriptedLLM({"search_agent_decision": ["search_and_rank", report(message="X" * 1201), report(message="X" * 1201)]})
    current = state()
    result = SubagentLoop(child, AgentController()).run(current,
        decision=AgentDecision(action="search_and_rank"), handlers={"search_and_rank": search})
    assert result.delegation_status == "needs_help"
    assert "汇报协议" in result.findings[0]
    assert len(child.calls) == 3
    assert current["agent_execution_budget"]["actions_committed"] == 1


@pytest.mark.parametrize("passes", [True, False])
def test_unchanged_candidate_is_validated_before_no_progress_decision(monkeypatch, passes):
    from app.agent.main_loop import MainAgentLoop
    from app.core.config import get_settings
    from tests.test_subagent_react import report
    monkeypatch.setattr(get_settings(), "agent_main_no_progress_limit", 1)
    current = state()
    current.update(paper_cards=[{"paper_id": "p1", "title": "证据"}], claim_plans=[{"claim_id": "c1"}],
                   review="相同的候选正文", quality_gate={"passed": False, "draft_released": False})
    seen = []
    def write(working, args):
        seen.append("write")
        working.update(review="相同的候选正文", pending_result_validation=True)
    def verify(working, args):
        seen.append("verify")
        working.update(quality_gate={"passed": passes, "draft_released": passes})
        from tests.agent_decision_fakes import checked_result
        checked_result(working)
    llm = ScriptedLLM({"main_agent_decision": ["generate_deliverables", "validate_result", "request_finish"],
                       "writing_agent_decision": ["generate_deliverables", report()],
                       "check_agent_decision": ["validate_result", report()]})
    result = MainAgentLoop(llm).run(current, handlers={"generate_deliverables": write, "validate_result": verify},
        finish_validator=lambda s: not s.get("pending_result_validation") and s["quality_gate"]["passed"])
    assert seen == ["write", "verify"]
    assert result == ("completed" if passes else "blocked")
    assert current["quality_gate"]["draft_released"] == passes
    assert current["required_reference_count"] == 40


def test_candidate_cycle_is_checked_after_validation(monkeypatch, store):
    from app.agent.main_loop import MainAgentLoop
    from app.core.config import get_settings
    from tests.test_subagent_react import report
    monkeypatch.setattr(get_settings(), "agent_main_no_progress_limit", 3)
    current = state()
    current.update(paper_cards=[{"paper_id": "p1", "title": "证据"}], claim_plans=[{"claim_id": "c1"}],
                   review="A", quality_gate={"passed": False})
    drafts = iter(["B", "A"])
    checks = []
    def write(working, args):
        working.update(review=next(drafts), pending_result_validation=True)
    def verify(working, args):
        checks.append(working["review"])
        working.update(quality_gate={"passed": len(checks) == 2})
        from tests.agent_decision_fakes import checked_result
        checked_result(working)
    llm = ScriptedLLM({"main_agent_decision": ["generate_deliverables", "validate_result", "generate_deliverables", "validate_result", "request_finish"],
                       "writing_agent_decision": ["generate_deliverables", report(), "generate_deliverables", report()],
                       "check_agent_decision": ["validate_result", report(), "validate_result", report()]})
    result = MainAgentLoop(llm).run(current, handlers={"generate_deliverables": write, "validate_result": verify},
        finish_validator=lambda s: not s.get("pending_result_validation") and s["quality_gate"]["passed"])
    assert checks == ["B", "A"]
    assert result == "completed"


@pytest.mark.parametrize("native", [True, False])
def test_finished_validation_cannot_be_repeated_and_repairs_to_report(native):
    from app.agent.controller import AgentController
    from app.agent.subagents.loop import SubagentLoop
    from tests.test_subagent_react import report
    current = state()
    current["review"] = "待验证正文"
    seen = []
    def verify(working, args):
        seen.append("verify")
        working["quality_gate"] = {"passed": False}
    llm = ScriptedLLM({"check_agent_decision": ["validate_result", "validate_result",
        report(status="needs_help", message="核验完成但引用不足，请主控安排恢复。", suggestions=["diagnose_evidence_gaps"])]}, native=native)
    result = SubagentLoop(llm, AgentController()).run(current,
        decision=AgentDecision(action="validate_result"), handlers={"validate_result": verify})
    assert result.delegation_status == "needs_help" and seen == ["verify"]
    correction = next(json.loads(m["content"]) for m in llm.calls[-1][1]
                      if m.get("role") == "user" and "decision_error" in json.loads(m["content"]))
    assert correction["allowed_actions"] == ["report_to_main", "recall_context"]
    assert correction["completed_operations"] == ["validate_result"]


def test_finished_validation_returns_main_after_bounded_illegal_report_attempts():
    from app.agent.controller import AgentController
    from app.agent.subagents.loop import SubagentLoop
    current = state()
    current["review"] = "待验证正文"
    seen = []
    def verify(working, args):
        seen.append("verify")
        working["quality_gate"] = {"passed": False}
    llm = ScriptedLLM({"check_agent_decision": ["validate_result", "validate_result", "validate_result"]})
    result = SubagentLoop(llm, AgentController()).run(current,
        decision=AgentDecision(action="validate_result"), handlers={"validate_result": verify})
    assert result.delegation_status == "needs_help"
    assert "汇报协议" in result.findings[0]
    assert seen == ["verify"] and len(llm.calls) == 3
