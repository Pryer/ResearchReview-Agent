"""独立循环、双向通信、权限与真实检查点恢复的行为证明。"""
import copy
import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.agent.controller import AgentController
from app.agent.execution_budget import AgentBudgetExceeded, AgentExecutionCancelled, budget_scope
from app.agent.main_loop import MainAgentLoop
from app.agent.main_policy import AgentDecisionError
from app.agent.react_context import ReActContext, encode
from app.agent.subagents.loop import SubagentLoop
from app.core.config import get_settings
from app.database.models import Base
from app.schemas.agent_decision_schema import AgentDecision
from app.services.durable_execution_service import durable_scope
from app.services.llm_service import LLMService
from app.services.research_artifact_service import ResearchArtifactService
from app.services.research_execution_service import _STORE


class ScriptedLLM:
    def __init__(self, scripts, native=True):
        self.scripts = {key: iter(items) for key, items in scripts.items()}
        self.native_tools_enabled = native
        self.calls = []

    def _next(self, messages, operation):
        LLMService._validate_messages(messages)
        self.calls.append((operation, copy.deepcopy(messages)))
        item = next(self.scripts[operation])
        if isinstance(item, BaseException):
            raise item
        return (item, {}) if isinstance(item, str) else item

    def complete_tool_call(self, messages, *, tools, operation):
        name, arguments = self._next(messages, operation)
        call_id = "call-" + str(len(self.calls))
        return {"name": name, "arguments": arguments, "tool_call_id": call_id,
                "assistant_message": {"role": "assistant", "content": None, "tool_calls": [{
                    "id": call_id, "type": "function", "function": {"name": name, "arguments": encode(arguments)},
                }]}}

    def complete_messages(self, messages, *, operation, **kwargs):
        name, arguments = self._next(messages, operation)
        return encode({"action": name, "arguments": arguments})


def report(status="completed", message="证据任务已完成", suggestions=()):
    return "report_to_main", {"status": status, "message": message, "suggested_actions": list(suggestions)}


def state():
    return {"session_id": "s", "topic": "主题", "user_query": "近三年主题研究，40篇文献",
            "required_reference_count": 40, "max_papers_explicit": True,
            "steps": [], "errors": [], "research_plan": {"task_graph": []}}


def search(current, args):
    current.update(candidate_papers=[{"paper_id": "p1", "title": "检索论文"}],
                   ranked_papers=[{"paper_id": "p1", "title": "检索论文"}])


def fetch(current, args):
    current["paper_details"] = [{"paper_id": "p1", "title": "已核验论文"}]


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


@pytest.mark.parametrize("native", [True, False])
def test_one_delegation_runs_multiple_decisions_and_reports_to_main(native):
    current = state()
    llm = ScriptedLLM({
        "main_agent_decision": ["search_and_rank", "request_finish"],
        "search_agent_decision": ["search_and_rank", "fetch_metadata", report(message="已检索并核验来源")],
    }, native)
    assert MainAgentLoop(llm).run(current, handlers={"search_and_rank": search, "fetch_metadata": fetch},
                                  finish_validator=lambda s: bool(s.get("paper_details"))) == "completed"
    child_calls = [m for op, m in llm.calls if op == "search_agent_decision"]
    assert len(child_calls) == 3
    assert child_calls[1][:len(child_calls[0])] == child_calls[0]
    assert "检索候选 1" in encode(child_calls[1])
    main_last = [m for op, m in llm.calls if op == "main_agent_decision"][-1]
    assert "已检索并核验来源" in encode(main_last)
    child = current["subagent_sessions"]["search"]
    assert [m["kind"] for m in child["messages"]] == ["task", "progress", "progress", "completed"]
    assert not current.get("active_subagent")
    assert len(current["main_agent_trajectory"]["memory"]) == 2
    assert len(child["main_agent_trajectory"]["memory"]) == 3
    assert current["agent_execution_budget"]["actions_committed"] == 2


def test_context_survives_new_delegation_without_sibling_private_history():
    current = state()
    current["subagent_sessions"] = {"writing": {"private_secret": "sibling-private-history"}}
    llm = ScriptedLLM({"search_agent_decision": ["search_and_rank", report(message="remember-search"),
                                                "fetch_metadata", report()]})
    runner = SubagentLoop(llm, AgentController())
    with budget_scope(current):
        runner.run(current, decision=AgentDecision(action="search_and_rank"),
                   handlers={"search_and_rank": search, "fetch_metadata": fetch})
        runner.run(current, decision=AgentDecision(action="fetch_metadata"),
                   handlers={"search_and_rank": search, "fetch_metadata": fetch})
    third = llm.calls[2][1]
    assert "remember-search" in encode(third)
    assert "sibling-private-history" not in encode(third)
    assert current["subagent_sessions"]["writing"] == {"private_secret": "sibling-private-history"}


def test_help_request_returns_to_main_and_reply_reaches_same_role():
    current = state()
    llm = ScriptedLLM({
        "main_agent_decision": ["search_and_rank", ("search_and_rank", {"delegation_message": "请按原时间范围检索"}), "request_finish"],
        "search_agent_decision": [report("needs_help", "请确认检索目标", ["search_and_rank"]),
                                  "search_and_rank", report()],
    })
    assert MainAgentLoop(llm).run(current, handlers={"search_and_rank": search},
                                  finish_validator=lambda s: bool(s.get("ranked_papers"))) == "completed"
    main_second = [m for op, m in llm.calls if op == "main_agent_decision"][1]
    assert "请确认检索目标" in encode(main_second)
    messages = current["subagent_sessions"]["search"]["messages"]
    assert [m["kind"] for m in messages] == ["task", "needs_help", "task", "progress", "completed"]
    assert messages[0]["delegation_id"] != messages[2]["delegation_id"]
    assert messages[2]["content"] == "请按原时间范围检索"
    child_second = [m for op, m in llm.calls if op == "search_agent_decision"][1]
    assert "请按原时间范围检索" in encode(child_second)
    assert "delegation_message" not in current["subagent_sessions"]["search"]["last_delegation"]["arguments"]


@pytest.mark.parametrize("native", [True, False])
def test_report_contract_lists_registered_suggestions_without_granting_execution(native):
    from app.agent.action_registry import ACTION_REGISTRY
    class ContractAwareLLM(ScriptedLLM):
        def check_contract(self, tools):
            tool = next(item for item in tools if item["function"]["name"] == "report_to_main")
            items = tool["function"]["parameters"]["properties"]["suggested_actions"]["items"]
            assert set(items.get("enum") or []) == set(ACTION_REGISTRY)
        def complete_tool_call(self, messages, *, tools, operation):
            self.check_contract(tools)
            return super().complete_tool_call(messages, tools=tools, operation=operation)
        def complete_messages(self, messages, *, operation, **kwargs):
            catalog = messages[0]["content"].split("核心工具目录（JSON 动作契约）：", 1)[1]
            self.check_contract(json.loads(catalog))
            return super().complete_messages(messages, operation=operation, **kwargs)
    current = state()
    llm = ContractAwareLLM({"search_agent_decision": [
        report("needs_help", "需要证据分析角色处理", ["extract_paper_cards"])]}, native=native)
    with budget_scope(current):
        result = SubagentLoop(llm, AgentController()).run(
            current, decision=AgentDecision(action="search_and_rank"), handlers={"search_and_rank": search})
    assert result.delegation_status == "needs_help"
    assert not current.get("candidate_papers")


def test_natural_language_suggestion_is_rejected_then_corrected_without_new_permissions():
    current = state()
    llm = ScriptedLLM({"search_agent_decision": [
        report("needs_help", "需要主控协助", ["请主控继续分析论文"]),
        report("needs_help", "需要主控协助", ["extract_paper_cards"]),
    ]})
    with budget_scope(current):
        result = SubagentLoop(llm, AgentController()).run(
            current, decision=AgentDecision(action="search_and_rank"), handlers={"search_and_rank": search})
    assert result.delegation_status == "needs_help"
    assert not current.get("paper_cards")
    assert "registered action names" in encode(llm.calls[1][1])


@pytest.mark.parametrize("action", ["generate_deliverables", "request_finish"])
def test_search_cannot_execute_other_role_or_release_review(action):
    current = state()
    llm = ScriptedLLM({"search_agent_decision": [action, action]})
    calls = []
    with pytest.raises(AgentDecisionError):
        SubagentLoop(llm, AgentController()).run(current, decision=AgentDecision(action="search_and_rank"),
            handlers={"generate_deliverables": lambda *args: calls.append(1), "search_and_rank": search})
    assert calls == []
    assert not ReActContext(current, namespace="search").data.get("pending")


def test_report_does_not_satisfy_required_operation_or_final_gate():
    current = state()
    llm = ScriptedLLM({
        "main_agent_decision": ["search_and_rank", "request_finish", "report_blocked"],
        "search_agent_decision": [report(), "search_and_rank", report()],
    })
    llm.scripts["main_agent_decision"] = iter(["search_and_rank", "request_finish",
        ("report_blocked", {"reason": "最终门禁未通过"})])
    assert MainAgentLoop(llm).run(current, handlers={"search_and_rank": search},
                                  finish_validator=lambda s: False) == "blocked"
    memories = current["subagent_sessions"]["search"]["main_agent_trajectory"]["memory"]
    assert memories[0]["error"] == "delegated_operation_not_completed"
    assert current["main_agent_rejections"][-1]["reason"] == "finish_rejected_by_quality_gates"


def test_rewrite_cannot_change_main_section_constraints():
    current = state()
    current.update(review="正文", writing_plans=[{"section_id": "approved"}])
    llm = ScriptedLLM({"writing_agent_decision": [
        ("rewrite_sections", {"section_ids": ["other"]}),
        ("rewrite_sections", {"section_ids": ["approved"]}), report()]})
    calls = []
    SubagentLoop(llm, AgentController()).run(current,
        decision=AgentDecision(action="rewrite_sections", arguments={"section_ids": ["approved"]}),
        handlers={"rewrite_sections": lambda s, a: calls.append(a)})
    assert calls == [{"section_ids": ["approved"]}]
    assert current["subagent_sessions"]["writing"]["main_agent_trajectory"]["memory"][0]["status"] == "rejected"


def test_evidence_invalidation_precedes_next_child_decision():
    current = state()
    current.update(ranked_papers=[{"paper_id": "p1"}], paper_details=[{"paper_id": "old"}],
                   paper_cards=[{"paper_id": "old"}], claim_plans=[{"claim_id": "stale"}],
                   validated_routes=[{"route_id": "stale"}], quality_gate={"passed": True})
    llm = ScriptedLLM({"search_agent_decision": ["fetch_metadata", report()]})
    SubagentLoop(llm, AgentController()).run(current, decision=AgentDecision(action="fetch_metadata"),
                                           handlers={"fetch_metadata": fetch})
    assert current["paper_cards"] == []
    assert not current.get("claim_plans") and not current.get("quality_gate")
    context = json.loads(llm.calls[1][1][-1]["content"])
    assert context["state"]["artifact_summary"]["paper_cards"] == 0


@pytest.mark.parametrize("stop", [AgentBudgetExceeded("budget"), AgentExecutionCancelled("cancel")])
def test_control_stop_propagates_without_additional_decision(stop):
    current = state()
    llm = ScriptedLLM({"search_agent_decision": ["search_and_rank", report()]})
    def fail(*args):
        raise stop
    with pytest.raises(type(stop)) as caught:
        SubagentLoop(llm, AgentController()).run(current, decision=AgentDecision(action="search_and_rank"),
                                               handlers={"search_and_rank": fail})
    assert caught.value is stop
    assert len(llm.calls) == 1
    assert not ReActContext(current, namespace="search").data.get("pending")


def test_no_progress_and_round_limits_return_blocked_to_main(monkeypatch):
    monkeypatch.setattr(get_settings(), "agent_subagent_no_progress_limit", 2)
    current = state()
    llm = ScriptedLLM({"search_agent_decision": ["search_and_rank"] * 4})
    result = SubagentLoop(llm, AgentController()).run(current, decision=AgentDecision(action="search_and_rank"),
                                                   handlers={"search_and_rank": lambda *args: None})
    assert result.outcome.value == "blocked"
    assert len(llm.calls) == 2
    monkeypatch.setattr(get_settings(), "agent_subagent_max_rounds", 1)
    current = state()
    llm = ScriptedLLM({"search_agent_decision": ["search_and_rank", report()]})
    result = SubagentLoop(llm, AgentController()).run(current, decision=AgentDecision(action="search_and_rank"),
                                                   handlers={"search_and_rank": search})
    assert result.outcome.value == "blocked"
    assert len(llm.calls) == 1


def test_checkpoint_resumes_child_without_replaying_committed_action(store):
    class Crash(BaseException):
        pass
    current = state()
    calls = []
    def counted_search(s, a):
        calls.append("search")
        search(s, a)
    first = ScriptedLLM({"main_agent_decision": ["search_and_rank"],
                        "search_agent_decision": ["search_and_rank", Crash()]})
    with budget_scope(current):
        with durable_scope(store.repo.db, "s", current, current["agent_execution_budget"]) as runtime:
            with pytest.raises(Crash):
                MainAgentLoop(first).run(current, handlers={"search_and_rank": counted_search, "fetch_metadata": fetch},
                                         finish_validator=lambda s: bool(s.get("paper_details")))
            restored = runtime.restore()
            assert restored["active_subagent"] == "search"
            child = restored["subagent_sessions"]["search"]
            assert child["active"]["completed_operations"] == ["search_and_rank"]
            assert not child["main_agent_trajectory"].get("pending")
            assert restored["main_agent_trajectory"]["pending"]["action"] == "search_and_rank"
            second = ScriptedLLM({"main_agent_decision": ["request_finish"],
                                 "search_agent_decision": ["fetch_metadata", report()]})
            assert MainAgentLoop(second).run(restored,
                handlers={"search_and_rank": counted_search, "fetch_metadata": fetch},
                finish_validator=lambda s: bool(s.get("paper_details"))) == "completed"
            saved = runtime.restore()
            assert saved["paper_details"]
            assert not saved.get("active_subagent")
            assert not saved["main_agent_trajectory"].get("pending")
    assert calls == ["search"]


def test_subagent_context_externalization_is_service_only_and_compacts_independently(store):
    current = state()
    main = ReActContext(current)
    child = ReActContext(current, namespace="analysis")
    for i in range(4):
        child.proposed(AgentDecision(action="validate_routes", reason=str(i)))
        child.observe({"status": "completed", "findings": ["观察" + str(i)]})
    before = copy.deepcopy(main.data)
    from app.agent.context_builder import build_main_agent_context
    child._compact(build_main_agent_context(current).model_dump(mode="json"))
    assert child.data["epoch"] == 1
    assert main.data == before
    stored = store.externalize_state("s", current)
    assert "subagent_sessions" not in stored
    restored = store.hydrate_state("s", stored)
    assert restored["subagent_sessions"] == current["subagent_sessions"]
    with pytest.raises(PermissionError):
        store.hydrate_state("s", stored, role="search")


def test_all_three_roles_have_separate_decisions_contexts_and_feedback():
    current = state()
    llm = ScriptedLLM({
        "main_agent_decision": ["search_and_rank", "extract_paper_cards", "generate_deliverables", "request_finish"],
        "search_agent_decision": ["search_and_rank", "fetch_metadata", report(message="search-private-turn")],
        "analysis_agent_decision": ["extract_paper_cards", "plan_claims", report(message="analysis-private-turn")],
        "writing_agent_decision": ["generate_deliverables", report(message="writing-private-turn")],
    })
    handlers = {
        "search_and_rank": search, "fetch_metadata": fetch,
        "extract_paper_cards": lambda s, a: s.update(paper_cards=[{"paper_id": "p1", "title": "证据"}]),
        "plan_claims": lambda s, a: s.update(claim_plans=[{"claim_id": "c1"}]),
        "generate_deliverables": lambda s, a: s.update(review="已授权正文", quality_gate={"passed": True}),
    }
    assert MainAgentLoop(llm).run(current, handlers=handlers,
                                  finish_validator=lambda s: bool((s.get("quality_gate") or {}).get("passed"))) == "completed"
    for role, other in (("analysis", "search"), ("writing", "analysis"), ("search", "writing")):
        calls = [messages for op, messages in llm.calls if op == role + "_agent_decision"]
        assert len(calls) >= 2
        assert other + "-private-turn" not in encode(calls)
        assert not current["subagent_sessions"][role]["main_agent_trajectory"].get("pending")


def test_subagent_actions_share_total_budget():
    current = state()
    current["agent_execution_budget"] = {"action_limit": 1}
    llm = ScriptedLLM({"main_agent_decision": ["search_and_rank"],
                       "search_agent_decision": ["search_and_rank", "fetch_metadata"]})
    assert MainAgentLoop(llm).run(current, handlers={"search_and_rank": search, "fetch_metadata": fetch},
                                  finish_validator=lambda s: False) == "blocked"
    assert not current.get("paper_details")
    assert current["agent_execution_budget"]["actions_committed"] == 1


def test_mandatory_recovery_cannot_be_preempted_by_child_choice():
    current = state()
    current.update(ranked_papers=[{"paper_id": "old"}],
                   agent_mandatory_actions=[{"action": "search_and_rank"}])
    llm = ScriptedLLM({"main_agent_decision": ["request_finish"],
        "search_agent_decision": ["fetch_metadata", "search_and_rank", report()]})
    calls = []
    assert MainAgentLoop(llm).run(current,
        handlers={"search_and_rank": search, "fetch_metadata": lambda s, a: calls.append(1)},
        finish_validator=lambda s: bool(s.get("candidate_papers"))) == "completed"
    assert calls == []
    assert not current.get("agent_mandatory_actions")


def test_local_rewrite_cannot_switch_to_full_generation():
    current = state()
    current.update(review="正文", writing_plans=[{"section_id": "approved"}],
                   paper_cards=[{"paper_id": "p1"}], claim_plans=[{"claim_id": "c1"}])
    llm = ScriptedLLM({"writing_agent_decision": ["generate_deliverables", "generate_deliverables"]})
    with pytest.raises(AgentDecisionError):
        SubagentLoop(llm, AgentController()).run(current,
            decision=AgentDecision(action="rewrite_sections", arguments={"section_ids": ["approved"]}),
            handlers={"rewrite_sections": lambda *a: None, "generate_deliverables": lambda *a: pytest.fail("bypass")})


def test_goal_change_invalidates_persisted_delegation_without_running_old_work(store):
    class Crash(BaseException):
        pass
    current = state()
    llm = ScriptedLLM({"search_agent_decision": [Crash()]})
    with pytest.raises(Crash):
        SubagentLoop(llm, AgentController()).run(current, decision=AgentDecision(action="search_and_rank"),
                                               handlers={"search_and_rank": search})
    stored = store.externalize_state("s", current)
    restored = store.hydrate_state("s", stored)
    restored["required_reference_count"] = 50
    result = SubagentLoop(ScriptedLLM({}), AgentController()).run(restored,
        handlers={"search_and_rank": lambda *a: pytest.fail("old objective")})
    assert result.open_questions[0]["category"] == "needs_help"
    assert restored["required_reference_count"] == 50


def test_child_decision_has_sidecar_check_before_extra_action(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(get_settings(), "agent_sidecar_enabled", True)
    monkeypatch.setattr(get_settings(), "agent_sidecar_review_actions", "fetch_metadata")
    current = state()
    llm = ScriptedLLM({"search_agent_decision": ["search_and_rank", "fetch_metadata", report()]})
    runner = SubagentLoop(llm, AgentController())
    monkeypatch.setattr(runner.sidecar, "review", lambda **kw: SimpleNamespace(verdict="deny", reason="review denied"))
    runner.run(current, decision=AgentDecision(action="search_and_rank"),
               handlers={"search_and_rank": search, "fetch_metadata": lambda *a: pytest.fail("denied action")})
    assert any(item["status"] == "deny" for item in current["subagent_sessions"]["search"]["main_agent_trajectory"]["memory"])


def test_native_fallback_is_scoped_to_child_context():
    from app.core.exceptions import NativeToolsUnsupportedError
    class ChildFallback(ScriptedLLM):
        def complete_tool_call(self, messages, *, tools, operation):
            if operation == "search_agent_decision":
                raise NativeToolsUnsupportedError("unsupported")
            return super().complete_tool_call(messages, tools=tools, operation=operation)
    llm = ChildFallback({"main_agent_decision": ["search_and_rank", "request_finish"],
                         "search_agent_decision": ["search_and_rank", report()]})
    current = state()
    assert MainAgentLoop(llm).run(current, handlers={"search_and_rank": search},
                                  finish_validator=lambda s: bool(s.get("ranked_papers"))) == "completed"
    assert current["subagent_sessions"]["search"]["main_agent_trajectory"]["native_tools_disabled"]
    assert not current["main_agent_trajectory"].get("native_tools_disabled")


def test_next_mandatory_action_is_not_executed_without_its_own_dequeue():
    current = state()
    current["agent_mandatory_actions"] = [{"action": "search_and_rank"}, {"action": "fetch_metadata"}]
    llm = ScriptedLLM({"main_agent_decision": ["request_finish"], "search_agent_decision": [
        "search_and_rank", "fetch_metadata", report(), "fetch_metadata", report(),
    ]})
    assert MainAgentLoop(llm).run(current, handlers={"search_and_rank": search, "fetch_metadata": fetch},
                                  finish_validator=lambda s: bool(s.get("paper_details"))) == "completed"
    assert not current.get("agent_mandatory_actions")
    assert current["agent_execution_budget"]["actions_committed"] == 2
    assert len([m for m in current["subagent_sessions"]["search"]["messages"] if m["kind"] == "task"]) == 2


def test_delegation_metadata_does_not_mutate_registry_or_execution_contract():
    from app.agent.action_registry import tool_schemas
    from app.agent.context_builder import build_main_agent_context
    from app.agent.main_policy import MainAgentPolicy
    before = copy.deepcopy(tool_schemas(["search_and_rank"]))
    llm = ScriptedLLM({"main_agent_decision": [("search_and_rank", {"delegation_message": "补充有来源的论文"})]})
    policy = MainAgentPolicy(llm)
    decision = policy.decide(build_main_agent_context(state()), allowed_actions=["search_and_rank"])
    assert decision.reason == "补充有来源的论文"
    assert decision.arguments == {}
    assert tool_schemas(["search_and_rank"]) == before
