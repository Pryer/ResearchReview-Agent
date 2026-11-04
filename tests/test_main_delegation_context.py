"""主控规划、上下文选择与操作级委派的行为证明。"""
import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.agent.context_builder import build_main_agent_context
from app.agent.main_policy import MainAgentPolicy, AgentDecisionError
from app.agent.subagents.loop import SubagentLoop
from app.agent.controller import AgentController
from app.schemas.agent_decision_schema import AgentDecision, ContextSelection
from tests.test_subagent_react import ScriptedLLM, state, search, report


@pytest.fixture
def agent_artifacts():
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


@pytest.mark.parametrize("native", [True, False])
def test_main_selects_context_and_plan_without_changing_execution_arguments(native):
    current = state()
    current.update(selected_scope={"label": "用户已确认范围", "confirmed_by_user": True},
                   evidence_gap_report={"needs_recovery": True, "gaps": [{"route_id": "gap-a", "reason": "缺少对照实验"}]},
                   research_memory_context={"matches": [{"memory_id": "m1", "summary": "该来源曾限流"}]})
    context = build_main_agent_context(current)
    payload = {"plan": [{"action": "search_and_rank", "objective": "补充候选"},
                        {"action": "fetch_metadata", "objective": "核验来源"}],
               "context_selection": {"decision_ids": ["selected-scope"], "question_ids": ["gap-a"], "memory_ids": ["m1"]}}
    llm = ScriptedLLM({"main_agent_decision": [("search_and_rank", payload)]}, native=native)
    decision = MainAgentPolicy(llm).decide(context, allowed_actions=["search_and_rank"])
    assert decision.arguments == {}
    assert decision.plan[1].action == "fetch_metadata"
    child = ScriptedLLM({"search_agent_decision": ["search_and_rank", report()]})
    SubagentLoop(child, AgentController()).run(current, decision=decision, handlers={"search_and_rank": search})
    received = json.loads(child.calls[0][1][-1]["content"])
    assert received["decisions"][0]["decision_id"] == "selected-scope"
    assert received["open_questions"][0]["question_id"] == "gap-a"
    assert received["state"]["research_memory"][0]["memory_id"] == "m1"
    assert received["state"]["delegation"]["context_selection"]["question_ids"] == ["gap-a"]
    assert received["goal"]["explicit_constraints"]


@pytest.mark.parametrize("selection", [{"evidence_ids": ["invented"]}, {"decision_ids": ["invented"]},
                                       {"question_ids": ["invented"]}, {"memory_ids": ["invented"]}])
def test_main_cannot_dispatch_unknown_context_ids(selection):
    llm = ScriptedLLM({"main_agent_decision": [("search_and_rank", {"context_selection": selection})]})
    with pytest.raises(AgentDecisionError, match="outside the current status bar"):
        MainAgentPolicy(llm, correction_attempts=0).decide(build_main_agent_context(state()), allowed_actions=["search_and_rank"])


def test_explicit_empty_selection_cannot_remove_constraints_or_latest_gate():
    current = state()
    current.update(quality_gate={"passed": False, "blocking_issues": [{"code": "missing_evidence"}]},
                   paper_details=[{"paper_id": "p1"}],
                   paper_cards=[{"paper_id": "p1", "title": "已读取论文"}],
                   claim_plans=[{"claims": [{"claim_id": "secret", "claim_text": "UNRELATED_CLAIM", "supporting_paper_ids": ["p1"]}]}])
    child = ScriptedLLM({"search_agent_decision": ["search_and_rank", report()]})
    SubagentLoop(child, AgentController()).run(current,
        decision=AgentDecision(action="search_and_rank", context_selection=ContextSelection()), handlers={"search_and_rank": search})
    received = json.loads(child.calls[0][1][-1]["content"])
    assert received["goal"]["explicit_constraints"]
    assert received["state"]["gate_status"]
    assert not received["key_evidence"]
    assert "UNRELATED_CLAIM" not in json.dumps(received)


def test_selected_packet_survives_checkpoint_without_granting_other_operations(agent_artifacts):
    class Crash(BaseException):
        pass
    current = state()
    current["selected_scope"] = {"label": "研究方向"}
    child = ScriptedLLM({"search_agent_decision": [Crash()]})
    with pytest.raises(Crash):
        SubagentLoop(child, AgentController()).run(current,
            decision=AgentDecision(action="search_and_rank", context_selection=ContextSelection(decision_ids=["selected-scope"])),
            handlers={"search_and_rank": search})
    saved = agent_artifacts.externalize_state("s", current)
    restored = agent_artifacts.hydrate_state("s", saved)
    resumed = ScriptedLLM({"search_agent_decision": ["search_and_rank", report()]})
    SubagentLoop(resumed, AgentController()).run(restored, handlers={"search_and_rank": search})
    received = json.loads(resumed.calls[0][1][-1]["content"])
    assert received["decisions"][0]["decision_id"] == "selected-scope"
    assert received["state"]["allowed_actions"] == ["search_and_rank", "report_to_main", "recall_context"]


def test_previous_candidate_comparison_is_service_only_and_fragmented(agent_artifacts):
    original = {"pending_generation_previous": {"products": {"review": "历史草稿" * 5000}}}
    stored = agent_artifacts.externalize_state("s", original)
    assert "pending_generation_previous" not in stored
    assert len(stored["artifact_manifest"]["pending_generation_previous"]["items"]) > 1
    assert agent_artifacts.hydrate_state("s", stored)["pending_generation_previous"] == original["pending_generation_previous"]
    with pytest.raises(PermissionError):
        agent_artifacts.hydrate_state("s", stored, role="writing")
