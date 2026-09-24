"""五字段主控循环的动作、终态与无进展边界。"""

from __future__ import annotations

from app.agent.main_loop import MainAgentLoop, _progress_fingerprint


class NativeDecisionLLM:
    native_tools_enabled = True

    def __init__(self, actions):
        self.actions = iter(actions)
        self.context_messages = []

    def complete_tool_call(self, messages, *, tools, operation):
        self.context_messages.append(messages)
        action = next(self.actions)
        return {"name": action, "arguments": {}, "tool_call_id": action}


def _state():
    return {
        "state_schema_version": "2", "user_query": "主题", "topic": "主题",
        "core_deliverables": [], "steps": [], "errors": [],
        "research_plan": {"task_graph": []},
    }


def test_finish_request_is_rejected_until_deterministic_validator_passes():
    state = _state()
    llm = NativeDecisionLLM(["request_finish", "search_and_rank", "request_finish"])

    def search(working, arguments):
        working["candidate_papers"] = [{"paper_id": "p1"}]

    def finish(current):
        if current.get("candidate_papers"):
            current["result_status"] = "completed"
            return True
        return False

    status = MainAgentLoop(llm).run(
        state, handlers={"search_and_rank": search}, finish_validator=finish
    )

    assert status == "completed"
    assert state["candidate_papers"] == [{"paper_id": "p1"}]
    assert state["main_agent_rejections"][0]["reason"] == "finish_rejected_by_quality_gates"
    assert len(llm.context_messages) == 3


def test_repeated_real_actions_without_progress_are_blocked():
    state = _state()
    llm = NativeDecisionLLM(["search_and_rank"] * 3)
    status = MainAgentLoop(llm).run(
        state, handlers={"search_and_rank": lambda s, a: None}, finish_validator=lambda s: False,
    )
    assert status == "blocked"
    assert state["errors"][-1]["code"] == "agent_no_progress"
    assert len(llm.context_messages) == 3


def test_rank_order_change_does_not_count_as_research_progress():
    first = _state()
    first["candidate_papers"] = [{"paper_id": "p1"}, {"paper_id": "p2"}]
    second = _state()
    second["candidate_papers"] = list(reversed(first["candidate_papers"]))
    assert _progress_fingerprint(first) == _progress_fingerprint(second)


def test_alternating_research_states_stop_before_round_budget():
    state = _state()
    llm = NativeDecisionLLM(["search_and_rank"] * 6)
    calls = 0

    def search(working, arguments):
        nonlocal calls
        calls += 1
        working["candidate_papers"] = [{"paper_id": "p1" if calls % 2 else "p2"}]

    status = MainAgentLoop(llm).run(
        state, handlers={"search_and_rank": search}, finish_validator=lambda s: False,
    )

    assert status == "blocked"
    assert state["errors"][-1]["code"] == "agent_state_cycle"
    assert calls == 3


def test_preselected_recovery_action_runs_before_model_finish():
    state = _state()
    state["agent_mandatory_actions"] = [{"action": "search_and_rank"}]
    llm = NativeDecisionLLM(["request_finish"])

    def search(working, arguments):
        working["candidate_papers"] = [{"paper_id": "p1"}]

    def finish(current):
        if current.get("candidate_papers"):
            current["result_status"] = "completed"
            return True
        return False

    status = MainAgentLoop(llm).run(
        state, handlers={"search_and_rank": search}, finish_validator=finish,
    )

    assert status == "completed"
    assert "agent_mandatory_actions" not in state
    assert [item["action"] for item in state["main_agent_decisions"]] == [
        "search_and_rank", "request_finish",
    ]
    assert len(llm.context_messages) == 1


def test_unavailable_preselected_recovery_action_blocks_explicitly():
    state = _state()
    state["agent_mandatory_actions"] = [{"action": "generate_deliverables"}]
    llm = NativeDecisionLLM([])

    status = MainAgentLoop(llm).run(
        state, handlers={"generate_deliverables": lambda working, arguments: None},
        finish_validator=lambda current: False,
    )

    assert status == "blocked"
    assert state["errors"][-1]["code"] == "required_recovery_action_unavailable"
    assert llm.context_messages == []


def test_autonomous_graph_pipeline_reuses_existing_nodes_and_quality_gate(monkeypatch):
    from app.agent import graph

    state = _state()
    state.update({
        "intent": "generate_review",
        "core_deliverables": ["narrative_review"],
        "research_plan": {"task_graph": []},
    })
    llm = NativeDecisionLLM([
        "search_and_rank", "fetch_metadata", "extract_paper_cards", "plan_claims",
        "generate_deliverables", "validate_result", "request_finish",
    ])
    monkeypatch.setattr(graph, "_get_llm", lambda: llm)
    monkeypatch.setattr(
        graph, "_search_rank_with_refinement",
        lambda current, **kwargs: current.update({
            "candidate_papers": [{"paper_id": "p1"}],
            "ranked_papers": [{"paper_id": "p1"}],
        }),
    )
    monkeypatch.setattr(
        graph, "fetch_detail_node",
        lambda current, **kwargs: current.update({"paper_details": [{"paper_id": "p1"}]}),
    )
    monkeypatch.setattr(graph, "download_pdf_node", lambda current, **kwargs: None)
    monkeypatch.setattr(graph, "should_parse_pdf", lambda current: False)
    monkeypatch.setattr(
        graph, "extract_card_node",
        lambda current, **kwargs: current.update({
            "paper_cards": [{"paper_id": "p1", "evidence_state": {"access_level": "abstract"}}]
        }),
    )
    monkeypatch.setattr(
        graph, "claim_plan_node",
        lambda current, **kwargs: current.update({
            "claim_plans": [{"route_id": "r1", "claims": [{
                "claim_id": "c1", "claim_text": "发现", "supporting_paper_ids": ["p1"],
            }]}]
        }),
    )
    monkeypatch.setattr(graph, "_run_claim_evidence_gate", lambda current: None)
    monkeypatch.setattr(graph, "global_evidence_gate_node", lambda current: None)
    monkeypatch.setattr(
        graph, "_generate_deliverables_or_block",
        lambda current, **kwargs: current.update({
            "review": "有证据的正文 [1]", "writing_plans": [{"section_id": "s1"}],
        }),
    )
    monkeypatch.setattr(
        graph, "_verify_generated_draft",
        lambda current, **kwargs: current.update({"quality_gate": {"passed": True}}),
    )
    monkeypatch.setattr(graph, "final_answer_node", lambda current: current.update({"answer": current.get("review", "")}))

    output = graph._run_autonomous_pipeline(state)

    assert output["status"] == "success"
    assert output["answer"] == "有证据的正文 [1]"
    assert [item["action"] for item in state["main_agent_decisions"]] == [
        "search_and_rank", "fetch_metadata", "extract_paper_cards", "plan_claims",
        "generate_deliverables", "validate_result", "request_finish",
    ]


def test_summary_action_is_never_offered_in_any_transport_or_mode():
    from app.agent.action_registry import ACTION_REGISTRY, allowed_actions, tool_schemas
    from app.agent.tool_registry import tools_for_role
    assert "read_summary" not in ACTION_REGISTRY
    assert all(t["function"]["name"] != "read_summary" for t in tool_schemas(list(ACTION_REGISTRY)))
    assert all(t["name"] != "read_summary" for t in tools_for_role("main"))
    for mode in ("initial", "regeneration", "verification_only"):
        assert "read_summary" not in allowed_actions({"agent_operation_mode": mode})


def test_search_finish_reports_explicit_count_and_focus_shortfalls(monkeypatch):
    from app.agent import graph

    state = _state()
    state.update({
        "intent": "search_papers", "max_papers_explicit": True,
        "required_reference_count": 3,
        "research_semantic_frame": {"required_focuses": ["因果推断"]},
        "ranked_papers": [
            {"paper_id": "p1", "title": "主题综述", "abstract": "主题研究"},
            {"paper_id": "p2", "title": "主题综述", "abstract": "主题研究"},
        ],
    })
    monkeypatch.setattr(graph, "_get_llm", lambda: NativeDecisionLLM(["request_finish"]))

    output = graph._run_autonomous_pipeline(state)

    assert output["status"] == "partial"
    assert output["search_result_quality"]["actual"] == 1
    assert output["search_result_quality"]["requested"] == 3
    assert "因果推断" in output["search_result_quality"]["issues"][-1]
    assert "要求返回至少 3 篇" in output["answer"]
    assert output["answer"].count("**主题综述**") == 1
    assert state["steps"][-1]["status"] == "partial"


def test_search_finish_with_no_papers_is_blocked(monkeypatch):
    from app.agent import graph

    state = _state()
    state.update({"intent": "search_papers", "ranked_papers": []})
    monkeypatch.setattr(graph, "_get_llm", lambda: NativeDecisionLLM(["request_finish"]))

    output = graph._run_autonomous_pipeline(state)

    assert output["status"] == "blocked"
    assert output["search_result_quality"]["actual"] == 0
    assert "未检索到符合当前范围的论文" in output["answer"]
    assert state["steps"][-1]["status"] == "blocked"


def test_no_progress_then_real_progress_resets_counter():
    state = _state()
    llm = NativeDecisionLLM(["request_finish", "request_finish", "search_and_rank", "request_finish"])
    def search(working, arguments):
        working["candidate_papers"] = [{"paper_id": "p1"}]
    def finish(current):
        if current.get("candidate_papers"):
            current["result_status"] = "completed"
            return True
        return False
    assert MainAgentLoop(llm).run(state, handlers={"search_and_rank": search}, finish_validator=finish) == "completed"
