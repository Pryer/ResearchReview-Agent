"""审计发现的生产路径回归：入口、等待恢复、资料边界、预算和修复回退。"""
import json
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.agent import graph
from app.agent.context_builder import build_context_snapshot
from app.agent.execution_budget import (
    AgentBudgetExceeded, budget_scope, budgeted_create, consume,
)
from app.core.config import get_settings
from app.database.models import Base, ResearchSessionModel
from app.database.repositories import ResearchSessionRepository
from app.schemas.agent_schema import AgentRequest
from app.schemas.agent_task_schema import AgentRole
from app.services.research_artifact_service import (
    ResearchArtifactService, ArtifactNotFoundError, ArtifactVersionError,
)
from app.services.research_conversation_service import ResearchConversationService


class Decisions:
    native_tools_enabled = True

    def __init__(self, *actions):
        self.actions = iter(actions)
        self.contexts = []

    def complete_tool_call(self, messages, **kwargs):
        context = json.loads(messages[-1]["content"])
        assert set(context) == {"goal", "state", "key_evidence", "decisions", "open_questions"}
        self.contexts.append(context)
        action = next(self.actions)
        if isinstance(action, tuple):
            name, arguments = action
        else:
            name, arguments = action, {}
        return {"name": name, "arguments": arguments}


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with sessionmaker(bind=engine)() as session:
        yield session
    engine.dispose()


def state():
    return {"user_query": "研究主题", "topic": "研究主题", "state_schema_version": "2",
            "agent_orchestration_mode": "autonomous", "research_plan": {"task_graph": []},
            "core_deliverables": [], "steps": [], "errors": []}


def test_new_default_entry_reaches_autonomous_loop(monkeypatch):
    llm = Decisions("search_and_rank", "request_finish")
    monkeypatch.setattr(graph, "_get_llm", lambda: llm)
    monkeypatch.setattr(graph, "plan_node", lambda current, **kw: current.update({"intent": "search_papers", "topic": "主题"}))
    monkeypatch.setattr(graph, "_search_rank_with_refinement", lambda current, **kw: current.update({"ranked_papers": [{"paper_id": "p1", "title": "论文"}]}))
    output = graph.run_research_agent("检索主题论文")
    assert output["status"] == "success"
    assert output["research_state"]["agent_orchestration_mode"] == "autonomous"
    assert len(llm.contexts) == 2


def test_autonomous_citation_repair_passes_all_required_arguments(monkeypatch):
    current = state()
    current.update(review="old", required_reference_count=40, unique_cited_paper_count=1,
                   candidate_papers=[{"paper_id": "p1"}], max_papers_explicit=True)
    llm = Decisions("targeted_search", ("report_blocked", {"reason": "证据仍不足"}))
    monkeypatch.setattr(graph, "_get_llm", lambda: llm)
    calls = []
    def repair(working, should_cancel, progress_callback, step_idx, total_steps):
        calls.append((step_idx, total_steps))
        working["citation_gap_repair_attempted"] = True
        working["candidate_papers"].append({"paper_id": "p2"})
    monkeypatch.setattr(graph, "_repair_citation_gap", repair)
    monkeypatch.setattr(graph, "_verify_generated_draft", lambda s: None)
    monkeypatch.setattr(graph, "final_answer_node", lambda s: None)
    output = graph._run_autonomous_pipeline(current)
    assert calls == [(0, get_settings().agent_main_max_rounds)]
    assert output["status"] == "blocked"
    assert current["agent_execution_budget"]["recovery_used"] == 1


def test_clarification_is_public_persisted_and_resumed_with_latest_constraints(db, monkeypatch):
    current = state()
    current.update(session_id="pause", required_reference_count=40, max_papers_explicit=True,
                   start_year=2020, end_year=2024, year_range_explicit=True,
                   paper_cards=[{"paper_id": "p1"}])
    monkeypatch.setattr(graph, "_get_llm", lambda: Decisions(("request_clarification", {"question": "时间范围？"})))
    output = graph._run_autonomous_pipeline(current)
    assert output["status"] == "needs_clarification"
    assert output["clarification"]["kind"] == "main_agent"
    service = ResearchConversationService(db, llm=None)
    first = service._persist_or_pause_result("pause", "研究主题，至少40篇", current, output)
    assert first["status"] == "needs_clarification"
    saved = ResearchSessionRepository(db).get("pause")
    assert saved["clarification"]["needed"]
    assert saved["state"]["editable_research_state"]["result_status"] == "waiting_user"
    llm = Decisions(("report_blocked", {"reason": "尚需更多证据"}))
    monkeypatch.setattr(graph, "_get_llm", lambda: llm)
    monkeypatch.setattr("app.agent.research_semantic_parser.parse_research_semantics", lambda *a, **k: SimpleNamespace(model_dump=lambda **kw: {}))
    result = ResearchConversationService(db, llm=None).handle(AgentRequest(
        session_id="pause", user_query="研究主题", clarification_answer="仅限2021-2023年，基于已有证据保守重写",
    ))
    assert result["status"] == "blocked"
    goal = llm.contexts[0]["goal"]
    assert goal["constraints"]["required_reference_count"] == 40
    assert goal["constraints"]["start_year"] == 2021
    assert goal["constraints"]["end_year"] == 2023
    assert any(item["name"] == "user_clarification" for item in goal["explicit_constraints"])
    assert ResearchSessionRepository(db).get("pause")["clarification"] is None


@pytest.mark.parametrize("status,expected", [("waiting_user", "needs_clarification"), ("cancelled", "cancelled"), ("failed", "failed"), ("blocked", "blocked")])
def test_public_status_honors_loop_terminal_state(status, expected):
    current = state()
    current["result_status"] = status
    assert graph._build_output(current)["status"] == expected


def test_long_goal_tail_and_user_decision_survive_compaction(monkeypatch):
    current = state()
    current["user_query"] = "背景描述" * 250 + "禁止动物实验；只纳入公开数据。"
    current["selected_scope"] = {"label": "已确认范围", "confirmed_by_user": True}
    current["route_decisions"] = [{"decision": "路线" + str(i), "reason": "a" * 300} for i in range(30)]
    monkeypatch.setattr(get_settings(), "main_context_max_decisions", 3)
    snap = build_context_snapshot(current)
    assert snap.context.goal.request.endswith("禁止动物实验；只纳入公开数据。")
    assert any(item.source == "user" for item in snap.context.decisions)
    current["user_query"] *= 30
    snap = build_context_snapshot(current)
    assert "context_budget_exceeded" in snap.validation_errors
    assert snap.context.goal.request == current["user_query"]


def test_artifact_roundtrip_externalizes_heavy_fields_and_document_locations(db):
    current = state()
    current.update(candidate_papers=[{"paper_id": "p1", "title": "raw"}], paper_details=[{"paper_id": "p1"}],
                   paper_cards=[{"paper_id": "p1", "evidence_refs": [{"page": 3, "quote": "来源片段"}]}],
                   claim_plans=[{"route_id": "r1"}], writing_plans=[{"section_id": "s1"}],
                   review="正文" * 17000, parsed_papers={"p1": {"pages": [{"page": 3, "text": "正文证据"}]}},
                   agent_task_results=[{"task_id": "t1", "status": "completed"}])
    repo = ResearchSessionRepository(db)
    repo.save("external", "running", "query", {"editable_research_state": current})
    db.commit()
    raw = json.loads(db.get(ResearchSessionModel, "external").state_json)["editable_research_state"]
    for field in ("candidate_papers", "paper_details", "paper_cards", "claim_plans", "writing_plans", "review", "parsed_papers", "agent_task_results"):
        assert field not in raw
        assert field in raw["artifact_manifest"]
        assert repo.get("external")["state"]["editable_research_state"][field] == current[field]
    fragments = ResearchArtifactService(db).repo.list("external", "document_fragment")
    assert fragments[0]["provenance"]["char_start"] == 0


def test_artifact_resolver_enforces_role_type_version_and_reports_missing(db):
    service = ResearchArtifactService(db)
    ref = service.persist_payload(session_id="a", artifact_type="draft", payload={"value": "正文"}, version=2)
    with pytest.raises(PermissionError):
        service.resolve("a", ref, role="search", allowed_types={"draft"}, expected_version=2)
    with pytest.raises(PermissionError):
        service.resolve("b", ref)
    with pytest.raises(ArtifactNotFoundError):
        service.resolve("a", "artifact://missing")
    with pytest.raises(ArtifactVersionError):
        service.resolve("a", ref, expected_version=1)
    with pytest.raises(PermissionError):
        service.resolve("a", ref, allowed_types={"paper_card"})
    assert service.resolve("a", ref, role="writing", expected_version=2)["payload"]["value"] == "正文"


@pytest.mark.parametrize("payload", [{"pdf_base64": "encoded"}, {"value": "x" * 300000}, {"value": b"PDF"}])
def test_artifact_rejects_embedded_or_unbounded_content(db, payload):
    with pytest.raises(ValueError):
        ResearchArtifactService(db).persist_payload(session_id="a", artifact_type="document_fragment", payload=payload)


def test_controlled_file_reference_checks_root_and_content(db, monkeypatch, tmp_path):
    monkeypatch.setattr(get_settings(), "pdf_save_dir", str(tmp_path / "pdfs"))
    folder = tmp_path / "pdfs"
    folder.mkdir()
    pdf = folder / "paper.pdf"
    pdf.write_bytes(b"%PDF-test")
    service = ResearchArtifactService(db)
    stored = service.externalize_state("s", {"pdf_paths": {"p": str(pdf)}})
    assert service.hydrate_state("s", stored)["pdf_paths"]["p"] == str(pdf)
    pdf.write_bytes(b"changed")
    with pytest.raises(ValueError):
        service.hydrate_state("s", stored)
    with pytest.raises(PermissionError):
        service.controlled_file_ref(str(tmp_path / "outside.pdf"))


def test_token_budget_reserves_before_request_and_estimates_missing_usage():
    # WHY: 缺失 usage 时按完整预留估算，不能把缓存或平均输出当作硬上界。
    current = {"agent_execution_budget": {"token_limit": 300}}
    calls = []
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kw: calls.append(kw) or SimpleNamespace(usage=None))))
    with budget_scope(current):
        budgeted_create(client, messages=[{"role": "user", "content": "hello"}], max_tokens=32)
        with pytest.raises(AgentBudgetExceeded):
            budgeted_create(client, messages=[{"role": "user", "content": "hello"}], max_tokens=32)
    ledger = current["agent_execution_budget"]
    assert len(calls) == 1
    assert ledger["usage_estimated"] and ledger["llm_tokens"] > 0
    assert ledger["tokens_reserved"] == 0


def test_shared_budget_retains_failed_subtask_consumption():
    from app.agent.controller import AgentController
    current = state()
    current["agent_execution_budget"] = {"retrieval_limit": 1}
    def failing(working):
        consume("retrieval")
        raise RuntimeError("external source failed")
    with budget_scope(current):
        with pytest.raises(RuntimeError):
            AgentController().execute(role=AgentRole.SEARCH, operation="search_and_rank", objective="search",
                                      state=current, handler=failing)
        with pytest.raises(AgentBudgetExceeded):
            consume("retrieval")
    assert current["agent_execution_budget"]["retrieval_used"] == 1
    assert current["agent_execution_budget"]["actions_reserved"] == 1


def test_old_session_mode_migrates_to_autonomous():
    from app.agent.orchestration import normalize_orchestration_mode
    for stored in (None, "legacy", "autonomous"):
        current = {"required_reference_count": 40, "paper_cards": [{"paper_id": "p1"}],
                   "autonomous_verified_fingerprint": "old-draft"}
        if stored is not None:
            current["agent_orchestration_mode"] = stored
        assert normalize_orchestration_mode(current) == "autonomous"
        assert current["agent_orchestration_mode"] == "autonomous"
        assert current["required_reference_count"] == 40
        assert current["paper_cards"] == [{"paper_id": "p1"}]
        if stored == "autonomous":
            assert current["autonomous_verified_fingerprint"] == "old-draft"
        else:
            assert "autonomous_verified_fingerprint" not in current
    with pytest.raises(ValueError, match="invalid persisted orchestration mode"):
        normalize_orchestration_mode({"agent_orchestration_mode": []})


def test_legacy_checkpoint_resumes_through_single_loop_without_losing_state(db, monkeypatch):
    repo = ResearchSessionRepository(db)
    editable = state()
    editable.update(
        agent_orchestration_mode="legacy",
        required_reference_count=40,
        paper_cards=[{"paper_id": "p1"}],
        agent_execution_budget={"retrieval_used": 3},
    )
    repo.save("old-checkpoint", "failed", "原研究请求", {
        "runtime_checkpoint_version": 2,
        "editable_research_state": editable,
    })
    db.commit()
    seen = {}

    def resume(current, **kwargs):
        seen.update(current)
        return {"status": "blocked", "research_state": current}

    monkeypatch.setattr(graph, "_run_autonomous_pipeline", resume)
    service = ResearchConversationService(db, llm=None)
    monkeypatch.setattr(service, "_persist_or_pause_result", lambda *args: args[-1])

    result = service._resume_checkpoint(AgentRequest(
        session_id="old-checkpoint", user_query="继续完成原研究",
    ))

    assert result["status"] == "blocked"
    assert seen["agent_orchestration_mode"] == "autonomous"
    assert seen["required_reference_count"] == 40
    assert seen["paper_cards"] == [{"paper_id": "p1"}]
    assert seen["agent_execution_budget"]["retrieval_used"] == 3
    assert seen["user_operation_sequence"] == 1


def test_autonomous_rewrite_rolls_back_a_worse_candidate(monkeypatch):
    current = state()
    current.update(review="原有正文", paper_cards=[{"paper_id": "p1"}],
                   claim_plans=[{"route_id": "r1"}], writing_plans=[{"section_id": "s1"}],
                   quality_gate={"passed": True})
    llm = Decisions("generate_deliverables", "request_finish")
    monkeypatch.setattr(graph, "_get_llm", lambda: llm)
    monkeypatch.setattr(graph, "_generate_deliverables_or_block", lambda s, **kw: s.update(review="更差的正文"))
    def verify(s):
        s["quality_gate"] = {"passed": s["review"] == "原有正文", "blocking_issues": [] if s["review"] == "原有正文" else [{"code": "unsupported_claims"}]}
    monkeypatch.setattr(graph, "_verify_generated_draft", verify)
    monkeypatch.setattr(graph, "final_answer_node", lambda s: s.update(answer=s["review"]))
    output = graph._run_autonomous_pipeline(current)
    assert output["status"] == "success"
    assert output["answer"] == "原有正文"
    assert current["recovery_candidate_rejections"]


def test_native_unsupported_falls_back_to_strict_five_field_json(monkeypatch):
    import httpx
    from openai import BadRequestError
    from app.services.llm_service import LLMService
    from app.agent.main_policy import MainAgentPolicy
    from app.agent.context_builder import build_main_agent_context
    calls = []
    def respond(**kwargs):
        calls.append(kwargs)
        if "tools" in kwargs:
            raise BadRequestError("tools not supported", response=httpx.Response(400, request=httpx.Request("POST", "https://example.invalid")), body=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=json.dumps({"action": "report_blocked", "arguments": {"reason": "证据不足"}, "evidence_refs": []}),
            reasoning_content=None), finish_reason="stop")], usage=SimpleNamespace(prompt_tokens=10, completion_tokens=15))
    llm = LLMService()
    llm.api_key = "test"
    llm.backup_enabled = False
    llm.native_tools_enabled = True
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=respond)))
    current = state()
    with budget_scope(current):
        decision = MainAgentPolicy(llm).decide(build_main_agent_context(current), allowed_actions=["report_blocked"])
    assert decision.transport == "json"
    assert len(calls) == 2
    native_prefix = calls[0]["messages"][0]["content"]
    json_prefix = calls[1]["messages"][0]["content"]
    assert "核心工具目录" not in native_prefix
    assert json_prefix.startswith(native_prefix)
    assert "核心工具目录" in json_prefix
    assert len(json.loads(calls[1]["messages"][-1]["content"])) == 5
    assert current["agent_execution_budget"]["llm_requests"] == 2
    assert current["agent_execution_budget"]["llm_tokens"] > 25


def test_nested_actions_do_not_double_charge_parent_budget():
    from app.agent.controller import AgentController
    current = state()
    def parent(working):
        AgentController().execute(role=AgentRole.SEARCH, operation="search_and_rank", objective="child",
                                  state=working, handler=lambda s: s.update(candidate_papers=[{"paper_id": "p1"}]))
    with budget_scope(current):
        AgentController().execute(role=AgentRole.SEARCH, operation="targeted_search", objective="parent",
                                  state=current, handler=parent)
    assert current["agent_execution_budget"]["actions_reserved"] == 1
    assert current["agent_execution_budget"]["actions_committed"] == 1


def test_deadline_after_task_discards_candidate_patch(monkeypatch):
    from app.agent.controller import AgentController
    from app.agent import execution_budget
    now = [0.0]
    monkeypatch.setattr(execution_budget.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(get_settings(), "agent_execution_deadline_seconds", 10)
    current = state()
    def slow(working):
        working["candidate_papers"] = [{"paper_id": "late"}]
        now[0] = 11.0
    with budget_scope(current):
        with pytest.raises(AgentBudgetExceeded):
            AgentController().execute(role=AgentRole.SEARCH, operation="search_and_rank", objective="search",
                                      state=current, handler=slow)
    assert "candidate_papers" not in current
    assert current["agent_execution_budget"]["actions_committed"] == 0


def test_context_propagation_shares_reservations_across_worker_threads():
    from concurrent.futures import ThreadPoolExecutor
    from app.agent.execution_budget import submit_with_context
    current = {"agent_execution_budget": {"retrieval_limit": 1}}
    with budget_scope(current), ThreadPoolExecutor(max_workers=1) as executor:
        submit_with_context(executor, consume, "retrieval").result()
        with pytest.raises(AgentBudgetExceeded):
            submit_with_context(executor, consume, "retrieval").result()
    assert current["agent_execution_budget"]["retrieval_used"] == 1


def test_section_rewrite_threads_share_budget_and_propagate_cancel():
    """章节改写线程必须继承预算上下文，且取消不得被当作可重试的模型失败。"""
    from app.agent.execution import AgentCancelledError
    from app.agent.execution_budget import check_execution
    from app.deliverables.renderers import _write_sections_in_chinese
    from app.schemas.deliverable_schema import (
        CoreDeliverableType, WritingPlan, WritingSection,
    )

    plan = WritingPlan(
        deliverable_type=CoreDeliverableType.RESEARCH_STATUS,
        purpose="测试",
        organizing_strategy="evidence_driven",
        sections=[
            WritingSection(
                id="theme_T1",
                title="自动识别",
                purpose="测试",
                supporting_paper_ids=["p1", "p2"],
            )
        ],
        citation_policy={"minimum_unique_references": 2},
    )
    state = {"topic": "课堂行为分析"}

    class BudgetAwareLLM:
        calls = 0

        def complete(self, prompt: str, **kwargs) -> str:
            # 真实 llm.complete 经 budgeted_create 先检查执行边界；此处复现同一
            # 边界。上下文未跨线程传播时 _ACTIVE 为 None，检查会静默通过。
            type(self).calls += 1
            check_execution()
            # 引用不足，迫使章节进入第二次尝试，取消在重试边界上生效。
            return "## 自动识别\n\n现有研究采用自动识别方法[p1]。"

    llm = BudgetAwareLLM()
    cancel_from_second_call = lambda: BudgetAwareLLM.calls >= 2
    with budget_scope(state, should_cancel=cancel_from_second_call):
        with pytest.raises(AgentCancelledError):
            _write_sections_in_chinese(
                "## 自动识别\n\nMethod one is reported by one study[p1].",
                plan,
                state,
                [],
                llm,
            )
    # 取消必须终止改写：不能继续第三次尝试，也不能把旧正文当作降级结果返回。
    assert BudgetAwareLLM.calls == 2
    assert not state.get("writer_section_diagnostics")


def test_controller_resolves_current_manifest_with_role_and_version(db, monkeypatch):
    from app.agent.controller import AgentController
    from app.services import research_execution_service as execution
    current = state()
    current.update(session_id="a", paper_cards=[{"paper_id": "p1"}])
    store = ResearchArtifactService(db)
    token = execution._STORE.set(store)
    calls = []
    resolve = store.resolve
    def spy(*args, **kwargs):
        calls.append(kwargs)
        return resolve(*args, **kwargs)
    monkeypatch.setattr(store, "resolve", spy)
    try:
        execution.checkpoint_artifacts(current)
        AgentController().execute(role=AgentRole.WRITING, operation="generate_deliverables", objective="write",
                                  state=current, handler=lambda s: s.update(review="draft"))
    finally:
        execution._STORE.reset(token)
    assert calls and all(call["role"] == "writing" and call["expected_version"] == 1 for call in calls)
    assert all("paper_card" in call["allowed_types"] for call in calls)


def test_verification_only_autonomous_entry_rejects_retrieval_and_keeps_writer_idle(monkeypatch):
    current = state()
    current.update(review="原有正文", verification_only_recovery=True)
    llm = Decisions("search_and_rank", "request_finish")
    monkeypatch.setattr(graph, "_get_llm", lambda: llm)
    def unexpected(*a, **k):
        pytest.fail("verification-only must not retrieve or write")
    monkeypatch.setattr(graph, "_search_rank_with_refinement", unexpected)
    monkeypatch.setattr(graph, "_generate_deliverables_or_block", unexpected)
    monkeypatch.setattr(graph, "_verify_generated_draft", lambda s: s.update(quality_gate={"passed": True}))
    monkeypatch.setattr(graph, "final_answer_node", lambda s: s.update(answer=s["review"]))
    result = graph.regenerate_research_agent(current)
    assert result["status"] == "success"
    assert result["answer"] == "原有正文"
    assert all("search_and_rank" not in context["state"]["allowed_actions"] for context in llm.contexts)


def test_incremental_entry_keeps_pinned_mode_and_evidence(monkeypatch):
    current = state()
    current.update(paper_details=[{"paper_id": "p1"}], paper_cards=[{"paper_id": "p1"}])
    llm = Decisions(("request_clarification", {"question": "补充的研究范围是什么？"}))
    monkeypatch.setattr(graph, "_get_llm", lambda: llm)
    result = graph.continue_research_agent(current)
    assert result["status"] == "needs_clarification"
    assert result["research_state"]["agent_operation_mode"] == "incremental"
    assert result["research_state"]["paper_cards"] == current["paper_cards"]
