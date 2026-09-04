"""Agent 工作流串联测试。"""

from __future__ import annotations

import json

import pytest

from app.agent.graph import _compute_total_steps, run_research_agent
from app.agent.retrieval_loop import diagnose_search_drift
from app.agent.nodes import (
    citation_check_node,
    extract_card_node,
    fetch_detail_node,
    final_answer_node,
    plan_node,
    refine_search_node,
    search_node,
)
from app.core.config import Settings


def _paper(**kwargs) -> dict:
    defaults = {
        "paper_id": "p1",
        "title": "Object Detection Survey",
        "authors": ["Author A"],
        "year": 2024,
        "venue": "CVPR",
        "abstract": "This paper studies object detection.",
        "doi": None,
        "arxiv_id": None,
        "url": "https://example.com/p1",
        "pdf_url": None,
        "citation_count": 10,
        "source": "test",
    }
    defaults.update(kwargs)
    return defaults


def test_search_drift_uses_semantic_topic_anchors():
    papers = [
        _paper(paper_id="edu", title="AI-supported classroom interaction analysis"),
        _paper(paper_id="generic1", title="Generic action recognition", abstract="student engagement"),
        _paper(paper_id="generic2", title="Human action recognition", abstract="student engagement"),
    ]

    diagnostics = diagnose_search_drift(
        papers,
        core_keywords=["classroom behavior analysis"],
        expanded_keywords=["student engagement"],
        topic_anchors=[["课堂互动", "classroom interaction"]],
    )

    assert diagnostics["anchor_hit_count"] == 1
    assert diagnostics["expanded_only_count"] == 2
    assert diagnostics["drift_detected"] is True
    assert "主题语义锚点覆盖不足" in diagnostics["reasons"]


def test_search_drift_without_expansion_is_observational_only():
    diagnostics = diagnose_search_drift(
        [_paper(title="Generic action recognition")],
        core_keywords=["classroom behavior analysis"],
        topic_anchors=[["课堂互动", "classroom interaction"]],
    )

    assert diagnostics["anchor_coverage_rate"] == 0
    assert diagnostics["drift_detected"] is False


class WorkflowLLM:
    """覆盖工作流控制面调用，确保单元测试不访问真实 API。"""

    def complete(self, prompt: str, **kwargs) -> str:
        operation = str(kwargs.get("operation") or "")
        if operation == "research_semantic_parsing":
            if "课堂行为" in prompt:
                return json.dumps({
                    "canonical_topic": "课堂行为分析",
                    "application_domains": [{"id": "education", "label": "education", "surface_text": "课堂", "explicit": True, "confidence": 0.95}],
                    "research_objects": [{"id": "teacher_student_behavior", "label": "teacher-student behavior", "surface_text": "老师或学生行为", "explicit": True, "confidence": 0.95}],
                    "methods": [
                        {"id": "action_recognition", "label": "action recognition", "surface_text": "自动识别", "category": "technical", "role": "intermediate_step", "explicit": True, "confidence": 0.95},
                        {"id": "st_analysis", "label": "S-T analysis", "surface_text": "S-T分析法", "category": "analytical", "explicit": True, "confidence": 0.95},
                        {"id": "lag_analysis", "label": "lag sequential analysis", "surface_text": "滞后分析法", "category": "analytical", "explicit": True, "confidence": 0.95},
                    ],
                    "research_actions": [{"id": "automatic_behavior_coding", "label": "automatic behavior coding", "surface_text": "自动行为编码", "explicit": True, "confidence": 0.95}],
                    "analysis_targets": [{"id": "teaching_interaction", "label": "teaching interaction", "surface_text": "课堂行为分析", "explicit": True, "confidence": 0.9}],
                    "terminal_goal": {"type": "domain_analysis", "target": "teaching_interaction"},
                    "task_chain": ["teacher_student_behavior_recognition", "automatic_behavior_coding", "st_or_lag_sequential_analysis", "teaching_structure_and_interaction_interpretation"],
                    "required_focuses": ["教师与学生行为自动识别", "自动行为编码", "S-T分析法或滞后序列分析法", "教学结构与师生互动解释"],
                }, ensure_ascii=False)
            return json.dumps({
                "canonical_topic": "目标检测",
                "application_domains": [], "research_objects": [],
                "methods": [{"id": "object_detection", "label": "object detection", "surface_text": "目标检测", "category": "technical", "explicit": True, "confidence": 0.95}],
                "research_actions": [], "analysis_targets": [],
                "terminal_goal": {"type": "method_analysis", "target": "object_detection"},
            }, ensure_ascii=False)
        if operation in {"initial_search_planning", "refined_search_planning"}:
            return json.dumps({
                "keywords": ["目标检测", "object detection", "object detection survey"],
                "topic_anchors": [{"concept": "目标检测", "terms": ["目标检测", "object detection"]}],
            }, ensure_ascii=False)
        if operation == "generate_search_keywords":
            return json.dumps({
                "zh": [{"keyword": "目标检测", "type": "exact"}],
                "en": [{"keyword": "object detection", "type": "exact"}],
            }, ensure_ascii=False)
        if operation == "screening_protocol_planning":
            return '{"corpus_goal":"目标主题证据池","hard_include_criteria":[],"soft_include_criteria":[],"hard_exclude_title_terms":[],"routes":[],"notes":[]}'
        return "{}"


def test_compute_total_steps_counts_all_checkpoint_slots(monkeypatch):
    """进度分母必须覆盖 validate_routes / claim_plan / claim_alignment 检查点，
    否则实际步骤数会超过分母，进度条超过 100%。"""
    monkeypatch.setattr(
        "app.agent.graph.get_settings",
        lambda: Settings(
            enable_pdf_pipeline=True,
            enable_evidence_recovery=True,
            enable_claim_verification=True,
        ),
    )
    state = {"intent": "generate_review", "core_deliverables": ["research_status"]}
    total = _compute_total_steps(state)
    # 基础6 + fetch + download/parse(2) + cards + validate_routes
    # + cluster + recovery + gate + claim_plan/gate(2) + generate
    # + alignment + verify + citation = 20
    assert total == 20

    # 只查论文：检索排序后即返回，不包含生成链路步骤。
    assert _compute_total_steps({"intent": "search_papers"}) == 6


def test_build_output_deliverable_type_is_always_scalar():
    """deliverable_type 必须类型稳定（始终为主交付物标量）。"""
    from app.agent.graph import _build_output

    multi = _build_output({
        "core_deliverables": ["research_status", "related_work"],
        "steps": [], "errors": [], "references": [], "paper_cards": [],
    })
    single = _build_output({"core_deliverables": ["research_status"]})
    empty = _build_output({})

    assert multi["deliverable_type"] == "research_status"
    assert single["deliverable_type"] == "research_status"
    assert empty["deliverable_type"] is None
    assert multi["core_deliverables"] == ["research_status", "related_work"]


def test_plan_uses_current_time_tool_for_relative_year_range():
    state = {
        "user_query": "帮我调研近五年目标检测论文，引用不少于5篇",
        "steps": [],
        "errors": [],
    }

    plan_node(state, llm=None)

    step_names = [step["step_name"] for step in state["steps"]]
    assert "get_current_time" in step_names
    assert state["current_time"]["year"] == state["end_year"]
    assert state["start_year"] == state["end_year"] - 5 + 1


def test_plan_skips_current_time_tool_for_absolute_closed_year_range():
    state = {
        "user_query": "帮我调研2022-2025年目标检测论文，引用不少于5篇",
        "steps": [],
        "errors": [],
    }

    plan_node(state, llm=None)

    step_names = [step["step_name"] for step in state["steps"]]
    assert "get_current_time" not in step_names
    assert state["start_year"] == 2022
    assert state["end_year"] == 2025


def test_plan_logs_explicit_multistage_task_chain_and_required_focuses():
    state = {
        "user_query": (
            "调研近三年课堂行为分析论文，并生成研究背景和研究现状。"
            "先进行老师或学生行为自动识别和自动行为编码，"
            "然后使用S-T分析法或滞后分析法"
        ),
        "steps": [],
        "errors": [],
    }

    plan_node(state, llm=WorkflowLLM(), current_year=2026)
    step = next(item for item in state["steps"] if item["step_name"] == "plan")

    assert state["canonical_topic"] == "课堂行为分析"
    assert step["output_data"]["task_chain"] == [
        "teacher_student_behavior_recognition",
        "automatic_behavior_coding",
        "st_or_lag_sequential_analysis",
        "teaching_structure_and_interaction_interpretation",
    ]
    assert step["output_data"]["required_focuses"] == [
        "教师与学生行为自动识别",
        "自动行为编码",
        "S-T分析法或滞后序列分析法",
        "教学结构与师生互动解释",
    ]


def test_plan_restores_constraints_changed_by_quality_decision():
    state = {
        "user_query": "survey classroom behavior analysis in recent three years",
        "research_request": {
            "start_year": 2021,
            "end_year": 2026,
            "year_range_explicit": True,
            "required_reference_count": 40,
            "max_papers": 40,
            "max_papers_explicit": True,
            "retrieval_target": 120,
            "generation_limit": 80,
            "requested_sections": ["research_status"],
        },
        "steps": [],
        "errors": [],
    }

    plan_node(state, llm=None, current_year=2026)

    assert (state["start_year"], state["end_year"]) == (2021, 2026)
    assert state["required_reference_count"] == 40
    assert state["retrieval_target"] == 120
    assert state["generation_limit"] == 80
    assert state["requested_sections"] == ["research_status"]


def test_incremental_search_queries_only_new_year_window_and_merges_candidates(monkeypatch):
    calls = []

    def fake_search_papers(**kwargs):
        calls.append(kwargs)
        return [_paper(
            paper_id="new",
            title="Classroom Behavior Analysis with Sequence Methods",
            year=2022,
        )]

    monkeypatch.setattr("app.tools.search_papers.search_papers", fake_search_papers)
    state = {
        "topic": "classroom behavior analysis",
        "keywords": ["classroom behavior analysis"],
        "start_year": 2022,
        "end_year": 2026,
        "max_papers": 40,
        "retrieval_target": 120,
        "candidate_papers": [_paper(paper_id="old", year=2025)],
        "searched_keywords": ["classroom behavior analysis"],
        "incremental_search_window": {"start_year": 2022, "end_year": 2023},
        "steps": [],
        "errors": [],
    }

    search_node(state)

    assert len(calls) == 1
    assert (calls[0]["start_year"], calls[0]["end_year"]) == (2022, 2023)
    assert {paper["paper_id"] for paper in state["candidate_papers"]} == {"old", "new"}
    assert state["incremental_search_new_candidates"] == 1


def test_incremental_detail_and_card_steps_reuse_existing_work(monkeypatch):
    old = _paper(paper_id="old", title="Classroom Behavior Analysis", year=2025)
    new = _paper(paper_id="new", title="Classroom Behavior Sequence Analysis", year=2022)
    fetched = []
    extracted = []

    def fake_fetch(papers):
        fetched.extend(papers)
        return list(papers)

    def fake_extract(papers, parsed, **kwargs):
        extracted.extend(papers)
        return [
            {**paper, "quality_status": "partial", "evidence_source": "abstract"}
            for paper in papers
        ]

    monkeypatch.setattr("app.tools.fetch_metadata.fetch_batch_details", fake_fetch)
    monkeypatch.setattr("app.tools.rank_papers.passes_topic_filter", lambda *a, **k: True)
    monkeypatch.setattr("app.tools.rank_papers.evaluate_scope_filter", lambda *a, **k: (True, ""))
    monkeypatch.setattr("app.tools.rank_papers.evaluate_search_branch_filter", lambda *a, **k: (True, ""))
    monkeypatch.setattr("app.tools.extract_paper_card.batch_extract_paper_cards", fake_extract)
    state = {
        "topic": "classroom behavior analysis",
        "keywords": ["classroom behavior analysis"],
        "ranked_papers": [old, new],
        "paper_details": [old],
        "paper_cards": [{**old, "quality_status": "partial", "evidence_source": "abstract"}],
        "required_reference_count": 2,
        "max_papers": 2,
        "generation_limit": 4,
        "generation_readiness": {"usable_reference_count": 1},
        "incremental_retrieval": True,
        "steps": [],
        "errors": [],
    }

    fetch_detail_node(state)
    extract_card_node(state, llm=None)

    assert [paper["paper_id"] for paper in fetched] == ["new"]
    assert [paper["paper_id"] for paper in extracted] == ["new"]
    assert state["incremental_new_paper_ids"] == ["new"]
    assert {paper["paper_id"] for paper in state["paper_details"]} == {"old", "new"}
    assert {card["paper_id"] for card in state["paper_cards"]} == {"old", "new"}


def test_fetch_detail_replays_scope_filter_when_protocol_exists(monkeypatch):
    """存在 screening protocol 时也不能绕过用户已确认的研究范围。"""
    off_scope = _paper(
        paper_id="off-scope",
        title="Visual Student Action Recognition with YOLO",
        abstract="Computer vision detection of student gestures.",
    )
    monkeypatch.setattr(
        "app.tools.fetch_metadata.fetch_batch_details", lambda papers: list(papers),
    )
    state = {
        "topic": "classroom behavior analysis",
        "keywords": ["classroom behavior analysis"],
        "ranked_papers": [off_scope],
        "required_reference_count": 1,
        "max_papers": 1,
        "generation_limit": 1,
        "screening_protocol": {"protocol_version": "1"},
        "selected_scope": {
            "include_terms": ["teaching behavior analysis", "classroom interaction analysis"],
            "exclude_terms": ["computer vision", "action recognition"],
        },
        "steps": [],
        "errors": [],
    }

    fetch_detail_node(state)

    assert state["paper_details"] == []
    assert state["retrieval_requirement_met"] is False


def test_standalone_search_returns_paper_list(monkeypatch):
    """独立找论文与 graph 的 search_papers 早退分支对齐：检索排序后直接返回。"""
    search_called = False

    def fake_search(**kwargs):
        nonlocal search_called
        search_called = True
        return [_paper(paper_id="p-search-1"), _paper(paper_id="p-search-2")]

    monkeypatch.setattr(
        "app.tools.search_papers.search_papers",
        fake_search,
    )

    result = run_research_agent("帮我找几篇关于目标检测的论文", current_year=2026)

    step_names = [s["step_name"] for s in result["steps"]]
    assert result["unsupported_task_guard"]["allowed"] is True
    assert search_called is True
    assert "final_answer" in step_names
    assert result["intent"] == "search_papers"
    assert result["references"] == []


def test_search_refines_keywords_after_low_recall(monkeypatch):
    """检索不足时，应把反馈交给 LLM 修正关键词并再次搜索。"""

    class FakeReActLLM:
        def complete(self, prompt: str, **kwargs) -> str:
            if "意图识别模块" in prompt:
                return '{"intent": "search_papers", "confidence": 0.95, "reason": "用户要求找论文"}'
            if "ReAct 查询修正模块" in prompt:
                return """
                {
                  "keywords": ["少样本动作识别", "FSAR", "few-shot action recognition"],
                  "required_concepts": [
                    {"concept": "少样本", "terms": ["few-shot", "few shot", "one-shot"]},
                    {"concept": "动作识别", "terms": ["action recognition", "activity recognition"]}
                  ],
                  "excluded_title_terms": ["zero-shot action recognition"]
                }
                """
            return """
            {
              "keywords": ["少样本动作识别", "action recognition"],
              "required_concepts": [
                {"concept": "少样本", "terms": ["few-shot", "few shot", "one-shot"]},
                {"concept": "动作识别", "terms": ["action recognition", "activity recognition"]}
              ],
              "excluded_title_terms": ["zero-shot action recognition"]
            }
            """

    seen_queries: list[str] = []

    def fake_search_papers(**kwargs):
        query = kwargs["query"]
        seen_queries.append(query)
        if query in {"FSAR", "few-shot action recognition"}:
            return [
                _paper(
                    paper_id=f"fsar-{query}",
                    title=f"Few-Shot Action Recognition Method {query}",
                    abstract="We study few-shot action recognition in videos.",
                )
            ]
        return [
            _paper(
                paper_id="generic",
                title="Generic Action Recognition",
                abstract="We study action recognition.",
            )
        ]

    monkeypatch.setattr("app.agent.graph._get_llm", lambda: FakeReActLLM())
    monkeypatch.setattr("app.tools.search_papers.search_papers", fake_search_papers)

    result = run_research_agent("生成少样本动作识别研究现状，至少引用2篇", current_year=2026)

    step_names = [step["step_name"] for step in result["steps"]]
    assert "refine_search" in step_names
    assert "FSAR" in seen_queries
    assert result["unsupported_task_guard"]["allowed"] is True


def test_refine_search_does_not_override_filter_boundary():
    class FakeRefineLLM:
        def complete(self, prompt: str, **kwargs) -> str:
            return """
            {
              "keywords": ["少样本动作识别", "FSAR"],
              "required_concepts": [
                {"concept": "宽泛动作", "terms": ["action recognition", "动作识别"]}
              ],
              "excluded_title_terms": ["few-shot action recognition"]
            }
            """

    state = {
        "user_query": "帮我找少样本动作识别论文",
        "topic": "少样本动作识别",
        "keywords": ["少样本动作识别", "few-shot action recognition"],
        "required_concepts": [["few-shot"], ["action recognition"]],
        "excluded_title_terms": ["zero-shot action recognition"],
        "candidate_papers": [],
        "ranked_papers": [],
        "searched_keywords": ["few-shot action recognition"],
        "steps": [
            {"step_name": "search", "output_data": {"count": 0}},
            {"step_name": "rank", "output_data": {"ranked": 0}},
        ],
        "errors": [],
    }

    refine_search_node(state, llm=FakeRefineLLM())

    assert "FSAR" in state["keywords"]
    assert state["required_concepts"] == [["few-shot"], ["action recognition"]]
    assert state["excluded_title_terms"] == ["zero-shot action recognition"]
    refine_step = state["steps"][-1]
    assert refine_step["output_data"]["ignored_refined_required_concepts"]


def test_refine_search_adds_conservative_queries_without_llm():
    state = {
        "user_query": "课堂行为识别后使用S-T分析法和滞后分析法",
        "topic": "课堂行为分析",
        "canonical_topic": "课堂行为分析",
        "keywords": ["classroom behavior analysis"],
        "required_concepts": [],
        "excluded_title_terms": [],
        "candidate_papers": [],
        "ranked_papers": [_paper(title="Classroom Behavior Detection", abstract="Visual behavior detection")],
        "searched_keywords": ["classroom behavior analysis"],
        "research_semantic_frame": {"required_focuses": ["S-T分析法", "滞后序列分析法"]},
        "steps": [
            {"step_name": "search", "output_data": {"count": 1}},
            {"step_name": "rank", "output_data": {"ranked": 1}},
        ],
        "errors": [],
    }

    refine_search_node(state, llm=None)

    joined = " ".join(state["keywords"]).lower()
    assert "s-t分析法" in joined
    assert "滞后序列分析法" in joined
    assert set(state["focus_coverage"]["missing_focuses"]) == {"S-T分析法", "滞后序列分析法"}


def test_review_flow_does_not_count_unverified_pdf_url_as_download_failure(monkeypatch):
    """综述默认尝试下载开放全文；单篇失败时继续使用摘要证据。"""
    calls: list[str] = []

    monkeypatch.setattr(
        "app.tools.search_papers.search_papers",
        lambda **kwargs: [_paper(pdf_url="https://example.com/p1.pdf")],
    )
    monkeypatch.setattr(
        "app.tools.fetch_metadata.fetch_batch_details",
        lambda papers: papers,
    )

    def fake_download(papers, existing=None, save_dir=None, should_cancel=None):
        calls.append("download")
        return {"p1": None}

    def fake_parse(pdf_paths, should_cancel=None):
        calls.append("parse")
        return {}

    monkeypatch.setattr("app.tools.download_pdf.batch_download_pdfs", fake_download)
    monkeypatch.setattr("app.tools.parse_pdf.batch_parse_pdfs", fake_parse)
    monkeypatch.setattr("app.agent.graph._get_llm", lambda: WorkflowLLM())

    result = run_research_agent(
        "帮我调研近五年目标检测相关论文，并生成中文文献综述，引用不少于5篇",
        current_year=2026,
    )

    assert "download" in calls
    assert "parse" not in calls
    assert result["references"] == []
    assert result["generation_blocked"] is True
    assert result["generation_readiness"]["ready"] is False
    assert result["generation_readiness"]["blocking_issues"][0]["code"] == "minimum_references_not_met"
    assert "正文生成已阻止" in result["answer"]
    download_step = next(s for s in result["steps"] if s["step_name"] == "download_pdf")
    assert download_step["output_data"]["downloaded"] == 0
    assert download_step["output_data"]["failed"] == 0
    assert download_step["output_data"]["unavailable"] == 1


def test_download_pdf_node_reports_cnki_as_policy_skip():
    """CNKI 仅使用摘要证据，不应被计为 PDF 下载失败。"""
    from app.agent.nodes import download_pdf_node

    state = {
        "paper_details": [
            {
                "paper_id": "cnki:policy-skip",
                "source": "cnki",
                "title": "课堂行为分析",
                "pdf_url": "https://kns.cnki.net/example.pdf",
                "is_open_access": True,
            }
        ],
        "pdf_paths": {"cnki:stale": "C:/stale/cnki.pdf"},
        "steps": [],
        "errors": [],
    }

    download_pdf_node(state)

    assert state["pdf_paths"]["cnki:policy-skip"] is None
    assert state["pdf_paths"]["cnki:stale"] is None
    output = state["steps"][-1]["output_data"]
    assert output["eligible"] == 0
    assert output["skipped_by_policy"] == 1
    assert output["policy_skipped_ids"] == ["cnki:policy-skip"]
    assert output["downloaded"] == 0
    assert output["failed"] == 0
    assert output["unavailable"] == 0


def test_review_returns_shortfall_message_when_no_cards_survive(monkeypatch):
    monkeypatch.setattr(
        "app.tools.search_papers.search_papers",
        lambda **kwargs: [_paper()],
    )
    monkeypatch.setattr(
        "app.tools.fetch_metadata.fetch_batch_details",
        lambda papers: [],
    )
    monkeypatch.setattr("app.agent.graph._get_llm", lambda: WorkflowLLM())

    result = run_research_agent(
        "帮我调研近五年目标检测论文，引用不少于5篇，并生成相关工作",
        current_year=2026,
        initial_state={
            "our_work": {
                "research_problem": "目标检测鲁棒性",
                "method_name": "检测模型",
                "method_summary": "改进目标检测模型",
            }
        },
    )

    assert "未生成相关工作" in result["answer"]
    assert "未获得可用论文" in result["answer"]
    assert result["answer"] != ""


def test_retrieval_shortfall_uses_requested_deliverable_names():
    from app.agent.nodes import retrieval_shortfall_node

    state = {
        "required_reference_count": 40,
        "ranked_papers": [],
        "start_year": 2023,
        "end_year": 2026,
        "max_papers_explicit": True,
        "requested_sections": ["background", "research_status"],
        "steps": [],
    }
    retrieval_shortfall_node(state)

    assert "未生成研究背景和研究现状" in state["review"]
    assert "未生成相关工作" not in state["review"]
    # 检索零结果必须阻断生成并反映到质量门禁，不能被误报为 success。
    assert state["generation_blocked"] is True
    assert state["quality_gate"]["passed"] is False
    assert state["quality_gate"]["phase"] == "pre_generation"


def test_fetch_detail_node_propagates_cancellation():
    from app.agent.execution import AgentCancelledError
    from app.agent.nodes import fetch_detail_node

    state = {
        "ranked_papers": [{"paper_id": "doi:x", "title": "t"}],
        "steps": [],
        "errors": [],
    }

    def _cancelled():
        return True

    with pytest.raises(AgentCancelledError):
        fetch_detail_node(state, should_cancel=_cancelled)
    # 取消不是失败：不得写入错误与失败步骤。
    assert state.get("errors") == []
    assert all(step.get("step_name") != "fetch_detail" for step in state["steps"])


def test_review_generates_with_available_papers_when_below_requested_count(monkeypatch):
    monkeypatch.setattr(
        "app.tools.search_papers.search_papers",
        lambda **kwargs: [_paper()],
    )
    monkeypatch.setattr(
        "app.tools.fetch_metadata.fetch_batch_details",
        lambda papers: papers,
    )
    monkeypatch.setattr("app.agent.graph._get_llm", lambda: WorkflowLLM())

    result = run_research_agent(
        "帮我调研近五年目标检测论文，引用不少于5篇，并生成相关工作",
        current_year=2026,
    )

    assert "相关工作暂未生成" in result["answer"]
    assert "主要解决什么问题" in result["answer"]
    assert result["generation_blocked"] is True
    assert result["references"] == []


def test_citation_node_prefers_paper_metadata_for_references():
    state = {
        "review": "代表性工作包括 [p1]。",
        "citation_style": "gbt7714",
        "paper_details": [
            {
                "paper_id": "p1",
                "title": "Object Detection",
                "authors": ["Alice Zhang"],
                "year": 2024,
                "venue": "CVPR",
            }
        ],
        "paper_cards": [
            {
                "paper_id": "p1",
                "title": "Object Detection",
                "year": 2024,
                "venue": "CVPR",
                "research_problem": "",
                "method": "",
                "relevance_reason": "",
                "evidence_source": "abstract",
            }
        ],
        "steps": [],
        "errors": [],
    }

    citation_check_node(state)

    assert state["references"]
    assert "Alice Zhang" in state["references"][0]


def test_search_node_uses_multiple_keywords(monkeypatch):
    from app.agent.nodes import search_node

    seen_queries: list[str] = []

    def fake_search_papers(**kwargs):
        seen_queries.append(kwargs["query"])
        # 并发派发下用查询派生 ID，避免 len() 计数竞态导致 ID 撞车。
        return [_paper(paper_id=f"p-{kwargs['query']}")]

    monkeypatch.setattr("app.tools.search_papers.search_papers", fake_search_papers)

    state = {
        "topic": "长视频理解",
        "keywords": ["长视频理解", "long video understanding", "long-form video understanding"],
        "start_year": 2022,
        "end_year": 2026,
        "max_papers": 10,
        "steps": [],
        "errors": [],
    }

    search_node(state)

    # 并发派发下调用记录顺序不确定，只断言检索词集合。
    assert set(seen_queries) == {
        "long video understanding",
        "long-form video understanding",
        "长视频理解",
    }


def test_search_node_selects_topic_and_english_keyword_variants(monkeypatch):
    from app.agent.nodes import search_node

    seen_queries: list[str] = []

    def fake_search_papers(**kwargs):
        seen_queries.append(kwargs["query"])
        # 并发派发下用查询派生 ID，避免 len() 计数竞态导致 ID 撞车。
        return [_paper(paper_id=f"p-{kwargs['query']}")]

    monkeypatch.setattr("app.tools.search_papers.search_papers", fake_search_papers)

    state = {
        "topic": "test",
        "keywords": ["主题", "中文变体", "english one", "english two", "english three", "english four"],
        "start_year": 2022,
        "end_year": 2026,
        "max_papers": 10,
        "steps": [],
        "errors": [],
    }

    search_node(state)

    # 并发派发下调用记录顺序不确定，只断言检索词集合。
    assert set(seen_queries) == {
        "english one",
        "english two",
        "english three",
        "主题",
    }


def test_search_node_keeps_chinese_keyword_when_first_keyword_is_english(monkeypatch):
    """回归测试：当关键词列表第一项是英文（如 canonical_topic 的
    snake_case 形式）而中文关键词排在后面时，中文关键词不应被 limit
    截断丢弃，否则 CNKI/Crossref 等中文数据源永远不会被触发检索。
    """
    from app.agent.nodes import search_node

    seen_queries: list[str] = []

    def fake_search_papers(**kwargs):
        seen_queries.append(kwargs["query"])
        # 并发派发下用查询派生 ID，避免 len() 计数竞态导致 ID 撞车。
        return [_paper(paper_id=f"p-{kwargs['query']}")]

    monkeypatch.setattr("app.tools.search_papers.search_papers", fake_search_papers)

    state = {
        "topic": "课堂行为分析",
        "keywords": [
            "classroom_behavior_analysis",
            "Classroom Behavior Analysis classroom behavior",
            "Education Classroom Behavior Analysis classroom behavior",
            "课堂行为分析",
            "classroom behavior analysis",
            "student behavior recognition in classroom",
        ],
        "start_year": 2023,
        "end_year": 2026,
        "max_papers": 40,
        "steps": [],
        "errors": [],
    }

    search_node(state)

    assert any("\u4e00" <= c <= "\u9fff" for q in seen_queries for c in q), (
        f"no chinese keyword was searched: {seen_queries}"
    )
    assert "课堂行为分析" in seen_queries


def test_search_node_calls_cnki_once_with_pure_chinese_core_query(monkeypatch):
    from app.agent.nodes import search_node

    calls: list[tuple[str, list[str]]] = []

    def fake_search_papers(**kwargs):
        calls.append((kwargs["query"], kwargs["sources"]))
        # 并发派发下用查询派生 ID，避免 len() 计数竞态导致 ID 撞车。
        return [_paper(paper_id=f"p-{kwargs['query']}")]

    monkeypatch.setattr("app.tools.search_papers.search_papers", fake_search_papers)
    state = {
        "topic": "课堂行为分析",
        "keywords": [
            "课堂行为分析",
            "action recognition multimodal learning classroom behavior 课堂行为",
            "classroom behavior recognition",
        ],
        "start_year": 2023,
        "end_year": 2026,
        "max_papers": 40,
        "retrieval_target": 120,
        "steps": [],
        "errors": [],
    }

    search_node(state)

    cnki_calls = [(query, sources) for query, sources in calls if "cnki" in sources]
    assert len(cnki_calls) == 1
    assert cnki_calls[0][0] == "课堂行为分析"


def test_search_node_upgrades_cnki_with_topic_anchor_keyword(monkeypatch):
    """同一批内含主题核心词的锚点词应独占 CNKI 检索位。"""
    from app.agent.nodes import search_node

    calls: list[tuple[str, list[str]]] = []

    def fake_search_papers(**kwargs):
        calls.append((kwargs["query"], kwargs["sources"]))
        # 并发派发下用查询派生 ID，避免 len() 计数竞态导致 ID 撞车。
        return [_paper(paper_id=f"p-{kwargs['query']}")]

    monkeypatch.setattr("app.tools.search_papers.search_papers", fake_search_papers)
    state = {
        "topic": "少样本动作识别",
        "keywords": ["少样本学习", "少样本动作识别", "课堂行为"],
        "start_year": 2022,
        "end_year": 2026,
        "max_papers": 40,
        "retrieval_target": 120,
        "steps": [],
        "errors": [],
    }

    search_node(state)

    cnki_queries = [query for query, sources in calls if "cnki" in sources]
    # 锚点词在场时，泛化词让位，只启动一次浏览器
    assert cnki_queries == ["少样本动作识别"]
    assert state["cnki_anchor_query_used"] == "少样本动作识别"


def test_search_node_cnki_anchor_upgrade_works_across_rounds(monkeypatch):
    """锚点词出现在后续 refine 轮次时，仍应允许一次 CNKI 升级检索。"""
    from app.agent.nodes import search_node

    calls: list[tuple[str, list[str]]] = []

    def fake_search_papers(**kwargs):
        calls.append((kwargs["query"], kwargs["sources"]))
        # 并发派发下用查询派生 ID，避免 len() 计数竞态导致 ID 撞车。
        return [_paper(paper_id=f"p-{kwargs['query']}")]

    monkeypatch.setattr("app.tools.search_papers.search_papers", fake_search_papers)
    state = {
        "topic": "少样本动作识别",
        "keywords": ["少样本学习"],
        "start_year": 2022,
        "end_year": 2026,
        "max_papers": 40,
        "retrieval_target": 120,
        "steps": [],
        "errors": [],
    }
    search_node(state)
    assert [q for q, s in calls if "cnki" in s] == ["少样本学习"]

    # 第二轮：诊断历史已有 CNKI，但锚点词首次出现，允许升级一次
    calls.clear()
    state["keywords"] = ["少样本动作识别"]
    state["source_diagnostics"] = [{"source": "cnki", "status": "success"}]
    search_node(state)
    assert [q for q, s in calls if "cnki" in s] == ["少样本动作识别"]

    # 第三轮：锚点名额已用完，其他纯中文词不再启动浏览器
    calls.clear()
    state["keywords"] = ["少样本视频分类"]
    search_node(state)
    assert [q for q, s in calls if "cnki" in s] == []


def test_search_node_selects_unseen_keywords_before_limiting(monkeypatch):
    from app.agent.nodes import search_node

    seen_queries: list[str] = []

    def fake_search_papers(**kwargs):
        seen_queries.append(kwargs["query"])
        # 并发派发下用查询派生 ID，避免 len() 计数竞态导致 ID 撞车。
        return [_paper(paper_id=f"p-{kwargs['query']}")]

    monkeypatch.setattr("app.tools.search_papers.search_papers", fake_search_papers)

    state = {
        "topic": "test",
        "keywords": ["主题", "english one", "english two", "english three", "english four"],
        "searched_keywords": ["english one"],
        "start_year": 2022,
        "end_year": 2026,
        "max_papers": 10,
        "steps": [],
        "errors": [],
    }

    search_node(state)

    # 并发派发下调用记录顺序不确定，只断言检索词集合。
    assert set(seen_queries) == {"english two", "english three", "english four", "主题"}


def test_drop_redundant_keywords_removes_token_overlapping_variants():
    """同批内 token 重叠 ≥80% 的近义词只保留首个，锚点词永不剔除。"""
    from app.agent.nodes.retrieval import (
        _drop_redundant_keywords,
        _normalized_window_key,
    )

    kept, dropped = _drop_redundant_keywords(
        [
            "few-shot learning action recognition",
            "few-shot learning human action",
            "graph neural network few-shot action",
            "少样本动作识别",
            "少样本学习",
        ],
        topic_anchor="少样本动作识别",
    )
    assert dropped == ["few-shot learning human action"]
    assert kept == [
        "few-shot learning action recognition",
        "graph neural network few-shot action",
        "少样本动作识别",
        "少样本学习",
    ]

    # 锚点词即使与已有词高度重叠也不剔除
    kept, dropped = _drop_redundant_keywords(
        ["few-shot learning", "few-shot learning 少样本动作识别"],
        topic_anchor="少样本动作识别",
    )
    assert dropped == []

    # 词序不同的等价查询共享同一窗口键
    assert _normalized_window_key(
        "action recognition video", 2022, 2026
    ) == _normalized_window_key("video recognition action", 2022, 2026)
    assert _normalized_window_key(
        "action recognition video", 2022, 2026
    ) != _normalized_window_key("action recognition video", 2021, 2026)


def test_search_node_skips_word_order_permuted_query_window(monkeypatch):
    """词序不同但等价的查询不应重复检索（归一化窗口键去重）。"""
    from app.agent.nodes import search_node

    seen_queries: list[str] = []

    def fake_search_papers(**kwargs):
        seen_queries.append(kwargs["query"])
        # 并发派发下用查询派生 ID，避免 len() 计数竞态导致 ID 撞车。
        return [_paper(paper_id=f"p-{kwargs['query']}")]

    monkeypatch.setattr("app.tools.search_papers.search_papers", fake_search_papers)

    state = {
        "topic": "test",
        "keywords": ["action recognition video understanding"],
        "start_year": 2022,
        "end_year": 2026,
        "max_papers": 10,
        "steps": [],
        "errors": [],
    }
    search_node(state)
    assert seen_queries == ["action recognition video understanding"]

    # 第二轮：同一批词的乱序变体应被窗口键拦截，不产生新检索
    seen_queries.clear()
    state["keywords"] = ["video understanding recognition action"]
    search_node(state)
    assert seen_queries == []


def test_search_node_promotes_topic_anchor_into_first_batch(monkeypatch):
    """锚点词被英文词挤出 4 个名额时，应强制替换末位词进入首批。"""
    from app.agent.nodes import search_node

    calls: list[tuple[str, list[str]]] = []

    def fake_search_papers(**kwargs):
        calls.append((kwargs["query"], kwargs["sources"]))
        # 并发派发下用查询派生 ID，避免 len() 计数竞态导致 ID 撞车。
        return [_paper(paper_id=f"p-{kwargs['query']}")]

    monkeypatch.setattr("app.tools.search_papers.search_papers", fake_search_papers)

    state = {
        "topic": "少样本动作识别",
        "core_keywords": ["少样本动作识别"],
        "expanded_keywords": [
            "english one", "english two", "english three", "english four", "少样本学习",
        ],
        "keywords": [
            "english one",
            "english two",
            "english three",
            "english four",
            "少样本学习",
            "少样本动作识别",
        ],
        "start_year": 2022,
        "end_year": 2026,
        "max_papers": 40,
        "retrieval_target": 120,
        "steps": [],
        "errors": [],
    }
    search_node(state)

    queries = [q for q, _ in calls]
    # 4 个名额：3 个英文词 + 锚点置换占位的泛化中文词“少样本学习”；
    # 并发派发下调用记录顺序不确定，只断言集合。
    assert set(queries) == {
        "english one", "english two", "english three", "少样本动作识别",
    }
    # 核心词已由排序直接进入首批，无需记录事后补救式置换。
    step = state["steps"][-1]
    assert step["output_data"]["anchor_promotion"] is None


def test_search_node_keeps_wide_classroom_behavior_log_query_and_routes_cnki(monkeypatch):
    """课堂行为日志场景：主题召回不被范围精确词降级，宽查询仍保留。"""
    from app.agent.nodes import search_node

    calls: list[tuple[str, list[str]]] = []

    def fake_search_papers(**kwargs):
        calls.append((kwargs["query"], kwargs["sources"]))
        return [_paper(paper_id=f"p-{kwargs['query']}")]

    monkeypatch.setattr("app.tools.search_papers.search_papers", fake_search_papers)
    state = {
        "topic": "课堂行为日志分析",
        "core_keywords": ["课堂行为日志分析"],
        "keywords": [
            "课堂行为日志分析",
            "课堂行为日志",
            "课堂行为日志分析 教学互动 学习参与",
            "classroom behavior log analysis",
        ],
        "scope_search_queries": ["课堂行为日志分析 教学互动 学习参与"],
        "scope_query_roles": {
            "课堂行为日志分析 教学互动 学习参与": "scope_precision",
        },
        "start_year": 2023,
        "end_year": 2026,
        "max_papers": 20,
        "retrieval_target": 40,
        "steps": [],
        "errors": [],
    }

    search_node(state)

    queries = [query for query, _sources in calls]
    assert "课堂行为日志分析" in queries
    assert "课堂行为日志" in queries
    cnki_queries = [query for query, sources in calls if "cnki" in sources]
    assert cnki_queries == ["课堂行为日志分析"]
    assert state["scope_query_roles"]["课堂行为日志分析"] == "topic_recall"


def test_drop_redundant_keywords_keeps_wider_query():
    from app.agent.nodes.retrieval import _drop_redundant_keywords

    kept, dropped = _drop_redundant_keywords(
        [
            "classroom behavior log analysis",
            "classroom behavior log analysis teaching",
        ],
        topic_anchor="",
    )

    assert kept == ["classroom behavior log analysis"]
    assert dropped == ["classroom behavior log analysis teaching"]


def test_search_node_records_source_contribution_metrics(monkeypatch):
    """新增唯一论文与累计候选池按来源统计，重复论文不重复计入。"""
    from app.agent.nodes import search_node

    round_results = [
        [
            _paper(paper_id="arxiv:1", title="Alpha", source="arxiv"),
            _paper(paper_id="openalex:1", title="Beta", source="openalex"),
        ],
        [
            _paper(paper_id="arxiv:1", title="Alpha", source="arxiv"),  # 重复
            _paper(paper_id="crossref:1", title="Gamma", source="crossref"),
        ],
    ]
    calls = {"n": 0}

    def fake_search_papers(**kwargs):
        result = round_results[min(calls["n"], len(round_results) - 1)]
        calls["n"] += 1
        return result

    monkeypatch.setattr("app.tools.search_papers.search_papers", fake_search_papers)

    state = {
        "topic": "test",
        "keywords": ["english one"],
        "start_year": 2022,
        "end_year": 2026,
        "max_papers": 10,
        "steps": [],
        "errors": [],
    }
    search_node(state)
    out = state["steps"][-1]["output_data"]
    assert out["new_by_source"] == {"arxiv": 1, "openalex": 1}
    assert out["pool_by_source"] == {"arxiv": 1, "openalex": 1}

    state["keywords"] = ["english two"]
    search_node(state)
    out = state["steps"][-1]["output_data"]
    # 第二轮：重复的 arxiv 论文不计入新增，仅 crossref 为新贡献
    assert out["new_by_source"] == {"crossref": 1}
    assert out["pool_by_source"] == {"arxiv": 1, "openalex": 1, "crossref": 1}


def test_search_node_handles_null_max_papers(monkeypatch):
    from app.agent.nodes import search_node

    seen_max_results: list[int] = []

    def fake_search_papers(**kwargs):
        seen_max_results.append(kwargs["max_results"])
        return [_paper()]

    monkeypatch.setattr("app.tools.search_papers.search_papers", fake_search_papers)

    state = {
        "topic": "时序动作定位",
        "keywords": ["时序动作定位"],
        "start_year": None,
        "end_year": None,
        "max_papers": None,
        "steps": [],
        "errors": [],
    }

    search_node(state)

    assert state["errors"] == []
    assert seen_max_results == [30]


def test_search_node_reports_raw_and_unique_candidate_counts(monkeypatch):
    from app.agent.nodes import search_node

    def fake_search_papers(**kwargs):
        # 并发派发下用查询派生唯一 DOI，避免计数器竞态导致撞车。
        return [
            _paper(
                paper_id=f"source-{kwargs['query']}-same",
                title="Shared Paper",
                doi="10.1000/shared",
            ),
            _paper(
                paper_id=f"source-{kwargs['query']}-unique",
                title=f"Unique Paper {kwargs['query']}",
                doi=f"10.1000/unique-{kwargs['query']}",
            ),
        ]

    monkeypatch.setattr("app.tools.search_papers.search_papers", fake_search_papers)
    state = {
        "topic": "test",
        "keywords": ["english query one", "english query two"],
        "start_year": 2023,
        "end_year": 2026,
        "max_papers": 10,
        "steps": [],
        "errors": [],
    }

    search_node(state)

    output = state["steps"][-1]["output_data"]
    assert output["raw_returned_count"] == 4
    assert output["count"] == 3
    assert output["new_unique_count"] == 3
    assert output["duplicate_removed_count"] == 1
    assert state["last_search_new_results"] == 3


def test_agent_stops_when_all_search_sources_return_empty(monkeypatch):
    class FakePlanningLLM:
        def complete(self, prompt: str, **kwargs) -> str:
            return """
            {
              "keywords": ["object detection", "目标检测"],
              "required_concepts": [
                {"concept": "目标检测", "terms": ["object detection", "目标检测"]}
              ],
              "excluded_title_terms": []
            }
            """

    monkeypatch.setattr("app.agent.graph._get_llm", lambda: FakePlanningLLM())
    monkeypatch.setattr("app.tools.search_papers.search_papers", lambda **kwargs: [])

    result = run_research_agent("帮我生成目标检测研究现状", current_year=2026)

    assert "论文检索失败" in result["answer"]


def test_final_answer_quarantines_draft_when_unique_citations_below_requested_minimum():
    state = {
        "intent": "generate_review",
        "review": "## 相关工作\n\n这里只引用了 [p1]。",
        "references": ["Author A. Paper One[J]. CVPR, 2024.", "Author B. Paper Two[J]. CVPR, 2024."],
        "citation_validation": {"valid": True, "cited_ids": ["p1"]},
        "unique_cited_paper_count": 1,
        "final_requirement_met": False,
        "retrieval_requirement_met": True,
        "max_papers_explicit": True,
        "required_reference_count": 2,
        "max_papers": 2,
        "paper_cards": [],
        "steps": [],
        "errors": [],
    }

    final_answer_node(state)

    # 引用数不足是硬约束失败：草稿必须隔离，不能以正式样式展示。
    assert "## 相关工作" not in state["answer"]
    assert "正式正文已被质量门禁阻止" in state["answer"]
    assert "实际有效引用 1 篇" in state["answer"]
    assert state["body"] == ""
    issue = next(
        item for item in state["quality_gate"]["blocking_issues"]
        if item["code"] == "minimum_cited_references_not_met"
    )
    assert issue["actual"] == 1
    assert state["quality_gate"]["draft_disposition"] == "quarantined"
    assert state["quarantined_draft"].startswith("## 相关工作")
    assert state["steps"][-1]["status"] == "blocked"


def test_expand_year_when_default_range_is_insufficient(monkeypatch):
    from app.agent.nodes import expand_search_year_node

    seen_ranges: list[tuple[int, int]] = []

    def fake_search_papers(**kwargs):
        seen_ranges.append((kwargs["start_year"], kwargs["end_year"]))
        return [_paper(
            paper_id="p2",
            title="Transformer Object Detection",
            year=2023,
        )]

    monkeypatch.setattr("app.tools.search_papers.search_papers", fake_search_papers)
    state = {
        "topic": "目标检测",
        "keywords": ["目标检测"],
        "start_year": 2024,
        "end_year": 2026,
        "max_papers": 2,
        "year_range_explicit": False,
        "candidate_papers": [_paper()],
        "ranked_papers": [_paper()],
        "steps": [],
        "errors": [],
    }

    expand_search_year_node(state)

    assert seen_ranges == [(2023, 2023)]
    assert state["start_year"] == 2023
    assert state["search_expanded"] is True
    assert len(state["ranked_papers"]) == 2
    assert state["steps"][-1]["output_data"]["expanded"] is True


def test_strict_year_range_is_never_auto_extended(monkeypatch):
    from app.agent.nodes import expand_search_year_node

    def fail_search(**kwargs):
        raise AssertionError("explicit year range must not be expanded")

    monkeypatch.setattr("app.tools.search_papers.search_papers", fail_search)
    state = {
        "start_year": 2024,
        "end_year": 2026,
        "max_papers": 30,
        "year_range_explicit": True,
        "strict_year_range": True,
        "ranked_papers": [_paper()],
        "steps": [],
        "errors": [],
    }

    expand_search_year_node(state)

    assert state["steps"][-1]["output_data"]["expanded"] is False
    assert state["steps"][-1]["output_data"]["reason"] == "explicit_user_defined_year_range"


def test_non_strict_explicit_relative_range_is_not_expanded(monkeypatch):
    from app.agent.nodes import expand_search_year_node

    seen_ranges: list[tuple[int, int]] = []

    def fake_search_papers(**kwargs):
        raise AssertionError("explicit relative range must not be expanded")

    monkeypatch.setattr("app.tools.search_papers.search_papers", fake_search_papers)
    state = {
        "topic": "目标检测",
        "keywords": ["目标检测"],
        "start_year": 2024,
        "end_year": 2026,
        "max_papers": 3,
        "year_range_explicit": True,
        "strict_year_range": False,
        "candidate_papers": [_paper()],
        "ranked_papers": [_paper()],
        "steps": [],
        "errors": [],
    }

    expand_search_year_node(state)

    assert seen_ranges == []
    assert state["start_year"] == 2024
    assert state["retrieval_requirement_met"] is False
    assert state["steps"][-1]["output_data"]["reason"] == "explicit_user_defined_year_range"


def test_incremental_rerank_only_covers_new_papers_and_boundary(monkeypatch):
    """增量轮末尾 LLM 重排只覆盖新论文 + 旧榜尾部边界段，头部段保持原序。"""
    from app.agent.retrieval_loop import search_rank_with_refinement
    from app.agent.nodes.base import _paper_identity_key

    old_papers = [_paper(paper_id=f"o{i}", title=f"Old Paper {i}") for i in range(1, 11)]
    new_papers = [_paper(paper_id=f"n{i}", title=f"New Paper {i}") for i in range(1, 3)]
    ranked_all = old_papers + new_papers

    def fake_search_node(state, should_cancel=None):
        state["candidate_papers"] = list(ranked_all)
        state["incremental_new_paper_keys"] = [
            _paper_identity_key(paper) for paper in new_papers
        ]

    def fake_rank_node(state, llm=None):
        state["ranked_papers"] = list(state["candidate_papers"])
        state["screening_report"] = {}

    rerank_input: list = []

    def fake_rerank(papers, **kwargs):
        rerank_input.extend(papers)
        return list(papers)

    monkeypatch.setattr("app.agent.retrieval_loop.search_node", fake_search_node)
    monkeypatch.setattr("app.agent.retrieval_loop.rank_node", fake_rank_node)
    monkeypatch.setattr("app.tools.rank_papers.llm_rerank_papers", fake_rerank)

    state = {
        "topic": "增量主题",
        "incremental_retrieval": True,
        "max_papers": 12,
        "steps": [],
        "errors": [],
    }

    search_rank_with_refinement(state, llm=object())

    # 重排输入 = 旧榜尾部 6 篇（target//2）+ 新增 2 篇，头部 4 篇不参与
    assert {p["paper_id"] for p in rerank_input} == {
        "o5", "o6", "o7", "o8", "o9", "o10", "n1", "n2",
    }
    assert state["screening_report"]["llm_rerank_mode"] == "incremental_merge"
    # 头部段保持原序拼接在重排结果之前
    final_ids = [p["paper_id"] for p in state["ranked_papers"]]
    assert final_ids[:4] == ["o1", "o2", "o3", "o4"]
    assert set(final_ids) == {f"o{i}" for i in range(1, 11)} | {"n1", "n2"}


def test_citation_gap_repair_snapshot_restores_polluted_state():
    """H4 回归：修复轮异常/退化后必须能整体恢复修复前产物。

    2026-08 审计发现该回滚逻辑无测试锁定。快照往返必须保证：
    引用表、正文、写作计划一起回到修复前状态，且增量检索残留被清除。
    """
    from app.agent.graph import (
        _snapshot_generation_products,
        _restore_pre_repair_snapshot,
    )

    pre = {
        "review": "修复前草稿[p1][p2]",
        "references": [{"id": 1}, {"id": 2}],
        "unique_cited_paper_count": 2,
        "writing_plans": [{"deliverable_type": "research_status"}],
    }
    state = dict(pre)

    state["_citation_gap_repair_snapshot"] = _snapshot_generation_products(state)
    # 模拟修复轮中途崩溃前的半途污染：产物被部分改写、增量残留写入
    state["review"] = "半途草稿[p9]"
    state["references"] = [{"id": 9}]
    state["unique_cited_paper_count"] = 1
    state["writing_plans"] = [{"deliverable_type": "research_status", "polluted": True}]
    state["incremental_retrieval"] = True
    state["incremental_search_window"] = {"start_year": 2022, "end_year": 2026}
    state["citation_shortfall_count"] = 12

    _restore_pre_repair_snapshot(state, clear_incremental=True)

    assert state["review"] == pre["review"]
    assert state["references"] == pre["references"]
    assert state["unique_cited_paper_count"] == 2
    assert state["writing_plans"] == pre["writing_plans"]
    # 增量残留与缺口计数随回滚清除，不会误导下一轮门禁
    assert "incremental_retrieval" not in state
    assert "incremental_search_window" not in state
    assert state["citation_shortfall_count"] == 0
    # 快照键本身被消费掉，重复回滚不会复活旧状态
    assert "_citation_gap_repair_snapshot" not in state
    assert _restore_pre_repair_snapshot(state) == {}


def test_reset_generation_products_clears_stale_plans_and_authorization():
    """H5 回归：三入口共用的重置 helper 必须清掉全部写作产物键。

    陈旧的写作计划/授权残留进入新一轮写作会造成章节错位或越权引用
    判定失真；非产物键（如 topic）不得被误删。
    """
    from app.agent.graph import _reset_generation_products, _GENERATION_PRODUCT_KEYS

    state = {key: "stale" for key in _GENERATION_PRODUCT_KEYS}
    state["topic"] = "课堂行为分析"
    state["paper_cards"] = [{"paper_id": "p1"}]

    _reset_generation_products(state)

    for key in _GENERATION_PRODUCT_KEYS:
        assert key not in state
    # 非产物键不受影响
    assert state["topic"] == "课堂行为分析"
    assert state["paper_cards"] == [{"paper_id": "p1"}]


def test_local_rewrite_targets_derive_from_repairs_ccc_and_section_diagnostics():
    """local_rewrite 同时派生 claim repair、CCC 与失败章节的目标句。"""
    from app.agent.graph import _build_regeneration_recovery_plan

    state = {
        "conservative_regeneration": True,
        "validated_routes": [{"route_id": "r1", "paper_ids": ["p1"]}],
        "claim_plans": [{"route_id": "r1", "claims": []}],
        "claim_verification": {
            "claims": [
                {"claim_id": "c001", "sentence": "第一句已有支持。", "factual": True, "support_status": "supported"},
                {"claim_id": "c002", "sentence": "第二句需要修复。", "factual": True, "support_status": "unsupported"},
                {"claim_id": "c003", "sentence": "第三句章节失败。", "factual": True, "support_status": "unsupported"},
            ]
        },
        "claim_repairs": {"removed_claim_ids": ["c002"]},
        "claim_citation_consistency": {
            "inconsistent_samples": [{"sentence": "第二句需要修复。"}],
        },
        "writer_section_diagnostics": [{
            "sections": [{"section_id": "theme_T1", "status": "fallback", "sentence_index": 3}],
        }],
        "quality_gate": {"blocking_issues": [{"code": "claim_evidence_quality_not_met"}]},
    }

    plan = _build_regeneration_recovery_plan(state)

    assert plan["mode"] == "local_rewrite"
    assert plan["target_sentence_indices"] == [2, 3]
    assert plan["target_claim_ids"] == ["c002"]


def test_conservative_regeneration_reuses_routes_and_claim_plan(monkeypatch):
    """纯文本/引用门禁失败时不得重新跑路线验证和 Claim Plan。"""
    from app.agent.graph import regenerate_research_agent

    def unexpected(*args, **kwargs):
        raise AssertionError("same-evidence local rewrite must reuse this stage")

    monkeypatch.setattr("app.agent.graph.validate_routes_node", unexpected)
    monkeypatch.setattr("app.agent.graph.claim_plan_node", unexpected)
    monkeypatch.setattr("app.agent.graph.global_evidence_gate_node", unexpected)

    def fake_generate(state, should_cancel=None):
        state["review"] = "## 研究现状\n\n保守重写后的正文。"

    monkeypatch.setattr("app.agent.graph._generate_deliverables_or_block", fake_generate)
    monkeypatch.setattr(
        "app.agent.graph.final_answer_node",
        lambda state: state.update({"answer": state.get("review", "")}),
    )
    state = {
        "intent": "generate_review",
        "topic": "课堂行为分析",
        "core_deliverables": ["research_status"],
        "paper_details": [_paper(paper_id="p1")],
        "paper_cards": [_paper(paper_id="p1")],
        "validated_routes": [{"route_id": "r1", "paper_ids": ["p1"]}],
        "claim_plans": [{"route_id": "r1", "claims": []}],
        "claim_evidence_gate": {"passed": True},
        "global_evidence_gate": {"status": "EVALUATED"},
        "theme_synthesis": [{"theme_id": "r1"}],
        "quality_gate": {
            "passed": False,
            "blocking_issues": [{"code": "final_text_integrity_not_met"}],
        },
        "conservative_regeneration": True,
        "steps": [],
        "errors": [],
    }

    result = regenerate_research_agent(state)

    recovery_step = next(
        item for item in result["steps"]
        if item["step_name"] == "regeneration_recovery_plan"
    )
    assert recovery_step["output_data"]["mode"] == "local_rewrite"
    assert result["research_state"]["claim_plans"] == state["claim_plans"]


def _record_post_writing_chain(monkeypatch) -> list[str]:
    """把写作后验证链的四个阶段替换为顺序记录器。"""
    calls: list[str] = []
    monkeypatch.setattr(
        "app.agent.graph._claim_alignment_check",
        lambda state: calls.append("claim_alignment"),
    )
    monkeypatch.setattr(
        "app.agent.graph.verify_claims_node",
        lambda state, llm=None, **kwargs: calls.append("verify_claims"),
    )
    monkeypatch.setattr(
        "app.agent.graph.citation_check_node",
        lambda state, llm=None: calls.append("citation_check"),
    )
    monkeypatch.setattr(
        "app.agent.graph._check_claim_citation_consistency",
        lambda state: calls.append("citation_authorization"),
    )
    return calls


def _draft_state() -> dict:
    return {
        "claim_plans": [{"route_id": "r1", "claims": []}],
        "review": "## 研究现状\n\n正文[1]。",
        "writing_plans": [{"sections": []}],
        "citation_map": {"p1": 1},
    }


def test_post_writing_chain_runs_every_stage_in_fixed_order(monkeypatch):
    """共享验证链固定顺序，且四个阶段一个都不能漏。"""
    from app.agent.graph import _verify_generated_draft

    calls = _record_post_writing_chain(monkeypatch)
    stages: list[str] = []

    _verify_generated_draft(_draft_state(), checkpoint=stages.append)

    assert calls == [
        "claim_alignment", "verify_claims", "citation_check", "citation_authorization",
    ]
    # 进度检查点由调用方决定，但阶段名必须与实际执行顺序一致。
    assert stages == ["claim_alignment", "verify_claims", "citation_check"]


def test_post_writing_chain_can_defer_citation_authorization(monkeypatch):
    """run 主链把授权一致性推迟到引用缺口修复之后再判定。"""
    from app.agent.graph import _verify_generated_draft

    calls = _record_post_writing_chain(monkeypatch)

    _verify_generated_draft(_draft_state(), check_citation_authorization=False)

    assert calls == ["claim_alignment", "verify_claims", "citation_check"]


def test_post_writing_chain_forwards_local_verification_targets(monkeypatch):
    """局部重写的目标句必须原样传给逐句验证节点。"""
    from app.agent.graph import _verify_generated_draft

    received: dict = {}
    monkeypatch.setattr("app.agent.graph._claim_alignment_check", lambda state: None)
    monkeypatch.setattr("app.agent.graph.citation_check_node", lambda state, llm=None: None)
    monkeypatch.setattr("app.agent.graph._check_claim_citation_consistency", lambda state: None)
    monkeypatch.setattr(
        "app.agent.graph.verify_claims_node",
        lambda state, llm=None, **kwargs: received.update(kwargs),
    )

    _verify_generated_draft(
        _draft_state(),
        verify_claims_kwargs={
            "target_sentence_indices": [2],
            "target_claim_ids": ["c002"],
            "verification_scope": {"mode": "local", "previous_report": {"claims": []}},
        },
    )

    assert received["target_sentence_indices"] == [2]
    assert received["target_claim_ids"] == ["c002"]
    assert received["verification_scope"]["mode"] == "local"


def test_post_writing_chain_skips_stages_without_their_inputs(monkeypatch):
    """缺少写作计划或正文时只跳过对应阶段，不改变其余阶段。"""
    from app.agent.graph import _verify_generated_draft

    calls = _record_post_writing_chain(monkeypatch)

    _verify_generated_draft({"claim_plans": [{"route_id": "r1"}], "review": "正文。"})

    assert calls == ["claim_alignment", "citation_authorization"]
