"""结构化研究计划兼容层测试。"""

from app.agent.research_plan import build_research_request_plan
from app.schemas.agent_schema import IntentResult, SlotResult
from app.schemas.research_plan_schema import DeliverableType, ResearchOperation


def test_composite_review_request_becomes_structured_plan():
    slots = SlotResult(
        topic="课堂行为分析",
        start_year=2024,
        end_year=2026,
        required_reference_count=30,
        retrieval_target=45,
        generation_limit=35,
        year_range_explicit=True,
        requested_sections=["background", "research_status"],
    )
    plan = build_research_request_plan(
        "调研近三年课堂行为分析论文，最少引用30篇，并生成研究背景和研究现状",
        IntentResult(intent="generate_review", confidence=0.9, reason="rule"),
        slots,
        search_plan={"topic": "课堂行为分析", "keywords": ["classroom behavior analysis"]},
    )

    assert plan.topic == "课堂行为分析"
    assert plan.constraints.time.raw_expression == "近三年"
    assert plan.constraints.time.mode == "calendar_year"
    assert "2024—2026" in plan.constraints.time.assumption
    assert plan.constraints.minimum_references == 30
    assert DeliverableType.RESEARCH_BACKGROUND in plan.deliverables
    assert DeliverableType.RESEARCH_STATUS in plan.deliverables
    assert ResearchOperation.METADATA_VERIFICATION in plan.operations
    assert ResearchOperation.VALIDATE_CITATIONS in plan.operations
    assert plan.task_graph[-1].depends_on == ["write"]


def test_search_only_plan_does_not_enable_writing():
    slots = SlotResult(topic="RAG", requested_sections=[])
    plan = build_research_request_plan(
        "检索 RAG 论文",
        IntentResult(intent="search_papers", confidence=0.9, reason="rule"),
        slots,
    )
    assert plan.deliverables == [DeliverableType.PAPER_LIST]
    assert ResearchOperation.SEARCH in plan.operations
    assert ResearchOperation.WRITE not in plan.operations


def test_unselected_scope_creates_clarification_task():
    slots = SlotResult(topic="课堂行为分析")
    plan = build_research_request_plan(
        "课堂行为分析综述",
        IntentResult(intent="generate_review", confidence=0.9, reason="rule"),
        slots,
        topic_interpretations=[
            {"scope_id": "education", "label": "教育学课堂观察"},
            {"scope_id": "vision", "label": "计算机视觉行为识别"},
        ],
    )
    assert plan.clarification.needed is True
    assert plan.operations[0] == ResearchOperation.SCOPE_DISAMBIGUATION
    assert plan.confidence.uncertain_fields == ["scope"]


# plan_patch 相关用例随 app/agent/plan_patch.py 一并移除：
# 该模块从未接入生产链路（多轮修订走 research_conversation_service 的指令解析）。
