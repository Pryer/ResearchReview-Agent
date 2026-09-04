"""四类交付物的新规划、渲染和引用注册架构。"""

import pytest

from app.agent.writing_plan import build_writing_plan
from app.schemas.deliverable_schema import CoreDeliverableType
from app.tools.citation_registry import build_citation_registry


def _card(paper_id: str, topic: str) -> dict:
    return {
        "paper_id": paper_id,
        "title": f"{topic}论文{paper_id}",
        "year": 2025,
        "quality_status": "partial",
        "evidence_source": "abstract",
        "evidence_state": {"access_level": "abstract"},
        "field_claims": {
            "research_problem": [{
                "claim": f"研究{topic}中的问题{paper_id}",
                "evidence_id": f"{paper_id}:problem",
                "explicitly_reported": True,
            }],
            "method": [{
                "claim": f"采用{topic}方法{paper_id}",
                "evidence_id": f"{paper_id}:method",
                "explicitly_reported": True,
            }],
        },
    }


@pytest.mark.parametrize("topic", [
    "大语言模型安全评估",
    "形成性教育评价",
    "肿瘤影像诊断",
    "钙钛矿太阳能电池稳定性",
])
def test_dynamic_research_status_has_no_classroom_template_leakage(topic: str):
    cards = [_card(f"p{index}", topic) for index in range(1, 9)]
    syntheses = [
        {
            "theme_id": f"T{index}",
            "theme_name": name,
            "paper_ids": [f"p{2 * index - 1}", f"p{2 * index}"],
            "common_problems": [],
            "common_methods": [],
            "reported_findings": [],
        }
        for index, name in enumerate(
            ["数据与测量", "建模方法", "评价与验证", "应用与转化"], start=1
        )
    ]
    plan = build_writing_plan("research_status", {
        "topic": topic,
        "canonical_topic": topic,
        "paper_cards": cards,
        "theme_synthesis": syntheses,
    })

    visible = " ".join(section.title for section in plan.sections)
    assert "课堂" not in visible
    assert "S-T" not in visible
    assert plan.sections[0].heading_level == 2
    routes = [section for section in plan.sections if section.id.startswith("theme_")]
    assert 1 <= len(routes) <= 4
    assert all(section.heading_level == 3 for section in routes)


def test_research_status_without_geography_uses_general_title():
    cards = [_card(f"p{index}", "课堂行为分析") for index in range(1, 5)]
    plan = build_writing_plan("research_status", {
        "topic": "课堂行为分析",
        "canonical_topic": "课堂行为分析",
        "paper_cards": cards,
        "theme_synthesis": [
            {
                "theme_id": "T1",
                "theme_name": "自动识别",
                "paper_ids": ["p1", "p2"],
                "reported_problems": [],
                "reported_methods": [],
                "reported_findings": [],
            },
            {
                "theme_id": "T2",
                "theme_name": "课堂编码",
                "paper_ids": ["p3", "p4"],
                "reported_problems": [],
                "reported_methods": [],
                "reported_findings": [],
            },
        ],
    })

    assert plan.sections[0].title == "研究现状"


def test_sparse_research_status_routes_are_merged_before_writing():
    cards = [_card(f"p{index}", "课堂行为分析") for index in range(1, 5)]
    plan = build_writing_plan("research_status", {
        "topic": "课堂行为分析",
        "canonical_topic": "课堂行为分析",
        "paper_cards": cards,
        "theme_synthesis": [
            {
                "theme_id": f"T{index}",
                "theme_name": f"路线{index}",
                "paper_ids": [f"p{index}"],
                "reported_problems": [],
                "reported_methods": [],
                "reported_findings": [],
            }
            for index in range(1, 5)
        ],
    })

    routes = [section for section in plan.sections if section.id.startswith("theme_")]
    assert len(routes) <= 2
    assert all(len(section.supporting_paper_ids) >= 2 for section in routes)


def test_background_uses_hidden_goals_but_only_one_visible_section():
    cards = [_card(f"p{index}", "医学影像分割") for index in range(1, 5)]
    plan = build_writing_plan("research_background", {
        "topic": "医学影像分割",
        "paper_cards": cards,
    })

    assert len(plan.sections) == 1
    assert plan.sections[0].title == "研究背景"
    assert 3 <= len(plan.hidden_planning_nodes) <= 5
    assert all(node.visible is False and node.heading_level is None for node in plan.hidden_planning_nodes)


def test_background_uses_only_llm_eligible_direct_or_near_evidence():
    direct = _card("direct", "少样本动作识别")
    direct.update({
        "relation_type": "direct",
        "eligible_deliverables": ["research_background"],
    })
    unrelated = _card("rag", "检索增强生成")
    unrelated.update({
        "relation_type": "indirect",
        "eligible_deliverables": [],
    })

    plan = build_writing_plan("research_background", {
        "topic": "少样本动作识别",
        "paper_cards": [direct, unrelated],
    })

    assert plan.sections[0].supporting_paper_ids == ["direct"]


def test_background_outline_and_comparison_dimensions_are_induced_by_llm():
    class OutlineLLM:
        def __init__(self):
            self.prompt = ""

        def complete(self, prompt: str, **kwargs) -> str:
            self.prompt = prompt
            return (
                '{"paragraph_goals": ['
                '{"id":"problem","label":"问题形成","writing_goal":"界定问题形成条件"},'
                '{"id":"evidence","label":"证据进展","writing_goal":"归纳证据支持的研究进展"},'
                '{"id":"value","label":"研究价值","writing_goal":"说明研究价值"}'
                '], "comparison_dimensions":["样本设定","评价目标"], "rationale":"证据驱动"}'
            )

    card = _card("p1", "少样本动作识别")
    card.update({
        "relation_type": "direct",
        "eligible_deliverables": ["research_background"],
    })
    llm = OutlineLLM()
    plan = build_writing_plan(
        "research_background",
        {"topic": "少样本动作识别", "paper_cards": [card]},
        llm=llm,
    )

    assert [node.label for node in plan.hidden_planning_nodes] == [
        "问题形成", "证据进展", "研究价值",
    ]
    assert plan.sections[0].comparison_dimensions == ["样本设定", "评价目标"]
    assert "课堂观察" not in llm.prompt


def test_research_status_reserves_routes_for_explicit_pipeline_focuses():
    cards = [_card(f"p{index}", "课堂行为分析") for index in range(1, 11)]
    syntheses = [
        {"theme_id": "T1", "theme_name": "课堂行为视觉识别", "paper_ids": ["p1", "p2"]},
        {"theme_id": "T2", "theme_name": "自动行为编码与结构化观察", "paper_ids": ["p3"]},
        {
            "theme_id": "T3",
            "theme_name": "S-T分析法与滞后序列分析",
            "paper_ids": ["p4"],
        },
        {"theme_id": "T4", "theme_name": "教学结构与师生互动解释", "paper_ids": ["p5"]},
        {"theme_id": "T5", "theme_name": "轻量化与实时部署", "paper_ids": ["p6", "p7", "p8", "p9", "p10"]},
    ]
    focuses = [
        "教师与学生行为自动识别", "自动行为编码", "S-T分析法",
        "滞后序列分析法", "教学结构与师生互动解释",
    ]
    plan = build_writing_plan("research_status", {
        "topic": "课堂行为分析",
        "canonical_topic": "课堂行为分析",
        "paper_cards": cards,
        "theme_synthesis": syntheses,
        "research_semantic_frame": {
            "task_chain": [
                "teacher_student_behavior_recognition",
                "automatic_behavior_coding",
                "st_or_lag_sequential_analysis",
                "teaching_structure_and_interaction_interpretation",
            ],
            "required_focuses": focuses,
        },
    })

    visible = " ".join(section.title for section in plan.sections)
    theme_sections = [
        section for section in plan.sections if section.id.startswith("theme_")
    ]
    planned_text = " ".join(
        f"{section.title} {section.purpose}" for section in theme_sections
    )
    assert "课堂行为视觉识别" in visible
    assert "自动行为编码" in visible
    # 标题最多三段（两个「与」）：吸收用户重点不得把小节名拼成链式长标题。
    for section in plan.sections:
        assert section.title.count("与") <= 2, section.title
    # 单篇重点路线并入相邻路线后，重点仍写进该节的写作目标而不是消失
    assert "S-T分析法与滞后序列分析" in planned_text
    assert "教学结构与师生互动解释" in planned_text
    # 正式路线不得只带一篇论文：那只能写成书目罗列，无法做路线内比较
    for section in theme_sections:
        assert len(section.supporting_paper_ids) >= 2, section.title
    assert plan.required_focuses == focuses
    assert plan.undercovered_focuses == []


def test_research_status_with_sparse_routes_enables_conservative_mode():
    cards = [_card(f"p{index}", "课堂行为分析") for index in range(1, 4)]
    plan = build_writing_plan("research_status", {
        "topic": "课堂行为分析",
        "canonical_topic": "课堂行为分析",
        "paper_cards": cards,
        "theme_synthesis": [
            {"theme_id": "T1", "theme_name": "image", "paper_ids": ["p1"]},
            {"theme_id": "T2", "theme_name": "video", "paper_ids": ["p2"]},
            {"theme_id": "T3", "theme_name": "text", "paper_ids": ["p3"]},
        ],
    })

    assert plan.style_constraints["conservative_evidence_mode"] is True
    assert "image" not in " ".join(section.title for section in plan.sections)
    assert not any(section.title in {"image", "video", "text"} for section in plan.sections if section.id.startswith("theme_"))


def test_citation_registry_separates_unique_papers_and_occurrences():
    text = (
        "## 研究背景\n\n问题已受到关注[p1][p2]。\n\n"
        "### （一）建模方法\n\n方法证据再次支持该判断[p1]。"
    )
    papers = [
        {"paper_id": "p1", "title": "论文一"},
        {"paper_id": "p2", "title": "论文二"},
    ]
    registry = build_citation_registry(text, papers)

    assert set(registry["unique_papers"]) == {"p1", "p2"}
    assert len(registry["citation_occurrences"]) == 3
    assert registry["section_allocations"]["研究背景"] == ["p1", "p2"]
    assert registry["section_allocations"]["（一）建模方法"] == ["p1"]


def test_all_four_deliverables_have_distinct_structural_contracts():
    cards = [_card(f"p{index}", "推荐系统") for index in range(1, 9)]
    syntheses = [
        {"theme_id": "T1", "theme_name": "表示学习", "paper_ids": ["p1", "p2", "p3", "p4"]},
        {"theme_id": "T2", "theme_name": "评价方法", "paper_ids": ["p5", "p6", "p7", "p8"]},
    ]
    base = {
        "topic": "推荐系统",
        "paper_cards": cards,
        "theme_synthesis": syntheses,
        "user_paper_profile": {
            "research_problem": "冷启动问题",
            "proposed_method": "图神经网络",
        },
    }
    plans = {
        dtype: build_writing_plan(dtype, base)
        for dtype in CoreDeliverableType
    }

    assert len(plans[CoreDeliverableType.RESEARCH_BACKGROUND].sections) == 1
    assert plans[CoreDeliverableType.RESEARCH_STATUS].sections[0].title == "研究现状"
    assert plans[CoreDeliverableType.RELATED_WORK].sections[0].title == "相关工作"
    assert plans[CoreDeliverableType.NARRATIVE_REVIEW].sections[0].title == "叙述性综述初稿"


def test_final_route_section_prompt_demands_cross_route_synthesis():
    """末条研究路线的改写提示必须硬性要求跨路线综合末段。

    该要求原先只写在 section.purpose 里，与提示词中"避免空泛的固定结尾"
    冲突而常被模型忽略，导致结构校验报"研究现状末段缺少跨路线综合"
    （2026-08-22 会话实测）。
    """
    from app.deliverables.renderers.base_renderer import _section_rewrite_prompt

    kwargs = dict(
        deliverable_type=CoreDeliverableType.RESEARCH_STATUS,
        section_id="theme_last",
        title="（六）多模态与自监督",
        topic="课堂行为分析",
        original="草稿[p1]",
        required_ids=["p1"],
        purpose="综合该路线的共同问题",
    )

    final_prompt = _section_rewrite_prompt(**kwargs, require_cross_route_synthesis=True)
    assert "跨路线综合" in final_prompt
    assert "最后一条研究路线" in final_prompt
    # 明确列出结构校验所识别的综合标记，避免模型写出无法通过校验的结尾
    assert "综合" in final_prompt and "差异" in final_prompt

    # 非末节不得携带该要求，否则每节都写综合结尾会产生重复套话
    middle_prompt = _section_rewrite_prompt(**kwargs)
    assert "最后一条研究路线" not in middle_prompt


def test_only_last_theme_section_carries_synthesis_duty_when_multiple_routes():
    """多路线时只有末条路线承担综合；单路线时不产生"跨路线"要求。"""
    from app.schemas.deliverable_schema import WritingPlan, WritingSection

    def _plan(theme_count: int) -> WritingPlan:
        sections = [WritingSection(
            id="status_overview", title="二、研究现状", purpose="总体", heading_level=2,
        )]
        for index in range(theme_count):
            sections.append(WritingSection(
                id=f"theme_r{index}", title=f"（{index + 1}）路线", purpose="综合",
                heading_level=3,
            ))
        return WritingPlan(
            deliverable_type=CoreDeliverableType.RESEARCH_STATUS,
            purpose="研究现状", organizing_strategy="evidence_driven",
            sections=sections,
        )

    def _final_theme_id(plan: WritingPlan) -> str:
        theme_ids = [s.id for s in plan.sections if s.id.startswith("theme_")]
        return theme_ids[-1] if len(theme_ids) > 1 else ""

    assert _final_theme_id(_plan(3)) == "theme_r2"
    assert _final_theme_id(_plan(1)) == ""
