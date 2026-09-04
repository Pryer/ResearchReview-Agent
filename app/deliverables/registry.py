"""只保存功能规则，不保存领域固定句式。"""

from __future__ import annotations

from app.schemas.deliverable_schema import (
    CoreDeliverableType,
    DeliverableSpec,
    InputRequirement,
    PlanningConstraints,
    SectionRequirement,
    StructureConstraints,
    ValidationConstraints,
)


_ABSTRACT_OR_BETTER = ["abstract", "partial_full_text", "full_text"]


DELIVERABLE_SPECS = {
    CoreDeliverableType.RESEARCH_BACKGROUND: DeliverableSpec(
        type=CoreDeliverableType.RESEARCH_BACKGROUND,
        purpose="说明研究问题的重要性、现实或理论价值、主要困难和继续研究的必要性",
        required_inputs=[
            InputRequirement(field="topic", reason="必须明确研究主题"),
            InputRequirement(field="verified_paper_cards", reason="背景判断需要多篇文献支持"),
        ],
        required_sections=[
            SectionRequirement(
                id="background_body",
                purpose="依据当前主题和证据，以连续自然段说明问题背景、发展条件与研究意义",
                allowed_evidence_levels=_ABSTRACT_OR_BETTER,
            ),
        ],
        rhetorical_moves=[
            "由当前主题和证据归纳问题场景",
            "说明证据支持的发展条件与既有局限",
            "界定研究对象、理论或实践价值",
            "由已证实问题自然引出继续研究的必要性",
        ],
        evidence_rules=["关键趋势至少由两篇文献支持", "元数据不得支持方法或结果结论"],
        forbidden_patterns=["逐篇论文卡片拼接", "空泛价值判断", "大量实验指标"],
        min_references=2,
        structure=StructureConstraints(
            visible_title="研究背景",
            allow_subsections=False,
            min_paragraphs=3,
            max_paragraphs=5,
            output_form="continuous_prose",
        ),
        planning=PlanningConstraints(
            planning_strategy="topic_adaptive_background",
            dynamic_topic_generation=True,
            allow_topic_merge=True,
        ),
        validation=ValidationConstraints(
            forbidden_visible_sections=[
                "研究问题与场景", "研究价值与必要性", "已有研究方式与主要困难",
                "进一步研究的必要性", "证据范围说明",
            ],
            target_char_range=(1200, 1800),
        ),
    ),
    CoreDeliverableType.RESEARCH_STATUS: DeliverableSpec(
        type=CoreDeliverableType.RESEARCH_STATUS,
        purpose="综合当前主题的主要研究路线、方法进展、证据差异与研究空白",
        required_inputs=[
            InputRequirement(field="topic", reason="必须明确研究主题"),
            InputRequirement(field="dynamic_taxonomy", reason="研究现状必须按当前主题的研究路线组织"),
            InputRequirement(field="theme_synthesis", reason="必须先完成跨论文综合"),
        ],
        required_sections=[
            SectionRequirement(id="status_overview", purpose="概括总体研究进展及主要路线；仅在地域证据可靠时比较不同国家或地区"),
            SectionRequirement(id="research_routes", purpose="按动态主题综合主要路线", allowed_evidence_levels=_ABSTRACT_OR_BETTER),
        ],
        rhetorical_moves=[
            "先概括总体研究进展；地域信息可靠时再比较不同国家或地区",
            "按一至四条研究路线综合近年文献",
            "在路线内部比较而非逐篇复述",
            "末段综合有证据支持的共性不足",
        ],
        evidence_rules=["每个正式主题原则上至少两篇论文；路线数量按当前证据动态确定", "区分作者局限与综合推断"],
        forbidden_patterns=["Other作为正式章节", "按论文卡片顺序拼接", "重复通用总结"],
        min_references=4,
        requires_dynamic_taxonomy=True,
        structure=StructureConstraints(
            visible_title="国内外研究现状",
            allow_subsections=True,
            min_subsections=1,
            # 高引用量综述下 4 个子节会让过宽路线拆分后的子路线又被合并
            # 回巨型小节（实测 3 条子路线被并回 2 节、单节 57 处引用）。
            max_subsections=6,
            output_form="thematic_synthesis",
        ),
        planning=PlanningConstraints(
            planning_strategy="evidence_driven_routes",
            dynamic_topic_generation=True,
            allow_topic_merge=True,
            require_comparative_synthesis=True,
            require_final_synthesis=True,
        ),
        validation=ValidationConstraints(
            forbidden_visible_sections=[
                "研究范围说明", "证据范围说明", "研究路线比较与共性问题",
                "研究空白", "进一步研究的必要性",
            ],
            # 上限随子节预算等比抬升（6 节 × 约 1000 字 + 概述）：否则子节
            # 数量放宽后每次都会触发“字符数未落入建议范围”的噪声告警。
            target_char_range=(3500, 7800),
        ),
    ),
    CoreDeliverableType.RELATED_WORK: DeliverableSpec(
        type=CoreDeliverableType.RELATED_WORK,
        purpose="定位用户研究与直接相关工作的联系、差异和研究位置",
        required_inputs=[
            InputRequirement(field="user_paper_profile.research_problem", reason="需要明确用户论文解决的问题", clarification_question="你的论文主要解决什么问题，并计划采用什么方法或研究路线？"),
            InputRequirement(field="user_paper_profile.method_or_direction", reason="需要依据用户方法或方向选择直接相关文献", clarification_question="你的论文主要解决什么问题，并计划采用什么方法或研究路线？"),
            InputRequirement(field="verified_paper_cards", reason="需要已有工作证据"),
        ],
        required_sections=[
            SectionRequirement(id="direct_research_routes", purpose="确定与用户论文直接相关的路线"),
            SectionRequirement(id="method_comparison", purpose="比较代表性工作和用户研究"),
            SectionRequirement(id="gap_and_positioning", purpose="说明已有不足及用户研究位置"),
        ],
        rhetorical_moves=["围绕用户问题选择文献", "比较已有路线", "谨慎定位用户研究"],
        evidence_rules=["用户贡献只能来自用户输入", "无全文不得比较详细实验设置"],
        forbidden_patterns=["复制完整研究现状", "虚构用户贡献", "声称用户方法全面优于已有工作"],
        min_references=3,
        requires_dynamic_taxonomy=True,
        requires_user_paper_profile=True,
        structure=StructureConstraints(
            visible_title="相关工作",
            allow_subsections=True,
            min_subsections=2,
            max_subsections=4,
            output_form="paper_centered_comparison",
        ),
        planning=PlanningConstraints(
            planning_strategy="paper_problem_method_alignment",
            dynamic_topic_generation=True,
            allow_topic_merge=True,
            require_comparative_synthesis=True,
        ),
        validation=ValidationConstraints(target_char_range=(2500, 4500)),
    ),
    CoreDeliverableType.NARRATIVE_REVIEW: DeliverableSpec(
        type=CoreDeliverableType.NARRATIVE_REVIEW,
        purpose="形成披露检索范围与证据边界的完整叙述性综述初稿",
        required_inputs=[
            InputRequirement(field="search_report", reason="必须真实说明检索和筛选过程"),
            InputRequirement(field="dynamic_taxonomy", reason="综述主体需要动态主题"),
            InputRequirement(field="theme_synthesis", reason="必须完成跨论文综合"),
            InputRequirement(field="evidence_summary", reason="必须披露证据可访问性"),
        ],
        required_sections=[
            SectionRequirement(id="abstract", purpose="概括范围和主要认识"),
            SectionRequirement(id="introduction", purpose="说明主题价值和综述目标"),
            SectionRequirement(id="search_scope", purpose="说明真实检索与筛选流程"),
            SectionRequirement(id="taxonomy_sections", purpose="按动态主题综合研究路线"),
            SectionRequirement(id="method_data_evaluation", purpose="比较方法、数据与评价"),
            SectionRequirement(id="challenges", purpose="归纳研究挑战"),
            SectionRequirement(id="future_directions", purpose="由挑战推导未来方向"),
            SectionRequirement(id="conclusion", purpose="总结主要认识"),
            SectionRequirement(id="evidence_statement", purpose="披露证据质量和生成边界"),
        ],
        rhetorical_moves=["说明综述范围", "建立动态知识结构", "跨路线比较", "公开证据不足"],
        evidence_rules=["只描述真实执行的检索流程", "元数据不得支持事实性结论"],
        forbidden_patterns=["冒充系统综述", "逐篇摘要罗列", "虚构PRISMA或双人筛选", "隐藏全文不足"],
        min_references=8,
        requires_dynamic_taxonomy=True,
        structure=StructureConstraints(
            visible_title="叙述性综述初稿",
            allow_subsections=True,
            min_subsections=4,
            max_subsections=7,
            output_form="full_narrative_review",
        ),
        planning=PlanningConstraints(
            planning_strategy="taxonomy_trajectory_comparison_gap",
            dynamic_topic_generation=True,
            allow_topic_merge=True,
            require_comparative_synthesis=True,
            require_final_synthesis=True,
        ),
        validation=ValidationConstraints(target_char_range=(7000, 12000)),
    ),
}


def get_deliverable_spec(deliverable_type: CoreDeliverableType | str) -> DeliverableSpec:
    return DELIVERABLE_SPECS[CoreDeliverableType(deliverable_type)]
