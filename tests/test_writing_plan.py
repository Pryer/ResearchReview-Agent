"""测试写作规划器与交付物渲染器质量改进。

重点验证：
1. 40 篇文献场景下，研究现状能稳定生成 3~4 个独立方法学路线子节，避免被单一全局锚点吞噬；
2. 子节标题符合学术规范，清洗内部流程标记，不出现机械词频拼接（如“行为自动识别研究”）；
3. 状态渲染器与提示词解耦，结构层次清晰且无常识复述。
"""

import pytest
from app.agent.writing_plan import (
    build_writing_plan,
    _absorb_focus_label,
    _evaluate_focus_coverage,
    _semantic_route_name,
    _merge_and_select_themes,
)
from app.agent.search_plan_builder import build_semantic_search_branches
from app.deliverables.renderers.status_renderer import StatusRenderer
from app.schemas.deliverable_schema import CoreDeliverableType, WritingPlan, WritingSection
from app.schemas.research_plan_schema import (
    ResearchSemanticFrame,
    ResearchMode,
    ResearchMethod,
    EvidenceRequirement,
)


def _make_paper_card(paper_id: str, title: str, summary: str = "", method: str = ""):
    return {
        "paper_id": paper_id,
        "title": title,
        "summary": summary or f"This paper studies {title} with {method}.",
        "abstract": summary or f"We propose {method} for {title}.",
        "quality_status": "valid",
        "relation_type": "direct",
        "eligible_deliverables": ["research_status", "related_work"],
        "evidence_source": "abstract",
        "evidence_state": {
            "access_level": "abstract",
            "full_text_available": False,
        },
        "claims": [
            {
                "field": "research_problem",
                "claim": f"{title}面临的关键挑战",
                "statement": f"{title}面临的关键挑战",
            },
            {
                "field": "method",
                "claim": f"采用{method or '特定机制'}进行特征建模",
                "statement": f"采用{method or '特定机制'}进行特征建模",
            },
            {
                "field": "results",
                "claim": f"在标准基准数据集上提升了识别准确率",
                "statement": f"在标准基准数据集上提升了识别准确率",
            },
        ],
        "field_claims": {
            "research_problem": [{"claim": f"{title}面临的关键挑战", "explicitly_reported": True}],
            "method": [{"claim": f"采用{method or '特定机制'}进行特征建模", "explicitly_reported": True}],
            "results": [{"claim": "在标准基准数据集上提升了识别准确率", "explicitly_reported": True}],
        },
    }


def test_structured_focus_ids_resolve_through_concepts_and_requirements():
    frame = {
        "research_objects": [
            {"id": "obj_classroom_behavior", "label": "classroom behavior", "surface_text": "课堂行为"},
            {"id": "obj_classroom_interaction", "label": "classroom interaction", "surface_text": "课堂互动"},
            {"id": "obj_teaching_behavior", "label": "teaching behavior", "surface_text": "教学行为"},
        ],
        "evidence_requirements": [
            {
                "requirement_id": "ev:obj_classroom_interaction",
                "source_ids": ["obj_classroom_interaction"],
                "aliases": ["课堂互动", "interaction analysis"],
            },
        ],
    }
    sections = [WritingSection(
        id="theme_interaction",
        title="互动模式编码分析与教学行为特征",
        purpose="梳理课堂行为观察与师生互动研究",
    )]

    covered, undercovered = _evaluate_focus_coverage(
        ["classroom_behavior", "classroom_interaction", "teaching_behavior"],
        sections,
        frame,
    )

    assert covered == ["classroom_behavior", "classroom_interaction", "teaching_behavior"]
    assert undercovered == []


def test_40_papers_generate_multi_route_subsections_without_monolithic_collapse():
    """验证40篇少样本动作识别文献场景下，写作规划能生成3~4个独立方法学子节。"""
    # 构造40篇分布在4个技术路线的文献
    temporal_papers = [f"t_{i}" for i in range(1, 11)]
    metric_papers = [f"m_{i}" for i in range(1, 11)]
    vlm_papers = [f"v_{i}" for i in range(1, 11)]
    cross_domain_papers = [f"c_{i}" for i in range(1, 11)]

    cards = []
    for pid in temporal_papers:
        cards.append(_make_paper_card(pid, f"Temporal Alignment Network {pid}", method="时序对齐与动态时间规整"))
    for pid in metric_papers:
        cards.append(_make_paper_card(pid, f"Metric Prototype Learning {pid}", method="度量学习与原型调制网络"))
    for pid in vlm_papers:
        cards.append(_make_paper_card(pid, f"Vision-Language Pretrained Video {pid}", method="多模态视觉语言模型迁移"))
    for pid in cross_domain_papers:
        cards.append(_make_paper_card(pid, f"Cross-Domain Action Adaptation {pid}", method="跨域特征对齐与无监督自适应"))

    # 4个细分技术路线聚类
    syntheses = [
        {
            "theme_id": "route_temporal",
            "theme_name": "时序对齐与动态规整机制",
            "paper_ids": temporal_papers,
            "reported_problems": [{"claim": "视频帧间时序不对齐与动作速度差异", "paper_id": "t_1"}],
            "reported_methods": [{"claim": "采用动态时间规整(DTW)与自注意力时序匹配", "paper_id": "t_2"}],
            "reported_findings": [{"claim": "在 Kinetics-100 上取得显著精度提升", "paper_id": "t_3"}],
            "synthesized_gaps": [{"statement": "长时序复杂动作的计算开销较高", "gap_type": "cross_paper_inference"}],
        },
        {
            "theme_id": "route_metric",
            "theme_name": "度量学习与原型调制网络",
            "paper_ids": metric_papers,
            "reported_problems": [{"claim": "支撑样本稀缺导致原型表征偏移", "paper_id": "m_1"}],
            "reported_methods": [{"claim": "构建任务自适应度量空间与原型修正模块", "paper_id": "m_2"}],
            "reported_findings": [{"claim": "小样本支持集下分类判别性更强", "paper_id": "m_3"}],
            "synthesized_gaps": [{"statement": "对复杂类内差异敏感", "gap_type": "author_reported"}],
        },
        {
            "theme_id": "route_vlm",
            "theme_name": "多模态与视觉语言预训练迁移",
            "paper_ids": vlm_papers,
            "reported_problems": [{"claim": "单视觉模态知识先验不足", "paper_id": "v_1"}],
            "reported_methods": [{"claim": "引入文本语义提示引导视频动作先验", "paper_id": "v_2"}],
            "reported_findings": [{"claim": "零样本与少样本迁移能力大幅提升", "paper_id": "v_3"}],
            "synthesized_gaps": [{"statement": "提示工程对下游动作特定微调要求高", "gap_type": "cross_paper_inference"}],
        },
        {
            "theme_id": "route_cross_domain",
            "theme_name": "跨域与无监督动作泛化",
            "paper_ids": cross_domain_papers,
            "reported_problems": [{"claim": "训练集与测试集分布迁移导致性能骤降", "paper_id": "c_1"}],
            "reported_methods": [{"claim": "基于领域对抗学习与分布对齐", "paper_id": "c_2"}],
            "reported_findings": [{"claim": "跨数据集迁移评测指标显著改善", "paper_id": "c_3"}],
            "synthesized_gaps": [{"statement": "未标注目标域样本上的稳定性有待提升", "gap_type": "cross_paper_inference"}],
        },
    ]

    # 包含一个宽泛任务锚点（模拟“动作识别”，吞噬了所有40篇论文）
    semantic_frame = {
        "canonical_topic": "少样本动作识别",
        "evidence_requirements": [
            {
                "requirement_id": "req_global_action",
                "label": "行为自动识别相关论文证据",
                "route_required": True,
                "route_group": "perception",
            }
        ],
    }

    state = {
        "topic": "少样本动作识别",
        "canonical_topic": "少样本动作识别",
        "paper_cards": cards,
        "theme_synthesis": syntheses,
        "research_semantic_frame": semantic_frame,
    }

    plan = build_writing_plan(CoreDeliverableType.RESEARCH_STATUS, state)

    # 验证子节数量：至少应有 status_overview + 3~4 个独立方法学路线子节
    theme_sections = [s for s in plan.sections if s.id.startswith("theme_")]
    assert len(theme_sections) >= 3, f"期望生成3~4个方法学子节，实际生成了 {len(theme_sections)} 个"

    # 验证子节标题：不应坍缩为单一“行为自动识别研究”
    theme_titles = [s.title for s in theme_sections]
    assert not any("行为自动识别研究" in t for t in theme_titles)

    # 验证标题保留了细分学术路线的特征
    all_titles_text = " ".join(theme_titles)
    assert "时序" in all_titles_text or "对齐" in all_titles_text
    assert "度量" in all_titles_text or "原型" in all_titles_text
    assert "多模态" in all_titles_text or "预训练" in all_titles_text or "视觉语言" in all_titles_text


def test_status_overview_supports_full_eligible_pool_beyond_route_papers():
    """主题综合未挂载的合格证据必须留在研究现状支撑池内。

    回归 2026-08 会话缺陷：overview 支撑集合被路线论文并集整个替换
    （路线 46 篇 vs 合格 79 篇），引用分配候选池与 M15 授权范围随之
    封顶，"不少于 60 篇"在路线并集不足时必然失败。
    """
    route_papers = ["r_1", "r_2", "r_3"]
    unassigned_papers = ["u_1", "u_2"]  # 未进任何路线但完全合格的证据
    cards = [
        *(
            _make_paper_card(pid, f"Metric Prototype Learning {pid}", method="度量学习与原型调制网络")
            for pid in route_papers
        ),
        *(
            _make_paper_card(pid, f"Cross-dataset Evaluation Survey {pid}", method="跨数据集评测")
            for pid in unassigned_papers
        ),
    ]
    syntheses = [{
        "theme_id": "route_metric",
        "theme_name": "度量学习与原型调制网络",
        "paper_ids": route_papers,
        "reported_problems": [{"claim": "支撑样本稀缺导致原型表征偏移", "paper_id": "r_1"}],
        "reported_methods": [{"claim": "构建任务自适应度量空间", "paper_id": "r_2"}],
        "reported_findings": [{"claim": "小样本支持集下分类判别性更强", "paper_id": "r_3"}],
        "synthesized_gaps": [{"statement": "对复杂类内差异敏感", "gap_type": "author_reported"}],
    }]
    state = {
        "topic": "少样本动作识别",
        "canonical_topic": "少样本动作识别",
        "paper_cards": cards,
        "theme_synthesis": syntheses,
        "research_semantic_frame": {
            "canonical_topic": "少样本动作识别",
            "evidence_requirements": [{
                "requirement_id": "req_metric",
                "label": "度量学习相关证据",
                "route_required": True,
                "route_group": "metric",
                "aliases": ["度量学习", "metric learning"],
            }],
        },
    }

    plan = build_writing_plan(CoreDeliverableType.RESEARCH_STATUS, state)

    overview = next(s for s in plan.sections if s.id == "status_overview")
    # 未进路线的合格证据也在 overview 的支撑/授权池中
    assert set(unassigned_papers) <= set(overview.supporting_paper_ids)
    # 路线论文保持优先序（分配层 fit score 仍优先主题论文）
    assert overview.supporting_paper_ids[:len(route_papers)] == route_papers
    # 主题章节支撑不扩散：仍只用本路线论文
    theme_sections = [s for s in plan.sections if s.id.startswith("theme_")]
    assert theme_sections
    for theme in theme_sections:
        assert set(theme.supporting_paper_ids) <= set(route_papers)


def test_semantic_route_name_cleans_internal_suffixes_and_preserves_academic_names():
    """验证学术标题命名优化：清洗内部标签后缀，优先使用高质量学术名称。"""
    # 场景1：带有流程后缀的标签，清洗后得到规范名称
    name1 = _semantic_route_name(
        labels=["时序对齐相关论文证据"],
        members=[{"label": "时序对齐相关论文证据"}],
        route={"theme_name": "时序建模"},
    )
    assert name1 == "时序对齐"

    # 场景2：多个标签组合清洗
    name2 = _semantic_route_name(
        labels=["原型度量相关证据", "时空特征匹配相关论文"],
        members=[],
        route={},
    )
    assert name2 == "原型度量与时空特征匹配"

    # 场景3：无显式有效标签时，使用既有高质量主题名称
    name3 = _semantic_route_name(
        labels=[],
        members=[],
        route={"theme_name": "多模态视觉语言提示迁移"},
    )
    assert name3 == "多模态视觉语言提示迁移"


def test_semantic_route_name_strips_internal_related_literature_labels():
    """"…相关文献"是内部覆盖用语，不能作为用户可见的小节标题。

    回归 2026-08-30 实测缺陷：正文出现
    "（一）课堂行为分析相关文献与师生行为相关文献与教学互动相关文献"。
    判据带"相关"前缀，因此"灰色文献"这类真实名称不受影响。
    """
    assert _semantic_route_name(
        labels=["课堂行为分析相关文献", "师生行为相关文献", "教学互动相关文献"],
        members=[],
        route={},
    ) == "课堂行为分析与师生行为与教学互动"
    assert _semantic_route_name(
        labels=["灰色文献"], members=[], route={},
    ) == "灰色文献"


def test_search_plan_branches_high_citation_target():
    """验证高引用目标 (>=40) 下生成独立细分方法学检索分支。"""
    frame = ResearchSemanticFrame(
        canonical_topic="少样本动作识别",
        research_mode=ResearchMode.TECHNOLOGY_APPLIED_TO_DOMAIN,
        methods=[
            ResearchMethod(id="m1", label="Temporal Alignment", explicit=True, category="technical"),
            ResearchMethod(id="m2", label="Metric Learning", explicit=True, category="technical"),
            ResearchMethod(id="m3", label="Vision-Language Models", explicit=True, category="technical"),
        ],
        evidence_requirements=[
            EvidenceRequirement(
                requirement_id="req1",
                label="时序规整方法",
                evidence_role="technical_method",
                route_group="temporal_alignment",
                aliases=["DTW", "temporal matching"],
            ),
            EvidenceRequirement(
                requirement_id="req2",
                label="原型度量网络",
                evidence_role="technical_method",
                route_group="metric_learning",
                aliases=["prototype rectification"],
            ),
        ],
    )

    # 目标为 40 篇
    branches = build_semantic_search_branches(frame, retrieval_target=40)
    branch_types = [b.branch_type for b in branches]

    # 应包含各个方法学子方向的独立分支
    assert any("method_subroute" in bt for bt in branch_types)
    assert any("requirement_route" in bt for bt in branch_types)


def test_status_renderer_fallback_structure_and_no_boilerplate():
    """验证研究现状渲染器 fallback 输出结构清晰且无僵化机械套话。"""
    cards = [
        _make_paper_card("p1", "Temporal Model", method="时序注意力"),
        _make_paper_card("p2", "Metric Model", method="原型度量"),
    ]
    syntheses = [
        {
            "theme_id": "theme_1",
            "theme_name": "时序注意力与特征对齐",
            "paper_ids": ["p1"],
            "reported_problems": [{"claim": "动作序列存在局部异步与时序形变", "paper_id": "p1"}],
            "reported_methods": [{"claim": "设计多尺度时序注意力机制", "paper_id": "p1"}],
            "reported_findings": [{"claim": "在 UCF101 上提升了 4.2% 准确率", "paper_id": "p1"}],
            "synthesized_gaps": [{"statement": "对超长视频的计算复杂度高", "gap_type": "cross_paper_inference"}],
        }
    ]

    state = {
        "canonical_topic": "少样本动作识别",
        "paper_cards": cards,
        "theme_synthesis": syntheses,
    }

    plan = build_writing_plan(CoreDeliverableType.RESEARCH_STATUS, state)
    renderer = StatusRenderer()
    rendered = renderer.render_fallback(plan, state, cards)

    # 验证不包含僵化五段式开头
    assert "在研究问题方面，" not in rendered
    assert "在方法路线方面，" not in rendered
    assert "在实验结果方面，" not in rendered

    # 验证包含层级式自然叙述
    assert "该路线主要聚焦于" in rendered or "主要聚焦于" in rendered
    assert "在具体方法机制上" in rendered or "方法机制" in rendered
    assert "[p1]" in rendered


def test_status_renderer_fallback_uses_nontechnical_route_and_evidence_language():
    plan = WritingPlan(
        deliverable_type=CoreDeliverableType.RESEARCH_STATUS,
        purpose="梳理研究现状",
        organizing_strategy="按研究路线组织",
        sections=[
            WritingSection(
                id="status_overview",
                title="研究现状概述",
                purpose="概述",
                supporting_paper_ids=["p1"],
            ),
            WritingSection(
                id="theme_history",
                title="（一）历史叙事与档案研究",
                purpose="梳理路线",
                supporting_paper_ids=["p1"],
            ),
        ],
    )
    cards = [{
        "paper_id": "p1",
        "claims": [{"field": "research_problem", "claim": "论文考察地方档案中的记忆建构"}],
    }]
    synthesis = [{
        "theme_id": "history",
        "theme_name": "历史叙事与档案研究",
        "reported_problems": [{"claim": "论文考察地方档案中的记忆建构", "paper_id": "p1"}],
    }]

    rendered = StatusRenderer().render_fallback(
        plan,
        {"canonical_topic": "地方文化记忆", "theme_synthesis": synthesis},
        cards,
    )

    assert "地方文化记忆" in rendered
    assert "历史叙事与档案研究" in rendered
    assert "论文考察地方档案中的记忆建构[p1]" in rendered
    assert all(term not in rendered for term in ("技术路径", "技术路线", "性能增益", "泛化能力", "计算效率", "可解释性"))


def test_status_fallback_neutralizes_source_voice_and_avoids_overview_repeat():
    plan = WritingPlan(
        deliverable_type=CoreDeliverableType.RESEARCH_STATUS,
        purpose="梳理研究现状",
        organizing_strategy="按研究路线组织",
        sections=[
            WritingSection(
                id="status_overview", title="研究现状概述", purpose="概述",
                supporting_paper_ids=["p1"],
            ),
            WritingSection(
                id="theme_interaction", title="课堂互动编码", purpose="梳理路线",
                supporting_paper_ids=["p1"],
            ),
        ],
    )
    claim = "本文提出课堂互动行为编码框架"
    cards = [{
        "paper_id": "p1",
        "claims": [{"field": "method", "claim": claim}],
    }]
    synthesis = [{
        "theme_id": "interaction",
        "theme_name": "课堂互动编码",
        "reported_methods": [{"claim": claim, "paper_id": "p1"}],
    }]

    rendered = StatusRenderer().render_fallback(
        plan,
        {"canonical_topic": "课堂行为分析", "theme_synthesis": synthesis},
        cards,
    )

    assert "本文提出" not in rendered
    assert rendered.count("该研究提出课堂互动行为编码框架[p1]") == 1


def test_status_renderer_fallback_keeps_technical_terms_from_evidence():
    plan = WritingPlan(
        deliverable_type=CoreDeliverableType.RESEARCH_STATUS,
        purpose="梳理研究现状",
        organizing_strategy="按研究路线组织",
        sections=[WritingSection(
            id="theme_model",
            title="（一）视觉识别模型",
            purpose="梳理路线",
            supporting_paper_ids=["p1", "p2"],
        )],
    )
    synthesis = [{
        "theme_id": "model",
        "theme_name": "视觉识别模型",
        "reported_methods": [{"claim": "采用卷积神经网络提取时空特征", "paper_id": "p1"}],
        "reported_findings": [{"claim": "模型提升了识别准确率", "paper_id": "p2"}],
    }]

    rendered = StatusRenderer().render_fallback(
        plan,
        {"canonical_topic": "动作识别", "theme_synthesis": synthesis},
        [{"paper_id": "p1", "claims": []}, {"paper_id": "p2", "claims": []}],
    )

    assert "卷积神经网络" in rendered
    assert "提升了识别准确率[p2]" in rendered



def test_research_status_supports_six_subsections_with_valid_numbering():
    """研究现状子节预算放宽到 6，且编号表能覆盖全部子节。

    回归：编号表原为“一二三四”四字，子节预算放宽后第 5、6 节会
    IndexError；同时验证 spec 的 max_subsections 真正被写作计划采纳。
    """
    from app.deliverables.registry import get_deliverable_spec

    spec = get_deliverable_spec(CoreDeliverableType.RESEARCH_STATUS)
    assert spec.structure.max_subsections == 6

    syntheses = []
    cards = []
    # 每条路线 5 篇：子节预算随证据缩放，证据不足时会被合并成更少的小节
    for index in range(1, 7):
        paper_ids = [f"r{index}_{n}" for n in range(1, 6)]
        for pid in paper_ids:
            cards.append(_make_paper_card(pid, f"Route {index} Study {pid}", method=f"机制{index}"))
        syntheses.append({
            "theme_id": f"route_{index}",
            "theme_name": f"技术路线{index}",
            "paper_ids": paper_ids,
            "reported_problems": [{"claim": f"路线{index}的关键问题", "paper_id": paper_ids[0]}],
            "reported_methods": [{"claim": f"路线{index}采用的机制", "paper_id": paper_ids[1]}],
            "reported_findings": [{"claim": f"路线{index}的性能增益", "paper_id": paper_ids[2]}],
            "author_stated_limitations": [{"claim": f"路线{index}的局限", "paper_id": paper_ids[3]}],
        })

    state = {
        "topic": "课堂行为分析",
        "canonical_topic": "课堂行为分析",
        "paper_cards": cards,
        "theme_synthesis": syntheses,
    }

    plan = build_writing_plan(CoreDeliverableType.RESEARCH_STATUS, state)

    theme_sections = [s for s in plan.sections if s.id.startswith("theme_")]
    assert len(theme_sections) == 6
    # 编号连续覆盖到第六节，不出现越界或重复编号
    prefixes = [s.title[:3] for s in theme_sections]
    assert prefixes == ["（一）", "（二）", "（三）", "（四）", "（五）", "（六）"]
    # 字数上限随子节预算抬升，避免每次都触发字符数告警
    assert plan.target_char_range == (3500, 7800)


# 语义上互不相似的主题：避免相似度归并干扰对“子节预算”本身的验证
_DISTINCT_THEMES = [
    ("行为编码", "coding scheme 编码体系 时间窗"),
    ("视觉识别", "object detection 目标检测 骨干网络"),
    ("语音分析", "speech recognition 语音识别 声学特征"),
    ("互动序列", "lag sequential 滞后序列 互动分析"),
    ("部署效率", "lightweight 轻量化 推理延迟"),
    ("评测基准", "benchmark 基准 数据集划分"),
]


def _scaling_state(theme_count: int, papers_per_theme: int, **extra):
    cards, syntheses = [], []
    for index in range(theme_count):
        name, text = _DISTINCT_THEMES[index]
        paper_ids = [f"{name}_{n}" for n in range(1, papers_per_theme + 1)]
        for pid in paper_ids:
            cards.append(_make_paper_card(pid, f"{text} {pid}", method=text))
        syntheses.append({
            "theme_id": f"t{index}",
            "theme_name": name,
            "paper_ids": paper_ids,
            "reported_problems": [{"claim": f"{text}的问题", "paper_id": paper_ids[0]}],
            "reported_methods": [{"claim": f"{text}的机制", "paper_id": paper_ids[-1]}],
        })
    return {
        "topic": "课堂行为分析",
        "canonical_topic": "课堂行为分析",
        "paper_cards": cards,
        "theme_synthesis": syntheses,
        **extra,
    }


def _theme_section_count(state) -> int:
    plan = build_writing_plan(CoreDeliverableType.RESEARCH_STATUS, state)
    return len([s for s in plan.sections if s.id.startswith("theme_")])


def test_subsection_budget_scales_with_available_evidence():
    """子节数随可用证据自适应：证据不足不把 spec 预算用满，充足时用满。

    每节至少 _MIN_PAPERS_PER_SUBSECTION 篇才独立成节，否则合并——避免
    出现一批“两篇一节”的碎片小节。
    """
    assert _theme_section_count(_scaling_state(6, 1)) == 2    # 6 篇 -> 2 节
    assert _theme_section_count(_scaling_state(6, 2)) == 4    # 12 篇 -> 4 节
    assert _theme_section_count(_scaling_state(6, 3)) == 6    # 18 篇 -> 达到 spec 上限
    assert _theme_section_count(_scaling_state(6, 5)) == 6    # 证据再多不超上限


def test_explicit_required_focuses_do_not_create_single_paper_sections():
    """显式研究重点抬高子节预算，但不得因此产出单篇论文的正式小节。

    回归 2026-08-30 实测缺陷：命中 required_focuses 的单篇路线被豁免合并，
    渲染层只能写成"本节纳入 1 篇文献"的书目罗列。重点覆盖改由合并后的
    路线名与章节写作目标承载，证据仍在同一节内，覆盖判定不受影响。
    """
    focuses = ["行为编码", "视觉识别", "语音分析", "互动序列", "部署效率"]
    state = _scaling_state(
        5, 1, research_semantic_frame={"required_focuses": focuses},
    )
    plan = build_writing_plan(CoreDeliverableType.RESEARCH_STATUS, state)
    theme_sections = [s for s in plan.sections if s.id.startswith("theme_")]

    assert theme_sections
    for section in theme_sections:
        assert len(section.supporting_paper_ids) >= 2, section.title
    # 用户点名的重点全部仍被覆盖：不因合并而丢失，也不谎报已覆盖
    assert plan.required_focuses == focuses
    assert plan.undercovered_focuses == []
    plan_text = " ".join(
        f"{section.title} {section.purpose}" for section in theme_sections
    )
    assert all(focus in plan_text for focus in focuses)


def test_multiple_dominant_required_routes_fall_back_to_syntheses():
    """多条全局锚点要求不得替换证据验证路线（泛化 monolithic 守卫）。

    回归 2026-08-23 会话：语义帧产出 research_object + method 两条要求，
    各命中 89/90 篇。原守卫只防“恰好 1 条”全局锚点，两条漏过，写作计划
    用两个 catch-all 小节（动作识别 / 动作识别方法）替换掉全部 7 条
    互斥的证据路线，正文退化为两节同名复述。
    """
    syntheses = []
    cards = []
    themes = [
        ("度量原型匹配", "metric prototype matching"),
        ("时序对齐建模", "temporal alignment"),
        ("图关系建模", "graph relation modeling"),
    ]
    for index, (name, text) in enumerate(themes, 1):
        paper_ids = [f"t{index}_{n}" for n in range(1, 6)]
        for pid in paper_ids:
            # 标题含要求别名 action recognition，使两条全局要求命中全池
            cards.append(_make_paper_card(
                pid, f"few-shot action recognition via {text} {pid}", method=text,
            ))
        syntheses.append({
            "theme_id": f"theme_{index}",
            "theme_name": name,
            "paper_ids": paper_ids,
            "reported_problems": [{"claim": f"{name}的问题", "paper_id": paper_ids[0]}],
            "reported_methods": [{"claim": f"{name}的机制", "paper_id": paper_ids[-1]}],
        })
    all_ids = [c["paper_id"] for c in cards]
    frame = {
        "canonical_topic": "少样本动作识别",
        "evidence_requirements": [
            {"requirement_id": "research_object:action", "label": "动作识别相关证据",
             "evidence_role": "主要研究目标", "aliases": ["动作识别", "action recognition"],
             "source_ids": ["action"], "route_required": True, "route_group": "research_object"},
            {"requirement_id": "method:action_recognition", "label": "动作识别方法证据",
             "evidence_role": "主要研究目标", "aliases": ["动作识别", "action recognition"],
             "source_ids": ["action_recognition"], "route_required": True, "route_group": "method"},
        ],
    }
    # 覆盖检查用真实逻辑：两条要求都命中全部论文
    from app.agent.evidence_roles import evidence_coverage
    coverage = evidence_coverage(
        frame,
        [{**c, "doi": None, "arxiv_id": None} for c in cards],
    )
    matched = coverage["matched_paper_ids"].get("research_object:action") or []
    # 要求实际命中不足时不构成全局锚点场景，测试前提不成立则跳过
    assert len(matched) >= len(all_ids) * 0.9, "测试前提：要求需覆盖全池"

    plan = build_writing_plan(CoreDeliverableType.RESEARCH_STATUS, {
        "topic": "少样本动作识别",
        "canonical_topic": "少样本动作识别",
        "paper_cards": cards,
        "theme_synthesis": syntheses,
        "research_semantic_frame": frame,
    })

    titles = [s.title for s in plan.sections if s.id.startswith("theme_")]
    # 证据验证路线主导小节，而不是两条全局锚点各吃全池
    assert len(titles) >= 3
    assert all("动作识别" not in title for title in titles)
    covered = {p for s in plan.sections for p in s.supporting_paper_ids}
    assert set(all_ids) <= covered, "路线小节合并后仍须覆盖全部证据"


def test_absorb_focus_label_caps_heading_at_three_segments():
    """合并重点名不得把小节标题拼成四段链式标题。

    回归 2026-09-01 实测：「（一）编码应用与实证研究与技术辅助编码与多模态
    融合」。旧护栏只数字数（≤30）并用 protected_focus_labels 记"吸收过几次"，
    但 _semantic_route_name 自身就会产出两段名且不计入吸收次数，21 字标题顺利
    放行。判据改为直接数合并后的「与」分段数：超过三段只记标签不改名，标签由
    _theme_section 写进章节写作目标，覆盖判定与证据都不丢。
    """
    overflow = _absorb_focus_label(
        {"theme_name": "编码应用与实证研究", "paper_ids": ["p1", "p2"]},
        "技术辅助编码与多模态融合",
    )
    assert overflow["theme_name"] == "编码应用与实证研究"
    assert "技术辅助编码与多模态融合" in overflow["protected_focus_labels"]

    within_limit = _absorb_focus_label(
        {"theme_name": "课堂行为视觉识别", "paper_ids": ["p1", "p2"]},
        "自动行为编码与结构化观察",
    )
    assert within_limit["theme_name"] == "课堂行为视觉识别与自动行为编码与结构化观察"
    assert within_limit["theme_name"].count("与") == 2


def test_overflow_routes_merge_instead_of_dropping_their_evidence():
    """小节名额溢出时并入存活路线，不得连同 paper_ids 一起丢弃。

    回归 2026-09-01 实测：5 条 KEEP 路线在 max_routes=3 下被
    `merged[:max_routes]` 裸切片丢掉 2 条，45 篇证据卡片最终只被引用 25 篇。
    """
    state = _scaling_state(5, 2)   # 5 条路线 × 2 篇 = 10 篇 -> scaled 名额 3
    plan = build_writing_plan(CoreDeliverableType.RESEARCH_STATUS, state)
    theme_sections = [s for s in plan.sections if s.id.startswith("theme_")]

    assert len(theme_sections) == 3
    expected = {
        f"{name}_{n}" for name, _ in _DISTINCT_THEMES[:5] for n in (1, 2)
    }
    covered = {p for s in plan.sections for p in s.supporting_paper_ids}
    assert expected <= covered, "溢出路线的证据必须并入存活小节"


def test_route_merge_diagnostics_records_every_absorption():
    """路线并入必须可观测：记录被并路线、目标路线与迁移论文数。

    WHY: 此前步骤输出只有 len(plans)，"证据被静默丢弃"只能靠 grep INFO 日志
    定位。
    """
    state = _scaling_state(5, 2)
    build_writing_plan(CoreDeliverableType.RESEARCH_STATUS, state)
    diagnostics = state.get("route_merge_diagnostics") or []

    assert len(diagnostics) == 2, diagnostics
    for record in diagnostics:
        assert record["reason"]
        assert record["merged_route"]
        assert record["target_route"]
        assert record["migrated_paper_count"] == 2
    merged_names = {record["merged_route"] for record in diagnostics}
    assert len(merged_names) == 2
