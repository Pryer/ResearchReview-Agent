from app.agent.evidence_roles import (
    evidence_coverage,
    is_scope_only_text,
    is_temporal_qualifier_text,
    temporal_requirement_window,
)
from app.agent.writing_plan import build_writing_plan
from app.schemas.deliverable_schema import CoreDeliverableType, WritingPlan, WritingSection
from app.tools.validate_deliverable import validate_deliverable
from app.utils.date_utils import current_year


def _frame():
    return {
        "canonical_topic": "课堂行为分析",
        "research_objects": [{
            "id": "classroom_behavior", "label": "classroom behavior",
            "surface_text": "教师和学生行为", "explicit": True,
        }],
        "task_chain": [
            "behavior_recognition", "structured_behavior_coding",
            "analytical_method", "interaction_interpretation",
        ],
        "required_focuses": [
            "教师与学生行为自动识别", "自动行为编码",
            "S-T分析法或滞后序列分析法", "教学结构与师生互动解释",
        ],
        "evidence_requirements": [
            {
                "requirement_id": "perception:behavior", "label": "行为自动识别",
                "evidence_role": "perception",
                "aliases": [
                    "teacher-student behavior", "student behavior", "classroom behavior",
                    "教师行为", "学生行为", "课堂行为",
                ],
                "source_ids": ["behavior_recognition"],
            },
            {
                "requirement_id": "structured_coding:behavior", "label": "自动行为编码",
                "evidence_role": "structured_coding",
                "aliases": ["automatic behavior coding", "自动行为编码"],
                "source_ids": ["structured_behavior_coding"],
            },
            {
                "requirement_id": "analytical_method:sequence", "label": "S-T分析法或滞后序列分析法",
                "evidence_role": "analytical_method",
                "aliases": ["S-T analysis", "S-T分析法", "lag sequential analysis", "滞后序列分析法"],
                "context_aliases": ["classroom behavior", "课堂行为"],
                "source_ids": ["st_analysis", "lag_sequential_analysis"],
                "selection_mode": "any",
            },
            {
                "requirement_id": "interpretation:interaction", "label": "教学结构与互动解释",
                "evidence_role": "interpretation",
                "aliases": ["teaching structure", "teacher-student interaction", "教学结构", "师生互动"],
                "source_ids": ["interaction_interpretation"],
            },
        ],
    }


def _card(paper_id: str, title: str, abstract: str = "") -> dict:
    return {
        "paper_id": paper_id,
        "title": title,
        "abstract": abstract,
        "quality_status": "partial",
        "evidence_source": "abstract",
        "evidence_state": {"access_level": "abstract"},
        "field_claims": {},
    }


def test_recognition_output_does_not_count_as_structured_behavior_coding():
    frame = _frame()
    papers = [
        _card("recognition", "Action recognition and classification of teacher-student behavior"),
        _card(
            "coding",
            "Automatic behavior coding for classroom observation",
            "A coding scheme defines time windows and evaluates inter-rater agreement.",
        ),
    ]
    coverage = evidence_coverage(frame, papers)
    coding_id = next(
        item["requirement_id"] for item in frame["evidence_requirements"]
        if item["evidence_role"] == "structured_coding"
    )
    assert coverage["matched_paper_ids"][coding_id] == ["coding"]
    assert "recognition" not in coverage["matched_paper_ids"][coding_id]


def test_perception_stage_uses_explicit_research_object_as_dynamic_alias():
    frame = _frame()
    paper = _card(
        "detection",
        "基于深度学习的学生课堂行为检测方法",
        "该方法对学生行为进行检测与分类。",
    )
    coverage = evidence_coverage(frame, [paper])
    perception_id = next(
        item["requirement_id"] for item in frame["evidence_requirements"]
        if item["evidence_role"] == "perception"
    )
    assert coverage["matched_paper_ids"][perception_id] == ["detection"]


def test_named_analysis_methods_require_direct_method_evidence():
    frame = _frame()
    papers = [
        _card("generic", "Temporal analysis of classroom interaction sequences"),
        _card("st", "S-T analysis of teaching structure"),
        _card("lag", "Lag sequential analysis of teacher-student interaction"),
    ]
    coverage = evidence_coverage(frame, papers)
    analytical = [
        item for item in frame["evidence_requirements"]
        if item["evidence_role"] == "analytical_method"
    ]
    assert len(analytical) == 1
    matched = coverage["matched_paper_ids"][analytical[0]["requirement_id"]]
    assert "generic" not in matched
    assert {"st", "lag"} <= set(matched)


def test_open_analysis_alternatives_accept_another_explicit_contextual_method():
    frame = _frame()
    analytical_requirement = next(
        item for item in frame["evidence_requirements"]
        if item["evidence_role"] == "analytical_method"
    )
    analytical_requirement["selection_mode"] = "open_any"
    papers = [
        _card(
            "other",
            "课堂行为的互动网络研究",
            "本研究采用社会网络分析方法分析课堂行为及其互动关系。",
        ),
        _card(
            "unrelated",
            "企业组织结构研究",
            "本研究采用社会网络分析方法分析企业合作关系。",
        ),
    ]
    coverage = evidence_coverage(frame, papers)
    analytical = next(
        item for item in frame["evidence_requirements"]
        if item["evidence_role"] == "analytical_method"
    )
    matched = coverage["matched_paper_ids"][analytical["requirement_id"]]

    assert "other" in matched
    assert "unrelated" not in matched


def test_required_routes_are_derived_from_stage_labels_and_weak_paper_is_unassigned():
    frame = _frame()
    cards = [
        _card("r1", "Action recognition for teacher and student behavior"),
        _card("c1", "Automatic behavior coding", "A coding scheme uses time windows and agreement."),
        _card("c2", "Automated behavior coding", "An annotation protocol defines event boundaries and reliability."),
        _card("s1", "S-T analysis of teaching structure"),
        _card("s2", "S-T analysis and teacher-student behavior ratio"),
        _card("l1", "Lag sequential analysis of interaction sequences"),
        _card("l2", "Lag sequential analysis of behavior transitions"),
        _card("i1", "Teaching structure and teacher-student interaction interpretation"),
        _card("weak", "University teacher performance evaluation and academic productivity"),
    ]
    syntheses = [
        {"theme_id": "a", "theme_name": "视觉方法", "paper_ids": ["r1", "weak"]},
        {"theme_id": "b", "theme_name": "编码方法", "paper_ids": ["c1", "c2"]},
        {"theme_id": "c", "theme_name": "序列方法", "paper_ids": ["s1", "s2", "l1", "l2"]},
        {"theme_id": "d", "theme_name": "解释方法", "paper_ids": ["i1"]},
    ]
    plan = build_writing_plan("research_status", {
        "topic": "课堂行为分析", "canonical_topic": "课堂行为分析",
        "paper_cards": cards, "theme_synthesis": syntheses,
        "research_semantic_frame": frame,
    })
    titles = " ".join(section.title for section in plan.sections)
    assert "S-T" in titles and "滞后序列" in titles
    assigned = {
        paper_id for section in plan.sections for paper_id in section.supporting_paper_ids
    }
    assert "weak" not in assigned


def test_validator_rejects_only_runtime_task_and_plan_identifiers():
    plan = WritingPlan(
        deliverable_type=CoreDeliverableType.RESEARCH_STATUS,
        purpose="test",
        organizing_strategy="evidence_driven",
        sections=[WritingSection(
            id="theme_dynamic_route", title="研究现状", purpose="test",
            supporting_paper_ids=["p1"], heading_level=2,
        )],
        citation_policy={"minimum_unique_references": 1},
    )
    state = {
        "canonical_topic": "课堂行为分析",
        "paper_cards": [{"paper_id": "p1"}],
        "research_semantic_frame": {
            "task_chain": ["produce_structured_observations"],
            "methods": [{"id": "object_detection"}],
        },
    }
    result = validate_deliverable(
        "## 研究现状\n\n课堂行为分析通过produce_structured_observations形成结果[p1]。",
        plan,
        state,
    )
    assert any("内部任务链或规划节点标识" in error for error in result["errors"])


def test_stage_routes_are_domain_independent():
    frame = _frame()
    frame["canonical_topic"] = "康复训练行为分析"
    frame["research_objects"][0].update({
        "id": "rehabilitation_training", "label": "rehabilitation training",
        "surface_text": "康复训练",
    })
    frame["required_focuses"] = [
        "康复训练动作识别", "康复训练行为编码", "滞后序列分析法", "训练质量分析",
    ]
    for requirement in frame["evidence_requirements"]:
        if requirement["evidence_role"] == "perception":
            requirement["aliases"] = ["rehabilitation training", "training action"]
            requirement["label"] = "康复训练动作识别"
        elif requirement["evidence_role"] == "structured_coding":
            requirement["aliases"] = ["automatic behavior coding", "automated behavior coding"]
        elif requirement["evidence_role"] == "analytical_method":
            requirement["aliases"] = ["lag sequential analysis", "滞后序列分析法"]
            requirement["label"] = "滞后序列分析法"
        elif requirement["evidence_role"] == "interpretation":
            requirement["aliases"] = ["training quality", "训练质量"]
            requirement["label"] = "训练质量分析"
    cards = [
        _card("r", "Action recognition for rehabilitation training"),
        _card("c1", "Automatic behavior coding in rehabilitation", "A coding scheme defines time windows."),
        _card("c2", "Automated behavior coding for training", "An annotation protocol reports reliability."),
        _card("l1", "Lag sequential analysis of rehabilitation actions"),
        _card("l2", "Lag sequential analysis of training transitions"),
        _card("t", "Training quality analysis from rehabilitation action sequences"),
    ]
    syntheses = [
        {"theme_id": "x1", "theme_name": "动作感知", "paper_ids": ["r"]},
        {"theme_id": "x2", "theme_name": "结构化数据", "paper_ids": ["c1", "c2"]},
        {"theme_id": "x3", "theme_name": "序列方法", "paper_ids": ["l1", "l2"]},
        {"theme_id": "x4", "theme_name": "质量解释", "paper_ids": ["t"]},
    ]
    plan = build_writing_plan("research_status", {
        "topic": "康复训练行为分析", "canonical_topic": "康复训练行为分析",
        "paper_cards": cards, "theme_synthesis": syntheses,
        "research_semantic_frame": frame,
    })
    titles = " ".join(section.title for section in plan.sections)
    assert "课堂" not in titles
    assert "康复" in titles or "训练质量" in titles
    assert "滞后序列分析法" in titles


def test_is_temporal_qualifier_text_distinguishes_window_from_topic():
    # 真实会话 label 含动词“调研”（“调研近五年…”），验证剥除后判为时间词
    assert is_temporal_qualifier_text("近五年文献证据") is True
    assert is_temporal_qualifier_text("近五年文献调研证据") is True
    assert is_temporal_qualifier_text("recent five years literature") is True
    assert is_temporal_qualifier_text("近年来研究进展") is True
    # 含领域内容的表述不是纯时间限定词
    assert is_temporal_qualifier_text("近五年少样本动作识别") is False
    assert is_temporal_qualifier_text("近五年少样本动作识别论文") is False
    # 不含时间词的表述恒为 False，即使全是通用学术词
    assert is_temporal_qualifier_text("研究综述") is False
    assert is_temporal_qualifier_text("") is False


def test_is_scope_only_text_judges_by_entity_mapping_not_word_enumeration():
    domain = ["少样本动作识别", "动作识别"]

    # 时间窗 + 任何非领域残差都是检索范围：判据是残差能否命中领域实体，
    # 与用什么动词无关——“梳理/汇总”从未出现在任何词表里同样判 True
    assert is_scope_only_text("近五年文献调研证据", domain) is True
    assert is_scope_only_text("近五年文献梳理证据", domain) is True
    assert is_scope_only_text("近五年文献汇总证据", domain) is True
    assert is_scope_only_text("近五年", domain) is True
    assert is_scope_only_text("近五年文献调研证据", []) is True

    # 残差命中领域实体：属于领域内容，保留
    assert is_scope_only_text("近五年少样本动作识别证据", domain) is False
    assert is_scope_only_text("近五年动作识别相关证据", domain) is False

    # 无时间词或空文本不归本判据管辖（交由 source_ids 溯源校验）
    assert is_scope_only_text("少样本学习相关证据", domain) is False
    assert is_scope_only_text("", domain) is False


def test_temporal_requirement_window_parses_user_window():
    window = temporal_requirement_window({
        "label": "近五年文献证据",
        "aliases": ["近五年", "recent five years"],
    })
    assert window == (current_year() - 5 + 1, current_year())

    bare = temporal_requirement_window({"label": "近年文献", "aliases": []})
    assert bare == (current_year() - 5 + 1, current_year())


def test_temporal_requirement_is_judged_by_publication_year_not_alias():
    frame = {"evidence_requirements": [{
        "requirement_id": "time:recent",
        "label": "近五年文献证据",
        "evidence_role": "recency",
        "aliases": ["近五年", "recent five years"],
        "minimum_direct_sources": 1,
    }]}
    papers = [
        {"paper_id": "in_window", "title": "few-shot action recognition", "year": 2024},
        {"paper_id": "too_old", "title": "few-shot action recognition", "year": 2018},
        {"paper_id": "no_year", "title": "few-shot action recognition"},
    ]

    coverage = evidence_coverage(frame, papers)

    # 词面匹配永远找不到“近五年”，年份窗判定才能正确放行窗内论文
    assert coverage["counts"]["time:recent"] == 1
    assert coverage["matched_paper_ids"]["time:recent"] == ["in_window"]
    assert coverage["ready"] is True
    assert coverage["missing_focuses"] == []


def test_temporal_requirement_reports_missing_when_window_uncovered():
    frame = {"evidence_requirements": [{
        "requirement_id": "time:recent",
        "label": "近五年文献证据",
        "evidence_role": "recency",
        "aliases": ["近五年"],
        "minimum_direct_sources": 1,
    }]}
    papers = [{"paper_id": "old", "title": "legacy method", "year": 2015}]

    coverage = evidence_coverage(frame, papers)

    assert coverage["ready"] is False
    assert coverage["missing_focuses"] == ["近五年文献证据"]


def _paper(paper_id: str, text: str, year: int = 2025) -> dict:
    return {
        "paper_id": paper_id,
        "title": text,
        "abstract": text,
        "research_problem": text,
        "year": year,
    }


def test_compound_alias_falls_back_to_corpus_derived_segments():
    """复合别名全语料零命中时，按语料切出的构件做合取匹配。

    回归 2026-08-29 实测缺陷：用户澄清"偏向于教学互动分析"，语义解析给出
    别名「教学互动分析」，但论文写的是"教学互动行为""师生互动"，完整别名
    59 篇里命中 0 次，覆盖门禁误报该重点缺失。切分完全由当前语料统计驱动，
    不含任何领域词表。
    """
    frame = {"evidence_requirements": [{
        "requirement_id": "method:interaction",
        "label": "教学互动分析方法相关文献",
        "evidence_role": "分析方法",
        "aliases": ["教学互动分析"],
        "minimum_direct_sources": 1,
    }]}
    papers = [
        _paper("p1", "智慧课堂环境下小学信息科技课堂教学互动行为双编码分析"),
        _paper("p2", "基于YOLOv8的学生课堂行为检测方法"),
        _paper("p3", "深度网络支撑下的课堂观察行为研究"),
        _paper("p4", "多模态融合的学生姿态估计"),
        _paper("p5", "课堂场景小目标检测网络优化"),
    ]

    coverage = evidence_coverage(frame, papers)

    assert coverage["counts"]["method:interaction"] == 1
    assert coverage["matched_paper_ids"]["method:interaction"] == ["p1"]
    assert coverage["ready"] is True
    assert "构件" in coverage["match_reasons"]["method:interaction"]["p1"]


def test_compound_alias_fallback_requires_all_segments_not_any():
    """构件是合取而非析取：只含其中一个构件的论文不得算命中。"""
    frame = {"evidence_requirements": [{
        "requirement_id": "method:interaction",
        "label": "教学互动分析方法相关文献",
        "evidence_role": "分析方法",
        "aliases": ["教学互动分析"],
        "minimum_direct_sources": 1,
    }]}
    # 语料里"教学互动"和"分析"都出现过，但没有任何一篇同时含有两者
    papers = [
        _paper("only_interaction", "课堂教学互动行为的结构描述"),
        _paper("only_analysis", "学生姿态估计与检测结果分析"),
        _paper("neither", "小目标检测网络的实时性优化"),
    ]

    coverage = evidence_coverage(frame, papers)

    assert coverage["counts"]["method:interaction"] == 0
    assert coverage["missing_focuses"] == ["教学互动分析方法相关文献"]


def test_compound_alias_fallback_abandoned_when_segments_are_too_generic():
    """构件覆盖率过半说明合取已无区分度：放弃回退，如实报告缺口。"""
    frame = {"evidence_requirements": [{
        "requirement_id": "method:interaction",
        "label": "教学互动分析方法相关文献",
        "evidence_role": "分析方法",
        "aliases": ["教学互动分析"],
        "minimum_direct_sources": 1,
    }]}
    # 每篇都同时含"教学互动"和"分析"，构件失去区分度
    papers = [
        _paper(f"p{index}", "课堂教学互动行为分析与教学改进")
        for index in range(4)
    ]

    coverage = evidence_coverage(frame, papers)

    assert coverage["counts"]["method:interaction"] == 0
    assert coverage["missing_focuses"] == ["教学互动分析方法相关文献"]


def test_temporal_requirement_is_not_rescued_by_segment_fallback():
    """时间窗要求按年份判定，不得被构件回退绕过。"""
    frame = {"evidence_requirements": [{
        "requirement_id": "time:recent",
        "label": "近五年文献证据",
        "evidence_role": "recency",
        "aliases": ["近五年"],
        "minimum_direct_sources": 1,
    }]}
    papers = [_paper("old", "近年来的课堂研究综述", year=2015)]

    coverage = evidence_coverage(frame, papers)

    assert coverage["counts"]["time:recent"] == 0
    assert coverage["ready"] is False


def test_full_alias_hit_does_not_trigger_segment_fallback():
    """完整别名有命中时不启用回退，匹配集合保持原样。

    回退只在"完整别名全语料零命中"时生效，绝不放宽本来有效的要求。
    """
    frame = {"evidence_requirements": [{
        "requirement_id": "method:interaction",
        "label": "教学互动分析方法相关文献",
        "evidence_role": "分析方法",
        "aliases": ["教学互动分析"],
        "minimum_direct_sources": 1,
    }]}
    papers = [
        _paper("exact", "面向教学互动分析的课堂观察框架"),
        # 只含构件、不含完整别名：完整别名已有命中，这篇不得被回退带进来
        _paper("segments_only", "课堂教学互动行为的结构分析"),
    ]

    coverage = evidence_coverage(frame, papers)

    assert coverage["matched_paper_ids"]["method:interaction"] == ["exact"]
