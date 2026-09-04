"""退化章节与路线兜底命名的可读性约束。"""

import pytest

from app.agent.route_validator import (
    _distinctive_cluster_labels,
    _is_boundary_aligned_gram,
)
from app.core.text_quality import AGENT_PROCESS_LANGUAGE_RE
from app.deliverables.renderers.base_renderer import _conservative_evidence_section


class _Section:
    def __init__(self, title="（五）多模态数据采集与处理", ids=None):
        self.title = title
        self.heading_level = 3
        self.supporting_paper_ids = ids or []


def _card(pid, title, problem="", method="", year=2025, venue="某期刊"):
    return {
        "paper_id": pid,
        "title": title,
        "research_problem": problem,
        "method": method,
        "year": year,
        "venue": venue,
    }


def test_degraded_section_states_problem_and_method_not_bare_titles():
    cards = [
        _card(
            "p1", "Real-Time Multimodal Student Behavior Analysis",
            problem="智能教室中学生行为难以实时感知。补充说明不应出现。",
            method="多模态流式融合与姿态估计",
        ),
        _card(
            "p2", "Leveraging Educational Technologies for Classroom Behavior Analysis",
            problem="现有课堂行为分析方法缺乏系统梳理",
            method="系统综述与方法学批判分析",
        ),
    ]
    text = _conservative_evidence_section(_Section(ids=["p1", "p2"]), cards)

    # 不再是"《标题》（年份）"式书目清单：问题设定与方法必须出现在正文里。
    assert "智能教室中学生行为难以实时感知" in text
    assert "多模态流式融合与姿态估计" in text
    assert "系统综述与方法学批判分析" in text
    # 只取首句，句号后的内容不应被带入。
    assert "补充说明不应出现" not in text
    assert "[p1]" in text and "[p2]" in text
    assert "本节纳入 2 篇文献" in text


def test_degraded_section_avoids_agent_process_language():
    cards = [_card("p1", "某研究", problem="问题", method="方法")]
    text = _conservative_evidence_section(_Section(ids=["p1"]), cards)
    assert not AGENT_PROCESS_LANGUAGE_RE.search(text)


def test_degraded_section_falls_back_to_titles_without_card_fields():
    """卡片没有问题与方法时仍要产出条目，不能整节丢空。"""
    cards = [_card("p1", "仅有标题的论文")]
    text = _conservative_evidence_section(_Section(ids=["p1"]), cards)
    assert "《仅有标题的论文》" in text
    assert "[p1]" in text


def test_degraded_section_reports_empty_allocation():
    text = _conservative_evidence_section(_Section(ids=["missing"]), [])
    assert "没有分配给本节的论文" in text


def test_boundary_check_rejects_cjk_sliding_window_fragment():
    texts = ["应用深度学习的学生课堂行为分析系统"]
    # "学生课堂"是滑窗片段：既非任何连续中文串的开头也非结尾。
    assert _is_boundary_aligned_gram("学生课堂", texts) is False
    # 词组开头与结尾的候选词判为合格。
    assert _is_boundary_aligned_gram("多模态", ["多模态融合的教学评估"]) is True
    assert _is_boundary_aligned_gram("姿态估计", ["骨骼点提取与姿态估计"]) is True
    # 英文词元由正则按词边界切出，始终合格。
    assert _is_boundary_aligned_gram("openpose", ["OpenPose 骨骼点"]) is True


def test_fallback_route_labels_are_not_truncated_fragments():
    """兜底名不得是"学生课堂"这类截断词组（实测缺陷）。"""
    cluster_a = ["a1", "a2", "a3", "a4"]
    cluster_b = ["b1", "b2", "b3", "b4"]
    card_map = {}
    for pid, title in zip(cluster_a, [
        "应用深度学习的学生课堂行为分析系统",
        "基于深度学习的学生课堂行为分析系统设计",
        "一种基于实时网络的学生课堂行为分析方法",
        "学生课堂行为分析中的姿态估计",
    ]):
        card_map[pid] = {"title": title, "research_problem": "", "method": "骨骼点提取与姿态估计"}
    for pid, title in zip(cluster_b, [
        "多模态融合的课堂教学质量评估",
        "跨注意力网络的课堂数据分析",
        "面向智能教室的多模态实时分析",
        "多模态数据驱动的教学评估",
    ]):
        card_map[pid] = {"title": title, "research_problem": "", "method": "跨注意力特征融合"}

    labels = _distinctive_cluster_labels(
        [cluster_a, cluster_b], card_map,
        parent_name="行为分析框架构建", topic="课堂行为分析",
    )

    assert len(labels) == 2
    for label in labels:
        suffix = label.split("：")[-1] if "：" in label else label
        assert suffix not in {"学生课堂", "课堂行为", "堂行为分", "生课堂行"}
    assert labels[0] != labels[1]


def test_split_abandoned_when_no_distinctive_candidate():
    """无合格候选时放弃拆分，不得编造"（子路线N）"这类内部编号标题。

    子路线名会直接成为正文小节标题；编号名对读者没有信息量，还暴露内部
    拆分机制。返回空列表让调用方保留父路线。
    """
    cluster_a = ["a1", "a2"]
    cluster_b = ["b1", "b2"]
    card_map = {
        pid: {"title": "课堂行为分析", "research_problem": "", "method": ""}
        for pid in [*cluster_a, *cluster_b]
    }

    labels = _distinctive_cluster_labels(
        [cluster_a, cluster_b], card_map,
        parent_name="行为分析", topic="课堂行为分析",
    )

    assert labels == []


def test_route_split_is_skipped_when_names_unavailable():
    """端到端：命名失败时父路线保持原样，正文标题里不出现编号名。"""
    from app.agent.route_validator import _split_oversized_routes

    big_ids = [f"p{i}" for i in range(12)]
    routes = [
        {
            "route_id": "R1",
            "name": "行为分析",
            "core_paper_ids": list(big_ids),
            "supporting_paper_ids": [],
            "paper_ids": list(big_ids),
            "status": "KEEP",
        },
        {
            "route_id": "R2",
            "name": "参与度评估",
            "core_paper_ids": ["q1", "q2"],
            "supporting_paper_ids": [],
            "paper_ids": ["q1", "q2"],
            "status": "KEEP",
        },
    ]
    # 所有卡片文本相同：切不出任何区分词项。
    card_map = {
        pid: {"title": "课堂行为分析", "research_problem": "", "method": ""}
        for pid in [*big_ids, "q1", "q2"]
    }
    primary_owner = {pid: "R1" for pid in big_ids}
    primary_owner.update({"q1": "R2", "q2": "R2"})

    _split_oversized_routes(
        routes, [], card_map, primary_owner, llm=None, topic="课堂行为分析",
    )

    names = [route["name"] for route in routes]
    assert "行为分析" in names
    assert not any("子路线" in name for name in names)


# ============================================================
# 跨章节免责套话去重
# ============================================================
from app.deliverables.renderers.base_renderer import (  # noqa: E402
    _deduplicate_boilerplate_clauses,
    _has_body_text,
)


def _sec(sid: str, body: str, title: str = "路线") -> tuple[str, str]:
    return sid, f"### （{sid}）{title}\n\n{body}"


def test_repeated_disclaimer_kept_once_across_sections():
    """蓝本免责句在多节重复时只保留首次出现。"""
    sections = [
        _sec("一", "方法上存在差异，不宜仅凭单项指标作统一排序。"),
        _sec("二", "两条路径各有侧重，不宜仅凭单项指标作统一排序。"),
        _sec("三", "其取舍取决于数据条件，不宜仅凭单项指标作统一排序。"),
    ]

    result = _deduplicate_boilerplate_clauses(sections)
    bodies = [text for _sid, text in result]

    assert sum("统一排序" in body for body in bodies) == 1
    assert "统一排序" in bodies[0]


def test_two_distinct_disclaimers_each_kept_once():
    """不同套话各自独立计数，互不影响。"""
    sections = [
        _sec("一", "存在差异，不宜仅凭单项指标作统一排序。"),
        _sec("二", "两种取向并不构成简单的优劣关系。"),
        _sec("三", "各有侧重，不宜仅凭单项指标作统一排序，并不构成简单的优劣关系。"),
    ]

    result = _deduplicate_boilerplate_clauses(sections)
    bodies = [text for _sid, text in result]

    assert sum("统一排序" in body for body in bodies) == 1
    assert sum("优劣" in body for body in bodies) == 1
    # 第三节两句都是重复，应被清空为纯正文并补回句末标点。
    assert bodies[2].rstrip().endswith("。")
    assert "统一排序" not in bodies[2]
    assert "优劣" not in bodies[2]


def test_removal_repairs_dangling_punctuation():
    """删除句尾子句后不得留下悬空逗号或"，。"。"""
    sections = [
        _sec("一", "首次出现，不宜仅凭单项指标作统一排序。"),
        _sec("二", "两条路径各有侧重，不宜仅凭单项指标作统一排序。"),
    ]

    body = _deduplicate_boilerplate_clauses(sections)[1][1]

    assert "，。" not in body
    assert not body.rstrip().endswith("，")
    assert body.rstrip().endswith("。")


def test_body_consisting_only_of_disclaimer_is_preserved():
    """整节正文只有这句套话时保留原文，避免产出裸标题。"""
    sections = [
        _sec("一", "首次出现，不宜仅凭单项指标作统一排序。"),
        _sec("二", "不宜仅凭单项指标作统一排序。"),
    ]

    result = _deduplicate_boilerplate_clauses(sections)

    assert "统一排序" in result[1][1]
    assert _has_body_text(result[1][1])


def test_sections_without_boilerplate_are_untouched():
    original = [
        _sec("一", "本节陈述具体证据，不含任何免责句式。"),
        _sec("二", "另一节同样只有正文。"),
    ]

    result = _deduplicate_boilerplate_clauses(original)

    assert result == original


def test_has_body_text_detects_heading_only_section():
    assert _has_body_text("### （一）路线\n\n正文") is True
    assert _has_body_text("### （一）路线") is False
    assert _has_body_text("### （一）路线\n\n   ") is False


def test_phrase_candidates_reject_english_modality_labels():
    """英文模态标签不得成为中文子路线名（实测"…框架：audio"）。

    ``data_modalities`` 存的是 video/audio 这类英文短标签；它们是数据模态
    描述而非论点名，拼进中文标题就成了中英混杂的怪名。区分工作交由
    n-gram 层的英文词元，那里按词边界对齐且会与保留名查重。
    """
    from app.agent.route_validator import _phrase_candidates

    card = {
        "data_modalities": ["video", "audio"],
        "behavior_categories": ["举手", "阅读"],
        "metrics": [],
        "study_design": "课堂实验",
        "dataset": "UGC",
    }
    candidates = _phrase_candidates(card)
    assert candidates == ["举手", "阅读", "课堂实验"]


def test_route_labels_never_contain_english_modality_tag():
    """端到端：仅英文模态标签可区分的簇，兜底名仍必须是可用词元。"""
    card_map = {
        "a1": {
            "title": "Multimodal classroom behavior recognition with audio",
            "research_problem": "", "method": "骨骼点提取",
            "data_modalities": ["video", "audio"],
        },
        "a2": {
            "title": "Audiovisual fusion for classroom behavior",
            "research_problem": "", "method": "骨骼点提取",
            "data_modalities": ["audio"],
        },
        "b1": {
            "title": "多模态教学评估", "research_problem": "",
            "method": "跨注意力特征融合", "data_modalities": ["video"],
        },
        "b2": {
            "title": "跨模态教学分析", "research_problem": "",
            "method": "跨注意力特征融合", "data_modalities": ["video"],
        },
    }
    labels = _distinctive_cluster_labels(
        [["a1", "a2"], ["b1", "b2"]], card_map,
        parent_name="课堂行为编码与分析框架", topic="课堂行为分析",
    )

    assert len(labels) == 2
    for label in labels:
        suffix = label.split("：")[-1] if "：" in label else label
        assert suffix not in {"audio", "video", "text"}, label
