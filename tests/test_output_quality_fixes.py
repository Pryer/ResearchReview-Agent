# -*- coding: utf-8 -*-
"""生成质量修复的行为测试：关键词清洗、主题锚点、引用堆砌、章节不删空、澄清解析。"""

from __future__ import annotations

from app.core.citation_density import break_citation_dumps, detect_citation_dumps
from app.core.source_capabilities import sanitize_search_keyword
from app.tools.generate_citation import render_in_text_citations
from app.tools.rank_papers import evaluate_topic_anchor_filter, rank_papers


# ---------- F1：中英混杂关键词清洗 ----------

def test_mixed_keyword_extracts_chinese_segment_and_strips_time_prefix():
    assert sanitize_search_keyword("survey 近三年少样本动作识别研究综述") == "少样本动作识别研究综述"


def test_mixed_keyword_with_short_chinese_segment_is_dropped():
    assert sanitize_search_keyword("FSAR 综述") is None


def test_pure_chinese_or_english_keyword_passes_through():
    assert sanitize_search_keyword("少样本动作识别") == "少样本动作识别"
    assert sanitize_search_keyword("few-shot action recognition") == "few-shot action recognition"
    assert sanitize_search_keyword("") is None


# ---------- F4：主题锚点硬下限 ----------

def test_topic_anchor_filter_kills_zero_dimension_paper():
    paper = {
        "title": "基于卷积神经网络的土体含水率智能识别",
        "abstract": "采集四种土壤表面照片构建样本库，采用卷积神经网络识别含水率。",
    }
    concepts = [["few-shot", "少样本"], ["action recognition", "动作识别"]]
    passed, reason = evaluate_topic_anchor_filter(
        paper, "少样本动作识别", required_concepts=concepts,
    )
    assert passed is False
    assert "未命中任何主题概念维度" in reason


def test_topic_anchor_filter_keeps_partial_dimension_hits():
    paper = {
        "title": "Zero-Shot Temporal Action Localization",
        "abstract": "A new formulation for temporal action recognition without exemplars.",
    }
    concepts = [["few-shot", "少样本"], ["action recognition", "动作识别"]]
    passed, _ = evaluate_topic_anchor_filter(
        paper, "少样本动作识别", required_concepts=concepts,
    )
    assert passed is True


def test_topic_anchor_filter_falls_back_to_topic_synonyms():
    paper = {
        "title": "Few-Shot Action Recognition via Temporal Alignment",
        "abstract": "",
    }
    passed, _ = evaluate_topic_anchor_filter(
        paper, "少样本动作识别", topic_synonyms=["few-shot action recognition"],
    )
    assert passed is True


def test_topic_anchor_filter_passes_paper_without_any_text():
    passed, reason = evaluate_topic_anchor_filter(
        {"title": "", "abstract": ""}, "少样本动作识别",
    )
    assert passed is True
    assert "语义筛选" in reason


def test_rank_papers_excludes_off_topic_paper():
    topic = "少样本动作识别"
    keywords = ["少样本动作识别", "few-shot action recognition"]
    papers = [
        {"paper_id": "p_relevant", "title": "Few-Shot Action Recognition via Temporal Alignment",
         "abstract": "few-shot video action recognition", "year": 2024},
        {"paper_id": "p_off_topic", "title": "基于卷积神经网络的土体含水率智能识别",
         "abstract": "土壤含水量图像识别模型", "year": 2024},
    ]
    ranked = rank_papers(papers, topic, top_k=2, keywords=keywords)
    ids = [p["paper_id"] for p in ranked]
    assert "p_relevant" in ids
    assert "p_off_topic" not in ids


# ---------- F3：引用堆砌检测与确定性拆散 ----------

def test_detect_citation_dumps_flags_consecutive_small_groups():
    text = "具有研究价值[1, 2, 3][4, 5, 6][7, 8, 9][10, 11, 12]。"
    dumps = detect_citation_dumps(text, max_per_group=3)
    assert len(dumps) == 1
    assert dumps[0]["citation_count"] == 12


def test_break_citation_dumps_truncates_to_threshold():
    text = "具有研究价值[1, 2, 3][4, 5, 6][7, 8, 9][10, 11, 12]。"
    result = break_citation_dumps(text, max_per_group=3)
    assert result == "具有研究价值[1, 2, 3]。"
    assert not detect_citation_dumps(result, max_per_group=3)


def test_render_in_text_citations_no_longer_splits_into_groups():
    papers = [{"paper_id": f"p{i}"} for i in range(1, 7)]
    text = "相关工作取得了进展[p1, p2, p3, p4, p5, p6]。"
    rendered = render_in_text_citations(text, papers, "gbt7714")
    assert rendered == "相关工作取得了进展[1, 2, 3]。"


# ---------- F2：章节不删空的保守证据段 ----------

def test_conservative_evidence_section_lists_allocated_cards():
    from app.deliverables.renderers.base_renderer import _conservative_evidence_section

    class _Section:
        id = "status_overview"
        title = "研究现状"
        heading_level = 2
        supporting_paper_ids = ["p1", "p2"]

    cards = [
        {"paper_id": "p1", "title": "论文一", "year": 2024, "venue": "期刊A"},
        {"paper_id": "p2", "title": "论文二", "year": 2025},
        {"paper_id": "p3", "title": "未分配论文", "year": 2025},
    ]
    text = _conservative_evidence_section(_Section(), cards)
    assert "## 研究现状" in text
    assert "《论文一》（2024，期刊A）[p1]。" in text
    assert "[p3]" not in text


def test_conservative_evidence_section_handles_no_cards():
    from app.deliverables.renderers.base_renderer import _conservative_evidence_section

    class _Section:
        id = "background_body"
        title = "研究背景"
        heading_level = 2
        supporting_paper_ids = []

    text = _conservative_evidence_section(_Section(), [])
    assert "没有分配给本节的论文" in text


# ---------- 背景 fallback 不再堆砌 ----------

def test_background_fallback_has_no_citation_runs():
    from app.deliverables.renderers.background_renderer import BackgroundRenderer
    from app.schemas.deliverable_schema import WritingSection

    plan = type("Plan", (), {})()
    plan.sections = [
        WritingSection(
            id="background_body", title="研究背景", purpose="",
            supporting_paper_ids=[f"p{i}" for i in range(20)],
            supporting_claim_ids=[], target_word_count=300, heading_level=2,
        )
    ]
    plan.citation_policy = {"minimum_unique_references": 20}
    cards = [
        {"paper_id": f"p{i}", "title": f"论文{i}", "year": 2024,
         "research_problem": "问题", "limitations": ["局限"]}
        for i in range(20)
    ]
    text = BackgroundRenderer().render_fallback(plan, {}, cards)
    # 逐点归因：每个问题/局限陈述只挂自己的来源 pid（不再成组钉前 3 张卡）
    assert "问题[p0]；问题[p1]；问题[p2]" in text
    assert "局限[p0]；局限[p1]；局限[p2]" in text
    assert "][" not in text
    assert not detect_citation_dumps(text, max_per_group=3)


def test_background_fallback_cites_each_points_own_source():
    """回归：兜底文本不得把甲论文的陈述挂到乙论文头上。

    背景：原实现把前 6 张卡的 research_problem/limitations 文本拼成一句，
    再统一引用前 3 张卡的 pid——当靠前的卡片缺字段时，后面卡片的内容
    会被错误归因到无关论文。现在每个陈述句只引用自己的来源卡片。
    """
    from app.deliverables.renderers.background_renderer import BackgroundRenderer
    from app.schemas.deliverable_schema import WritingSection

    plan = type("Plan", (), {})()
    plan.sections = [
        WritingSection(
            id="background_body", title="研究背景", purpose="",
            supporting_paper_ids=["p1", "p2", "p3", "p4"],
            supporting_claim_ids=[], target_word_count=300, heading_level=2,
        )
    ]
    plan.citation_policy = {"minimum_unique_references": 4}
    # p1、p2 没有 research_problem / limitations，内容实际来自 p3、p4
    cards = [
        {"paper_id": "p1", "title": "论文一"},
        {"paper_id": "p2", "title": "论文二"},
        {"paper_id": "p3", "title": "论文三",
         "research_problem": "三号论文的研究问题", "limitations": ["三号论文的局限"]},
        {"paper_id": "p4", "title": "论文四",
         "research_problem": "四号论文的研究问题", "limitations": ["四号论文的局限"]},
    ]
    text = BackgroundRenderer().render_fallback(plan, {}, cards)

    # 每条陈述只引用自己的来源：p3/p4 的内容不得再被钉上 [p1, p2, ...]
    assert "三号论文的研究问题[p3]" in text
    assert "四号论文的研究问题[p4]" in text
    assert "三号论文的局限[p3]" in text
    assert "四号论文的局限[p4]" in text
    assert "[p1" not in text.replace("[p1]", "")  # p1 无内容，不应出现在任何引用组里


def test_background_fallback_neutralizes_paper_self_reference_only():
    from app.deliverables.renderers.background_renderer import BackgroundRenderer
    from app.schemas.deliverable_schema import WritingSection

    plan = type("Plan", (), {})()
    plan.sections = [WritingSection(
        id="background_body", title="研究背景", purpose="",
        supporting_paper_ids=["p1"], supporting_claim_ids=[],
        target_word_count=200, heading_level=2,
    )]
    plan.citation_policy = {"minimum_unique_references": 1}
    cards = [{
        "paper_id": "p1",
        "title": "课堂互动研究",
        "research_problem": "本研究旨在分析课堂互动行为",
        "limitations": ["本文未覆盖跨学段样本"],
    }]

    text = BackgroundRenderer().render_fallback(plan, {}, cards)

    assert "本研究旨在" not in text
    assert "本文未覆盖" not in text
    assert "该研究旨在分析课堂互动行为[p1]" in text
    assert "该研究未覆盖跨学段样本[p1]" in text


# ---------- F6：澄清回答宽口径识别 ----------

def test_quality_decision_recognizes_verbose_direct_generation_answers():
    from app.services.research_conversation_service import ResearchConversationService

    for answer in (
        "直接基于现有证据生成最佳可用草稿",
        "直接生成当前最佳草稿",
        "就用当前的最佳草稿",
        "基于现有证据直接写",
    ):
        decision = ResearchConversationService._parse_quality_decision(
            answer, {"phase": "pre_generation"},
        )
        assert decision == {"action": "force_generate"}, answer
