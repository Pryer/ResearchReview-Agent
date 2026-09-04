# -*- coding: utf-8 -*-
"""意图/检索词规划优化的行为测试：主题提取、补充查询、通用词过滤、动词前缀剥离。"""

from __future__ import annotations

from app.agent.focus_coverage import supplemental_focus_queries
from app.agent.slot_extractor import extract_topic
from app.core.source_capabilities import (
    is_generic_search_keyword,
    sanitize_search_keyword,
)


# ---------- G3：主题提取 ----------

def test_topic_extraction_from_generate_section_phrase():
    """“生成X研究现状”句式：交付物名词之前的内容才是主题。"""
    assert extract_topic("生成少样本动作识别研究现状，至少引用2篇") == "少样本动作识别"
    assert extract_topic("帮我生成联邦学习的研究背景") == "联邦学习"
    assert extract_topic("写课堂行为分析相关工作") == "课堂行为分析"


def test_topic_extraction_verb_leading_pattern_is_anchored():
    """动词引导模式必须锚定句首，防止“识别【研究】现状”中间的“研究”截断主题。"""
    assert extract_topic("调研少样本动作识别的研究现状") == "少样本动作识别"
    assert extract_topic("检索图神经网络异常检测相关论文") == "图神经网络异常检测"


def test_topic_extraction_unaffected_for_common_phrasings():
    assert extract_topic("帮我调研近五年目标检测的论文，引用不少于5篇") == "目标检测"
    assert extract_topic("关于联邦学习的综述") == "联邦学习"


# ---------- H2：无内容主题防线 ----------

def test_contentless_topic_is_rejected():
    """“近三年综述”式残片没有任何领域内容，不得作为主题进入规划。"""
    from app.agent.slot_extractor import has_topic_content

    assert has_topic_content("少样本动作识别") is True
    assert has_topic_content("目标检测") is True
    assert has_topic_content("object detection") is True
    for residual in ("近三年综述", "综述", "研究", "参考文献", ""):
        assert has_topic_content(residual) is False, residual


def test_survey_only_queries_never_produce_contentless_topic():
    """各种“近三年综述”句式：主题要么 None（交 LLM 补充），要么有领域内容。"""
    from app.agent.slot_extractor import has_topic_content

    queries = [
        "近三年综述",
        "帮我生成近三年综述",
        "调研近三年综述论文",
        "生成少样本动作识别近三年综述",
        "近三年少样本动作识别综述",
        "写一篇近三年综述",
        "生成近三年综述并引用30篇",
        "综述",
    ]
    for query in queries:
        topic = extract_topic(query)
        assert topic is None or has_topic_content(topic), (query, topic)
    # 含领域词的句式仍正确提取
    assert extract_topic("生成少样本动作识别近三年综述") == "少样本动作识别"


def test_survey_only_queries_pass_the_guard():
    from app.agent.intent import recognize_intent
    from app.agent.unsupported_task_guard import check_unsupported_task

    for query in ("近三年综述", "帮我生成近三年综述", "生成近三年综述并引用30篇"):
        intent = recognize_intent(query, llm=None)
        guard = check_unsupported_task(query, intent=intent.intent)
        assert guard.allowed is True, query
        assert guard.supported_deliverables, query


# ---------- G4a：补充查询生成 ----------

def test_supplemental_queries_skip_generic_verb_aliases():
    frame = {
        "evidence_requirements": [
            {
                "requirement_id": "action:survey",
                "label": "综述调研",
                "aliases": ["调研", "survey", "综述"],
            }
        ]
    }
    queries = supplemental_focus_queries(
        ["action:survey"], "少样本动作识别", frame,
    )
    assert queries == ["少样本动作识别 综述"]


def test_supplemental_queries_put_topic_first():
    frame = {
        "evidence_requirements": [
            {"requirement_id": "m:metric", "label": "评价指标", "aliases": ["benchmark"]}
        ]
    }
    queries = supplemental_focus_queries(["m:metric"], "少样本动作识别", frame)
    assert queries == ["少样本动作识别 benchmark"]


# ---------- G4b/G4c：检索词清洗 ----------

def test_sanitize_strips_verb_prefix():
    assert sanitize_search_keyword("调研 少样本动作识别") == "少样本动作识别"
    assert sanitize_search_keyword("检索少样本动作识别研究综述") == "少样本动作识别研究综述"


def test_generic_search_keywords_are_recognized():
    assert is_generic_search_keyword("现状")
    assert is_generic_search_keyword("survey")
    assert not is_generic_search_keyword("少样本动作识别")
    assert not is_generic_search_keyword("action recognition")
