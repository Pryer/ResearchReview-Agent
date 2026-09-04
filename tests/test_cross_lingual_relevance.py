# -*- coding: utf-8 -*-
"""测试跨语言相关性打分功能。"""

import pytest
from app.tools.rank_papers import (
    _extract_topic_synonyms,
    compute_relevance_score,
    compute_protocol_relevance_score,
    rank_papers,
)


def test_extract_topic_synonyms():
    """测试从 keywords 提取跨语言与同义变体。"""
    topic = "少样本动作识别"
    keywords = [
        "少样本动作识别",
        "few-shot action recognition",
        "few-shot video classification",
        "小样本动作识别",
        "action recognition",
    ]
    synonyms = _extract_topic_synonyms(topic, keywords)
    assert "few-shot action recognition" in synonyms
    assert "few-shot video classification" in synonyms
    assert "小样本动作识别" in synonyms
    assert topic not in synonyms  # 原 topic 不应重复作为同义词


def test_extract_topic_synonyms_preserves_late_english_aliases():
    """中文扩展词先占位时，英文锚点仍必须进入硬筛选语境。"""
    keywords = [
        "课堂行为分析", "课堂观察", "教学互动", "行为编码", "质性编码",
        "课堂提问", "教师反馈", "互动模式", "教学行为特征",
        "classroom behavior analysis", "teaching behavior analysis",
    ]

    synonyms = _extract_topic_synonyms("课堂行为分析", keywords)

    assert "classroom behavior analysis" in synonyms
    assert "teaching behavior analysis" in synonyms


def test_cross_lingual_relevance_chinese_topic_english_paper():
    """测试中文 topic 对英文相关论文的打分提升。"""
    topic = "少样本动作识别"
    topic_synonyms = ["few-shot action recognition", "few-shot video classification"]

    # 1. 英文高度相关论文 (标题命中英文同义词)
    paper_en_relevant = {
        "paper_id": "p1",
        "title": "Few-Shot Action Recognition via Temporal Alignment",
        "abstract": "We propose a novel framework for few-shot video action recognition.",
        "year": 2024,
    }
    score_with_synonyms = compute_relevance_score(
        paper_en_relevant, topic, topic_synonyms=topic_synonyms
    )
    score_without_synonyms = compute_relevance_score(
        paper_en_relevant, topic, topic_synonyms=[]
    )
    assert score_with_synonyms >= 0.35
    assert score_with_synonyms > score_without_synonyms

    # 2. 完全无关论文
    paper_irrelevant = {
        "paper_id": "p2",
        "title": "Quantum Cryptography and Entanglement Routing",
        "abstract": "We study quantum key distribution in optical networks.",
        "year": 2024,
    }
    score_irrelevant = compute_relevance_score(
        paper_irrelevant, topic, topic_synonyms=topic_synonyms
    )
    assert score_irrelevant < 0.1


def test_cross_lingual_relevance_english_topic_chinese_paper():
    """测试英文 topic 对中文相关论文的打分提升。"""
    topic = "few-shot action recognition"
    topic_synonyms = ["少样本动作识别", "小样本动作识别"]

    paper_zh_relevant = {
        "paper_id": "p1",
        "title": "基于度量学习的少样本动作识别算法研究",
        "abstract": "本文提出一种针对视频动作的小样本分类网络。",
        "year": 2024,
    }
    score = compute_relevance_score(
        paper_zh_relevant, topic, topic_synonyms=topic_synonyms
    )
    assert score >= 0.35


def test_rank_papers_cross_lingual_ordering():
    """测试 rank_papers 在有中英文混合论文时按主题贴合度优先排序。"""
    topic = "少样本动作识别"
    keywords = ["少样本动作识别", "few-shot action recognition", "video classification"]

    papers = [
        {
            "paper_id": "p_irrelevant",
            "title": "A Survey of Database Indexing Algorithms",
            "abstract": "We review modern indexing techniques in relational databases.",
            "year": 2024,
            "citation_count": 100,
        },
        {
            "paper_id": "p_relevant_en",
            "title": "Few-Shot Action Recognition via Temporal Alignment",
            "abstract": "Few-shot video action recognition under low data regime.",
            "year": 2024,
            "citation_count": 5,
        },
        {
            "paper_id": "p_relevant_zh",
            "title": "基于原型网络的少样本动作识别方法",
            "abstract": "本文针对视频少样本动作分类提出了新结构。",
            "year": 2023,
            "citation_count": 2,
        },
    ]

    ranked = rank_papers(papers, topic, top_k=3, keywords=keywords)
    # 主题锚点硬下限：无关论文（数据库索引综述）不再进入结果池
    assert len(ranked) == 2
    top_ids = [p["paper_id"] for p in ranked]
    assert "p_relevant_en" in top_ids
    assert "p_relevant_zh" in top_ids
    assert "p_irrelevant" not in top_ids


def test_penalty_activates_cross_language_group_via_synonym_anchor():
    """纯中文主题下，英文单语概念组经英文同义词锚点激活惩罚，不再静默跳过。"""
    from app.tools.rank_papers import _apply_task_mismatch_penalty

    topic = "少样本动作识别"
    # 单语英文组（可能来自历史会话或分支 core_concepts），主题本身不含英文词
    concepts = [["action recognition", "activity recognition"]]
    synonyms = ["few-shot action recognition"]
    off_topic_title = "Remote Sensing Image Change Detection"

    penalized = _apply_task_mismatch_penalty(
        0.8, topic, off_topic_title, "",
        required_concepts=concepts, topic_synonyms=synonyms,
    )
    without_synonyms = _apply_task_mismatch_penalty(
        0.8, topic, off_topic_title, "",
        required_concepts=concepts,
    )
    assert penalized == 0.55  # 0.8 - 0.25：同义词锚点激活维度校验
    assert without_synonyms == 0.8  # 旧行为：维度静默跳过


def test_penalty_matcher_consistent_with_scoring_path():
    """连字符/空格变体标题不得在打分路径命中概念组、却在惩罚路径被误判缺失。"""
    from app.tools.rank_papers import _apply_task_mismatch_penalty, _term_matches_haystack

    topic = "少样本动作识别"
    concepts = [["few-shot", "少样本"]]
    title = "Few Shot Action Recognition via Temporal Alignment"

    # 打分路径：归一化匹配器把 "few shot" 视为命中 "few-shot"
    assert _term_matches_haystack("few-shot", title.lower())
    # 惩罚路径必须使用同一匹配器：标题已含该维度，不惩罚
    score = _apply_task_mismatch_penalty(
        0.9, topic, title, "", required_concepts=concepts,
    )
    assert score == 0.9


def test_penalty_still_punishes_title_missing_active_dimension():
    """维度激活后，标题与摘要均缺失该维度的论文仍被惩罚。"""
    from app.tools.rank_papers import _apply_task_mismatch_penalty

    topic = "少样本动作识别"
    concepts = [["few-shot", "少样本"], ["action recognition", "动作识别"]]
    title = "A Survey of Database Indexing Algorithms"

    score = _apply_task_mismatch_penalty(
        0.9, topic, title, "", required_concepts=concepts,
    )
    # 两个维度均缺失：0.25 * 2 + 0.15 = 0.65
    assert score == pytest.approx(0.25)
