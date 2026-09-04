"""论文排序模块测试。"""

from __future__ import annotations

import json
import re

import pytest

from app.agent.nodes import rank_node
from app.tools.paper_matching import compile_scope
from app.tools.rank_papers import (
    deduplicate_and_rank,
    evaluate_document_type_filter,
    evaluate_screening_protocol_hard_filter,
    evaluate_search_branch_filter,
    evaluate_scope_filter,
    llm_rerank_papers,
    normalize_title,
    passes_topic_filter,
    title_similarity,
)


def test_compile_scope_is_stable_and_shared_by_scope_filters():
    scope = {
        "include_terms": ["classroom interaction", "课堂互动"],
        "exclude_terms": ["physics"],
        "seed_queries": ["classroom interaction coding"],
        "branches": [{"scope_id": "education", "seed_queries": ["classroom interaction coding"]}],
    }
    protocol = {
        "hard_include_criteria": [{
            "label": "课堂场景", "terms_zh": ["课堂互动"],
            "terms_en": ["classroom interaction"], "source": "confirmed_scope",
            "applies_to_each_paper": True,
        }],
        "hard_exclude_title_terms": ["physics"],
    }
    kwargs = dict(
        selected_scope=scope,
        semantic_frame={"research_objects": [{"surface_text": "classroom interaction", "aliases": ["课堂互动"]}]},
        screening_protocol=protocol,
        required_concepts=[["classroom interaction", "课堂互动"]],
        topic_anchors=[["classroom interaction", "课堂互动"]],
        topic="classroom interaction",
    )
    first = compile_scope(**kwargs)
    second = compile_scope(**kwargs)
    assert first["version"] == "compiled_scope.v1"
    assert first["fingerprint"] == second["fingerprint"]
    assert first["aliases"]["topic_anchor"] == ["classroom interaction", "课堂互动"]
    assert "interact" in first["tokens"]["en"]
    assert compile_scope(**{**kwargs, "selected_scope": {**scope, "exclude_terms": ["physics", "mechanics"]}})["fingerprint"] != first["fingerprint"]


def test_compiled_scope_keeps_bilingual_scope_and_protocol_decisions_consistent():
    scope = {"include_terms": ["classroom interaction", "课堂互动"], "exclude_terms": ["physics"]}
    protocol = {"hard_include_criteria": [{
        "label": "课堂场景", "terms_zh": ["课堂互动"],
        "terms_en": ["classroom interaction"], "source": "confirmed_scope",
        "applies_to_each_paper": True,
    }]}
    compiled = compile_scope(selected_scope=scope, screening_protocol=protocol, topic_anchors=[["classroom interaction", "课堂互动"]])
    english = {"title": "Classroom-interaction coding", "abstract": "A classroom interaction study.", "venue": "Journal"}
    chinese = {"title": "课堂互动编码研究", "abstract": "课堂互动研究", "venue": "期刊"}
    outside = {"title": "Physics interaction", "abstract": "A physics study", "venue": "Journal"}
    from app.tools.language_filter import evaluate_language_hard_filter
    assert evaluate_scope_filter(english, scope, compiled)[0] is True
    assert evaluate_scope_filter(chinese, scope, compiled)[0] is True
    assert evaluate_scope_filter(outside, scope, compiled)[0] is False
    assert evaluate_language_hard_filter(english, protocol, "en", compiled)[0] is True
    assert evaluate_language_hard_filter(chinese, protocol, "zh", compiled)[0] is True


def _context_screening_protocol() -> dict:
    return {
        "version": "1.0",
        "corpus_goal": "课堂行为自动识别、编码与教育学分析由证据池共同覆盖",
        "hard_include_criteria": [{
            "criterion_id": "classroom_context",
            "label": "课堂场景",
            "terms": ["classroom", "课堂"],
            "source": "user_explicit",
            "applies_to_each_paper": True,
        }],
        "soft_include_criteria": [{
            "criterion_id": "preferred_routes",
            "label": "技术与教育分析",
            "terms": [
                "behavior recognition", "行为识别",
                "teacher-student interaction", "师生互动",
            ],
            "source": "confirmed_scope",
            "applies_to_each_paper": False,
        }],
        "hard_exclude_title_terms": [],
        "routes": [
            {
                "route_id": "automatic_recognition",
                "label": "自动识别",
                "terms": ["behavior recognition", "automatic coding"],
                "weight": 0.4,
            },
            {
                "route_id": "educational_analysis",
                "label": "教育学分析",
                "terms": ["teacher-student interaction", "classroom observation"],
                "weight": 0.6,
            },
        ],
        "generated_by": "llm",
    }


def test_context_protocol_treats_research_routes_as_corpus_level_soft_conditions():
    papers = [
        _make_paper(
            paper_id="technical",
            title="Automatic Classroom Behavior Recognition with YOLO",
            abstract="The system detects student actions.",
        ),
        _make_paper(
            paper_id="education",
            title="Teacher-Student Interaction Coding in Classroom Observation",
            abstract="An educational analysis of instructional practice.",
        ),
        _make_paper(
            paper_id="unrelated",
            title="Automatic Defect Recognition in Industrial Production",
            abstract="A computer vision inspection system.",
            citation_count=1000,
        ),
    ]
    diagnostics: dict = {}

    result = deduplicate_and_rank(
        papers,
        topic="课堂行为分析",
        top_k=10,
        # 长检索词不再成为上下文协议模式下的逐篇硬门槛。
        keywords=["课堂行为 自动识别 行为编码 教育学分析"],
        screening_protocol=_context_screening_protocol(),
        filter_diagnostics=diagnostics,
    )

    assert {paper["paper_id"] for paper in result} == {"technical", "education"}
    assert diagnostics["filtered_count"] == 1
    assert diagnostics["filtered_by_stage"] == {"protocol_hard_filter": 1}
    assert diagnostics["passed_hard_filters"] == 2


def test_context_protocol_hard_filter_only_enforces_each_paper_criteria():
    protocol = _context_screening_protocol()
    protocol["soft_include_criteria"].append({
        "criterion_id": "ai_preference",
        "label": "人工智能",
        "terms": ["artificial intelligence"],
        "source": "user_explicit",
        "applies_to_each_paper": False,
    })
    education_paper = _make_paper(
        title="Classroom Observation and Teacher-Student Interaction",
        abstract="A qualitative coding study.",
    )

    passed, reason = evaluate_screening_protocol_hard_filter(
        education_paper, protocol,
    )

    assert passed is True
    assert "通过上下文硬条件" in reason


def test_generic_hard_anchor_is_lexical_and_defers_semantics_to_llm():
    protocol = _context_screening_protocol()
    protocol["hard_include_criteria"] = [{
        "criterion_id": "classroom_behavior_anchor",
        "label": "课堂行为研究对象",
        "terms": ["classroom behavior", "课堂行为"],
        "source": "confirmed_scope",
        "applies_to_each_paper": True,
    }]
    adjacent = _make_paper(
        title="AI Robot for Learning Behavior in Laboratory Safety Courses",
        abstract=(
            "The course is delivered in a classroom setting. "
            "The robot improves learning behavior and motivation."
        ),
    )
    direct = _make_paper(
        title="Recognition of Student Behavior in the Classroom",
        abstract="A vision model detects classroom student behaviors.",
    )

    assert evaluate_screening_protocol_hard_filter(adjacent, protocol)[0] is True
    assert evaluate_screening_protocol_hard_filter(direct, protocol)[0] is True


def test_hard_filter_has_no_domain_specific_pipeline_exceptions():
    protocol = _context_screening_protocol()
    protocol["hard_include_criteria"] = [{
        "criterion_id": "classroom_behavior_anchor",
        "label": "课堂行为研究对象",
        "terms": ["classroom behavior", "课堂行为"],
        "source": "confirmed_scope",
        "applies_to_each_paper": True,
    }]
    ifias = _make_paper(
        title="基于iFIAS的课堂师生互动行为分析",
        abstract="使用课堂观察编码分析师生互动结构。",
    )
    lag = _make_paper(
        title="Lag Sequential Analysis of Teacher-Student Classroom Interaction",
        abstract="The study codes interaction sequences in teaching.",
    )
    trauma = _make_paper(
        title="Impact of Trauma-Informed Teaching on Classroom Behavior",
        abstract="The intervention measures engagement and wellbeing outcomes.",
    )

    assert evaluate_screening_protocol_hard_filter(ifias, protocol)[0] is False
    assert evaluate_screening_protocol_hard_filter(lag, protocol)[0] is False
    assert evaluate_screening_protocol_hard_filter(trauma, protocol)[0] is True


def test_context_llm_screening_retains_uncertain_and_excludes_only_high_confidence():
    papers = [
        _make_paper(paper_id="uncertain", title="Classroom Interaction Study"),
        _make_paper(paper_id="other", title="Industrial Inspection Study"),
    ]

    class ScreeningLLM:
        def complete(self, prompt: str, **kwargs) -> str:
            paper_ids = re.findall(r'"paper_id":\s*"([^"]+)"', prompt)
            results = []
            for paper_id in paper_ids:
                if paper_id == "other":
                    results.append({
                        "paper_id": paper_id,
                        "topic_relevance": 1,
                        "scope_alignment": 1,
                        "method_alignment": 2,
                        "decision": "exclude",
                        "confidence": 0.95,
                        "route_id": None,
                    })
                else:
                    results.append({
                        "paper_id": paper_id,
                        "topic_relevance": 6,
                        "scope_alignment": 6,
                        "method_alignment": 5,
                        "decision": "uncertain",
                        "confidence": 0.6,
                        "route_id": "educational_analysis",
                    })
            return json.dumps({"results": results})

    diagnostics: dict = {}
    result = llm_rerank_papers(
        papers,
        topic="课堂行为分析",
        llm=ScreeningLLM(),
        top_k=10,
        screening_protocol=_context_screening_protocol(),
        rerank_diagnostics=diagnostics,
    )

    assert [paper["paper_id"] for paper in result] == ["uncertain"]
    assert diagnostics["excluded_count"] == 1
    assert diagnostics["uncertain_retained_count"] == 1


def test_high_confidence_exclusions_are_never_restored_as_reserve():
    papers = [
        _make_paper(paper_id=f"p{index}", title=f"Unrelated Study {index}")
        for index in range(5)
    ]

    class ExcludingLLM:
        def complete(self, prompt: str, **kwargs) -> str:
            paper_ids = re.findall(r'"paper_id":\s*"([^"]+)"', prompt)
            return json.dumps({"results": [{
                "paper_id": paper_id,
                "topic_relevance": 1,
                "scope_alignment": 1,
                "method_alignment": 1,
                "decision": "exclude",
                "confidence": 0.99,
                "relation_type": "unrelated",
                "eligible_deliverables": [],
            } for paper_id in paper_ids]})

    diagnostics: dict = {}
    result = llm_rerank_papers(
        papers,
        topic="少样本动作识别",
        llm=ExcludingLLM(),
        top_k=5,
        minimum_required=4,
        rerank_diagnostics=diagnostics,
    )

    assert result == []
    assert diagnostics["excluded_count"] == 5
    assert diagnostics["reserve_backfilled_count"] == 0
    assert diagnostics["hard_excluded_paper_ids"] == [f"p{i}" for i in range(5)]


def test_scope_filter_happens_before_top_k_and_backfills():
    candidates = [
        _make_paper(paper_id="outside-1", title="Unrelated High Impact", citation_count=1000),
        _make_paper(paper_id="outside-2", title="Another Unrelated", citation_count=900),
        _make_paper(paper_id="inside-1", title="Keep Topic One", abstract="keep topic"),
        _make_paper(paper_id="inside-2", title="Keep Topic Two", abstract="keep topic"),
    ]
    state = {
        "candidate_papers": candidates,
        "topic": "unknown topic",
        "keywords": [],
        "max_papers": 2,
        "retrieval_target": 2,
        "selected_scope": {"include_terms": ["keep topic"]},
        "steps": [],
        "errors": [],
    }

    rank_node(state)

    assert [paper["paper_id"] for paper in state["ranked_papers"]] == [
        "inside-1", "inside-2"
    ]


def test_recovery_branch_minimum_lifts_targeted_recall_into_top_k():
    """定向补检索召回的论文不能因全局分数偏低而在 top_k 截断处消失。"""
    papers = [
        _make_paper(
            paper_id=f"strong-{index}",
            title=f"Classroom Behavior Recognition {index}",
            abstract="classroom behavior recognition deep learning",
            citation_count=900 - index,
        )
        for index in range(10)
    ]
    recovered = _make_paper(
        paper_id="recovered-1",
        title="Classroom Behavior Recognition Teacher Feedback",
        abstract="classroom behavior recognition teacher feedback",
        citation_count=0,
    )
    recovered["_search_branches"] = ["evidence_recovery_1"]
    diagnostics: dict = {}

    result = deduplicate_and_rank(
        [*papers, recovered],
        topic="classroom behavior recognition",
        top_k=5,
        branch_minimums={"evidence_recovery_1": 1},
        filter_diagnostics=diagnostics,
    )

    assert "recovered-1" in {paper["paper_id"] for paper in result}
    assert diagnostics["branch_minimums"] == {"evidence_recovery_1": 1}


def test_recovery_branch_minimum_does_not_relax_hard_filters():
    """配额只影响名额分配；未通过范围硬过滤的定向召回仍必须被剔除。"""
    inside = _make_paper(
        paper_id="inside-1",
        title="Keep Topic One",
        abstract="keep topic",
    )
    outside = _make_paper(
        paper_id="recovered-out",
        title="Completely Unrelated Study",
        abstract="unrelated content",
    )
    outside["_search_branches"] = ["evidence_recovery_1"]

    result = deduplicate_and_rank(
        [inside, outside],
        topic="keep topic",
        top_k=5,
        scope={"include_terms": ["keep topic"]},
        branch_minimums={"evidence_recovery_1": 2},
    )

    assert {paper["paper_id"] for paper in result} == {"inside-1"}


def test_rule_ranking_preserves_configured_minimum_for_cnki_before_top_k():
    papers = [
        _make_paper(
            paper_id=f"international-{index}",
            title=f"Classroom Behavior Recognition {index}",
            abstract="classroom behavior recognition",
            citation_count=500 - index,
            source="semantic_scholar",
        )
        for index in range(130)
    ] + [
        _make_paper(
            paper_id="cnki-1",
            title="智慧教学环境下混合式课堂行为研究",
            abstract="课堂行为分析 滞后序列分析",
            source="cnki",
        ),
        _make_paper(
            paper_id="cnki-2",
            title="开放环境学生课堂行为识别与应用",
            abstract="课堂行为分析 目标检测",
            source="cnki",
        ),
        _make_paper(
            paper_id="cnki-3",
            title="基于目标检测的课堂参与行为分析",
            abstract="课堂行为分析 自动编码",
            source="cnki",
        ),
    ]

    result = deduplicate_and_rank(
        papers,
        topic="课堂行为分析",
        top_k=20,
        source_minimums={"cnki": 3},
    )

    assert sum(paper["source"] == "cnki" for paper in result) == 3


def test_rank_node_with_cnki_candidates_computes_source_quota():
    """回归：rank_node 内部按 settings.search_sources_list 计算 CNKI 配额，
    曾因未绑定 settings 抛 NameError，导致整条流水线拿到 0 篇论文。"""
    candidates = [
        _make_paper(
            paper_id=f"cnki-{index}",
            title=f"基于深度学习的课堂行为分析研究{index}",
            abstract="课堂行为分析 深度学习",
            source="cnki",
        )
        for index in range(5)
    ] + [
        _make_paper(
            paper_id=f"intl-{index}",
            title=f"Classroom Behavior Analysis {index}",
            abstract="classroom behavior analysis deep learning",
            source="openalex",
        )
        for index in range(30)
    ]
    state = {
        "candidate_papers": candidates,
        "topic": "课堂行为分析",
        "keywords": ["课堂行为分析"],
        "max_papers": 40,
        "retrieval_target": 120,
        "required_reference_count": 40,
    }

    out = rank_node(state)

    assert out.get("errors") is None
    assert out["screening_report"]["rule_filter"]["source_minimums"].get("cnki") >= 1
    ranked = out["ranked_papers"]
    assert ranked, "CNKI 候选存在时 rank_node 应产出排序结果"
    assert any(paper["source"] == "cnki" for paper in ranked)


def test_llm_rerank_scores_all_batches_and_does_not_backfill_below_threshold():
    papers = [
        _make_paper(
            paper_id=f"p{index}",
            title=f"Paper {index}",
            abstract="same evidence",
        )
        for index in range(80)
    ]

    class GlobalScoreLLM:
        def complete(self, prompt: str, **kwargs) -> str:
            paper_ids = re.findall(r'"paper_id":\s*"(p\d+)"', prompt)
            return json.dumps({
                "results": [
                    {
                        "paper_id": paper_id,
                        "topic_relevance": int(paper_id[1:]) / 8,
                        "scope_alignment": int(paper_id[1:]) / 8,
                        "method_alignment": int(paper_id[1:]) / 8,
                        "reason": "global score",
                    }
                    for paper_id in paper_ids
                ]
            })

    result = llm_rerank_papers(
        papers,
        topic="test",
        llm=GlobalScoreLLM(),
        top_k=60,
    )

    # 低分论文不再被 t_rel/s_align 阈值排除：全部参与排序，仅按 top_k 截断
    assert len(result) == 60
    assert result[0]["paper_id"] == "p79"
    assert result[-1]["paper_id"] == "p20"


def test_mixed_review_rerank_reserves_domain_observation_route():
    papers = [
        _make_paper(
            paper_id=f"tech-{index}",
            title=f"YOLO Classroom Detection {index}",
            abstract="A technical classroom behavior detection model.",
        )
        for index in range(16)
    ] + [
        _make_paper(
            paper_id=f"domain-{index}",
            title=f"Classroom Observation and Teacher-Student Interaction {index}",
            abstract="A classroom observation coding study of instructional practice.",
            _search_branches=["domain_foundation"],
        )
        for index in range(4)
    ]

    class RouteScoreLLM:
        def complete(self, prompt: str, **kwargs) -> str:
            paper_ids = re.findall(r'"paper_id":\s*"([^"]+)"', prompt)
            return json.dumps({
                "results": [
                    {
                        "paper_id": paper_id,
                        "topic_relevance": 6 if paper_id.startswith("domain-") else 9,
                        "scope_alignment": 7 if paper_id.startswith("domain-") else 9,
                        "method_alignment": 6 if paper_id.startswith("domain-") else 9,
                        "decision": "include",
                        "confidence": 0.9,
                        "relation_type": "direct",
                        "eligible_deliverables": ["research_status"],
                        "route_id": "domain_evidence" if paper_id.startswith("domain-") else "technical_evidence",
                        "reason": "route score",
                    }
                    for paper_id in paper_ids
                ]
            })

    result = llm_rerank_papers(
        papers,
        topic="课堂行为分析",
        llm=RouteScoreLLM(),
        top_k=10,
        research_mode="technology_assisted_domain_analysis",
    )

    assert len(result) == 10
    assert sum(
        paper["paper_id"].startswith("domain-") for paper in result
    ) == 4


def test_llm_rerank_parameters_are_configurable_per_call():
    papers = [
        _make_paper(paper_id=f"p{i}", title=f"Paper {i}")
        for i in range(10)
    ] + [
        _make_paper(paper_id="cnki-extra", title="CNKI Paper", source="cnki")
    ]

    class RecordingLLM:
        def __init__(self):
            self.batches = []

        def complete(self, prompt: str, **kwargs) -> str:
            ids = list(dict.fromkeys(re.findall(r'"paper_id":\s*"(p\d+|cnki-extra)"', prompt)))
            self.batches.append(ids)
            return json.dumps({"results": [
                {"paper_id": paper_id, "decision": "include", "confidence": 1.0,
                 "topic_relevance": 8, "scope_alignment": 8, "method_alignment": 8}
                for paper_id in ids
            ]})

    llm = RecordingLLM()
    result = llm_rerank_papers(
        papers, topic="test", llm=llm, top_k=2,
        cnki_quota=1, candidate_min=4, candidate_max=4, batch_size=3,
    )
    assert len(result) == 2
    assert [len(batch) for batch in llm.batches] == [3, 2]
    assert any(paper["paper_id"] == "cnki-extra" for paper in result) is False



def _make_paper(**kwargs) -> dict:
    """构造测试用论文字典。"""
    defaults = {
        "paper_id": "test:1",
        "title": "Test Paper",
        "authors": [],
        "year": 2023,
        "venue": "CVPR",
        "abstract": "",
        "doi": None,
        "arxiv_id": None,
        "url": None,
        "pdf_url": None,
        "citation_count": 0,
        "source": "test",
    }
    defaults.update(kwargs)
    return defaults


class TestNormalizeTitle:
    def test_lowercase_and_remove_punct(self):
        assert normalize_title("Hello, World!") == "hello world"

    def test_empty_returns_empty(self):
        assert normalize_title("") == ""


class TestTitleSimilarity:
    def test_identical_titles(self):
        assert title_similarity("Hello World", "Hello World") == 1.0

    def test_completely_different(self):
        assert title_similarity("abc", "xyz") == 0.0

    def test_partial_overlap(self):
        sim = title_similarity(
            "Vision Transformer for Image Classification",
            "Vision Transformer for Object Detection"
        )
        assert 0 < sim < 1


class TestDeduplicateAndRank:
    def test_deduplicate_similar_titles(self):
        papers = [
            _make_paper(paper_id="1", title="Vision Transformer for Image Classification"),
            _make_paper(paper_id="2", title="Vision Transformer for Image Classification"),
            _make_paper(paper_id="3", title="CNN for Object Detection"),
        ]
        result = deduplicate_and_rank(papers, topic="vision transformer", top_k=10)
        assert len(result) <= 2

    def test_deduplicate_keeps_distinct_few_shot_action_methods(self):
        papers = [
            _make_paper(
                paper_id="soap",
                title="SOAP: Enhancing Spatio-Temporal Relation and Motion Information Capturing for Few-Shot Action Recognition",
            ),
            _make_paper(
                paper_id="mvp",
                title="MVP-Shot: Multi-Velocity Progressive-Alignment Framework for Few-Shot Action Recognition",
            ),
        ]

        result = deduplicate_and_rank(
            papers,
            topic="少样本动作识别",
            top_k=10,
            required_concepts=[
                ["few-shot", "few shot", "fewshot", "one-shot", "one shot"],
                ["action recognition", "activity recognition", "human action"],
            ],
        )

        assert {p["paper_id"] for p in result} == {"soap", "mvp"}

    def test_rank_by_relevance(self):
        papers = [
            _make_paper(paper_id="1", title="Vision Transformer for Image Classification"),
            _make_paper(paper_id="2", title="Random Paper About Bananas", citation_count=100),
        ]
        result = deduplicate_and_rank(papers, topic="vision transformer", top_k=10)
        assert result[0]["paper_id"] == "1"  # 更相关的排前面

    def test_abstract_evidence_is_preferred_over_metadata_only_record(self):
        papers = [
            _make_paper(
                paper_id="metadata",
                title="Classroom Behavior Analysis",
                abstract="",
            ),
            _make_paper(
                paper_id="abstract",
                title="Classroom Behavior Analysis with Evidence",
                abstract="This study analyzes classroom behavior and interaction.",
            ),
        ]
        result = deduplicate_and_rank(papers, topic="classroom behavior", top_k=2)
        assert result[0]["paper_id"] == "abstract"

    def test_ranking_preserves_multi_year_coverage_when_candidates_span_three_years(self):
        papers = [
            _make_paper(
                paper_id=f"y2024-{index}",
                title=f"Classroom Behavior Study {index}",
                abstract="Behavior analysis in education.",
                year=2024,
            )
            for index in range(2)
        ] + [
            _make_paper(
                paper_id=f"y2025-{index}",
                title=f"Classroom Behavior Study {index + 2}",
                abstract="Behavior analysis in education.",
                year=2025,
            )
            for index in range(2)
        ] + [
            _make_paper(
                paper_id=f"y2026-{index}",
                title=f"Classroom Behavior Study {index + 4}",
                abstract="Behavior analysis in education.",
                year=2026,
            )
            for index in range(2)
        ]

        result = deduplicate_and_rank(papers, topic="classroom behavior", top_k=6)

        assert {paper["year"] for paper in result} == {2024, 2025, 2026}

    def test_top_k_limit(self):
        papers = [_make_paper(paper_id=str(i), title=f"Paper {i}") for i in range(10)]
        result = deduplicate_and_rank(papers, topic="paper", top_k=5)
        assert len(result) <= 5

    def test_long_video_filter_removes_unrelated_papers(self):
        papers = [
            _make_paper(
                paper_id="related",
                title="Long-Form Video Understanding with Temporal Memory",
                abstract="We study long video understanding and video-language reasoning.",
            ),
            _make_paper(
                paper_id="unrelated",
                title="Supply Chain Inventory Finance Optimization",
                abstract="This paper studies business management.",
                citation_count=1000,
            ),
        ]

        result = deduplicate_and_rank(papers, topic="长视频理解", top_k=10)

        assert {p["paper_id"] for p in result} == {"unrelated", "related"}

    def test_long_video_filter_removes_generic_video_or_cv_papers(self):
        papers = [
            _make_paper(
                paper_id="generic",
                title="Attention Mechanisms in Computer Vision: A Survey",
                abstract="This survey covers image and video recognition tasks.",
                citation_count=1000,
            ),
            _make_paper(
                paper_id="specific",
                title="MLVU: Benchmarking Multi-task Long Video Understanding",
                abstract="A benchmark for long video understanding.",
            ),
        ]

        result = deduplicate_and_rank(papers, topic="长视频理解", top_k=10)

        assert {p["paper_id"] for p in result} == {"generic", "specific"}

    def test_few_shot_action_filter_requires_few_shot_and_action_terms(self):
        papers = [
            _make_paper(
                paper_id="related",
                title="Few-Shot Action Recognition with Temporal Alignment",
                abstract="We study few-shot human action recognition in videos.",
            ),
            _make_paper(
                paper_id="sensor",
                title="Data Augmentation for Optical Fiber Pattern Recognition",
                abstract="A conditional adversarial network for signal pattern recognition.",
                citation_count=1000,
            ),
            _make_paper(
                paper_id="psychology",
                title="Suicidal Ideation Recognition Based on Large Language Models",
                abstract="Data augmentation and text recognition for psychology.",
                citation_count=1000,
            ),
            _make_paper(
                paper_id="knowledge_graph",
                title="Knowledge Graph Construction for Disaster Emergency Response",
                abstract="A geoscience knowledge graph method.",
                citation_count=1000,
            ),
        ]

        result = deduplicate_and_rank(papers, topic="少样本动作识别", top_k=10)

        assert {p["paper_id"] for p in result} == {"related", "sensor", "psychology", "knowledge_graph"}

    def test_dynamic_keywords_filter_arbitrary_topic(self):
        papers = [
            _make_paper(
                paper_id="related",
                title="Graph Neural Network Anomaly Detection on Attributed Networks",
                abstract="We propose a GNN anomaly detection method for graph outlier detection.",
            ),
            _make_paper(
                paper_id="unrelated",
                title="Climate Change Policy Performance in Local Governance",
                abstract="This paper studies public policy and local governance.",
                citation_count=1000,
            ),
        ]

        result = deduplicate_and_rank(
            papers,
            topic="图神经网络异常检测",
            top_k=10,
            keywords=[
                "图神经网络异常检测",
                "graph neural network anomaly detection",
                "GNN anomaly detection",
                "graph outlier detection",
            ],
        )

        # 主题锚点硬下限：偏题论文（气候变化政策）不再与主题论文并列存活
        assert {p["paper_id"] for p in result} == {"related"}

    def test_required_concepts_allow_wording_variation(self):
        paper = _make_paper(
            title="Few-Shot Learning for Human Action Recognition",
            abstract="A low-shot method for recognizing human activities in video.",
        )

        assert passes_topic_filter(
            paper,
            topic="少样本动作识别",
            required_concepts=[
                ["few-shot", "low-shot", "少样本"],
                ["action recognition", "human activity", "动作识别"],
            ],
        )

    def test_required_concepts_boost_chinese_synonym_relevance(self):
        papers = [
            _make_paper(
                paper_id="zh",
                title="小样本行为识别方法研究",
                abstract="",
                citation_count=0,
            ),
            _make_paper(
                paper_id="unrelated",
                title="Few-Shot Image Classification with Prototypes",
                abstract="We study low-shot visual classification.",
                citation_count=1000,
            ),
        ]

        result = deduplicate_and_rank(
            papers,
            topic="少样本动作识别",
            top_k=10,
            required_concepts=[
                ["few-shot", "low-shot", "少样本", "小样本"],
                ["action recognition", "human activity", "动作识别", "行为识别"],
            ],
        )

        # 不再因概念组部分命中而拒绝；unrelated 论文不会被删除
        assert {p["paper_id"] for p in result} == {"unrelated", "zh"}
        zh_paper = next(p for p in result if p["paper_id"] == "zh")
        assert zh_paper["_relevance_score"] >= 0.5

    def test_required_concepts_reject_partial_topic_match(self):
        paper = _make_paper(
            title="Few-Shot Image Classification with Prototypes",
            abstract="We study low-shot visual classification.",
        )

        # 不再因为概念组部分命中而拒绝论文；语义判断交由 LLM screening
        result = passes_topic_filter(
            paper,
            topic="少样本动作识别",
            required_concepts=[
                ["few-shot", "low-shot", "少样本"],
                ["action recognition", "human activity", "动作识别"],
            ],
        )
        assert result is True

    def test_required_concepts_must_describe_title_not_only_abstract(self):
        paper = _make_paper(
            title="Vision Transformer Recognition Tasks: A Survey",
            abstract="The survey briefly discusses few-shot action recognition.",
        )

        # 不再因"标题未命中"而拒绝论文：摘要已包含所有必要语义概念
        result = passes_topic_filter(
            paper,
            topic="少样本动作识别",
            required_concepts=[
                ["few-shot", "low-shot", "少样本"],
                ["action recognition", "human activity", "动作识别"],
            ],
        )
        assert result is True

    def test_excluded_neighboring_topic_is_rejected_by_title(self):
        paper = _make_paper(
            title="Zero-Shot Action Recognition with Language Knowledge",
            abstract="We compare against few-shot action recognition.",
        )

        assert not passes_topic_filter(
            paper,
            topic="少样本动作识别",
            required_concepts=[["shot"], ["action recognition"]],
            excluded_title_terms=["zero-shot action recognition"],
        )

    def test_topic_filter_is_only_strict_for_known_domains(self):
        paper = _make_paper(title="General Paper")
        assert passes_topic_filter(paper, topic="unknown topic")


class TestConfirmedScopeFilter:
    scope = {
        "include_terms": ["课堂互动", "学习行为", "课堂观察"],
        "exclude_terms": [
            "目标检测", "计算机视觉", "深度学习", "图像识别",
            "object detection", "computer vision", "deep learning", "image recognition",
        ],
        "seed_queries": ["classroom observation learning behavior"],
    }

    def test_cross_scope_accepts_english_classroom_behavior_evidence(self):
        from app.tools.rank_papers import evaluate_scope_filter

        scope = {
            "research_mode": "mixed",
            "include_terms": [
                "课堂行为分析", "自动行为编码", "领域解释", "多模态学习分析",
                "classroom behavior analysis", "automatic behavior coding",
                "educational analysis", "multimodal learning analytics",
            ],
            "seed_queries": ['"课堂行为分析" automatic coding domain analysis'],
        }
        paper = _make_paper(
            title="Automatic Classroom Behavior Recognition for Learning Analytics",
            abstract="A multimodal system detects student actions and supports educational analysis.",
        )

        assert evaluate_scope_filter(paper, scope)[0] is True

    def test_cross_scope_still_rejects_generic_action_recognition(self):
        from app.tools.rank_papers import evaluate_scope_filter

        scope = {
            "research_mode": "mixed",
            "include_terms": ["课堂行为分析", "自动行为编码", "领域解释", "多模态学习分析"],
            "seed_queries": ['"课堂行为分析" automatic coding domain analysis'],
        }
        paper = _make_paper(
            title="Action Recognition for Autonomous Driving",
            abstract="A transformer recognizes road actions from vehicle cameras.",
        )

        assert evaluate_scope_filter(paper, scope)[0] is False

    def test_rejects_unrelated_behavior_analysis_from_other_domain(self):
        paper = _make_paper(
            title="AI-Driven Cloud Security with User Behavior Analysis",
            abstract="Threat detection for cloud computing environments.",
        )
        passed, reason = evaluate_scope_filter(paper, self.scope)
        assert passed is False
        assert "纳入语境" in reason

    def test_rejects_explicitly_excluded_technical_method_cross_language(self):
        paper = _make_paper(
            title="Classroom Behavior Detection with Deep Learning",
            abstract="A computer vision model recognizes student actions.",
        )
        passed, reason = evaluate_scope_filter(paper, self.scope)
        assert passed is False
        assert "排除概念" in reason

    def test_accepts_paper_matching_confirmed_education_scope(self):
        paper = _make_paper(
            title="Classroom Observation of Learning Behavior",
            abstract="We examine teacher-student interaction and engagement.",
        )
        passed, _ = evaluate_scope_filter(paper, self.scope)
        assert passed is True

    @pytest.mark.parametrize("title", [
        "Research on Effective Classroom Teaching Behavior Based on Behavior Sequence Analysis",
        "Analysis of Characteristics of Primary School Classroom Teaching Behaviors and Optimization Strategies",
        "A Comparative Study on the Analysis of Interactive Behavior in Smart Class Teaching",
    ])
    def test_accepts_reordered_and_inflected_english_scope_terms(self, title):
        scope = {
            "include_terms": [
                "classroom interaction analysis",
                "classroom observation",
                "teaching behavior analysis",
            ],
            "seed_queries": ["classroom behavior analysis interaction coding"],
        }

        passed, reason = evaluate_scope_filter(
            _make_paper(title=title, abstract=""), scope,
        )

        assert passed is True, reason


def test_combined_scope_rejects_generic_method_without_domain_context():
    scope = {
        "include_terms": ["classroom behavior recognition", "deep learning", "object detection"],
        "branches": [{
            "scope_id": "technical",
            "label": "技术驱动",
            "seed_queries": ["classroom behavior recognition deep learning"],
        }],
    }
    generic = _make_paper(
        title="YOLO for Remote Sensing Object Detection",
        abstract="A deep learning object detector for satellite imagery.",
    )
    classroom = _make_paper(
        title="Classroom Behavior Recognition with YOLO",
        abstract="A deep model recognizes student behaviors in classrooms.",
    )

    assert evaluate_scope_filter(generic, scope)[0] is False
    assert evaluate_scope_filter(classroom, scope)[0] is True


def test_domain_application_rejects_technical_method_only_candidate():
    branches = [{
        "branch_type": "technical_method",
        "required_concepts": [["action recognition", "object detection"]],
        "constraint_level": "hard",
    }]
    paper = _make_paper(
        title="Object Detection for Industrial Defects",
        abstract="An object detection model for manufacturing.",
        _search_branches=["technical_method"],
    )

    passed, reason = evaluate_search_branch_filter(
        paper,
        branches,
        "technology_applied_to_domain",
    )
    assert passed is False
    assert "领域锚点" in reason


def test_refined_topic_core_result_must_still_match_domain_anchor():
    branches = [{
        "branch_type": "domain_foundation",
        "required_concepts": [["classroom behavior", "课堂行为"]],
        "constraint_level": "soft",
    }]
    unrelated = _make_paper(
        title="Fact-Check Game for Information Literacy",
        abstract="Log-based evidence is used to evaluate a digital learning game.",
        _search_branches=["topic_core"],
    )
    related = _make_paper(
        title="Classroom Behavior Analysis for Learning Analytics",
        abstract="The study interprets classroom behavior patterns.",
        _search_branches=["topic_core"],
    )

    # 主题过滤不再硬拒绝：未命中领域锚点的论文同样通过，仅影响打分排序
    assert evaluate_search_branch_filter(
        unrelated, branches, topic="课堂行为分析"
    )[0] is True
    assert evaluate_search_branch_filter(
        related, branches, topic="课堂行为分析"
    )[0] is True


# ============================================================
# 中英文双分支测试
# ============================================================


def _bilingual_protocol():
    """返回一个包含 terms_zh / terms_en 的测试协议。"""
    return {
        "version": "1.0",
        "corpus_goal": "课堂行为分析交叉研究",
        "hard_include_criteria": [
            {
                "criterion_id": "classroom_context",
                "label": "课堂场景",
                "terms_zh": ["课堂行为", "课堂教学", "课堂互动"],
                "terms_en": ["classroom behavior", "classroom teaching", "classroom interaction"],
                "source": "confirmed_scope",
                "applies_to_each_paper": True,
            },
        ],
        "soft_include_criteria": [
            {
                "criterion_id": "deep_learning",
                "label": "深度学习方法",
                "terms_zh": ["深度学习", "神经网络"],
                "terms_en": ["deep learning", "neural network"],
                "source": "confirmed_scope",
                "applies_to_each_paper": False,
            },
        ],
        "hard_exclude_title_terms": [],
        "routes": [
            {
                "route_id": "technical_method",
                "label": "技术方法",
                "terms_zh": ["计算机视觉", "行为识别"],
                "terms_en": ["computer vision", "behavior recognition"],
                "weight": 0.5,
                "rationale": "",
            },
            {
                "route_id": "educational_analysis",
                "label": "教育分析",
                "terms_zh": ["课堂观察", "教学互动"],
                "terms_en": ["classroom observation", "teaching interaction"],
                "weight": 0.5,
                "rationale": "",
            },
        ],
        "generated_by": "llm",
    }


class TestLanguageRouter:
    """语言检测与分支拆分。"""

    def test_detect_chinese_paper_by_cjk_title(self):
        from app.tools.language_router import detect_paper_language

        paper = {"title": "基于深度学习的课堂行为识别研究", "source": "openalex"}
        assert detect_paper_language(paper) == "zh"
        assert "_detected_language" not in paper

    def test_detect_english_paper(self):
        from app.tools.language_router import detect_paper_language

        paper = {"title": "Deep Learning for Classroom Behavior Recognition", "source": "arxiv"}
        assert detect_paper_language(paper) == "en"

    def test_detect_cnki_source_as_chinese(self):
        from app.tools.language_router import detect_paper_language

        paper = {"title": "Some English Title", "source": "cnki"}
        # CNKI 来源强制识别为中文，即使标题是英文
        assert detect_paper_language(paper) == "zh"

    def test_detect_by_metadata_language_field(self):
        from app.tools.language_router import detect_paper_language

        paper = {"title": "Mixed content", "language": "zh-cn"}
        assert detect_paper_language(paper) == "zh"

    def test_split_papers_by_language(self):
        from app.tools.language_router import split_papers_by_language

        papers = [
            {"title": "课堂行为分析", "source": "cnki"},
            {"title": "Classroom Behavior Analysis", "source": "arxiv"},
            {"title": "深度学习综述", "source": "openalex"},
            {"title": "Computer Vision Applications", "source": "semantic_scholar"},
        ]
        zh, en = split_papers_by_language(papers)
        assert len(zh) == 2
        assert len(en) == 2
        assert all(p["_language_branch"] == "zh" for p in zh)
        assert all(p["_language_branch"] == "en" for p in en)
        assert all("_language_branch" not in paper for paper in papers)

    def test_cjk_ratio_threshold(self):
        from app.tools.language_router import calculate_cjk_ratio

        assert calculate_cjk_ratio("纯中文标题") > 0.5
        assert calculate_cjk_ratio("English only") == 0.0
        # 混合文本 CJK 占比不足 15% 时不判定为中文
        mixed = "This is a study about 教育 technology"
        ratio = calculate_cjk_ratio(mixed)
        assert ratio < 0.15

    def test_japanese_kana_title_goes_to_english_branch(self):
        """含假名的日文文献不得因汉字占比高而误入 CNKI 中文支线。"""
        from app.tools.language_router import detect_paper_language

        paper = {"title": "深層学習による教室行動認識", "source": "openalex"}
        assert detect_paper_language(paper) == "en"

    def test_mixed_chinese_title_with_english_terms_stays_chinese(self):
        from app.tools.language_router import detect_paper_language

        paper = {"title": "基于BERT的课堂行为文本分类方法", "source": "openalex"}
        assert detect_paper_language(paper) == "zh"

    def test_english_title_with_incidental_cjk_is_not_chinese(self):
        """英文标题混入少量汉字时不判为中文（旧阈值 0.15 的误报场景）。"""
        from app.tools.language_router import detect_paper_language

        paper = {"title": "A survey of 中文信息处理 techniques", "source": "openalex"}
        assert detect_paper_language(paper) == "en"


class TestLanguageHardFilter:
    """中英文独立硬过滤。"""

    def test_chinese_hard_filter_uses_terms_zh(self):
        from app.tools.language_filter import evaluate_language_hard_filter

        protocol = _bilingual_protocol()
        cn_paper = {
            "title": "基于深度学习的课堂行为识别研究",
            "abstract": "本文研究了课堂行为识别方法",
            "venue": "计算机学报",
        }
        passed, reason = evaluate_language_hard_filter(cn_paper, protocol, "zh")
        assert passed is True
        assert "zh" in reason

    def test_english_hard_filter_uses_terms_en(self):
        from app.tools.language_filter import evaluate_language_hard_filter

        protocol = _bilingual_protocol()
        en_paper = {
            "title": "Deep Learning for Classroom Behavior Recognition",
            "abstract": "We study classroom behavior recognition methods",
            "venue": "CVPR",
        }
        passed, reason = evaluate_language_hard_filter(en_paper, protocol, "en")
        assert passed is True
        assert "en" in reason

    def test_english_paper_not_rejected_by_chinese_term(self):
        """英文论文不应被中文 terms_zh 误杀。"""
        from app.tools.language_filter import evaluate_language_hard_filter

        protocol = _bilingual_protocol()
        # 英文论文标题只有英文，不应该因 terms_zh 中的"课堂行为"而失败
        en_paper = {
            "title": "Deep Learning for Classroom Behavior Recognition",
            "abstract": "We propose a method for recognizing student behaviors in classroom settings.",
            "venue": "CVPR",
        }
        passed, reason = evaluate_language_hard_filter(en_paper, protocol, "en")
        assert passed is True, f"英文论文不应被中文 term 误杀: {reason}"

    def test_chinese_paper_rejected_when_no_match(self):
        from app.tools.language_filter import evaluate_language_hard_filter

        protocol = _bilingual_protocol()
        # 一篇完全不相关的论文（非课堂非行为）
        cn_paper = {
            "title": "基于深度学习的图像分割算法研究",
            "abstract": "本文研究了图像分割算法",
            "venue": "计算机学报",
        }
        passed, reason = evaluate_language_hard_filter(cn_paper, protocol, "zh")
        assert passed is False

    def test_fallback_to_terms_when_terms_zh_empty(self):
        from app.tools.language_filter import evaluate_language_hard_filter

        protocol = {
            "version": "1.0",
            "corpus_goal": "test",
            "hard_include_criteria": [
                {
                    "criterion_id": "test",
                    "label": "测试",
                    "terms": ["课堂行为", "classroom behavior"],
                    "terms_zh": [],  # 空 → 回退到 terms
                    "terms_en": [],
                    "source": "user_explicit",
                    "applies_to_each_paper": True,
                },
            ],
            "hard_exclude_title_terms": [],
        }
        cn_paper = {"title": "课堂行为分析研究", "abstract": ""}
        passed, _ = evaluate_language_hard_filter(cn_paper, protocol, "zh")
        assert passed is True


class TestBranchMerge:
    """分支合并、归一化、配额、跨语言去重。"""

    def test_percentile_normalization(self):
        from app.tools.branch_merge import normalize_scores_by_percentile

        papers = [
            {"_branch_final_score": 0.5, "id": "a"},
            {"_branch_final_score": 1.0, "id": "b"},
            {"_branch_final_score": 0.0, "id": "c"},
        ]
        result = normalize_scores_by_percentile(papers)
        percs = {p["id"]: p["_branch_percentile"] for p in result}
        assert percs["c"] == 0.0  # 最低分
        assert percs["b"] == 1.0  # 最高分
        assert 0.4 < percs["a"] < 0.6  # 中间分

    def test_percentile_single_paper(self):
        from app.tools.branch_merge import normalize_scores_by_percentile

        papers = [{"_branch_final_score": 0.7}]
        result = normalize_scores_by_percentile(papers)
        assert result[0]["_branch_percentile"] == 1.0

    def test_calculate_branch_targets(self):
        from app.tools.branch_merge import calculate_branch_targets

        zh_target, en_target = calculate_branch_targets(
            top_k=60, zh_ratio=0.40, zh_count=100, en_count=200,
        )
        assert zh_target == 24  # 60 * 0.4
        assert en_target == 36  # 60 - 24

    def test_calculate_branch_targets_respects_available_count(self):
        from app.tools.branch_merge import calculate_branch_targets

        # 中文只有 5 篇可用，目标被限制
        zh_target, en_target = calculate_branch_targets(
            top_k=60, zh_ratio=0.40, zh_count=5, en_count=200,
        )
        assert zh_target == 5
        assert en_target == 55

    def test_calculate_branch_targets_never_exceeds_small_top_k(self):
        from app.tools.branch_merge import calculate_branch_targets

        zh_target, en_target = calculate_branch_targets(
            top_k=10, zh_ratio=0.40, zh_count=100, en_count=100,
        )

        assert (zh_target, en_target) == (4, 6)
        assert zh_target + en_target == 10

    def test_cross_language_dedup_by_doi(self):
        from app.tools.branch_merge import global_cross_language_deduplicate

        zh = [
            {
                "doi": "10.1234/same-paper", "title": "课堂行为分析",
                "_merge_score": 0.9,
            },
        ]
        en = [
            {
                "doi": "10.1234/same-paper", "title": "Classroom Behavior Analysis",
                "abstract": "Full abstract", "authors": ["A. Author"], "year": 2024,
                "venue": "TPAMI", "citation_count": 10,
                "_merge_score": 0.5,
            },
        ]
        clean_zh, clean_en = global_cross_language_deduplicate(zh, en)
        # 同一 DOI 保留元数据更完整的记录，与 merge_score 高低无关
        assert clean_zh == []
        assert [p["title"] for p in clean_en] == ["Classroom Behavior Analysis"]

    def test_percentile_normalization_covers_quality(self):
        from app.tools.branch_merge import normalize_scores_by_percentile

        papers = [
            {"_branch_final_score": 0.5, "_quality_score": 0.2, "id": "a"},
            {"_branch_final_score": 1.0, "_quality_score": 0.9, "id": "b"},
            {"_branch_final_score": 0.0, "_quality_score": 0.5, "id": "c"},
        ]
        result = normalize_scores_by_percentile(papers)
        percs = {p["id"]: p["_quality_percentile"] for p in result}
        assert percs["a"] == 0.0  # 质量分最低
        assert percs["c"] == 0.5
        assert percs["b"] == 1.0  # 质量分最高

    def test_merge_score_blocks_absolute_quality_bias_across_branches(self):
        from app.tools.branch_merge import merge_language_branches

        # 两分支相关度排序一致，中文分支元数据普遍稀疏（质量分低 0.4）。
        # 若质量分不做分支内归一化，同排名英文论文会系统性高出 0.08。
        zh_papers = [
            {"paper_id": f"zh:{i}", "_branch_final_score": 1.0 - i * 0.1,
             "_rank_score": 0.9 - i * 0.1, "_quality_score": 0.4}
            for i in range(12)
        ]
        en_papers = [
            {"paper_id": f"en:{i}", "_branch_final_score": 1.0 - i * 0.1,
             "_rank_score": 0.9 - i * 0.1, "_quality_score": 0.8}
            for i in range(12)
        ]

        result = merge_language_branches(
            zh_papers, en_papers, top_k=24, zh_ratio=0.5,
        )
        scores = {p["paper_id"]: p["_merge_score"] for p in result}
        # 分支内同排名论文的 merge_score 应对齐，不受绝对质量分差影响
        assert abs(scores["zh:0"] - scores["en:0"]) < 0.02
        assert abs(scores["zh:5"] - scores["en:5"]) < 0.02

    def test_identifier_dedup_keeps_more_complete_record_without_language_bias(self):
        from app.tools.branch_merge import identifier_level_cross_dedup

        zh = [{
            "paper_id": "zh:rich", "doi": "10.1/same", "title": "完整记录",
            "authors": ["作者"], "year": 2025, "venue": "期刊", "abstract": "完整摘要",
        }]
        en = [{"paper_id": "en:sparse", "doi": "10.1/same", "title": "Sparse"}]

        clean_zh, clean_en = identifier_level_cross_dedup(zh, en)
        assert [paper["paper_id"] for paper in clean_zh] == ["zh:rich"]
        assert clean_en == []

    def test_merge_language_branches_basic(self):
        from app.tools.branch_merge import merge_language_branches

        zh_papers = [
            {"paper_id": "zh:1", "title": f"中文论文{i}", "_branch_final_score": 0.9 - i * 0.05,
             "_rank_score": 0.9 - i * 0.05, "_quality_score": 0.7}
            for i in range(20)
        ]
        en_papers = [
            {"paper_id": "en:1", "title": f"English Paper {i}", "_branch_final_score": 0.95 - i * 0.05,
             "_rank_score": 0.95 - i * 0.05, "_quality_score": 0.8}
            for i in range(40)
        ]

        result = merge_language_branches(
            zh_papers, en_papers, top_k=30, zh_ratio=0.4,
        )
        assert 0 < len(result) <= 30
        # 应同时包含中英文论文
        zh_in_result = sum(1 for p in result if str(p["paper_id"]).startswith("zh:"))
        en_in_result = sum(1 for p in result if str(p["paper_id"]).startswith("en:"))
        assert zh_in_result > 0, "应有中文论文"
        assert en_in_result > 0, "应有英文论文"

    def test_merge_language_branches_empty_branch(self):
        from app.tools.branch_merge import merge_language_branches

        # 中文分支为空时，全部由英文补位
        en_papers = [
            {"paper_id": f"en:{i}", "title": f"Paper {i}", "_branch_final_score": 0.9 - i * 0.01,
             "_rank_score": 0.9 - i * 0.01, "_quality_score": 0.7}
            for i in range(50)
        ]
        result = merge_language_branches(
            [], en_papers, top_k=20, zh_ratio=0.4,
        )
        assert len(result) <= 20
        assert all(str(p["paper_id"]).startswith("en:") for p in result)


class TestRankNodeBranchedMode:
    """rank_node 双分支模式端到端测试。"""

    def test_rank_node_branched_with_bilingual_protocol(self):
        """模拟中英文混合候选池，验证 rank_node 双分支模式输出。"""
        protocol = _bilingual_protocol()

        # 混合中英文候选
        candidates = []
        for i in range(15):
            candidates.append({
                "paper_id": f"zh:test:{i}",
                "title": f"基于深度学习的课堂行为识别研究第{i}篇",
                "abstract": "本文研究课堂行为识别方法",
                "authors": ["作者"],
                "year": 2024,
                "venue": "计算机学报",
                "source": "cnki",
                "citation_count": 5,
            })
        for i in range(25):
            candidates.append({
                "paper_id": f"en:test:{i}",
                "title": f"Deep Learning for Classroom Behavior Recognition Study {i}",
                "abstract": "We propose methods for recognizing student behaviors in classroom settings.",
                "authors": ["Author"],
                "year": 2024,
                "venue": "CVPR",
                "source": "arxiv",
                "citation_count": 20,
            })

        state = {
            "candidate_papers": candidates,
            "topic": "课堂行为分析",
            "keywords": ["课堂行为分析", "classroom behavior"],
            "max_papers": 20,
            "retrieval_target": 20,
            "required_concepts": [],
            "excluded_title_terms": [],
            "selected_scope": {},
            "search_branches": [],
            "screening_protocol": protocol,
            "steps": [],
            "errors": [],
        }

        result_state = rank_node(state, llm=None)

        assert result_state.get("ranked_papers") is not None
        assert len(result_state["ranked_papers"]) > 0
        # 不应有错误
        assert not result_state.get("errors")

        report = result_state.get("screening_report", {})
        rule_filter = report.get("rule_filter", {})
        # 双分支模式
        assert rule_filter.get("mode") == "branched"
        branch_stats = rule_filter.get("branch_stats", {})
        assert branch_stats.get("zh_initial") == 15
        assert branch_stats.get("en_initial") == 25


def test_llm_rerank_survives_null_and_non_numeric_scores():
    """回归：LLM 返回 null/非数值评分时不得炸掉整个重排阶段。

    背景：原实现用裸 float(llm_info.get(...))，dict.get 的默认值只在
    键缺失时生效，键存在但值为 null 时 float(None) 抛 TypeError，
    单篇脏数据导致全批重排失败。
    """
    papers = [
        _make_paper(paper_id="dirty", title="Dirty Paper"),
        _make_paper(paper_id="clean", title="Clean Paper"),
    ]

    class DirtyScoreLLM:
        def complete(self, prompt: str, **kwargs) -> str:
            paper_ids = re.findall(r'"paper_id":\s*"(\w+)"', prompt)
            results = []
            for paper_id in paper_ids:
                if paper_id == "dirty":
                    results.append({
                        "paper_id": paper_id,
                        "topic_relevance": None,      # 键存在但为 null
                        "scope_alignment": "high",    # 非数值字符串
                        "decision": "include",
                        "confidence": None,
                    })                                # method_alignment 键缺失
                else:
                    results.append({
                        "paper_id": paper_id,
                        "topic_relevance": 8,
                        "scope_alignment": 8,
                        "method_alignment": 8,
                        "decision": "include",
                        "confidence": 0.9,
                    })
            return json.dumps({"results": results})

    result = llm_rerank_papers(papers, topic="test", llm=DirtyScoreLLM(), top_k=10)

    assert {p["paper_id"] for p in result} == {"dirty", "clean"}
    dirty = next(p for p in result if p["paper_id"] == "dirty")
    # null/非数值按缺省 5 分（0.5）处理，而非崩溃
    assert dirty["_llm_semantic_score"] == pytest.approx(0.5, abs=1e-6)
    clean = next(p for p in result if p["paper_id"] == "clean")
    assert clean["_llm_semantic_score"] > dirty["_llm_semantic_score"]


def test_reserve_backfill_fires_from_papers_outside_candidate_pool():
    """回归：minimum_required 回填安全网此前是死代码。

    原实现要求回填论文有 _topic_relation ∈ {direct, near}，但该字段只在
    LLM 打分循环内赋值，候选池外的论文永远没有，导致候选池被高置信排除
    后回填从未生效。现在应从池外未排除论文按规则分回填。
    """
    # 70 篇：前 60 篇进入候选池（minimum_required=3 → 池上限 60），
    # 后 10 篇留在池外作为回填来源。
    papers = [
        _make_paper(paper_id=f"p{index}", title=f"Paper {index}")
        for index in range(70)
    ]
    for index, paper in enumerate(papers):
        paper["_quality_score"] = index / 100  # 越靠后规则分越高

    class ExcludeAllLLM:
        def complete(self, prompt: str, **kwargs) -> str:
            paper_ids = re.findall(r'"paper_id":\s*"(p\d+)"', prompt)
            return json.dumps({
                "results": [
                    {
                        "paper_id": paper_id,
                        "topic_relevance": 1,
                        "scope_alignment": 1,
                        "method_alignment": 1,
                        "decision": "exclude",
                        "confidence": 0.95,
                        "relation_type": "unrelated",
                    }
                    for paper_id in paper_ids
                ]
            })

    diagnostics: dict = {}
    result = llm_rerank_papers(
        papers,
        topic="test",
        llm=ExcludeAllLLM(),
        top_k=10,
        minimum_required=3,
        rerank_diagnostics=diagnostics,
    )

    # 候选池 60 篇全被高置信排除，回填从池外 10 篇按规则分补足
    # reserve_target = min(top_k, max(3, int(3*1.5+0.5))) = 5 篇
    assert diagnostics["excluded_count"] == 60
    assert diagnostics["reserve_backfilled_count"] == 5
    assert len(result) == 5
    assert all(
        p["_screening_decision"] == "rule_screened_reserve" for p in result
    )
    # 回填的是池外规则分最高的 5 篇（p69..p65）
    assert [p["paper_id"] for p in result] == ["p69", "p68", "p67", "p66", "p65"]


def test_retrieval_loop_keeps_rule_order_when_rerank_fails(monkeypatch):
    """回归：末尾 LLM rerank 异常不得炸掉整条检索链（#4 逃逸口）。

    rerank 只是排序增强，失败时应保留规则粗排顺序并在诊断中记录降级。
    """
    from app.agent import retrieval_loop

    state = {
        "topic": "测试主题",
        "retrieval_target": 5,
        "ranked_papers": [
            _make_paper(paper_id="p1", title="P1"),
            _make_paper(paper_id="p2", title="P2"),
        ],
    }

    monkeypatch.setattr(retrieval_loop, "search_node", lambda state, **k: None)
    monkeypatch.setattr(retrieval_loop, "rank_node", lambda state, llm=None: None)

    def _boom(*args, **kwargs):
        raise RuntimeError("rerank exploded")

    monkeypatch.setattr("app.tools.rank_papers.llm_rerank_papers", _boom)

    retrieval_loop.search_rank_with_refinement(
        state, llm=object(), should_cancel=None, progress_callback=None,
    )

    # 规则粗排顺序原样保留
    assert [p["paper_id"] for p in state["ranked_papers"]] == ["p1", "p2"]
    report = state["screening_report"]
    assert report["llm_rerank"]["mode"] == "rule_order_fallback"
    assert "rerank exploded" in report["llm_rerank"]["error"]


def test_safe_float_degrades_instead_of_crashing_rerank():
    """H7 回归：LLM 打分为 null/非数值时退回默认值，不炸重排阶段。"""
    from app.tools.paper_rerank import _safe_float

    assert _safe_float(None, 5.0) == 5.0
    assert _safe_float("not-a-number") == 0.0
    assert _safe_float([1, 2]) == 0.0
    assert _safe_float("7.5") == 7.5
    assert _safe_float(7) == 7.0


def test_deduplicate_cross_key_closure_collapses_multi_source_records():
    """M14 回归：同一论文经不同键路径进入必须收敛为一条记录。

    A(仅DOI) + B(仅arXiv) + C(DOI+arXiv 双键)：C 出现时折叠 A、B，
    三条跨键记录闭包为一条，不再各占配额/产生双条目。
    """
    from app.utils.deduplicate import deduplicate_papers

    shared_doi = "10.1109/tpami.2024.00001"
    shared_arxiv = "2401.00001"
    papers = [
        {"paper_id": "a", "title": "Few-Shot Action Recognition via Metric Alignment", "doi": shared_doi},
        {"paper_id": "b", "title": "Few-Shot Action Recognition via Metric Alignment", "arxiv_id": shared_arxiv},
        {"paper_id": "c", "title": "Few-Shot Action Recognition via Metric Alignment",
         "doi": shared_doi, "arxiv_id": shared_arxiv},
    ]
    kept = deduplicate_papers(papers)
    assert len(kept) == 1
    # 合并后补全的键让后续身份判定看到同一组 DOI/arXiv
    assert kept[0]["doi"] == shared_doi
    assert kept[0]["arxiv_id"] == shared_arxiv


def test_infer_publication_profile_prefers_retrieval_layer_metadata():
    """M13 回归：出处判定以检索层 source/venue 为准，正文文本不能翻转。"""
    from app.tools.extract_paper_card import _infer_publication_profile

    # crossref 源 + 带 DOI 的正式期刊：即使正文提到 arxiv/conference 也不翻转
    paper = {
        "venue": "IEEE Transactions on Pattern Analysis and Machine Intelligence",
        "doi": "10.1109/tpami.2024.1",
        "source": "crossref",
        "title": "Few-shot action recognition",
    }
    profile = _infer_publication_profile(
        paper, "the arxiv preprint version was presented at a conference workshop",
    )
    assert profile["publication_type"] == "journal_article"
    assert profile["peer_review_status"] == "likely_peer_reviewed"

    # arxiv 源且无出版方 DOI：预印本、未经同行评审
    preprint = _infer_publication_profile(
        {"source": "arxiv", "title": "Same topic", "arxiv_id": "2401.00001"}, ""
    )
    assert preprint["publication_type"] == "preprint"
    assert preprint["peer_review_status"] == "not_peer_reviewed"


def test_arxiv_id_does_not_override_formal_publication():
    """回归：只要有 arxiv_id 就判 preprint，正式出版信息被压过。

    真实参考文献表（logs/fsar_review3.md）里 CVPR / ACM MM / AAAI / IJCV /
    Knowledge-Based Systems 的正式论文被全部标成 [EB/OL]。正式发表的论文
    普遍同时挂着 arXiv 预印本，arxiv_id 只能在没有出版方 DOI 时定罪。
    """
    from app.tools.extract_paper_card import _infer_publication_profile

    # 裸会议名 + 出版方 DOI + arxiv_id：会议论文，不是预印本
    cvpr = _infer_publication_profile({
        "venue": "Computer Vision and Pattern Recognition",
        "doi": "10.1109/CVPR52729.2023.01727",
        "source": "s2",
        "arxiv_id": "2304.00415",
        "title": "MoLo",
    }, "")
    assert cvpr["publication_type"] == "conference_paper"

    # ACM Multimedia 同样不含 conference 字面词
    assert _infer_publication_profile({
        "venue": "ACM Multimedia", "doi": "10.1145/3581783.3612221",
        "source": "s2", "arxiv_id": "2307.00001", "title": "X",
    }, "")["publication_type"] == "conference_paper"

    # 期刊仍是期刊：Pattern Recognition 不能因含 "pattern recognition" 被当成 CVPR
    assert _infer_publication_profile({
        "venue": "Pattern Recognition", "doi": "10.1016/j.patcog.2023.110110",
        "source": "crossref", "title": "P",
    }, "")["publication_type"] == "journal_article"
    assert _infer_publication_profile({
        "venue": "International Journal of Computer Vision",
        "doi": "10.1007/s11263-023-01917-4", "source": "s2",
        "arxiv_id": "2303.00001", "title": "Z",
    }, "")["publication_type"] == "journal_article"

    # 真预印本不受影响：arXiv 自有 DOI 前缀
    genuine_preprint = _infer_publication_profile({
        "venue": "arXiv.org", "doi": "10.48550/arXiv.2411.11335",
        "source": "arxiv", "arxiv_id": "2411.11335", "title": "A",
    }, "")
    assert genuine_preprint["publication_type"] == "preprint"
    assert genuine_preprint["peer_review_status"] == "not_peer_reviewed"


def test_arxiv_placeholder_venue_does_not_override_formal_publication():
    """回归：venue 字面为 "arXiv" 的正式论文仍被判 preprint。

    arxiv_client.py 对每条记录硬编码 ``venue="arXiv"``，同时又从
    ``<link title="doi">`` 取回正式发表后回填的出版方 DOI。上一版的出版方
    DOI 守卫只挡住 arxiv_id 分支，平台 venue 分支排在最前直接短路——真实
    运行的 60 条参考文献里 IJCV / CVPR / ICCV / AAAI / TCSVT 共 7 条被标
    成 [EB/OL]。
    """
    from app.tools.extract_paper_card import _infer_publication_profile

    def profile(doi: str) -> dict:
        return _infer_publication_profile({
            "venue": "arXiv", "doi": doi, "source": "arxiv",
            "arxiv_id": "2304.00415", "title": "Few-shot action recognition",
        }, "")

    for doi in ["10.1109/CVPR52688.2022.01932", "10.1109/ICCV51070.2023.00963",
                "10.1609/AAAI.V37I3.25403"]:
        assert profile(doi)["publication_type"] == "conference_paper", doi

    for doi in ["10.1007/s11263-023-01917-4", "10.1109/TCSVT.2023.3287201",
                "10.1016/j.knosys.2024.112539"]:
        result = profile(doi)
        assert result["publication_type"] == "journal_article", doi
        assert result["peer_review_status"] == "likely_peer_reviewed", doi

    # 真预印本不受影响
    assert profile("10.48550/arXiv.2411.11335")["publication_type"] == "preprint"
    assert profile("")["publication_type"] == "preprint"


def test_ssrn_is_preprint_not_journal_article():
    """回归：SSRN 在打分时是 preprint，在参考文献里却是 [J] + 已评审。

    Crossref 给所有 SSRN 预印本套同一个假刊名 "SSRN Electronic Journal"，
    含 "Journal" 字样，只看 venue 会被当成期刊。
    """
    from app.tools.extract_paper_card import _infer_publication_profile

    profile = _infer_publication_profile({
        "venue": "SSRN Electronic Journal",
        "doi": "10.2139/ssrn.5055942",
        "source": "crossref",
        "title": "S",
    }, "")
    assert profile["publication_type"] == "preprint"
    assert profile["peer_review_status"] == "not_peer_reviewed"


def test_degree_theses_are_excluded_from_evidence_pool():
    """学位论文未经期刊/会议同行评审，不计入证据池。

    判定信号有两类：CNKI 学位论文 DOI 的 /d.cnki. 段，以及详情页把培养单位
    连同属性标签塞进 venue 的形态。刊名（南京邮电大学学报）不得误判成培养单位。
    """
    passed, reason = evaluate_document_type_filter(
        {"title": "基于分阶段注意时序对齐的少样本动作识别",
         "doi": "10.27170/d.cnki.gjsuu.2022.001834", "venue": ""}
    )
    assert passed is False
    assert "学位论文" in reason

    assert evaluate_document_type_filter(
        {"title": "小样本动作识别研究",
         "venue": "合肥工业大学安徽省211工程院校教育部直属院校", "doi": ""}
    )[0] is False

    # 刊物与正式出版物照常放行
    assert evaluate_document_type_filter(
        {"title": "少样本动作识别综述", "venue": "南京邮电大学学报(自然科学版)",
         "doi": "10.1/x"}
    )[0] is True
    assert evaluate_document_type_filter(
        {"title": "Few-shot action recognition",
         "venue": "International Journal of Computer Vision", "doi": "10.1007/s11263-1"}
    )[0] is True


def test_thesis_exclusion_is_attributable_in_diagnostics():
    """排除必须记账：篇数不足时可归因到 document_type_filter，不静默缺口。"""
    papers = [
        {"paper_id": "t1", "title": "少样本动作识别的度量优化",
         "venue": "", "doi": "10.27170/d.cnki.gjsuu.2022.001834",
         "year": 2022, "source": "cnki", "abstract": "少样本动作识别研究"},
        {"paper_id": "j1", "title": "Few-shot action recognition survey",
         "venue": "International Journal of Computer Vision", "doi": "10.1007/s11263-1",
         "year": 2024, "source": "s2", "abstract": "few-shot action recognition"},
    ]
    diagnostics: dict = {}
    kept = deduplicate_and_rank(
        papers, "少样本动作识别", 10, filter_diagnostics=diagnostics,
    )
    assert [paper["paper_id"] for paper in kept] == ["j1"]
    assert diagnostics["filtered_by_stage"]["document_type_filter"] == 1


# ============================================================
# 自适应加深筛选（规则粗排后备窗口 + LLM 重排加深）
# ============================================================
# 标题必须互不相似：deduplicate_papers 用 0.85 的 Jaccard 阈值折叠候选，
# 而中文标题按单字切词，只靠序号区分会被判为同一篇（现有双分支测试的
# 15 篇中文候选实际只剩 1 篇）。这里让每篇从一个字符池各取一个互不相同
# 的字，任意两篇的相似度都是 9/11。
_ZH_FIRST_POOL = "甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉戌亥"
_ZH_SECOND_POOL = "金木水火土风雨雷电云雪花月星辰海山林川泽湖泊泉"
_EN_METHOD_POOL = (
    "YOLO", "Transformer", "GCN", "LSTM", "ResNet", "ViT", "MLP", "SVM",
    "BERT", "CLIP", "DETR", "UNet", "RNN", "GAN", "SNN", "KNN", "PCA",
    "ICA", "HMM", "CRF", "SOM", "ELM",
)
_EN_OBJECT_POOL = (
    "engagement", "attention", "questioning", "discussion", "discipline",
    "gesture", "posture", "expression", "handraising", "grouping", "pace",
    "blackboard", "seatwork", "eyecontact", "turntaking", "feedback",
    "scaffolding", "modeling", "prompting", "circling", "waittime", "probing",
)


def _branched_candidate_pool(size: int) -> list[dict]:
    """构造标题互不相似的中英文候选池。"""
    candidates: list[dict] = []
    for index in range(size):
        slot = index % len(_ZH_FIRST_POOL)
        group = (index // len(_ZH_FIRST_POOL)) % len(_ZH_SECOND_POOL)
        candidates.append({
            "paper_id": f"zh:test:{index}",
            "title": f"课堂行为分析{_ZH_FIRST_POOL[slot]}{_ZH_SECOND_POOL[group]}研究",
            "abstract": "本文研究课堂行为识别方法",
            "authors": ["作者"],
            "year": 2024,
            "venue": "计算机学报",
            "source": "cnki",
            "citation_count": 5,
        })
        candidates.append({
            "paper_id": f"en:test:{index}",
            "title": (
                "Classroom Behavior Recognition of "
                f"{_EN_OBJECT_POOL[index % len(_EN_OBJECT_POOL)]} with "
                f"{_EN_METHOD_POOL[(index // len(_EN_METHOD_POOL)) % len(_EN_METHOD_POOL)]}"
            ),
            "abstract": "We recognize student behaviors in classroom settings.",
            "authors": ["Author"],
            "year": 2024,
            "venue": "CVPR",
            "source": "arxiv",
            "citation_count": 20,
        })
    return candidates


def _branched_state(candidates: list[dict], **overrides) -> dict:
    state = {
        "candidate_papers": candidates,
        "topic": "课堂行为分析",
        "keywords": ["课堂行为分析", "classroom behavior"],
        "max_papers": 72,
        "retrieval_target": 72,
        "required_concepts": [],
        "excluded_title_terms": [],
        "selected_scope": {},
        "search_branches": [],
        "screening_protocol": _bilingual_protocol(),
        "steps": [],
        "errors": [],
    }
    state.update(overrides)
    return state


class _HalfExcludingLLM:
    """paper_id 尾部序号为奇数的 include，偶数的高置信排除（排除率 50%）。

    按尾部数字判定而非绑定某一种 id 形式：单分支夹具用 ``p{i}``，双分支夹具
    用 ``zh:test:{i}`` / ``en:test:{i}``，同一个替身要能服务两种。
    """

    def complete(self, prompt: str, **kwargs) -> str:
        results = []
        for paper_id in re.findall(r'"paper_id":\s*"([^"]+)"', prompt):
            trailing = re.search(r"(\d+)$", paper_id)
            keep = bool(trailing) and int(trailing.group(1)) % 2 == 1
            results.append({
                "paper_id": paper_id,
                "topic_relevance": 8 if keep else 1,
                "scope_alignment": 8 if keep else 1,
                "method_alignment": 8 if keep else 1,
                "decision": "include" if keep else "exclude",
                "confidence": 0.95,
                "relation_type": "direct" if keep else "unrelated",
                "eligible_deliverables": ["research_status"] if keep else [],
            })
        return json.dumps({"results": results})


class _SparseRetainLLM:
    """每批只保留序号最大的一篇，其余高置信排除。"""

    def complete(self, prompt: str, **kwargs) -> str:
        paper_ids = re.findall(r'"paper_id":\s*"(p\d+)"', prompt)
        keep = max(paper_ids, key=lambda pid: int(pid[1:])) if paper_ids else None
        return json.dumps({"results": [
            {
                "paper_id": paper_id,
                "topic_relevance": 8 if paper_id == keep else 1,
                "scope_alignment": 8 if paper_id == keep else 1,
                "method_alignment": 8 if paper_id == keep else 1,
                "decision": "include" if paper_id == keep else "exclude",
                "confidence": 0.95,
                "relation_type": "direct" if paper_id == keep else "unrelated",
                "eligible_deliverables": ["research_status"] if paper_id == keep else [],
            }
            for paper_id in paper_ids
        ]})


def test_adaptive_screening_deepens_window_when_exclusion_rate_high():
    """回归：高排除率下必须继续向尾部加深筛选，而不是拿未筛论文回填。

    2026-09-01 实测 1101 篇候选被规则粗排截到 64 篇、其中 34 篇被高置信
    排除，重排的回填池结构性为空 → 40 篇引用要求只落地 25 篇。
    """
    papers = [
        _make_paper(paper_id=f"p{index}", title=f"Classroom Behavior Study {index}")
        for index in range(128)
    ]
    diagnostics: dict = {}

    result = llm_rerank_papers(
        papers,
        topic="课堂行为分析",
        llm=_HalfExcludingLLM(),
        top_k=72,
        minimum_required=40,
        rerank_diagnostics=diagnostics,
    )

    # 初始候选池 = min(128, min(max(40*2, 60), 120)) = 80
    # 达标线 = min(top_k=72, max(40, int(40*1.5+0.5)=60)) = 60
    assert diagnostics["reserve_target"] == 60
    # 排除率 50%：80 篇只留 40 篇，必须继续筛到 120 篇（candidate_max）才够 60
    assert diagnostics["candidate_count"] == 120
    assert diagnostics["deepened_batch_count"] == 4
    assert diagnostics["retained_count"] == 60
    assert diagnostics["excluded_count"] == 60
    # 关键：达标靠真实语义筛选，不靠未经 LLM 确认的回填
    assert diagnostics["reserve_backfilled_count"] == 0
    assert len(result) == 60
    assert all(paper["_screening_decision"] == "include" for paper in result)


def test_shortfall_is_reported_not_padded_with_unscreened_papers():
    """加深筛选耗尽整个列表后仍不足 → 如实上报缺口，不拿未筛论文冒充达标。"""
    papers = [
        _make_paper(paper_id=f"p{index}", title=f"Classroom Behavior Study {index}")
        for index in range(100)
    ]
    diagnostics: dict = {}

    result = llm_rerank_papers(
        papers,
        topic="课堂行为分析",
        llm=_SparseRetainLLM(),
        top_k=72,
        minimum_required=40,
        rerank_diagnostics=diagnostics,
    )

    # 每批只留 1 篇 → 一直加深到列表耗尽（100 篇 < candidate_max 120）
    assert diagnostics["candidate_count"] == 100
    assert diagnostics["deepened_batch_count"] == 2
    assert diagnostics["reserve_target"] == 60
    assert diagnostics["retained_count"] == 9
    # 缺口如实暴露：返回数量低于达标线，且没有任何一篇是未筛回填
    assert len(result) == 9
    assert len(result) < diagnostics["reserve_target"]
    assert diagnostics["reserve_backfilled_count"] == 0
    assert not any(
        paper.get("_screening_decision") == "rule_screened_reserve" for paper in result
    )


def test_deepening_stops_when_a_batch_retains_nothing():
    """成本护栏：整批一篇都没留下时不再继续送 LLM，交给回填安全网。

    papers 已按规则分降序，整批被否说明更深的尾部只会更差。
    """
    papers = [
        _make_paper(paper_id=f"p{index}", title=f"Unrelated Study {index}")
        for index in range(70)
    ]

    class ExcludeAllLLM:
        def complete(self, prompt: str, **kwargs) -> str:
            paper_ids = re.findall(r'"paper_id":\s*"(p\d+)"', prompt)
            return json.dumps({"results": [
                {
                    "paper_id": paper_id,
                    "topic_relevance": 1,
                    "scope_alignment": 1,
                    "method_alignment": 1,
                    "decision": "exclude",
                    "confidence": 0.95,
                    "relation_type": "unrelated",
                }
                for paper_id in paper_ids
            ]})

    diagnostics: dict = {}
    llm_rerank_papers(
        papers,
        topic="test",
        llm=ExcludeAllLLM(),
        top_k=10,
        minimum_required=3,
        rerank_diagnostics=diagnostics,
    )

    assert diagnostics["deepened_batch_count"] == 0
    assert diagnostics["candidate_count"] == 60


def test_reserve_k_appends_marked_rule_screened_tail():
    """规则粗排在 top_k 之外附加后备尾部，并让诊断反映加宽后的窗口。"""
    papers = [
        _make_paper(paper_id=str(index), title=f"Paper {index}")
        for index in range(200)
    ]
    diagnostics: dict = {}

    result = deduplicate_and_rank(
        papers, topic="paper", top_k=72, reserve_k=48, filter_diagnostics=diagnostics,
    )

    assert len(result) == 120
    assert not any(paper.get("_rule_screened_reserve") for paper in result[:72])
    assert all(paper.get("_rule_screened_reserve") for paper in result[72:])
    # selected_count 保持"主窗口"语义，后备尾部单独记账
    assert diagnostics["selected_count"] == 72
    assert diagnostics["reserve_selected_count"] == 48
    assert diagnostics["truncated_by_top_k"] == 80


def test_reserve_k_defaults_to_zero_and_keeps_top_k_contract():
    """truncated_by_top_k 护栏：此前全仓库没有任何测试断言这个字段。"""
    papers = [
        _make_paper(paper_id=str(index), title=f"Paper {index}")
        for index in range(200)
    ]
    diagnostics: dict = {}

    result = deduplicate_and_rank(
        papers, topic="paper", top_k=72, filter_diagnostics=diagnostics,
    )

    assert len(result) == 72
    assert diagnostics["selected_count"] == 72
    assert diagnostics["reserve_selected_count"] == 0
    assert diagnostics["truncated_by_top_k"] == 128


def test_screening_reserve_k_only_applies_with_explicit_reference_requirement():
    from app.agent.nodes.retrieval import _screening_reserve_k

    # 无显式引用要求 → 没有"缺口"可言，不加宽窗口
    assert _screening_reserve_k(0, 72, 120) == 0
    assert _screening_reserve_k(40, 72, 120) == 48
    assert _screening_reserve_k(40, 0, 120) == 0
    # 主窗口已不低于重排上限时不留负数后备
    assert _screening_reserve_k(40, 200, 120) == 0


class TestRankNodeScreeningReserve:
    """rank_node 必须把后备窗口交到下游全局重排手里（两种模式都要）。"""

    @staticmethod
    def _single_branch_state(**overrides) -> dict:
        state = {
            "candidate_papers": [
                _make_paper(paper_id=str(index), title=f"Paper {index}")
                for index in range(200)
            ],
            "topic": "paper",
            "keywords": ["paper"],
            "max_papers": 72,
            "retrieval_target": 72,
            "required_concepts": [],
            "excluded_title_terms": [],
            "selected_scope": {},
            "search_branches": [],
            "screening_protocol": {},
            "steps": [],
            "errors": [],
        }
        state.update(overrides)
        return state

    def test_single_branch_pool_exceeds_primary_window(self):
        out = rank_node(
            self._single_branch_state(required_reference_count=40), llm=None,
        )

        ranked = out["ranked_papers"]
        assert len(ranked) == 120
        assert sum(1 for paper in ranked if paper.get("_rule_screened_reserve")) == 48
        rank_step = next(
            step for step in out["steps"] if step.get("step_name") == "rank"
        )
        assert rank_step["output_data"]["screening_reserve_k"] == 48
        assert rank_step["output_data"]["screening_reserve_count"] == 48

    def test_single_branch_pool_stays_at_primary_window_without_requirement(self):
        out = rank_node(self._single_branch_state(), llm=None)

        assert len(out["ranked_papers"]) == 72

    def test_branched_merged_pool_exceeds_primary_window(self):
        """回归：2026-09-01 走的正是双分支模式，合并把证据池压回 64。"""
        out = rank_node(
            _branched_state(
                _branched_candidate_pool(150), required_reference_count=40,
            ),
            llm=None,
        )

        assert not out.get("errors")
        rule_filter = out["screening_report"]["rule_filter"]
        assert rule_filter["mode"] == "branched"
        # 前置条件：候选确实够宽，否则下面的断言会被过滤器变化悄悄放行
        assert rule_filter["branch_stats"]["zh_after_hard_filter"] >= 120
        assert rule_filter["branch_stats"]["en_after_hard_filter"] >= 120
        assert len(out["ranked_papers"]) > 72, "合并窗口必须带上筛选后备段"
        assert len(out["ranked_papers"]) <= 120
        # 后备段走 reserve_k 而不是抬高 top_k：两支各自仍按主窗口配额选 72 篇，
        # 再各追加 48 篇带标记的尾部，合并后一起交到下游全局重排手里。
        rank_step = next(
            step for step in out["steps"] if step.get("step_name") == "rank"
        )
        assert rank_step["output_data"]["screening_reserve_k"] == 48
        assert rank_step["output_data"]["screening_reserve_count"] == 96
        assert rank_step["output_data"]["rule_selected_count"] == 144
        assert rule_filter["zh_filter"]["selected_count"] == 72
        assert rule_filter["en_filter"]["selected_count"] == 72

    def test_branched_merged_pool_stays_at_primary_window_without_requirement(self):
        out = rank_node(_branched_state(_branched_candidate_pool(150)), llm=None)

        assert len(out["ranked_papers"]) <= 72


class _ExplodingLLM:
    def complete(self, prompt: str, **kwargs) -> str:
        raise RuntimeError("branch rerank exploded")


def test_filter_reasons_aggregates_by_specific_reason():
    """filter_reasons 必须是"原因 → 篇数"，阶段粒度看不出是哪条排除词生效。"""
    papers = [
        _make_paper(
            paper_id=f"bad:{index}",
            title=f"Zero-shot action recognition method {index}",
        )
        for index in range(5)
    ] + [
        _make_paper(
            paper_id=f"good:{index}",
            title=f"Few-shot learning backbone {index}",
        )
        for index in range(5)
    ]
    diagnostics: dict = {}

    result = deduplicate_and_rank(
        papers,
        topic="few-shot learning",
        top_k=10,
        excluded_title_terms=["zero-shot action recognition"],
        filter_diagnostics=diagnostics,
    )

    assert len(result) == 5
    assert diagnostics["filtered_count"] == 5
    assert sum(diagnostics["filter_reasons"].values()) == diagnostics["filtered_count"]
    assert diagnostics["filter_reasons"] == {
        "命中用户明确排除词: zero-shot action recognition": 5,
    }


class TestRankNodeBranchedDiagnostics:
    """双分支模式必须把逐分支诊断带出来，而不是只留 branch_stats 里手抄的一个计数。"""

    def test_branched_rule_filter_keeps_per_branch_diagnostics(self):
        out = rank_node(
            _branched_state(
                _branched_candidate_pool(150), required_reference_count=40,
            ),
            llm=None,
        )

        assert not out.get("errors")
        rule_filter = out["screening_report"]["rule_filter"]
        assert rule_filter["mode"] == "branched"
        zh_filter = rule_filter["zh_filter"]
        en_filter = rule_filter["en_filter"]
        assert zh_filter and en_filter
        # 一级字段是两支之和；此前双分支模式这些键全部缺失，步骤输出只有 0。
        assert rule_filter["passed_hard_filters"] == (
            zh_filter["passed_hard_filters"] + en_filter["passed_hard_filters"]
        )
        assert rule_filter["passed_hard_filters"] == (
            rule_filter["branch_stats"]["zh_after_hard_filter"]
            + rule_filter["branch_stats"]["en_after_hard_filter"]
        )
        assert rule_filter["truncated_by_top_k"] == (
            zh_filter["truncated_by_top_k"] + en_filter["truncated_by_top_k"]
        )

        output = next(
            step for step in out["steps"] if step.get("step_name") == "rank"
        )["output_data"]
        assert output["passed_hard_filters"] == rule_filter["passed_hard_filters"]
        assert output["truncated_by_top_k"] == rule_filter["truncated_by_top_k"]
        assert output["rule_selected_count"] == (
            zh_filter["selected_count"] + en_filter["selected_count"]
        )

    def test_branched_rerank_diagnostics_are_merged_and_promoted(self):
        out = rank_node(
            _branched_state(_branched_candidate_pool(30), required_reference_count=40),
            llm=_HalfExcludingLLM(),
        )

        assert not out.get("errors")
        branches = out["screening_report"]["llm_rerank"]["branches"]
        assert branches["zh"]["candidate_count"] > 0
        assert branches["en"]["candidate_count"] > 0

        output = next(
            step for step in out["steps"] if step.get("step_name") == "rank"
        )["output_data"]
        assert output["rerank_candidate_count"] == (
            branches["zh"]["candidate_count"] + branches["en"]["candidate_count"]
        )
        assert output["rerank_excluded_count"] == (
            branches["zh"]["excluded_count"] + branches["en"]["excluded_count"]
        )
        # _HalfExcludingLLM 高置信排除一半，这个数必须是正的才证明诊断真的接上了
        assert output["rerank_excluded_count"] > 0

    def test_branch_rerank_batch_degradation_is_visible_in_diagnostics(self):
        """LLM 每批失败时 paper_rerank 内部降级，降级篇数必须能看见。"""
        out = rank_node(
            _branched_state(_branched_candidate_pool(20)), llm=_ExplodingLLM(),
        )

        branches = out["screening_report"]["llm_rerank"]["branches"]
        assert branches["zh"]["screening_degraded_count"] == 20
        assert branches["en"]["screening_degraded_count"] == 20
        output = next(
            step for step in out["steps"] if step.get("step_name") == "rank"
        )["output_data"]
        assert output["rerank_screening_degraded_count"] == 40
        assert out["ranked_papers"], "重排降级必须保留论文，不能清空论文池"

    def test_branch_rerank_total_failure_is_recorded_not_swallowed(
        self, monkeypatch,
    ):
        def _boom(*args, **kwargs):
            raise RuntimeError("branch rerank exploded")

        monkeypatch.setattr("app.tools.rank_papers.llm_rerank_papers", _boom)
        out = rank_node(
            _branched_state(_branched_candidate_pool(20)), llm=object(),
        )

        branches = out["screening_report"]["llm_rerank"]["branches"]
        assert branches["zh"]["mode"] == "rule_order_fallback"
        assert branches["en"]["mode"] == "rule_order_fallback"
        assert "branch rerank exploded" in branches["zh"]["error"]
        assert out["ranked_papers"], "重排失败必须退回规则顺序，不能清空论文池"
