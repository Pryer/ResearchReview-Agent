"""证据感知分类测试。"""

import json

from app.agent.nodes import cluster_node
from app.tools.cluster_papers import cluster_papers, compile_dynamic_taxonomy


class ClusterLLM:
    def complete(self, prompt: str, **kwargs) -> str:
        assert "每篇论文只进入一个主类" in prompt
        return """{
          "clusters": [
            {
              "cluster_name": "课堂观察与行为编码",
              "description": "人工观察量表与编码体系",
              "paper_ids": ["p1", "p1", "invented"],
              "representative_papers": ["p1"]
            },
            {
              "cluster_name": "多模态自动分析",
              "description": "视频、语音与姿态融合",
              "paper_ids": ["p2"],
              "representative_papers": ["p2"]
            }
          ]
        }"""


def test_llm_clusters_are_exclusive_complete_and_do_not_invent_ids():
    cards = [
        {"paper_id": "p1", "title": "Observation Coding"},
        {"paper_id": "p2", "title": "Multimodal Analysis"},
        {"paper_id": "p3", "title": "Uncertain Study"},
    ]
    result = cluster_papers(cards, llm=ClusterLLM(), topic="课堂行为分析")
    clusters = result["clusters"]
    ids = [paper_id for cluster in clusters for paper_id in cluster["paper_ids"]]
    assert sorted(ids) == ["p1", "p2", "p3"]
    assert len(ids) == len(set(ids))
    assert "invented" not in ids
    assert clusters[-1]["cluster_name"] == "其他相关研究"


def test_rule_fallback_uses_explicit_card_fields_without_domain_labels():
    cards = [
        {
            "paper_id": "vision",
            "title": "YOLO Classroom Behavior Detection Framework",
            "method": "We propose a neural object detection model.",
            "study_design": "experiment",
        },
        {
            "paper_id": "ifias",
            "title": "Classroom Interaction Analysis Based on iFIAS",
            "method": "本文基于iFIAS开展课堂观察与行为编码。",
        },
        {
            "paper_id": "review",
            "title": "A Systematic Review of Classroom Behavior Analysis",
        },
    ]
    result = cluster_papers(cards, llm=None, topic="课堂行为分析")
    clusters = result["clusters"]
    category = {
        paper_id: cluster["cluster_name"]
        for cluster in clusters
        for paper_id in cluster["paper_ids"]
    }
    assert category["vision"] == "We propose a neural object detection model"
    assert category["ifias"] == "本文基于iFIAS开展课堂观察与行为编码"
    assert category["review"].endswith("待补充证据")
    assert result["dynamic_taxonomy"]["organizing_principle"] == "method"


def test_rule_fallback_normalizes_generic_study_design_labels():
    cards = [
        {
            "paper_id": "p1",
            "title": "Study One",
            "study_design": "experiment",
        },
        {
            "paper_id": "p2",
            "title": "Study Two",
            "study_design": "真实",
        },
    ]
    result = cluster_papers(cards, llm=None, topic="课堂行为分析")
    names = [cluster["cluster_name"] for cluster in result["clusters"]]
    assert "实验研究" in names
    assert "研究设计待补充证据" in names


def test_rule_fallback_uses_domain_independent_axes_when_llm_is_unavailable():
    cards = [
        {
            "paper_id": "detector",
            "title": "YOLO Classroom Behavior Detection",
            "method": "computer vision object detection",
        },
        {
            "paper_id": "sequence",
            "title": "Lag Sequential Analysis of Teacher Student Interaction",
            "method": "lag sequence coding",
        },
        {
            "paper_id": "ifias",
            "title": "Classroom Interaction Analysis Based on iFIAS",
            "method": "teacher student interaction and classroom observation",
        },
    ]
    taxonomy, validation = compile_dynamic_taxonomy(
        cards,
        llm=None,
        topic="课堂行为分析",
        scope={
            "scope_id": "technology_assisted_domain_analysis",
            "description": "先自动识别和行为编码，再做教育学分析",
        },
    )
    assigned_ids = {assignment.paper_id for assignment in taxonomy.assignments}
    assert assigned_ids == {"detector", "sequence", "ifias"}
    assert all(
        name not in {"计算与自动化分析", "观察、编码与质性分析", "综述与证据综合"}
        for name in (theme.name for theme in taxonomy.themes)
    )
    assert validation.paper_coverage == 1.0


def test_route_labels_are_semantic_not_raw_modalities():
    cards = [
        {
            "paper_id": "vision",
            "title": "Classroom Behavior Detection with YOLO and OpenPose",
            "method": "video detection and pose estimation",
        },
        {
            "paper_id": "coding",
            "title": "Classroom Interaction Analysis Based on iFIAS",
            "method": "observation coding and interaction analysis",
        },
        {
            "paper_id": "text",
            "title": "A Review of Classroom Behavior Analysis",
            "method": "survey and review",
        },
    ]
    result = cluster_papers(cards, llm=None, topic="课堂行为分析")
    names = [cluster["cluster_name"] for cluster in result["clusters"]]
    assert not any(name in {"image", "video", "text", "pose", "skeleton"} for name in names)
    assert any(
        any(token in name for token in {"识别", "编码", "理解", "应用", "研究", "分析"})
        for name in names
    )


class DynamicTaxonomyLLM:
    calls = 0

    def complete(self, prompt: str, **kwargs) -> str:
        self.calls += 1
        assert '"scope"' in prompt
        return """{
          "organizing_principle": "研究任务与分析路线",
          "rationale": "同时覆盖人工观察和自动分析",
          "themes": [
            {"theme_id": "T1", "name": "课堂观察与互动编码", "description": "人工编码路线",
             "inclusion_criteria": ["使用观察或编码体系"], "exclusion_criteria": ["纯自动检测"],
             "representative_papers": ["p1"]},
            {"theme_id": "T2", "name": "视觉与多模态行为识别", "description": "自动分析路线",
             "inclusion_criteria": ["使用视觉或多模态模型"], "exclusion_criteria": ["纯人工观察"],
             "representative_papers": ["p2"]}
          ],
          "assignments": [
            {"paper_id": "p1", "primary_theme_id": "T1", "confidence": 0.9,
             "rationale": "采用课堂观察编码", "evidence_fields": ["method"]},
            {"paper_id": "p2", "primary_theme_id": "T2", "secondary_theme_ids": ["T1"],
             "confidence": 0.85, "rationale": "使用多模态识别", "evidence_fields": ["method", "data_modalities"]}
          ]
        }"""


def test_dynamic_taxonomy_keeps_scope_assignments_and_validation():
    cards = [
        {"paper_id": "p1", "title": "Observation", "method": "coding"},
        {"paper_id": "p2", "title": "Vision", "method": "multimodal"},
    ]
    taxonomy, validation = compile_dynamic_taxonomy(
        cards,
        llm=DynamicTaxonomyLLM(),
        topic="课堂行为分析",
        scope={"label": "交叉方向"},
    )
    assert taxonomy.scope["label"] == "交叉方向"
    assert taxonomy.organizing_principle == "研究任务与分析路线"
    assert validation.paper_coverage == 1.0
    assert validation.valid is True
    assert {a.paper_id for a in taxonomy.assignments} == {"p1", "p2"}


def test_dynamic_taxonomy_replaces_abnormal_english_theme_names_from_members():
    class AbnormalNameLLM:
        def complete(self, prompt: str, **kwargs) -> str:
            return json.dumps({
                "organizing_principle": "研究路线",
                "themes": [{
                    "theme_id": "T1",
                    "name": "Vision Detection",
                    "description": "课堂互动分析",
                    "inclusion_criteria": ["互动序列编码"],
                }],
                "assignments": [
                    {"paper_id": "p1", "primary_theme_id": "T1"},
                    {"paper_id": "p2", "primary_theme_id": "T1"},
                ],
            })

    cards = [
        {
            "paper_id": "p1",
            "title": "课堂互动序列编码研究",
            "research_problem": "课堂互动序列如何编码",
            "method": "互动序列编码与滞后分析",
        },
        {
            "paper_id": "p2",
            "title": "课堂互动序列分析方法",
            "research_problem": "课堂互动序列如何分析",
            "method": "互动序列编码与转移分析",
        },
    ]

    taxonomy, _ = compile_dynamic_taxonomy(
        cards,
        llm=AbnormalNameLLM(),
        topic="课堂行为分析",
    )

    names = [theme.name for theme in taxonomy.themes if theme.name]
    assert names
    assert "Vision Detection" not in names
    assert any("互动" in name or "序列" in name or "编码" in name for name in names)


def test_cluster_node_removes_fallback_and_fragment_themes_when_count_allows():
    abstract_evidence = {"access_level": "abstract"}
    cards = [
        {
            "paper_id": f"vision-{index}",
            "title": f"Classroom Behavior Detection {index}",
            "method": "computer vision yolo object detection",
            "evidence_state": abstract_evidence,
            "quality_status": "valid",
        }
        for index in range(7)
    ] + [
        {
            "paper_id": "intervention",
            "title": "Classroom Behavior Intervention",
            "method": "behavior intervention trial",
            "evidence_state": abstract_evidence,
            "quality_status": "valid",
        },
        {
            "paper_id": "unknown",
            "title": "Classroom Behavior Study",
            "method": "",
            "evidence_state": abstract_evidence,
            "quality_status": "valid",
        },
    ]
    state = {
        "topic": "课堂行为分析",
        "canonical_topic": "课堂行为分析",
        "paper_cards": cards,
        "paper_details": list(cards),
        "required_reference_count": 7,
        "max_papers_explicit": True,
        "steps": [],
        "errors": [],
    }

    result = cluster_node(state, llm=None)

    assert result["taxonomy_remediation"]["applied"] is True
    assert set(result["taxonomy_remediation"]["excluded_paper_ids"]) == {
        "intervention",
        "unknown",
    }
    assert len(result["paper_cards"]) == 7
    assert result["taxonomy_validation"]["valid"] is True
    assert result["taxonomy_validation"]["requires_revision"] is False
    assert all(cluster["cluster_name"] != "其他相关研究" for cluster in result["clusters"])


def test_taxonomy_remediation_does_not_delete_required_stage_evidence_or_break_minimum():
    from app.agent.nodes import _remediate_invalid_taxonomy
    from app.agent.research_semantic_parser import parse_research_semantics

    frame = parse_research_semantics(
        "学生课堂行为识别研究现状",
        "课堂行为识别",
        llm=None,
    ).model_dump(mode="json")
    cards = [
        {
            "paper_id": "direct", "title": "Student Classroom Behavior Recognition",
            "abstract": "Deep learning recognition and classification of student behavior.",
            "evidence_state": {"access_level": "abstract"}, "quality_status": "valid",
        },
        {
            "paper_id": "related-1", "title": "学生课堂行为研究",
            "evidence_state": {"access_level": "abstract"}, "quality_status": "valid",
        },
        {
            "paper_id": "related-2", "title": "课堂行为识别评价",
            "evidence_state": {"access_level": "abstract"}, "quality_status": "valid",
        },
    ]
    state = {
        "topic": "课堂行为识别", "canonical_topic": "课堂行为识别",
        "paper_cards": cards, "paper_details": list(cards),
        "research_semantic_frame": frame,
        "required_reference_count": 3, "max_papers_explicit": True,
        "steps": [],
    }
    cluster_result = {
        "dynamic_taxonomy": {
            "themes": [{"theme_id": "T_OTHER", "name": "其他相关研究"}],
            "assignments": [{"paper_id": "direct", "primary_theme_id": "T_OTHER"}],
        },
        "taxonomy_validation": {"undersized_theme_ids": ["T_OTHER"], "errors": ["fragment"]},
        "clusters": [],
    }

    result = _remediate_invalid_taxonomy(state, cluster_result)

    assert result is cluster_result
    assert {card["paper_id"] for card in state["paper_cards"]} == {"direct", "related-1", "related-2"}
    assert "taxonomy_remediation" not in state


class LargeTaxonomyLLM:
    received_paper_count = 0
    calls = 0

    def complete(self, prompt: str, **kwargs) -> str:
        self.calls += 1
        marker = "论文样本 JSON：\n" if "论文样本 JSON：" in prompt else "待分配论文批次：\n"
        start = prompt.index(marker) + len(marker)
        end = prompt.index("\n\n", start)
        payload = json.loads(prompt[start:end])
        papers = payload["papers"]
        if "论文样本 JSON：" in prompt:
            self.received_paper_count = len(papers)
            return json.dumps({
                "organizing_principle": "research route",
                "themes": [
                    {
                        "theme_id": "T1",
                        "name": "Vision Detection",
                        "description": "vision detection models",
                        "inclusion_criteria": ["vision detection"],
                    },
                    {
                        "theme_id": "T2",
                        "name": "Observation Coding",
                        "description": "observation coding schemes",
                        "inclusion_criteria": ["observation coding"],
                    },
                ],
            })
        assignments = [
            {
                "paper_id": paper["paper_id"],
                "primary_theme_id": "T1" if "Vision" in paper["title"] else "T2",
                "confidence": 0.9,
                "rationale": "sample assignment",
                "evidence_fields": ["title"],
            }
            for paper in papers
        ]
        return json.dumps({"assignments": assignments})


def test_large_taxonomy_uses_bounded_induction_sample_and_completes_assignments():
    cards = [
        {
            "paper_id": f"p{index}",
            "title": (
                f"Vision Detection Study {index}"
                if index % 2 == 0
                else f"Observation Coding Study {index}"
            ),
        }
        for index in range(80)
    ]
    llm = LargeTaxonomyLLM()

    taxonomy, validation = compile_dynamic_taxonomy(
        cards, llm=llm, topic="课堂行为分析"
    )

    assert llm.received_paper_count == 24
    assert llm.calls == 1
    assert len(taxonomy.assignments) == 80
    assert validation.paper_coverage == 1.0
    assert validation.valid is True


def test_fallback_taxonomy_refines_dominant_technical_theme_at_realistic_scale():
    cards = [
        {
            "paper_id": f"vision-{index}",
            "title": (
                f"Video Temporal Action Recognition {index}"
                if index % 2 == 0
                else f"Multimodal Audio Visual Behavior Detection {index}"
            ),
            "method": "deep learning computer vision",
        }
        for index in range(12)
    ] + [
        {
            "paper_id": f"observation-{index}",
            "title": f"Classroom Observation Coding Study {index}",
            "method": "observation coding scheme qualitative",
        }
        for index in range(8)
    ]

    taxonomy, validation = compile_dynamic_taxonomy(
        cards,
        llm=None,
        topic="课堂行为分析",
        scope={"research_mode": "mixed"},
    )

    names = {theme.name for theme in taxonomy.themes}
    assert taxonomy.source == "evidence_axis_refined"
    assert any("包含“" in name for name in names)
    assert any("不包含“" in name for name in names)
    assert validation.valid is True
    assert validation.largest_theme_ratio <= 0.5
