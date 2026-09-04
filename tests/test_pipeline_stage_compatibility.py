"""研究链阶段兼容性：上游产物不得替代下游证据。

这些测试全部使用与教育、课堂无关的合成主题（矿物光谱勘探、水文监测），
以确保阶段判定与偏序规则是领域中立的通用编排规则，而不是对某一学科的
词表适配。若某天有人把学科术语写进 ``pipeline_stages``，这些用例仍应通过，
但 ``test_stage_module_contains_no_domain_vocabulary`` 会失败。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.agent.pipeline_stages import (
    STAGE_ORDER,
    annotate_card_stages,
    card_is_stage_agnostic,
    card_stages,
    derive_card_stages,
    normalize_stage,
    route_stage,
    stage_compatible,
    stage_gap_reason,
    stage_rank,
)
from app.agent.route_validator import extract_route_paper_features


# ---------- 合成语义帧：矿物光谱勘探，与教育领域无关 ----------

def _frame() -> dict:
    return {
        "canonical_topic": "矿物光谱勘探的标注规范与成因解释",
        "research_objects": [{"label": "矿物光谱"}],
        "evidence_requirements": [
            {
                "requirement_id": "perception:spectral",
                "label": "光谱感知",
                "evidence_role": "perception",
                "aliases": ["spectral sensor", "光谱传感"],
                "minimum_direct_sources": 1,
            },
            {
                "requirement_id": "structured_coding:protocol",
                "label": "标注规范",
                "evidence_role": "structured_coding",
                "aliases": ["annotation protocol", "标注规范"],
                "minimum_direct_sources": 1,
            },
            {
                "requirement_id": "interpretation:genesis",
                "label": "成因解释",
                "evidence_role": "interpretation",
                "aliases": ["ore genesis", "成矿机理"],
                "minimum_direct_sources": 1,
            },
        ],
    }


def _upstream_card() -> dict:
    """只产出感知/识别结果的上游论文。"""
    return {
        "paper_id": "p_sense",
        "title": "Spectral sensor based mineral detection",
        "abstract": (
            "We propose a spectral sensor recognition pipeline for mineral "
            "detection with improved classification accuracy."
        ),
        "method": "spectral sensor detection network",
    }


def _structured_card() -> dict:
    """产出结构化编码规则的下游论文。"""
    return {
        "paper_id": "p_code",
        "title": "An annotation protocol for mineral survey coding",
        "abstract": (
            "We define a coding scheme with time window, event boundary and "
            "inter-rater reliability for the annotation protocol."
        ),
        "method": "structured observation coding framework",
    }


def _interpretation_card() -> dict:
    return {
        "paper_id": "p_interp",
        "title": "Ore genesis interpretation from survey records",
        "abstract": (
            "We interpret ore genesis mechanisms using structured survey "
            "records and discuss implications for exploration decisions."
        ),
        "method": "成矿机理解释分析",
    }


# ---------- 阶段词汇归一与偏序 ----------

def test_route_and_evidence_vocabularies_map_to_one_axis():
    """路线侧与证据侧两套词汇必须归一到同一条阶段轴。

    历史缺陷：路线用 sensing/formalization/...，卡片用 survey/method/...，
    两套枚举只有 application 偶然重合，导致阶段校验永远匹配不上。
    """
    assert normalize_stage("sensing") == "perception"
    assert normalize_stage("formalization") == "structured_coding"
    assert normalize_stage("analysis") == "analytical_method"
    assert normalize_stage("application") == "interpretation"
    # 规范阶段自身幂等
    for stage in STAGE_ORDER:
        assert normalize_stage(stage) == stage


def test_unknown_vocabulary_imposes_no_constraint():
    """未收录词汇不得被猜成某个阶段，否则会误伤未知学科。"""
    assert normalize_stage("whatever-role") == ""
    assert stage_rank("whatever-role") == -1
    route = {"route_id": "R", "route_role": "whatever-role"}
    assert stage_compatible(route, _upstream_card()) is True


def test_stage_order_is_strictly_increasing():
    ranks = [stage_rank(stage) for stage in STAGE_ORDER]
    assert ranks == sorted(ranks)
    assert len(set(ranks)) == len(ranks)
    assert stage_rank("perception") < stage_rank("structured_coding")
    assert stage_rank("structured_coding") < stage_rank("interpretation")


# ---------- 阶段推导 ----------

def test_stage_derived_from_evidence_requirements_not_keywords():
    """阶段来自语义帧的证据要求判定，而不是标题关键词。"""
    frame = _frame()
    assert sorted(derive_card_stages(_upstream_card(), frame)) == ["perception"]
    assert sorted(derive_card_stages(_structured_card(), frame)) == ["structured_coding"]
    assert "interpretation" in derive_card_stages(_interpretation_card(), frame)


def test_annotate_writes_pipeline_stages_and_preserves_existing():
    cards = [_upstream_card(), _structured_card()]
    annotate_card_stages(cards, _frame())
    assert cards[0]["pipeline_stages"] == ["perception"]
    assert cards[1]["pipeline_stages"] == ["structured_coding"]

    preset = {"paper_id": "x", "title": "t", "pipeline_stages": ["interpretation"]}
    annotate_card_stages([preset], _frame())
    assert preset["pipeline_stages"] == ["interpretation"]


def test_legacy_singular_field_is_still_read():
    """卡片只有单数 evidence_role 时也必须能读出阶段。

    这是历史缺陷的直接护栏：读取方曾假定复数 ``evidence_roles`` 存在。
    """
    assert card_stages({"evidence_role": "application"}) == {"interpretation"}
    assert card_stages({"evidence_roles": ["sensing"]}) == {"perception"}
    assert card_stages({"pipeline_stages": ["structured_coding"]}) == {"structured_coding"}


def test_missing_frame_falls_back_to_card_annotation():
    card = {"paper_id": "p", "title": "t", "evidence_role": "sensing"}
    assert derive_card_stages(card, None) == {"perception"}
    assert derive_card_stages(card, {}) == {"perception"}


# ---------- 兼容性规则 ----------

def test_upstream_paper_cannot_support_downstream_route():
    """核心规则：感知论文不能支撑结构化编码路线。

    ``stage_compatible`` 读取的是已标注的 ``pipeline_stages``，标注由
    ``validate_route_evidence`` 在匹配前统一执行（见 route_validator）。
    这里按生产顺序先标注再判定。
    """
    route = {"route_id": "R1", "name": "标注规范构建", "route_role": "formalization"}
    card = _upstream_card()
    annotate_card_stages([card], _frame())

    assert stage_compatible(route, card) is False
    reason = stage_gap_reason(route, card)
    assert "上游" in reason and "structured_coding" in reason


def test_downstream_paper_supports_upstream_and_same_stage_routes():
    """下游论文可以支撑同级与更上游的路线。"""
    structured = _structured_card()
    annotate_card_stages([structured], _frame())
    assert stage_compatible({"route_role": "formalization"}, structured) is True
    assert stage_compatible({"route_role": "sensing"}, structured) is True


def test_interpretation_paper_supports_all_earlier_stages():
    card = _interpretation_card()
    annotate_card_stages([card], _frame())
    for role in ("sensing", "formalization", "analysis", "application"):
        assert stage_compatible({"route_role": role}, card) is True


def test_survey_evidence_is_exempt_from_stage_order():
    """综述类证据跨阶段概括，不应被偏序规则挡住。"""
    survey = {"paper_id": "s", "title": "A survey", "evidence_role": "survey"}
    assert card_is_stage_agnostic(survey) is True
    assert stage_compatible({"route_role": "formalization"}, survey) is True
    assert stage_compatible({"route_role": "application"}, survey) is True


def test_benchmark_evidence_is_exempt_from_stage_order():
    bench = {"paper_id": "b", "title": "A benchmark", "evidence_role": "benchmark"}
    assert card_is_stage_agnostic(bench) is True
    assert stage_compatible({"route_role": "application"}, bench) is True


def test_undetermined_paper_stage_imposes_no_constraint():
    """无法判定阶段的论文不被阶段规则误杀，仍由词面信号决定。"""
    blank = {"paper_id": "u", "title": "Untyped work"}
    assert card_stages(blank) == set()
    assert stage_compatible({"route_role": "formalization"}, blank) is True


def test_route_stage_reads_route_role():
    assert route_stage({"route_role": "formalization"}) == "structured_coding"
    assert route_stage({}) == ""


# ---------- 接入点：匹配特征不得把上游论文升为核心证据 ----------

def _route_for_matching(role: str) -> dict:
    """构造一条锚点齐备的路线，使词面信号足以判定 core。"""
    return {
        "route_id": "R1",
        "name": "标注规范构建",
        "research_question": "如何为矿物光谱勘探建立标注规范？",
        "route_role": role,
        "core_concepts": ["spectral sensor", "mineral detection"],
        "semantic_anchors": ["spectral sensor", "mineral detection"],
        "method_concepts": ["spectral sensor detection network"],
        "task_anchors": ["mineral detection"],
        "negative_anchors": [],
    }


def test_lexically_strong_upstream_paper_is_demoted_to_supporting():
    """词面高度命中但阶段错位的论文只能是 supporting，不能是 core。

    这是语义漂移的执行点：路线要的是结构化编码产物，论文只有识别输出。
    """
    card = _upstream_card()
    annotate_card_stages([card], _frame())

    same_stage = extract_route_paper_features(_route_for_matching("sensing"), card)
    assert same_stage.stage_compatible is True
    assert same_stage.match_level == "core"

    downstream = extract_route_paper_features(_route_for_matching("formalization"), card)
    assert downstream.stage_compatible is False
    assert downstream.stage_conflict_reason
    assert downstream.match_level == "supporting"
    # 词面信号本身不应被削弱，只有最终等级被降级。
    assert downstream.lexical_anchor_score == same_stage.lexical_anchor_score


def test_match_features_expose_stage_diagnostics():
    card = _upstream_card()
    annotate_card_stages([card], _frame())
    features = extract_route_paper_features(_route_for_matching("formalization"), card)
    assert features.route_stage == "structured_coding"
    assert features.paper_stages == ["perception"]


def test_route_without_declared_stage_keeps_lexical_behaviour():
    """路线未声明阶段时行为不变，保证旧会话不回归。"""
    card = _upstream_card()
    annotate_card_stages([card], _frame())
    route = _route_for_matching("sensing")
    route.pop("route_role")
    features = extract_route_paper_features(route, card)
    assert features.route_stage == ""
    assert features.stage_compatible is True
    assert features.match_level == "core"


# ---------- 领域中立契约 ----------

_DOMAIN_TOKENS = (
    "课堂", "教师", "学生", "师生", "教育", "教学",
    "classroom", "teacher", "student",
    "yolo", "openpose", "flanders",
    "目标检测", "少样本", "行为识别",
)


def test_stage_module_contains_no_domain_vocabulary():
    """阶段模块必须只表达通用产物次序，不得出现任何学科术语。"""
    source = Path("app/agent/pipeline_stages.py").read_text(encoding="utf-8").lower()
    for token in _DOMAIN_TOKENS:
        assert token not in source, f"pipeline_stages 出现领域词: {token}"


@pytest.mark.parametrize(
    "role,expected",
    [
        ("sensing", "perception"),
        ("perception", "perception"),
        ("formalization", "structured_coding"),
        ("coding", "structured_coding"),
        ("analysis", "analytical_method"),
        ("interpretation", "interpretation"),
        ("application", "interpretation"),
    ],
)
def test_vocabulary_mapping_is_exhaustive_for_known_roles(role, expected):
    assert normalize_stage(role) == expected


# ---------- 召回侧：缺失阶段驱动的补检索 ----------

def test_supplemental_queries_add_stage_probes_for_missing_stage():
    """缺口查询必须带上阶段产物特征词，把召回从上游推向缺失的下游。

    仅用「主题 + 别名」时，检索源往往仍返回上游论文——它们同样含主题词。
    """
    from app.agent.focus_coverage import supplemental_focus_queries

    queries = supplemental_focus_queries(
        ["structured_coding:protocol"], "矿物光谱勘探", _frame(),
    )
    assert "矿物光谱勘探 标注规范" in queries
    # 中文别名只搭配中文判据词，英文别名只搭配英文判据词
    assert any("标注规范 编码体系" in q for q in queries)
    assert any("annotation protocol coding scheme" in q for q in queries)
    for query in queries:
        has_cjk = bool(re.search(r"[\u4e00-\u9fff]", query.replace("矿物光谱勘探", "")))
        has_latin = bool(re.search(r"[A-Za-z]", query))
        assert not (has_cjk and has_latin), f"中英混杂查询: {query}"


def test_supplemental_queries_do_not_repeat_alias_as_probe():
    """别名与判据词相同时不得拼出「X X」这类重复查询。"""
    from app.agent.focus_coverage import supplemental_focus_queries

    queries = supplemental_focus_queries(
        ["structured_coding:protocol"], "矿物光谱勘探", _frame(),
    )
    assert "矿物光谱勘探 标注规范 标注规范" not in queries


def test_supplemental_queries_isolate_each_missing_stage():
    """不同阶段的缺口使用各自阶段的判据词，不得互相串用。"""
    from app.agent.focus_coverage import supplemental_focus_queries

    interp = supplemental_focus_queries(
        ["interpretation:genesis"], "矿物光谱勘探", _frame(),
    )
    assert any("成矿机理 作用机制" in q for q in interp)
    # 解释阶段不应带上结构化阶段的判据词
    assert not any("编码体系" in q for q in interp)


def test_stage_probe_terms_are_language_grouped_and_domain_free():
    from app.agent.pipeline_stages import stage_probe_terms

    probes = stage_probe_terms("structured_coding")
    assert set(probes) == {"zh", "en"}
    assert probes["zh"] and probes["en"]
    joined = " ".join(probes["zh"] + probes["en"]).lower()
    for token in _DOMAIN_TOKENS:
        assert token not in joined
    assert stage_probe_terms("unknown-role") == {}


def test_supplemental_queries_fall_back_without_frame():
    """无语义帧时退回「主题 + 缺口」，且过滤纯动词缺口。"""
    from app.agent.focus_coverage import supplemental_focus_queries

    assert supplemental_focus_queries(["某缺口"], "矿物光谱勘探", None) == [
        "矿物光谱勘探 某缺口"
    ]
    assert supplemental_focus_queries(["调研"], "矿物光谱勘探", None) == []


# ---------- 路线阶段推导：LLM 未声明 route_role 时不得整体休眠 ----------

def _unlabeled_routes() -> list[dict]:
    """实测形态：候选路线生成未输出 route_role，5 条路线全为 None。"""
    return [
        {
            "route_id": "R1", "name": "标注规范体系构建",
            "research_question": "如何为勘探记录建立标注规范？",
            "core_concepts": ["annotation protocol"],
        },
        {
            "route_id": "R2", "name": "光谱感知技术应用",
            "research_question": "如何用光谱传感实现自动识别？",
            "core_concepts": ["spectral sensor"],
        },
        {
            "route_id": "R3", "name": "成因解释关联",
            "research_question": "如何把记录关联到成矿机理？",
            "core_concepts": ["ore genesis"],
        },
        {
            "route_id": "R4", "name": "与语义帧无关的路线",
            "research_question": "完全不同的问题",
            "core_concepts": ["unrelated concept"],
        },
    ]


def test_route_stage_inferred_when_llm_omits_route_role():
    """route_role 缺失时必须能从路线文本 + 语义帧别名推导阶段。

    这是阶段约束在真实运行中失效的根因：实测 5 条路线 route_role 全为 None，
    ``stage_compatible`` 因此无条件放行，下游路线仍被上游论文填满。
    """
    from app.agent.pipeline_stages import annotate_route_stages, infer_route_stage

    frame = _frame()
    routes = _unlabeled_routes()
    assert all(route.get("route_role") is None for route in routes)

    assert infer_route_stage(routes[0], frame) == "structured_coding"
    assert infer_route_stage(routes[1], frame) == "perception"
    assert infer_route_stage(routes[2], frame) == "interpretation"
    # 与语义帧无任何别名交集的路线不猜阶段
    assert infer_route_stage(routes[3], frame) == ""

    annotate_route_stages(routes, frame)
    assert routes[0]["pipeline_stage"] == "structured_coding"
    assert "pipeline_stage" not in routes[3]


def test_declared_route_role_takes_precedence_over_inference():
    """LLM 显式声明时以声明为准，不被文本推导覆盖。"""
    from app.agent.pipeline_stages import infer_route_stage

    route = {
        "route_id": "R", "name": "标注规范体系构建",
        "route_role": "sensing",
        "core_concepts": ["annotation protocol"],
    }
    assert infer_route_stage(route, _frame()) == "perception"


def test_inference_picks_most_downstream_stage_on_multi_hit():
    """路线同时含上游手段与下游目标时取最下游阶段。

    路线名常写成「借助X实现Y」，而路线要交付的是最下游产物。
    """
    from app.agent.pipeline_stages import infer_route_stage

    route = {
        "route_id": "R",
        "name": "基于光谱传感的成矿机理解释",
        "research_question": "如何用 spectral sensor 支撑 ore genesis 解释？",
        "core_concepts": ["spectral sensor", "ore genesis"],
    }
    assert infer_route_stage(route, _frame()) == "interpretation"


def test_unlabeled_downstream_route_now_rejects_upstream_paper():
    """端到端：route_role 缺失的下游路线经推导后仍能挡住上游论文。"""
    from app.agent.pipeline_stages import annotate_route_stages

    frame = _frame()
    card = _upstream_card()
    annotate_card_stages([card], frame)
    route = {
        "route_id": "R1", "name": "标注规范体系构建",
        "research_question": "如何为勘探记录建立标注规范？",
        "core_concepts": ["annotation protocol", "spectral sensor"],
        "semantic_anchors": ["spectral sensor", "mineral detection"],
        "method_concepts": ["spectral sensor detection network"],
        "task_anchors": ["mineral detection"],
        "negative_anchors": [],
    }
    # 未标注时阶段约束休眠
    assert stage_compatible(route, card) is True

    annotate_route_stages([route], frame)
    assert route["pipeline_stage"] == "structured_coding"
    assert stage_compatible(route, card) is False

    features = extract_route_paper_features(route, card)
    assert features.route_stage == "structured_coding"
    assert features.match_level == "supporting"


def test_route_inference_without_frame_is_noop():
    from app.agent.pipeline_stages import annotate_route_stages, infer_route_stage

    route = {"route_id": "R", "name": "标注规范体系构建", "core_concepts": ["x"]}
    assert infer_route_stage(route, None) == ""
    annotate_route_stages([route], None)
    assert "pipeline_stage" not in route
