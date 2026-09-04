"""证据池目标回归：绝对目标持久化 + 按实测成品率倒推。

回归 2026-09-01 实测：40 篇显式引用要求下，首轮池目标 60，增量轮
required_to_fetch = max(1, 40-30) = 10 → ceil(10×1.5) = 15，池目标从 60
缩到 15；同时 generation_limit = 40 + ceil(40×0.60) = 64 把任何更大的绝对
目标夹回 64。结果 45 篇证据卡片只产出 25 篇引用，门禁却记达标。
"""

import math

import pytest

from app.agent.nodes.retrieval import (
    absolute_evidence_pool_target,
    evidence_yield_report,
    fetch_detail_node,
)
from app.agent.slot_extractor import _derive_count_targets


# ============================================================
# 成品率观测：两个率必须分别落盘
# ============================================================
def test_yield_report_separates_availability_from_realization():
    """45 卡片 → 32 可用 → 25 引用：两段损耗必须分开可见。"""
    state = {
        "paper_cards": [{"paper_id": f"p{i}"} for i in range(45)],
        "generation_readiness": {"usable_reference_count": 32},
        "unique_cited_paper_count": 25,
    }

    report = evidence_yield_report(state)

    assert report["evidence_availability_rate"] == pytest.approx(32 / 45, abs=1e-4)
    assert report["citation_realization_rate"] == pytest.approx(25 / 32, abs=1e-4)
    assert report["end_to_end_rate"] == pytest.approx(25 / 45, abs=1e-4)


@pytest.mark.parametrize("missing", ["paper_cards", "generation_readiness", "unique_cited_paper_count"])
def test_yield_report_requires_a_complete_observation(missing):
    """观测不完整时不得产出成品率，避免用半截数据倒推池目标。"""
    state = {
        "paper_cards": [{"paper_id": f"p{i}"} for i in range(45)],
        "generation_readiness": {"usable_reference_count": 32},
        "unique_cited_paper_count": 25,
    }
    state.pop(missing)

    assert evidence_yield_report(state) == {}


# ============================================================
# 绝对目标：按端到端成品率倒推
# ============================================================
def test_absolute_target_is_derived_from_observed_end_to_end_yield():
    """端到端 0.556 时 40 篇要求需要约 72 篇池。"""
    target = absolute_evidence_pool_target(
        40, {"end_to_end_rate": 25 / 45}, reserve_ratio=0.5
    )

    assert target >= 72


def test_absolute_target_falls_back_to_reserve_ratio_without_observation():
    """首轮无观测时退回配置默认余量，保持既有 60 的行为。"""
    assert absolute_evidence_pool_target(40, {}, reserve_ratio=0.5) == 60


def test_absolute_target_is_floored_against_anomalous_yield():
    """一次异常低的成品率不得把池目标推到无节制规模。"""
    assert absolute_evidence_pool_target(40, {"end_to_end_rate": 0.01}, 0.5) == 200


def test_absolute_target_is_zero_without_a_requirement():
    assert absolute_evidence_pool_target(0, {"end_to_end_rate": 0.5}, 0.5) == 0


# ============================================================
# C1 天花板：绝对目标不得被 generation_limit 夹回 64
# ============================================================
def test_generation_limit_leaves_room_for_the_absolute_target():
    """40 篇显式要求的 generation_limit 必须容得下 72 篇绝对目标。"""
    retrieval_target, generation_limit = _derive_count_targets(40, True)

    assert retrieval_target == generation_limit == 72
    assert generation_limit >= absolute_evidence_pool_target(
        40, {"end_to_end_rate": 25 / 45}, reserve_ratio=0.5
    )


def test_generation_limit_stays_bounded_for_large_requests():
    """抬高余量比例不得解除大额请求的上限。"""
    assert _derive_count_targets(200, True)[0] <= 240


# ============================================================
# 池目标跨轮持久化：增量轮不得缩回增量余量
# ============================================================
def _paper(index: int) -> dict:
    return {
        "paper_id": f"p{index}",
        "title": f"课堂行为分析中的师生互动编码研究 {index}",
        "authors": ["张三"],
        "year": 2024,
        "venue": "电化教育研究",
        "doi": None,
        "url": None,
    }


def _run_fetch_detail(state: dict, ranked_count: int = 90) -> dict:
    from app.tools import fetch_metadata

    state.setdefault("steps", [])
    state["ranked_papers"] = [_paper(index) for index in range(ranked_count)]
    original = fetch_metadata.fetch_batch_details
    fetch_metadata.fetch_batch_details = lambda papers: list(papers)
    try:
        fetch_detail_node(state)
    finally:
        fetch_metadata.fetch_batch_details = original
    return state


def _first_round_state() -> dict:
    return {
        "topic": "课堂行为分析",
        "keywords": ["课堂行为分析"],
        "required_reference_count": 40,
        "max_papers": 40,
        "retrieval_target": 72,
        "generation_limit": 72,
        "selected_scope": {},
        "search_branches": [],
        "screening_protocol": {},
        "research_semantic_frame": {},
    }


def test_first_round_pool_target_keeps_the_reserve():
    state = _run_fetch_detail(_first_round_state())

    assert state["evidence_pool_target"] == 60


def test_incremental_round_does_not_shrink_the_pool_to_the_increment():
    """增量轮 required_to_fetch=10 时，池目标不得从 60 掉到 15。"""
    state = _first_round_state()
    state.update({
        "incremental_retrieval": True,
        "paper_details": [_paper(index) for index in range(30)],
        "generation_readiness": {"usable_reference_count": 30},
        "evidence_pool_target": 60,
    })

    _run_fetch_detail(state)

    assert state["evidence_pool_target"] >= 60


def test_incremental_round_uses_the_observed_yield_and_is_not_clamped():
    """观测到端到端 0.556 后，增量轮池目标升到 72 且不被 generation_limit 夹回。"""
    state = _first_round_state()
    state.update({
        "incremental_retrieval": True,
        "paper_details": [_paper(index) for index in range(30)],
        "paper_cards": [{"paper_id": f"p{i}"} for i in range(45)],
        "generation_readiness": {"usable_reference_count": 32},
        "unique_cited_paper_count": 25,
        "evidence_pool_target": 60,
        "evidence_yield": evidence_yield_report({
            "paper_cards": [{"paper_id": f"p{i}"} for i in range(45)],
            "generation_readiness": {"usable_reference_count": 32},
            "unique_cited_paper_count": 25,
        }),
    })

    _run_fetch_detail(state)

    assert state["evidence_pool_target"] == math.ceil(40 / (25 / 45)) == 72


def test_pool_target_is_recorded_in_the_step_trace():
    """池目标的推导依据必须可诊断，不能只留一个结果数字。"""
    state = _first_round_state()
    state["evidence_yield"] = {"end_to_end_rate": 25 / 45}

    _run_fetch_detail(state)

    step = next(
        item for item in state["steps"] if item.get("step_name") == "fetch_detail"
    )
    assert step["input_data"]["absolute_pool_target"] == 72
    assert step["input_data"]["evidence_yield"] == {"end_to_end_rate": 25 / 45}
    assert step["input_data"]["evidence_pool_target"] == 72
