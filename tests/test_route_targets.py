"""路线级证据目标口径测试。

目标篇数必须从交付物类型和证据角色派生，不能沿用一个全局常数，
也不能与 route_validator 自身的充分性判定重复。
"""

from __future__ import annotations

from app.agent.route_targets import derive_route_core_targets, route_diversity_deficits


def _state(**kwargs) -> dict:
    base = {
        "core_deliverables": ["research_status"],
        "required_reference_count": 40,
        "validated_routes": [
            {"route_id": "R1", "name": "学生行为识别", "core_paper_ids": ["p1"]},
            {"route_id": "R2", "name": "师生互动分析", "core_paper_ids": []},
        ],
        "route_decisions": [
            {"route_id": "R1"},
            {"route_id": "R2"},
        ],
    }
    base.update(kwargs)
    return base


def test_non_route_deliverable_gets_no_extra_target():
    """研究背景不走路线体系，不额外设目标，行为与上线前一致。"""
    targets = derive_route_core_targets(_state(core_deliverables=["research_background"]))
    assert targets == {}


def test_research_status_target_scales_with_requested_reference_count():
    """研究现状按用户要求的引用篇数在路线间分摊，而不是固定 3 篇。"""
    targets = derive_route_core_targets(_state())
    assert set(targets) == {"R1", "R2"}
    # 40 篇 × 0.85 / 2 条路线 = 17 → 被 target_max=12 钳制
    assert all(value == 12 for value in targets.values())


def test_small_request_falls_back_to_floor_not_below():
    """小规模请求不得把目标压到判定阈值以下。"""
    targets = derive_route_core_targets(_state(required_reference_count=4))
    assert all(value >= 3 for value in targets.values())


def test_competing_work_route_target_exceeds_prior_work_route():
    """相关工作中竞争工作缺失会让比较失效，其目标应高于前置工作。"""
    state = _state(
        core_deliverables=["related_work"],
        required_reference_count=12,
        validated_routes=[
            {"route_id": "R1", "name": "前置工作", "core_paper_ids": []},
            {"route_id": "R2", "name": "竞争方法对比", "core_paper_ids": []},
        ],
        route_decisions=[{"route_id": "R1"}, {"route_id": "R2"}],
    )
    targets = derive_route_core_targets(state)
    assert targets["R2"] > targets["R1"]


def test_split_sub_routes_receive_targets():
    """SPLIT 产出的子路线不在 provisional_routes 里，也必须拿到目标。"""
    state = _state(
        validated_routes=[
            {"route_id": "R1_S1", "name": "子路线一", "core_paper_ids": []},
            {"route_id": "R1_S2", "name": "子路线二", "core_paper_ids": []},
        ],
        route_decisions=[{"route_id": "R1_S1"}, {"route_id": "R1_S2"}],
        provisional_framework={"provisional_routes": [{"route_id": "R1"}]},
    )
    targets = derive_route_core_targets(state)
    assert "R1_S1" in targets and "R1_S2" in targets


def test_narrative_review_reports_year_span_deficit_separately():
    """补足篇数不代表能写研究脉络，多样性缺口独立记录。"""
    state = _state(
        core_deliverables=["narrative_review"],
        paper_cards=[
            {"paper_id": "p1", "year": 2025},
            {"paper_id": "p2", "year": 2025},
            {"paper_id": "p3", "year": 2024},
            {"paper_id": "p4", "year": 2026},
        ],
    )
    deficits = route_diversity_deficits(
        state,
        {"R1": ["p1", "p2"], "R2": ["p3", "p4"]},
    )
    assert "R1" in deficits
    assert deficits["R1"]["year_span"] == 1
    assert "R2" not in deficits


def test_diversity_deficit_only_applies_to_narrative_review():
    state = _state(paper_cards=[{"paper_id": "p1", "year": 2025}])
    assert route_diversity_deficits(state, {"R1": ["p1"]}) == {}
