"""路线级证据目标口径。

``route_min_core_evidence`` 只回答"这条路线是否算充分"，不回答"要补到几篇"。
本模块把后者从交付物类型和证据角色确定性地派生出来，让证据恢复闭环有明确
的 per-route 目标，而不是沿用一个全局常数。

不做任何领域词判断，也不调用 LLM：目标只来自用户显式要求（引用篇数）、
交付物类型和已有的证据角色标签。
"""

from __future__ import annotations

import math
from typing import Any

from app.core.config import get_review_threshold_policy

# 走路线体系的交付物；研究背景按论证角色组织，不在此列。
_ROUTE_BASED_DELIVERABLES = ("research_status", "related_work", "narrative_review")

# 竞争工作缺失会直接让"比较"失去意义，因此目标高于前置/同类工作。
_COMPETING_WORK_MARKERS = ("competing", "competitor", "baseline", "竞争", "对比", "基线")


def _route_ids(state: dict[str, Any]) -> list[str]:
    """收集所有需要目标的路线，包含 SPLIT 产出的子路线。"""
    ids: list[str] = []
    for route in state.get("validated_routes") or []:
        route_id = str(route.get("route_id") or "")
        if route_id:
            ids.append(route_id)
    framework = state.get("provisional_framework") or {}
    for route in framework.get("provisional_routes") or []:
        route_id = str(route.get("route_id") or "")
        if route_id:
            ids.append(route_id)
    for decision in state.get("route_decisions") or []:
        route_id = str(decision.get("route_id") or "")
        if route_id:
            ids.append(route_id)
    return list(dict.fromkeys(ids))


def _route_name(state: dict[str, Any], route_id: str) -> str:
    for source in (
        state.get("validated_routes") or [],
        (state.get("provisional_framework") or {}).get("provisional_routes") or [],
        state.get("route_decisions") or [],
    ):
        for route in source:
            if str(route.get("route_id") or "") == route_id:
                name = str(route.get("name") or route.get("route_name") or "").strip()
                if name:
                    return name
    return ""


def _is_competing_work_route(state: dict[str, Any], route_id: str) -> bool:
    """判断该路线是否承担竞争工作比较职责。

    只读已有标签：路线自身的 evidence_role / route_role，或名称中的显式标记。
    """
    for source in (
        state.get("validated_routes") or [],
        (state.get("provisional_framework") or {}).get("provisional_routes") or [],
    ):
        for route in source:
            if str(route.get("route_id") or "") != route_id:
                continue
            role = " ".join(
                str(route.get(field) or "")
                for field in ("evidence_role", "route_role", "comparison_role")
            ).lower()
            if any(marker in role for marker in _COMPETING_WORK_MARKERS):
                return True
    name = _route_name(state, route_id).lower()
    return any(marker in name for marker in _COMPETING_WORK_MARKERS)


def derive_route_core_targets(state: dict[str, Any]) -> dict[str, int]:
    """按交付物类型派生每条路线的核心证据目标篇数。

    只有走路线体系的交付物才有独立目标。其他情况返回空字典，表示"不额外设
    目标"，路线是否充分完全由 route_validator 的 ``route_min_core_evidence``
    判定，行为与本能力上线前一致。

    WHY: 若在此回落成 ``route_min_core_evidence``，per-route 目标就与验证器
    自身的充分性判定重复，两处口径一旦不同步会互相矛盾。
    """
    policy = get_review_threshold_policy()
    floor = max(1, policy.route_min_core_evidence)
    lower = max(1, policy.route_recovery_target_min)
    upper = max(lower, policy.route_recovery_target_max)
    route_ids = _route_ids(state)
    if not route_ids:
        return {}

    deliverables = {str(item) for item in state.get("core_deliverables") or []}
    active = [item for item in _ROUTE_BASED_DELIVERABLES if item in deliverables]
    if not active:
        return {}

    required = int(
        state.get("required_reference_count")
        or state.get("max_papers")
        or 0
    )
    if required > 0:
        share = max(0.0, min(1.0, policy.route_recovery_status_share))
        derived = math.ceil(required * share / max(1, len(route_ids)))
    else:
        derived = floor
    base = max(floor, lower, derived)
    base = min(base, upper)

    targets: dict[str, int] = {}
    competing_bonus = max(0, policy.route_recovery_competing_work_bonus)
    for route_id in route_ids:
        target = base
        if "related_work" in active and _is_competing_work_route(state, route_id):
            target = min(upper, target + competing_bonus)
        targets[route_id] = max(1, target)
    return targets


def route_diversity_deficits(
    state: dict[str, Any],
    route_core_papers: dict[str, list[str]],
) -> dict[str, dict[str, Any]]:
    """叙述性综述额外要求年份跨度；缺口单独记录，不并入篇数目标。

    WHY: 补到目标篇数但全是同一年的论文仍写不出研究脉络，因此多样性是独立
    维度，不能通过提高篇数目标来代替。
    """
    deliverables = {str(item) for item in state.get("core_deliverables") or []}
    if "narrative_review" not in deliverables:
        return {}
    policy = get_review_threshold_policy()
    min_span = max(1, policy.route_recovery_diversity_min_years)
    years_by_paper = {
        str(card.get("paper_id") or ""): card.get("year")
        for card in state.get("paper_cards") or []
        if card.get("paper_id")
    }
    deficits: dict[str, dict[str, Any]] = {}
    for route_id, paper_ids in route_core_papers.items():
        years = {
            int(years_by_paper[paper_id])
            for paper_id in paper_ids
            if paper_id in years_by_paper and isinstance(years_by_paper[paper_id], int)
        }
        if not years:
            continue
        span = max(years) - min(years) + 1
        if span < min_span:
            deficits[route_id] = {
                "year_span": span,
                "required_year_span": min_span,
                "reason": "叙述性综述需要跨年份证据才能刻画研究脉络",
            }
    return deficits
