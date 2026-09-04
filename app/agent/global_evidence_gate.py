"""全局证据门：综述级证据充分性评估与推荐。

本模块只测量与推荐，不执行任何恢复动作（v1 冻结范围）：
不自动扩年份、不改 scope、不降低用户约束、不重分配论文。
LLM 不参与任何决策；``evaluate_global_sufficiency`` 只读 state，绝不修改它。
"""

from __future__ import annotations

import math
from typing import Any

from app.agent.slot_extractor import extract_peer_review_requirement
from app.schemas.global_evidence_schema import (
    DeficitSeverity,
    DimensionStatus,
    EvidenceDeficit,
    GlobalAction,
    GlobalEvidenceMetrics,
    GlobalSufficiencyResult,
    RouteEvidenceStat,
    SufficiencyDimension,
)

# 同行评审状态集合：likely_peer_reviewed 视为已评审（来源推断），
# unknown 不计入分母（无可测量的证据），但单独计数供审计。
_PEER_REVIEWED_STATUSES = {"peer_reviewed", "likely_peer_reviewed"}
_KNOWN_PEER_STATUSES = _PEER_REVIEWED_STATUSES | {"not_peer_reviewed"}

# 推荐动作固定优先级；v1 只输出给调用方，不执行。
_ACTION_PRIORITY = (
    GlobalAction.REBALANCE_ROUTE,
    GlobalAction.TARGETED_GLOBAL_SEARCH,
    GlobalAction.ASK_USER,
    GlobalAction.CONTINUE,
)


def evaluate_global_sufficiency(
    state: dict[str, Any],
    *,
    min_recency_ratio: float = 0.9,
    route_balance_min_ratio: float = 0.25,
    peer_review_ratio_threshold: float = 0.8,
) -> dict[str, Any]:
    """评估综述级证据充分性。

    返回 ``GlobalSufficiencyResult.model_dump(mode="json")``。只读 ``state``。
    阻断规则：引用数量/年份范围/同行评审仅在用户**显式**提出对应要求时
    判定为 blocking；隐式默认值只测量（non_blocking）。路线均衡缺口恒为
    blocking（综述结构失衡），但不计入 ``explicit_constraint_unmet``。
    """
    deficits: list[EvidenceDeficit] = []
    notes: list[str] = []
    evaluated_flags: list[bool] = []

    citation_deficits, citation_metrics, citation_evaluated = _evaluate_citation_count(state)
    deficits.extend(citation_deficits)
    evaluated_flags.append(citation_evaluated)

    recency_deficits, recency_metrics, recency_evaluated = _evaluate_recency(
        state, min_recency_ratio
    )
    deficits.extend(recency_deficits)
    evaluated_flags.append(recency_evaluated)

    route_deficits, route_metrics, route_evaluated = _evaluate_route_coverage(
        state, route_balance_min_ratio
    )
    deficits.extend(route_deficits)
    evaluated_flags.append(route_evaluated)

    quality_deficits, quality_metrics, quality_evaluated = _evaluate_quality(
        state, peer_review_ratio_threshold
    )
    deficits.extend(quality_deficits)
    evaluated_flags.append(quality_evaluated)

    claim_metrics = _claim_support_metrics(state)
    if claim_metrics["claim_support_source"] == "claim_plans":
        notes.append(
            "主张级证据强度取自 claim_plans："
            f"{claim_metrics['claim_strong_plus_count']}/{claim_metrics['claim_total_count']}"
            " 条主张具备多篇独立证据；是否阻断由 claim_evidence_gate 在写作前判定。"
        )
    else:
        notes.append(
            "claim_plans 尚未生成，主张强度暂以 KEEP 路线证据体量代替"
            "（该回退值不反映主张交叉验证程度）。"
        )

    metrics = GlobalEvidenceMetrics(
        total_papers=len(state.get("paper_details") or []),
        **citation_metrics,
        **recency_metrics,
        **route_metrics,
        **quality_metrics,
        **claim_metrics,
        evidence_recovery_status=str(state.get("evidence_recovery_status") or ""),
    )

    blocked = [d for d in deficits if d.severity == DeficitSeverity.BLOCKING]
    passed = not blocked
    # 只有源于用户显式约束的阻断缺口才影响结果状态（partial）。
    # 路线均衡是综述质量信号而非用户显式要求，只记录提示。
    explicit_constraint_unmet = any(
        d.severity == DeficitSeverity.BLOCKING
        and d.type
        in {
            SufficiencyDimension.CITATION_COUNT,
            SufficiencyDimension.RECENCY,
            SufficiencyDimension.QUALITY,
        }
        for d in deficits
    )
    evidence_debt = {d.type.value: d.missing for d in deficits}
    recommended_actions = _recommend_actions(
        passed, deficits, state.get("evidence_recovery_status")
    )

    if not any(evaluated_flags):
        status = DimensionStatus.NOT_REQUIRED
    else:
        status = DimensionStatus.EVALUATED

    result = GlobalSufficiencyResult(
        passed=passed,
        status=status,
        explicit_constraint_unmet=explicit_constraint_unmet,
        deficits=deficits,
        evidence_debt=evidence_debt,
        recommended_actions=recommended_actions,
        metrics=metrics,
        notes=notes,
    )
    return result.model_dump(mode="json")


def _evaluate_citation_count(
    state: dict[str, Any],
) -> tuple[list[EvidenceDeficit], dict[str, Any], bool]:
    """引用数量维度：证据池规模 vs 用户要求。显式要求（max_papers_explicit）
    未满足 → blocking；隐式默认目标未达到只作 non_blocking 提示。"""
    required = int(
        state.get("required_reference_count") or state.get("max_papers") or 0
    )
    available = len(state.get("paper_details") or [])
    metrics: dict[str, Any] = {
        "citation_required": required,
        "citation_available": available,
    }
    if required <= 0:
        return [], metrics, False
    missing = max(0, required - available)
    if missing == 0:
        return [], metrics, True
    explicit = bool(state.get("max_papers_explicit"))
    ratio = available / required
    reason = (
        f"用户要求引用不少于 {required} 篇，当前证据池仅 {available} 篇。"
        if explicit
        else f"默认引用目标 {required} 篇，当前证据池仅 {available} 篇。"
    )
    deficit = EvidenceDeficit(
        type=SufficiencyDimension.CITATION_COUNT,
        severity=DeficitSeverity.BLOCKING if explicit else DeficitSeverity.NON_BLOCKING,
        required=required,
        available=available,
        missing=missing,
        ratio=min(ratio, 1.0),
        reason=reason,
    )
    return [deficit], metrics, True


def _evaluate_recency(
    state: dict[str, Any], min_recency_ratio: float
) -> tuple[list[EvidenceDeficit], dict[str, Any], bool]:
    """年份窗口维度：窗口内论文占比。缺失年份视为窗口外（无法证明满足）。
    显式年份范围（year_range_explicit）未满足 → blocking。"""
    start = state.get("start_year")
    end = state.get("end_year")
    details = state.get("paper_details") or []
    total = len(details)
    metrics: dict[str, Any] = {"in_window_papers": 0, "out_window_papers": 0}
    if start is None or end is None or total == 0:
        return [], metrics, False
    in_window = 0
    for paper in details:
        year = paper.get("year")
        if isinstance(year, int) and not isinstance(year, bool) and start <= year <= end:
            in_window += 1
    out_window = total - in_window
    ratio = in_window / total
    metrics = {
        "in_window_papers": in_window,
        "out_window_papers": out_window,
        "recency_ratio": ratio,
    }
    if ratio >= min_recency_ratio:
        return [], metrics, True
    explicit = bool(state.get("year_range_explicit"))
    deficit = EvidenceDeficit(
        type=SufficiencyDimension.RECENCY,
        severity=DeficitSeverity.BLOCKING if explicit else DeficitSeverity.NON_BLOCKING,
        required=total,
        available=in_window,
        missing=out_window,
        ratio=ratio,
        reason=(
            f"仅 {in_window}/{total} 篇论文落在要求的年份范围 {start}-{end} 内。"
        ),
    )
    return [deficit], metrics, True


def _evaluate_route_coverage(
    state: dict[str, Any], route_balance_min_ratio: float
) -> tuple[list[EvidenceDeficit], dict[str, Any], bool]:
    """路线覆盖维度：KEEP 路线证据数量均衡（最少/平均）。

    WEAK 路线已由 Route Validator 判定证据不充分，重复计入会双重计分，
    因此只进入 metrics.weak_route_count，不产生 deficit。
    """
    routes = state.get("validated_routes") or []
    keep_routes = [r for r in routes if r.get("status") == "KEEP"]
    weak_count = sum(1 for r in routes if r.get("status") == "WEAK")
    if len(keep_routes) < 2:
        metrics: dict[str, Any] = {
            "weak_route_count": weak_count,
            "zero_paper_route_count": sum(
                1 for r in keep_routes if not r.get("paper_ids")
            ),
            "route_stats": [
                RouteEvidenceStat(
                    route_id=str(r.get("route_id") or ""),
                    name=str(r.get("name") or ""),
                    status=str(r.get("status") or ""),
                    paper_count=len(r.get("paper_ids") or []),
                    core_paper_count=len(r.get("core_paper_ids") or []),
                    balanced=True,
                )
                for r in keep_routes
            ],
        }
        return [], metrics, False
    counts = [len(r.get("paper_ids") or []) for r in keep_routes]
    avg = sum(counts) / len(counts)
    min_count = min(counts)
    ratio = min_count / avg if avg > 0 else 1.0
    route_stats = [
        RouteEvidenceStat(
            route_id=str(r.get("route_id") or ""),
            name=str(r.get("name") or ""),
            status=str(r.get("status") or ""),
            paper_count=count,
            core_paper_count=len(r.get("core_paper_ids") or []),
            balanced=count >= route_balance_min_ratio * avg,
        )
        for r, count in zip(keep_routes, counts)
    ]
    metrics = {
        "weak_route_count": weak_count,
        "zero_paper_route_count": sum(1 for c in counts if c == 0),
        "min_route_count": min_count,
        "avg_route_count": avg,
        "route_balance_ratio": ratio,
        "route_stats": route_stats,
    }
    if ratio >= route_balance_min_ratio:
        return [], metrics, True
    balanced_count = sum(1 for s in route_stats if s.balanced)
    unbalanced = len(keep_routes) - balanced_count
    deficit = EvidenceDeficit(
        type=SufficiencyDimension.ROUTE_COVERAGE,
        severity=DeficitSeverity.BLOCKING,
        required=len(keep_routes),
        available=balanced_count,
        missing=unbalanced,
        ratio=ratio,
        reason=(
            f"KEEP 路线证据分布失衡：最少路线 {min_count} 篇 vs 平均 {avg:.1f} 篇"
            f"（比值 {ratio:.2f} < {route_balance_min_ratio}）。"
        ),
    )
    return [deficit], metrics, True


def _evaluate_quality(
    state: dict[str, Any], peer_review_ratio_threshold: float
) -> tuple[list[EvidenceDeficit], dict[str, Any], bool]:
    """质量维度：同行评审占比（peer_reviewed/likely_peer_reviewed / 已知状态）。

    仅用户显式提及同行评审/期刊/SCI/EI 时 blocking，否则只测量。
    unknown 状态不计入分母（无可测量证据），单独计数供审计。
    """
    cards = state.get("paper_cards") or []
    known = [c for c in cards if c.get("peer_review_status") in _KNOWN_PEER_STATUSES]
    unknown_count = len(cards) - len(known)
    metrics: dict[str, Any] = {
        "peer_reviewed_count": 0,
        "peer_review_known_count": len(known),
        "peer_review_unknown_count": unknown_count,
    }
    if not known:
        return [], metrics, False
    peer_reviewed = sum(
        1 for c in known if c.get("peer_review_status") in _PEER_REVIEWED_STATUSES
    )
    ratio = peer_reviewed / len(known)
    metrics["peer_reviewed_count"] = peer_reviewed
    metrics["peer_review_ratio"] = ratio
    if ratio >= peer_review_ratio_threshold:
        return [], metrics, True
    required = math.ceil(peer_review_ratio_threshold * len(known))
    missing = max(0, required - peer_reviewed)
    explicit = extract_peer_review_requirement(str(state.get("user_query") or ""))
    deficit = EvidenceDeficit(
        type=SufficiencyDimension.QUALITY,
        severity=DeficitSeverity.BLOCKING if explicit else DeficitSeverity.NON_BLOCKING,
        required=required,
        available=peer_reviewed,
        missing=missing,
        ratio=ratio,
        reason=(
            f"同行评审论文占比 {ratio:.2f}，低于阈值 {peer_review_ratio_threshold}"
            f"{'，用户显式要求同行评审/期刊来源' if explicit else '（未显式要求，仅提示）'}。"
        ),
    )
    return [deficit], metrics, True


def _route_evidence_volume_proxy(state: dict[str, Any]) -> float:
    """回退指标：KEEP 路线 evidence_sufficiency 均值（证据体量，非主张强度）。

    route_validator 的 evidence_sufficiency 有 dict（``{"score": x}``）和
    裸 float 两种形态，均需兼容。

    注意该分数的分母是 ``route_min_core_evidence``（默认 3）这一固定低阈值，
    分子是路线实际论文数，因此任何正常规模的路线都会被 ``min(1.0, ...)``
    削平到 1.0。它只能说明"路线凑够了最低证据数"，不反映主张是否得到交叉
    验证，仅在主张统计缺失时作为兜底。
    """
    scores: list[float] = []
    for route in state.get("validated_routes") or []:
        if route.get("status") != "KEEP":
            continue
        sufficiency = route.get("evidence_sufficiency")
        if isinstance(sufficiency, dict):
            score = sufficiency.get("score")
        else:
            score = sufficiency
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            scores.append(float(score))
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def _claim_support_metrics(state: dict[str, Any]) -> dict[str, Any]:
    """主张级证据强度：strong/established 主张占比。

    直接复用 ``claim_plans`` 已算出的分级计数，使本门禁与 claim_plan /
    claim_evidence_gate 观察同一件事——此前的路线体量代理恒为 1.0，会让
    全局门禁在 227/233 条主张仅有单篇证据时仍显示"健康"。

    ``claim_plans`` 尚未生成（本门禁在其之前也会运行一次）时回退到路线证据
    体量，并用 ``claim_support_source`` 标明来源。
    """
    plans = state.get("claim_plans") or []
    total = 0
    strong_plus = 0
    single_only = 0
    for plan in plans:
        if not isinstance(plan, dict):
            continue
        total += int(plan.get("total_claims") or 0)
        strong_plus += int(plan.get("strong_plus_claims") or 0)
        single_only += int(plan.get("single_evidence_claims") or 0)

    if total > 0:
        return {
            "claim_support_proxy": strong_plus / total,
            "claim_total_count": total,
            "claim_strong_plus_count": strong_plus,
            "claim_single_evidence_count": single_only,
            "claim_support_source": "claim_plans",
        }
    return {
        "claim_support_proxy": _route_evidence_volume_proxy(state),
        "claim_total_count": 0,
        "claim_strong_plus_count": 0,
        "claim_single_evidence_count": 0,
        "claim_support_source": "route_evidence_volume_fallback",
    }


def _recommend_actions(
    passed: bool,
    deficits: list[EvidenceDeficit],
    recovery_status: str | None,
) -> list[GlobalAction]:
    """确定性推荐动作（v1 只推荐不执行）。

    路线失衡 → REBALANCE_ROUTE；其余阻断缺口 → TARGETED_GLOBAL_SEARCH；
    已有阻断缺口且路线级恢复已耗尽 → 追加 ASK_USER（需用户决策）。
    """
    if passed:
        return [GlobalAction.CONTINUE]
    blocking = [d for d in deficits if d.severity == DeficitSeverity.BLOCKING]
    if not blocking:
        # 只有非阻断提示，继续流程，缺口由最终答复呈现。
        return [GlobalAction.CONTINUE]
    actions: set[GlobalAction] = set()
    for deficit in blocking:
        if deficit.type == SufficiencyDimension.ROUTE_COVERAGE:
            actions.add(GlobalAction.REBALANCE_ROUTE)
        else:
            actions.add(GlobalAction.TARGETED_GLOBAL_SEARCH)
    if recovery_status == "EXHAUSTED":
        actions.add(GlobalAction.ASK_USER)
    return sorted(actions, key=_ACTION_PRIORITY.index)
