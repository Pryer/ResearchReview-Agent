"""Research state invariants checked at write/recovery boundaries.

The checks are intentionally observational and deterministic: they do not infer
missing metadata or repair state silently. Callers decide whether a violation
should quarantine a draft or trigger a recovery action.
"""

from __future__ import annotations

from typing import Any


_HARD_GAP_CODES = {
    "claim_evidence_quality_not_met",
    "route_validation_failed",
    "taxonomy_not_ready",
    "minimum_cited_references_not_met",
    "minimum_references_not_met",
    "language_coverage_not_met",
}


def _diagnostic_source_health(state: dict[str, Any]) -> str:
    statuses = []
    for item in state.get("source_diagnostics") or []:
        if isinstance(item, dict):
            status = item.get("status")
        else:
            status = getattr(item, "status", None)
        if status:
            statuses.append(str(status).strip().lower())
    if not statuses:
        return "unknown"
    if any(status in {"failed", "human_action_required"} for status in statuses):
        return "partial" if any(status == "success" for status in statuses) else "unavailable"
    if any(status == "empty" for status in statuses):
        return "partial" if any(status == "success" for status in statuses) else "empty"
    return "healthy" if any(status == "success" for status in statuses) else "unknown"


def is_stale_evidence_snapshot(report: dict[str, Any], state: dict[str, Any]) -> bool:
    """派生报告是否落后于当前证据快照。

    报告或状态任一侧未带版本号时返回 False：历史会话没有该字段，不能因为
    缺少版本就把它判成陈旧并丢弃。
    """
    if not report:
        return False
    current_version = state.get("evidence_snapshot_version")
    report_version = report.get("evidence_snapshot_version")
    if current_version is None or report_version is None:
        return False
    if int(report_version or 0) != int(current_version or 0):
        return True
    current_fingerprint = str(state.get("evidence_snapshot_fingerprint") or "")
    report_fingerprint = str(report.get("evidence_snapshot_fingerprint") or "")
    return bool(
        current_fingerprint
        and report_fingerprint
        and current_fingerprint != report_fingerprint
    )


def validate_research_state_invariants(state: dict[str, Any]) -> dict[str, Any]:
    """Return deterministic blocking violations and non-blocking warnings.

    Missing optional snapshots are not violations, which keeps this validator
    usable at early planning stages. Once a snapshot exists, its version and
    fingerprint must agree with the current evidence state.
    """
    blocking: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    request = state.get("research_request") or {}
    top_start = state.get("start_year")
    top_end = state.get("end_year")
    request_start = request.get("start_year")
    request_end = request.get("end_year")
    if (
        top_start is not None
        and request_start is not None
        and int(top_start) != int(request_start)
    ) or (
        top_end is not None
        and request_end is not None
        and int(top_end) != int(request_end)
    ):
        blocking.append({
            "code": "state_time_window_mismatch",
            "message": "研究请求与顶层年份窗口不一致，不能在陈旧时间解释上写作",
            "details": {
                "request": {"start_year": request_start, "end_year": request_end},
                "state": {"start_year": top_start, "end_year": top_end},
            },
        })

    gap = state.get("evidence_gap_report") or {}
    current_version = state.get("evidence_snapshot_version")
    current_fingerprint = str(state.get("evidence_snapshot_fingerprint") or "")
    if is_stale_evidence_snapshot(gap, state):
        blocking.append({
            "code": "stale_evidence_snapshot",
            "message": "证据缺口诊断不是当前证据快照，旧诊断不能驱动写作",
            "details": {
                "state_version": current_version,
                "gap_version": gap.get("evidence_snapshot_version"),
                "state_fingerprint": current_fingerprint,
                "gap_fingerprint": str(gap.get("evidence_snapshot_fingerprint") or ""),
            },
        })

    # WHY: 门禁与缺口报告消费同一份证据，但过去只有缺口报告带快照版本。证据恢复
    # 轮会 SPLIT 路线并改写 validated_routes，陈旧门禁的 route_stats 于是继续描述
    # 拆分前的路线论域，而 derive_result_status 仍按它判定 success/partial。
    gate = state.get("global_evidence_gate") or {}
    if gate.get("status") == "EVALUATED" and is_stale_evidence_snapshot(gate, state):
        blocking.append({
            "code": "stale_global_evidence_gate",
            "message": "全局证据门禁不是当前证据快照，旧门禁不能决定结果状态",
            "details": {
                "state_version": current_version,
                "gate_version": gate.get("evidence_snapshot_version"),
                "state_fingerprint": current_fingerprint,
                "gate_fingerprint": str(gate.get("evidence_snapshot_fingerprint") or ""),
            },
        })

    if gap.get("needs_recovery") is True and current_version is not None:
        readiness = state.get("generation_readiness") or {}
        ready = readiness.get("ready") is True
        issue_codes = {
            str(item.get("code") or "")
            for item in (readiness.get("blocking_issues") or [])
            if isinstance(item, dict)
        }
        if ready and not (_HARD_GAP_CODES & issue_codes):
            blocking.append({
                "code": "recovery_readiness_conflict",
                "message": "证据缺口仍需恢复，但生成状态无结构化解释地标记为 ready",
                "details": {"needs_recovery": True, "readiness": readiness},
            })

    # WHY: 门禁的 route_stats 必须描述当前 validated_routes 里的路线。恢复轮 SPLIT
    # 路线后若门禁未重算，它会继续引用已不存在的父路线 ID，此时 route_balance_ratio
    # 等指标与当前路线论域无关。历史会话的门禁没有快照版本，故只告警不阻断。
    gate_route_ids = {
        str(stat.get("route_id") or "")
        for stat in ((gate.get("metrics") or {}).get("route_stats") or [])
        if isinstance(stat, dict)
    }
    gate_route_ids.discard("")
    current_route_ids = {
        str(route.get("route_id") or "")
        for route in state.get("validated_routes") or []
        if isinstance(route, dict)
    }
    current_route_ids.discard("")
    orphan_gate_routes = sorted(gate_route_ids - current_route_ids)
    if gate_route_ids and current_route_ids and orphan_gate_routes:
        warnings.append({
            "code": "gate_route_universe_mismatch",
            "message": "全局证据门禁统计了当前路线中不存在的路线，指标与本轮路线论域不一致",
            "details": {
                "gate_only_routes": orphan_gate_routes,
                "gate_routes": sorted(gate_route_ids),
                "current_routes": sorted(current_route_ids),
            },
        })

    reported_health = str(gap.get("source_health") or "").strip().lower()
    actual_health = _diagnostic_source_health(state)
    if reported_health and reported_health not in {"unknown", actual_health}:
        warnings.append({
            "code": "source_health_snapshot_mismatch",
            "message": "来源健康度与本轮来源诊断不一致，以下游诊断为准",
            "details": {"reported": reported_health, "actual": actual_health},
        })

    return {
        "valid": not blocking,
        "blocking_issues": blocking,
        "warnings": warnings,
        "source_health": actual_health,
    }
