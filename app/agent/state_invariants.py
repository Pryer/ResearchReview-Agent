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
    gap_version = gap.get("evidence_snapshot_version")
    current_fingerprint = str(state.get("evidence_snapshot_fingerprint") or "")
    gap_fingerprint = str(gap.get("evidence_snapshot_fingerprint") or "")
    if gap and current_version is not None and gap_version is not None:
        if int(gap_version or 0) != int(current_version or 0) or (
            current_fingerprint and gap_fingerprint and current_fingerprint != gap_fingerprint
        ):
            blocking.append({
                "code": "stale_evidence_snapshot",
                "message": "证据缺口诊断不是当前证据快照，旧诊断不能驱动写作",
                "details": {
                    "state_version": current_version,
                    "gap_version": gap_version,
                    "state_fingerprint": current_fingerprint,
                    "gap_fingerprint": gap_fingerprint,
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
