"""单次补检索与分析结算；角色之间始终返回 Main，不嵌套整条生成链。"""
from __future__ import annotations

from app.agent.execution_budget import consume, check_execution
from app.agent.recovery_loop import (
    _append_recovery_history, _route_core_counts, _refresh_evidence_snapshot,
    _record_route_evidence_deficits,
)
from app.core.config import get_settings


def search_once(state, *, search, rank, should_cancel=None):
    from app.agent.evidence_recovery import targeted_search_kind
    from app.agent.generation_recovery import recovery_action_count
    kind = targeted_search_kind(state)
    if not kind:
        raise ValueError("targeted search is unavailable in current state")
    settings = get_settings()
    decision = state.get("recovery_decision") or {}
    if kind == "route" and (decision.get("action") not in {"TARGETED_SEARCH", "QUERY_FILTER_REVISION"}
                             or not decision.get("queries")):
        raise ValueError("route search requires queries diagnosed by Analysis and selected by Main")
    consume("recovery")
    state["recovery_action_count"] = recovery_action_count(state) + 1
    pending = {
        "kind": kind, "decision": dict(decision),
        "before_card_ids": [str(card.get("paper_id")) for card in state.get("paper_cards") or []],
        "coverage_before": float((state.get("evidence_gap_report") or {}).get("coverage_score") or 0),
        "core_before": _route_core_counts(state),
    }
    if kind == "citation":
        state["citation_gap_repair_attempted"] = True
        state["_citation_gap_repair_previous_cited"] = int(state.get("unique_cited_paper_count") or 0)
        state["citation_shortfall_count"] = max(0, int(state.get("required_reference_count") or 0)
                                                - int(state.get("unique_cited_paper_count") or 0))
    else:
        queries = list(dict.fromkeys(str(q).strip() for q in decision["queries"] if str(q).strip()))
        round_number = int(state.get("recovery_round") or 0) + 1
        state["recovery_round"] = round_number
        attempts = dict(state.get("route_recovery_attempts") or {})
        for route_id in decision.get("affected_route_ids") or []:
            attempts[route_id] = int(attempts.get(route_id) or 0) + 1
        state["route_recovery_attempts"] = attempts
        state["keywords"] = list(dict.fromkeys([*(state.get("keywords") or []), *queries]))
        state.setdefault("search_branches", []).append({
            "branch_type": f"evidence_recovery_{round_number}", "queries": queries,
            "required_concepts": decision.get("missing_constraints") or [],
            "rationale": decision.get("reason") or "route evidence recovery",
            "constraint_level": "targeted_recovery", "affected_route_ids": decision.get("affected_route_ids") or [],
        })
    state["incremental_retrieval"] = True
    state["incremental_search_window"] = {key: state[key] for key in ("start_year", "end_year") if state.get(key) is not None}
    state["incremental_required_new_evidence"] = settings.evidence_recovery_min_new_evidence
    state["last_search_new_results"] = 0
    state["pending_evidence_recovery"] = pending
    check_execution()
    search(state, should_cancel=should_cancel)
    check_execution()
    if int(state.get("last_search_new_results") or 0) > 0:
        rank(state)
    # WHY: 新论文只能提交为候选；详情、证据与写作由 Main 另行委派，预算不跨检查点重置。


def settle_after_analysis(state):
    """新证据经过路线分析后才结算边际收益，不能把搜索命中数当证据数。"""
    from app.agent.evidence_recovery import count_new_relevant_evidence
    from app.agent.nodes import diagnose_evidence_gaps_node, global_evidence_gate_node
    from app.schemas.recovery_schema import RecoveryAction, RecoveryStatus
    pending = state.get("pending_evidence_recovery")
    if not pending:
        return
    settings = get_settings()
    _refresh_evidence_snapshot(state)
    diagnose_evidence_gaps_node(state, llm=None)
    if pending["kind"] == "route":
        decision = pending["decision"]
        validation = state.get("route_validation_report") or {}
        after_coverage = float((validation.get("coverage") or {}).get("evidence_understood_rate") or 0)
        new_ids = sorted({str(c.get("paper_id")) for c in state.get("paper_cards") or []}
                         - set(pending["before_card_ids"]))
        core_after = _route_core_counts(state)
        attempts = state.get("route_recovery_attempts") or {}
        progress = {route: {
            "target": int((decision.get("route_targets") or {}).get(route) or 0),
            "core_before": int(pending["core_before"].get(route) or 0),
            "core_after": int(core_after.get(route) or 0),
            "new_relevant": count_new_relevant_evidence(validation, new_ids, route_id=route),
            "attempts": int(attempts.get(route) or 0),
        } for route in decision.get("affected_route_ids") or []}
        state["route_recovery_progress"] = {**(state.get("route_recovery_progress") or {}), **progress}
        gain = after_coverage - pending["coverage_before"]
        unmet = any(p["target"] and p["core_after"] < p["target"] for p in progress.values())
        from app.schemas.recovery_schema import RouteGapType
        remaining = any(g.get("gap_type") in {RouteGapType.SEARCH_COVERAGE_GAP.value, RouteGapType.SEARCH_PRECISION_GAP.value}
                        for g in (state.get("evidence_gap_report") or {}).get("gaps") or [])
        status, reason = RecoveryStatus.RECOVERABLE, ""
        if not remaining and not unmet:
            status, reason = RecoveryStatus.NOT_REQUIRED, "route evidence gaps resolved"
        elif sum(p["new_relevant"] for p in progress.values()) < settings.evidence_recovery_min_new_evidence:
            status, reason = RecoveryStatus.EXHAUSTED, "new route-bound evidence below marginal-gain minimum"
        elif gain < settings.evidence_recovery_min_coverage_gain:
            status, reason = RecoveryStatus.EXHAUSTED, "coverage gain below marginal-gain minimum"
        state["evidence_recovery_status"] = status.value
        _append_recovery_history(state, round=int(state.get("recovery_round") or 0),
            action=RecoveryAction(decision["action"]), status=status,
            affected_route_ids=list(progress), queries=decision.get("queries") or [],
            new_evidence=len(new_ids), new_relevant_evidence=count_new_relevant_evidence(validation, new_ids),
            coverage_before=pending["coverage_before"], coverage_after=after_coverage, coverage_gain=gain,
            query_novelty=float(decision.get("query_novelty") or 0), stop_reason=reason, route_progress=progress)
    if state.get("paper_details") and settings.enable_global_evidence_gate:
        global_evidence_gate_node(state)
    _record_route_evidence_deficits(state)
    for field in ("pending_evidence_recovery", "incremental_retrieval", "incremental_search_window",
                  "incremental_required_new_evidence", "_citation_gap_repair_previous_cited"):
        state.pop(field, None)
    state["citation_shortfall_count"] = 0
