"""路线证据恢复闭环。

从 graph.py 编排层下沉的恢复状态机：预算推导、恢复动作分派
（CONTINUE/DEGRADE/SCOPE_REVISION/TARGETED_SEARCH/QUERY_FILTER_REVISION）、
增量检索批次的状态注入、级联节点调用与边际收益终止判定。
graph 只负责决定"何时进入本闭环"。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

from app.agent.execution import AgentCancelledError
from app.agent.execution import get_llm as _get_llm
from app.agent.nodes import (
    diagnose_evidence_gaps_node,
    download_pdf_node,
    extract_card_node,
    fetch_detail_node,
    parse_pdf_node,
    rank_node,
    recovery_controller_node,
    scope_revision_node,
    search_node,
    validate_routes_node,
)
from app.agent.router import should_parse_pdf
from app.agent.state import ResearchAgentState
from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)


def _record_recovery_statistics(
    state: ResearchAgentState,
    *,
    reused_claims: int = 0,
    recomputed_claims: int = 0,
    llm_calls_before: int | None = None,
    stage_durations: dict[str, int] | None = None,
) -> None:
    """Record stable per-round reuse/recompute counters for diagnostics."""
    stats = dict(state.get("recovery_statistics") or {})
    stats["rounds"] = int(stats.get("rounds") or 0) + 1
    stats["reused_claims"] = int(stats.get("reused_claims") or 0) + int(reused_claims)
    stats["recomputed_claims"] = int(stats.get("recomputed_claims") or 0) + int(recomputed_claims)
    if llm_calls_before is not None:
        from app.core.metrics import get_metrics_collector
        current = get_metrics_collector().get_token_report().get("total_calls", 0)
        stats["llm_calls"] = int(stats.get("llm_calls") or 0) + max(0, int(current) - int(llm_calls_before))
    if stage_durations:
        durations = dict(stats.get("stage_durations_ms") or {})
        for stage, duration in stage_durations.items():
            durations[stage] = int(durations.get(stage) or 0) + max(0, int(duration))
        stats["stage_durations_ms"] = durations
    state["recovery_statistics"] = stats


def _refresh_evidence_snapshot(state: ResearchAgentState) -> None:
    """在论文或路线归属变化时递增证据快照版本。"""
    payload = {
        "papers": sorted(
            str(card.get("paper_id") or "")
            for card in state.get("paper_cards") or []
            if card.get("paper_id")
        ),
        "routes": sorted(
            (
                str(route.get("route_id") or ""),
                tuple(sorted(str(pid) for pid in route.get("core_paper_ids") or [])),
            )
            for route in state.get("validated_routes") or []
            if route.get("route_id")
        ),
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    if fingerprint != str(state.get("evidence_snapshot_fingerprint") or ""):
        state["evidence_snapshot_version"] = int(
            state.get("evidence_snapshot_version") or 0
        ) + 1
        state["evidence_snapshot_fingerprint"] = fingerprint


def refresh_recovery_queries_from_framework(state: ResearchAgentState) -> None:
    """将修订后的路线查询更新到检索状态，不改写用户显式 scope。"""
    from app.agent.provisional_routes import (
        generate_global_recall_queries,
        route_aware_search_queries,
    )

    framework = state.get("provisional_framework") or {}
    routes = framework.get("provisional_routes") or []
    topic = state.get("canonical_topic") or state.get("topic", "")
    global_queries = generate_global_recall_queries(
        topic,
        state.get("research_semantic_frame"),
    )
    route_branches = route_aware_search_queries(routes, global_topic=topic)
    revised_queries = [
        str(query).strip()
        for branch in route_branches
        for query in branch.get("queries") or []
        if str(query).strip()
    ]
    state["keywords"] = list(dict.fromkeys([
        *(state.get("keywords") or []),
        *global_queries,
        *revised_queries,
    ]))
    retained_branches = [
        branch for branch in (state.get("search_branches") or [])
        if not str(branch.get("branch_type") or "").startswith("route_")
    ]
    state["search_branches"] = retained_branches + [
        {
            "branch_type": f"route_{branch['route_id']}",
            "queries": branch["queries"],
            "required_concepts": branch.get("core_concepts", []),
            "rationale": f"修订后候选路线「{branch['route_name']}」的定向检索",
            "constraint_level": "exploratory",
        }
        for branch in route_branches
    ]
    state["global_recall_queries"] = global_queries


def _route_core_counts(state: ResearchAgentState) -> dict[str, int]:
    """当前每条路线的核心证据篇数。"""
    return {
        str(route.get("route_id") or ""): len(route.get("core_paper_ids") or [])
        for route in state.get("validated_routes") or []
        if route.get("route_id")
    }


def _append_recovery_history(
    state: ResearchAgentState,
    **entry_data: Any,
) -> None:
    from app.schemas.recovery_schema import RecoveryHistoryEntry

    entry = RecoveryHistoryEntry(**entry_data).model_dump(mode="json")
    state.setdefault("recovery_history", []).append(entry)


def run_route_evidence_recovery(
    state: ResearchAgentState,
    *,
    should_cancel: Callable[[], bool] | None = None,
) -> None:
    """在路线验证和 Claim Planning 之间执行有界、增量的证据恢复闭环。"""
    from app.agent.evidence_recovery import count_new_relevant_evidence
    from app.schemas.recovery_schema import RecoveryAction, RecoveryStatus

    settings = get_settings()
    state.setdefault("recovery_round", 0)
    state.setdefault("route_recovery_attempts", {})
    state.setdefault("scope_revision_count", 0)
    state.setdefault("recovery_history", [])
    action_budget = (
        settings.evidence_recovery_max_rounds
        + settings.evidence_recovery_max_scope_revisions
        + 1
    )
    llm = _get_llm()
    _refresh_evidence_snapshot(state)

    try:
        for _ in range(max(1, action_budget)):
            if should_cancel and should_cancel():
                raise AgentCancelledError("任务已在证据恢复轮次之间取消")

            diagnose_evidence_gaps_node(state, llm=llm)
            recovery_controller_node(state)
            decision = state.get("recovery_decision") or {}
            action = str(decision.get("action") or RecoveryAction.DEGRADE.value)
            status = str(decision.get("status") or RecoveryStatus.DEGRADED.value)

            if action in {RecoveryAction.CONTINUE.value, RecoveryAction.DEGRADE.value}:
                state["evidence_recovery_status"] = status
                if action == RecoveryAction.DEGRADE.value:
                    _append_recovery_history(
                        state,
                        round=int(state.get("recovery_round") or 0),
                        action=RecoveryAction.DEGRADE,
                        status=RecoveryStatus(status),
                        affected_route_ids=decision.get("affected_route_ids") or [],
                        stop_reason=str(decision.get("reason") or "recovery degraded"),
                    )
                break

            if action == RecoveryAction.SCOPE_REVISION.value:
                before = float(
                    (state.get("evidence_gap_report") or {}).get("coverage_score") or 0.0
                )
                scope_revision_node(state, llm=llm)
                if state.get("scope_revision_failed"):
                    state["evidence_recovery_status"] = RecoveryStatus.DEGRADED.value
                    _append_recovery_history(
                        state,
                        round=int(state.get("recovery_round") or 0),
                        action=RecoveryAction.SCOPE_REVISION,
                        status=RecoveryStatus.DEGRADED,
                        affected_route_ids=decision.get("affected_route_ids") or [],
                        coverage_before=before,
                        coverage_after=before,
                        stop_reason="scope revision failed contract validation",
                    )
                    break
                refresh_recovery_queries_from_framework(state)
                validate_routes_node(state, llm=llm)
                _refresh_evidence_snapshot(state)
                after = float(
                    ((state.get("route_validation_report") or {}).get("coverage") or {}).get(
                        "evidence_understood_rate"
                    ) or 0.0
                )
                _append_recovery_history(
                    state,
                    round=int(state.get("recovery_round") or 0),
                    action=RecoveryAction.SCOPE_REVISION,
                    status=RecoveryStatus.RECOVERABLE,
                    affected_route_ids=decision.get("affected_route_ids") or [],
                    coverage_before=before,
                    coverage_after=after,
                    coverage_gain=after - before,
                )
                continue

            if action not in {
                RecoveryAction.TARGETED_SEARCH.value,
                RecoveryAction.QUERY_FILTER_REVISION.value,
            }:
                # 结构型修订已由 validate_routes_against_evidence 执行；控制器不应
                # 让同一组证据在这里重复 MERGE/SPLIT 并造成振荡。
                state["evidence_recovery_status"] = RecoveryStatus.NOT_REQUIRED.value
                break

            queries = [
                str(query).strip()
                for query in decision.get("queries") or []
                if str(query).strip()
            ]
            if not queries:
                state["evidence_recovery_status"] = RecoveryStatus.EXHAUSTED.value
                break

            round_number = int(state.get("recovery_round") or 0) + 1
            affected_routes = [
                str(route_id) for route_id in decision.get("affected_route_ids") or []
                if str(route_id)
            ]
            before_coverage = float(
                (state.get("evidence_gap_report") or {}).get("coverage_score") or 0.0
            )
            before_card_ids = {
                str(card.get("paper_id") or "")
                for card in (state.get("paper_cards") or [])
                if card.get("paper_id")
            }
            route_targets = {
                str(route_id): int(target or 0)
                for route_id, target in (decision.get("route_targets") or {}).items()
            }
            core_before = _route_core_counts(state)

            state["keywords"] = list(dict.fromkeys([
                *(state.get("keywords") or []),
                *queries,
            ]))
            state.setdefault("search_branches", []).append({
                "branch_type": f"evidence_recovery_{round_number}",
                "queries": queries,
                "required_concepts": decision.get("missing_constraints") or [],
                "rationale": str(decision.get("reason") or "route evidence recovery"),
                "constraint_level": "targeted_recovery",
                "affected_route_ids": affected_routes,
            })
            state["incremental_retrieval"] = True
            state["incremental_search_window"] = {
                "start_year": state.get("start_year"),
                "end_year": state.get("end_year"),
            }
            state["incremental_required_new_evidence"] = (
                settings.evidence_recovery_min_new_evidence
            )
            state["last_search_new_results"] = 0
            search_node(state, should_cancel=should_cancel)

            if int(state.get("last_search_new_results") or 0) > 0:
                # Recovery is incremental. Keep the existing deterministic
                # screening/ranking rules instead of re-running batched LLM
                # reranking over the full accumulated pool on every round.
                # The LLM remains responsible for semantic gap diagnosis and
                # novel recovery-query planning upstream.
                rank_node(state, llm=None)
                fetch_detail_node(state, should_cancel=should_cancel)
                if settings.enable_pdf_pipeline:
                    download_pdf_node(state)
                    if should_parse_pdf(state):
                        parse_pdf_node(state)
                card_llm = llm if settings.enable_llm_card_extraction else None
                extract_card_node(state, llm=card_llm)

            state["recovery_round"] = round_number
            attempts = dict(state.get("route_recovery_attempts") or {})
            for route_id in affected_routes:
                attempts[route_id] = int(attempts.get(route_id) or 0) + 1
            state["route_recovery_attempts"] = attempts

            validate_routes_node(state, llm=llm)
            _refresh_evidence_snapshot(state)
            validation = state.get("route_validation_report") or {}
            after_coverage = float(
                (validation.get("coverage") or {}).get("evidence_understood_rate")
                or 0.0
            )
            after_card_ids = {
                str(card.get("paper_id") or "")
                for card in (state.get("paper_cards") or [])
                if card.get("paper_id")
            }
            new_card_ids = sorted(after_card_ids - before_card_ids)
            new_relevant = count_new_relevant_evidence(validation, new_card_ids)
            coverage_gain = after_coverage - before_coverage
            core_after = _route_core_counts(state)
            # per-route 进度：目标、补前、补后篇数，以及真正落到该路线的新证据。
            route_progress = {
                route_id: {
                    "target": int(route_targets.get(route_id) or 0),
                    "core_before": int(core_before.get(route_id) or 0),
                    "core_after": int(core_after.get(route_id) or 0),
                    "new_relevant": count_new_relevant_evidence(
                        validation, new_card_ids, route_id=route_id,
                    ),
                    "attempts": int(attempts.get(route_id) or 0),
                }
                for route_id in affected_routes
            }
            state["route_recovery_progress"] = {
                **(state.get("route_recovery_progress") or {}),
                **route_progress,
            }
            unmet_routes = [
                route_id for route_id, progress in route_progress.items()
                if progress["target"] and progress["core_after"] < progress["target"]
            ]
            targeted_new_relevant = sum(
                progress["new_relevant"] for progress in route_progress.values()
            )
            remaining_search_gap = any(
                str(item.get("action") or "").upper() in {"DROP", "TARGETED_SEARCH"}
                or str(item.get("status") or "").upper() == "WEAK"
                or str(item.get("diagnosis") or "").upper() == "INSUFFICIENT_EVIDENCE"
                for item in (state.get("route_decisions") or [])
            )

            stop_reason = ""
            history_status = RecoveryStatus.RECOVERABLE
            if not remaining_search_gap and not unmet_routes:
                stop_reason = "route evidence gaps resolved"
                history_status = RecoveryStatus.NOT_REQUIRED
            elif targeted_new_relevant < settings.evidence_recovery_min_new_evidence:
                # 只有真正落到缺证路线的新证据才算进展；补到已达标路线不算。
                stop_reason = "new route-bound evidence below marginal-gain minimum"
                history_status = RecoveryStatus.EXHAUSTED
            elif coverage_gain < settings.evidence_recovery_min_coverage_gain:
                stop_reason = "coverage gain below marginal-gain minimum"
                history_status = RecoveryStatus.EXHAUSTED

            _append_recovery_history(
                state,
                round=round_number,
                action=RecoveryAction(action),
                status=history_status,
                affected_route_ids=affected_routes,
                queries=queries,
                new_evidence=len(new_card_ids),
                new_relevant_evidence=new_relevant,
                coverage_before=before_coverage,
                coverage_after=after_coverage,
                coverage_gain=coverage_gain,
                query_novelty=float(decision.get("query_novelty") or 0.0),
                stop_reason=stop_reason,
                route_progress=route_progress,
            )
            if stop_reason:
                state["evidence_recovery_status"] = history_status.value
                break
        else:
            state["evidence_recovery_status"] = RecoveryStatus.EXHAUSTED.value
    finally:
        # 保留新查询、论文和历史；仅清除控制当前增量批次的临时字段。
        state.pop("incremental_retrieval", None)
        state.pop("incremental_search_window", None)
        state.pop("incremental_required_new_evidence", None)
        # 最后一轮恢复可能在 validate 后因“已解决”或边际收益不足直接 break；
        # 必须针对新快照重算 gap，不能把补检索前的 0/N 报告留给后续门禁。
        if state.get("route_validation_report") is not None:
            diagnose_evidence_gaps_node(state, llm=None)
        _record_route_evidence_deficits(state)


def _record_route_evidence_deficits(state: ResearchAgentState) -> None:
    """把恢复结束后仍未达目标的路线如实记录下来。

    WHY: 这些路线随后会被 merge_weak_routes_for_writing 合并掉，缺口若不在这里
    落盘，最终交付就只剩"看不出哪条方向证据不足"的合并结果。
    """
    from app.agent.route_targets import derive_route_core_targets

    targets = derive_route_core_targets(state)
    if not targets:
        state["route_evidence_deficits"] = []
        return
    core_counts = _route_core_counts(state)
    names = {
        str(route.get("route_id") or ""): str(route.get("name") or "")
        for route in state.get("validated_routes") or []
        if route.get("route_id")
    }
    progress = state.get("route_recovery_progress") or {}
    deficits = [
        {
            "route_id": route_id,
            "route_name": names.get(route_id, ""),
            "core_evidence_count": int(core_counts.get(route_id) or 0),
            "target_core_evidence": int(target),
            "core_evidence_deficit": int(target) - int(core_counts.get(route_id) or 0),
            "recovery_attempts": int((progress.get(route_id) or {}).get("attempts") or 0),
        }
        for route_id, target in targets.items()
        if int(target) > int(core_counts.get(route_id) or 0)
        # 只报告仍存在于验证结果中的路线；已被结构性移除的不再计入。
        and route_id in core_counts
    ]
    state["route_evidence_deficits"] = deficits
