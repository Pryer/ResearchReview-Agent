"""路线证据恢复：语义诊断、确定性决策和边际收益计算。

本模块不请求论文数据源，也不控制工作流循环。外部检索仍由现有 search/rank/fetch
节点完成；这里只生成结构化诊断和可审计的控制决策。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from app.core.json_utils import parse_json_object
from app.core.logger import get_logger
from app.schemas.recovery_schema import (
    EvidenceGapReport,
    RecoveryAction,
    RecoveryDecision,
    RecoveryStatus,
    RouteEvidenceGap,
    RouteGapType,
)

logger = get_logger(__name__)


def _query_tokens(text: str) -> set[str]:
    normalized = str(text or "").strip().casefold()
    latin = set(re.findall(r"[a-z0-9]+", normalized))
    chinese_chunks = re.findall(r"[\u4e00-\u9fff]+", normalized)
    chinese = {
        chunk[index:index + 2]
        for chunk in chinese_chunks
        for index in range(max(1, len(chunk) - 1))
        if chunk[index:index + 2]
    }
    return latin | chinese


def query_novelty(query: str, historical_queries: list[str]) -> float:
    """返回查询相对历史查询的语义词项新颖度（1 为完全新）。"""
    tokens = _query_tokens(query)
    if not tokens:
        return 0.0
    similarities: list[float] = []
    for previous in historical_queries:
        old_tokens = _query_tokens(previous)
        union = tokens | old_tokens
        similarities.append(len(tokens & old_tokens) / len(union) if union else 0.0)
    return max(0.0, 1.0 - max(similarities, default=0.0))


def _source_health(state: dict[str, Any]) -> str:
    diagnostics = state.get("source_diagnostics") or []
    outcomes = [
        str(
            item.get("outcome") or item.get("status")
            if isinstance(item, dict)
            else getattr(item, "outcome", None) or getattr(item, "status", "")
        ).strip().lower()
        for item in diagnostics
    ]
    outcomes = [item for item in outcomes if item]
    if not outcomes:
        return "unknown"
    succeeded = sum(item in {"success_with_results", "success"} for item in outcomes)
    empty = sum(item in {"success_empty", "empty"} for item in outcomes)
    failed = sum(item in {
        "rate_limited", "timeout", "authentication_failed", "api_failed",
        "failed", "human_action_required",
    } for item in outcomes)
    skipped = sum(item in {"query_not_adapted", "skipped"} for item in outcomes)
    if failed and not succeeded and not empty:
        return "unavailable"
    if not succeeded and not empty and skipped:
        return "empty"
    if failed:
        return "partial"
    if empty and not succeeded:
        return "empty"
    if empty:
        # WHY: 请求成功但零结果只证明来源可访问，不证明它为本任务提供了证据。
        # 与有结果的 success 混为 healthy 会隐藏英文源空召回等覆盖缺口。
        return "partial"
    return "healthy"


def _fallback_route_queries(route: dict[str, Any]) -> list[str]:
    queries = [
        str(item).strip()
        for item in route.get("search_queries") or []
        if str(item).strip()
    ]
    if not queries:
        concepts = [
            str(item).strip()
            for item in route.get("core_concepts") or []
            if str(item).strip()
        ]
        if concepts:
            queries.append(" ".join(concepts[:4]))
    return list(dict.fromkeys(queries))


def _regenerated_route_queries(
    route: dict[str, Any],
    historical: list[str],
    min_novelty: float,
) -> list[str]:
    """为该路线重新生成检索词（确定性，不依赖 LLM）。

    WHY: 直接复用路线首轮查询会让 query_novelty 接近 0，控制器随即 DEGRADE，
    补检索空转。这里的再生成只用路线自己的核心概念做窄化组合：单概念与
    概念对相对历史宽查询是词项子集，天然具备新颖度；更窄的检索式恰好是
    补特定方向时需要的口径。全部候选仍受新颖度门槛约束。
    """
    base = _fallback_route_queries(route)
    novel = [
        query for query in base
        if query_novelty(query, historical) >= min_novelty
    ]
    if novel:
        return novel

    concepts = [
        str(item).strip()
        for item in route.get("core_concepts") or []
        if str(item).strip()
    ]
    candidates: list[str] = [*concepts]
    for index, first in enumerate(concepts):
        for second in concepts[index + 1:]:
            candidates.append(f"{first} {second}")
    regenerated = [
        query for query in dict.fromkeys(candidates)
        if query_novelty(query, historical) >= min_novelty
    ]
    return regenerated or base


def _deterministic_route_gaps(state: dict[str, Any]) -> list[RouteEvidenceGap]:
    from app.agent.route_targets import derive_route_core_targets, route_diversity_deficits

    framework = state.get("provisional_framework") or {}
    # WHY: SPLIT 产出的子路线（{parent}_S{i}）和 scope 修订后的路线不在
    # provisional_routes 里，只用它建 route_map 会让这些路线永远诊断不出缺口。
    route_map = {
        str(route.get("route_id") or ""): route
        for route in [
            *(framework.get("provisional_routes") or []),
            *(state.get("validated_routes") or []),
        ]
        if route.get("route_id")
    }
    targets = derive_route_core_targets(state)
    core_papers = {
        str(route.get("route_id") or ""): [
            str(paper_id) for paper_id in route.get("core_paper_ids") or []
        ]
        for route in state.get("validated_routes") or []
        if route.get("route_id")
    }
    diversity = route_diversity_deficits(state, core_papers)
    settings_min_novelty = 0.0
    try:
        from app.core.config import get_settings

        settings_min_novelty = float(get_settings().evidence_recovery_min_query_novelty)
    except Exception:  # pragma: no cover - 配置不可用时退回无门槛
        settings_min_novelty = 0.0
    historical = list(state.get("searched_keywords") or [])
    gaps: list[RouteEvidenceGap] = []
    for decision in state.get("route_decisions") or []:
        route_id = str(decision.get("route_id") or "")
        if not route_id or route_id not in route_map:
            continue
        route = route_map[route_id]
        action = str(decision.get("action") or "").upper()
        status = str(decision.get("status") or "").upper()
        diagnosis = str(decision.get("diagnosis") or "").upper()
        scores = dict(decision.get("scores") or {})
        core_count = int(scores.get("core_paper_count") or 0)
        paper_count = int(scores.get("paper_count") or 0)
        mean_fit = float(scores.get("mean_route_fit") or 0.0)
        target = int(targets.get(route_id) or 0)
        deficit = max(0, target - core_count)

        if (
            action in {"DROP", "TARGETED_SEARCH"}
            or status == "WEAK"
            or diagnosis == "INSUFFICIENT_EVIDENCE"
            or deficit > 0
        ):
            gap_type = (
                RouteGapType.SEARCH_PRECISION_GAP
                if (
                    paper_count > core_count
                    and mean_fit <= float(scores.get("supporting_threshold") or 0.0)
                )
                else RouteGapType.SEARCH_COVERAGE_GAP
            )
            validity = decision.get("route_validity") or {}
            sufficiency = decision.get("evidence_sufficiency") or {}
            reason = (
                f"路线有效性 {float(validity.get('score') or 0.0):.2f}，"
                f"证据充分性 {float(sufficiency.get('score') or 0.0):.2f}；"
                f"当前 {core_count} 篇核心证据，目标 {target} 篇，缺口 {deficit} 篇，"
                f"状态为 {status or action or diagnosis}。"
            )
            resolved = False
        elif action in {
            "MERGE", "MERGED_INTO", "SPLIT", "SPLIT_INTO", "REDEFINE_BOUNDARY",
            "OUTLIER_CHECK", "ADD_NEW_ROUTE", "ADD_NEW_ROUTE_CANDIDATE",
        }:
            gap_type = RouteGapType.ROUTE_STRUCTURE_GAP
            reason = f"路线结构已由验证阶段执行或提出 {action}。"
            resolved = action in {
                "MERGED_INTO", "SPLIT_INTO", "REDEFINE_BOUNDARY", "OUTLIER_CHECK",
                "ADD_NEW_ROUTE",
            }
        else:
            continue

        gaps.append(RouteEvidenceGap(
            route_id=route_id,
            route_name=str(route.get("name") or decision.get("route_name") or ""),
            gap_type=gap_type,
            reason=reason,
            metrics=scores,
            suggested_queries=_regenerated_route_queries(
                route, historical, settings_min_novelty,
            ),
            structurally_resolved=resolved,
            core_evidence_count=core_count,
            target_core_evidence=target,
            core_evidence_deficit=deficit,
            diversity_deficit=diversity.get(route_id),
        ))
    return gaps


def _merge_llm_diagnosis(
    gaps: list[RouteEvidenceGap],
    data: dict[str, Any],
) -> tuple[list[RouteEvidenceGap], bool, list[str]]:
    by_route = {gap.route_id: gap for gap in gaps}
    allowed = {item.value for item in RouteGapType}
    for item in data.get("route_diagnoses") or []:
        if not isinstance(item, dict):
            continue
        route_id = str(item.get("route_id") or "")
        gap = by_route.get(route_id)
        if gap is None:
            continue
        proposed_type = str(item.get("gap_type") or "")
        # LLM 可以在两个检索型缺口之间细化，但不能把确定性的结构问题改写成检索问题。
        if (
            proposed_type in allowed
            and gap.gap_type in {
                RouteGapType.SEARCH_COVERAGE_GAP,
                RouteGapType.SEARCH_PRECISION_GAP,
            }
            and proposed_type in {
                RouteGapType.SEARCH_COVERAGE_GAP.value,
                RouteGapType.SEARCH_PRECISION_GAP.value,
            }
        ):
            gap.gap_type = RouteGapType(proposed_type)
        reason = str(item.get("reason") or "").strip()
        if reason:
            gap.reason = reason
        for field in ("suggested_queries", "missing_constraints", "exclusion_candidates"):
            values = [
                str(value).strip()
                for value in item.get(field) or []
                if str(value).strip()
            ]
            if values:
                setattr(gap, field, list(dict.fromkeys(values)))
    return (
        gaps,
        bool(data.get("scope_revision_recommended")),
        [str(item).strip() for item in data.get("notes") or [] if str(item).strip()],
    )


def diagnose_evidence_gaps(
    state: dict[str, Any],
    llm=None,
) -> EvidenceGapReport:
    """用确定性指标建立事实底座，再让 LLM 补充术语和语义原因。"""
    validation = state.get("route_validation_report") or {}
    coverage = validation.get("coverage") or {}
    coverage_score = float(coverage.get("evidence_understood_rate") or 0.0)
    gaps = _deterministic_route_gaps(state)
    source_health = _source_health(state)
    diagnosis_source = "deterministic"
    llm_scope_recommended = False
    notes: list[str] = []

    if llm is not None and gaps:
        try:
            from app.prompt.evidence_recovery import EVIDENCE_GAP_DIAGNOSIS_PROMPT

            request = {
                "user_query": state.get("user_query"),
                "canonical_topic": state.get("canonical_topic") or state.get("topic"),
                "selected_scope": state.get("selected_scope") or {},
                "semantic_frame": state.get("research_semantic_frame") or {},
                "start_year": state.get("start_year"),
                "end_year": state.get("end_year"),
            }
            prompt = EVIDENCE_GAP_DIAGNOSIS_PROMPT.format(
                request_json=json.dumps(request, ensure_ascii=False),
                validation_json=json.dumps({
                    "decisions": state.get("route_decisions") or [],
                    "coverage": coverage,
                    "gaps": [item.model_dump(mode="json") for item in gaps],
                    # 明确告知每条路线要补多少篇，让建议查询有规模指引；
                    # 目标本身由代码派生，LLM 不得改写。
                    "route_evidence_targets": [
                        {
                            "route_id": item.route_id,
                            "route_name": item.route_name,
                            "core_evidence_count": item.core_evidence_count,
                            "target_core_evidence": item.target_core_evidence,
                            "core_evidence_deficit": item.core_evidence_deficit,
                        }
                        for item in gaps
                    ],
                }, ensure_ascii=False),
                searched_queries_json=json.dumps(
                    state.get("searched_keywords") or [], ensure_ascii=False
                ),
            )
            response = llm.complete(
                prompt,
                response_format="json",
                temperature=0.0,
                operation="diagnose_evidence_gaps",
            )
            data = parse_json_object(response if isinstance(response, str) else str(response))
            if data:
                gaps, llm_scope_recommended, notes = _merge_llm_diagnosis(gaps, data)
                diagnosis_source = "hybrid"
        except Exception as exc:  # 诊断可降级，检索证据不受影响
            logger.warning("LLM evidence gap diagnosis failed; using metrics: %s", exc)
            notes.append(f"LLM diagnosis unavailable: {type(exc).__name__}")

    unresolved = [gap for gap in gaps if not gap.structurally_resolved]
    search_gaps = [
        gap for gap in unresolved
        if gap.gap_type in {
            RouteGapType.SEARCH_COVERAGE_GAP,
            RouteGapType.SEARCH_PRECISION_GAP,
        }
    ]
    from app.core.config import get_settings

    route_count = len((state.get("provisional_framework") or {}).get("provisional_routes") or [])
    systemic_gap = bool(
        route_count
        and len(search_gaps) / route_count
        >= get_settings().evidence_recovery_scope_gap_ratio
    )
    scope_revision_recommended = bool(
        source_health != "unavailable"
        and systemic_gap
        and llm_scope_recommended
    )
    if scope_revision_recommended:
        for gap in search_gaps:
            gap.gap_type = RouteGapType.SCOPE_GAP

    return EvidenceGapReport(
        evidence_snapshot_version=int(state.get("evidence_snapshot_version") or 0),
        evidence_snapshot_fingerprint=str(
            state.get("evidence_snapshot_fingerprint") or ""
        ),
        needs_recovery=bool(search_gaps),
        gaps=gaps,
        affected_route_ids=list(dict.fromkeys(gap.route_id for gap in search_gaps)),
        coverage_score=max(0.0, min(1.0, coverage_score)),
        source_health=source_health,
        diagnosis_source=diagnosis_source,
        scope_revision_recommended=scope_revision_recommended,
        notes=notes,
    )


def decide_recovery(
    state: dict[str, Any],
    report: EvidenceGapReport,
    *,
    max_rounds: int,
    max_route_attempts: int,
    min_query_novelty: float,
    max_scope_revisions: int,
    max_queries: int,
) -> RecoveryDecision:
    """根据报告和预算做确定性决策；LLM 无权增加循环预算。"""
    if not report.needs_recovery:
        return RecoveryDecision(
            action=RecoveryAction.CONTINUE,
            status=RecoveryStatus.NOT_REQUIRED,
            reason="当前路线缺口已由结构修订解决，或证据覆盖满足进入 Claim Planning 的条件。",
        )
    if report.source_health == "unavailable":
        return RecoveryDecision(
            action=RecoveryAction.DEGRADE,
            status=RecoveryStatus.DEGRADED,
            reason="当前数据源不可用，继续改写查询不能证明证据不存在。",
            affected_route_ids=report.affected_route_ids,
        )

    round_count = int(state.get("recovery_round") or 0)
    if round_count >= max_rounds:
        return RecoveryDecision(
            action=RecoveryAction.DEGRADE,
            status=RecoveryStatus.EXHAUSTED,
            reason=f"已达到证据恢复检索上限 {max_rounds} 轮。",
            affected_route_ids=report.affected_route_ids,
        )

    if (
        report.scope_revision_recommended
        and int(state.get("scope_revision_count") or 0) < max_scope_revisions
    ):
        return RecoveryDecision(
            action=RecoveryAction.SCOPE_REVISION,
            status=RecoveryStatus.RECOVERABLE,
            reason="多数候选路线同时缺少覆盖，先在用户边界内修订概念框架。",
            affected_route_ids=report.affected_route_ids,
        )

    route_attempts = state.get("route_recovery_attempts") or {}
    eligible_routes = [
        route_id for route_id in report.affected_route_ids
        if int(route_attempts.get(route_id) or 0) < max_route_attempts
    ]
    if not eligible_routes:
        return RecoveryDecision(
            action=RecoveryAction.DEGRADE,
            status=RecoveryStatus.EXHAUSTED,
            reason="所有受影响路线均达到各自的补搜次数上限。",
            affected_route_ids=report.affected_route_ids,
        )

    historical = list(state.get("searched_keywords") or [])
    # WHY: 按缺口大小轮询分配查询名额，保证每条 eligible 路线在预算内至少拿到
    # 一条查询。此前所有路线的建议查询混在一起取前 N 条，缺口大的路线可能
    # 一条都分不到。
    per_route_candidates: dict[str, list[tuple[str, float]]] = {}
    route_targets: dict[str, int] = {}
    missing_constraints: list[str] = []
    exclusion_candidates: list[str] = []
    gap_counts: Counter[RouteGapType] = Counter()
    for gap in report.gaps:
        if gap.route_id not in eligible_routes:
            continue
        gap_counts[gap.gap_type] += 1
        missing_constraints.extend(gap.missing_constraints)
        exclusion_candidates.extend(gap.exclusion_candidates)
        route_targets[gap.route_id] = int(gap.target_core_evidence or 0)
        scored = [
            (query, query_novelty(query, historical))
            for query in gap.suggested_queries
        ]
        per_route_candidates[gap.route_id] = sorted(
            [item for item in scored if item[1] >= min_query_novelty],
            key=lambda item: -item[1],
        )

    ordered_routes = sorted(
        per_route_candidates,
        key=lambda route_id: -next(
            (
                gap.core_evidence_deficit
                for gap in report.gaps
                if gap.route_id == route_id
            ),
            0,
        ),
    )
    budget = max(1, max_queries)
    allocation: dict[str, list[str]] = {route_id: [] for route_id in ordered_routes}
    novelty_by_query: dict[str, float] = {}
    selected_queries: list[str] = []
    cursor = 0
    while len(selected_queries) < budget and ordered_routes:
        progressed = False
        for route_id in ordered_routes:
            if len(selected_queries) >= budget:
                break
            candidates = per_route_candidates.get(route_id) or []
            if cursor >= len(candidates):
                continue
            query, novelty = candidates[cursor]
            if query in novelty_by_query:
                progressed = True
                continue
            novelty_by_query[query] = novelty
            allocation[route_id].append(query)
            selected_queries.append(query)
            progressed = True
        if not progressed:
            break
        cursor += 1

    if not selected_queries:
        return RecoveryDecision(
            action=RecoveryAction.DEGRADE,
            status=RecoveryStatus.EXHAUSTED,
            reason="没有生成达到新颖度要求的增量查询，继续搜索预计只会重复召回。",
            affected_route_ids=eligible_routes,
            route_targets=route_targets,
        )

    precision_dominant = (
        gap_counts[RouteGapType.SEARCH_PRECISION_GAP]
        > gap_counts[RouteGapType.SEARCH_COVERAGE_GAP]
    )
    action = (
        RecoveryAction.QUERY_FILTER_REVISION
        if precision_dominant else RecoveryAction.TARGETED_SEARCH
    )
    return RecoveryDecision(
        action=action,
        status=RecoveryStatus.RECOVERABLE,
        reason=(
            "当前召回噪声主导，使用缺失约束收窄受影响分支。"
            if precision_dominant
            else "当前路线方向可保留，使用新术语执行定向补搜。"
        ),
        affected_route_ids=eligible_routes,
        queries=selected_queries,
        missing_constraints=list(dict.fromkeys(missing_constraints)),
        # 仅保留为审计建议；执行层不会把它直接写入全局硬排除词。
        exclusion_candidates=list(dict.fromkeys(exclusion_candidates)),
        query_novelty=sum(novelty_by_query[q] for q in selected_queries) / len(selected_queries),
        route_targets=route_targets,
        route_query_allocation={
            route_id: queries
            for route_id, queries in allocation.items()
            if queries
        },
    )


def revise_scope_framework(
    state: dict[str, Any],
    report: EvidenceGapReport,
    llm,
) -> dict[str, Any]:
    """在用户显式边界内修订候选框架；失败时返回空字典。"""
    if llm is None:
        return {}
    try:
        from app.agent.provisional_routes import _MIN_ROUTES, _validate_and_normalize_routes
        from app.prompt.evidence_recovery import SCOPE_REVISION_PROMPT

        request = {
            "user_query": state.get("user_query"),
            "canonical_topic": state.get("canonical_topic") or state.get("topic"),
            "selected_scope": state.get("selected_scope") or {},
            "semantic_frame": state.get("research_semantic_frame") or {},
            "start_year": state.get("start_year"),
            "end_year": state.get("end_year"),
        }
        prompt = SCOPE_REVISION_PROMPT.format(
            request_json=json.dumps(request, ensure_ascii=False),
            framework_json=json.dumps(
                state.get("provisional_framework") or {}, ensure_ascii=False
            ),
            gap_report_json=report.model_dump_json(),
        )
        response = llm.complete(
            prompt,
            response_format="json",
            temperature=0.0,
            operation="revise_evidence_scope",
        )
        data = parse_json_object(response if isinstance(response, str) else str(response))
        routes = _validate_and_normalize_routes(data.get("provisional_routes") or [])
        if len(routes) < _MIN_ROUTES:
            return {}
        previous = state.get("provisional_framework") or {}
        return {
            "research_scope": data.get("research_scope") or previous.get("research_scope") or {},
            "background_outline": previous.get("background_outline") or {},
            "provisional_routes": routes,
        }
    except Exception as exc:
        logger.warning("Scope framework revision failed: %s", exc)
        return {}


def count_new_relevant_evidence(
    validation_report: dict[str, Any],
    new_paper_ids: list[str],
    route_id: str | None = None,
) -> int:
    """统计新增证据中真正被路线接纳的篇数。

    传入 ``route_id`` 时只统计落到该路线的新证据；否则统计任意路线归属。
    WHY: 全局计数会把"补的都落在已达标路线"误判为有进展，让缺证路线永远
    等不到针对自己的下一轮补搜。
    """
    assignments = validation_report.get("assignment_map") or {}
    routes = {
        str(route.get("route_id") or ""): {
            str(paper_id)
            for paper_id in [
                *(route.get("core_paper_ids") or []),
                *(route.get("supporting_paper_ids") or []),
            ]
        }
        for route in validation_report.get("validated_routes") or []
        if route.get("route_id")
    }
    members = routes.get(str(route_id or ""), set())
    return sum(
        1
        for paper_id in set(new_paper_ids)
        if (assignments.get(paper_id) or {}).get("type") in {"single_route", "cross_route"}
        and (route_id is None or paper_id in members)
    )
