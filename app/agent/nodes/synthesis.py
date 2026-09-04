"""路线验证、聚类、综述与交付物生成节点。"""

from __future__ import annotations

import json
import math
import re
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from app.agent.decorators import node, optional, provides, requires
from app.agent.nodes.base import (
    _compact_debug_value,
    _latest_step,
    _needs_current_time_tool,
    _paper_debug_item,
    _paper_identity_key,
    _preview_text,
    _select_branch_diverse_keywords,
    _select_search_keywords,
    _summarize_papers,
    append_step,
)
from app.core.config import get_review_threshold_policy, get_settings
from app.core.logger import get_logger
from app.schemas.paper_schema import SourceDiagnostic

if TYPE_CHECKING:
    from app.agent.state import ResearchAgentState

logger = get_logger(__name__)


@node(name="validate_routes", category="generation", description="用检索证据验证和修正候选路线")
@requires("paper_cards")
@provides("validated_routes", "route_decisions", "route_validation_report")
@optional("provisional_framework", "canonical_topic", "dynamic_taxonomy", "taxonomy_validation")
def validate_routes_node(state: "ResearchAgentState", llm=None) -> "ResearchAgentState":
    """用检索到的 Evidence Cards 验证候选路线，执行 KEEP/MERGE/SPLIT/DROP/ADD。

    当 provisional_framework 存在时，本节点验证候选路线；
    当 provisional_framework 不存在时，回退到原始 cluster_node 逻辑。
    """
    t0 = time.time()
    provisional = state.get("provisional_framework") or {}
    provisional_routes = provisional.get("provisional_routes") or []

    if not provisional_routes:
        # 回退：没有候选路线时保持原聚类行为
        logger.info("No provisional routes available, skipping validation")
        state["validated_routes"] = []
        state["route_decisions"] = []
        state["route_validation_report"] = {
            "validated_routes": [],
            "decisions": [],
            "coverage": {},
            "assignment_map": {},
        }
        append_step(state, "validate_routes", "skipped",
                    output_data={"reason": "no provisional routes"},
                    duration_ms=int((time.time() - t0) * 1000))
        return state

    try:
        from app.agent.provisional_routes import validate_routes_against_evidence

        result = validate_routes_against_evidence(
            provisional_routes,
            state.get("paper_cards") or [],
            llm=llm,
            topic=str(state.get("canonical_topic") or state.get("topic") or ""),
            semantic_frame=state.get("research_semantic_frame") or {},
        )
        state["validated_routes"] = result.get("validated_routes") or []
        state["route_decisions"] = result.get("decisions") or []
        # 保存完整诊断，供 Evidence Gap Diagnosis 使用。此前只保留最终路线和
        # action，导致 coverage / unassigned / assignment 信息在节点边界丢失。
        state["route_validation_report"] = result
        state["coverage"] = result.get("coverage") or {}
        prepared_routes = result.get("prepared_routes") or []
        if prepared_routes:
            state["provisional_framework"] = {
                **provisional,
                "provisional_routes": prepared_routes,
            }

        # 用验证后的路线覆盖 dynamic_taxonomy（写作阶段使用）。WEAK 路线
        # 继续留在 validated_routes 供 Recovery 诊断，但不得被扩写成独立小节。
        if state["validated_routes"]:
            from app.agent.route_validator import merge_weak_routes_for_writing

            writing_routes = merge_weak_routes_for_writing(state["validated_routes"])
            themes = []
            assignments = []
            evidence_backed_routes = [
                route for route in writing_routes
                if route.get("paper_ids")
            ]
            route_theme_ids = [
                (f"VR{i}", route)
                for i, route in enumerate(evidence_backed_routes, 1)
            ]
            for theme_id, route in route_theme_ids:
                themes.append({
                    "theme_id": theme_id,
                    "name": route.get("name", ""),
                    "description": route.get("research_question", ""),
                    "inclusion_criteria": route.get("core_concepts", []),
                })
            # primary_theme_id 必须唯一：路线 paper_ids 含 cross_route 成员，
            # 逐路线直接写入会让同一论文获得多个“primary”归属，下游
            # synthesize_themes 按该字段聚合后主题互相吞并（实测 5 个主题
            # 各 72-81 篇、两两重叠 65-77 篇），主题节授权面随之铺满全池，
            # 造成跨小节重复引用。
            # 归属判据优先复用路线验证阶段已算出的逐篇最佳匹配
            # （assignment_map[pid]["primary_route"]，基于 token fit 排序），
            # 而不是路线书写顺序；该路线被合并或缺少记录时退回“核心成员
            # 优先 + 路线顺序”的确定性兜底。
            theme_id_by_route = {
                str(route.get("route_id") or ""): theme_id
                for theme_id, route in route_theme_ids
                if route.get("route_id")
            }
            assignment_map = result.get("assignment_map") or {}
            primary_owner: dict[str, str] = {}
            for paper_id, info in assignment_map.items():
                theme_id = theme_id_by_route.get(
                    str((info or {}).get("primary_route") or "")
                )
                if theme_id:
                    primary_owner.setdefault(str(paper_id), theme_id)
            for theme_id, route in route_theme_ids:
                for paper_id in route.get("core_paper_ids") or []:
                    primary_owner.setdefault(str(paper_id), theme_id)
            for theme_id, route in route_theme_ids:
                for paper_id in route.get("paper_ids") or []:
                    primary_owner.setdefault(str(paper_id), theme_id)
            seen_assignment_ids: set[str] = set()
            for theme_id, route in route_theme_ids:
                for paper_id in route.get("paper_ids", []):
                    paper_id = str(paper_id)
                    if paper_id in seen_assignment_ids:
                        continue
                    seen_assignment_ids.add(paper_id)
                    assignments.append({
                        "paper_id": paper_id,
                        "primary_theme_id": primary_owner.get(paper_id, theme_id),
                        "confidence": 0.8,
                    })
            if themes:
                state["dynamic_taxonomy"] = {
                    "organizing_principle": "evidence_validated_routes",
                    "themes": themes,
                    "assignments": assignments,
                }
                state["taxonomy_validation"] = {
                    "valid": True,
                    "requires_revision": False,
                    "status": "valid",
                    "paper_coverage": len(
                        set(a["paper_id"] for a in assignments)
                    ) / max(1, len(state.get("paper_cards") or [])),
                }

        decisions_summary = [
            f"{d.get('action')}: {d.get('route_id', '')} ({d.get('reason', '')})"
            for d in (result.get("decisions") or [])[:10]
        ]
        append_step(
            state, "validate_routes", "success",
            tool_name="validate_routes_against_evidence",
            input_data={"provisional_routes": len(provisional_routes)},
            output_data={
                "validated_count": len(state["validated_routes"]),
                "decisions": decisions_summary,
            },
            duration_ms=int((time.time() - t0) * 1000),
        )
        logger.info(
            "Route validation: %d provisional → %d validated (%d decisions)",
            len(provisional_routes),
            len(state["validated_routes"]),
            len(state["route_decisions"]),
        )
    except Exception as e:
        logger.warning("validate_routes_node failed: %s", e)
        state["validated_routes"] = []
        state["route_decisions"] = []
        state["route_validation_report"] = {
            "validated_routes": [],
            "decisions": [],
            "coverage": {},
            "assignment_map": {},
            "error": str(e),
        }
        append_step(state, "validate_routes", "failed", error=str(e))

    return state




# ============================================================
# Cluster 节点
# ============================================================
@node(name="cluster_papers", category="generation", description="按方法、任务、数据集、年份聚类论文")
@requires("paper_cards", "topic")
@provides("clusters", "dynamic_taxonomy", "taxonomy_validation")
@optional("canonical_topic", "selected_scope", "research_semantic_frame", "force_taxonomy_remediation", "taxonomy_remediation")
def cluster_node(state: "ResearchAgentState", llm=None) -> "ResearchAgentState":
    """按方法、任务、数据集、年份聚类论文。"""
    t0 = time.time()
    try:
        from app.tools.cluster_papers import cluster_papers

        effective_llm = None if state.get("force_taxonomy_remediation") else llm
        cluster_result = cluster_papers(
            state.get("paper_cards") or [],
            llm=effective_llm,
            topic=state.get("canonical_topic") or state.get("topic", ""),
            scope=state.get("selected_scope") or {},
        )
        cluster_result = _remediate_invalid_taxonomy(state, cluster_result)
        clusters = cluster_result["clusters"]
        state["clusters"] = clusters
        state["dynamic_taxonomy"] = cluster_result["dynamic_taxonomy"]
        state["taxonomy_validation"] = cluster_result["taxonomy_validation"]
        append_step(
            state, "cluster", "success",
            tool_name="cluster_papers",
            input_data={
                "cards": len(state.get("paper_cards") or []),
                "topic": state.get("canonical_topic") or state.get("topic", ""),
                "scope": state.get("selected_scope") or {},
            },
            output_data={
                "clusters": len(clusters),
                "cluster_sample": clusters[:8],
                "taxonomy_validation": state["taxonomy_validation"],
            },
            duration_ms=int((time.time() - t0) * 1000),
        )
    except Exception as e:
        from app.agent.exceptions import LLMGenerationError

        error = LLMGenerationError(str(e), step="cluster", original_error=e)
        logger.error("cluster_node failed: %s", error.message)
        state.setdefault("errors", []).append(f"cluster: {e}")
        append_step(state, "cluster", "failed", error=str(e))
    return state


def _remediate_invalid_taxonomy(
    state: "ResearchAgentState",
    cluster_result: Dict[str, Any],
) -> Dict[str, Any]:
    """在仍满足篇数约束时，排除兜底/碎片主题论文并确定性重建分类。"""
    validation = cluster_result.get("taxonomy_validation") or {}
    taxonomy = cluster_result.get("dynamic_taxonomy") or {}
    themes = taxonomy.get("themes") or []
    problem_theme_ids = {
        str(theme.get("theme_id") or "")
        for theme in themes
        if re.search(r"其他|其它|未分类|other|misc", str(theme.get("name") or ""), re.I)
    }
    problem_theme_ids.update(str(value) for value in validation.get("undersized_theme_ids") or [])
    if not problem_theme_ids:
        return cluster_result
    excluded_ids = {
        str(item.get("paper_id") or "")
        for item in taxonomy.get("assignments") or []
        if str(item.get("primary_theme_id") or "") in problem_theme_ids
    }
    if not excluded_ids:
        return cluster_result

    semantic_frame = state.get("research_semantic_frame") or {}
    protected_ids: set[str] = set()
    if semantic_frame.get("evidence_requirements"):
        from app.agent.evidence_roles import evidence_coverage

        coverage = evidence_coverage(semantic_frame, state.get("paper_cards") or [])
        protected_ids = {
            str(paper_id)
            for ids in coverage.get("matched_paper_ids", {}).values()
            for paper_id in ids
        }
        excluded_ids -= protected_ids
        if not excluded_ids:
            return cluster_result

    remaining_cards = [
        card for card in state.get("paper_cards") or []
        if str(card.get("paper_id") or "") not in excluded_ids
    ]
    usable_remaining = [
        card for card in remaining_cards
        if card.get("quality_status") != "invalid"
        and str((card.get("evidence_state") or {}).get("access_level") or card.get("evidence_source") or "")
        in {"abstract", "partial_full_text", "full_text"}
    ]
    minimum = (
        int(state.get("required_reference_count") or 0)
        if state.get("max_papers_explicit", False)
        else 4
    )
    eligible_remaining_count = len(usable_remaining)
    if semantic_frame.get("evidence_requirements"):
        from app.agent.evidence_roles import citation_eligible_paper_ids

        usable_ids = {
            str(card.get("paper_id") or "") for card in usable_remaining
        }
        eligible_remaining_count = len(
            citation_eligible_paper_ids(semantic_frame, usable_remaining) & usable_ids
        )
    if eligible_remaining_count < max(1, minimum):
        return cluster_result

    from app.tools.cluster_papers import cluster_papers

    rebuilt = cluster_papers(
        remaining_cards,
        llm=None,
        topic=state.get("canonical_topic") or state.get("topic", ""),
        scope=state.get("selected_scope") or {},
    )
    old_validation = cluster_result.get("taxonomy_validation") or {}
    new_validation = rebuilt.get("taxonomy_validation") or {}
    old_errors = len(old_validation.get("errors") or []) + len(old_validation.get("undersized_theme_ids") or [])
    new_errors = len(new_validation.get("errors") or []) + len(new_validation.get("undersized_theme_ids") or [])
    if new_errors >= old_errors and not new_validation.get("valid", False):
        return cluster_result

    state["paper_cards"] = remaining_cards
    state["paper_details"] = [
        paper for paper in state.get("paper_details") or []
        if str(paper.get("paper_id") or "") not in excluded_ids
    ]
    state["taxonomy_remediation"] = {
        "applied": True,
        "excluded_paper_ids": sorted(excluded_ids),
        "excluded_count": len(excluded_ids),
        "protected_direct_evidence_count": len(protected_ids),
        "eligible_remaining_count": eligible_remaining_count,
        "remaining_cards": len(remaining_cards),
        "reason": "fallback_or_undersized_themes",
    }
    append_step(
        state,
        "taxonomy_remediation",
        "success",
        tool_name="deterministic_taxonomy_remediation",
        input_data={
            "problem_theme_ids": sorted(problem_theme_ids),
            "previous_cards": len(state.get("paper_cards") or []) + len(excluded_ids),
        },
        output_data={
            **state["taxonomy_remediation"],
            "taxonomy_validation": new_validation,
        },
        duration_ms=0,
    )
    return rebuilt


# ============================================================
# 四交付物统一综合、规划与写作节点
# ============================================================
def _global_deliverable_citation_quotas(
    deliverables: list[str],
    target: int,
) -> dict[str, int]:
    """把整份正文的最低引用量分配到多个交付物，且总和严格等于 target。"""
    ordered = list(dict.fromkeys(str(item) for item in deliverables if item))
    if not ordered or target <= 0:
        return {item: 0 for item in ordered}
    if len(ordered) == 1:
        return {ordered[0]: target}

    # 权重近似各交付物的事实性章节数量。研究背景不应独自承担整份正文
    # 的 40 篇要求；研究现状通常包含更多主题章节，承担更高覆盖量。
    weights = {
        "research_background": 4,
        "research_status": 12,
        "related_work": 6,
        "narrative_review": 12,
    }
    minimums = {
        "research_background": 2,
        "research_status": 4,
        "related_work": 2,
        "narrative_review": 5,
    }
    quotas = {
        item: min(target, minimums.get(item, 1))
        for item in ordered
    }
    assigned = sum(quotas.values())
    if assigned > target:
        # 极小目标下按请求顺序每类最多分到一篇，避免配额和超过总目标。
        quotas = {item: 0 for item in ordered}
        for item in ordered[:target]:
            quotas[item] = 1
        return quotas

    remaining = target - assigned
    total_weight = sum(weights.get(item, 1) for item in ordered)
    raw_shares = {
        item: remaining * weights.get(item, 1) / max(1, total_weight)
        for item in ordered
    }
    for item in ordered:
        quotas[item] += int(raw_shares[item])
    leftover = target - sum(quotas.values())
    for item in sorted(
        ordered,
        key=lambda value: (
            raw_shares[value] - int(raw_shares[value]),
            weights.get(value, 1),
        ),
        reverse=True,
    )[:leftover]:
        quotas[item] += 1
    return quotas


def _citation_allocation_budget(
    required: int,
    usable: int,
    generation_limit: int,
) -> int:
    """引用分配预算：验收下限之外的写作证据上限。

    “不少于 N 篇”的 N 是校验下限；分配预算允许写手用满可用证据，
    至多到 ``generation_limit``（检索冗余与生成预算的既定上限）。
    可用证据不足下限时退化为可用量（降级路径，与旧行为一致）。
    """
    required = max(0, int(required or 0))
    usable = max(0, int(usable or 0))
    generation_limit = max(0, int(generation_limit or 0))
    ceiling = max(required, generation_limit)
    return min(usable, ceiling)


def _is_two_part_background_status_request(state: "ResearchAgentState") -> bool:
    """是否为仅交付研究背景和研究现状的论文正文请求。"""
    requested = set(state.get("requested_sections") or [])
    core = set(state.get("core_deliverables") or [])
    return (
        requested == {"background", "research_status"}
        or core == {"research_background", "research_status"}
    )


def _with_display_number(plan, ordinal: int, *, enabled: bool):
    """在装配层给可见一级部分编号，规划器只保留语义标题。"""
    if not enabled or not plan.sections:
        return plan
    numerals = "一二三四五六七八九十"
    rendered = plan.model_copy(deep=True)
    prefix = numerals[ordinal - 1] if 0 < ordinal <= len(numerals) else str(ordinal)
    rendered.sections[0].title = f"{prefix}、{rendered.sections[0].title}"
    return rendered


@node(name="generate_deliverables", category="generation", description="通过 Spec → Synthesis → WritingPlan → SectionWriter 生成四类交付物")
@requires("paper_details", "paper_cards", "core_deliverables", "topic")
@provides(
    "writing_plans", "deliverable_readiness", "generation_readiness",
    "search_report", "evidence_quality_report", "theme_synthesis",
    "deliverable_validation", "deliverable_downgrades",
    "generation_blocked", "review", "body",
)
@optional(
    "clusters", "dynamic_taxonomy", "theme_synthesis", "canonical_topic",
    "our_work", "background", "existing_limitations", "verified_results",
    "language", "citation_style"
)
def generate_deliverables_node(
    state: "ResearchAgentState",
    llm=None,
    should_cancel=None,
) -> "ResearchAgentState":
    """通过 Spec → Synthesis → WritingPlan → SectionWriter 生成四类交付物。"""
    t0 = time.time()
    try:
        from app.agent.deliverable_router import (
            check_deliverable_readiness,
            check_generation_readiness,
            resolve_core_deliverables,
        )
        from app.agent.writing_plan import build_writing_plan
        from app.schemas.deliverable_schema import CoreDeliverableType
        from app.tools.synthesize_themes import build_search_report, synthesize_themes
        from app.tools.validate_deliverable import validate_deliverable
        from app.tools.verify_claims import build_evidence_quality_report
        from app.tools.write_deliverable import write_deliverable

        requested = state.get("core_deliverables") or [
            item.value for item in resolve_core_deliverables(
                str(state.get("intent") or ""),
                state.get("requested_sections") or [],
            )
        ]
        state["core_deliverables"] = requested
        state["search_report"] = build_search_report(state)
        state["evidence_quality_report"] = build_evidence_quality_report(
            state.get("paper_cards") or []
        )
        state["theme_synthesis"] = synthesize_themes(
            state.get("paper_cards") or [],
            state.get("dynamic_taxonomy") or {},
        )

        global_readiness = check_generation_readiness(state)
        if (
            not global_readiness.ready
            and state.get("best_effort_generation")
            and int(global_readiness.usable_reference_count or 0) > 0
        ):
            state["forced_generation_issues"] = list(global_readiness.blocking_issues)
            global_readiness = global_readiness.model_copy(update={"ready": True})
        state["generation_readiness"] = global_readiness.model_dump(mode="json")
        if not global_readiness.ready:
            issue_lines = "\n".join(
                f"- {item.get('message')}" for item in global_readiness.blocking_issues
            )
            option_lines = "\n".join(
                f"- {item}" for item in global_readiness.recovery_options
            )
            state["generation_blocked"] = True
            state["quality_gate"] = {
                "passed": False,
                "draft_available": False,
                "draft_released": False,
                "draft_disposition": "none",
                "partial_success": False,
                "phase": "pre_generation",
                "blocking_issues": global_readiness.blocking_issues,
                "recovery_options": global_readiness.recovery_options,
            }
            state["writing_plans"] = []
            state["deliverable_validation"] = []
            state["deliverable_readiness"] = []
            state["review"] = (
                "## 正文生成已阻止\n\n"
                "当前检索与证据状态未满足用户要求，因此系统没有生成研究背景、研究现状或其他正式正文。\n\n"
                f"### 阻断原因\n\n{issue_lines}\n\n"
                f"### 可选处理方式\n\n{option_lines or '- 补充满足约束的证据后重新执行'}"
            )
            append_step(
                state,
                "generation_readiness",
                "blocked",
                tool_name="check_generation_readiness",
                input_data={
                    "deliverables": requested,
                    "required_reference_count": state.get("required_reference_count"),
                    "paper_cards": len(state.get("paper_cards") or []),
                },
                output_data=state["generation_readiness"],
                duration_ms=int((time.time() - t0) * 1000),
            )
            return state

        outputs: list[str] = []
        plans: list[dict[str, Any]] = []
        allocation_plans: list[dict[str, Any]] = []
        validations: list[dict[str, Any]] = []
        readiness_results: list[dict[str, Any]] = []
        downgrades: list[dict[str, Any]] = []
        generated_types: set[CoreDeliverableType] = set()
        display_names = {
            CoreDeliverableType.RESEARCH_BACKGROUND: "研究背景",
            CoreDeliverableType.RESEARCH_STATUS: "研究现状",
            CoreDeliverableType.RELATED_WORK: "相关工作",
            CoreDeliverableType.NARRATIVE_REVIEW: "叙述性综述初稿",
        }
        global_reference_target = min(
            int(state.get("required_reference_count") or 0),
            int(global_readiness.usable_reference_count or 0),
        )
        # 引用分配预算独立于验收下限：“不少于 N 篇”的 N 只是校验下限，
        # 分配层允许在预算内多选证据，让写手引用超过最低量的论文，
        # 而不是把下限当目标值精确执行（每次都恰好 N 篇）。
        citation_allocation_budget = _citation_allocation_budget(
            required=int(state.get("required_reference_count") or 0),
            usable=int(global_readiness.usable_reference_count or 0),
            generation_limit=int(state.get("generation_limit") or 0),
        )

        # 先完成全部交付物的 readiness / downgrade 解析，再分配全局引用量。
        # 旧逻辑先按原请求把 40 篇拆成 10+30，之后若研究现状降级并被
        # 去重跳过，30 篇配额会直接消失。预解析后，配额只在最终真正会
        # 生成的有效类型之间分配。
        resolved_deliverables: list[dict[str, Any]] = []
        for raw_type in requested:
            requested_dtype = CoreDeliverableType(raw_type)
            dtype = requested_dtype
            readiness = check_deliverable_readiness(dtype, state, phase="post_evidence")
            downgrade_reasons: list[str] = []
            visited_types = {dtype}
            while not readiness.ready and readiness.downgrade_suggestion:
                downgrade_reasons.extend([
                    *readiness.missing_inputs,
                    *readiness.insufficient_evidence,
                ])
                next_dtype = readiness.downgrade_suggestion
                if next_dtype in visited_types:
                    break
                dtype = next_dtype
                visited_types.add(dtype)
                readiness = check_deliverable_readiness(
                    dtype, state, phase="post_evidence"
                )

            unique_reasons = list(dict.fromkeys(downgrade_reasons))
            readiness_data = readiness.model_dump(mode="json")
            readiness_data["requested_type"] = requested_dtype.value
            readiness_data["effective_type"] = dtype.value
            readiness_data["downgrade_applied"] = (
                dtype != requested_dtype and readiness.ready
            )
            readiness_data["downgrade_reasons"] = unique_reasons
            readiness_results.append(readiness_data)
            if dtype != requested_dtype and readiness.ready:
                downgrades.append({
                    "requested_type": requested_dtype.value,
                    "effective_type": dtype.value,
                    "reasons": unique_reasons,
                })
            resolved_deliverables.append({
                "requested": requested_dtype,
                "effective": dtype,
                "readiness": readiness,
                "downgrade_reasons": unique_reasons,
            })

        effective_type_values: list[str] = []
        for item in resolved_deliverables:
            dtype = item["effective"]
            if (
                item["readiness"].ready
                and dtype.value not in effective_type_values
            ):
                effective_type_values.append(dtype.value)
        # 引用预算按用户请求的交付物分配，而不是按降级后的唯一类型重算。
        # 否则“研究现状未生成”会把整份40篇配额全部压到研究背景，破坏文体。
        citation_quotas = _global_deliverable_citation_quotas(
            [item["requested"].value for item in resolved_deliverables],
            global_reference_target,
        )
        # 分配预算沿用同一权重函数拆分到各交付物；只有预算高于验收下限
        # 时才计算，避免与下限配额完全一致造成误导性的两套数字。
        budget_quotas = (
            _global_deliverable_citation_quotas(
                [item["requested"].value for item in resolved_deliverables],
                citation_allocation_budget,
            )
            if citation_allocation_budget > global_reference_target
            else {}
        )
        globally_allocated_ids: set[str] = set()
        # 每轮生成重置路线并入诊断：build_writing_plan 会为每份交付物追加事件。
        state["route_merge_diagnostics"] = []
        visible_ordinal = 0
        # 每份正式交付物都由装配层统一编号；规划器与领域主题不携带编号。
        number_primary_sections = True
        for item in resolved_deliverables:
            # 分钟级写作循环的取消检查点：每份交付物之间响应取消，
            # 不再让用户等全部正文写完。
            if should_cancel and should_cancel():
                from app.agent.graph import AgentCancelledError

                raise AgentCancelledError("任务已在交付物生成间隙取消")
            requested_dtype = item["requested"]
            dtype = item["effective"]
            readiness = item["readiness"]
            downgrade_reasons = item["downgrade_reasons"]
            if not readiness.ready:
                continue
            if dtype in generated_types:
                continue
            plan = build_writing_plan(dtype, state, llm=llm)
            visible_ordinal += 1
            plan = _with_display_number(
                plan,
                visible_ordinal,
                enabled=number_primary_sections,
            )
            quota = citation_quotas.get(requested_dtype.value, 0)
            if quota:
                plan.citation_policy["minimum_unique_references"] = min(
                    quota,
                    len({
                        str(paper_id)
                        for section in plan.sections
                        for paper_id in section.supporting_paper_ids
                        if paper_id
                    }),
                )
            from app.core.config import get_settings
            citation_plan = _plan_citation_allocation(
                state=state,
                llm=(
                    llm
                    if get_settings().enable_llm_citation_planning
                    else None
                ),
                ranked_papers=state.get("ranked_papers") or state.get("paper_cards") or [],
                required_count=int(
                    plan.citation_policy.get("minimum_unique_references") or 0
                ),
                writing_plan=plan,
                excluded_ids=globally_allocated_ids,
                selection_target=budget_quotas.get(requested_dtype.value) or 0,
            )
            # Writer 和确定性 fallback 都消费同一个、已经归一化为 paper_id
            # 的逐章节计划，不能再把 LLM 返回的论文序号当作可选提示。
            state["citation_allocation_plan"] = citation_plan
            globally_allocated_ids.update(
                str(paper_id)
                for section in citation_plan.get("sections") or []
                for paper_id in section.get("paper_ids") or []
                if paper_id
            )
            allocation_plans.append({
                "deliverable_type": dtype.value,
                **citation_plan,
            })
            effective_llm = None if state.get("conservative_regeneration") else llm
            text = write_deliverable(plan, state, llm=effective_llm)
            validation = validate_deliverable(text, plan, state)
            plans.append(plan.model_dump(mode="json"))
            validations.append(validation)
            generated_types.add(dtype)
            outputs.append(text)

        # 跨交付物引用并集兜底：分配层允许为缺失研究路线复用已引论文，
        # 逐节 LLM 改写也可能丢弃个别计划引用，两者都会让两份正文出现
        # 重叠（如 10+30 中重叠 2 篇、全局唯一仅 38/40）。此时把已分配但
        # 未被引用的论文补写进所属交付物，而不是留给门禁直接拦截。
        outputs, validations = _backfill_global_citation_union(
            state,
            llm=None if state.get("conservative_regeneration") else llm,
            outputs=outputs,
            validations=validations,
            plans=plans,
            allocation_plans=allocation_plans,
        )

        state["deliverable_readiness"] = readiness_results
        state["writing_plans"] = plans
        state["citation_allocation_plans"] = allocation_plans
        state["deliverable_validation"] = validations
        state["deliverable_downgrades"] = downgrades
        state["generation_blocked"] = not bool(plans)
        state["review"] = "\n\n".join(outputs)
        state["body"] = state["review"]
        if requested == [CoreDeliverableType.RELATED_WORK.value]:
            state["related_work"] = state["review"]
        append_step(
            state,
            "generate_deliverables",
            "success",
            tool_name="write_deliverable",
            input_data={
                "deliverables": requested,
                "papers": len(state.get("paper_cards") or []),
                "themes": len(state.get("theme_synthesis") or []),
            },
            output_data={
                "writing_plans": len(plans),
                "route_merge_diagnostics": state.get("route_merge_diagnostics") or [],
                "readiness": readiness_results,
                "validations": validations,
                "writer_diagnostics": state.get("writer_diagnostics") or [],
                "writer_section_diagnostics": (
                    state.get("writer_section_diagnostics") or []
                ),
                "answer_preview": _preview_text(state["review"], 800),
            },
            duration_ms=int((time.time() - t0) * 1000),
        )
    except Exception as exc:
        from app.agent.graph import AgentCancelledError

        if isinstance(exc, AgentCancelledError):
            # 协作式取消必须继续向上传播，不能被包装成生成失败。
            raise
        logger.error("generate_deliverables_node failed: %s", exc)
        # P1 集成：LLM 生成/写作管线失败可降级（fallback 模板），用
        # LLMGenerationError 统一记录结构化上下文；errors 仍写字符串。
        from app.agent.exceptions import LLMGenerationError

        error = LLMGenerationError(str(exc), step="generate_deliverables", original_error=exc)
        state.setdefault("errors", []).append(f"generate_deliverables: {exc}")
        append_step(state, "generate_deliverables", "failed", error=str(exc), output_data=error.to_dict())
        # Writer 自身已经负责 LLM 失败时的模板降级；能逃逸到这里的异常属于
        # 管线或数据契约错误，必须让后台任务失败，不能伪装成零篇检索结果。
        raise error from exc
    return state


def _research_status_evidence_disclosure(
    cards: list[dict[str, Any]],
) -> str:
    """披露研究现状实际使用的全文/摘要证据分布。"""
    counts = {
        "full_text": 0,
        "partial_full_text": 0,
        "abstract": 0,
    }
    for card in cards:
        level = str(getattr(
            (card.get("evidence_state") or {}).get("access_level")
            or card.get("evidence_source"),
            "value",
            (card.get("evidence_state") or {}).get("access_level")
            or card.get("evidence_source"),
        ))
        if level in counts and card.get("quality_status") != "invalid":
            counts[level] += 1
    abstract_count = counts["abstract"]
    full_count = counts["full_text"] + counts["partial_full_text"]
    if not abstract_count:
        return ""
    if full_count:
        scope = (
            f"当前证据池包含 {full_count} 篇全文或部分全文论文、"
            f"{abstract_count} 篇摘要论文。"
        )
    else:
        scope = (
            f"当前可用的 {abstract_count} 篇论文均为摘要级证据；"
            "PDF 下载或解析失败不阻断研究现状生成。"
        )
    return (
        "> **证据范围说明：** "
        + scope
        + "摘要级论文仅用于原文明确报告的研究问题、方法概述和主要发现，"
        "不用于详细模型结构、数据划分、消融实验、作者局限或公平指标比较。"
    )


def _add_scope_disclosure(text: str, scope: Dict[str, Any], language: str) -> str:
    """在正文开头明确用户确认的概念边界，避免把相邻任务混为一谈。"""
    label = str(scope.get("label") or "").strip()
    description = str(scope.get("description") or "").strip()
    if not text or not label:
        return text
    if language == "zh":
        disclosure = f"**研究范围说明：** 本文按“{label}”界定主题"
        if description:
            disclosure += f"，即{description.rstrip('。')}"
        disclosure += "。相邻含义仅在与该范围存在直接证据关系时作为背景纳入。"
    else:
        disclosure = f"**Scope:** This review uses the “{label}” interpretation"
        if description:
            disclosure += f": {description.rstrip('.')}"
        disclosure += ". Neighboring meanings are included only when directly supported."
    lines = text.splitlines()
    insert_at = 1 if lines and lines[0].lstrip().startswith("#") else 0
    lines[insert_at:insert_at] = ["", disclosure, ""]
    return "\n".join(lines)


def _infer_evidence_role(paper: Dict[str, Any]) -> str:
    """根据论文标题和筛选决策推断其在综述中的证据角色。

    优先采用聚类阶段已写入卡片的 ``evidence_role``（单一事实来源），
    未写入时按标题关键词 + 筛选决策兜底推断。

    Returns:
        "survey" | "method" | "benchmark" | "application"
    """
    preset = paper.get("evidence_role")
    if preset in ("survey", "method", "benchmark", "application"):
        return preset
    title = str(paper.get("title", "")).lower()
    if any(kw in title for kw in [
        "survey", "review", "综述", "comprehensive", "systematic",
        "bibliometric", "meta-analysis", "元分析",
    ]):
        return "survey"
    if any(kw in title for kw in [
        "dataset", "benchmark", "数据集", "基准",
    ]):
        return "benchmark"
    from app.tools.paper_rerank import RULE_SCREENED_RESERVE

    screening_decision = str(paper.get("_screening_decision", ""))
    if screening_decision in ("uncertain", RULE_SCREENED_RESERVE):
        return "application"
    return "method"



# ============================================================
# Final Answer 节点
# ============================================================
@node(name="final_answer", category="execution", description="组装最终输出")
@optional(
    "ranked_papers", "paper_details", "review", "related_work", "introduction",
    "references", "writing_plans", "quality_gate", "generation_blocked"
)
@provides("answer", "body")
def final_answer_node(state: "ResearchAgentState") -> "ResearchAgentState":
    """组装最终输出。"""
    t0 = time.time()
    try:
        _apply_final_quality_gate(state)
        gate = state.get("quality_gate") or {}
        can_release_draft = (
            gate.get("passed") is True
            or gate.get("draft_released") is True
        )
        state["body"] = (
            state.get("review")
            or state.get("related_work")
            or state.get("introduction")
            or ""
        ) if can_release_draft else ""
        answer = _assemble_answer(state)
        state["answer"] = answer
        gate = state.get("quality_gate") or {}
        final_status = (
            "partial"
            if gate.get("passed") is False and gate.get("partial_success")
            else "blocked"
            if gate.get("passed") is False
            else "success"
        )
        append_step(
            state, "final_answer", final_status,
            input_data={
                "threshold_policy": get_review_threshold_policy().snapshot(),
                "intent": state.get("intent"),
                "review_length": len(state.get("review", "")),
                "references": len(state.get("references") or []),
                "paper_cards": len(state.get("paper_cards") or []),
                "errors": state.get("errors", []),
            },
            output_data={
                "answer_length": len(answer),
                "answer_preview": _preview_text(answer, limit=1200),
            },
            duration_ms=int((time.time() - t0) * 1000),
        )
    except Exception as e:
        from app.agent.exceptions import DegradableAgentError

        error = DegradableAgentError(str(e), step="final_answer", original_error=e)
        logger.error("final_answer_node failed: %s", error.message)
        state.setdefault("errors", []).append(f"final_answer: {e}")
        state["answer"] = f"执行过程中出现错误：{chr(10).join(state.get('errors', []))}"
        append_step(state, "final_answer", "failed", error=str(e))
    return state


def _backfill_global_citation_union(
    state: "ResearchAgentState",
    llm,
    outputs: list[str],
    validations: list[dict[str, Any]],
    plans: list[dict[str, Any]],
    allocation_plans: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    """把已分配但未被引用的论文补写进所属交付物，保证全局引用并集达标。

    引用配额按交付物分配（如 10+30=40），但主题代表复用和逐节 LLM 改写
    可能造成两份正文重叠、唯一并集不足用户硬性要求。这里对拥有未引用
    计划论文的交付物做一次定向 LLM 补写：论文必须自然融入既有段落，
    不得新增凑数段落；补写结果必须保留原有引用集合并重新通过交付物
    校验才会被接受，否则保留原稿。
    """
    required = int(state.get("required_reference_count") or 0)
    if required <= 0 or llm is None or not outputs:
        return outputs, validations

    from app.core.citation_syntax import extract_citation_ids, normalize_citation_syntax
    from app.schemas.deliverable_schema import WritingPlan
    from app.tools.validate_deliverable import validate_deliverable

    cited_union: set[str] = set()
    for text in outputs:
        cited_union.update(extract_citation_ids(text))
    if len(cited_union) >= required:
        return outputs, validations

    cards_by_id = {
        str(card.get("paper_id") or ""): card
        for card in state.get("paper_cards") or []
        if card.get("paper_id")
    }
    outputs = list(outputs)
    validations = list(validations)
    for index, allocation in enumerate(allocation_plans):
        if index >= len(outputs) or index >= len(plans) or len(cited_union) >= required:
            break
        allocated_ids = list(dict.fromkeys(
            str(paper_id)
            for section in allocation.get("sections") or []
            for paper_id in section.get("paper_ids") or []
            if paper_id
        ))
        shortfall = required - len(cited_union)
        missing_ids = [
            paper_id for paper_id in allocated_ids
            if paper_id not in cited_union and paper_id in cards_by_id
        ][:shortfall]
        if not missing_ids:
            continue

        paper_lines = []
        for paper_id in missing_ids:
            card = cards_by_id.get(paper_id) or {}
            summary = " ".join(filter(None, [
                str(card.get("method") or ""),
                str(card.get("contributions") or ""),
                str(card.get("research_problem") or ""),
            ]))[:200]
            paper_lines.append(
                f"- [{paper_id}] {card.get('title') or '未题名论文'}"
                f"（{card.get('year') or '年份未知'}）：{summary}"
            )
        papers_block = "\n".join(paper_lines)
        prompt = (
            "你是中文学术编辑。下面正文的唯一引用数未达到硬性要求，"
            "请把列出的论文自然融入最相关的段落：可以并入已有复合引用，"
            "也可以补一句基于要点的综合转述，但不得新增独立小节或文末凑数段落，"
            "不得改动既有引用标记和事实表述。引用标记必须原样保留 [paper_id] 形式。"
            "请完整输出修改后的全文，不要任何解释。\n\n"
            f"需补入的论文：\n{papers_block}\n\n正文：\n{outputs[index]}"
        )
        try:
            revised = llm.complete(
                prompt,
                temperature=0.05,
                operation=(
                    "backfill_citations:"
                    f"{allocation.get('deliverable_type') or index}"
                ),
                thinking_enabled=True,
            )
        except Exception as exc:
            logger.warning("Citation backfill LLM call failed: %s", exc)
            continue
        if not revised or not revised.strip():
            continue

        valid_ids = set(cards_by_id)
        revised = normalize_citation_syntax(revised.strip(), valid_ids)
        new_ids = set(extract_citation_ids(revised, valid_ids=valid_ids))
        old_ids = set(
            extract_citation_ids(outputs[index], valid_ids=valid_ids)
        )
        # 补写必须无损：新增了缺失论文且没有丢失任何既有引用。
        if not set(missing_ids) <= new_ids or not old_ids <= new_ids:
            logger.info(
                "Citation backfill rejected: coverage not monotonic "
                "(deliverable=%s)",
                allocation.get("deliverable_type"),
            )
            continue

        # 防截断保护：补写后的正文长度不得显著小于原正文
        if len(revised.strip()) < int(len(outputs[index].strip()) * 0.85):
            logger.info(
                "Citation backfill rejected: output truncated (len=%d vs orig=%d)",
                len(revised),
                len(outputs[index]),
            )
            continue

        # 结构完整性保护：原正文中的所有 Markdown 标题必须全部保留
        orig_headings = re.findall(r"^(#{2,4}\s+[^\n]+)", outputs[index], re.M)
        rev_headings = re.findall(r"^(#{2,4}\s+[^\n]+)", revised, re.M)
        if set(orig_headings) - set(rev_headings):
            logger.info(
                "Citation backfill rejected: missing headings %s",
                sorted(set(orig_headings) - set(rev_headings)),
            )
            continue

        try:
            plan_obj = WritingPlan.model_validate(plans[index])
            new_validation = validate_deliverable(revised, plan_obj, state)
        except Exception as exc:
            logger.warning("Citation backfill validation failed: %s", exc)
            continue
        old_errors = set(validations[index].get("errors") or [])
        new_errors = set(new_validation.get("errors") or [])
        if new_errors - old_errors:
            logger.info(
                "Citation backfill rejected: new validation errors %s",
                sorted(new_errors - old_errors),
            )
            continue

        outputs[index] = revised
        validations[index] = new_validation
        cited_union.update(new_ids)
        logger.info(
            "Citation backfill applied: deliverable=%s added=%d union=%d/%d",
            allocation.get("deliverable_type"),
            len(set(missing_ids) & new_ids),
            len(cited_union),
            required,
        )
    return outputs, validations


def _plan_citation_allocation(
    state: "ResearchAgentState",
    llm,
    ranked_papers: list,
    required_count: int,
    writing_plan=None,
    excluded_ids: set[str] | None = None,
    selection_target: int = 0,
) -> dict:
    """生成可执行的逐章节引用计划，并拒绝用低相关论文补齐数量。

    LLM 只负责提供语义分配建议；最终输出统一使用 ``paper_id``，并由
    写作计划限定可分配证据。相关证据不足时由写作前门禁阻止生成。

    ``selection_target`` 是本交付物的引用分配预算（来自
    ``_citation_allocation_budget`` 的权重拆分）：高于 ``required_count``
    时按预算多分配证据，让写手能引用超过最低量的论文；验收仍以
    ``minimum_unique_references``（= 用户下限）为准。缺省 0 保持旧的
    精确按下限分配行为。
    """
    from app.core.json_utils import parse_json_object
    from app.schemas.deliverable_schema import CoreDeliverableType, WritingPlan

    plan = (
        writing_plan
        if isinstance(writing_plan, WritingPlan)
        else WritingPlan.model_validate(writing_plan)
        if writing_plan
        else None
    )
    cards_by_id = {
        str(card.get("paper_id") or ""): card
        for card in state.get("paper_cards") or []
        if card.get("paper_id") and card.get("quality_status") != "invalid"
    }
    allowed_ids = {
        str(paper_id)
        for section in (plan.sections if plan else [])
        for paper_id in section.supporting_paper_ids
        if str(paper_id) in cards_by_id
    } or set(cards_by_id)

    ordered_ids: list[str] = []
    seen_ids: set[str] = set()
    for paper in [*ranked_papers, *(state.get("paper_cards") or [])]:
        paper_id = str(paper.get("paper_id") or "")
        if paper_id in allowed_ids and paper_id not in seen_ids:
            seen_ids.add(paper_id)
            ordered_ids.append(paper_id)
    target = min(max(0, int(required_count)), len(ordered_ids))
    selection_budget = max(0, int(selection_target or 0))
    if selection_budget > target:
        # 用户下限之上有预算余量时按预算多分配（仍受实际可用池约束），
        # 避免把“不少于 N 篇”的 N 当精确目标执行——那会让每次输出都
        # 恰好 N 篇引用，浪费已筛选的证据。
        target = min(selection_budget, len(ordered_ids))
    all_ordered_ids = list(ordered_ids)
    excluded_ids = {str(value) for value in (excluded_ids or set()) if value}
    fresh_ids = [
        paper_id for paper_id in ordered_ids
        if paper_id not in excluded_ids
    ]
    # 优先使用尚未被其他交付物占用的论文。只有剩余论文确实不足时，
    # 才允许复用，确保多个交付物的全局并集能够达到用户要求。
    if len(fresh_ids) >= target:
        ordered_ids = fresh_ids
    else:
        ordered_ids = [
            *fresh_ids,
            *[
                paper_id for paper_id in ordered_ids
                if paper_id in excluded_ids
            ],
        ]

    # 按动态研究路线轮转选择最低引用集合，避免单一路线占满全部名额，而其他有证据支持的路线永远进不了 Writer。
    taxonomy = state.get("dynamic_taxonomy") or {}
    fallback_theme_ids = {
        str(theme.get("theme_id") or "")
        for theme in taxonomy.get("themes") or []
        if theme.get("theme_id")
        and re.search(
            r"其他|其它|未分类|兜底|other|misc",
            str(theme.get("name") or ""),
            re.I,
        )
    }
    theme_order = [
        str(theme.get("theme_id") or "")
        for theme in taxonomy.get("themes") or []
        if theme.get("theme_id")
        and str(theme.get("theme_id") or "") not in fallback_theme_ids
    ]
    theme_by_paper = {
        str(item.get("paper_id") or ""): str(item.get("primary_theme_id") or "")
        for item in taxonomy.get("assignments") or []
        if item.get("paper_id")
    }
    preferred_ordered_ids = [
        paper_id
        for paper_id in ordered_ids
        if theme_by_paper.get(paper_id) not in fallback_theme_ids
    ]
    fallback_ordered_ids = [
        paper_id
        for paper_id in ordered_ids
        if theme_by_paper.get(paper_id) in fallback_theme_ids
    ]
    # 当主要研究路线已有足够论文时，兜底主题不参与最低引用集合；
    # 只有主要路线数量确实不足，才把兜底论文放到最后补足。
    selection_order = [*preferred_ordered_ids, *fallback_ordered_ids]
    theme_buckets = {
        theme_id: [
            paper_id for paper_id in preferred_ordered_ids
            if theme_by_paper.get(paper_id) == theme_id
        ]
        for theme_id in theme_order
    }
    selected_ids: list[str] = []
    bucket_offsets = {theme_id: 0 for theme_id in theme_order}
    while len(selected_ids) < target:
        added = False
        for theme_id in theme_order:
            bucket = theme_buckets.get(theme_id) or []
            offset = bucket_offsets[theme_id]
            if offset >= len(bucket):
                continue
            paper_id = bucket[offset]
            bucket_offsets[theme_id] += 1
            if paper_id not in selected_ids:
                selected_ids.append(paper_id)
                added = True
            if len(selected_ids) >= target:
                break
        if not added:
            break
    for paper_id in selection_order:
        if len(selected_ids) >= target:
            break
        if paper_id not in selected_ids:
            selected_ids.append(paper_id)

    # 前一个交付物可能已用完某个小众但重要的研究路线。
    # 后续“研究现状”若完全排除这些论文，会出现有主题标题却
    # 没有任何证据的空节。允许每个缺失路线复用一篇代表论文，
    # 同时保留 target 篇未使用论文，因此不降低整份正文的全局引用并集。
    if excluded_ids and plan and plan.deliverable_type == CoreDeliverableType.RESEARCH_STATUS:
        represented_themes = {
            theme_by_paper.get(paper_id) for paper_id in selected_ids
        }
        for theme_id in theme_order:
            if theme_id in represented_themes:
                continue
            reusable = next(
                (
                    paper_id for paper_id in all_ordered_ids
                    if paper_id in excluded_ids
                    and theme_by_paper.get(paper_id) == theme_id
                ),
                None,
            )
            if reusable and reusable not in selected_ids:
                selected_ids.append(reusable)

    # 章节级下限的选择兜底：轮转选择按 dynamic_taxonomy 主题分桶，而写作
    # 计划的路线小节可能来自证据要求路线或合并路线（section_id 不在
    # taxonomy 里），两者不同构时某条正式路线只会被选中一篇；独占归属又
    # 让它无法从别的小节借调证据，最终写成"本节纳入 1 篇文献"（2026-08-30
    # 实测）。这里只按该小节自身的授权论文补选，不复用他节论文、不放宽
    # 全局唯一引用要求；授权池本身不足时留给下面的缺口诊断如实报告。
    if plan:
        for section in plan.sections:
            floor = int(section.minimum_unique_references or 0)
            if floor <= 0:
                continue
            own_ids = [
                str(paper_id) for paper_id in section.supporting_paper_ids
                if str(paper_id) in cards_by_id
            ]
            chosen = [paper_id for paper_id in own_ids if paper_id in selected_ids]
            if len(chosen) >= floor:
                continue
            for candidate_pool in (ordered_ids, all_ordered_ids):
                for paper_id in own_ids:
                    if len(chosen) >= floor:
                        break
                    if paper_id in selected_ids or paper_id not in candidate_pool:
                        continue
                    selected_ids.append(paper_id)
                    chosen.append(paper_id)
                if len(chosen) >= floor:
                    break

    details_by_id = {
        str(paper.get("paper_id") or ""): paper
        for paper in state.get("paper_details") or []
        if paper.get("paper_id")
    }
    summary_ids = [
        *selected_ids,
        *[
            paper_id for paper_id in ordered_ids
            if paper_id not in selected_ids
        ][:10],
    ]
    paper_summaries = [
        {
            "idx": i + 1,
            "paper_id": paper_id,
            "title": (
                details_by_id.get(paper_id, {}).get("title")
                or cards_by_id.get(paper_id, {}).get("title")
                or ""
            ),
            "year": (
                details_by_id.get(paper_id, {}).get("year")
                or cards_by_id.get(paper_id, {}).get("year")
            ),
            "abstract": (
                details_by_id.get(paper_id, {}).get("abstract") or ""
            )[:200],
        }
        for i, paper_id in enumerate(summary_ids)
    ]

    structural_ids = {
        "scope_definition", "search_scope", "evidence_statement", "abstract", "conclusion"
    }
    section_specs: list[dict[str, Any]] = []
    if plan:
        for section in plan.sections:
            section_allowed = [
                str(paper_id) for paper_id in section.supporting_paper_ids
                if str(paper_id) in selected_ids
            ]
            if section_allowed and section.id not in structural_ids:
                section_specs.append({
                    "section_id": section.id,
                    "section": section.title,
                    "purpose": section.purpose,
                    "allowed_ids": section_allowed,
                })
    if not section_specs and selected_ids:
        section_specs = [{
            "section_id": "body",
            "section": "正文",
            "purpose": "按证据组织研究现状",
            "allowed_ids": list(selected_ids),
        }]

    proposed: dict[str, list[str]] = {
        spec["section_id"]: [] for spec in section_specs
    }
    summary_id_by_index = {
        item["idx"]: item["paper_id"] for item in paper_summaries
    }
    section_by_id = {
        spec["section_id"]: spec for spec in section_specs
    }
    section_id_by_title = {
        spec["section"]: spec["section_id"] for spec in section_specs
    }

    prompt = f"""你是学术写作规划助手。给定以下 {len(paper_summaries)} 篇论文和写作章节，
请为每个章节分配与其研究问题直接相关的论文。
用户确认的研究重点：
{json.dumps(state.get("selected_scope") or {}, ensure_ascii=False)}

论文列表：
{json.dumps(paper_summaries, ensure_ascii=False, indent=2)}

写作章节：{json.dumps([
    {
        "section_id": item["section_id"],
        "section": item["section"],
        "purpose": item.get("purpose") or "",
    }
    for item in section_specs
], ensure_ascii=False)}

请返回 JSON：
{{"sections": [{{"section_id": "章节ID", "section": "章节名", "paper_ids": ["p1", "p2"]}}]}}

要求：
1. 只使用论文列表中的 paper_id
2. 每篇论文只分配给一个与其研究问题最匹配的章节，禁止重复分配
3. 优先匹配章节目的和用户研究重点；不得为均匀分配而将与当前主题无直接关系的论文硬塞进任何章节。相关性须依据用户语义框架和论文证据判断。
4. 允许低相关论文不分配；不得为了达到数量要求而强行塞入章节
5. 综述/调研类论文（标题含 review/survey/综述）优先分配到研究现状主题章节，作为领域格局与宏观背景的引用锚点
"""

    if llm is not None and paper_summaries and section_specs:
        try:
            from app.core.config import get_settings
            response = llm.complete(
                prompt,
                response_format="json_object",
                temperature=0.1,
                timeout=get_settings().llm_control_plane_timeout,
                operation="citation_allocation_planning",
            )
            result = parse_json_object(response)
            for raw_section in result.get("sections") or []:
                section_id = str(raw_section.get("section_id") or "")
                if section_id not in section_by_id:
                    section_id = section_id_by_title.get(
                        str(raw_section.get("section") or ""), ""
                    )
                if section_id not in section_by_id:
                    continue
                raw_ids = [
                    str(paper_id) for paper_id in raw_section.get("paper_ids") or []
                ]
                raw_ids.extend(
                    summary_id_by_index.get(index, "")
                    for index in raw_section.get("paper_indices") or []
                    if isinstance(index, int)
                )
                allowed = set(section_by_id[section_id]["allowed_ids"])
                proposed[section_id].extend(
                    paper_id for paper_id in raw_ids
                    if paper_id in allowed and paper_id not in proposed[section_id]
                )
        except Exception as exc:
            logger.warning(
                "Citation allocation planning failed: %s; using deterministic allocation",
                exc,
            )

    # 将 LLM 结果当作语义偏好，再确定性编译为“单篇单节”计划。
    # 旧实现允许同一论文被分配到多节，且只按当前章节长度补齐，
    # 会把语义不相关的证据硬塞进短章节。
    preferred_sections: dict[str, list[str]] = {paper_id: [] for paper_id in selected_ids}
    for section_id, paper_ids in proposed.items():
        for paper_id in paper_ids:
            if (
                paper_id in preferred_sections
                and section_id not in preferred_sections[paper_id]
            ):
                preferred_sections[paper_id].append(section_id)

    proposed = {spec["section_id"]: [] for spec in section_specs}
    capacity = max(3, (target + max(1, len(section_specs)) - 1) // max(1, len(section_specs)) + 2)
    for paper_id in selected_ids:
        eligible = [
            spec for spec in section_specs if paper_id in spec["allowed_ids"]
        ]
        if not eligible:
            continue
        under_capacity = [
            spec for spec in eligible
            if len(proposed[spec["section_id"]]) < capacity
        ]
        candidates = under_capacity or eligible
        card = cards_by_id.get(paper_id) or details_by_id.get(paper_id) or {}
        chosen = max(
            candidates,
            key=lambda spec: (
                _citation_section_fit_score(card, spec),
                2.0 if spec["section_id"] in preferred_sections[paper_id] else 0.0,
                -0.8 * len(proposed[spec["section_id"]]),
                1.0 if spec["section_id"].startswith("theme_") else 0.0,
                -section_specs.index(spec),
            ),
        )
        proposed[chosen["section_id"]].append(paper_id)

    # 章节级最低引用优先读写作计划的章节契约（WritingSection
    # .minimum_unique_references），计划未携带契约时沿用"高引用量研究现状
    # 每节至少两篇"的既有口径。无可移动证据时不再静默保留稀疏分配，而是把
    # 缺口写成确定性诊断，由交付物校验与最终门禁如实报告。
    section_floor_deficits: list[dict[str, Any]] = []
    if plan:
        planned_floors = {
            str(section.id): int(section.minimum_unique_references or 0)
            for section in plan.sections
        }
        legacy_route_floor = (
            2
            if plan.deliverable_type == CoreDeliverableType.RESEARCH_STATUS
            and target >= 20
            else 0
        )
        minimum_by_section: dict[str, int] = {}
        for spec in section_specs:
            section_id = str(spec["section_id"])
            is_route_section = (
                section_id.startswith("theme_")
                or section_id in {"cross_route_comparison", "research_gaps"}
            )
            floor = planned_floors.get(section_id) or (
                legacy_route_floor if is_route_section else 0
            )
            if floor > 0:
                minimum_by_section[section_id] = floor
        for target_section_id, minimum in minimum_by_section.items():
            target_spec = section_by_id[target_section_id]
            while len(proposed[target_section_id]) < minimum:
                movable: list[tuple[float, str, str]] = []
                for donor_id, donor_ids in proposed.items():
                    if donor_id == target_section_id or len(donor_ids) <= minimum_by_section.get(donor_id, 1):
                        continue
                    donor_spec = section_by_id[donor_id]
                    for paper_id in donor_ids:
                        if paper_id not in target_spec["allowed_ids"]:
                            continue
                        card = cards_by_id.get(paper_id) or details_by_id.get(paper_id) or {}
                        gain = (
                            _citation_section_fit_score(card, target_spec)
                            - _citation_section_fit_score(card, donor_spec)
                        )
                        movable.append((gain, donor_id, paper_id))
                if not movable:
                    break
                _, donor_id, paper_id = max(movable, key=lambda item: item[0])
                proposed[donor_id].remove(paper_id)
                proposed[target_section_id].append(paper_id)
            if len(proposed[target_section_id]) < minimum:
                section_floor_deficits.append({
                    "section_id": target_section_id,
                    "section": target_spec["section"],
                    "required_unique_references": minimum,
                    "assigned_unique_references": len(proposed[target_section_id]),
                    "authorized_paper_count": len(target_spec["allowed_ids"]),
                })
        if section_floor_deficits:
            logger.warning(
                "Citation allocation left %d section(s) below the evidence floor: %s",
                len(section_floor_deficits),
                [item["section_id"] for item in section_floor_deficits],
            )

    already_assigned = {
        paper_id for paper_ids in proposed.values() for paper_id in paper_ids
    }

    normalized_sections = [
        {
            "section_id": spec["section_id"],
            "section": spec["section"],
            "paper_ids": proposed[spec["section_id"]],
        }
        for spec in section_specs
    ]
    logger.info(
        "Citation allocation plan normalized: target=%d assigned=%d sections=%d",
        target,
        len(already_assigned),
        len(normalized_sections),
    )
    return {
        "minimum_unique_references": target,
        "assigned_unique_references": len(already_assigned),
        "sections": normalized_sections,
        "section_floor_deficits": section_floor_deficits,
    }


def _citation_section_fit_score(
    card: dict[str, Any],
    section: dict[str, Any],
) -> float:
    """评估论文证据与章节功能的匹配度，仅用于约束引用分配。"""
    section_id = str(section.get("section_id") or "")
    section_text = " ".join([
        str(section.get("section") or ""),
        str(section.get("purpose") or ""),
    ]).lower()
    card_text = " ".join([
        str(card.get("title") or ""),
        str(card.get("research_problem") or ""),
        str(card.get("method") or ""),
        str(card.get("study_design") or ""),
        " ".join(str(value) for value in card.get("behavior_categories") or []),
    ]).lower()
    fields = {
        str(field)
        for field, claims in (card.get("field_claims") or {}).items()
        if claims
    }
    score = 0.0
    if section_id.startswith("theme_"):
        score += 8.0
    field_preferences = {
        "problem_context": {"research_problem": 3.0, "contributions": 1.0},
        "importance": {"contributions": 3.0, "results": 1.0, "research_problem": 1.0},
        "existing_approaches": {"method": 3.0, "dataset": 1.5, "results": 1.0},
        "research_need": {"limitations": 4.0, "research_problem": 1.0},
        "research_gaps": {"limitations": 4.0, "research_problem": 1.0},
        "cross_route_comparison": {"method": 2.0, "dataset": 1.5, "metrics": 1.5},
    }
    for field, weight in field_preferences.get(section_id, {}).items():
        if field in fields:
            score += weight
    keyword_groups = {
        "research_need": ["limitation", "challenge", "gap", "future", "局限", "挑战", "不足", "未来"],
        "research_gaps": ["limitation", "challenge", "gap", "future", "局限", "挑战", "不足", "空白"],
        "existing_approaches": ["model", "method", "network", "algorithm", "模型", "方法", "网络", "算法"],
        "importance": [
            "importance", "significance", "critical", "essential", "value", "demand",
            "drive", "application", "重要", "意义", "关键", "价值", "需求", "驱动", "应用", "前景",
        ],
    }
    score += 0.5 * sum(
        keyword in card_text
        for keyword in keyword_groups.get(section_id, [])
    )
    # 小量词项重合用于同分时的稳定排序，不作事实判定。
    section_tokens = set(re.findall(r"[a-z][a-z0-9_-]{2,}|[\u4e00-\u9fff]{2,4}", section_text))
    card_tokens = set(re.findall(r"[a-z][a-z0-9_-]{2,}|[\u4e00-\u9fff]{2,4}", card_text))
    score += min(2.0, 0.25 * len(section_tokens & card_tokens))
    return score


def _apply_final_quality_gate(state: "ResearchAgentState") -> None:
    """聚合生成后验证结果；失败时隔离草稿而不是继续正式交付。"""
    if state.get("intent") == "search_papers":
        return
    existing = state.get("quality_gate") or {}
    if (
        existing.get("passed") is False
        and existing.get("phase") == "pre_generation"
        and not state.get("best_effort_generation")
    ):
        return

    issues: list[dict[str, Any]] = []
    recovery: list[str] = []
    warnings: list[dict[str, Any]] = []
    if state.get("best_effort_generation"):
        forced_issues = list(state.get("forced_generation_issues") or [])
        taxonomy_validation = state.get("taxonomy_validation") or {}
        issues.append({
            "code": "user_accepted_best_effort_generation",
            "message": (
                "用户已确认基于当前证据直接生成；动态分类或部分证据重点未完全达标"
                if forced_issues or not taxonomy_validation.get("valid", False)
                else "用户已确认基于当前证据直接生成最佳可用草稿"
            ),
            "details": forced_issues,
        })
    if state.get("max_papers_explicit", False) and state.get("citation_validation"):
        requested = int(state.get("required_reference_count") or state.get("max_papers") or 0)
        consistency = state.get("claim_citation_consistency") or {}
        cited = int(
            state.get("unique_valid_cited_paper_count")
            if consistency
            else state.get("unique_cited_paper_count") or 0
        )
        # “至少引用 N 篇”是用户明确硬约束，不能再把 80% 当成达标。
        # 旧逻辑使 40 篇要求在 32 篇时被错误放行。
        effective_target = requested
        if requested and cited < effective_target:
            issues.append({
                "code": "minimum_cited_references_not_met",
                "message": f"要求正文至少引用 {requested} 篇（达标线 {effective_target} 篇），实际有效引用 {cited} 篇",
                "requested": requested,
                "actual": cited,
            })
            recovery.append("扩展检索或经用户确认降低最低引用篇数")

    selected_scope = state.get("selected_scope") or {}
    if selected_scope and state.get("reference_papers"):
        from app.tools.rank_papers import evaluate_scope_filter

        off_scope = []
        for paper in state.get("reference_papers") or []:
            passed, reason = evaluate_scope_filter(paper, selected_scope)
            if not passed:
                off_scope.append({
                    "title": str(paper.get("title") or ""),
                    "reason": reason,
                })
        if off_scope:
            issues.append({
                "code": "citation_scope_not_met",
                "message": f"有 {len(off_scope)} 条实际引用文献不满足已确认研究范围",
                "details": off_scope[:20],
            })
            recovery.append("移除越界引用并从已确认范围内重新分配证据")

    language_target = state.get("language_coverage_target") or {}
    if language_target.get("enabled"):
        from app.tools.language_router import detect_paper_language

        cited_papers = state.get("reference_papers") or []
        cited_zh = sum(
            1 for paper in cited_papers if detect_paper_language(paper) == "zh"
        )
        cited_en = sum(
            1 for paper in cited_papers if detect_paper_language(paper) == "en"
        )
        minimum_zh = int(language_target.get("minimum_zh") or 0)
        minimum_en = int(language_target.get("minimum_en") or 0)
        deficits = {
            "zh": max(0, minimum_zh - cited_zh),
            "en": max(0, minimum_en - cited_en),
        }
        state["language_coverage"] = {
            **language_target,
            "cited_zh": cited_zh,
            "cited_en": cited_en,
            "deficits": deficits,
            "satisfied": not any(deficits.values()),
        }
        if any(deficits.values()):
            issues.append({
                "code": "language_coverage_not_met",
                "message": (
                    "最终有效引用未满足双语覆盖契约："
                    f"中文 {cited_zh}/{minimum_zh}，英文 {cited_en}/{minimum_en}"
                ),
                "details": deficits,
            })
            recovery.append("按缺失语言补充检索与证据卡片，并重新执行引用分配")

    from app.tools.validate_deliverable import validate_final_review_integrity

    final_integrity = validate_final_review_integrity(
        state.get("review") or state.get("related_work") or state.get("introduction") or "",
        state,
    )
    state["final_review_integrity"] = final_integrity
    if not final_integrity.get("valid", True):
        integrity_errors = list(final_integrity.get("errors") or [])
        issues.append({
            "code": "final_text_integrity_not_met",
            "message": (
                "最终正文完整性检查未通过：" + "；".join(integrity_errors)
                if integrity_errors
                else "最终正文完整性检查未通过"
            ),
            "details": integrity_errors,
        })
        recovery.append("按章节重新生成并在引用检查前后各执行一次成文完整性验证")

    citation_validation = state.get("citation_validation") or {}
    unknown_publication_status = list(
        citation_validation.get("unknown_publication_status") or []
    )
    if unknown_publication_status:
        warnings.append({
            "code": "publication_status_unknown",
            "message": (
                f"有 {len(unknown_publication_status)} 条实际引用文献的出版状态尚未确认；"
                "不得据此宣称均已正式发表"
            ),
            "details": unknown_publication_status[:20],
        })
    if citation_validation and (
        not citation_validation.get("valid", True)
        or not citation_validation.get("metadata_quality_valid", True)
    ):
        suggestions = citation_validation.get("suggestions") or []
        metadata_errors = (
            list(citation_validation.get("incomplete_metadata") or [])
            + list(citation_validation.get("unverified_metadata") or [])
            + list(citation_validation.get("duplicate_dois") or [])
        )
        if metadata_errors:
            issues.append({
                "code": "citation_metadata_not_verified",
                "message": "参考文献元数据未通过最终核验",
                "details": metadata_errors,
            })
            recovery.append("补齐题名、作者、年份、来源及稳定标识后重新生成参考文献")
        missing_citations = citation_validation.get("missing_citations") or []
        # 复合引用（如 [1; 2; 3]）缺失属于格式问题，降级为警告
        # 仅当有非复合的缺失引用时才阻塞
        _composite_missing = [m for m in missing_citations if re.search(r"[;；,，、]", m)]
        _single_missing = [m for m in missing_citations if not re.search(r"[;；,，、]", m)]
        if _single_missing:
            issues.append({
                "code": "invalid_citations",
                "message": "正文存在无效、未闭合或无法映射的引用",
                "details": suggestions,
            })
            recovery.append("修复引用映射和参考文献元数据后重新验证")
        else:
            # 仅有复合引用缺失或元数据不完整，降级为警告
            warnings.append({
                "code": "incomplete_citation_metadata",
                "message": "部分参考文献元数据不完整或无法校验，但不阻塞生成",
                "details": suggestions,
            })

    quality = state.get("generation_quality") or {}
    # support_rate 为 None 时视为未计算，降级为 0（阻断而非放行）
    _raw_support = quality.get("support_rate")
    _support_rate = float(_raw_support) if _raw_support is not None else 0.0

    paper_cards = state.get("paper_cards") or []
    abstract_or_meta_count = sum(
        1 for card in paper_cards
        if (card.get("evidence_state") or {}).get("access_level") in ("abstract", "metadata_only", "title_and_keywords")
        or str(card.get("evidence_source") or "metadata") in ("metadata", "abstract")
    )
    policy = get_review_threshold_policy()
    is_abstract_dominant = len(paper_cards) > 0 and (abstract_or_meta_count / len(paper_cards)) >= policy.synthesis_abstract_dominance
    blocking_support_threshold = (policy.synthesis_abstract_support_rate
                                  if is_abstract_dominant
                                  else policy.synthesis_fulltext_support_rate)

    if quality and not quality.get("passed", True) and _support_rate < blocking_support_threshold:
        issues.append({
            "code": "claim_evidence_quality_not_met",
            "message": (
                f"主张—证据加权支持率为 {float(quality.get('support_rate') or 0.0):.1%}，"
                f"仍有 {int(quality.get('unsupported_claims') or 0)} 条事实性主张缺少充分证据"
            ),
        })
        recovery.append("删除、弱化或补证后重新执行逐句主张验证")
    elif quality and not quality.get("passed", True) and _support_rate >= blocking_support_threshold:
        warnings.append({
            "code": "claim_evidence_quality_low",
            "message": (
                f"主张—证据加权支持率为 {float(quality.get('support_rate') or 0.0):.1%}，"
                f"仍有 {int(quality.get('unsupported_claims') or 0)} 条事实性主张缺少充分证据（已降级为警告）"
            ),
        })

    # 主张—引用一致性：正文每句的引用应由其匹配主张的证据授权。此前该指标只由
    # graph 写一条 warning 日志，实测 9/43 句错配既不阻断也不降级（规则 5）。
    consistency = state.get("claim_citation_consistency") or {}
    mismatched_sentences = int(consistency.get("inconsistent_sentences") or 0)
    if mismatched_sentences:
        consistency_rate = float(consistency.get("consistency_rate") or 0.0)
        consistency_samples = [
            str(item.get("sentence") or "")[:120]
            for item in consistency.get("inconsistent_samples") or []
        ][:3]
        # 引用错配不是可按总体比例稀释的软质量指标：哪怕只有一句，引用也没有
        # 获得对应主张的证据授权，必须阻断交付并触发定点修复。
        issues.append({
            "code": "claim_citation_consistency_not_met",
            "message": (
                f"正文引用与主张授权证据的一致率为 {consistency_rate:.1%}，"
                f"有 {mismatched_sentences} 句引用了未由匹配主张授权的文献"
            ),
            "details": consistency_samples,
        })
        recovery.append("按主张计划重新分配引用，或移除未获授权的引用后重新生成")

    invalid_deliverables = [
        validation for validation in state.get("deliverable_validation") or []
        if not validation.get("valid", False)
    ]
    failed_sections = [
        item
        for run in state.get("writer_section_diagnostics") or []
        for item in run.get("sections") or []
        if item.get("status") in {"evidence_limited", "fallback"}
    ]
    if failed_sections:
        issues.append({
            "code": "section_generation_failed",
            "message": "部分章节未能完成可靠改写，已按证据边界降级",
            "details": [
                {
                    "section_id": item.get("section_id"),
                    "status": item.get("status"),
                    "errors": item.get("errors") or [],
                }
                for item in failed_sections
            ],
        })
        recovery.append("仅重写失败章节，或补充该章节所需证据后重新生成")
    if invalid_deliverables:
        issues.append({
            "code": "deliverable_structure_invalid",
            "message": "交付物结构或写作边界检查未通过",
            "details": [error for item in invalid_deliverables for error in item.get("errors") or []],
        })
        recovery.append("修复章节、主题、引用或越权表述后重新验证")

    # 引用数量缺口：成文后实际引用数低于用户要求时显式提示，
    # 避免草稿静默低于“不少于 N 篇”的显式约束。
    required_refs = int(state.get("required_reference_count") or 0)
    reference_count = len(state.get("references") or [])
    if required_refs and reference_count and reference_count < required_refs:
        warnings.append({
            "code": "citation_count_shortfall",
            "message": (
                f"草稿实际引用 {reference_count} 篇，低于要求的 {required_refs} 篇"
            ),
        })

    # 路线级证据缺口：定向补检索耗尽后仍未达目标的方向如实报告。这些路线随后
    # 由 merge_weak_routes_for_writing 合并，不单独阻断交付。
    route_deficits = [
        item for item in (state.get("route_evidence_deficits") or [])
        if int(item.get("core_evidence_deficit") or 0) > 0
    ]
    section_deficits = [
        deficit
        for plan in (state.get("citation_allocation_plans") or [])
        for deficit in (plan.get("section_floor_deficits") or [])
        if deficit
    ]
    if route_deficits or section_deficits:
        details = [
            f"{item.get('route_name') or item.get('route_id')}："
            f"{item.get('core_evidence_count')}/{item.get('target_core_evidence')} 篇"
            for item in route_deficits
        ]
        warnings.append({
            "code": "route_evidence_target_not_met",
            "message": (
                "以下研究方向的证据篇数未达目标，已按现有证据合并或降级："
                + "；".join(details)
                if details
                else "部分章节的授权证据不足，已按现有证据分配"
            ),
            "routes": route_deficits,
            "sections": section_deficits,
        })

    if issues:
        draft = state.get("review") or state.get("related_work") or state.get("introduction") or ""
        if draft:
            state["quarantined_draft"] = draft
        state["generation_blocked"] = True
        best_effort_released = bool(state.get("best_effort_generation"))
        state["quality_gate"] = {
            "passed": False,
            "draft_available": bool(draft),
            "draft_released": best_effort_released and bool(draft),
            "draft_disposition": (
                "released_best_effort" if best_effort_released and draft else "quarantined"
            ) if draft else "none",
            "partial_success": best_effort_released and bool(draft),
            "phase": "post_generation",
            "blocking_issues": issues,
            "warnings": warnings,
            "recovery_options": list(dict.fromkeys(recovery)),
        }
    else:
        state["quality_gate"] = {
            "passed": True,
            "draft_available": bool(state.get("review")),
            "draft_released": True,
            "draft_disposition": "approved",
            "partial_success": False,
            "phase": "post_generation",
            "blocking_issues": [],
            "warnings": warnings,
            "recovery_options": [],
        }


def _global_gate_warning_banner(state: "ResearchAgentState") -> str:
    """渲染全局证据门的阻断缺口提示；只展示 blocking 缺口。

    non_blocking 缺口不在此展示，仍可通过 API 的 global_evidence_gate 字段读取。
    """
    gate = state.get("global_evidence_gate") or {}
    if gate.get("status") != "EVALUATED" or gate.get("passed", True):
        return ""
    lines = []
    for deficit in gate.get("deficits") or []:
        if deficit.get("severity") != "blocking":
            continue
        d_type = deficit.get("type")
        if d_type == "citation_count":
            lines.append(
                f"检索到 {deficit.get('available')}/{deficit.get('required')} 篇论文，"
                "未满足引用数量要求。"
            )
        elif d_type == "recency":
            lines.append(
                f"仅 {deficit.get('available')}/{deficit.get('required')} 篇论文落在要求的年份范围内"
                f"（{state.get('start_year')}-{state.get('end_year')}）。"
            )
        elif d_type == "route_coverage":
            lines.append("研究路线证据分布明显不均，部分路线支撑论文过少。")
        elif d_type == "quality":
            lines.append(f"同行评审论文占比约 {deficit.get('ratio', 0):.0%}，未达到要求。")
    non_blocking = sum(
        1 for d in gate.get("deficits") or [] if d.get("severity") != "blocking"
    )
    if non_blocking:
        lines.append(f"另有 {non_blocking} 项非阻断性提示。")
    if not lines:
        return ""
    body = "\n".join(f"> {line}" for line in lines)
    return (
        "\n\n> ⚠️ **全局证据门提示（部分满足）**\n>\n"
        f"{body}\n>\n"
        "> 综述已按现有证据继续生成；建议补充检索或调整要求后重新生成。\n\n"
    )


def _assemble_answer(state: "ResearchAgentState") -> str:
    """根据 review + references 组装最终回复文本。"""
    parts: list[str] = []

    quality_gate = state.get("quality_gate") or {}
    if quality_gate.get("passed") is False:
        # 只有显式 best-effort 授权才可发布未完全通过门禁的草稿。
        if quality_gate.get("draft_released") is True:
            draft = state.get("quarantined_draft") or state.get("review") or ""
            issues = quality_gate.get("blocking_issues") or []
            issue_lines = "\n".join(f"- {item.get('message')}" for item in issues)
            gate_warnings = quality_gate.get("warnings") or []
            warning_lines = "\n".join(
                f"- {item.get('message')}" for item in gate_warnings if item.get("message")
            )
            warning_block = f"\n> {warning_lines}" if warning_lines else ""
            warning_banner = (
                "\n\n> ⚠️ **质量门禁提示（部分满足）**\n>\n"
                f"> {issue_lines}{warning_block}\n>\n"
                "> 以下内容为未完全达标草稿，仅供参考，请补充检索或降低引用要求后重新生成。\n\n"
            )
            references = state.get("references") or []
            reference_block = ""
            if references:
                reference_block = "\n\n## 参考文献\n\n" + "\n".join(
                    f"[{index}] {reference}"
                    for index, reference in enumerate(references, 1)
                )
            return warning_banner + draft + reference_block
        
        # 原有硬拦截逻辑
        if quality_gate.get("phase") == "pre_generation" and state.get("review"):
            return str(state["review"])
        issues = quality_gate.get("blocking_issues") or []
        options = quality_gate.get("recovery_options") or []
        issue_lines = "\n".join(f"- {item.get('message')}" for item in issues)
        option_lines = "\n".join(f"- {item}" for item in options)
        return (
            "## 正式正文已被质量门禁阻止\n\n"
            "系统已完成草稿验证，但结果未达到可交付标准，因此未展示未经验证的正文。\n\n"
            f"### 阻断原因\n\n{issue_lines}\n\n"
            f"### 建议处理\n\n{option_lines or '- 补充或修正证据后重新生成'}"
        )

    banner = _global_gate_warning_banner(state)
    if banner:
        parts.append(banner)

    if state.get("intent") == "search_papers":
        papers = state.get("ranked_papers") or []
        topic = state.get("topic") or state.get("user_query", "")
        parts.append(f"# {topic} 相关论文\n")
        if not papers:
            parts.append("未检索到相关论文。")
            return "\n".join(parts)
        for i, paper in enumerate(papers, 1):
            title = paper.get("title", "Untitled")
            year = paper.get("year") or "未知年份"
            venue = paper.get("venue") or "未知来源"
            paper_id = paper.get("paper_id") or ""
            url = paper.get("url") or paper.get("pdf_url") or ""
            reason = paper.get("ranking_reason") or paper.get("relevance_reason") or ""
            line = f"{i}. **{title}** ({year}, {venue})"
            if paper_id:
                line += f" `{paper_id}`"
            if url:
                line += f"\n   {url}"
            if reason:
                line += f"\n   入选理由：{reason}"
            parts.append(line)
        return "\n\n".join(parts)

    # 综述正文
    review = (
        state.get("review")
        or state.get("related_work")
        or state.get("introduction")
        or ""
    )
    references = state.get("references") or []
    if review:
        parts.append(review)

    # 参考文献
    if (
        state.get("max_papers_explicit", False)
        and not references
        and review.startswith("## 未生成相关工作")
    ):
        return "\n".join(parts)

    if references:
        parts.append("\n\n## 参考文献\n")
        for i, ref in enumerate(references, 1):
            parts.append(f"[{i}] {ref}")

    return "\n".join(parts)
