"""从权威研究状态构建主 Agent 的五字段工作视图。"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from app.core.config import get_settings
from app.schemas.context_schema import (
    AgentStateContext,
    ContextSnapshot,
    DecisionRecord,
    GoalContext,
    KeyEvidence,
    MainAgentContext,
    OpenQuestion,
)


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:20]


def context_source_payload(state: dict[str, Any]) -> dict[str, Any]:
    """只纳入会改变主控判断的权威字段，避免运行日志令快照无意义失效。"""
    return {
        "request": state.get("research_request") or {"original_query": state.get("user_query")},
        "explicit_constraints": {
            "year_range_explicit": bool(state.get("year_range_explicit")),
            "max_papers_explicit": bool(state.get("max_papers_explicit")),
            "start_year": state.get("start_year"),
            "end_year": state.get("end_year"),
            "required_reference_count": state.get("required_reference_count"),
        },
        "plan": state.get("research_plan") or {},
        "semantic_frame": state.get("research_semantic_frame") or {},
        "selected_scope": state.get("selected_scope") or {},
        "deliverables": state.get("core_deliverables") or [],
        "evidence_snapshot": state.get("evidence_snapshot_fingerprint") or "",
        "evidence_version": state.get("evidence_snapshot_version") or 0,
        "evidence_content": _stable_hash(state.get("paper_cards") or []),
        "source_material": _stable_hash({key: state.get(key) for key in (
            "candidate_papers", "paper_details", "parsed_papers", "pdf_paths",
        )}),
        "claim_authorization": _stable_hash(state.get("claim_plans") or []),
        "writing_version": state.get("writing_version") or 0,
        "writing_content": _stable_hash({
            "review": state.get("review") or "",
            "writing_plans": state.get("writing_plans") or [],
            "citation_map": state.get("citation_map") or {},
        }),
        "routes": state.get("route_decisions") or [],
        "recovery": state.get("recovery_decision") or state.get("quality_recovery_decision") or {},
        "quality_gate": state.get("quality_gate") or {},
        "global_evidence_gate": state.get("global_evidence_gate") or {},
        "clarification": state.get("clarification") or {},
        "user_clarifications": state.get("user_clarifications") or [],
    }


def context_source_fingerprint(state: dict[str, Any]) -> str:
    return _stable_hash(context_source_payload(state))


def source_state_version(state: dict[str, Any]) -> str:
    return ":".join([
        str(state.get("state_schema_version") or "1"),
        str(state.get("state_revision") or 0),
        str(state.get("evidence_snapshot_version") or 0),
        str(state.get("writing_version") or 0),
    ])


def _clean_text(value: Any, limit: int = 360) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= limit else text[: max(0, limit - 1)].rstrip() + "…"


def _explicit_constraints(state: dict[str, Any]) -> list[dict[str, Any]]:
    request = state.get("research_request") or {}
    constraints: list[dict[str, Any]] = []
    if state.get("year_range_explicit") or request.get("year_range_explicit"):
        constraints.append({
            "name": "time_range",
            "value": {key: state.get(key) if state.get(key) is not None else request.get(key)
                      for key in ("start_year", "end_year")},
            "source": "user",
            "hard": True,
        })
    if state.get("max_papers_explicit") or request.get("max_papers_explicit"):
        constraints.append({
            "name": "required_reference_count",
            "value": state.get("required_reference_count") or request.get("required_reference_count"),
            "source": "user",
            "hard": True,
        })
    # WHY: 语义帧的真实生产字段是 evidence_requirements；仅 user_explicit 可升为
    # 硬约束。同时保留原始请求，避免未被解析器识别的否定或尾部条件被静默裁掉。
    semantic = state.get("research_semantic_frame") or {}
    for item in semantic.get("evidence_requirements") or []:
        if item.get("source") == "user_explicit" or item.get("explicit"):
            constraints.append({"name": "evidence_requirement", "value": item, "source": "user", "hard": True})
    for answer in state.get("user_clarifications") or []:
        constraints.append({"name": "user_clarification", "value": answer, "source": "user", "hard": True})
    return constraints


def _goal(state: dict[str, Any]) -> GoalContext:
    request = state.get("research_request") or {}
    semantic = state.get("research_semantic_frame") or {}
    return GoalContext(
        request=str(request.get("original_query") or state.get("user_query") or ""),
        topic=_clean_text(state.get("canonical_topic") or state.get("topic"), 240),
        deliverables=[str(item) for item in state.get("core_deliverables") or []],
        scope=dict(state.get("selected_scope") or {}),
        constraints={
            "start_year": state.get("start_year"),
            "end_year": state.get("end_year"),
            "required_reference_count": state.get("required_reference_count"),
            "language": state.get("language"),
            "citation_style": state.get("citation_style"),
            "required_focuses": semantic.get("required_focuses") or [],
        },
        explicit_constraints=_explicit_constraints(state),
    )


def _current_stage(state: dict[str, Any]) -> str:
    steps = state.get("steps") or []
    if steps:
        return str(
            steps[-1].get("step_name")
            or steps[-1].get("step")
            or steps[-1].get("name")
            or "planned"
        )
    return "planned" if state.get("research_plan") else "created"


def _next_actions(state: dict[str, Any]) -> list[str]:
    plan = state.get("research_plan") or {}
    pending = [
        str(item.get("operation") or item.get("id") or "")
        for item in plan.get("task_graph") or []
        if str(item.get("status") or "pending") in {"pending", "running"}
    ]
    decision = state.get("quality_recovery_decision") or state.get("recovery_decision") or {}
    if decision.get("action") and str(decision.get("action")) not in pending:
        pending.insert(0, str(decision["action"]))
    from app.agent.evidence_recovery import targeted_search_kind
    return [item for item in dict.fromkeys(pending) if item and (
        item.lower() != "targeted_search" or targeted_search_kind(state)
    )][:8]


def _agent_state(state: dict[str, Any]) -> AgentStateContext:
    settings = get_settings()
    steps = state.get("steps") or []
    completed = [
        str(item.get("step_name") or item.get("step") or item.get("name") or "")
        for item in steps
        if str(item.get("status") or "") in {"success", "completed"}
    ]
    quality = state.get("quality_gate") or {}
    global_gate = state.get("global_evidence_gate") or {}
    used = int(state.get("recovery_action_count") or 0)
    ledger = state.get("agent_execution_budget") or {}
    execution_status = str(state.get("agent_execution_status") or "idle")
    research_status = str(
        state.get("result_status")
        or ("blocked" if state.get("generation_blocked") else "running")
    )
    recent = [{
        "action": str(
            item.get("step_name") or item.get("step") or item.get("name") or ""
        ),
        "status": str(item.get("status") or "unknown"),
        "error": _clean_text(item.get("error"), 180),
    } for item in steps[-8:]]
    allowed = [
        str(item) for item in state.get("allowed_agent_actions") or _next_actions(state)
    ]
    failures: list[str] = []
    failures.extend(str(item.get("reason") or "") for item in (state.get("main_agent_rejections") or [])[-3:])
    for item in (state.get("errors") or [])[-6:]:
        value = (
            item.get("message") or item.get("error") or item.get("code")
            if isinstance(item, dict)
            else item
        )
        cleaned = _clean_text(value, 240)
        if cleaned:
            failures.append(cleaned)
    return AgentStateContext(
        status=research_status,
        execution_status=execution_status,
        research_status=research_status,
        current_stage=_current_stage(state),
        next_actions=_next_actions(state),
        allowed_actions=list(dict.fromkeys(item for item in allowed if item))[:16],
        completed_actions=list(dict.fromkeys(item for item in completed if item))[-10:],
        recent_trajectory=recent,
        failure_reasons=list(dict.fromkeys(failures)),
        gate_status={
            "quality": quality.get("passed"),
            "quality_phase": quality.get("phase"),
            "evidence": global_gate.get("status"),
            "explicit_constraint_unmet": bool(global_gate.get("explicit_constraint_unmet")),
            "generation_blocked": bool(state.get("generation_blocked")),
            "route_recovery_status": state.get("evidence_recovery_status"),
        },
        artifact_summary={
            "candidate_papers": len(state.get("candidate_papers") or []),
            "paper_details": len(state.get("paper_details") or []),
            "paper_cards": len(state.get("paper_cards") or []),
            "claim_plans": len(state.get("claim_plans") or []),
            "has_review": bool(str(state.get("review") or "").strip()),
            "valid_citations": int(state.get("unique_valid_cited_paper_count") or 0),
        },
        remaining_action_budget=max(0, int(settings.recovery_total_action_budget) - used),
        budget_remaining={
            "actions": max(0, int(ledger.get("action_limit", settings.agent_execution_action_budget)) - int(ledger.get("actions_reserved") or 0)),
            "tokens": max(0, int(ledger.get("token_limit", settings.agent_main_token_budget)) - int(ledger.get("llm_tokens") or 0) - int(ledger.get("tokens_reserved") or 0)),
            "retrieval": max(0, int(ledger.get("retrieval_limit", settings.agent_retrieval_budget)) - int(ledger.get("retrieval_used") or 0)),
            "recovery": max(0, int(settings.recovery_total_action_budget) - max(used, int(ledger.get("recovery_used") or 0))),
            "route_recovery_rounds": max(0, int(settings.evidence_recovery_max_rounds) - int(state.get("recovery_round") or 0)),
        },
        source_state_version=source_state_version(state),
    )


def _key_evidence(state: dict[str, Any], limit: int) -> list[KeyEvidence]:
    refs = {str(item.get("paper_id")): item["ref"]
            for item in (state.get("artifact_manifest") or {}).get("paper_cards", {}).get("items", [])}
    cards = {str(card.get("paper_id") or ""): card for card in state.get("paper_cards") or []}
    items: list[KeyEvidence] = []
    seen: set[str] = set()
    for plan in state.get("claim_plans") or []:
        for claim in plan.get("claims") or []:
            finding = _clean_text(
                claim.get("claim_text") or claim.get("claim") or claim.get("statement"),
                420,
            )
            if not finding:
                continue
            paper_ids = [str(pid) for pid in claim.get("supporting_paper_ids") or claim.get("paper_ids") or [] if pid]
            if not paper_ids:
                paper_ids = list(dict.fromkeys(
                    str(evidence_id).rsplit(":", 1)[0]
                    for evidence_id in claim.get("evidence_ids") or []
                    if ":" in str(evidence_id)
                ))
            key = _stable_hash([finding, paper_ids])
            if key in seen:
                continue
            seen.add(key)
            levels = [
                str((cards.get(pid, {}).get("evidence_state") or {}).get("access_level") or cards.get(pid, {}).get("evidence_source") or "unknown")
                for pid in paper_ids
            ]
            items.append(KeyEvidence(
                evidence_id=str(claim.get("claim_id") or key),
                finding=finding,
                paper_ids=paper_ids,
                source_refs=[refs.get(pid, f"state://paper-card/{pid}") for pid in paper_ids],
                evidence_level=", ".join(dict.fromkeys(levels)) or "unknown",
                uncertainty=_clean_text(claim.get("uncertainty") or claim.get("limitation"), 220) or None,
                supports=[str(plan.get("section_id") or plan.get("route_id") or "")],
            ))
            if len(items) >= limit:
                return items
    for theme in state.get("theme_synthesis") or []:
        for raw in theme.get("reported_findings") or []:
            finding = _clean_text(raw.get("claim") if isinstance(raw, dict) else raw, 420)
            if not finding:
                continue
            paper_ids = [str(pid) for pid in (raw.get("supporting_paper_ids") or [raw.get("paper_id")]) if pid] if isinstance(raw, dict) else []
            key = _stable_hash([finding, paper_ids])
            if key in seen:
                continue
            seen.add(key)
            items.append(KeyEvidence(
                evidence_id=str(raw.get("claim_id") or key) if isinstance(raw, dict) else key,
                finding=finding,
                paper_ids=paper_ids,
                source_refs=[refs.get(pid, f"state://paper-card/{pid}") for pid in paper_ids],
                evidence_level="mixed",
                supports=[str(theme.get("theme_id") or theme.get("theme_name") or "")],
            ))
            if len(items) >= limit:
                return items
    return items


def _decisions(state: dict[str, Any], limit: int) -> list[DecisionRecord]:
    output: list[DecisionRecord] = []
    selected = state.get("selected_scope") or {}
    if selected:
        output.append(DecisionRecord(
            decision_id="selected-scope",
            category="scope",
            decision=_clean_text(selected.get("label") or selected.get("description") or selected, 420),
            reason=_clean_text(selected.get("reason") or "用户确认或范围消歧结果", 260),
            source="user" if selected.get("confirmed_by_user") else "agent",
            invalidated_by=["user_scope_change"],
        ))
    for index, raw in enumerate(state.get("route_decisions") or []):
        output.append(DecisionRecord(
            decision_id=str(raw.get("decision_id") or raw.get("route_id") or f"route-{index}"),
            category="route",
            decision=_clean_text(raw.get("decision") or raw.get("action") or raw, 420),
            reason=_clean_text(raw.get("reason"), 260),
            source="evidence",
            invalidated_by=["evidence_snapshot_change", "scope_change"],
        ))
    for category, raw in (
        ("recovery", state.get("recovery_decision") or {}),
        ("quality_recovery", state.get("quality_recovery_decision") or {}),
    ):
        if raw.get("action"):
            output.append(DecisionRecord(
                decision_id=f"{category}-{_stable_hash(raw)}",
                category=category,
                decision=str(raw.get("action")),
                reason=_clean_text(raw.get("reason"), 260),
                source="policy",
                invalidated_by=["state_progress", "budget_change"],
            ))
    user = [item for item in output if item.source == "user"]
    others = [item for item in output if item.source != "user"]
    remaining = max(0, limit - len(user))
    return user + (others[-remaining:] if remaining else [])


def _open_questions(state: dict[str, Any], limit: int) -> list[OpenQuestion]:
    output: list[OpenQuestion] = []
    clarification = state.get("clarification") or (state.get("research_plan") or {}).get("clarification") or {}
    if clarification.get("needed") and clarification.get("question"):
        output.append(OpenQuestion(
            question_id="clarification",
            category="user_input",
            question=_clean_text(clarification.get("question"), 420),
            blocking=True,
            next_action="request_user_input",
        ))
    gap = state.get("evidence_gap_report") or {}
    for index, item in enumerate(gap.get("gaps") or []):
        output.append(OpenQuestion(
            question_id=str(item.get("route_id") or f"evidence-gap-{index}"),
            category="evidence_gap",
            question=_clean_text(item.get("reason") or item.get("gap_type"), 420),
            blocking=bool(gap.get("needs_recovery")),
            needed_evidence=[str(value) for value in item.get("missing_constraints") or []],
            next_action="targeted_search" if item.get("suggested_queries") else "review_scope",
        ))
    for index, item in enumerate((state.get("quality_gate") or {}).get("blocking_issues") or []):
        output.append(OpenQuestion(
            question_id=str(item.get("code") or f"quality-{index}"),
            category="quality_gate",
            question=_clean_text(item.get("message") or item.get("code"), 420),
            blocking=True,
            next_action="quality_recovery",
        ))
    # WHY: blocking 问题决定是否能安全继续，必须在数量限制前优先保留。
    blocking = [item for item in output if item.blocking]
    advisory = [item for item in output if not item.blocking]
    return blocking + advisory[:max(0, limit - len(blocking))]


def build_main_agent_context(state: dict[str, Any]) -> MainAgentContext:
    settings = get_settings()
    return MainAgentContext(
        goal=_goal(state),
        state=_agent_state(state),
        key_evidence=_key_evidence(state, int(settings.main_context_max_evidence)),
        decisions=_decisions(state, int(settings.main_context_max_decisions)),
        open_questions=_open_questions(state, int(settings.main_context_max_open_questions)),
    )


def build_context_snapshot(state: dict[str, Any]) -> ContextSnapshot:
    from app.agent.context_compaction import compact_main_context, validate_context_snapshot

    settings = get_settings()
    # WHY: main_context_max_chars 是整次主控请求预算；动态五字段必须为稳定
    # system/tool 前缀与结构化动作输出预留空间，不能把三者分别按满额计算。
    max_chars = max(
        2000,
        int(settings.main_context_max_chars)
        - int(settings.main_context_system_reserve_chars)
        - int(settings.main_context_output_reserve_chars),
    )
    context = compact_main_context(
        build_main_agent_context(state),
        max_chars=max_chars,
    )
    estimated_chars = len(json.dumps(context.model_dump(mode="json"), ensure_ascii=False))
    snapshot = ContextSnapshot(
        context=context,
        source_fingerprint=context_source_fingerprint(state),
        source_state_version=source_state_version(state),
        estimated_chars=estimated_chars,
        max_chars=max_chars,
    )
    errors = validate_context_snapshot(snapshot, state)
    return snapshot.model_copy(update={"validation_errors": errors})


def refresh_main_agent_context(state: dict[str, Any]) -> ContextSnapshot:
    snapshot = build_context_snapshot(state)
    state["main_agent_context"] = snapshot.context.model_dump(mode="json")
    state["main_context_snapshot"] = snapshot.model_dump(mode="json")
    return snapshot
