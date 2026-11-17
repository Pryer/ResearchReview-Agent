"""供Main决策的有界质量反馈；恢复计数仍由generation_recovery拥有。"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any


def candidate_usability(state: dict[str, Any]) -> dict[str, Any]:
    review = str(state.get("review") or "").strip()
    problems = [str(error)[:180]
                for report in state.get("deliverable_validation") or []
                if isinstance(report, dict) and report.get("valid") is False
                for error in report.get("errors") or []]
    from app.services.writer_support import substantive
    problems.extend(str(i.get("reason") or i.get("code")) for i in state.get("writer_attempt_issues") or [])
    return {"present": bool(review), "chars": len(review),
            "substantive": substantive(review), "usable": bool(review) and not problems,
            "errors": problems[:4],
            "pending_validation": bool(state.get("pending_result_validation"))}


def quality_feedback(state: dict[str, Any], *, limit: int = 3) -> dict[str, Any]:
    gate = state.get("quality_gate") or {}
    report = state.get("claim_verification") or {}
    if not any(state.get(k) for k in ("review", "quality_gate", "claim_verification", "deliverable_validation")):
        return {}
    weak = []
    for claim in report.get("claims") or []:
        if not isinstance(claim, dict) or claim.get("support_status") not in {"unsupported", "partially_supported"}:
            continue
        weak.append({"claim_id": str(claim.get("claim_id") or ""),
                     "sentence": str(claim.get("sentence") or "")[:180],
                     "support_status": claim.get("support_status"),
                     "evidence_ids": list(claim.get("evidence_ids") or [])[:4],
                     "issues": [str(i)[:140] for i in claim.get("issues") or []][:2],
                     "suggested_revision": str(claim.get("suggested_revision") or "")[:160]})
    pending = bool(state.get("pending_result_validation"))
    from app.agent.checking import check_is_current
    check = state.get("check_report") or {}
    current_check = bool(check) and check_is_current(state)
    recommendations = ["validate_result"] if pending else []
    decision = state.get("quality_recovery_decision") or {}
    mapping = {"REWRITE_SECTIONS": "rewrite_sections", "REBUILD_CLAIMS": "plan_claims",
               "TARGETED_SEARCH": "targeted_search", "REVERIFY_CLAIMS": "validate_result",
               "REFRESH_EVIDENCE": "fetch_metadata"}
    if not pending and gate.get("passed") is True:
        # WHY: 已通过的当前候选应申请最终门禁；旧恢复建议不能引导再次改写。
        if candidate_usability(state)["usable"]:
            recommendations.append("request_finish")
    elif not pending and gate.get("passed") is False:
        operation = mapping.get(decision.get("action"))
        if operation == "rewrite_sections":
            from app.agent.generation_recovery import rewrite_attempt_exhausted
            if rewrite_attempt_exhausted(state):
                operation = None
        if operation:
            recommendations.append(operation)
        recommendations.append("diagnose_evidence_gaps")
        if (state.get("recovery_decision") or {}).get("queries"):
            recommendations.append("targeted_search")
    sections = [{"section_id": str(section.get("id") or section.get("section_id") or ""),
                 "title": str(section.get("title") or "")[:80]}
                for plan in state.get("writing_plans") or []
                for section in plan.get("sections") or [] if section.get("id") or section.get("section_id")]
    return {"candidate": candidate_usability(state), "writer_issues": (state.get("writer_attempt_issues") or [])[:6],
            "authorization_coverage": state.get("writer_authorization_coverage") or {}, "section_targets": sections[:8],
            "check": {"current": current_check, "report_id": check.get("report_id"),
                      "draft_version": check.get("draft_version"),
                      "execution_status": check.get("execution_status"), "verdict": check.get("verdict"),
                      "coverage": check.get("coverage") or {}},
            # WHY: 写作会使旧核验失效；保留失败证据供诊断，但明确它不是当前版本的判定。
            "verification_state": "previous_candidate" if pending else "stale" if check and not current_check else "reported",
            "verification": {k: report.get(k) for k in
                ("valid", "factual_claims", "supported", "unsupported", "unverified", "support_rate") if k in report},
            "blocking_issues": [{"code": str(i.get("code") or ""), "message": str(i.get("message") or "")[:200]}
                                for i in gate.get("blocking_issues") or [] if isinstance(i, dict)][:3],
            "weak_claim_count": len(weak), "weak_claims": weak[:limit],
            "suggested_actions": list(dict.fromkeys(recommendations)),
            "recovery_reason": str(decision.get("reason") or "")[:200]}


def checked_recovery(state: dict[str, Any]):
    """从当前版本的 Check 报告重新计算恢复路径。"""
    if (state.get("active_quality_recovery") or state.get("pending_result_validation")
            or (state.get("check_report") or {}).get("verdict") != "needs_revision"):
        return None
    from app.agent.checking import check_is_current
    from app.agent.generation_recovery import decide_generation_recovery
    from app.core.config import get_settings

    if not check_is_current(state):
        return None
    return decide_generation_recovery(state, max_actions=get_settings().recovery_total_action_budget)


def checked_section_revision(state: dict[str, Any], recovery):
    """由当前 Check 和恢复策略签出有界的 Writing 请求。"""
    from app.schemas.delegation_schema import DelegationRequest

    plans = state.get("writing_plans") or []
    known = list(dict.fromkeys(section["id"] for plan in plans
                              for section in plan.get("sections") or [] if section.get("id")))
    permitted = set(state.get("target_section_ids") or known)
    if recovery.target_section_ids:
        targets = recovery.target_section_ids
    elif len(plans) == 1:
        # WHY: 旧报告没有句子到章节的绑定时，只用单份交付物中已有权限的章节作有界修订。
        targets = [section for section in known if section in permitted]
    else:
        return None
    if not 1 <= len(targets) <= 3 or not set(targets) <= (permitted & set(known)):
        return None
    return DelegationRequest(agent="writing", profile="section_revision", goal=recovery.reason,
                             scope={"section_ids": targets})


def diagnose_quality_recovery(state: dict[str, Any]) -> None:
    """分析角色提交缺口和备选查询，检索与写作仍由Main另行选择。"""
    from app.agent.generation_recovery import decide_generation_recovery, input_fingerprint
    from app.agent.evidence_recovery import query_novelty, route_recovery_stop_reason
    from app.core.config import get_settings
    if state.get("pending_result_validation") or (state.get("quality_gate") or {}).get("passed") is not False:
        return
    settings = get_settings()
    decision = decide_generation_recovery(state, max_actions=settings.recovery_total_action_budget)
    state["quality_recovery_decision"] = decision.model_dump(mode="json")
    feedback = quality_feedback(state, limit=12)
    gaps = feedback.get("weak_claims") or []
    report = dict(state.get("evidence_gap_report") or {})
    report.update(quality_claim_gaps=gaps, quality_issues=feedback.get("blocking_issues") or [],
                  quality_input_fingerprint=input_fingerprint(state))
    state["evidence_gap_report"] = report
    # 没有路线的背景综述也需要可执行查询；只从真实失败主张生成，不能凭空编造缺口。
    historical = [str(q) for q in state.get("searched_keywords") or []]
    queries = []
    topic = str(state.get("canonical_topic") or state.get("topic") or "")[:100]
    for gap in gaps:
        sentence = re.sub(r"\[[^\]]*\]", "", gap["sentence"]).strip()
        query = (topic + " " + sentence).strip()[:240]
        if sentence and query_novelty(query, historical) >= settings.evidence_recovery_min_query_novelty:
            queries.append(query)
    queries = list(dict.fromkeys(queries))[:min(4, settings.max_search_keywords)]
    if queries and not route_recovery_stop_reason(state):
        report["needs_recovery"] = True
        state["recovery_decision"] = {
            "action": "TARGETED_SEARCH", "status": "RECOVERABLE", "kind": "quality",
            "reason": "按核验失败的具体主张补充证据；Main仍可选择已有证据内的改写",
            "queries": queries, "affected_route_ids": [], "missing_constraints": []}
    elif (state.get("recovery_decision") or {}).get("kind") == "quality":
        state.pop("recovery_decision", None)


def begin_main_recovery(state: dict[str, Any], operation: str, arguments: dict[str, Any]) -> None:
    """Main选择研究动作后，复用统一恢复账本预留一次恢复；不在子任务里嵌套整条链。"""
    if (state.get("active_quality_recovery") or state.get("pending_result_validation")
            or (state.get("quality_gate") or {}).get("passed") is not False):
        return
    from app.agent.generation_recovery import decide_generation_recovery, start_recovery_action
    from app.agent.execution_budget import AgentBudgetExceeded
    from app.core.config import get_settings
    from app.schemas.recovery_schema import RecoveryAction
    mapping = {"rewrite_sections": RecoveryAction.REWRITE_SECTIONS,
               "generate_deliverables": RecoveryAction.REWRITE_SECTIONS,
               "plan_claims": RecoveryAction.REBUILD_CLAIMS,
               "targeted_search": RecoveryAction.TARGETED_SEARCH}
    selected = mapping.get(operation)
    if selected is None:
        return
    if operation == "rewrite_sections":
        from app.agent.generation_recovery import resolve_rewrite_sections
        arguments = {**arguments, "section_ids": resolve_rewrite_sections(state, arguments.get("section_ids") or [])}
    decision = decide_generation_recovery(state, max_actions=get_settings().recovery_total_action_budget)
    if decision.remaining_budget <= 0:
        raise AgentBudgetExceeded("quality recovery action budget exhausted")
    if not decision.issues:
        return
    # WHY: 统一控制器给建议与预算，Main保留合法动作选择权；选中的动作身份必须单独记账。
    decision = decision.model_copy(update={"action": selected,
        "target_section_ids": list(arguments.get("section_ids") or decision.target_section_ids),
        "action_fingerprint": hashlib.sha256(json.dumps({"input": decision.action_fingerprint,
            "operation": operation, "arguments": arguments}, sort_keys=True).encode()).hexdigest()[:20]})
    start_recovery_action(state, decision)
    state["active_quality_recovery"]["origin"] = "main"


def complete_main_recovery(state: dict[str, Any]) -> None:
    if (state.get("active_quality_recovery") or {}).get("origin") == "main":
        from app.agent.generation_recovery import complete_recovery_action
        complete_recovery_action(state)
