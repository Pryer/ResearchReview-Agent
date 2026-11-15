"""核验身份、问题反馈与Controller提交阶段的质量比较。"""
import copy
from app.schemas.check_schema import CheckReport, ResultIssue, SuggestedAction


def check_input_fingerprint(state):
    from app.agent.context_builder import _stable_hash
    from app.agent.claim_plan import CLAIM_SUPPORT_POLICY_VERSION
    from app.services.evidence_support import EVIDENCE_POLICY_VERSION
    from app.tools.verify_claims import _ENTAILMENT_VERIFIER_VERSION
    from app.services.rag_runtime import RAG_PIPELINE_VERSION
    from app.core.config import get_settings
    # WHY: 报告与调度字段不属于检查输入；所有影响事实、授权及交付的依赖均参与绑定。
    keys = ("review", "related_work", "introduction", "paper_cards", "paper_details", "parsed_papers",
            "claim_plans", "writing_plans", "citation_source_text", "citation_rendered_text",
            "research_request", "research_semantic_frame", "user_query", "required_reference_count",
            "start_year", "end_year", "year_range_explicit", "strict_year_range", "max_papers_explicit", "selected_scope", "core_deliverables",
            "requested_sections", "target_length", "language", "citation_style", "writing_version",
            "evidence_snapshot_fingerprint", "best_effort_on_failure", "best_effort_generation")
    settings = get_settings()
    from app.core.config import get_review_threshold_policy
    return _stable_hash({"inputs": {k: state.get(k) for k in keys},
        "policy": EVIDENCE_POLICY_VERSION, "claim_support_policy": CLAIM_SUPPORT_POLICY_VERSION,
        "verifier": _ENTAILMENT_VERIFIER_VERSION,
        "pipeline": RAG_PIPELINE_VERSION, "thresholds": get_review_threshold_policy().snapshot(),
        "verification_enabled": settings.enable_claim_verification,
        "context_limit": settings.claim_verification_max_context_chars,
        "reference_best_effort": settings.enable_reference_coverage_best_effort_release,
        "reference_best_effort_ratio": settings.reference_coverage_best_effort_ratio})


def check_is_current(state):
    report = state.get("check_report") or {}
    return bool(report and report.get("input_fingerprint") == check_input_fingerprint(state))


def build_check_report(state, fingerprint, *, failure=""):
    from app.core.config import get_settings
    from app.agent.claim_plan import claim_source_strength_violations
    verification = state.get("claim_verification") or {}
    gate = state.get("quality_gate") or {}
    incomplete = bool(failure or not gate or not str(state.get("review") or "").strip()
                      or (get_settings().enable_claim_verification and (
                          not verification or int(verification.get("unverified") or 0) > 0
                          or verification.get("verification_failed")
                          or (int(verification.get("factual_claims") or 0) > 0
                              and float(verification.get("verification_coverage", 1.0)) < 1.0))))
    issues = []
    strength_issues = claim_source_strength_violations(state.get("claim_plans") or [], state.get("paper_cards") or [])
    if strength_issues:
        issues.extend(ResultIssue(code="claim_source_independence_overstated", **item) for item in strength_issues)
        # WHY: 恢复旧计划也须遵守现行来源分组；Check只阻断放行，不改Analysis的授权字段。
        gate = {**gate, "passed": False, "draft_released": False, "blocking_issues": [
            *list(gate.get("blocking_issues") or []),
            {"code": "claim_evidence_quality_not_met", "message": "主张语言强度超过当前独立来源支持，需重新分析"}]}
        state["quality_gate"] = gate
    if incomplete:
        issues.append(ResultIssue(code="check_execution_incomplete", reason=failure or "部分必需核验尚未完成"))
    for item in gate.get("blocking_issues") or []:
        issues.append(ResultIssue(code=str(item.get("code") or "quality_gate_failed"),
                                  reason=str(item.get("message") or item.get("code") or "交付要求未满足")))
    for item in verification.get("claims") or []:
        if item.get("support_status") not in {"unsupported", "partially_supported"}:
            continue
        issues.append(ResultIssue(code="unsupported_claim" if item.get("support_status") == "unsupported" else "partially_supported_claim",
            severity="blocking" if item.get("support_status") == "unsupported" else "warning",
            claim_id=str(item.get("claim_id") or ""), section_id=str(item.get("section_id") or ""),
            evidence_ids=[str(i) for i in item.get("evidence_ids") or []],
            reason="；".join(str(i) for i in item.get("issues") or []) or "主张超出当前证据支持范围"))
    # WHY: Check不能一边报告阻断事实问题，一边给出passed；修复仍由Main委派。
    unsupported = [i for i in issues if i.code == "unsupported_claim" and i.severity == "blocking"]
    if unsupported and gate.get("passed") is True:
        gate = {**gate, "passed": False, "draft_released": False, "blocking_issues": [
            *list(gate.get("blocking_issues") or []),
            {"code": "claim_evidence_quality_not_met", "message": "存在未获证据支持的事实主张，需修复后复验"}]}
        state["quality_gate"] = gate
    suggestions = []
    if incomplete:
        suggestions.append(SuggestedAction(action="validate_result", reason="完成未完成的检查；受现有重试与预算限制"))
    if not incomplete and gate.get("passed") is True:
        suggestions.append(SuggestedAction(action="request_finish", reason="申请当前版本交付"))
    else:
        suggestions.append(SuggestedAction(action="diagnose_evidence_gaps", reason="区分可局部修复的问题和确需补证的缺口"))
        if verification.get("claims"):
            suggestions.append(SuggestedAction(action="rewrite_sections", target_issue_ids=[i.issue_id for i in issues if i.claim_id],
                                               reason="已有证据可支持时局部修复，不删除用户明确要求的主题"))
    scope = verification.get("verification_scope") or {}
    report = CheckReport(input_fingerprint=fingerprint, draft_version=int(state.get("writing_version") or 0),
        execution_status="failed" if failure else "incomplete" if incomplete else "completed",
        verdict="unable_to_verify" if incomplete else "passed" if gate.get("passed") is True else "needs_revision",
        coverage={"mode": str(scope.get("mode") or "full"),
                  "factual_claims": int(verification.get("factual_claims") or 0),
                  "unverified": int(verification.get("unverified") or 0),
                  "verification_coverage": float(verification.get("verification_coverage") or 0.0)},
        issues=issues, suggested_actions=suggestions)
    for issue in report.issues:
        # WHY: 支持不足与已知矛盾、身份错误后果不同，不能统一放行或一律隔离。
        issue.blocks_final = issue.severity == "blocking"
        known_hard = any(term in issue.code for term in ("identity", "citation_protocol", "citation_scope_not_met", "contradict", "fabricat"))
        issue.blocks_draft = known_hard
        issue.verification_status = "failed" if known_hard else "incomplete" if incomplete else "insufficient"
        issue.suggested_repair = "修正或移除错误内容" if known_hard else "缩小断言或补证；草稿标明待核验范围"
    for claim in verification.get("claims") or []:
        if any("contradict" in str(i) for i in claim.get("issues") or []):
            report.issues.append(ResultIssue(code="claim_contradicted_by_evidence", reason="主张与原文矛盾，必须修正或移除",
                claim_id=str(claim.get("claim_id") or ""), section_id=str(claim.get("section_id") or ""),
                blocks_draft=True, blocks_final=True, verification_status="contradicted"))
    state["check_report"] = report.model_dump(mode="json")
    if incomplete:
        state["generation_blocked"] = True
        state["quality_gate"] = {**gate, "passed": False, "draft_released": False,
            "blocking_issues": list(gate.get("blocking_issues") or []) + [
                {"code": "claim_verification_incomplete", "message": issues[0].reason}]}
    return report


def settle_checked_candidate(state):
    report = state.get("check_report") or {}
    if not report:
        return  # 兼容旧操作适配器；生产核验入口始终生成报告。
    if not check_is_current(state):
        raise ValueError("check report does not match current candidate and evidence")
    from app.agent.graph import _draft_fingerprint, _GENERATION_PRODUCT_KEYS, _snapshot_generation_products
    state["pending_result_validation"] = False
    state["autonomous_verified_fingerprint"] = _draft_fingerprint(state)
    previous = state.pop("pending_generation_previous", None)
    if previous:
        from app.agent.generation_recovery import candidate_is_not_worse, input_fingerprint
        # WHY: 上一稿缺省的字段也必须恢复为缺省，不能继承新稿引用快照使旧报告失效。
        restored = {k:v for k,v in state.items() if k not in (*_GENERATION_PRODUCT_KEYS, "writing_version")}
        restored.update(copy.deepcopy(previous["products"]))
        if check_is_current(restored) and not candidate_is_not_worse(restored, state):
            for key in _GENERATION_PRODUCT_KEYS:
                state.pop(key, None)
            state.update(copy.deepcopy(_snapshot_generation_products(restored)))
            state["pending_result_validation"] = False
            state["autonomous_verified_fingerprint"] = _draft_fingerprint(state)
            state.setdefault("recovery_candidate_rejections", []).append({
                "reason": "独立Check发现候选质量退化，Controller保留同证据版本的较优稿", "operation": previous["operation"],
                "input_fingerprint": previous.get("input_fingerprint") or input_fingerprint(restored),
                "check_report": copy.deepcopy(report)})
    from app.agent.quality_feedback import complete_main_recovery
    complete_main_recovery(state)
