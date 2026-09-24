"""写作质量失败的统一诊断、动作选择和进展判定。"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from app.schemas.recovery_schema import (
    GenerationRecoveryDecision,
    GenerationRecoveryHistoryEntry,
    GenerationRecoveryIssue,
    RecoveryAction,
    RecoveryProgressVector,
    RecoveryStatus,
)


_INTERNAL_STATE_CODES = {
    "recovery_readiness_conflict", "stale_evidence_snapshot",
    "state_time_window_mismatch", "stale_global_evidence_gate",
}
_STRUCTURE_CODES = {
    "taxonomy_not_ready", "fallback_theme_present", "route_validation_failed",
}
_CLAIM_CODES = {
    "claim_evidence_quality_not_met", "claim_citation_consistency_not_met",
    "invalid_citations",
}
_VERIFICATION_CODES = {"claim_verification_incomplete"}
_SECTION_CODES = {
    "section_generation_failed", "deliverable_structure_invalid",
    "final_text_integrity_not_met", "deliverable_generation_failed",
    "language_coverage_not_met",
}
_COUNT_CODES = {
    "minimum_references_not_met", "minimum_planned_references_not_met",
    "minimum_cited_references_not_met",
}
# WHY: 用户显式研究重点缺证据与引用篇数不足是两类问题。篇数不足可以靠重分配引用
# 或重建授权在现有证据内缓解，重点缺证据不行——重写和重分配都无法凭空造出某个
# 研究重点的直接证据，唯一真实手段是定向补检索，耗尽后只能降级并标注未覆盖重点。
_FOCUS_CODES = {"required_focus_evidence_not_met"}
_USER_INPUT_CODES = {
    "authentication_required", "human_action_required", "missing_user_material",
}


def active_focus_recovery_targets(state: dict[str, Any]) -> list[str]:
    """返回当前已选定补检索动作需要覆盖的研究重点。"""
    recovery = state.get("active_quality_recovery") or {}
    if recovery.get("action") != RecoveryAction.TARGETED_SEARCH.value:
        return []
    targets: list[str] = []
    has_focus_issue = False
    for issue in recovery.get("issues") or []:
        if not isinstance(issue, dict) or issue.get("code") not in _FOCUS_CODES:
            continue
        has_focus_issue = True
        details = issue.get("details") or {}
        for key in ("missing_requirement_ids", "missing_focuses"):
            targets.extend(str(item).strip() for item in details.get(key) or [] if str(item).strip())
    if has_focus_issue and not targets:
        coverage = state.get("focus_coverage") or {}
        targets.extend(str(item).strip() for item in coverage.get("missing_focuses") or [] if str(item).strip())
    return list(dict.fromkeys(targets))


def _ints(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def recovery_action_count(state: dict[str, Any]) -> int:
    """兼容旧会话计数，并返回路线与写作恢复共用的已消耗动作数。"""
    return max(
        _ints(state.get("recovery_action_count")),
        _ints(state.get("quality_recovery_attempts"))
        + _ints(state.get("recovery_round")),
    )


def _collect_ids(value: Any, *keys: str) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in keys:
                if isinstance(item, (list, tuple, set)):
                    found.extend(str(entry) for entry in item if entry)
                elif item:
                    found.append(str(item))
            found.extend(_collect_ids(item, *keys))
    elif isinstance(value, list):
        for item in value:
            found.extend(_collect_ids(item, *keys))
    return list(dict.fromkeys(found))


def quality_progress_vector(state: dict[str, Any]) -> RecoveryProgressVector:
    gate = state.get("quality_gate") or {}
    issues = [item for item in gate.get("blocking_issues") or [] if isinstance(item, dict)]
    codes = sorted({str(item.get("code") or "") for item in issues if item.get("code")})
    required = _ints(state.get("required_reference_count") or gate.get("requested"))
    readiness = state.get("generation_readiness") or {}
    coverage = (
        state.get("reference_coverage_stats")
        or readiness.get("reference_coverage_stats")
        or {}
    )
    valid = _ints(
        state.get("unique_valid_cited_paper_count")
        or (state.get("claim_citation_consistency") or {}).get("validly_authorized_paper_count")
        # WHY: 会话持久化状态只保证保存阶段覆盖统计；恢复向量必须沿用
        # final_valid，不能把已验证的引用误算成 0 并报告整个目标均缺失。
        or coverage.get("final_valid")
        or state.get("unique_cited_paper_count")
    )
    claim_report = state.get("claim_verification") or {}
    mismatch_report = state.get("claim_citation_consistency") or {}
    integrity = state.get("final_review_integrity") or {}
    missing_sections = _ints(integrity.get("missing_required_sections_count"))
    if not missing_sections:
        missing_sections = len(integrity.get("missing_sections") or [])
    structure_codes = _STRUCTURE_CODES | _SECTION_CODES
    return RecoveryProgressVector(
        missing_required_sections=missing_sections,
        unsupported_claims=_ints(
            claim_report.get("unsupported")
            or claim_report.get("unsupported_claims")
        ),
        unverified_claims=_ints(
            claim_report.get("unverified")
            or (state.get("generation_quality") or {}).get("unverified_claims")
        ),
        citation_mismatches=_ints(
            mismatch_report.get("inconsistent_sentences")
            or mismatch_report.get("mismatch_count")
        ),
        valid_reference_shortfall=max(required - valid, 0),
        structure_issues=sum(code in structure_codes for code in codes),
        metadata_issues=sum("metadata" in code for code in codes),
        hard_issue_codes=codes,
    )


def progress_fingerprint(vector: RecoveryProgressVector) -> str:
    payload = vector.model_dump(mode="json")
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:20]


def input_fingerprint(state: dict[str, Any]) -> str:
    """覆盖影响恢复复用边界的证据、结构、策略和用户约束。"""
    payload = {
        "papers": sorted(
            (
                str(card.get("paper_id") or ""),
                str((card.get("evidence_state") or {}).get("access_level") or card.get("evidence_source") or ""),
                str(card.get("quality_status") or ""),
                hashlib.sha256(json.dumps(
                    {
                        "field_evidence": card.get("field_evidence") or {},
                        "field_claims": card.get("field_claims") or {},
                        "evidence_spans": card.get("evidence_spans") or [],
                    },
                    sort_keys=True,
                    ensure_ascii=False,
                    default=str,
                ).encode("utf-8")).hexdigest()[:16],
            )
            for card in state.get("paper_cards") or []
            if card.get("paper_id")
        ),
        "routes": state.get("validated_routes") or state.get("dynamic_taxonomy") or {},
        "screening": sorted(
            (
                str(paper.get("paper_id") or ""),
                str(paper.get("_screening_decision") or ""),
            )
            for paper in state.get("paper_details") or []
            if paper.get("paper_id")
        ),
        "scope": state.get("selected_scope") or {},
        "semantic_frame": state.get("research_semantic_frame") or {},
        "required_reference_count": _ints(state.get("required_reference_count")),
        "allow_evidence_expansion": state.get("allow_evidence_expansion", True),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()[:20]


def _issue_category(code: str) -> str:
    if code in _INTERNAL_STATE_CODES:
        return "internal_state"
    if code in _STRUCTURE_CODES:
        return "structure"
    if code in _CLAIM_CODES:
        return "claim_or_citation"
    if code in _VERIFICATION_CODES:
        return "claim_verification"
    if code in _SECTION_CODES:
        return "section_generation"
    if code in _FOCUS_CODES:
        return "focus_coverage"
    if code in _COUNT_CODES:
        return "reference_coverage"
    if code in _USER_INPUT_CODES:
        return "user_input"
    if "metadata" in code:
        return "metadata"
    return "quality"


def diagnose_generation_issues(state: dict[str, Any]) -> list[GenerationRecoveryIssue]:
    gate = state.get("quality_gate") or {}
    phase = str(gate.get("phase") or "post_generation")
    issues: list[GenerationRecoveryIssue] = []
    for raw in gate.get("blocking_issues") or []:
        if not isinstance(raw, dict):
            continue
        code = str(raw.get("code") or "unknown_quality_failure")
        category = _issue_category(code)
        actions: list[RecoveryAction]
        if category == "internal_state":
            actions = [RecoveryAction.RECOMPUTE_STATE]
        elif category == "structure":
            actions = [RecoveryAction.REBUILD_STRUCTURE, RecoveryAction.TARGETED_SEARCH]
        elif category == "reference_coverage":
            actions = [
                RecoveryAction.REFRESH_EVIDENCE,
                RecoveryAction.REALLOCATE_CITATIONS,
                RecoveryAction.REBUILD_CLAIMS,
                RecoveryAction.TARGETED_SEARCH,
            ]
        elif category == "focus_coverage":
            # 只广告真实可达的手段：重写章节与重分配引用都无法为缺失的研究重点
            # 造出直接证据，广告它们会让诊断结果承诺阶梯永远不会执行的动作。
            actions = [RecoveryAction.TARGETED_SEARCH]
        elif category == "claim_or_citation":
            actions = [RecoveryAction.REWRITE_SECTIONS, RecoveryAction.REBUILD_CLAIMS]
        elif category == "claim_verification":
            actions = [RecoveryAction.REVERIFY_CLAIMS]
        elif category == "section_generation":
            actions = [RecoveryAction.REWRITE_SECTIONS]
        elif category == "user_input":
            actions = [RecoveryAction.REQUEST_USER_INPUT]
        else:
            actions = [RecoveryAction.REWRITE_SECTIONS, RecoveryAction.TARGETED_SEARCH]
        issues.append(GenerationRecoveryIssue(
            code=code,
            phase=phase,
            category=category,
            message=str(raw.get("message") or ""),
            affected_section_ids=_collect_ids(raw, "section_id", "section_ids"),
            affected_claim_ids=_collect_ids(raw, "claim_id", "claim_ids"),
            available_actions=actions,
            details=dict(raw),
        ))
    return issues


def _missing_allocation_sections(state: dict[str, Any]) -> list[str]:
    from app.core.citation_syntax import extract_citation_ids

    cited = set(extract_citation_ids(str(state.get("review") or "")))
    section_ids: list[str] = []
    for allocation in state.get("citation_allocation_plans") or []:
        for section in allocation.get("sections") or []:
            assigned = {str(value) for value in section.get("paper_ids") or [] if value}
            if assigned - cited and section.get("section_id"):
                section_ids.append(str(section["section_id"]))
    return list(dict.fromkeys(section_ids))


def _used_actions_for_same_input(state: dict[str, Any], fingerprint: str) -> list[str]:
    return [
        str(item.get("action") or "")
        for item in state.get("quality_recovery_history") or []
        if str(item.get("input_fingerprint") or "") == fingerprint
        and str(item.get("outcome") or "") in {"started", "no_progress", "failed"}
    ]


def decide_generation_recovery(
    state: dict[str, Any],
    *,
    max_actions: int,
) -> GenerationRecoveryDecision:
    issues = diagnose_generation_issues(state)
    progress = quality_progress_vector(state)
    fingerprint = input_fingerprint(state)
    action_count = recovery_action_count(state)
    remaining = max(0, int(max_actions) - action_count)
    if not issues:
        return GenerationRecoveryDecision(
            action=RecoveryAction.CONTINUE,
            status=RecoveryStatus.NOT_REQUIRED,
            reason="当前没有阻断正式交付的质量问题",
            progress=progress,
            action_fingerprint=fingerprint,
            remaining_budget=remaining,
        )
    if remaining <= 0:
        return GenerationRecoveryDecision(
            action=RecoveryAction.DEGRADE,
            status=RecoveryStatus.EXHAUSTED,
            reason="任务级恢复预算已耗尽",
            issues=issues,
            progress=progress,
            action_fingerprint=fingerprint,
            remaining_budget=0,
        )

    codes = {item.code for item in issues}
    categories = {item.category for item in issues}
    target_sections = list(dict.fromkeys(
        section_id for issue in issues for section_id in issue.affected_section_ids
    ))
    target_claims = list(dict.fromkeys(
        claim_id for issue in issues for claim_id in issue.affected_claim_ids
    ))
    used = _used_actions_for_same_input(state, fingerprint)
    allow_search = state.get("allow_evidence_expansion", True) is not False
    # WHY: 任务级恢复预算与证据级恢复预算是两套独立边界。证据恢复已耗尽时
    # TARGETED_SEARCH 必然是空动作，仍按 RECOVERABLE 下发会让主循环重复同一诊断
    # 直到 no-progress 阻断，且永远进不了 EXHAUSTED 分支的最佳努力兜底。
    from app.agent.evidence_recovery import route_recovery_stop_reason

    search_stop_reason = route_recovery_stop_reason(state) if allow_search else ""
    search_viable = allow_search and not search_stop_reason
    readiness = state.get("generation_readiness") or {}
    stats = (
        state.get("reference_coverage_stats")
        or readiness.get("reference_coverage_stats")
        or {}
    )
    requested = _ints(state.get("required_reference_count"))
    eligible = _ints(
        stats.get("evidence_backed")
        or stats.get("eligible")
        or readiness.get("usable_reference_count")
    )
    authorized = _ints(
        stats.get("claim_authorized")
        or readiness.get("authorized_reference_count")
    )
    confirmed = _ints(stats.get("confirmed_in_scope"))

    if categories & {"user_input"}:
        action = RecoveryAction.REQUEST_USER_INPUT
        reason = "继续执行需要登录、机构访问或用户材料"
    elif codes & _INTERNAL_STATE_CODES:
        action = RecoveryAction.RECOMPUTE_STATE
        reason = "重算与当前证据版本不一致的派生状态"
    elif codes & _VERIFICATION_CODES and RecoveryAction.REVERIFY_CLAIMS.value not in used:
        action = RecoveryAction.REVERIFY_CLAIMS
        reason = "保留当前正文与已完成判定，仅重试未完成的语义主张验证"
    elif codes & _VERIFICATION_CODES:
        action = RecoveryAction.DEGRADE
        reason = "相同正文与证据上的语义主张重验未取得进展"
    elif codes & _STRUCTURE_CODES and RecoveryAction.REBUILD_STRUCTURE.value not in used:
        action = RecoveryAction.REBUILD_STRUCTURE
        reason = "基于现有证据重建无效或碎片化的研究结构"
    elif codes & _COUNT_CODES:
        target_sections = target_sections or _missing_allocation_sections(state)
        if (
            confirmed >= requested > eligible
            and RecoveryAction.REFRESH_EVIDENCE.value not in used
        ):
            action = RecoveryAction.REFRESH_EVIDENCE
            reason = "范围内论文数量足够但证据内容或元数据不足，重新获取并提取现有论文"
        elif authorized >= requested > 0 and RecoveryAction.REALLOCATE_CITATIONS.value not in used:
            action = RecoveryAction.REALLOCATE_CITATIONS
            reason = "已有足够授权论文，重新分配缺失引用并重写受影响章节"
        elif (
            codes & (_CLAIM_CODES | _SECTION_CODES)
            and RecoveryAction.REWRITE_SECTIONS.value not in used
        ):
            # 篇数缺口与正文质量问题并存时，引用重分配后必须真正重写章节；
            # 否则控制器会在分配、主张重建和检索间循环，却不改变问题正文。
            action = RecoveryAction.REWRITE_SECTIONS
            reason = "引用重分配后仍有正文质量问题，重写受影响章节并重新验证"
        elif eligible >= requested > 0 and RecoveryAction.REBUILD_CLAIMS.value not in used:
            action = RecoveryAction.REBUILD_CLAIMS
            reason = "证据池足够但主张授权不足，重建覆盖计划"
        elif search_viable and RecoveryAction.TARGETED_SEARCH.value not in used:
            action = RecoveryAction.TARGETED_SEARCH
            reason = "当前证据无法覆盖引用硬约束，按原范围定向补证"
        elif search_stop_reason:
            # 证据补充已停止且没有其他可自动推进的动作时转为 DEGRADE：状态随之
            # 变为 EXHAUSTED，由调用方执行一次明确标注限制的最佳努力生成，
            # 而不是把已达标的证据整体丢弃。
            action = RecoveryAction.DEGRADE
            reason = f"证据补充已停止（{search_stop_reason}），基于现有证据降级生成"
        else:
            action = RecoveryAction.REQUEST_USER_INPUT
            reason = "当前允许的证据范围无法满足引用硬约束"
    elif codes & _FOCUS_CODES:
        # WHY: 排在篇数缺口之后，以保留本文件既有的升级顺序——先穷尽证据内手段
        # （重分配引用、重建授权），最后才发起外部检索。两类缺口并存时，篇数分支
        # 的廉价补救可能先解决篇数问题，下一轮再由本分支处理重点缺口。
        # 该动作经 _continue_retrieval_and_persist → continue_research_agent；
        # 后者在唯一自主模式中先检索缺失重点。主 Agent 的
        # targeted_search 动作走 recovery_loop，只用路线缺口查询，不等价。
        if search_viable and RecoveryAction.TARGETED_SEARCH.value not in used:
            action = RecoveryAction.TARGETED_SEARCH
            reason = "针对缺失的用户研究重点执行专项定向补检索"
        elif search_stop_reason:
            # 与门禁、前端既有语义对齐：can_offer_best_effort_draft 接受该码，
            # 最佳努力草稿会把它写进 forced_generation_issues 并保留 gate 失败，
            # 因此耗尽后应降级并标注未覆盖重点，而不是把已有证据整体丢弃。
            action = RecoveryAction.DEGRADE
            reason = (
                f"证据补充已停止（{search_stop_reason}），"
                "基于现有证据降级生成并标注未覆盖的研究重点"
            )
        else:
            action = RecoveryAction.REQUEST_USER_INPUT
            reason = "当前允许的证据范围无法覆盖用户明确的研究重点"
    elif categories & {"metadata"}:
        if RecoveryAction.REFRESH_EVIDENCE.value not in used:
            action = RecoveryAction.REFRESH_EVIDENCE
            reason = "重新核验并提取现有论文的元数据与可访问证据"
        elif search_viable and RecoveryAction.TARGETED_SEARCH.value not in used:
            action = RecoveryAction.TARGETED_SEARCH
            reason = "现有论文重新核验后仍不足，按原范围定向补证"
        else:
            action = RecoveryAction.REQUEST_USER_INPUT
            reason = "现有论文的元数据或证据仍不足，继续需要额外访问条件"
    elif codes & (_CLAIM_CODES | _SECTION_CODES):
        if RecoveryAction.REWRITE_SECTIONS.value not in used:
            action = RecoveryAction.REWRITE_SECTIONS
            reason = "仅重写存在主张、引用或章节问题的部分"
        elif RecoveryAction.REBUILD_CLAIMS.value not in used:
            action = RecoveryAction.REBUILD_CLAIMS
            reason = "局部重写无进展，重建主张授权后再次验证"
        elif search_viable:
            action = RecoveryAction.TARGETED_SEARCH
            reason = "现有主张修复无进展，按缺口补充证据"
        else:
            action = RecoveryAction.REQUEST_USER_INPUT
            reason = "现有证据内的写作修复已无可验证进展"
    elif search_viable and RecoveryAction.TARGETED_SEARCH.value not in used:
        action = RecoveryAction.TARGETED_SEARCH
        reason = "未知质量问题先按原始范围补足可验证证据"
    else:
        action = RecoveryAction.REQUEST_USER_INPUT
        reason = "当前约束下没有剩余的自动恢复动作"

    requires_input = action == RecoveryAction.REQUEST_USER_INPUT
    action_key = hashlib.sha256(
        json.dumps({
            "input": fingerprint,
            "action": action.value,
            "issues": sorted(codes),
            "sections": sorted(target_sections),
        }, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:20]
    return GenerationRecoveryDecision(
        action=action,
        status=(
            RecoveryStatus.EXHAUSTED
            if action == RecoveryAction.DEGRADE
            else RecoveryStatus.RECOVERABLE if not requires_input else RecoveryStatus.DEGRADED
        ),
        reason=reason,
        issues=issues,
        target_section_ids=target_sections,
        target_claim_ids=target_claims,
        progress=progress,
        action_fingerprint=action_key,
        requires_user_input=requires_input,
        user_input_reason=reason if requires_input else "",
        remaining_budget=remaining,
    )


def start_recovery_action(
    state: dict[str, Any],
    decision: GenerationRecoveryDecision,
) -> dict[str, Any]:
    from app.agent.execution_budget import reserve_recovery
    reserve_recovery(state)
    entry = GenerationRecoveryHistoryEntry(
        action=decision.action,
        action_fingerprint=decision.action_fingerprint,
        issue_codes=[item.code for item in decision.issues],
        target_section_ids=decision.target_section_ids,
        progress_before=decision.progress,
        input_fingerprint=input_fingerprint(state),
    ).model_dump(mode="json")
    state.setdefault("quality_recovery_history", []).append(entry)
    state["recovery_action_count"] = recovery_action_count(state) + 1
    state["active_quality_recovery"] = decision.model_dump(mode="json")
    return entry


def complete_recovery_action(state: dict[str, Any]) -> None:
    history = state.get("quality_recovery_history") or []
    if not history:
        return
    entry = history[-1]
    before = RecoveryProgressVector.model_validate(entry.get("progress_before") or {})
    after = quality_progress_vector(state)
    before_tuple = (
        before.missing_required_sections, before.unsupported_claims,
        before.unverified_claims,
        before.citation_mismatches, before.valid_reference_shortfall,
        before.structure_issues, before.metadata_issues,
    )
    after_tuple = (
        after.missing_required_sections, after.unsupported_claims,
        after.unverified_claims,
        after.citation_mismatches, after.valid_reference_shortfall,
        after.structure_issues, after.metadata_issues,
    )
    entry["progress_after"] = after.model_dump(mode="json")
    # WHY: 各维质量指标都是独立硬边界。不能让首个指标改善后，以元组字典序
    # 掩盖另一个指标恶化；只有所有维度不增且至少一维下降才算取得进展。
    old_codes = set(before.hard_issue_codes)
    new_codes = set(after.hard_issue_codes)
    # WHY: 重点缺口等硬问题没有独立数值维度；代码消失本身就是可验证进展。
    # 新增其他硬问题不能被引用数等改善抵消。
    non_worse = all(new <= old for old, new in zip(before_tuple, after_tuple)) and new_codes <= old_codes
    improved = non_worse and (
        any(new < old for old, new in zip(before_tuple, after_tuple))
        or bool(old_codes - new_codes)
    )
    entry["outcome"] = "improved" if improved else "no_progress"
    if entry["outcome"] == "no_progress":
        entry["stop_reason"] = "恢复后完整质量向量未改善"
    state.pop("active_quality_recovery", None)


def candidate_is_not_worse(before: dict[str, Any], after: dict[str, Any]) -> bool:
    """候选不能用引用增长抵消新增硬错误。"""
    old = quality_progress_vector(before)
    new = quality_progress_vector(after)
    old_hard = set(old.hard_issue_codes)
    new_hard = set(new.hard_issue_codes)
    if new_hard - old_hard:
        return False
    old_values = (
        old.missing_required_sections,
        old.unsupported_claims,
        old.unverified_claims,
        old.citation_mismatches,
        old.valid_reference_shortfall,
        old.structure_issues,
        old.metadata_issues,
    )
    new_values = (
        new.missing_required_sections,
        new.unsupported_claims,
        new.unverified_claims,
        new.citation_mismatches,
        new.valid_reference_shortfall,
        new.structure_issues,
        new.metadata_issues,
    )
    return all(current <= previous for previous, current in zip(old_values, new_values))
