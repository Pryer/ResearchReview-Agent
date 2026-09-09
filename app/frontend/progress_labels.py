"""将后台内部步骤转换为用户可理解的进度说明。"""

from __future__ import annotations


_RECOVERY_STEP_LABELS = {
    "quality_recovery:recompute_state": "正在刷新证据状态并重新检查写作条件",
    "quality_recovery:refresh_evidence": "正在重新核验现有论文的元数据与证据",
    "quality_recovery:rebuild_structure": "正在依据证据重建研究结构",
    "quality_recovery:rebuild_claims": "正在重建主张授权和引用覆盖",
    "quality_recovery:reallocate_citations": "正在重新分配引用",
    "quality_recovery:rewrite_sections": "正在重写未通过验证的章节",
    "quality_recovery:reverify_claims": "正在重试未完成的语义主张验证",
    "quality_recovery:targeted_search": "正在原范围内定向补充证据",
    "quality_recovery:conservative_rewrite": "正在按已授权证据保守重写",
    "quality_recovery:final_best_effort_generation": "正在生成最终最佳可用草稿",
}

_RECOVERY_ACTION_LABELS = {
    "RECOMPUTE_STATE": "刷新证据状态",
    "REFRESH_EVIDENCE": "重新核验现有证据",
    "REBUILD_STRUCTURE": "重建研究结构",
    "REBUILD_CLAIMS": "重建主张授权",
    "REALLOCATE_CITATIONS": "重新分配引用",
    "REWRITE_SECTIONS": "重写失败章节",
    "REVERIFY_CLAIMS": "重试语义主张验证",
    "TARGETED_SEARCH": "定向补充证据",
    "REQUEST_USER_INPUT": "等待必要输入",
    "DEGRADE": "恢复预算已耗尽",
}


def describe_job_step(step: str) -> str:
    """返回公开进度文案；未知步骤保持原值，便于兼容新增节点。"""
    return _RECOVERY_STEP_LABELS.get(str(step or ""), str(step or "等待执行"))


def recovery_progress_view(
    decision: dict | None,
    history: list[dict] | None,
    coverage: dict | None,
    quality_gate: dict | None,
) -> dict[str, object]:
    """构造自动恢复的公开展示数据，不暴露内部指纹和检查点正文。"""
    decision = decision or {}
    history = history or []
    coverage = coverage or {}
    quality_gate = quality_gate or {}
    latest = history[-1] if history else {}
    action = str(decision.get("action") or latest.get("action") or "")
    outcome = str(latest.get("outcome") or "")
    if outcome == "started" or not quality_gate:
        verification = "尚未重新验证"
    elif quality_gate.get("passed") is True:
        verification = "重新验证通过"
    else:
        verification = "重新验证未通过"
    def metric(name: str) -> int | str:
        value = coverage.get(name)
        return "尚未统计" if value is None else int(value or 0)

    return {
        "action": _RECOVERY_ACTION_LABELS.get(action, action),
        "outcome": outcome,
        "verification": verification,
        "claim_authorized": metric("claim_authorized"),
        "planned": metric("planned"),
        "final_valid": metric("final_valid"),
    }


def can_offer_best_effort_draft(metadata: dict | None) -> bool:
    """判断 blocked 消息是否可展示“生成可用草稿”动作。"""
    metadata = metadata or {}
    gate = metadata.get("quality_gate") or {}
    status = metadata.get("status")
    clarification = metadata.get("clarification") or {}
    if status not in {"blocked", "needs_clarification"}:
        return False
    if status == "needs_clarification" and clarification.get("kind") != "quality_decision":
        return False
    if gate.get("passed") is not False or gate.get("draft_released") is True:
        return False
    issue_codes = {
        str(issue.get("code") or "")
        for issue in gate.get("blocking_issues") or []
        if isinstance(issue, dict)
    }
    # 这些问题需要先修复程序或取得外部访问/材料，按钮不能暗示现有证据足以成文。
    non_generatable = {
        "deliverable_generation_failed",
        "authentication_required",
        "human_action_required",
        "missing_user_material",
        "state_time_window_mismatch",
        "stale_evidence_snapshot",
        "recovery_readiness_conflict",
    }
    return bool(metadata.get("paper_cards")) and not bool(issue_codes & non_generatable)
