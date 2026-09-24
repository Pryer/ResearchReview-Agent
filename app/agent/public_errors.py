"""将内部停止诊断转换为用户可读原因；原始错误留在诊断记录中。"""

import re


_LABELS = {
    "AgentBoundaryViolation": "研究动作的结果未通过系统字段校验，任务已停止，请更新服务后从检查点恢复",
    "agent_budget_exhausted": "执行预算已耗尽",
    "agent_token_budget_exhausted": "模型 token 预算已耗尽",
    "agent_no_progress": "连续动作未产生实质研究进展，已停止重复执行",
    "agent_state_cycle": "研究状态反复往返，未产生持续进展，已停止重复执行",
    "agent_round_limit": "已达到最大决策轮数",
    "search_results_empty": "未检索到符合当前范围的论文",
    "required_recovery_action_unavailable": "已选定的恢复动作当前无法执行",
    "main_context_invalid": "研究上下文校验未通过",
    "invalid_agent_decision": "模型未返回当前允许的有效动作",
    "AgentDecisionError": "模型未返回当前允许的有效动作",
    "LLMInvocationError": "模型服务调用失败，请检查服务状态后重试",
    "RuntimeConflict": "研究状态已变化或会话正在执行，请等待当前任务结束后重试",
    "AgentCancelledError": "用户已取消研究任务",
    "AgentExecutionCancelled": "用户已取消研究任务",
}
_BUDGET_LABELS = {
    "token reservation exceeds remaining budget": "剩余 token 预算不足以预留下一次完整请求",
    "actual token usage exceeded the reserved budget": "实际 token 用量已超过预算，响应已计费但未继续用于研究",
    "research execution deadline exceeded": "本次研究执行已超过时限",
    "task deadline exceeded before commit": "任务执行已超过时限，结果未提交",
    "retrieval budget exhausted": "检索次数预算已耗尽",
    "recovery budget exhausted": "证据恢复次数预算已耗尽",
    "agent action budget exhausted": "专业动作次数预算已耗尽",
}


def public_stop_reason(errors) -> str:
    if not errors:
        return ""
    # WHY: 最后的停止错误决定本轮终态；较早的数据源失败可能已经恢复。
    error = errors[-1]
    if not isinstance(error, dict):
        return "研究执行出现异常，请查看服务端诊断"
    code = str(error.get("code") or "")
    message = str(error.get("message") or "")
    if code == "AgentBudgetExceeded":
        return _BUDGET_LABELS.get(message, "执行预算不足，无法继续研究")
    if code == "main_agent_blocked":
        # WHY: 仅允许简短自然语言阻断说明；地址、路径、标识符和工具诊断不能进入正文。
        safe = (0 < len(message) <= 240 and re.search(r"[\u4e00-\u9fff]", message)
                and re.fullmatch(r"[\u4e00-\u9fffA-Za-z0-9 ，。；：、！？（）《》‘’“”%+\-]+", message)
                and not re.search(r"[A-Za-z0-9-]{24,}|localhost|traceback|api.?key|token=|prompt", message, re.I))
        return message.rstrip("。；") if safe else "当前证据或必要输入不足，无法继续交付"
    return _LABELS.get(code, "研究执行出现内部错误，请查看服务端诊断")
