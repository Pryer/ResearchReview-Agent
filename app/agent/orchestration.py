"""把旧会话的编排标记规范为唯一受支持的自主执行路径。"""


def normalize_orchestration_mode(state: dict) -> str:
    mode = state.get("agent_orchestration_mode")
    if mode is not None and (not isinstance(mode, str) or mode not in {"legacy", "autonomous"}):
        raise ValueError("invalid persisted orchestration mode")
    # WHY: 历史会话的标记只表明原调度算法；研究目标、证据、约束和预算仍由
    # 原状态持有。旧调度的验证指纹不能用于自主模式的 finish 授权。
    if mode != "autonomous":
        state.pop("autonomous_verified_fingerprint", None)
    state["agent_orchestration_mode"] = "autonomous"
    return "autonomous"
