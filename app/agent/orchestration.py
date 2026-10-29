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


def normalize_retrieval_profile(state: dict) -> dict:
    """旧规则榜单仅作历史记录，恢复执行时重新取得混合检索与语义准入资格。"""
    profile = state.get("retrieval_profile") or {}
    if profile.get("mode") not in {None, "rules", "hybrid_shadow", "hybrid"}:
        raise ValueError("invalid persisted retrieval mode")
    if profile.get("mode") != "hybrid":
        # WHY: 不重置原始证据、用户约束或累计预算；只撤销旧排序和派生授权。
        state["retrieval_profile"] = {"mode": "hybrid", "version": "hybrid.v1"}
        if profile.get("mode") is None:
            # WHY: 未开始检索的新状态也没有 profile；现有证据仍由版本化门禁复核。
            return state["retrieval_profile"]
        if state.get("ranked_papers"):
            from app.agent.nodes.base import _paper_identity_key
            candidates = state.setdefault("candidate_papers", [])
            seen = {_paper_identity_key(p) for p in candidates}
            candidates.extend(p for p in state["ranked_papers"] if _paper_identity_key(p) not in seen)
            state["ranked_papers"] = []
        for key in ("retrieval_screening_cache", "screening_report", "hybrid_shadow_report",
                    "claim_plans", "claim_verification", "quality_gate", "autonomous_verified_fingerprint"):
            state.pop(key, None)
    return state["retrieval_profile"]
