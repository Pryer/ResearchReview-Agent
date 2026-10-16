"""证据变更与旧授权失效必须在同一次 Controller 提交内完成。"""
import json


def evidence_snapshot(state, operation):
    if operation not in {"fetch_metadata", "extract_paper_cards"}:
        return None
    from app.agent.nodes.base import _paper_identity_key
    return {
        "details": {_paper_identity_key(p): json.dumps(p, ensure_ascii=False, sort_keys=True, default=str)
                    for p in state.get("paper_details") or []},
        "fields": {key: json.dumps(state.get(key) or [], ensure_ascii=False, sort_keys=True, default=str)
                   for key in ("paper_details", "paper_cards")},
    }


def invalidate_changed_evidence(state, operation, before):
    if before is None or all(before["fields"][key] == json.dumps(
            state.get(key) or [], ensure_ascii=False, sort_keys=True, default=str)
            for key in before["fields"]):
        return
    from app.agent.graph import _GENERATION_PRODUCT_KEYS
    from app.agent.nodes.base import _paper_identity_key
    if operation == "fetch_metadata":
        details = {_paper_identity_key(p): json.dumps(p, ensure_ascii=False, sort_keys=True, default=str)
                   for p in state.get("paper_details") or []}
        state["paper_cards"] = [card for card in state.get("paper_cards") or []
                                if (key := _paper_identity_key(card)) in details
                                and before["details"].get(key) == details[key]]
    # WHY: 子循环可能马上选择下一个动作；不能等返回主控后才失效旧证据授权。
    for key in (*_GENERATION_PRODUCT_KEYS, "validated_routes", "route_decisions",
                "autonomous_verified_fingerprint"):
        state.pop(key, None)
