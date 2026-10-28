"""证据变更与旧授权失效必须在同一次 Controller 提交内完成。"""
import json


def _detail_signature(paper):
    # WHY: 排序分数不改变证据；缺省来源集合与单一主来源等价。保留全部正文、
    # 元数据与准入字段（含置信度/交付资格），未知字段变化仍按失效处理。
    value = {key: item for key, item in paper.items()
             if key not in {"_rank_score", "_llm_semantic_score", "_sources"}}
    sources = paper.get("_sources") or ([paper["source"]] if paper.get("source") else [])
    value["_sources"] = sorted(set(sources))
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def evidence_snapshot(state, operation):
    if operation not in {"fetch_metadata", "extract_paper_cards"}:
        return None
    from app.agent.nodes.base import _paper_identity_key
    details = {_paper_identity_key(p): _detail_signature(p)
               for p in state.get("paper_details") or []}
    return {
        "details": details,
        "fields": {
            "paper_details": json.dumps(details, ensure_ascii=False, sort_keys=True),
            "paper_cards": json.dumps(state.get("paper_cards") or [], ensure_ascii=False,
                                      sort_keys=True, default=str),
        },
    }


def invalidate_changed_evidence(state, operation, before):
    if operation == "extract_paper_cards":
        current = {str(card.get("paper_id") or "") for card in state.get("paper_cards") or []}
        state["pending_card_rebuild_ids"] = [pid for pid in state.get("pending_card_rebuild_ids") or []
                                              if pid not in current]
    if before is None or before["fields"] == evidence_snapshot(state, operation)["fields"]:
        return
    from app.agent.graph import _GENERATION_PRODUCT_KEYS
    from app.agent.nodes.base import _paper_identity_key
    if operation == "fetch_metadata":
        details = {_paper_identity_key(p): _detail_signature(p)
                   for p in state.get("paper_details") or []}
        state["paper_cards"] = [card for card in state.get("paper_cards") or []
                                if (key := _paper_identity_key(card)) in details
                                and before["details"].get(key) == details[key]]
        if before["fields"]["paper_cards"] != "[]" or state.get("pending_card_rebuild_ids"):
            # WHY: 已提取的证据失效后先重建，再允许新的检索/详情循环；标记与
            # 元数据同次提交、随检查点恢复，不能恢复旧卡或旧主张授权。
            retained = {str(card.get("paper_id") or "") for card in state["paper_cards"]}
            state["pending_card_rebuild_ids"] = [str(p.get("paper_id"))
                for p in state.get("paper_details") or []
                if p.get("paper_id") and str(p["paper_id"]) not in retained]
    # WHY: 子循环可能马上选择下一个动作；不能等返回主控后才失效旧证据授权。
    for key in (*_GENERATION_PRODUCT_KEYS, "validated_routes", "route_decisions",
                "autonomous_verified_fingerprint"):
        state.pop(key, None)
