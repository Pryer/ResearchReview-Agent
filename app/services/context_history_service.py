"""从当前上下文可达的归档回查观察；不向模型返回原始提示词或思考协议。"""
from collections import deque
import hashlib
import json

from app.agent.execution_budget import check_execution


def evidence_keys(value):
    keys = set()
    if isinstance(value, dict):
        for name, item in value.items():
            if name in {"evidence_id", "paper_id"} and isinstance(item, str):
                keys.add(item)
            elif name in {"evidence_ids", "evidence_refs", "output_artifact_refs", "artifact_refs"} and isinstance(item, list):
                keys.update(x for x in item if isinstance(x, str))
            if isinstance(item, (dict, list)):
                keys.update(evidence_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(evidence_keys(item))
    return keys


def _load_archive(store, session_id, ref, scope_id, remaining_bytes):
    index = store.resolve(session_id, ref, role="service", allowed_types={"context_archive_index"})["payload"]
    if index.get("scope_id") not in {None, scope_id}:
        raise PermissionError("context archive belongs to another trajectory")
    kind = index["artifact_type"]
    if kind not in {"context_epoch", "tool_observation"}:
        raise PermissionError("not a context history archive")
    chunks = []
    for fragment in index["fragments"]:
        check_execution()
        text = store.resolve(session_id, fragment, role="service", allowed_types={kind})["payload"]["value"]
        remaining_bytes -= len(text.encode("utf-8"))
        if remaining_bytes < 0:
            return None, remaining_bytes
        chunks.append(text)
    payload = json.loads("".join(chunks))
    if kind == "tool_observation":
        payload = {**payload, "sequence": index.get("observation_sequence"),
                   "action": index.get("operation") or payload.get("operation")}
    return payload, remaining_bytes


def recall_history(state, trajectory, evidence_id, *, limit=3):
    from app.services.research_execution_service import artifact_store
    store = artifact_store()
    if store is None or not state.get("session_id"):
        raise ValueError("context history requires a session artifact store")
    terms = {evidence_id}
    for card in state.get("paper_cards") or []:
        if evidence_id in evidence_keys(card):
            pid = card.get("paper_id")
            terms.update({str(pid), f"state://paper-card/{pid}"})
    queue = deque((ref, None) for ref in trajectory.data.get("archive_refs", []))
    records = deque((item, None) for item in reversed(trajectory.data["memory"]))
    seen_refs, seen_records, matches = set(), set(), []
    remaining_bytes = 8 * 1024 * 1024
    scanned = 0
    # WHY: 只遍历当前轨迹的已提交引用链；不能让模型指定任意资料引用来读取别人的历史。
    while records or queue:
        check_execution()
        if not records:
            if scanned >= 128:
                break
            ref, _ = queue.popleft()
            if ref in seen_refs:
                continue
            seen_refs.add(ref)
            payload, remaining_bytes = _load_archive(store, str(state["session_id"]), ref,
                                                      trajectory.data["scope_id"], remaining_bytes)
            if payload is None:
                break
            scanned += 1
            queue.extend((old, None) for old in payload.get("previous_archive_refs", []))
            observations = payload.get("memory")
            if observations is None:
                # 兼容旧 Epoch：观察可能只保存在消息或 checkpoint 摘要中。
                observations = []
                for message in payload.get("messages", []):
                    try:
                        body = json.loads(message.get("content") or "{}")
                    except (ValueError, TypeError):
                        continue
                    if message.get("role") == "tool":
                        observations.append(body)
                    elif "observation" in body:
                        observations.append(body["observation"])
                    observations.extend((body.get("context_checkpoint") or {}).get("observations", []))
                if "messages" not in payload:
                    observations = [payload]
            records = deque((item, ref) for item in reversed(observations))
            if not records:
                continue
        item, ref = records.popleft()
        for raw_ref in item.get("artifact_refs") or []:
            if raw_ref not in seen_refs:
                queue.append((raw_ref, None))
        identity = str(item.get("sequence") or hashlib.sha256(
            json.dumps(item, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest())
        if (item.get("action") == "recall_context" or item.get("history")
                or identity in seen_records or not terms.intersection(evidence_keys(item))):
            continue
        seen_records.add(identity)
        matches.append({
            "sequence": item.get("sequence"), "action": item.get("action") or item.get("operation"),
            "status": item.get("status"), "outcome": item.get("outcome"),
            "finding": "; ".join(str(x) for x in item.get("findings", [])[:2])[:240],
            "error": str(item.get("error") or "")[:160], "archive_ref": ref,
        })
        if len(matches) >= limit:
            break
    result = {"evidence_id": evidence_id, "matches": matches, "search_complete": not records and not queue}
    # 结果大小独立有界；未扫描完不得把“未找到”解释为不存在证据。
    from app.core.config import get_settings
    max_chars = max(512, get_settings().agent_context_tool_result_chars // 2)
    while len(json.dumps(result, ensure_ascii=False)) > max_chars and result["matches"]:
        result["matches"].pop()
        result["search_complete"] = False
    return {"status": "completed", "outcome": "no_change", "history": result,
            "findings": [f"找到 {len(result['matches'])} 条历史观察；完整扫描={result['search_complete']}"],
            "evidence_refs": []}
