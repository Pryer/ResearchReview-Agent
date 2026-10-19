"""会话内研究记忆检索；历史描述不能覆盖当前约束或授予证据权限。"""
import hashlib
import json
import re
import unicodedata

from sqlalchemy import select

from app.agent.execution_budget import check_execution
from app.database.models import ResearchEventModel
from app.services.research_memory_service import ResearchMemoryService


def tokens(text):
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    result = set(re.findall(r"[a-z0-9_]+", normalized))
    for span in re.findall(r"[\u3400-\u9fff]+", normalized):
        result.update(span[i:i+2] for i in range(max(1, len(span)-1)))
    return result


def _entry(kind, value, source_ref, *, event_id=None):
    if not isinstance(value, dict):
        value = {"finding": str(value)}
    text = str(next((value[k] for k in ("finding", "decision", "question", "error", "claim_text") if value.get(k)), ""))
    if not text:
        return None
    identity = json.dumps([kind, event_id, value], ensure_ascii=False, sort_keys=True)
    return {"memory_id": "mem-" + hashlib.sha256(identity.encode()).hexdigest()[:24], "kind": kind,
            "text": text[:480], "source_ref": source_ref, "event_id": event_id,
            "historical": True, "requires_revalidation": True, "authorizes_evidence": False}


class ResearchMemoryRetriever:
    def __init__(self, store):
        self.store = store
        self.db = store.repo.db
        self.memory = ResearchMemoryService(self.db)

    def available(self, session_id):
        return self.db.scalar(select(ResearchEventModel.event_id).where(
            ResearchEventModel.session_id == session_id).limit(1)) is not None

    def current_summary(self, session_id, requested_ref=None):
        check_execution()
        if requested_ref:
            self.store.resolve(session_id, requested_ref, role="main", allowed_types={"conversation_summary"})
        if not self.available(session_id):
            return None, {}, []
        ref = self.memory.rebuild_summary(session_id)
        summary = self.store.resolve(session_id, ref, role="main", allowed_types={"conversation_summary"})["payload"]
        entries = []
        for field, kind in (("findings", "finding"), ("decisions", "decision"), ("open_questions", "question")):
            for value in reversed(summary.get(field) or []):
                entry = _entry(kind, value, ref)
                if entry:
                    entries.append(entry)
        return ref, summary, entries

    def retrieve(self, session_id, query, *, kind="all", limit=3, requested_ref=None):
        ref, summary, entries = self.current_summary(session_id, requested_ref)
        for entry in entries:
            entry["validity"] = "current_snapshot"
        # 旧摘要只召回研究发现；已经失效的决定和已解决问题不重新进入工作状态。
        previous = ref
        seen_ids = {item["memory_id"] for item in entries}
        visited = set()
        for _ in range(16):
            if not previous or previous in visited:
                break
            visited.add(previous)
            check_execution()
            row = self.store.resolve(session_id, previous, role="main", allowed_types={"conversation_summary"})
            if previous != ref:
                for value in reversed(row["payload"].get("findings") or []):
                    item = _entry("finding", value, previous)
                    if item and item["memory_id"] not in seen_ids:
                        item["validity"] = "historical_summary"
                        entries.append(item)
                        seen_ids.add(item["memory_id"])
            previous = row["provenance"].get("previous_summary")
        failures = self.db.scalars(select(ResearchEventModel).where(
            ResearchEventModel.session_id == session_id,
            ResearchEventModel.event_type == "agent_failure",
        ).order_by(ResearchEventModel.event_id.desc()).limit(201)).all()
        complete = len(failures) <= 200 and not previous
        for row in failures[:200]:
            check_execution()
            body = json.loads(row.payload_json)["payload"]
            entry = _entry("failure", body, (body.get("source_refs") or [ref])[0], event_id=row.event_id)
            if entry:
                entry["operation"] = str(body.get("operation") or "")
                entry["task"] = str(body.get("task") or "")[:240]
                entries.append(entry)
        query_tokens = tokens(query)
        ranked = []
        for index, entry in enumerate(entries):
            if kind != "all" and entry["kind"] != kind:
                continue
            overlap = len(query_tokens & tokens(entry["text"] + " " + entry.get("task", "") + " " + entry.get("operation", "")))
            if overlap:
                ranked.append((overlap / max(1, len(query_tokens)), entry["kind"] == "failure", -index, entry))
        ranked.sort(key=lambda item: item[:3], reverse=True)
        matches = [item[-1] for item in ranked[:limit]]
        from app.core.config import get_settings
        cap = min(get_settings().research_memory_context_chars, get_settings().agent_context_tool_result_chars // 2)
        while matches and len(json.dumps(matches, ensure_ascii=False)) > cap:
            matches.pop()
        return {"query": query, "kind": kind, "summary_ref": ref, "cursor": summary.get("cursor", 0),
                "matches": matches, "search_complete": complete,
                "result_limited": len(ranked) > len(matches), "authorizes_evidence": False}


def memory_service(state):
    from app.services.research_execution_service import artifact_store
    store = artifact_store()
    return ResearchMemoryRetriever(store) if store is not None and state.get("session_id") else None


def task_fingerprint(state):
    from app.agent.context_builder import _goal, _stable_hash
    return _stable_hash(_goal(state).model_dump(mode="json"))


def prepare_memory_context(state):
    service = memory_service(state)
    context = state.get("research_memory_context") or {}
    if context.get("task_fingerprint") != task_fingerprint(state):
        state.pop("research_memory_context", None)
        context = {}
    if service is None:
        state.pop("research_memory_context", None)
        state["research_memory_status"] = {"available": False, "reason": "session_store_unavailable"}
        return
    session_id = str(state["session_id"])
    try:
        check_execution()
        available = service.available(session_id)
        state["research_memory_status"] = {"available": available, "tool": "search_research_memory"}
        if context:
            ref, summary, entries = service.current_summary(session_id, state.get("research_memory_artifact_ref"))
            current_ids = {item["memory_id"] for item in entries}
            # WHY: 决定失效/问题解决后不能凭旧检索缓存复活；失败记录只是历史经验，不是当前阻断项。
            context["matches"] = [item for item in context["matches"]
                                  if item["kind"] in {"failure", "finding"} or item["memory_id"] in current_ids]
            for item in context["matches"]:
                if item["memory_id"] not in current_ids:
                    item["validity"] = "historical"
            # WHY: 最新摘要游标不等于上次查询已扫描的游标，不能把旧命中伪装成新检索结果。
            context["current_cursor"] = summary.get("cursor", 0)
            state["research_memory_artifact_ref"] = ref
    except Exception as exc:
        from app.agent.execution_budget import is_control_exception
        if is_control_exception(exc):
            raise
        state.pop("research_memory_context", None)
        state["research_memory_status"] = {"available": False, "error": type(exc).__name__}
        diagnostic = {"code": "research_memory_unavailable", "message": type(exc).__name__}
        if diagnostic not in state.setdefault("errors", []):
            state["errors"].append(diagnostic)


def execute_memory_search(state, trajectory, arguments):
    try:
        return _execute_memory_search(state, trajectory, arguments)
    except Exception as exc:
        from app.agent.execution_budget import is_control_exception
        from app.agent.react_context import ContextBudgetExceeded
        if is_control_exception(exc) or isinstance(exc, ContextBudgetExceeded):
            raise
        state.pop("research_memory_context", None)
        state.setdefault("errors", []).append({"code": "research_memory_search_failed", "message": type(exc).__name__})
        payload = {"status": "failed", "outcome": "blocked", "error": "research_memory_search_failed:" + type(exc).__name__}
        trajectory.observe(payload)
        return payload


def _execute_memory_search(state, trajectory, arguments):
    service = memory_service(state)
    if service is None:
        payload = {"status": "failed", "outcome": "blocked", "error": "research_memory_store_unavailable"}
    else:
        result = service.retrieve(str(state["session_id"]), **arguments,
                                  requested_ref=state.get("research_memory_artifact_ref"))
        state["research_memory_context"] = {**result, "task_fingerprint": task_fingerprint(state)}
        state["research_memory_artifact_ref"] = result["summary_ref"]
        service.memory.append_event(session_id=str(state["session_id"]), event_type="memory_retrieved",
            payload={"query": result["query"], "kind": result["kind"], "cursor": result["cursor"],
                     "memory_ids": [item["memory_id"] for item in result["matches"]]},
            source_refs=[result["summary_ref"]] if result["summary_ref"] else [])
        payload = {"status": "completed", "outcome": "no_change", "findings": [
            f"检索到 {len(result['matches'])} 条研究记忆；历史不授予当前证据权限"], "research_memory": result["matches"]}
    trajectory.observe(payload)
    from app.services.durable_execution_service import active_runtime
    runtime = active_runtime()
    if runtime:
        runtime.snapshot(state, ledger=state.get("agent_execution_budget") or {})
    return payload


def _append_audit(state, service, **event):
    from app.agent.execution_budget import is_control_exception
    try:
        check_execution()
        service.memory.append_event(session_id=str(state["session_id"]), **event)
    except Exception as exc:
        if is_control_exception(exc):
            raise
        # WHY: 辅助使用记录失败必须可诊断，但不能改变证据门禁或把已成功的研究动作判为失败。
        state.setdefault("errors", []).append({"code": "research_memory_write_failed", "message": type(exc).__name__})


def record_memory_feedback(state, decision, *, presented_matches=None):
    service = memory_service(state)
    matches = (state.get("research_memory_context") or {}).get("matches") or []
    if presented_matches is not None:
        matches = presented_matches
    if not service or not matches:
        return
    by_id = {item["memory_id"]: item for item in matches}
    feedback = [item.model_dump(mode="json") for item in decision.memory_feedback]
    # 未声明使用只能记录“已呈现”，不能伪称模型实际采用了检索结果。
    payload = {"decision_id": decision.decision_id, "action": decision.action,
               "presented_ids": list(by_id), "feedback": feedback, "usage_reported": bool(feedback)}
    _append_audit(state, service, event_type="memory_feedback", payload=payload,
                  event_key="memory-feedback:" + decision.decision_id)
    state["research_memory_pending_feedback"] = {"decision_id": decision.decision_id,
                                                  "memory_ids": [item["memory_id"] for item in feedback if item["disposition"] == "used"]}


def record_memory_outcome(state, result):
    pending = state.pop("research_memory_pending_feedback", None)
    service = memory_service(state)
    if pending and service:
        _append_audit(state, service, event_type="memory_action_outcome",
            payload={**pending, "operation": result.operation, "status": result.status.value,
                     "outcome": result.outcome.value, "delegation_status": result.delegation_status},
            event_key="memory-outcome:" + pending["decision_id"])


def record_memory_control_outcome(state, operation, status, outcome):
    from types import SimpleNamespace
    record_memory_outcome(state, SimpleNamespace(operation=operation,
        status=SimpleNamespace(value=status), outcome=SimpleNamespace(value=outcome), delegation_status=None))


def record_failure(state, trajectory, feedback):
    if not (feedback.get("error") or feedback.get("status") in {"failed", "rejected", "blocked", "deny", "escalate"}
            or feedback.get("outcome") in {"blocked", "degraded"}):
        return
    service = memory_service(state)
    if service is None:
        return
    try:
        # WHY: 停止后的辅助记忆不能再次写库或覆盖原取消/预算异常；研究提交边界仍会传播停止信号。
        check_execution()
        service.memory.append_event(session_id=str(state["session_id"]), event_type="agent_failure",
            payload={"operation": feedback["action"], "error": feedback.get("error") or
                         "; ".join(feedback.get("findings") or []) or str(feedback.get("outcome") or feedback.get("status")),
                     "status": feedback.get("status"), "outcome": feedback.get("outcome"),
                     "task": str(state.get("canonical_topic") or state.get("topic") or state.get("user_query") or "")[:240],
                     "source_refs": feedback.get("artifact_refs") or []},
            event_key=f"agent-failure:{trajectory.data['scope_id']}:{feedback['sequence']}")
    except Exception as exc:
        from app.agent.execution_budget import is_control_exception
        if not is_control_exception(exc):
            state.setdefault("errors", []).append({"code": "research_memory_write_failed", "message": type(exc).__name__})
