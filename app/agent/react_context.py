"""Append-only ReAct epochs; authoritative research data stays in Artifact Store."""

from __future__ import annotations

import copy
import json
from uuid import uuid4
from typing import Any

from app.core.config import get_settings


class ContextBudgetExceeded(ValueError):
    pass


def is_failure(item):
    return bool(item.get("error") or item.get("status") in {"failed", "rejected", "blocked", "deny", "escalate"}
                or item.get("outcome") in {"blocked", "degraded"})


def working_memory(records, settings):
    latest = {item.get("action"): i for i, item in enumerate(records)}
    failures = [i for i in latest.values() if is_failure(records[i])]
    pinned = set(sorted(failures, reverse=True)[:settings.agent_context_failure_items])
    priority = sorted(pinned, reverse=True) + [i for i in range(len(records)-1, -1, -1) if i not in pinned]
    selected = set()
    for i in priority:
        candidate = [records[j] for j in sorted(selected | {i})]
        if len(candidate) <= settings.agent_context_memory_items and len(encode(candidate)) <= settings.agent_context_memory_chars:
            selected.add(i)
    return [copy.deepcopy(records[i]) for i in sorted(selected)], {
        records[i].get("sequence") for i in pinned if i in selected}


def encode(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def estimate_tokens(messages: list[dict], tools: list[dict]) -> int:
    # WHY: 不混用不同厂商的 tokenizer；UTF-8 字节上界加消息开销是保守估算，
    # 实际 usage 单独记录。窗口必须配置为主备模型可用窗口的较小值。
    return len(encode({"messages": messages, "tools": tools}).encode("utf-8")) + 32 * (
        len(messages) + len(tools)
    )


def archive_payload(state: dict, kind: str, payload: dict, *, scope_id: str | None = None,
                    observation_sequence: int | None = None, operation: str | None = None) -> list[str]:
    from app.services.research_execution_service import artifact_store

    store = artifact_store()
    if store is None or not state.get("session_id"):
        raise ContextBudgetExceeded("context_archive_store_unavailable")
    try:
        serialized = encode(payload)
        refs = [
            store.persist_payload(
                session_id=str(state["session_id"]),
                artifact_type=kind,
                payload={"value": serialized[offset : offset + 16000]},
                provenance={"encoding": "json", "offset": offset},
            )
            for offset in range(0, len(serialized), 16000)
        ]
        # WHY: 大结果的分片引用本身也可能超预算；模型只看到一个受控索引引用。
        index = store.persist_payload(
            session_id=str(state["session_id"]),
            artifact_type="context_archive_index",
            payload={"fragments": refs, "artifact_type": kind, "encoding": "json", "scope_id": scope_id,
                     "observation_sequence": observation_sequence, "operation": operation},
        )
        store.repo.db.flush()
        return [index]
    except Exception as exc:
        from app.agent.execution_budget import is_control_exception

        if is_control_exception(exc):
            raise
        raise ContextBudgetExceeded("context_archive_failed:" + type(exc).__name__) from exc


class ReActContext:
    def __init__(self, state: dict, *, namespace: str | None = None) -> None:
        self.state = state
        self.namespace = namespace
        self.container.setdefault(
            "main_agent_trajectory",
            {
                "version": 1,
                "epoch": 0,
                "messages": [],
                "goals": [],
                "memory": [],
                "archive_refs": [],
            },
        )
        if self.data.get("version") != 1:
            raise ContextBudgetExceeded("unsupported_trajectory_version")
        self.data.setdefault("scope_id", uuid4().hex)
        # WHY: 旧检查点的 memory 是全历史，长度可用于一次性迁移；之后累计序号不随裁剪回退。
        self.data.setdefault("observation_count", len(self.data["memory"]))

    @property
    def container(self) -> dict:
        if self.namespace is None:
            return self.state
        return self.state.setdefault("subagent_sessions", {}).setdefault(self.namespace, {})

    @property
    def data(self) -> dict:
        # WHY: Controller 提交会替换 state 内容，不能持有提交前的嵌套字典引用。
        return self.container["main_agent_trajectory"]

    def request(
        self, context, *, system: str, tools: list[dict], allowed_actions: list[str]
    ) -> list[dict]:
        if self.data.get("pending"):
            raise ContextBudgetExceeded("unresolved_tool_call_in_checkpoint")
        goal = context.goal.model_dump(mode="json")
        if not self.data["goals"] or self.data["goals"][-1] != goal:
            self.data["goals"].append(copy.deepcopy(goal))
            self.data["messages"].append({"role": "user", "content": encode({"goal": goal})})
        status = context.model_dump(mode="json")
        status["state"]["allowed_actions"] = list(allowed_actions)
        # WHY: 状态栏也追加保留；下一次请求完全保留本次输入前缀，直至 epoch 压缩。
        prefix = {"role": "system", "content": system}
        settings = get_settings()
        reserve = int(settings.llm_control_plane_max_tokens)
        window = int(settings.agent_context_window_tokens)
        threshold = float(settings.agent_context_compact_ratio)
        if reserve < 1 or window <= reserve or not 0 < threshold < 1:
            raise ContextBudgetExceeded("invalid_context_budget_configuration")
        status["state"]["context_budget"].update(
            {
                "window_tokens": window,
                "output_reserve_tokens": reserve,
                "estimate_method": "utf8_upper_bound",
            }
        )
        status_message = {"role": "user", "content": encode(status)}

        def build():
            return [prefix, *copy.deepcopy(self.data["messages"]), copy.deepcopy(status_message)]

        messages = build()
        before = estimate_tokens(messages, tools)
        due = before + reserve >= int(window * threshold) and (
            not self.data["epoch"]
            or self.data["observation_count"] - self.data.get("last_compacted_turn", 0)
            > max(0, int(settings.agent_context_recent_turns))
        )
        memory_over = (len(self.data["memory"]) > settings.agent_context_memory_items
                       or len(encode(self.data["memory"])) > settings.agent_context_memory_chars)
        if (due or before + reserve > window or memory_over) and self.data["memory"]:
            self._compact(status, input_budget=window - reserve, system=system, tools=tools)
            messages = build()
        estimate = estimate_tokens(messages, tools)
        self.container["agent_context_usage"] = {
            "estimated_input_tokens": estimate,
            "estimate_method": "utf8_upper_bound",
            "output_reserve_tokens": reserve,
            "window_tokens": window,
            "before_compaction_tokens": before,
            "epoch": self.data["epoch"],
        }
        if estimate + reserve > window:
            raise ContextBudgetExceeded("context_window_exceeded_after_compaction")
        self.data["messages"].append(status_message)
        return messages

    def _compact(self, status: dict, *, input_budget: int | None = None,
                 system: str = "", tools: list[dict] | None = None) -> None:
        data = self.data
        if data.get("pending"):
            raise ContextBudgetExceeded("cannot_compact_pending_tool_call")
        refs = archive_payload(
            self.state,
            "context_epoch",
            {
                "epoch": data["epoch"],
                "messages": data["messages"],
                "memory": data["memory"],
                "goals": data["goals"],
                "observation_count": data["observation_count"],
                "previous_archive_refs": data["archive_refs"],
            },
            scope_id=data["scope_id"],
        )
        settings = get_settings()
        # WHY: 少量尚未被同动作成功观察取代的失败经验优先于纯时间顺序；仍不能挤掉硬约束。
        retained, pinned = working_memory(data["memory"], settings)
        # WHY: 最新门禁/证据在紧随其后的 Status Bar 中重建，不在 checkpoint 再复制一整份动态状态。
        summary = {}
        summary["observation_count"] = data["observation_count"]
        summary["history_recall"] = "recall_context(evidence_id)"
        summary["archive_refs"] = refs
        # WHY: 新 Epoch 用结构化动作/观察重建，不重放旧 assistant/tool/reasoning_content。
        # 同一 Epoch 内仍原样保留真实工具调用与 Provider 必需的协议字段。
        latest_goal = copy.deepcopy(status.get("goal") or (data["goals"][-1] if data["goals"] else {}))
        def rebuild():
            summary["observations"] = copy.deepcopy(retained)
            return [{"role": "user", "content": encode({"goal": latest_goal})},
                    {"role": "user", "content": encode({"context_checkpoint": summary})}]
        messages = rebuild()
        while retained and input_budget is not None and estimate_tokens([
            {"role": "system", "content": system}, *messages,
            {"role": "user", "content": encode(status)},
        ], tools or []) > input_budget:
            removable = next((i for i, item in enumerate(retained) if item.get("sequence") not in pinned), 0)
            retained.pop(removable)
            messages = rebuild()
        data["messages"] = messages
        data["goals"] = [latest_goal]
        data["memory"] = retained
        data["archive_refs"] = refs
        data["epoch"] += 1
        data["last_compacted_turn"] = data["observation_count"]

    def proposed(self, decision, *, assistant_message: dict | None = None) -> None:
        if self.data.get("pending"):
            raise ContextBudgetExceeded("tool_call_already_pending")
        native = decision.transport == "native_tool"
        if native:
            message = assistant_message or {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": decision.tool_call_id or decision.decision_id,
                        "type": "function",
                        "function": {
                            "name": decision.action,
                            "arguments": encode(decision.arguments),
                        },
                    }
                ],
            }
        else:
            message = {
                "role": "assistant",
                "content": encode(
                    {
                        "action": decision.action,
                        "arguments": decision.arguments,
                        "reason": decision.reason,
                        "evidence_refs": decision.evidence_refs,
                    }
                ),
            }
        self.data["messages"].append(copy.deepcopy(message))
        self.data["pending"] = {
            "action": decision.action,
            "native": native,
            "id": message["tool_calls"][0]["id"] if native else decision.decision_id,
            "decision": {
                "action": decision.action,
                "arguments": decision.arguments,
                "reason": decision.reason,
                "evidence_refs": decision.evidence_refs,
            },
        }

    def observe(self, payload: dict) -> None:
        pending = self.data.get("pending")
        if not pending:
            return
        settings = get_settings()
        limit = max(1024, int(settings.agent_context_tool_result_chars))
        refs = []
        # WHY: 小观察也要保存完整载荷；工作摘要被裁剪后，原始字段仍可审计回查。
        from app.services.research_execution_service import artifact_store

        if artifact_store() is not None and self.state.get("session_id"):
            refs = archive_payload(self.state, "tool_observation", payload, scope_id=self.data["scope_id"],
                                   observation_sequence=self.data["observation_count"] + 1, operation=pending["action"])
        feedback = {
            "action": pending["action"],
            "status": payload.get("status", "unknown"),
            "outcome": payload.get("outcome"),
            "error": str(payload.get("error") or "")[:240],
            "findings": [str(item)[:240] for item in (payload.get("findings") or [])[:3]],
            "evidence_refs": payload.get("evidence_refs") or [],
            "output_artifact_refs": payload.get("output_artifact_refs") or [],
            "artifact_refs": refs,
        }
        if payload.get("history") is not None:
            feedback["history"] = payload["history"]
        if payload.get("research_memory") is not None:
            feedback["research_memory"] = payload["research_memory"]
        if payload.get("delegation_status") is not None:
            feedback["delegation_status"] = payload["delegation_status"]
        visible = encode(feedback)
        if len(visible) > limit and refs:
            # WHY: 全量引用保留在归档中，不能截断 JSON 或让大引用数组挤掉执行反馈。
            feedback.update(
                {
                    "evidence_refs": [],
                    "output_artifact_refs": [],
                    "references_archived": True,
                    "evidence_ref_count": len(payload.get("evidence_refs") or []),
                    "output_artifact_ref_count": len(payload.get("output_artifact_refs") or []),
                }
            )
            visible = encode(feedback)
        # 不以截断 JSON/引用方式挤入预算；无法表达保留集时明确熔断。
        if len(visible) > limit:
            raise ContextBudgetExceeded("tool_observation_retention_set_exceeds_limit")
        message = (
            {"role": "tool", "tool_call_id": pending["id"], "content": visible}
            if pending["native"]
            else {"role": "user", "content": encode({"observation": json.loads(visible)})}
        )
        self.data["messages"].append(message)
        self.data["observation_count"] += 1
        self.data["memory"].append({**feedback, "sequence": self.data["observation_count"],
                                    "decision": copy.deepcopy(pending["decision"])})
        self.data.pop("pending")
        from app.services.research_memory_retrieval import record_failure
        record_failure(self.state, self, self.data["memory"][-1])


def execute_context_recall(trajectory: ReActContext, arguments: dict) -> dict:
    from app.services.context_history_service import recall_history
    from app.agent.execution_budget import is_control_exception
    from app.services.durable_execution_service import active_runtime
    try:
        payload = recall_history(trajectory.state, trajectory, **arguments)
    except Exception as exc:
        if is_control_exception(exc):
            raise
        payload = {"status": "failed", "outcome": "blocked", "error": "context_recall_failed:" + type(exc).__name__}
        trajectory.state.setdefault("errors", []).append({"code": "context_recall_failed", "message": type(exc).__name__})
    trajectory.observe(payload)
    runtime = active_runtime()
    if runtime:
        runtime.snapshot(trajectory.state, ledger=trajectory.state.get("agent_execution_budget") or {})
    return payload


def record_tool_observation(state: dict, payload: dict) -> None:
    role = state.get("active_subagent")
    if role:
        from app.agent.subagents.loop import record_subagent_observation
        record_subagent_observation(state, payload)
        return
    data = state.get("main_agent_trajectory") or {}
    if (data.get("pending") or {}).get("action") == payload.get("operation"):
        ReActContext(state).observe(payload)
