"""Append-only ReAct epochs; authoritative research data stays in Artifact Store."""

from __future__ import annotations

import copy
import json
from typing import Any

from app.core.config import get_settings


class ContextBudgetExceeded(ValueError):
    pass


def encode(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def estimate_tokens(messages: list[dict], tools: list[dict]) -> int:
    # WHY: 不混用不同厂商的 tokenizer；UTF-8 字节上界加消息开销是保守估算，
    # 实际 usage 单独记录。窗口必须配置为主备模型可用窗口的较小值。
    return len(encode({"messages": messages, "tools": tools}).encode("utf-8")) + 32 * (
        len(messages) + len(tools)
    )


def archive_payload(state: dict, kind: str, payload: dict) -> list[str]:
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
            payload={"fragments": refs, "artifact_type": kind, "encoding": "json"},
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
            or len(self.data["memory"]) - self.data.get("last_compacted_turn", 0)
            > max(0, int(settings.agent_context_recent_turns))
        )
        if (due or before + reserve > window) and self.data["memory"]:
            self._compact(status)
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

    def _compact(self, status: dict) -> None:
        data = self.data
        refs = archive_payload(
            self.state,
            "context_epoch",
            {
                "epoch": data["epoch"],
                "messages": data["messages"],
                "previous_archive_refs": data["archive_refs"],
            },
        )
        # WHY: 只按完整 assistant/result 交互边界保留尾部；不截断 tool_call_id 配对。
        ends = [
            i + 1
            for i, item in enumerate(data["messages"])
            if item["role"] == "tool" or (item["role"] == "user" and "observation" in _object(item))
        ]
        keep = max(0, int(get_settings().agent_context_recent_turns))
        start = ends[-keep - 1] if keep and len(ends) > keep else 0
        recent = data["messages"][start : ends[-1]] if keep and ends else []
        # Drop status-only material before the first retained decision.
        first = next(
            (i for i, item in enumerate(recent) if item["role"] == "assistant"), len(recent)
        )
        recent = recent[first:]
        summary = {
            key: status[key] for key in ("state", "key_evidence", "decisions", "open_questions")
        }
        summary["observations"] = copy.deepcopy(data["memory"])
        summary["archive_refs"] = refs
        data["messages"] = [
            *[{"role": "user", "content": encode({"goal": goal})} for goal in data["goals"]],
            {"role": "user", "content": encode({"context_checkpoint": summary})},
            *copy.deepcopy(recent),
        ]
        data["archive_refs"] = refs
        data["epoch"] += 1
        data["last_compacted_turn"] = len(data["memory"])

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
        raw = encode(payload)
        refs = []
        if len(raw) > limit:
            # 原始任务结果也留在权威 state；有持久化存储时额外给出不可变引用。
            from app.services.research_execution_service import artifact_store

            if artifact_store() is not None and self.state.get("session_id"):
                refs = archive_payload(self.state, "tool_observation", payload)
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
        self.data["memory"].append({**feedback, "decision": copy.deepcopy(pending["decision"])})
        self.data.pop("pending")


def _object(message: dict) -> dict:
    try:
        return json.loads(message.get("content") or "{}")
    except (ValueError, TypeError):
        return {}


def record_tool_observation(state: dict, payload: dict) -> None:
    role = state.get("active_subagent")
    if role:
        from app.agent.subagents.loop import record_subagent_observation
        record_subagent_observation(state, payload)
        return
    data = state.get("main_agent_trajectory") or {}
    if (data.get("pending") or {}).get("action") == payload.get("operation"):
        ReActContext(state).observe(payload)
