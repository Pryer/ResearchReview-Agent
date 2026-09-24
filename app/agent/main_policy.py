"""仅基于五字段上下文产生单个主 Agent 动作。"""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError
from app.core.exceptions import NativeToolsUnsupportedError

from app.agent.action_registry import tool_schemas, validate_arguments
from app.schemas.agent_decision_schema import AgentDecision
from app.schemas.context_schema import MainAgentContext


class AgentDecisionError(ValueError):
    pass


class MainAgentPolicy:
    def __init__(self, llm, *, correction_attempts: int = 1) -> None:
        binder = getattr(llm, "for_agent", None)
        self.llm = binder("main") if callable(binder) else llm
        self.correction_attempts = max(0, int(correction_attempts))

    def decide(
        self,
        context: MainAgentContext,
        *,
        allowed_actions: list[str],
    ) -> AgentDecision:
        """发送一次干净的五字段消息；纠错轮也不附带原始资料或旧对话。"""
        # WHY: 原生 tool_choice=required；工具清单必须与本轮允许动作一致，
        # 否则模型可能被迫选到当前状态不可执行的工具并浪费纠错轮。
        tools = tool_schemas(allowed_actions)
        native = getattr(self.llm, "complete_tool_call", None)
        native_enabled = bool(getattr(self.llm, "native_tools_enabled", False))
        last_error = ""
        for attempt in range(self.correction_attempts + 1):
            current = context.model_copy(deep=True)
            current.state.allowed_actions = list(allowed_actions)
            if last_error:
                current.state.failure_reasons.append(last_error)
            payload = json.dumps(current.model_dump(mode="json"), ensure_ascii=False)
            try:
                if callable(native) and native_enabled:
                    raw = native(
                        [{"role": "user", "content": payload}],
                        tools=tools,
                        operation="main_agent_decision",
                    )
                    decision = AgentDecision(
                        action=str(raw.get("name") or ""),
                        arguments=dict(raw.get("arguments") or {}),
                        reason=str(raw.get("reason") or ""),
                        evidence_refs=list(raw.get("evidence_refs") or []),
                        transport="native_tool",
                        tool_call_id=raw.get("tool_call_id"),
                        usage=dict(raw.get("usage") or {}),
                    )
                else:
                    content = self.llm.complete_messages(
                        [{"role": "user", "content": payload}],
                        response_format="json",
                        retry_empty=False,
                        operation="main_agent_decision",
                    )
                    decision = AgentDecision.model_validate_json(content)
                if decision.action not in allowed_actions:
                    raise AgentDecisionError(
                        f"action {decision.action!r} is not allowed in current state"
                    )
                validate_arguments(decision.action, decision.arguments)
                known_refs = {ref for item in current.key_evidence for ref in item.source_refs}
                if not set(decision.evidence_refs).issubset(known_refs):
                    raise AgentDecisionError("decision references evidence outside the current context")
                return decision
            except NativeToolsUnsupportedError:
                native_enabled = False
                last_error = "原生工具协议不可用，请按 JSON 动作契约返回。"
                if attempt >= self.correction_attempts:
                    break
            except (ValidationError, ValueError, TypeError, json.JSONDecodeError) as exc:
                last_error = str(exc)
                if attempt >= self.correction_attempts:
                    break
        raise AgentDecisionError(last_error or "model did not return one valid action")
