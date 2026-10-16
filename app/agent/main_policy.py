"""Validated single-action decisions over a persistent ReAct trajectory."""

from __future__ import annotations

from pydantic import ValidationError

from app.agent.action_registry import ACTION_REGISTRY, tool_schemas, validate_arguments
from app.agent.react_context import ContextBudgetExceeded, ReActContext, encode
from app.core.exceptions import NativeToolsUnsupportedError
from app.prompt.agent_system import MAIN_AGENT_SYSTEM_PROMPT
from app.schemas.agent_decision_schema import AgentDecision
from app.schemas.context_schema import MainAgentContext


class AgentDecisionError(ValueError):
    pass


class MainAgentPolicy:
    def __init__(self, llm, *, correction_attempts: int = 1) -> None:
        self.llm = llm
        self.correction_attempts = max(0, int(correction_attempts))
        # WHY: 工具目录仅暴露协议，不授予执行权；合法动作始终由本轮状态与 Controller 检查。
        self.tools = tool_schemas(sorted(ACTION_REGISTRY))
        self.native_system = MAIN_AGENT_SYSTEM_PROMPT.strip()
        self.system = (
            MAIN_AGENT_SYSTEM_PROMPT.strip()
            + "\n核心工具目录（JSON 动作契约）："
            + encode(self.tools)
        )

    def decide(
        self,
        context: MainAgentContext,
        *,
        allowed_actions: list[str],
        trajectory: ReActContext | None = None,
    ) -> AgentDecision:
        trajectory = trajectory or ReActContext({})
        native = getattr(self.llm, "complete_tool_call", None)
        native_enabled = bool(
            getattr(self.llm, "native_tools_enabled", False)
        ) and not trajectory.data.get("native_tools_disabled", False)
        last_error = ""
        for attempt in range(self.correction_attempts + 1):
            messages = trajectory.request(
                context,
                system=self.native_system if native_enabled else self.system,
                tools=self.tools if native_enabled else [],
                allowed_actions=allowed_actions,
            )
            try:
                if callable(native) and native_enabled:
                    raw = native(messages, tools=self.tools, operation="main_agent_decision")
                    trajectory.state["agent_context_last_usage"] = dict(raw.get("usage") or {})
                    decision = AgentDecision(
                        action=str(raw.get("name") or ""),
                        arguments=dict(raw.get("arguments") or {}),
                        reason=str(raw.get("reason") or ""),
                        evidence_refs=list(raw.get("evidence_refs") or []),
                        transport="native_tool",
                        tool_call_id=raw.get("tool_call_id"),
                        usage=dict(raw.get("usage") or {}),
                    )
                    trajectory.proposed(decision, assistant_message=raw.get("assistant_message"))
                else:
                    content = self.llm.complete_messages(
                        messages,
                        response_format="json",
                        retry_empty=False,
                        operation="main_agent_decision",
                    )
                    trajectory.state["agent_context_last_usage"] = dict(
                        getattr(self.llm, "last_usage", {}) or {}
                    )
                    decision = AgentDecision.model_validate_json(content)
                    trajectory.proposed(decision)
                if decision.action not in allowed_actions:
                    raise AgentDecisionError(
                        f"action {decision.action!r} is not allowed in current state"
                    )
                validate_arguments(decision.action, decision.arguments)
                known_refs = {ref for item in context.key_evidence for ref in item.source_refs}
                if not set(decision.evidence_refs).issubset(known_refs):
                    raise AgentDecisionError(
                        "decision references evidence outside the current context"
                    )
                return decision
            except ContextBudgetExceeded:
                raise
            except NativeToolsUnsupportedError:
                native_enabled = False
                # WHY: 已确认不支持的协议在本会话保持 JSON，避免每轮重试破坏前缀并额外计费。
                trajectory.data["native_tools_disabled"] = True
                last_error = "原生工具协议不可用，请按 JSON 动作契约返回。"
            except (ValidationError, ValueError, TypeError) as exc:
                last_error = str(exc)
            trajectory.observe({"status": "rejected", "error": last_error})
            trajectory.data["messages"].append(
                {"role": "user", "content": encode({"decision_error": last_error})}
            )
            if attempt >= self.correction_attempts:
                break
        raise AgentDecisionError(last_error or "model did not return one valid action")
