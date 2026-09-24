"""五字段驱动的主 Agent 单动作决策循环。"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from collections.abc import Callable
from typing import Any

from app.agent.action_registry import action_spec, allowed_actions, validate_arguments
from app.agent.context_builder import refresh_main_agent_context
from app.agent.controller import AgentController
from app.agent.execution_budget import AgentBudgetExceeded, budget_scope, check_execution
from app.agent.execution import AgentCancelledError
from app.core.exceptions import LLMInvocationError
from app.agent.main_policy import AgentDecisionError, MainAgentPolicy
from app.schemas.agent_decision_schema import AgentDecision
from app.core.config import get_settings
from app.database.runtime_repository import RuntimeConflict


TERMINAL_STATUSES = {"completed", "partial", "blocked", "failed", "cancelled", "waiting_user"}


def _progress_fingerprint(state: dict[str, Any]) -> str:
    def stable_items(value: Any) -> list[str]:
        # WHY: 排名或列表顺序抖动不代表证据集合变化；保留每项内容以识别真正的补证。
        return sorted(json.dumps(item, ensure_ascii=False, sort_keys=True, default=str) for item in (value or []))

    payload = {
        "scope": state.get("selected_scope") or {},
        "candidates": stable_items(state.get("candidate_papers")),
        "details": stable_items(state.get("paper_details")),
        "cards": stable_items(state.get("paper_cards")),
        "routes": stable_items(state.get("validated_routes")),
        "taxonomy": state.get("dynamic_taxonomy") or {},
        "claims": stable_items(state.get("claim_plans")),
        "gaps": state.get("evidence_gap_report") or {},
        "review": state.get("review") or "",
        "quality": state.get("quality_gate") or {},
        "valid_citations": state.get("unique_valid_cited_paper_count") or 0,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:20]


class MainAgentLoop:
    def __init__(self, llm, *, controller: AgentController | None = None) -> None:
        self.policy = MainAgentPolicy(llm)
        self.controller = controller or AgentController()

    def run(
        self, state, *, handlers, finish_validator, should_cancel=None,
    ) -> str:
        try:
            with budget_scope(state, should_cancel):
                return self._run(state, handlers=handlers, finish_validator=finish_validator,
                                 should_cancel=should_cancel)
        except (AgentBudgetExceeded, AgentCancelledError, AgentDecisionError, LLMInvocationError, RuntimeConflict) as exc:
            status = "cancelled" if isinstance(exc, AgentCancelledError) else "blocked" if isinstance(exc, AgentBudgetExceeded) else "failed"
            state["result_status"] = status
            if status == "blocked":
                state["generation_blocked"] = True
            state.setdefault("errors", []).append({"code": type(exc).__name__, "message": str(exc)})
            return status

    def _run(
        self,
        state: dict[str, Any],
        *,
        handlers: dict[str, Callable[[dict[str, Any], dict[str, Any]], Any]],
        finish_validator: Callable[[dict[str, Any]], bool],
        should_cancel: Callable[[], bool] | None = None,
    ) -> str:
        settings = get_settings()
        no_progress = 0
        previous_progress = _progress_fingerprint(state)
        progress_limit = max(1, int(settings.agent_main_no_progress_limit))
        recent_progress = deque([previous_progress], maxlen=max(4, 2 * progress_limit + 1))
        for round_index in range(int(settings.agent_main_max_rounds)):
            check_execution()
            if should_cancel and should_cancel():
                state["result_status"] = "cancelled"
                return "cancelled"
            names = allowed_actions(state, set(handlers))
            mandatory = list(state.get("agent_mandatory_actions") or [])
            if mandatory:
                required = mandatory[0]
                action_name = str(required.get("action") or "") if isinstance(required, dict) else ""
                if action_name not in names:
                    state["result_status"] = "blocked"
                    state["generation_blocked"] = True
                    state.setdefault("errors", []).append({
                        "code": "required_recovery_action_unavailable",
                        "message": action_name,
                    })
                    return "blocked"
                names = [action_name]
            from app.services.research_execution_service import checkpoint_artifacts
            checkpoint_artifacts(state)
            state["allowed_agent_actions"] = names
            snapshot = refresh_main_agent_context(state)
            if snapshot.validation_errors:
                state["errors"] = list(state.get("errors") or []) + [
                    {"code": "main_context_invalid", "message": ",".join(snapshot.validation_errors)}
                ]
                state["result_status"] = "failed"
                return "failed"
            try:
                if mandatory:
                    arguments = dict(required.get("arguments") or {})
                    validate_arguments(action_name, arguments)
                    decision = AgentDecision(
                        action=action_name,
                        arguments=arguments,
                        reason="履行已选定的质量恢复动作",
                    )
                else:
                    decision = self.policy.decide(snapshot.context, allowed_actions=names)
            except (AgentDecisionError, ValueError) as exc:
                state["errors"] = list(state.get("errors") or []) + [
                    {"code": "invalid_agent_decision", "message": str(exc)}
                ]
                state["result_status"] = "failed"
                return "failed"
            state.setdefault("main_agent_decisions", []).append(
                decision.model_dump(mode="json")
            )
            ledger = state.setdefault("agent_execution_budget", {})
            ledger["decision_rounds"] = int(ledger.get("decision_rounds") or 0) + 1
            if int(ledger.get("llm_tokens") or 0) > int(ledger["token_limit"]):
                state["result_status"] = "blocked"
                state["generation_blocked"] = True
                state.setdefault("errors", []).append({
                    "code": "agent_token_budget_exhausted",
                    "message": "主 Agent 模型 token 预算已耗尽",
                })
                return "blocked"
            action = decision.action
            if action == "request_clarification":
                state["clarification"] = {
                    "needed": True,
                    "question": decision.arguments["question"],
                    "kind": "main_agent",
                }
                state["result_status"] = "waiting_user"
                return "waiting_user"
            if action == "report_blocked":
                state["generation_blocked"] = True
                state["errors"] = list(state.get("errors") or []) + [{
                    "code": "main_agent_blocked",
                    "message": decision.arguments["reason"],
                }]
                state["result_status"] = "blocked"
                return "blocked"
            if action == "request_finish":
                if finish_validator(state):
                    return str(state.get("result_status") or "completed")
                state.setdefault("main_agent_rejections", []).append({
                    "action": action,
                    "reason": "finish_rejected_by_quality_gates",
                    "round": round_index,
                })
            else:
                spec = action_spec(action)
                handler = handlers.get(action)
                if spec.role is None or handler is None:
                    state["result_status"] = "failed"
                    state.setdefault("errors", []).append({
                        "code": "unbound_agent_action", "message": action,
                    })
                    return "failed"
                try:
                    result = self.controller.execute(
                        role=spec.role,
                        operation=action,
                        objective=decision.reason or spec.description,
                        state=state,
                        handler=lambda working, fn=handler, args=decision.arguments: fn(working, args),
                        constraints=decision.arguments,
                        expected_output=list(spec.output_fields),
                        input_artifact_refs=decision.evidence_refs,
                        should_cancel=should_cancel,
                        consume_mandatory_action=bool(mandatory),
                    )
                except AgentBudgetExceeded as exc:
                    state["result_status"] = "blocked"
                    state["generation_blocked"] = True
                    state.setdefault("errors", []).append({
                        "code": "AgentBudgetExceeded",
                        "message": str(exc),
                    })
                    return "blocked"
                if result.status.value in {"failed", "cancelled"}:
                    state["result_status"] = result.status.value
                    return result.status.value

            current_progress = _progress_fingerprint(state)
            no_progress = no_progress + 1 if current_progress == previous_progress else 0
            state_cycled = current_progress != previous_progress and current_progress in recent_progress
            recent_progress.append(current_progress)
            previous_progress = current_progress
            if no_progress >= progress_limit:
                state["generation_blocked"] = True
                state["result_status"] = "blocked"
                state.setdefault("errors", []).append({
                    "code": "agent_no_progress",
                    "message": "主 Agent 连续动作未产生实质研究进展",
                })
                return "blocked"
            if state_cycled:
                state["generation_blocked"] = True
                state["result_status"] = "blocked"
                state.setdefault("errors", []).append({
                    "code": "agent_state_cycle",
                    "message": "主 Agent 研究状态反复往返，未产生持续进展",
                })
                return "blocked"
        state["generation_blocked"] = True
        state["result_status"] = "blocked"
        state.setdefault("errors", []).append({
            "code": "agent_round_limit",
            "message": "主 Agent 达到最大决策轮数",
        })
        return "blocked"
