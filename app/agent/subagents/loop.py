"""持久化的专业 ReAct 循环；任务执行仍通过确定性 Controller。"""
from __future__ import annotations

import copy
from uuid import uuid4

from app.agent.action_contracts import CONTRACTS
from app.agent.action_registry import (
    ACTION_REGISTRY, allowed_actions, recovery_handoff_action, tool_schemas, validate_arguments,
)
from app.agent.context_builder import build_main_agent_context, _goal, _stable_hash, context_source_fingerprint
from app.agent.execution_budget import check_execution, AgentExecutionCancelled
from app.agent.main_policy import MainAgentPolicy, AgentDecisionError
from app.agent.react_context import ReActContext, execute_context_recall
from app.agent.sidecar import ActionSidecar
from app.core.config import get_settings
from app.prompt.agent_system import SYSTEM_PROMPTS, SUBAGENT_REACT_PROMPT
from app.schemas.agent_message_schema import AgentMessage, SubagentReport
from app.schemas.agent_task_schema import AgentRole, AgentTaskResult, AgentTaskStatus, AgentTaskOutcome


def _validate(name, arguments):
    if name == "report_to_main":
        report = SubagentReport.model_validate(arguments)
        if not report.message.strip() or any(action not in ACTION_REGISTRY for action in report.suggested_actions):
            raise ValueError("invalid report message or suggested action")
    else:
        validate_arguments(name, arguments)


def _session(state, role):
    return state.setdefault("subagent_sessions", {}).setdefault(role, {})


def _message(state, role, kind, content, suggested_actions=()):
    session = _session(state, role)
    message = AgentMessage(
        delegation_id=session["active"]["id"], sender="main" if kind == "task" else role,
        recipient=role if kind == "task" else "main", kind=kind,
        content=content[:1200], suggested_actions=list(suggested_actions),
    ).model_dump(mode="json")
    session.setdefault("messages", []).append(message)


def _checkpoint(state):
    from app.services.durable_execution_service import active_runtime
    runtime = active_runtime()
    if runtime:
        runtime.snapshot(state, ledger=state.get("agent_execution_budget") or {})


def record_subagent_observation(state, payload):
    """Controller 在同一原子快照中保存工具配对、进展与汇报，不提前结束主控调用。"""
    role = state["active_subagent"]
    trajectory = ReActContext(state, namespace=role)
    if (trajectory.data.get("pending") or {}).get("action") != payload.get("operation"):
        return
    trajectory.observe(payload)
    active = _session(state, role)["active"]
    active["last_result"] = payload
    if payload.get("status") == "completed":
        active["completed_operations"].append(payload["operation"])
    from app.agent.main_loop import _progress_fingerprint
    progress = _progress_fingerprint(state)
    active["no_progress"] = active["no_progress"] + 1 if progress == active["progress"] else 0
    active["cycled"] = progress != active["progress"] and progress in active["recent_progress"]
    active["recent_progress"] = (active["recent_progress"] + [progress])[-8:]
    active["progress"] = progress
    _message(state, role, "progress", "; ".join(str(s) for s in payload.get("findings", [])) or payload["operation"])


class SubagentLoop:
    def __init__(self, llm, controller):
        self.llm = llm
        self.controller = controller
        self.sidecar = ActionSidecar(llm)

    def run(self, state, *, decision=None, handlers, should_cancel=None,
            consume_mandatory_action=False, required_first=False):
        # WHY: 当前共享研究状态只支持串行；新委派不能冒用已有角色活动轨迹，恢复必须 decision=None。
        if decision is not None and state.get("active_subagent"):
            raise AgentDecisionError("another delegation is active; resume it explicitly")
        try:
            return self._run(state, decision=decision, handlers=handlers, should_cancel=should_cancel,
                             consume_mandatory_action=consume_mandatory_action, required_first=required_first)
        except Exception as exc:
            role = state.get("active_subagent")
            if role:
                # WHY: 控制异常必须原样上抛；只补齐内存中的失败观察，不在取消/失权后提交。
                ReActContext(state, namespace=role).observe({"status": "failed", "error": type(exc).__name__})
            raise

    def _run(self, state, *, decision, handlers, should_cancel,
             consume_mandatory_action, required_first):
        if state.get("active_subagent"):
            role = state["active_subagent"]
        else:
            spec = ACTION_REGISTRY[decision.action]
            role = spec.role.value
            from app.agent.main_loop import _progress_fingerprint
            progress = _progress_fingerprint(state)
            _session(state, role)["active"] = {
                "id": uuid4().hex, "operation": decision.action,
                "objective": decision.reason or spec.description,
                "arguments": copy.deepcopy(decision.arguments),
                "evidence_refs": list(decision.evidence_refs),
                "source_fingerprint": context_source_fingerprint(state),
                "goal_fingerprint": _stable_hash(_goal(state).model_dump(mode="json")),
                "rounds": 0, "completed_operations": [], "no_progress": 0,
                "progress": progress, "recent_progress": [progress], "cycled": False,
                "consume_mandatory_action": consume_mandatory_action,
                "required_first": required_first or consume_mandatory_action,
            }
            state["active_subagent"] = role
            _message(state, role, "task", decision.reason or spec.description)
        if role not in {r.value for r in AgentRole}:
            raise AgentDecisionError("invalid persisted subagent role")
        active = _session(state, role).get("active") or {}
        spec = ACTION_REGISTRY.get(active.get("operation"))
        if spec is None or spec.role is None or spec.role.value != role:
            raise AgentDecisionError("persisted delegation role does not own operation")
        trajectory = ReActContext(state, namespace=role)
        names = [name for name, spec in ACTION_REGISTRY.items() if spec.role and spec.role.value == role]
        policy = MainAgentPolicy(
            self.llm, system_prompt=SYSTEM_PROMPTS[role] + SUBAGENT_REACT_PROMPT,
            tools=tool_schemas([*names, "recall_context"]) + [{"type": "function", "function": {
                "name": "report_to_main", "description": "向主控汇报子任务结果、请求协助或说明阻断",
                "parameters": SubagentReport.model_json_schema(),
            }}], argument_validator=_validate, operation=role + "_agent_decision", delegation_messages=False,
        )
        settings = get_settings()
        while True:
            check_execution()
            if should_cancel and should_cancel():
                raise AgentExecutionCancelled("subagent cancelled")
            active = _session(state, role)["active"]
            if active["goal_fingerprint"] != _stable_hash(_goal(state).model_dump(mode="json")):
                return self._finish(state, role, "needs_help", "用户目标已更新，请主控重新委派。")
            if active["rounds"] >= settings.agent_subagent_max_rounds:
                return self._finish(state, role, "blocked", "子 Agent 决策轮数已达上限。")
            if active["no_progress"] >= settings.agent_subagent_no_progress_limit or active["cycled"]:
                return self._finish(state, role, "blocked", "子 Agent 无实质进展或状态循环，请主控调整任务。")
            available = [name for name in allowed_actions(state, set(handlers)) if name in names and name in handlers]
            required = active["operation"]
            # WHY: 局部改写的授权范围是主控指定章节；不能改选全篇生成来绕过约束。
            if required == "rewrite_sections":
                available = [name for name in available if name == required]
            mandatory = state.get("agent_mandatory_actions") or []
            if mandatory:
                # WHY: 下一项必做任务由主控另行委派；子循环不能执行后遗漏原子出队。
                available = ([name for name in available if name == mandatory[0].get("action")]
                             if required not in active["completed_operations"] else [])
            else:
                handoff = recovery_handoff_action(state)
                if handoff:
                    available = [name for name in available if name == handoff]
            if active["required_first"] and required not in active["completed_operations"]:
                available = [name for name in available if name == required]
            available.append("report_to_main")
            available.append("recall_context")
            context = self._context(state, role, names, available)
            active["rounds"] += 1
            ledger = state.setdefault("agent_execution_budget", {})
            ledger["subagent_decision_rounds"] = int(ledger.get("subagent_decision_rounds") or 0) + 1
            # WHY: 恢复后继续原委派的轮数与已提交观察，不重置局部预算或重放已完成动作。
            _checkpoint(state)
            chosen = policy.decide(context, allowed_actions=available, trajectory=trajectory)
            check_execution()
            active = _session(state, role)["active"]
            if chosen.action == "recall_context":
                execute_context_recall(trajectory, chosen.arguments)
                continue
            if chosen.action == "report_to_main":
                report = SubagentReport.model_validate(chosen.arguments)
                last = active.get("last_result") or {}
                if report.status == "completed" and (
                    required not in active["completed_operations"]
                    or last.get("status") != "completed"
                ):
                    trajectory.observe({"status": "rejected", "error": "delegated_operation_not_completed"})
                    continue
                return self._finish(state, role, report.status, report.message, report.suggested_actions)
            if chosen.action == required and chosen.arguments != active["arguments"]:
                trajectory.observe({"status": "rejected", "error": "delegated_arguments_are_immutable"})
                continue
            if settings.agent_sidecar_enabled and chosen.action in {
                item.strip() for item in settings.agent_sidecar_review_actions.split(",")
            }:
                review = self.sidecar.review(operation=chosen.action, arguments=chosen.arguments)
                _session(state, role).setdefault("sidecar_reviews", []).append({
                    "action": chosen.action, "verdict": review.verdict, "reason": review.reason,
                })
                if review.verdict != "allow":
                    trajectory.observe({"status": review.verdict, "error": review.reason})
                    if review.verdict == "escalate":
                        return self._finish(state, role, "needs_help", review.reason)
                    continue
            spec = ACTION_REGISTRY[chosen.action]
            result = self.controller.execute(
                role=spec.role, operation=chosen.action, objective=chosen.reason or active["objective"],
                state=state, handler=lambda working: handlers[chosen.action](working, chosen.arguments),
                constraints=chosen.arguments, expected_output=list(spec.output_fields),
                input_artifact_refs=(chosen.evidence_refs or active["evidence_refs"]
                                     if chosen.action == required else chosen.evidence_refs),
                should_cancel=should_cancel,
                consume_mandatory_action=bool(active["consume_mandatory_action"]
                    and chosen.action == required and required not in active["completed_operations"]),
                invalidate_evidence=True,
            )
            # 无持久化 runtime 或命中幂等结果时仍记录；已原子记录的观察不会重复。
            record_subagent_observation(state, result.model_dump(mode="json"))
            if result.status != AgentTaskStatus.COMPLETED:
                return self._finish(state, role, "blocked", result.error or "专业动作未成功提交。")

    @staticmethod
    def _context(state, role, names, available):
        fields = set().union(*(CONTRACTS[name].inputs for name in names))
        projected = {key: copy.deepcopy(value) for key, value in state.items() if key in fields}
        projected["artifact_manifest"] = {
            key: copy.deepcopy(value) for key, value in (state.get("artifact_manifest") or {}).items()
            if key in fields
        }
        session = _session(state, role)
        projected["agent_task_results"] = [item for item in state.get("agent_task_results", []) if item.get("role") == role]
        projected["agent_execution_budget"] = copy.deepcopy(state.get("agent_execution_budget") or {})
        projected["agent_context_usage"] = session.get("agent_context_usage", {})
        projected["agent_context_last_usage"] = session.get("agent_context_last_usage", {})
        context = build_main_agent_context(projected)
        active = session["active"]
        context.state.allowed_actions = available
        context.state.next_actions = available
        context.state.delegation = {key: active[key] for key in (
            "id", "operation", "objective", "arguments", "evidence_refs", "completed_operations", "rounds",
        )}
        context.state.agent_messages = copy.deepcopy(session.get("messages", [])[-6:])
        return context

    @staticmethod
    def _finish(state, role, status, message, suggestions=()):
        session = _session(state, role)
        active = session["active"]
        _message(state, role, status, message, suggestions)
        result = AgentTaskResult(
            task_id=active["id"], role=AgentRole(role), operation=active["operation"],
            status=AgentTaskStatus.COMPLETED,
            outcome=(AgentTaskOutcome((active.get("last_result") or {}).get("outcome", "unknown"))
                     if status == "completed" else AgentTaskOutcome.BLOCKED),
            delegation_status=status,
            findings=[message], open_questions=[] if status == "completed" else [{
                "category": status, "question": message, "suggested_actions": list(suggestions),
            }], input_state_fingerprint=active["source_fingerprint"],
            output_state_fingerprint=context_source_fingerprint(state),
        )
        ReActContext(state, namespace=role).observe({"status": status, "findings": [message]})
        ReActContext(state).observe(result.model_dump(mode="json"))
        session["last_delegation"] = copy.deepcopy(active)
        session.pop("active")
        state.pop("active_subagent")
        _checkpoint(state)
        return result
