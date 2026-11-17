"""持久化的专业 ReAct 循环；任务执行仍通过确定性 Controller。"""
from __future__ import annotations

import copy
from uuid import uuid4

from app.agent.action_registry import (
    ACTION_REGISTRY, allowed_actions, recovery_handoff_action, tool_schemas, validate_arguments,
)
from app.agent.context_builder import _goal, _stable_hash, context_source_fingerprint
from app.agent.execution_budget import check_execution, AgentExecutionCancelled
from tests.reference_v1.main_policy import MainAgentPolicy, AgentDecisionError
from app.agent.react_context import ReActContext, execute_context_recall, observe_terminal
from app.agent.sidecar import ActionSidecar
from tests.reference_v1.delegation_context import context_packet, subagent_context
from app.core.config import get_settings
from app.core.exceptions import LLMInvocationError
from app.prompt.agent_system import SYSTEM_PROMPTS
from tests.reference_v1.prompts import SUBAGENT_REACT_PROMPT, SUBAGENT_REPORT_REPAIR_HINT
from app.schemas.agent_message_schema import AgentMessage, SubagentReport
from app.schemas.agent_task_schema import AgentRole, AgentTaskResult, AgentTaskStatus, AgentTaskOutcome


def _validate(name, arguments):
    if name == "report_to_main":
        message = arguments.get("message")
        limit = SubagentReport.model_json_schema()["properties"]["message"]["maxLength"]
        if isinstance(message, str) and len(message) > limit:
            # WHY: 不截断或放宽上限；只返回实际长度和修复提示，避免错误回显整段汇报。
            raise ValueError(f"report_to_main.message length {len(message)} exceeds {limit}. " + SUBAGENT_REPORT_REPAIR_HINT)
        report = SubagentReport.model_validate(arguments)
        if not report.message.strip():
            raise ValueError("report message must not be blank")
        if any(action not in ACTION_REGISTRY for action in report.suggested_actions):
            raise ValueError("suggested_actions must contain registered action names; put prose in message or use []")
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
    from tests.reference_v1.main_loop import _progress_fingerprint
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
            consume_mandatory_action=False, required_first=False, decision_context=None):
        # WHY: 当前共享研究状态只支持串行；新委派不能冒用已有角色活动轨迹，恢复必须 decision=None。
        if decision is not None and state.get("active_subagent"):
            raise AgentDecisionError("another delegation is active; resume it explicitly")
        try:
            return self._run(state, decision=decision, handlers=handlers, should_cancel=should_cancel,
                             consume_mandatory_action=consume_mandatory_action, required_first=required_first,
                             decision_context=decision_context)
        except Exception as exc:
            role = state.get("active_subagent")
            if role:
                # WHY: 控制异常必须原样上抛；只补齐内存中的失败观察，不在取消/失权后提交。
                observe_terminal(state, {"status": "failed", "error": type(exc).__name__}, namespace=role)
            raise

    def _run(self, state, *, decision, handlers, should_cancel,
             consume_mandatory_action, required_first, decision_context):
        if state.get("active_subagent"):
            role = state["active_subagent"]
            old = _session(state, role).get("active") or {}
            if role == "analysis" and old.get("operation") == "validate_result":
                if "validate_result" in (old.get("completed_operations") or []):
                    return self._finish(state, role, "needs_help", "历史核验已执行；请Main按当前版本决定是否委派Check复验。")
                from app.schemas.agent_decision_schema import AgentDecision
                _session(state, role)["last_delegation"] = copy.deepcopy(old)
                _session(state, role).pop("active", None)
                state.pop("active_subagent")
                decision = AgentDecision(action="validate_result", arguments={}, reason=old.get("objective") or "恢复核验")
                return self._run(state, decision=decision, handlers=handlers, should_cancel=should_cancel,
                    consume_mandatory_action=old.get("consume_mandatory_action", False),
                    required_first=True, decision_context=None)
        else:
            spec = ACTION_REGISTRY[decision.action]
            role = spec.role.value
            from tests.reference_v1.main_loop import _progress_fingerprint
            progress = _progress_fingerprint(state)
            try:
                packet = context_packet(state, decision, decision_context=decision_context)
            except ValueError as exc:
                raise AgentDecisionError(str(exc)) from exc
            from app.agent.delegation_memory import prepare_work_context, begin_delegation
            _session(state, role)
            work_context = prepare_work_context(state, role)
            begin_delegation(state, role)
            _session(state, role)['active'] = {
                "id": uuid4().hex, "operation": decision.action,
                "objective": decision.reason or spec.description,
                "arguments": copy.deepcopy(decision.arguments),
                "context_packet": packet, "work_context": work_context,
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
        report_parameters = SubagentReport.model_json_schema()
        # WHY: 汇报可以建议其他角色的注册动作，但不能执行它们；Schema 与运行时
        # 校验同源，避免模型按普通字符串 Schema 填自然语言后被反复拒绝。
        report_parameters["properties"]["suggested_actions"]["items"]["enum"] = sorted(ACTION_REGISTRY)
        policy = MainAgentPolicy(
            self.llm, system_prompt=SYSTEM_PROMPTS[role] + SUBAGENT_REACT_PROMPT,
            tools=tool_schemas([*names, "recall_context"]) + [{"type": "function", "function": {
                "name": "report_to_main", "description": "向主控汇报子任务结果、请求协助或说明阻断",
                "parameters": report_parameters,
            }}], argument_validator=_validate, operation=role + "_agent_decision", delegation_messages=False,
        )
        settings = get_settings()
        while True:
            check_execution()
            if should_cancel and should_cancel():
                raise AgentExecutionCancelled("subagent cancelled")
            active = _session(state, role)["active"]
            last = active.get("last_result") or {}
            if (settings.agent_deterministic_subagent_report and "delivery_mode" in state
                    and active["operation"] in active["completed_operations"] and last.get("status") == "completed"):
                outcome = str(last.get("outcome") or "unknown")
                status = "completed" if outcome == "succeeded" else "needs_help"
                suggestions = [item["action"] for item in last.get("suggested_actions") or []
                               if item.get("action") in ACTION_REGISTRY]
                return self._finish(state, role, status,
                    "; ".join(str(x) for x in last.get("findings") or []) or f"{active['operation']}已提交，业务状态：{outcome}", suggestions)
            if active["goal_fingerprint"] != _stable_hash(_goal(state).model_dump(mode="json")):
                return self._finish(state, role, "needs_help", "用户目标已更新，请主控重新委派。")
            if active["rounds"] >= settings.agent_subagent_max_rounds:
                return self._finish(state, role, "blocked", "子 Agent 决策轮数已达上限。")
            if active["no_progress"] >= settings.agent_subagent_no_progress_limit or active["cycled"]:
                return self._finish(state, role, "blocked", "子 Agent 无实质进展或状态循环，请主控调整任务。")
            available = [name for name in allowed_actions(state, set(handlers)) if name in names and name in handlers]
            required = active["operation"]
            available = [name for name in available if name == required]
            # WHY: 一次委派只授权一个研究动作；完成后必须汇报，后续计划由 Main 重新决定。
            if required in active["completed_operations"]:
                available = []
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
            direct = settings.agent_deterministic_subagent_execute and "delivery_mode" in state
            ledger["subagent_dispatch_rounds"] = int(ledger.get("subagent_dispatch_rounds") or 0) + 1
            if not direct:
                ledger["subagent_decision_rounds"] = int(ledger.get("subagent_decision_rounds") or 0) + 1
            # WHY: 恢复后继续原委派的轮数与已提交观察，不重置局部预算或重放已完成动作。
            _checkpoint(state)
            from app.services.execution_report_capture import record_delegation
            record_delegation(state,role,active)
            try:
                if direct:
                    if required not in available:
                        return self._finish(state,role,"needs_help","本次授权动作在当前状态不可执行，请Main重新规划。")
                    from app.schemas.agent_decision_schema import AgentDecision
                    chosen=AgentDecision(action=required,arguments=copy.deepcopy(active["arguments"]),reason=active["objective"])
                    trajectory.proposed(chosen)
                else:
                    chosen = policy.decide(context, allowed_actions=available, trajectory=trajectory)
            except (AgentDecisionError, LLMInvocationError) as exc:
                from app.agent.execution_budget import is_control_exception
                if is_control_exception(exc):
                    raise
                last = active.get("last_result") or {}
                if required in active["completed_operations"] and last.get("status") == "completed":
                    # WHY: 操作已原子提交，汇报失败不能重放或抹掉结果；显式needs_help交回Main处理。
                    trajectory.observe({"status": "rejected", "error": "subagent_report_protocol_failed",
                                        "detail": str(exc)[:240]})
                    return self._finish(state, role, "needs_help",
                        f"{required}已提交，但汇报协议未通过有界纠错（{str(exc)[:200]}）。请主控依据结构化结果和当前门禁决定下一步；不要重复本次工具。")
                if isinstance(exc, LLMInvocationError):
                    # WHY: 尚未执行的模型通信失败交回Main选择重试；余额/预算/失权已在上方透传。
                    active["last_result"] = {"status": "failed", "outcome": "blocked", "operation": required,
                        "error": str(exc)[:500], "issues": [{"code": "subagent_decision_failed", "reason": str(exc)[:500]}]}
                    trajectory.observe(active["last_result"])
                    return self._finish(state, role, "needs_help", f"{required}尚未执行，模型决策失败，请Main决定是否重试。")
                raise
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
            try:
                result = self.controller.execute(
                    role=spec.role, operation=chosen.action, objective=active["objective"],
                    state=state, handler=lambda working: handlers[chosen.action](working, chosen.arguments),
                    constraints=chosen.arguments, expected_output=list(spec.output_fields),
                    input_artifact_refs=(chosen.evidence_refs or active["evidence_refs"]
                                         if chosen.action == required else chosen.evidence_refs),
                    should_cancel=should_cancel,
                    consume_mandatory_action=bool(active["consume_mandatory_action"]
                        and chosen.action == required and required not in active["completed_operations"]),
                    invalidate_evidence=True,
                )
            except Exception as exc:
                from app.agent.execution_budget import is_control_exception
                if is_control_exception(exc):
                    raise
                raw = getattr(exc, "agent_task_result", None) or {
                    "status": "failed", "outcome": "blocked", "operation": chosen.action,
                    "error": str(exc)[:500], "issues": [{"code": "execution_failed", "reason": str(exc)[:500]}],
                }
                active["last_result"] = raw
                trajectory.observe(raw)
                return self._finish(state, role, "needs_help", f"{chosen.action}执行失败：{type(exc).__name__}；请Main选择恢复动作。")
            # 无持久化 runtime 或命中幂等结果时仍记录；已原子记录的观察不会重复。
            record_subagent_observation(state, result.model_dump(mode="json"))
            if result.status != AgentTaskStatus.COMPLETED:
                return self._finish(state, role, "blocked", result.error or "专业动作未成功提交。")

    @staticmethod
    def _context(state, role, names, available):
        return subagent_context(state, role, _session(state, role)["active"], available)

    @staticmethod
    def _finish(state, role, status, message, suggestions=()):
        session = _session(state, role)
        active = session["active"]
        _message(state, role, status, message, suggestions)
        last = active.get("last_result") or {}
        executed = active['operation'] in active.get('completed_operations', []) and last.get('status') == 'completed'
        if not executed:
            last = {**last, 'issues': [*last.get('issues', []), {'code': 'delegated_action_not_executed',
                'reason': '本次委派没有完成工具观察；历史执行不能替代本次执行', 'recoverable': True}]}
            message = '本次尚未执行' + active['operation'] + '。子Agent报告（未验证）：' + message
        typed_suggestions = copy.deepcopy(last.get("suggested_actions") or [])
        for action in suggestions:
            if action not in {item["action"] for item in typed_suggestions}:
                typed_suggestions.append({"action": action, "reason": message[:300]})
        result = AgentTaskResult(
            task_id=active["id"], role=AgentRole(role), operation=active["operation"],
            status=AgentTaskStatus.COMPLETED,
            outcome=(AgentTaskOutcome.BLOCKED if status == "blocked" else
                     AgentTaskOutcome(last.get("outcome", "unknown")) if last else AgentTaskOutcome.UNKNOWN),
            delegation_status=status,
            findings=[message],
            proposed_decisions=copy.deepcopy(last.get("proposed_decisions") or []),
            issues=copy.deepcopy(last.get("issues") or []),
            suggested_actions=typed_suggestions,
            check_report_ref=last.get("check_report_ref"),
            output_artifact_refs=list(last.get("output_artifact_refs") or []),
            error=last.get("error"),
            evidence_refs=list(last.get("evidence_refs") or []),
            changed_state_fields=list(last.get("changed_state_fields") or []),
            usage=dict(last.get("usage") or {}),
            open_questions=[] if status == "completed" else [{
                "category": status, "question": message, "suggested_actions": list(suggestions),
            }], input_state_fingerprint=active["source_fingerprint"],
            output_state_fingerprint=context_source_fingerprint(state),
        )
        ReActContext(state, namespace=role).observe({"status": status, "findings": [message]})
        ReActContext(state).observe(result.model_dump(mode="json"))
        from app.agent.delegation_memory import complete_work_context
        complete_work_context(state, role, {**result.model_dump(mode='json'),
            'findings': copy.deepcopy(last.get('findings') or ['本次没有已提交工具观察']),
            'task_executed': executed, 'model_report': message[:300]})
        session["last_delegation"] = copy.deepcopy(active)
        session.pop("active")
        state.pop("active_subagent")
        _checkpoint(state)
        from app.services.execution_report_capture import record_delegation_result
        record_delegation_result(state,role,active,result)
        return result
