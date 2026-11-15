"""Main 选择补充上下文；执行权限和必需输入始终由动作契约决定。"""
from __future__ import annotations

import copy

from app.agent.action_contracts import CONTRACTS
from app.agent.action_registry import ACTION_REGISTRY
from app.agent.context_builder import build_main_agent_context, _goal, _gate_status, _budget_remaining, _failure_reasons
from app.schemas.agent_decision_schema import ContextSelection
from app.schemas.context_schema import KeyEvidence, DecisionRecord, OpenQuestion


def selectable_context_ids(context):
    """只返回当前可见摘要的ID；论文身份和资料引用不在此命名空间。"""
    return {
        "evidence_ids": [item.evidence_id for item in context.key_evidence],
        "decision_ids": [item.decision_id for item in context.decisions],
        "question_ids": [item.question_id for item in context.open_questions],
        "memory_ids": [item["memory_id"] for item in context.state.research_memory],
    }


class ContextSelectionError(ValueError):
    def __init__(self, invalid_ids, allowed_ids):
        # WHY: 公开错误保持简短；具体ID只用于模型纠错和本地脱敏诊断。
        super().__init__("selected context ID is outside the current status bar")
        self.invalid_ids = invalid_ids
        self.allowed_ids = allowed_ids

    def repair_context(self):
        return {
            "code": "invalid_context_selection",
            "invalid_ids": self.invalid_ids,
            "allowed_ids": self.allowed_ids,
            "repair": "只从本轮allowed_ids对应字段选值；空字段必须传[]。paper_id、DOI和artifact引用不是摘要ID。必需论文资料由动作契约提供，无补充摘要时可用空选择。",
        }


def validate_planning(decision, context):
    for step in decision.plan:
        if step.action not in ACTION_REGISTRY:
            raise ValueError("plan must contain registered actions")
    selection = decision.context_selection
    if selection is None:
        return
    spec = ACTION_REGISTRY.get(decision.action)
    if not spec or spec.role is None:
        raise ValueError("context_selection requires a delegated action")
    allowed = selectable_context_ids(context)
    invalid = {name: sorted(set(requested) - set(allowed[name]))
               for name, requested in selection.model_dump().items()
               if not set(requested).issubset(allowed[name])}
    if invalid:
        raise ContextSelectionError(invalid, allowed)


def context_packet(state, decision, *, decision_context=None):
    # WHY: 新plan入账会替换展示的计划ID；选择应按决策时已见快照校验，不在同一委派内换视图。
    context = decision_context if decision_context is not None else build_main_agent_context(state)
    if context.goal != _goal(state):
        raise ValueError("delegation goal changed after context selection")
    validate_planning(decision, context)
    if decision.context_selection is None:
        return None
    selected = decision.context_selection
    return {
        "selection": selected.model_dump(mode="json"),
        "key_evidence": [item.model_dump(mode="json") for item in context.key_evidence
                         if item.evidence_id in selected.evidence_ids],
        "decisions": [item.model_dump(mode="json") for item in context.decisions
                      if item.decision_id in selected.decision_ids],
        "open_questions": [item.model_dump(mode="json") for item in context.open_questions
                           if item.question_id in selected.question_ids],
        "research_memory": [copy.deepcopy(item) for item in context.state.research_memory
                            if item["memory_id"] in selected.memory_ids],
    }


def subagent_context(state, role, active, available):
    fields = CONTRACTS[active["operation"]].inputs
    projected = {key: copy.deepcopy(value) for key, value in state.items() if key in fields}
    projected["artifact_manifest"] = {
        key: copy.deepcopy(value) for key, value in (state.get("artifact_manifest") or {}).items()
        if key in fields
    }
    session = state["subagent_sessions"][role]
    # WHY: 当前完成态只来自本次委派的观察；同角色旧操作放在明确标记的work_context历史中。
    projected['agent_task_results'] = [copy.deepcopy(active['last_result'])] if active.get('last_result') else []
    for key in ("agent_execution_budget",):
        projected[key] = copy.deepcopy(state.get(key) or {})
    for key in ("agent_context_usage", "agent_context_last_usage"):
        projected[key] = copy.deepcopy(session.get(key) or {})
    context = build_main_agent_context(projected)
    # WHY: 模型只能裁剪补充材料，不能省略硬约束、最新门禁、失败信息或剩余预算。
    context.goal = _goal(state)
    context.state.gate_status = _gate_status(state)
    context.state.failure_reasons = _failure_reasons(state)
    context.state.budget_remaining = _budget_remaining(state)
    context.state.remaining_action_budget = context.state.budget_remaining['recovery']
    packet = active.get("context_packet")
    if packet is not None:
        # 历史摘要只作任务说明；最新状态与当前操作产物仍由代码构造，不能当作新授权。
        selection = ContextSelection.model_validate(packet["selection"])
        # WHY: 新包已有决策时摘要；仅旧检查点缺少材料时才补建完整权威视图。
        authoritative = (build_main_agent_context(state) if any(key not in packet for key in
            ("key_evidence", "decisions", "open_questions")) else None)
        # WHY: 摘要是本次委派选中的历史材料，不授予当前证据权限；旧检查点无快照时保持原投影。
        context.key_evidence = ([KeyEvidence.model_validate(item) for item in packet["key_evidence"]]
                                if "key_evidence" in packet else
                                [item for item in authoritative.key_evidence if item.evidence_id in selection.evidence_ids])
        context.decisions = ([DecisionRecord.model_validate(item) for item in packet["decisions"]]
                             if "decisions" in packet else
                             [item for item in authoritative.decisions if item.decision_id in selection.decision_ids])
        context.open_questions = ([OpenQuestion.model_validate(item) for item in packet["open_questions"]]
                                  if "open_questions" in packet else
                                  [item for item in authoritative.open_questions if item.question_id in selection.question_ids])
        context.state.research_memory = copy.deepcopy(packet["research_memory"])
    context.state.allowed_actions = available
    context.state.next_actions = available
    context.state.delegation = {key: copy.deepcopy(active[key]) for key in (
        "id", "operation", "objective", "arguments", "evidence_refs", "completed_operations", "rounds",
    )}
    context.state.delegation["context_selection"] = copy.deepcopy((packet or {}).get("selection"))
    context.state.delegation["input_contract"] = active["operation"]
    context.state.delegation["expected_output"] = list(ACTION_REGISTRY[active["operation"]].output_fields)
    context.state.delegation['work_context'] = copy.deepcopy(active.get('work_context') or {})
    executed = active['operation'] in active.get('completed_operations', [])
    context.state.delegation['execution_status'] = 'observed' if executed else 'not_started'
    context.state.delegation['must_execute_before_completed_report'] = not executed
    # WHY: Builder会从旧同角色任务推断完成态；这里必须覆盖为本次实际结果。
    context.state.completed_actions = list(active.get('completed_operations') or [])
    context.state.last_task_result = copy.deepcopy(active.get('last_result') or {})
    context.state.agent_messages = [copy.deepcopy(m) for m in session.get('messages', [])
                                    if m.get('delegation_id') == active['id']][-6:]
    return context
