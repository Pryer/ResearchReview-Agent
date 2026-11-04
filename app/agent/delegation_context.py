"""Main 选择补充上下文；执行权限和必需输入始终由动作契约决定。"""
from __future__ import annotations

import copy

from app.agent.action_contracts import CONTRACTS
from app.agent.action_registry import ACTION_REGISTRY
from app.agent.context_builder import build_main_agent_context
from app.schemas.agent_decision_schema import ContextSelection


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
    groups = (
        (selection.evidence_ids, {item.evidence_id for item in context.key_evidence}),
        (selection.decision_ids, {item.decision_id for item in context.decisions}),
        (selection.question_ids, {item.question_id for item in context.open_questions}),
        (selection.memory_ids, {item["memory_id"] for item in context.state.research_memory}),
    )
    if any(not set(requested).issubset(known) for requested, known in groups):
        raise ValueError("selected context ID is outside the current status bar")


def context_packet(state, decision):
    # WHY: 旧检查点/无选择元数据使用操作级最小视图；空选择则明确不附加历史摘要。
    context = build_main_agent_context(state)
    validate_planning(decision, context)
    if decision.context_selection is None:
        return None
    selected = decision.context_selection
    return {
        "selection": selected.model_dump(mode="json"),
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
    projected["agent_task_results"] = [item for item in state.get("agent_task_results", [])
                                       if item.get("role") == role and item.get("operation") == active["operation"]]
    for key in ("agent_execution_budget",):
        projected[key] = copy.deepcopy(state.get(key) or {})
    for key in ("agent_context_usage", "agent_context_last_usage"):
        projected[key] = copy.deepcopy(session.get(key) or {})
    context = build_main_agent_context(projected)
    authoritative = build_main_agent_context(state)
    # WHY: 模型只能裁剪补充材料，不能省略硬约束、最新门禁、失败信息或剩余预算。
    context.goal = authoritative.goal
    context.state.gate_status = authoritative.state.gate_status
    context.state.failure_reasons = authoritative.state.failure_reasons
    context.state.budget_remaining = authoritative.state.budget_remaining
    packet = active.get("context_packet")
    if packet is not None:
        # 历史摘要只作任务说明；最新状态与当前操作产物仍由代码构造，不能当作新授权。
        selection = ContextSelection.model_validate(packet["selection"])
        context.key_evidence = [item for item in authoritative.key_evidence if item.evidence_id in selection.evidence_ids]
        context.decisions = [item for item in authoritative.decisions if item.decision_id in selection.decision_ids]
        context.open_questions = [item for item in authoritative.open_questions if item.question_id in selection.question_ids]
        context.state.research_memory = copy.deepcopy(packet["research_memory"])
    context.state.allowed_actions = available
    context.state.next_actions = available
    context.state.delegation = {key: copy.deepcopy(active[key]) for key in (
        "id", "operation", "objective", "arguments", "evidence_refs", "completed_operations", "rounds",
    )}
    context.state.delegation["context_selection"] = copy.deepcopy((packet or {}).get("selection"))
    context.state.delegation["input_contract"] = active["operation"]
    context.state.delegation["expected_output"] = list(ACTION_REGISTRY[active["operation"]].output_fields)
    context.state.agent_messages = copy.deepcopy(session.get("messages", [])[-6:])
    return context
