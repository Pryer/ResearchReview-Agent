"""按动作契约投影专业任务输入，不把新状态字段自动暴露给专业任务。"""
import copy
from contextvars import ContextVar
from contextlib import contextmanager
from app.schemas.agent_task_schema import AgentRole
from app.agent.action_contracts import CONTRACTS, schema_for
from app.agent.tool_registry import operation_allowed


def build_task_state(state: dict, role: AgentRole, operation: str = "") -> dict:
    if not operation_allowed(role.value, operation):
        raise ValueError("role cannot execute action")
    names = CONTRACTS[operation].inputs
    projected = {key: copy.deepcopy(value) for key, value in state.items() if key in names}
    schema_for(operation).model_validate(projected)
    projected["active_agent_role"] = role.value
    # WHY: 预算通过执行上下文管理，禁止任务直接拿到共享可变账本。
    return projected



# WHY: Main的委派说明必须到达实际执行工具的模型，而不只到达子Agent选工具的一轮。

_EXECUTION_BRIEF = ContextVar("research_execution_brief", default=None)


@contextmanager
def execution_task_brief(task, packet=None):
    value = {"delegation_id": task.task_id, "role": task.role.value, "operation": task.operation,
             "objective": task.objective, "constraints": copy.deepcopy(task.constraints)}
    if packet:
        value["selected_context"] = copy.deepcopy(packet)
    token = _EXECUTION_BRIEF.set(value)
    try:
        yield
    finally:
        _EXECUTION_BRIEF.reset(token)


def current_task_brief(role):
    value = _EXECUTION_BRIEF.get()
    return copy.deepcopy(value) if value and value["role"] == role else None
