"""按动作契约投影专业任务输入，不把新状态字段自动暴露给专业任务。"""
import copy
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
