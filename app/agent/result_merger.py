"""专业任务结果的版本复核与受限状态合并。"""

from __future__ import annotations

import copy
from typing import Any

from app.agent.context_builder import context_source_fingerprint, source_state_version
from app.schemas.agent_task_schema import AgentTask, AgentTaskResult, AgentTaskStatus


class StaleAgentResult(RuntimeError):
    """专业任务完成时，权威输入已改变。"""


def merge_task_result(
    state: dict[str, Any], task: AgentTask, result: AgentTaskResult
) -> None:
    from app.agent.action_contracts import validate_patch
    if (result.task_id, result.role, result.operation) != (task.task_id, task.role, task.operation):
        raise ValueError("task result identity mismatch")
    if result.input_state_fingerprint != task.source_state_fingerprint:
        raise ValueError("task result input fingerprint mismatch")
    if result.status != AgentTaskStatus.COMPLETED:
        raise StaleAgentResult("task did not complete")
    validate_patch(task.operation, result.state_patch, result.removed_state_fields)
    current = context_source_fingerprint(state)
    if (
        current != task.source_state_fingerprint
        or source_state_version(state) != task.source_state_version
    ):
        result.status = AgentTaskStatus.STALE
        result.error = "source_state_changed_before_commit"
        result.output_state_fingerprint = current
        raise StaleAgentResult(result.error)
    for key in result.removed_state_fields:
        state.pop(key, None)
    for key, value in result.state_patch.items():
        state[key] = copy.deepcopy(value)
    state["state_revision"] = int(state.get("state_revision") or 0) + 1
