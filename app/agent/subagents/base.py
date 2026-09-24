"""专业 Agent 的公共权限、版本和结果边界。"""

from __future__ import annotations

import copy
import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from app.agent.context_builder import context_source_fingerprint
from app.agent.task_context import build_task_state
from app.agent.tool_registry import operation_allowed
from app.schemas.agent_task_schema import (
    AgentRole,
    AgentTask,
    AgentTaskOutcome,
    AgentTaskResult,
    AgentTaskStatus,
)


class AgentBoundaryViolation(RuntimeError):
    """专业 Agent 试图越过目标、证据或写作权限边界。"""


_GOAL_FIELDS = {
    "user_query", "research_request", "required_reference_count",
    "year_range_explicit", "max_papers_explicit", "start_year", "end_year",
    "selected_scope", "core_deliverables",
    "session_id", "agent_orchestration_mode", "user_clarifications",
}


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:20]


def _changed_fields(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    return sorted(
        key for key in set(before) | set(after)
        if key not in before or key not in after
        or _fingerprint(before.get(key)) != _fingerprint(after.get(key))
    )


class BaseSubAgent(ABC):
    role: AgentRole
    protected_fields: set[str] = set(_GOAL_FIELDS)

    def execute(
        self,
        task: AgentTask,
        state: dict[str, Any],
        handler: Callable[[dict[str, Any]], Any],
    ) -> AgentTaskResult:
        if task.role != self.role:
            raise AgentBoundaryViolation(f"task role {task.role} cannot run on {self.role}")
        if not operation_allowed(self.role.value, task.operation):
            raise AgentBoundaryViolation(
                f"operation {task.operation!r} is not allowed for {self.role.value} agent"
            )
        current = context_source_fingerprint(state)
        if current != task.source_state_fingerprint:
            return AgentTaskResult(
                task_id=task.task_id,
                role=self.role,
                operation=task.operation,
                status=AgentTaskStatus.STALE,
                input_state_fingerprint=task.source_state_fingerprint,
                output_state_fingerprint=current,
                error="source_state_changed_before_execution",
            )

        working_state = build_task_state(state, self.role, task.operation)
        before = copy.deepcopy(working_state)
        try:
            handler(working_state)
        except Exception as exc:
            working_state.pop("active_agent_role", None)
            before.pop("active_agent_role", None)
            changed = _changed_fields(before, working_state)
            status = (
                AgentTaskStatus.CANCELLED
                if exc.__class__.__name__ == "AgentCancelledError"
                else AgentTaskStatus.FAILED
            )
            result = self._result(task, working_state, status, changed, str(exc))
            setattr(exc, "agent_task_result", result.model_dump(mode="json"))
            raise

        working_state.pop("active_agent_role", None)
        before.pop("active_agent_role", None)
        from app.agent.action_contracts import DERIVED, validate_patch
        changed = _changed_fields(before, working_state)
        changed = [key for key in changed if key not in DERIVED and key != "agent_execution_budget"]
        violations = sorted(set(changed) & self.protected_fields)
        if violations:
            raise AgentBoundaryViolation(
                f"{self.role.value} agent modified protected fields: {', '.join(violations)}"
            )
        result = self._result(task, working_state, AgentTaskStatus.COMPLETED, changed)
        result.state_patch = {
            key: copy.deepcopy(working_state[key]) for key in changed if key in working_state
        }
        result.removed_state_fields = [key for key in changed if key not in working_state]
        try:
            validate_patch(task.operation, result.state_patch, result.removed_state_fields)
        except ValueError as exc:
            raise AgentBoundaryViolation(str(exc)) from exc
        return result

    def _result(
        self,
        task: AgentTask,
        state: dict[str, Any],
        status: AgentTaskStatus,
        changed: list[str],
        error: str | None = None,
    ) -> AgentTaskResult:
        findings, evidence_refs, questions = self.summarize(state)
        return AgentTaskResult(
            task_id=task.task_id,
            role=self.role,
            operation=task.operation,
            status=status,
            outcome=self._outcome(state, status, changed),
            findings=findings,
            evidence_refs=evidence_refs,
            open_questions=questions,
            changed_state_fields=changed,
            input_state_fingerprint=task.source_state_fingerprint,
            output_state_fingerprint=context_source_fingerprint(state),
            idempotency_key=task.idempotency_key,
            error=error,
        )

    def _outcome(
        self,
        state: dict[str, Any],
        status: AgentTaskStatus,
        changed: list[str],
    ) -> AgentTaskOutcome:
        if status != AgentTaskStatus.COMPLETED:
            return AgentTaskOutcome.UNKNOWN
        if state.get("generation_blocked"):
            return AgentTaskOutcome.BLOCKED
        if state.get("search_failed") and not state.get("candidate_papers"):
            return AgentTaskOutcome.BLOCKED
        if not changed:
            return AgentTaskOutcome.NO_CHANGE
        if (state.get("evidence_gap_report") or {}).get("needs_recovery"):
            return AgentTaskOutcome.DEGRADED
        return AgentTaskOutcome.SUCCEEDED

    @abstractmethod
    def summarize(
        self, state: dict[str, Any]
    ) -> tuple[list[str], list[str], list[dict[str, Any]]]:
        raise NotImplementedError
