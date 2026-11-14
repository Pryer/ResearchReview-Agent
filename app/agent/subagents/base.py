"""专业 Agent 的公共权限、版本和结果边界。"""

from __future__ import annotations

import copy
import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from app.agent.context_builder import context_source_fingerprint
from app.agent.task_context import build_task_state, execution_task_brief
from app.agent.quality_feedback import quality_feedback
from app.schemas.check_schema import ResultIssue, SuggestedAction
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
    "selected_scope", "core_deliverables", "target_length",
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
        from app.agent.execution_budget import AgentFinalizationRequired
        partial_reason = ""
        try:
            active = ((state.get("subagent_sessions") or {}).get(self.role.value) or {}).get("active") or {}
            packet = active.get("context_packet") if active.get("operation") == task.operation else None
            if active.get('operation') == task.operation:
                packet = {**(packet or {}), 'work_context': copy.deepcopy(active.get('work_context') or {})}
            with execution_task_brief(task, packet):
                handler(working_state)
        except AgentFinalizationRequired as exc:
            # WHY: 软收尾只停止后续远程工作；已经完成的阶段结果仍须经过同一权限和CAS校验提交。
            partial_reason = str(exc)
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
            from app.agent.execution_budget import is_control_exception
            if self.role == AgentRole.WRITING and not is_control_exception(exc):
                old_ids = {r['candidate_id'] for r in before.get('writer_candidates') or []}
                # WHY: 失败时只返回追加审计，不返回失败工具的研究状态补丁或公开稿指针。
                setattr(exc, 'writer_audit_records', [copy.deepcopy(r) for r in working_state.get('writer_candidates') or []
                                                     if r['candidate_id'] not in old_ids])
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
        if partial_reason:
            result.outcome = AgentTaskOutcome.DEGRADED
            result.issues.append(ResultIssue(code="budget_low", severity="warning",
                reason=partial_reason[:500], blocks_draft=False, blocks_final=False, issue_type="budget"))
            result.findings.append("已提交完成的阶段结果，保留收尾预算交回Main规划。")
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
        feedback = quality_feedback(state) if self.role in {AgentRole.ANALYSIS, AgentRole.WRITING, AgentRole.CHECK} else {}
        check = (state.get("check_report") or {}) if self.role == AgentRole.CHECK else {}
        issues = [ResultIssue.model_validate(i) for i in check.get("issues") or []]
        if error:
            issues.append(ResultIssue(code="execution_failed", reason=error[:500]))
        if not check:
            issues.extend(ResultIssue(code=str(q.get("category") or "task_gap"),
                severity="blocking" if q.get("blocking") else "warning",
                reason=str(q.get("question") or "任务存在缺口")) for q in questions)
        suggestions = ([SuggestedAction.model_validate(i) for i in check.get("suggested_actions") or []]
                       if check else [SuggestedAction(action=a) for a in feedback.get("suggested_actions") or []])
        return AgentTaskResult(
            task_id=task.task_id,
            role=self.role,
            operation=task.operation,
            status=status,
            outcome=self._outcome(state, status, changed),
            findings=findings,
            proposed_decisions=([{"kind": "quality_feedback", "feedback": feedback}] if feedback else []),
            evidence_refs=evidence_refs,
            open_questions=questions,
            changed_state_fields=changed,
            input_state_fingerprint=task.source_state_fingerprint,
            output_state_fingerprint=context_source_fingerprint(state),
            idempotency_key=task.idempotency_key,
            error=error,
            issues=issues, suggested_actions=suggestions,
            check_report_ref=(f"state://check-report/{check['report_id']}" if check.get("report_id") else None),
        )

    def _outcome(
        self,
        state: dict[str, Any],
        status: AgentTaskStatus,
        changed: list[str],
    ) -> AgentTaskOutcome:
        if status != AgentTaskStatus.COMPLETED:
            return AgentTaskOutcome.UNKNOWN
        if self.role == AgentRole.WRITING and state.get("generation_blocked"):
            return AgentTaskOutcome.BLOCKED
        if self.role == AgentRole.SEARCH and state.get("search_failed") and not state.get("candidate_papers"):
            return AgentTaskOutcome.BLOCKED
        if not changed:
            return AgentTaskOutcome.NO_CHANGE
        if self.role == AgentRole.ANALYSIS and (state.get("evidence_gap_report") or {}).get("needs_recovery"):
            return AgentTaskOutcome.DEGRADED
        return AgentTaskOutcome.SUCCEEDED

    @abstractmethod
    def summarize(
        self, state: dict[str, Any]
    ) -> tuple[list[str], list[str], list[dict[str, Any]]]:
        raise NotImplementedError
