"""主 Agent 的确定性任务委派控制器。"""

from __future__ import annotations

import hashlib
import copy
from contextlib import nullcontext
import json
import time
from collections.abc import Callable
from typing import Any

from app.agent.context_builder import (
    context_source_fingerprint,
    refresh_main_agent_context,
    source_state_version,
)
from app.agent.execution_budget import (
    AgentBudgetExceeded,
    AgentExecutionCancelled,
    AgentExecutionStale,
    commit_action,
    reserve_action,
    check_execution,
    single_action_charge,
)
from app.agent.result_merger import StaleAgentResult, merge_task_result
from app.agent.subagents import AnalysisAgent, SearchAgent, WritingAgent, CheckAgent
from app.schemas.agent_task_schema import AgentRole, AgentTask, AgentTaskResult
from app.core.config import get_settings


_AGENTS = {
    AgentRole.SEARCH: SearchAgent,
    AgentRole.ANALYSIS: AnalysisAgent,
    AgentRole.WRITING: WritingAgent,
    AgentRole.CHECK: CheckAgent,
}

_PLAN_OPERATIONS = {
    "search_and_rank": {"search", "deduplicate", "screen"},
    "extract_paper_cards": {"extract_paper_cards"},
    "fetch_metadata": {"metadata_verification"},
    "validate_routes": {"classify_papers"},
    "cluster_papers": {"classify_papers"},
    "plan_claims": {"synthesize"},
    "generate_deliverables": {"write"},
}


def _record_result(state, raw, *, durable=False):
    history = state.setdefault("agent_task_results", [])
    history.append(raw)
    if durable:
        # WHY: 完整结果在research_tasks按幂等键保存；上下文只需近期结果，纯内存模式保留旧行为。
        del history[:-16]


def _update_plan_status(state: dict[str, Any], operation: str, status: str) -> None:
    plan = state.get("research_plan")
    if not isinstance(plan, dict):
        return
    expected = _PLAN_OPERATIONS.get(operation, {operation})
    for node in plan.get("task_graph") or []:
        if str(node.get("operation") or "") in expected:
            node["status"] = status


class AgentController:
    """验证版本和权限后执行专业任务，并把精简结果并回主状态。"""

    def publish_draft(self, state, *, expected_version=None):
        from app.agent.delivery import publish_draft
        from app.services.durable_execution_service import active_runtime
        check_execution()
        if expected_version is not None and source_state_version(state) != expected_version:
            raise ValueError("publication state version changed")
        candidate = copy.deepcopy(state)
        record = publish_draft(candidate)
        pending = (candidate.get('main_agent_trajectory') or {}).get('pending') or {}
        paired = pending.get('action') == 'publish_draft'
        if paired:
            from app.agent.react_context import ReActContext
            unchanged = state.get('published_draft_version') == record['version']
            # WHY: 控制观察仅内存配对，随发布快照持久化；CAS前不另写“已发布”归档。
            ReActContext(candidate).observe({'status': 'no_progress' if unchanged else 'completed',
                'reason': 'already_published' if unchanged else 'published', 'version': record['version'],
                'grade': record['grade'], 'delivery_status': record['grade']}, archive=False)
        check_execution()
        runtime = active_runtime()
        if runtime:
            runtime.snapshot(candidate, ledger=state.get("agent_execution_budget") or {})
            state["_runtime_version"]=candidate["_runtime_version"]
        state.update({key: candidate[key] for key in ("published_versions", "delivery_status",
            "published_artifact_ref", "published_draft_version", "verification_status",
            "requirements_satisfied", "delivery_metrics") if key in candidate})
        if paired:
            state['main_agent_trajectory'] = candidate['main_agent_trajectory']
        return record

    @single_action_charge
    def execute(
        self,
        *,
        role: AgentRole,
        operation: str,
        objective: str,
        state: dict[str, Any],
        handler: Callable[[dict[str, Any]], Any],
        constraints: dict[str, Any] | None = None,
        expected_output: list[str] | None = None,
        input_artifact_refs: list[str] | None = None,
        budget: dict[str, int] | None = None,
        should_cancel: Callable[[], bool] | None = None,
        consume_mandatory_action: bool = False,
        invalidate_evidence: bool = False,
        paper_scope: tuple[str, ...] = (),
    ) -> AgentTaskResult:
        # WHY: 直接调用同样受候选版本协议约束，不能仅依靠给LLM的动作白名单。
        if state.get("intent") != "search_papers" and state.get("pending_result_validation") and operation != "validate_result":
            raise ValueError("candidate is awaiting independent validation before further research actions")
        if operation in {"rewrite_sections", "generate_deliverables"}:
            if state.get("pending_evidence_recovery"):
                raise ValueError("new search results require Analysis authorization before writing")
            if state.get("pending_result_validation"):
                raise ValueError("candidate is awaiting independent validation")
            from app.agent.recovery.generation_recovery import rewrite_attempt_exhausted

            # WHY: 直接调用 Controller 同样受恢复限制，在创建任务、预留预算和
            # 幂等缓存复用前拦截；不能仅靠可用动作列表约束模型决策。
            if rewrite_attempt_exhausted(state):
                raise ValueError("rewrite attempts exhausted for current research input")
        if consume_mandatory_action:
            pending = list(state.get("agent_mandatory_actions") or [])
            if not pending or pending[0].get("action") != operation:
                raise ValueError("mandatory action does not match current task")
        _update_plan_status(state, operation, "running")
        snapshot = refresh_main_agent_context(state)
        if snapshot.validation_errors:
            raise ValueError(
                "main agent context failed validation: "
                + ", ".join(snapshot.validation_errors)
            )
        effective_budget = budget or {}
        from app.services.research_execution_service import artifact_store
        store = artifact_store()
        from app.schemas.artifact_schema import ARTIFACT_TYPES, STATE_ARTIFACT_FIELDS
        from app.agent.action_contracts import CONTRACTS
        allowed_fields = CONTRACTS[operation].inputs
        allowed_types = {kind for field, kind in STATE_ARTIFACT_FIELDS.items() if field in allowed_fields}
        if "parsed_papers" in allowed_fields:
            allowed_types.add("document_fragment")
        readable = {kind for kind, spec in ARTIFACT_TYPES.items()
                    if role.value in spec.roles and kind in allowed_types}
        if store is not None and not input_artifact_refs:
            input_artifact_refs = [item["ref"]
                for manifest in (state.get("artifact_manifest") or {}).values()
                for item in manifest["items"] if item["type"] in readable
                and (not paper_scope or item.get('paper_id') in paper_scope)]
        for ref in input_artifact_refs or []:
            if ref.startswith("state://"):
                prefix_fields = {"state://paper-card/": "paper_cards", "state://paper/": "paper_details",
                                 "state://writing-plan/": "writing_plans", "state://check-report/": "check_report", "state://writer-candidate/": "writer_candidates"}
                field = next((field for prefix, field in prefix_fields.items() if ref.startswith(prefix)), None)
                if field not in allowed_fields:
                    raise ValueError("artifact reference is not allowed by action input contract")
            if store is None:
                if not ref.startswith("state://"):
                    raise ValueError("artifact store is required for persisted references")
                from app.services.research_artifact_service import ResearchArtifactService
                if ResearchArtifactService._resolve_legacy_state_ref(ref, state) is None:
                    raise ValueError("task artifact does not exist")
            else:
                entries = [item for manifest in (state.get("artifact_manifest") or {}).values()
                           for item in manifest["items"] if item["ref"] == ref]
                if ref.startswith("artifact://") and not entries:
                    raise ValueError("task reference is not in the current artifact manifest")
                store.resolve(str(state.get("session_id") or ""), ref, state=state,
                              allowed_types=readable, role=role.value,
                              expected_version=entries[0]["version"] if entries else 1)
        deadline_ms = int(effective_budget.get("deadline_ms") or 0)
        deadline = time.monotonic() + deadline_ms / 1000 if deadline_ms > 0 else None
        source_fingerprint = context_source_fingerprint(state)
        from app.services.invocation_scope import current_invocation
        invocation = current_invocation()
        if paper_scope and invocation is None:
            raise ValueError('paper subset requires a server-signed v2 invocation')
        idempotency_key = hashlib.sha256(json.dumps({
            **({"invocation_attempt": invocation[1]} if invocation else {}),
            "session": state.get("session_id") or "",
            "operation_sequence": state.get("user_operation_sequence") or 0,
            "role": role.value,
            "operation": operation,
            "objective": objective,
            "constraints": constraints or {},
            "source": source_fingerprint,
        }, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:32]
        from app.services.durable_execution_service import active_runtime, active_task_id, task_scope
        # WHY: 有界复合动作的内层 Controller 不另起数据库提交；由外层一次提交。
        runtime = active_runtime() if not active_task_id() else None
        stored_result = runtime.completed_task(idempotency_key) if runtime else None
        for raw in ([stored_result] if stored_result else reversed(state.get("agent_task_results") or [])):
            if raw.get("idempotency_key") == idempotency_key and raw.get("status") == "completed":
                check_execution()
                if should_cancel and should_cancel():
                    raise AgentExecutionCancelled("cached task cancelled")
                cached = AgentTaskResult.model_validate(raw)
                state["agent_execution_status"] = "completed"
                _update_plan_status(state, operation, "blocked" if cached.outcome.value == "blocked" else "completed")
                if consume_mandatory_action:
                    _consume_mandatory_action(state)
                    from app.services.durable_execution_service import active_runtime
                    runtime = active_runtime()
                    if runtime:
                        runtime.snapshot(state, ledger=state.get("agent_execution_budget") or {})
                refresh_main_agent_context(state)
                return cached
        reserve_action(
            state,
            limit=int(
                effective_budget["action_limit"]
                if "action_limit" in effective_budget
                else get_settings().agent_execution_action_budget
            ),
            should_cancel=should_cancel,
            deadline_monotonic=deadline,
        )
        task = AgentTask(
            role=role,
            operation=operation,
            objective=objective,
            constraints={**(constraints or {}),
                'explicit_constraints': snapshot.context.goal.explicit_constraints,
                'target_length': state.get('target_length'), 'start_year': state.get('start_year'),
                'end_year': state.get('end_year'), 'minimum_references': state.get('required_reference_count'),
                'citation_protocol': 'paper_id-v1', 'acceptance_criteria': expected_output or []},
            input_artifact_refs=input_artifact_refs or [],
            paper_scope=list(paper_scope),
            expected_output=expected_output or [],
            budget=effective_budget,
            source_state_version=source_state_version(state),
            source_state_fingerprint=source_fingerprint,
            idempotency_key=idempotency_key,
        )
        agent = _AGENTS[role]()
        from app.agent.evidence_invalidation import evidence_snapshot, invalidate_changed_evidence
        evidence_before = evidence_snapshot(state, operation) if invalidate_evidence else None
        from app.database.runtime_repository import RuntimeConflict
        if runtime:
            runtime.begin_task(task, state.get("agent_execution_budget") or {}, resource_state=state)
        try:
            state["agent_execution_status"] = "running"
            with task_scope(task.task_id) if runtime else nullcontext():
                from app.agent.execution_budget import research_stage
                with research_stage(state, operation):
                    result = agent.execute(task, state, handler)
            check_execution()
            if deadline is not None and time.monotonic() >= deadline:
                raise AgentBudgetExceeded("task deadline exceeded before commit")
            if should_cancel and should_cancel():
                raise AgentExecutionCancelled("agent task cancelled before commit")
            candidate = copy.deepcopy(state) if runtime else state
            merge_task_result(candidate, task, result)
            if operation in {"generate_deliverables", "rewrite_sections"}:
                candidate.pop("check_report", None)
                candidate.pop("autonomous_verified_fingerprint", None)
            if operation == "validate_result":
                # WHY: 回滚与恢复结算由Controller提交，Check本身不能改写候选。
                from app.agent.checking import settle_checked_candidate
                settle_checked_candidate(candidate)
                from app.agent.delivery import review_published_delivery
                review_published_delivery(candidate)
            invalidate_changed_evidence(candidate, operation, evidence_before, paper_scope=paper_scope)
            commit_action(candidate)
            if consume_mandatory_action and result.status.value == "completed":
                # WHY: 已选动作与任务结果必须进入同一持久化快照；否则进程在
                # 提交后、主循环出队前崩溃，会把已付费动作再执行一次。
                _consume_mandatory_action(candidate)
            if runtime:
                _update_plan_status(candidate, operation,
                    "blocked" if result.outcome.value == "blocked" else "completed")
                candidate["agent_execution_status"] = "completed"
                result.output_state_fingerprint = context_source_fingerprint(candidate)
                _record_result(candidate, result.model_dump(mode="json"), durable=True)
                # WHY: 观察与研究提交进入同一检查点，恢复后不会留下未配对调用或重复执行。
                from app.agent.react_context import record_tool_observation
                record_tool_observation(candidate, result.model_dump(mode="json"))
                # WHY: 合并、失效清理与观察构造也消耗时间；正式提交前重新确认Action期限。
                check_execution()
                runtime.snapshot(candidate, task=task, result=result,
                                 ledger=candidate.get("agent_execution_budget") or {})
                ledger = state.get("agent_execution_budget")
                if ledger is not None:
                    ledger.update(candidate.get("agent_execution_budget") or {})
                    candidate["agent_execution_budget"] = ledger
                state.clear()
                state.update(candidate)
                refresh_main_agent_context(state)
                return result
        except StaleAgentResult:
            if runtime:
                runtime.fail_task(task.task_id, "stale", state.get("agent_execution_budget") or {},
                                  result=result.model_dump(mode="json"))
            _update_plan_status(state, operation, "invalidated")
            state["agent_execution_status"] = "stale"
            _record_result(state, result.model_dump(mode="json"), durable=bool(runtime))
            refresh_main_agent_context(state)
            return result
        except (AgentBudgetExceeded, AgentExecutionCancelled) as exc:
            # WHY: 租约失效复用取消透传，但任务账本状态记 stale（非用户取消）。
            if isinstance(exc, AgentExecutionStale):
                terminal = "stale"
            elif isinstance(exc, AgentBudgetExceeded):
                terminal = "blocked"
            else:
                terminal = "cancelled"
            if runtime:
                runtime.fail_task(task.task_id, terminal, state.get("agent_execution_budget") or {})
            _update_plan_status(state, operation, "failed")
            state["agent_execution_status"] = terminal
            refresh_main_agent_context(state)
            raise
        except Exception as exc:
            _update_plan_status(state, operation, "failed")
            state["agent_execution_status"] = "failed"
            from app.agent.execution_budget import is_control_exception
            if role == AgentRole.WRITING and not is_control_exception(exc):
                check_execution()
                audit = getattr(exc, 'writer_audit_records', [])
                existing = {r['candidate_id'] for r in state.get('writer_candidates') or []}
                state.setdefault('writer_candidates', []).extend(copy.deepcopy(r) for r in audit if r['candidate_id'] not in existing)
            raw = getattr(exc, "agent_task_result", None)
            if raw:
                raw["output_state_fingerprint"] = context_source_fingerprint(state)
                _record_result(state, raw, durable=bool(runtime))
            if runtime:
                if role == AgentRole.WRITING and not is_control_exception(exc) and audit:
                    # WHY: 父状态未合并失败工具补丁；仅失败审计立即落盘，恢复无需等待子循环收尾。
                    runtime.fail_task(task.task_id, "failed", state.get("agent_execution_budget") or {},
                                      audit_state=state)
                else:
                    runtime.fail_task(task.task_id, "stale" if isinstance(exc, RuntimeConflict) else "failed",
                                      state.get("agent_execution_budget") or {}, result=raw)
            refresh_main_agent_context(state)
            raise
        plan_status = {
            "completed": "completed",
            "stale": "invalidated",
            "cancelled": "failed",
            "failed": "failed",
        }.get(result.status.value, "pending")
        if result.outcome.value == "blocked":
            plan_status = "blocked"
        _update_plan_status(state, operation, plan_status)
        state["agent_execution_status"] = "completed"
        result.output_state_fingerprint = context_source_fingerprint(state)
        _record_result(state, result.model_dump(mode="json"))
        refresh_main_agent_context(state)
        return result


def _consume_mandatory_action(state: dict[str, Any]) -> None:
    pending = list(state.get("agent_mandatory_actions") or [])
    if len(pending) > 1:
        state["agent_mandatory_actions"] = pending[1:]
    else:
        state.pop("agent_mandatory_actions", None)
