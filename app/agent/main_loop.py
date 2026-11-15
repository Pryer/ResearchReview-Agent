"""Stable Prefix + ReAct Trajectory + Agent Status Bar 驱动的主控委派循环。"""

from __future__ import annotations

import hashlib
import copy
import json
from contextvars import ContextVar
from contextlib import contextmanager
from collections import deque
from collections.abc import Callable
from typing import Any

from app.agent.action_registry import (
    action_spec, admitted_ranked_papers, allowed_actions, recovery_handoff_action,
    validate_arguments,
)
from app.agent.context_builder import refresh_main_agent_context
from app.agent.controller import AgentController
from app.agent.sidecar import ActionSidecar
from app.agent.execution_budget import (
    AgentBudgetExceeded,
    AgentExecutionStale,
    budget_scope,
    check_execution,
)
from app.agent.execution import AgentCancelledError
from app.core.exceptions import LLMInvocationError
from app.agent.main_policy import AgentDecisionError, MainAgentPolicy
from app.agent.react_context import ContextBudgetExceeded, ReActContext, execute_context_recall, observe_terminal
from app.schemas.agent_decision_schema import AgentDecision
from app.core.config import get_settings
from app.database.runtime_repository import RuntimeConflict


TERMINAL_STATUSES = {"completed", "partial", "blocked", "failed", "cancelled", "waiting_user"}

_PROGRESS_CACHE = ContextVar("progress_digest_cache", default=None)


@contextmanager
def progress_cache_scope():
    if _PROGRESS_CACHE.get() is not None:
        yield
        return
    token = _PROGRESS_CACHE.set({})
    try:
        yield
    finally:
        _PROGRESS_CACHE.reset(token)


def _progress_fingerprint(state: dict[str, Any]) -> str:
    from app.agent.evidence_invalidation import _detail_signature
    def stable_items(name: str) -> list[str]:
        value = state.get(name) or []
        cache = _PROGRESS_CACHE.get()
        previous = cache.get(name, []) if cache is not None else []
        current, hashes = [], []
        for index, item in enumerate(value):
            # WHY: 运行内缓存用独立快照比较，原地修改嵌套Span也会失效；不能仅信任版本号或对象身份。
            if index < len(previous) and previous[index][0] == item:
                saved, digest = previous[index]
            else:
                digest = hashlib.sha256(json.dumps(item, ensure_ascii=False, sort_keys=True,
                    default=str).encode("utf-8")).hexdigest()
                saved = copy.deepcopy(item) if cache is not None else None
            hashes.append(digest)
            if len(value) <= 512:
                current.append((saved, digest))
        if cache is not None:
            cache[name] = current
        # WHY: 内容摘要排序保留集合语义，避免再次JSON转义整段证据原文；缓存不进入持久化状态。
        return sorted(hashes)

    payload = {
        "scope": state.get("selected_scope") or {},
        "candidates": sorted({str(p.get("doi") or p.get("arxiv_id") or p.get("paper_id") or p.get("title")) for p in state.get("candidate_papers") or []}),
        # WHY: 排序抖动不算进展；逐篇语义合同确认后才算新增可处理材料。
        "admitted_ranked": sorted({json.dumps({
            "id": paper.get("paper_id") or paper.get("id") or paper.get("title"),
            "content": paper.get("_screened_content_fingerprint"),
            "scope": paper.get("_screened_scope_fingerprint"),
            "relation": paper.get("_topic_relation"),
            "eligible": sorted(paper.get("_eligible_deliverables") or []),
        }, ensure_ascii=False, sort_keys=True) for paper in admitted_ranked_papers(state)})
        if (state.get("retrieval_profile") or {}).get("mode") == "hybrid" else [],
        "details": sorted(_detail_signature(p) for p in state.get("paper_details") or []),
        "cards": stable_items("paper_cards"),
        "routes": stable_items("validated_routes"),
        "taxonomy": state.get("dynamic_taxonomy") or {},
        "claims": stable_items("claim_plans"),
        "gaps": state.get("evidence_gap_report") or {},
        "review": state.get("review") or "",
        "published_version": state.get("published_draft_version"),
        "quality": state.get("quality_gate") or {},
        # WHY: 完成独立复验是研究进展；同一报告重放不变，仍会被无进展/循环限制。
        "pending_validation": bool(state.get("pending_result_validation")),
        "verification": {key: (state.get("claim_verification") or {}).get(key)
                         for key in ("valid", "supported", "unsupported", "unverified", "support_rate")},
        "valid_citations": state.get("unique_valid_cited_paper_count") or 0,
    }
    cache = _PROGRESS_CACHE.get()
    cached = cache.get("final_payload") if cache is not None else None
    if cached is not None and cached[0] == payload:
        return cached[1]
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:20]
    if cache is not None:
        # WHY: 只复用实际内容相等的进展载荷；独立快照能发现原地质量/范围修改，不持久化最新指纹。
        cache["final_payload"] = (copy.deepcopy(payload), digest)
    return digest


class MainAgentLoop:
    def __init__(self, llm, *, controller: AgentController | None = None) -> None:
        self.policy = MainAgentPolicy(llm)
        self.controller = controller or AgentController()
        self.sidecar = ActionSidecar(llm)
        from app.agent.subagents.loop import SubagentLoop
        self.specialists = SubagentLoop(llm, self.controller)

    def run(
        self, state, *, handlers, finish_validator, should_cancel=None,
    ) -> str:
        try:
            with budget_scope(state, should_cancel), progress_cache_scope():
                status = self._run(state, handlers=handlers, finish_validator=finish_validator,
                                   should_cancel=should_cancel)
                if (state.get("main_agent_trajectory") or {}).get("pending"):
                    observe_terminal(state, {"status": status, "error": str((state.get("errors") or [{}])[-1].get("code") or "")})
                return status
        except (AgentBudgetExceeded, ContextBudgetExceeded, AgentExecutionStale, AgentCancelledError, AgentDecisionError, LLMInvocationError, RuntimeConflict) as exc:
            # WHY: AgentExecutionStale 继承 AgentCancelledError 以复用取消
            # 透传，但租约失效不是用户取消，结果状态必须是 failed。
            if isinstance(exc, AgentExecutionStale):
                status = "failed"
            elif isinstance(exc, AgentCancelledError):
                status = "cancelled"
            elif isinstance(exc, (AgentBudgetExceeded, ContextBudgetExceeded)):
                status = "blocked"
            else:
                status = "failed"
            state["result_status"] = status
            if status == "blocked":
                state["generation_blocked"] = True
            state.setdefault("errors", []).append({"code": type(exc).__name__, "message": str(exc)})
            if (state.get("main_agent_trajectory") or {}).get("pending"):
                observe_terminal(state, {"status": status, "error": type(exc).__name__})
            return status

    def _run(
        self,
        state: dict[str, Any],
        *,
        handlers: dict[str, Callable[[dict[str, Any], dict[str, Any]], Any]],
        finish_validator: Callable[[dict[str, Any]], bool],
        should_cancel: Callable[[], bool] | None = None,
    ) -> str:
        settings = get_settings()
        trajectory = ReActContext(state)
        if state.get("active_subagent"):
            # WHY: 检查点可能停在子循环内；先续接其观察与汇报，再允许主控生成新决策。
            restored_result = self.specialists.run(state, handlers=handlers, should_cancel=should_cancel)
            self._consume_delegation_result(state, restored_result)
        no_progress = 0
        previous_progress = _progress_fingerprint(state)
        progress_limit = max(1, int(settings.agent_main_no_progress_limit))
        recent_progress = deque([previous_progress], maxlen=max(4, 2 * progress_limit + 1))
        ledger = state.setdefault("agent_execution_budget", {})
        limit = int(ledger.setdefault("main_round_limit", settings.agent_main_max_rounds))
        for round_index in range(max(0, limit - int(ledger.get("decision_rounds") or 0))):
            check_execution()
            from app.agent.execution_budget import update_finalization
            update_finalization(state)
            rollback_count_before = len(state.get("recovery_candidate_rejections") or [])
            if should_cancel and should_cancel():
                state["result_status"] = "cancelled"
                return "cancelled"
            from app.services.research_memory_retrieval import prepare_memory_context
            prepare_memory_context(state)
            names = allowed_actions(state, set(handlers))
            mandatory = list(state.get("agent_mandatory_actions") or [])
            if mandatory:
                required = mandatory[0]
                action_name = str(required.get("action") or "") if isinstance(required, dict) else ""
                if action_name not in names:
                    state["result_status"] = "blocked"
                    state["generation_blocked"] = True
                    state.setdefault("errors", []).append({
                        "code": "required_recovery_action_unavailable",
                        "message": action_name,
                    })
                    return "blocked"
                names = [action_name]
            handoff = recovery_handoff_action(state) if not mandatory else ""
            if handoff and handoff in names:
                names = [handoff]
            from app.services.research_execution_service import checkpoint_artifacts
            checkpoint_artifacts(state)
            state["allowed_agent_actions"] = names
            snapshot = refresh_main_agent_context(state)
            if snapshot.validation_errors:
                state["errors"] = list(state.get("errors") or []) + [
                    {"code": "main_context_invalid", "message": ",".join(snapshot.validation_errors)}
                ]
                status = "blocked" if set(snapshot.validation_errors) == {"context_budget_exceeded"} else "failed"
                state["result_status"] = status
                if status == "blocked":
                    state["generation_blocked"] = True
                return status
            try:
                if mandatory:
                    arguments = dict(required.get("arguments") or {})
                    validate_arguments(action_name, arguments)
                    decision = AgentDecision(
                        action=action_name,
                        arguments=arguments,
                        reason="履行已选定的质量恢复动作",
                    )
                elif handoff and handoff in names:
                    decision = AgentDecision(
                        action=handoff, arguments={},
                        reason="按草稿优先策略先发布可用版本" if handoff == "publish_draft" else "推进已确认论文到当前证据与主张授权",
                    )
                else:
                    decision = self.policy.decide(snapshot.context, allowed_actions=names, trajectory=trajectory)
            except ContextBudgetExceeded:
                raise
            except (AgentDecisionError, ValueError) as exc:
                state["errors"] = list(state.get("errors") or []) + [
                    {"code": "invalid_agent_decision", "message": str(exc)}
                ]
                state["result_status"] = "failed"
                return "failed"
            from app.services.execution_report_capture import record_main_context
            record_main_context(state,snapshot.context,decision,model_input=not bool(mandatory or (handoff and handoff in names)))
            state.setdefault("main_agent_decisions", []).append(
                decision.model_dump(mode="json")
            )
            from app.services.research_memory_retrieval import record_memory_feedback
            record_memory_feedback(state, decision, presented_matches=snapshot.context.state.research_memory)
            if mandatory or (handoff and handoff in names):
                trajectory.proposed(decision)
            ledger = state.setdefault("agent_execution_budget", {})
            ledger["decision_rounds"] = int(ledger.get("decision_rounds") or 0) + 1
            if int(ledger.get("llm_tokens") or 0) > int(ledger["token_limit"]):
                state["result_status"] = "blocked"
                state["generation_blocked"] = True
                state.setdefault("errors", []).append({
                    "code": "agent_token_budget_exhausted",
                    "message": "主 Agent 模型 token 预算已耗尽",
                })
                return "blocked"
            action = decision.action
            from app.services.research_memory_retrieval import record_memory_control_outcome
            # Review only explicitly configured actions, before Controller effects.
            # The reviewer cannot override deterministic Controller authorization.
            if (settings.agent_sidecar_enabled
                    and action in {item.strip() for item in settings.agent_sidecar_review_actions.split(",") if item.strip()}
                    and action not in {"request_finish", "report_blocked", "request_clarification"}):
                review = self.sidecar.review(operation=action, arguments=decision.arguments)
                state.setdefault("agent_sidecar_reviews", []).append({
                    "action": action, "verdict": review.verdict, "reason": review.reason,
                })
                if review.verdict != "allow":
                    trajectory.observe({"status": review.verdict, "error": review.reason})
                    record_memory_control_outcome(state, action, review.verdict, "blocked")
                    state.setdefault("main_agent_rejections", []).append({
                        "action": action,
                        "reason": "sidecar_" + review.verdict + ":" + review.reason,
                        "round": round_index,
                    })
                    if review.verdict == "escalate":
                        state["clarification"] = {
                            "needed": True, "kind": "sidecar_review",
                            "question": "此项研究操作需要人工确认：" + review.reason,
                        }
                        state["result_status"] = "waiting_user"
                        return "waiting_user"
                    # Denial becomes a new observation; no automatic re-execution.
                    continue
            if action == "recall_context":
                recalled = execute_context_recall(trajectory, decision.arguments)
                record_memory_control_outcome(state, action, recalled["status"], recalled["outcome"])
            elif action == "search_research_memory":
                from app.services.research_memory_retrieval import execute_memory_search
                recalled = execute_memory_search(state, trajectory, decision.arguments)
                record_memory_control_outcome(state, action, recalled["status"], recalled["outcome"])
            elif action == "request_clarification":
                trajectory.observe({"status": "waiting_user", "findings": [decision.arguments["question"]]})
                record_memory_control_outcome(state, action, "waiting_user", "unknown")
                state["clarification"] = {
                    "needed": True,
                    "question": decision.arguments["question"],
                    "kind": "main_agent",
                }
                state["result_status"] = "waiting_user"
                return "waiting_user"
            elif action == "report_blocked":
                trajectory.observe({"status": "blocked", "error": decision.arguments["reason"]})
                record_memory_control_outcome(state, action, "blocked", "blocked")
                state["generation_blocked"] = True
                state["errors"] = list(state.get("errors") or []) + [{
                    "code": "main_agent_blocked",
                    "message": decision.arguments["reason"],
                }]
                state["result_status"] = "blocked"
                return "blocked"
            elif action == "publish_draft":
                try:
                    previous_version = state.get("published_draft_version")
                    record = self.controller.publish_draft(state)
                    unchanged = previous_version == record["version"]
                    trajectory.observe({"status":"no_progress" if unchanged else "completed",
                        "reason":"already_published" if unchanged else "published",
                        "delivery_status":record["grade"], "version":record["version"]})
                except ValueError as exc:
                    trajectory.observe({"status":"rejected", "error":str(exc)[:300]})
            elif action == "request_finish":
                if finish_validator(state):
                    trajectory.observe({"status": str(state.get("result_status") or "completed")})
                    record_memory_control_outcome(state, action, str(state.get("result_status") or "completed"),
                                                  "degraded" if state.get("result_status") == "partial" else "succeeded")
                    return str(state.get("result_status") or "completed")
                trajectory.observe({"status": "rejected", "error": "finish_rejected_by_quality_gates"})
                record_memory_control_outcome(state, action, "rejected", "blocked")
                state.setdefault("main_agent_rejections", []).append({
                    "action": action,
                    "reason": "finish_rejected_by_quality_gates",
                    "round": round_index,
                })
            else:
                spec = action_spec(action)
                handler = handlers.get(action)
                if spec.role is None or handler is None:
                    state["result_status"] = "failed"
                    state.setdefault("errors", []).append({
                        "code": "unbound_agent_action", "message": action,
                    })
                    return "failed"
                from app.agent.generation_recovery import RewriteTargetError
                result = None
                try:
                    from app.agent.quality_feedback import begin_main_recovery
                    begin_main_recovery(state, action, decision.arguments)
                    result = self.specialists.run(
                        state, decision=decision, handlers=handlers,
                        decision_context=snapshot.context,
                        should_cancel=should_cancel,
                        consume_mandatory_action=bool(mandatory),
                        required_first=bool(handoff),
                    )
                    self._consume_delegation_result(state, result)
                except RewriteTargetError as exc:
                    trajectory.observe({"status": "rejected", "error": "invalid_rewrite_target", "detail": str(exc)[:240]})
                    state.setdefault("main_agent_rejections", []).append({"action": action, "reason": "invalid_rewrite_target"})
                except AgentBudgetExceeded as exc:
                    from app.agent.execution_budget import AgentFinalizationRequired
                    if isinstance(exc, AgentFinalizationRequired):
                        role=state.get("active_subagent")
                        if role:
                            result=self.specialists._finish(state, role, "needs_help", "budget_low：已保留收尾预算，请基于已提交材料写作或发布草稿。")
                            self._consume_delegation_result(state,result)
                        continue
                    state["result_status"] = "blocked"
                    state["generation_blocked"] = True
                    state.setdefault("errors", []).append({
                        "code": "AgentBudgetExceeded",
                        "message": str(exc),
                    })
                    return "blocked"
                if result is not None and result.status.value in {"failed", "cancelled"}:
                    state["result_status"] = result.status.value
                    return result.status.value

            current_progress = _progress_fingerprint(state)
            no_progress = no_progress + 1 if current_progress == previous_progress else 0
            state_cycled = current_progress != previous_progress and current_progress in recent_progress
            recent_progress.append(current_progress)
            previous_progress = current_progress
            # WHY: 独立核验拒绝退化候选、恢复已验证旧稿会回到历史状态；
            # 仍须当前版本通过最终门禁，不能把安全回滚误判为无效循环。
            if (state_cycled and action == "validate_result" and not state.get("pending_result_validation")
                    and len(state.get("recovery_candidate_rejections") or []) > rollback_count_before):
                from app.agent.graph import _draft_fingerprint
                # WHY: 仅本次独立核验确认的安全回滚可交回Main换方案；历史回滚不能关闭循环保护。
                if state.get("autonomous_verified_fingerprint") == _draft_fingerprint(state):
                    state_cycled = False
            # WHY: 生成/改写只提交候选，验证前无法判断质量进展；只延期这一次写作动作的判定。
            # 下一轮仍受动作准入、轮数/预算限制；验证后照常检查无进展和循环，不能直接发布。
            if (action in {"generate_deliverables", "rewrite_sections"}
                    and state.get("pending_result_validation")):
                continue
            if no_progress >= progress_limit:
                from app.agent.delivery import draft_first, draft_materials
                if draft_first(state) and draft_materials(state) and state.get("finalization_phase") != "finalizing":
                    # WHY: 素材已可写时先给Main一次收尾机会；再次无进展仍停止，不延长硬预算。
                    state["finalization_phase"]="finalizing"
                    state["finalization_reason"]="research_no_progress"
                    no_progress=0
                    recent_progress.clear()
                    continue
                state["generation_blocked"] = True
                state["result_status"] = "blocked"
                state.setdefault("errors", []).append({
                    "code": "agent_no_progress",
                    "message": "主 Agent 连续动作未产生实质研究进展",
                })
                return "blocked"
            if state_cycled:
                state["generation_blocked"] = True
                state["result_status"] = "blocked"
                state.setdefault("errors", []).append({
                    "code": "agent_state_cycle",
                    "message": "主 Agent 研究状态反复往返，未产生持续进展",
                })
                return "blocked"
        state["generation_blocked"] = True
        state["result_status"] = "blocked"
        state.setdefault("errors", []).append({
            "code": "agent_round_limit",
            "message": "主 Agent 达到最大决策轮数",
        })
        return "blocked"

    @staticmethod
    def _consume_delegation_result(state, result):
        from app.services.research_memory_retrieval import record_memory_outcome
        record_memory_outcome(state, result)
        # WHY: COMPLETED 只说明委派已结束；阻塞/协助报告必须反馈给下一轮主控，不能当业务成功。
        from app.schemas.agent_task_schema import AgentTaskOutcome
        if result.delegation_status in {"needs_help", "blocked"} or result.outcome in {
            AgentTaskOutcome.BLOCKED, AgentTaskOutcome.DEGRADED, AgentTaskOutcome.UNKNOWN,
        }:
            feedback = (result.delegation_status if result.delegation_status in {"needs_help", "blocked"}
                        else result.outcome.value)
            state.setdefault("main_agent_rejections", []).append({
                "action": result.operation,
                "reason": "subagent_" + feedback,
                "delegation_status": result.delegation_status,
                "outcome": result.outcome.value,
                "message": "; ".join(result.findings)[:600],
            })
