"""Agent 工作流编排。

``run_research_agent`` / ``continue_research_agent`` / ``regenerate_research_agent``
是三个公共入口，统一进入五字段主 Agent 的单动作决策循环。历史状态在执行入口
规范为自主编排；节点权限、预算、提交版本复核与质量门禁仍由代码裁决。

本文件同时承载写作质量边界（主张对齐、引用一致性、引用缺口修复与草稿回滚）和
最终输出组装。执行原语（协作式取消、节点边界检查、LLM 工厂）在
``app.agent.execution``，检索精化循环在 ``app.agent.retrieval_loop``，
证据恢复状态机在 ``app.agent.recovery_loop``。
"""

from __future__ import annotations

import re
import copy

from typing import Any, Callable, Dict, List, Optional

from app.agent.controller import AgentController
from app.agent.orchestration import normalize_orchestration_mode
from app.agent.execution import AgentCancelledError
from app.agent.execution_budget import research_budget
from app.agent.execution import checkpoint as _checkpoint
from app.agent.execution import get_llm as _get_llm
from app.agent.nodes import (
    append_step,
    citation_check_node,
    claim_evidence_gate_node,
    claim_plan_node,
    cluster_node,
    download_pdf_node,
    fetch_detail_node,
    final_answer_node,
    extract_card_node,
    generate_deliverables_node,
    global_evidence_gate_node,
    parse_pdf_node,
    plan_node,
    validate_routes_node,
    verify_claims_node,
)
from app.agent.recovery_loop import run_route_evidence_recovery as _run_route_evidence_recovery
from app.agent.retrieval_loop import (
    search_rank_with_refinement as _search_rank_with_refinement,
)
from app.agent.router import should_parse_pdf
from app.agent.state import ResearchAgentState
from app.core.config import get_settings
from app.core.logger import get_logger
from app.schemas.agent_task_schema import AgentRole

logger = get_logger(__name__)


def _role_llm(llm, role: str):
    """生产客户端使用稳定角色前缀；测试/旧适配器保持 duck typing 兼容。"""
    binder = getattr(llm, "for_agent", None)
    return binder(role) if callable(binder) else llm


def _draft_fingerprint(state):
    from app.agent.context_builder import _stable_hash
    # WHY: 调度状态在任务提交时改变，不属于正文验证输入；证据和授权改变则必须重验。
    return _stable_hash({key: state.get(key) for key in (
        "review", "paper_cards", "paper_details", "claim_plans", "writing_plans",
        "citation_map", "citation_registry", "required_reference_count", "start_year",
        "end_year", "quality_gate", "claim_verification", "core_deliverables",
    )})


def _run_autonomous_pipeline(
    state: ResearchAgentState,
    *,
    should_cancel: Callable[[], bool] | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
    local_verification: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """以五字段主控循环驱动既有研究节点；门禁仍由确定性代码执行。"""
    from app.agent.main_loop import MainAgentLoop

    normalize_orchestration_mode(state)
    settings = get_settings()
    llm = _get_llm()
    if state.get("agent_operation_mode") == "verification_only" and not state.get("agent_mandatory_actions"):
        # WHY: 用户/恢复阶梯已选定只重验，模型不能用 finish 或 blocked
        # 跳过本轮验证；检查点恢复也必须继续履行同一动作。
        state["agent_mandatory_actions"] = [{"action": "validate_result"}]
    state.pop("result_status", None)
    state.pop("clarification", None)
    from app.services.durable_execution_service import active_runtime
    runtime = active_runtime()
    if runtime:
        runtime.snapshot(state, ledger=state.get("agent_execution_budget") or {})
    from app.agent.context_builder import _stable_hash

    # WHY: 局部重验只对同一证据和同一授权成立；主循环若先改了任一依赖，
    # 验证动作必须自动退回全文验证，不能复用上一稿的句级报告。
    local_source = _stable_hash({key: state.get(key) for key in (
        "paper_details", "paper_cards", "claim_plans", "validated_routes",
    )}) if local_verification else None

    def _search(current: ResearchAgentState, _arguments: dict[str, Any]) -> None:
        _search_rank_with_refinement(
            current,
            llm=_role_llm(llm, "search"),
            should_cancel=should_cancel,
            progress_callback=progress_callback,
            total_steps=max(1, int(settings.agent_main_max_rounds)),
        )

    def _targeted_search(current: ResearchAgentState, _arguments: dict[str, Any]) -> None:
        from app.agent.evidence_recovery import targeted_search_kind
        kind = targeted_search_kind(current)
        if not kind:
            raise ValueError("targeted search is unavailable in current state")
        # WHY: 两种补检索都受现有有界恢复预算约束；这里是单个主控动作，
        # 内部算法不会再次启动主 Agent 循环。
        if kind == "citation":
            previous = copy.deepcopy(current)
            from app.agent.execution_budget import consume
            consume("recovery")
            _repair_citation_gap(current, should_cancel, progress_callback,
                                 step_idx=0, total_steps=max(1, settings.agent_main_max_rounds))
            _validate_result(current, {})
            _retain_better_generation(previous, current)
            current.pop("_citation_gap_repair_snapshot", None)
            current.pop("_citation_gap_repair_previous_cited", None)
        else:
            dependency_keys = ("paper_cards", "paper_details", "validated_routes", "dynamic_taxonomy")
            before = copy.deepcopy({key: current.get(key) for key in dependency_keys})
            _run_route_evidence_recovery(current, should_cancel=should_cancel)
            if before != {key: current.get(key) for key in dependency_keys} or not current.get("claim_plans"):
                # WHY: 补搜改变证据或路线后，旧授权/门禁不能继续代表当前证据。
                # 与既有增量流程一致，在同一复合动作内重建授权、写作并重验；硬约束不变。
                _reset_generation_products(current)
                current.pop("autonomous_verified_fingerprint", None)
                _claims(current, {})
                _write(current, {})

    def _fetch(current: ResearchAgentState, _arguments: dict[str, Any]) -> None:
        fetch_detail_node(current, should_cancel=should_cancel)
        if settings.enable_pdf_pipeline:
            download_pdf_node(current, should_cancel=should_cancel)
            if should_parse_pdf(current):
                parse_pdf_node(current, should_cancel=should_cancel)

    def _extract(current: ResearchAgentState, _arguments: dict[str, Any]) -> None:
        card_llm = (
            _role_llm(llm, "analysis") if settings.enable_llm_card_extraction else None
        )
        extract_card_node(current, llm=card_llm, should_cancel=should_cancel)

    def _validate_routes(current: ResearchAgentState, _arguments: dict[str, Any]) -> None:
        validate_routes_node(current, llm=_role_llm(llm, "analysis"))
        if not any(route.get("paper_ids") for route in (current.get("validated_routes") or [])):
            # WHY: 恢复动作可能强制先重验路线；没有可写路线时，仍须在同一
            # 授权动作内建立证据驱动的回退分类，随后才能重建主张授权。
            cluster_node(
                current,
                llm=_role_llm(llm, "analysis") if settings.enable_llm_clustering else None,
            )

    def _cluster(current: ResearchAgentState, _arguments: dict[str, Any]) -> None:
        cluster_node(
            current,
            llm=_role_llm(llm, "analysis") if settings.enable_llm_clustering else None,
        )

    def _claims(current: ResearchAgentState, _arguments: dict[str, Any]) -> None:
        claim_plan_node(current, llm=_role_llm(llm, "analysis"))
        _run_claim_evidence_gate(current)
        if current.get("paper_details") and settings.enable_global_evidence_gate:
            global_evidence_gate_node(current)

    def _write(current: ResearchAgentState, arguments: dict[str, Any]) -> None:
        previous = copy.deepcopy(current)
        # WHY: 清除上轮就绪/隔离诊断再写；已有草稿仅用于验证后质量比较。
        for key in ("generation_readiness", "deliverable_readiness", "generation_blocked", "generation_quality",
                    "writer_section_diagnostics", "writer_diagnostics", "quarantined_draft"):
            current.pop(key, None)
        if arguments.get("section_ids"):
            current["target_section_ids"] = list(arguments["section_ids"])
        current.pop("autonomous_verified_fingerprint", None)
        current.pop("quality_gate", None)
        _generate_deliverables_or_block(current, should_cancel=should_cancel)
        _validate_result(current, {})
        _retain_better_generation(previous, current)

    def _retain_better_generation(previous, current):
        from app.agent.generation_recovery import candidate_is_not_worse
        if previous.get("review") and not candidate_is_not_worse(previous, current):
            for key in _GENERATION_PRODUCT_KEYS:
                current.pop(key, None)
            current.update(copy.deepcopy(_snapshot_generation_products(previous)))
            current.setdefault("recovery_candidate_rejections", []).append({
                "reason": "自主修复候选质量退化，保留已验证版本",
            })
            _validate_result(current, {})

    def _validate_result(current: ResearchAgentState, _arguments: dict[str, Any]) -> None:
        verify_kwargs = None
        if local_verification and local_source == _stable_hash({key: current.get(key) for key in (
            "paper_details", "paper_cards", "claim_plans", "validated_routes",
        )}):
            verify_kwargs = local_verification
        if verify_kwargs:
            _verify_generated_draft(current, verify_claims_kwargs=verify_kwargs)
        else:
            _verify_generated_draft(current)
        final_answer_node(current)
        current["autonomous_verified_fingerprint"] = _draft_fingerprint(current)

    handlers = {
        "search_and_rank": _search,
        "targeted_search": _targeted_search,
        "fetch_metadata": _fetch,
        "extract_paper_cards": _extract,
        "validate_routes": _validate_routes,
        "cluster_papers": _cluster,
        "plan_claims": _claims,
        "generate_deliverables": _write,
        "rewrite_sections": _write,
        "validate_result": _validate_result,
    }

    def _finish(current: ResearchAgentState) -> bool:
        if current.get("intent") == "search_papers":
            from app.agent.focus_coverage import required_focus_coverage
            from app.agent.nodes.base import _paper_identity_key

            papers = list(current.get("ranked_papers") or [])
            actual = len({_paper_identity_key(paper) for paper in papers})
            requested = int(current.get("required_reference_count") or 0) if current.get("max_papers_explicit") else 0
            coverage = required_focus_coverage(current.get("research_semantic_frame") or {}, papers)
            issues = []
            if not actual:
                issues.append("未检索到符合当前范围的论文")
            if requested and actual < requested:
                issues.append(f"要求返回至少 {requested} 篇，当前只有 {actual} 篇有效论文")
            missing_focuses = [str(item) for item in coverage.get("missing_focuses") or [] if str(item).strip()]
            if missing_focuses:
                issues.append("以下研究重点尚无匹配论文：" + "、".join(missing_focuses))
            current["search_result_quality"] = {
                "passed": not issues,
                "requested": requested,
                "actual": actual,
                "issues": issues,
                "warnings": ["部分检索源失败，结果可能不完整"] if current.get("search_failed") else [],
            }
            # WHY: 检索任务没有正文，写作门禁和旧会话草稿状态不能决定论文列表的发布。
            current.pop("quality_gate", None)
            current["result_status"] = "blocked" if not actual else "partial" if issues else "completed"
            if not actual:
                current.setdefault("errors", []).append({"code": "search_results_empty", "message": ""})
            final_answer_node(current)
            return True
        if not str(current.get("review") or "").strip():
            return False
        # 只有当前正文已经过验证时才允许交付；模型不能用 finish 跳过验证。
        if not current.get("quality_gate"):
            return False
        if current.get("autonomous_verified_fingerprint") != _draft_fingerprint(current):
            return False
        # WHY: validate_result 已对同一指纹运行最终门禁；再次调用会重复写入
        # 质量诊断，且可能使恢复轮的事件和计数不再幂等。
        status = derive_result_status(current)
        if status == "success":
            current["result_status"] = "completed"
            return True
        quality = current.get("quality_gate") or {}
        if status == "partial" and quality.get("draft_released"):
            current["result_status"] = "partial"
            return True
        return False

    status = MainAgentLoop(llm).run(
        state,
        handlers=handlers,
        finish_validator=_finish,
        should_cancel=should_cancel,
    )
    state["result_status"] = status
    if status == "waiting_user":
        state["answer"] = (state.get("clarification") or {}).get("question", "请补充研究范围。")
    if (state.get("active_quality_recovery") and not state.get("agent_mandatory_actions")
            and status in {"completed", "partial", "blocked", "failed", "cancelled"}):
        from app.agent.generation_recovery import complete_recovery_action

        complete_recovery_action(state)
    if progress_callback:
        progress_callback(status, 1, 1)
    return _build_output(state)


def _run_search_subagent(
    state: ResearchAgentState,
    *,
    objective: str,
    llm,
    should_cancel=None,
    progress_callback=None,
    total_steps: int,
) -> None:
    AgentController().execute(
        role=AgentRole.SEARCH,
        operation="search_and_rank",
        objective=objective,
        state=state,
        handler=lambda current: _search_rank_with_refinement(
            current,
            llm=llm,
            should_cancel=should_cancel,
            progress_callback=progress_callback,
            total_steps=total_steps,
        ),
        constraints={
            "start_year": state.get("start_year"),
            "end_year": state.get("end_year"),
            "required_reference_count": state.get("required_reference_count"),
        },
        expected_output=["candidate_papers", "ranked_papers", "source_diagnostics"],
        budget={"remaining_recovery_actions": max(
            0,
            int(get_settings().recovery_total_action_budget)
            - int(state.get("recovery_action_count") or 0),
        )},
        should_cancel=should_cancel,
    )


def _run_search_stage(
    state: ResearchAgentState,
    *,
    operation: str,
    objective: str,
    handler: Callable[[ResearchAgentState], Any],
    expected_output: list[str],
    should_cancel=None,
) -> None:
    AgentController().execute(
        role=AgentRole.SEARCH,
        operation=operation,
        objective=objective,
        state=state,
        handler=handler,
        expected_output=expected_output,
        should_cancel=should_cancel,
    )


def _run_analysis_subagent(
    state: ResearchAgentState,
    *,
    operation: str,
    objective: str,
    handler: Callable[[ResearchAgentState], Any],
    expected_output: list[str],
    should_cancel=None,
) -> None:
    AgentController().execute(
        role=AgentRole.ANALYSIS,
        operation=operation,
        objective=objective,
        state=state,
        handler=handler,
        expected_output=expected_output,
        input_artifact_refs=[
            f"state://paper/{paper.get('paper_id')}"
            for paper in (state.get("paper_details") or [])
            if paper.get("paper_id")
        ],
        should_cancel=should_cancel,
    )


@research_budget
def run_research_agent(
    user_query: str,
    current_year: Optional[int] = None,
    initial_state: Optional[Dict[str, Any]] = None,
    should_cancel: Callable[[], bool] | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> Dict[str, Any]:
    """运行完整 Agent 流程。

    Args:
        user_query: 用户自然语言请求。
        current_year: 当前年份（用于计算时间范围）。
        initial_state: 可选的初始状态，用于传入本文工作信息、研究背景等写作所需字段。
        should_cancel: 取消检查回调函数，返回 True 时停止执行。
        progress_callback: 进度更新回调函数，接收 (step_name, current, total)。

    Returns:
        包含 answer / steps / references / paper_cards 的结果字典。
        
    Raises:
        AgentCancelledError: 任务被取消时抛出。
    """
    # ---------- 初始化状态 ----------
    state: ResearchAgentState = {
        "user_query": user_query,
        "state_schema_version": "2",
        "allow_evidence_expansion": True,
        "recovery_action_count": 0,
        "quality_recovery_history": [],
        "section_checkpoints": {},
        "section_candidate_checkpoints": {},
        "writing_version": 0,
        "steps": [],
        "errors": [],
    }
    
    # 合并用户传入的初始状态（如 our_work, background 等）
    if initial_state:
        state.update(initial_state)
        # 这些字段定义一次执行的身份与审计边界，不能被初始上下文覆盖。
        state["user_query"] = user_query
        state["state_schema_version"] = "2"
        state["steps"] = []
        state["errors"] = []

    logger.info("Agent started: %s", user_query[:100])

    # 进度单调化：引用缺口修复轮会用钳制后的步骤位重复报告较早索引，
    # 直接透传会让进度条回跳；这里保证 current 只前进不后退。
    raw_progress_callback = progress_callback
    progress_high_water = {"current": -1}

    def progress_callback(step: str, current: int, total: int) -> None:  # noqa: F811
        if current < progress_high_water["current"]:
            return
        progress_high_water["current"] = current
        if raw_progress_callback:
            raw_progress_callback(step, current, total)

    # WHY: 主 Agent 的动作序列由模型决定，进度分母只能按决策轮预算估计。
    total_steps = max(1, int(get_settings().agent_main_max_rounds))

    # 取消优先级高于任何能力判断或节点执行。
    if should_cancel and should_cancel():
        logger.info("Agent cancelled before unsupported task guard")
        raise AgentCancelledError("任务已在执行前取消")

    # 在规划与检索之前执行能力门禁。会话入口通常已检查一次，但直接 API
    # 也能调用本函数，因此这里必须保留独立防线。
    from app.agent.unsupported_task_guard import check_unsupported_task

    original_query = str(
        (state.get("research_request") or {}).get("original_query") or user_query
    )
    guard = check_unsupported_task(original_query)
    state["unsupported_task_guard"] = guard.model_dump(mode="json")
    append_step(
        state,
        "unsupported_task_guard",
        "success" if guard.allowed else "blocked",
        input_data={"user_query": original_query},
        output_data=state["unsupported_task_guard"],
        duration_ms=0,
    )
    if not guard.allowed:
        state["generation_blocked"] = True
        state["answer"] = guard.message
        logger.info("Agent blocked before retrieval: %s", guard.unsupported_requests)
        return _build_output(state)

    # ---------- 1. 规划 ----------
    _checkpoint(state, "plan", 0, total_steps, should_cancel, progress_callback)
    # WHY: 原始需求解析属于检索规划适配器；主决策角色仅在五字段循环调用。
    plan_node(state, llm=_role_llm(_get_llm(), "search"), current_year=current_year)
    if state.get("planning_failed"):
        state["answer"] = (
            "## 检索规划失败\n\n"
            "大模型未能生成可靠的中英文检索策略，因此本次请求未执行论文检索。"
            "请稍后重试；系统不会再用中文兜底词查询国际论文库并误报论文数量。"
        )
        append_step(
            state,
            "final_answer",
            "failed",
            error="search_planning_failed",
            duration_ms=0,
        )
        return _build_output(state)
    if not state.get("canonical_topic"):
        state["canonical_topic"] = state.get("topic")

    # ---------- 1.5. 搜索前概念规划（Provisional Routes）----------
    # 在检索之前生成候选研究路线框架，引导后续定向检索。
    # 这是 Layer 1: Conceptual Planning — 解决"先搜再聚类"导致的
    # text/video/期刊论文等无意义分类问题。
    if "research_status" in (state.get("core_deliverables") or []):
        from app.agent.nodes import provisional_route_node
        provisional_route_node(state, llm=_get_llm())

    # 非会话API同样必须在检索前检查相关工作的用户论文信息。
    if "related_work" in (state.get("core_deliverables") or []):
        from app.agent.deliverable_router import check_deliverable_readiness
        from app.schemas.deliverable_schema import CoreDeliverableType

        readiness = check_deliverable_readiness(
            CoreDeliverableType.RELATED_WORK, state, phase="pre_retrieval"
        )
        if not readiness.ready:
            state["deliverable_readiness"] = [readiness.model_dump(mode="json")]
            state["generation_blocked"] = True
            state["review"] = (
                "## 相关工作暂未生成\n\n"
                + (readiness.clarification_question or "请补充用户论文的研究问题和方法路线。")
            )
            final_answer_node(state)
            return _build_output(state)

    normalize_orchestration_mode(state)
    state["agent_operation_mode"] = "initial"
    return _run_autonomous_pipeline(
        state,
        should_cancel=should_cancel,
        progress_callback=progress_callback,
    )



def _run_claim_evidence_gate(state: ResearchAgentState) -> None:
    """执行写作前主张门禁，并兼容旧版单参数节点替身。"""
    import inspect

    if "llm" in inspect.signature(claim_evidence_gate_node).parameters:
        claim_evidence_gate_node(state, llm=_get_llm())
    else:
        claim_evidence_gate_node(state)


def _claim_alignment_check(state: ResearchAgentState) -> None:
    """Post-writing：检查生成文本中的事实主张是否在 Claim Plan 授权范围内。"""
    # 生成门禁阻断时，review 中保存的是阻断说明而不是学术正文，不能对它
    # 执行主张对齐；否则诊断步骤可能掩盖真正的证据不足状态。
    if state.get("generation_blocked"):
        return

    from app.agent.claim_plan import validate_claim_support
    from app.core.logger import get_logger

    logger = get_logger(__name__)
    review_text = str(state.get("review") or "")
    claim_plans = state.get("claim_plans") or []

    if not review_text or not claim_plans:
        return

    result = validate_claim_support(review_text, claim_plans)
    state["claim_alignment"] = result

    unsupported_count = int(result.get("unsupported_sentences") or 0)
    if unsupported_count:
        # support_rate 是 0~1 的比例；原格式串 "%.0%%" 缺类型字符，
        # logging 内部格式化会抛 ValueError 导致告警整体丢失。
        logger.warning(
            "Claim alignment: %d/%d sentences unauthorized (support_rate=%.0f%%)",
            unsupported_count,
            int(result.get("total_factual_sentences") or 0),
            100.0 * float(result.get("support_rate") or 0.0),
        )
    overclaimed_samples = result.get("overclaimed_samples") or []
    if overclaimed_samples:
        logger.warning(
            "Claim alignment: %d overclaimed sentences detected",
            len(overclaimed_samples),
        )


# 一轮写作产出的全部产物键。引用缺口修复的快照/回滚、continue 与
# regenerate 的旧产物清理共用同一清单：此前三套手写列表互不相同且都缺项
# （continue 缺 claim_plans/writing_plans，regenerate 缺得更多），导致正文
# 已重置而计划/授权仍是上一轮的版本，writer 按失效授权写作、完整性检查误报。
_GENERATION_PRODUCT_KEYS = (
    "body", "review", "related_work", "related_work_data", "introduction",
    "introduction_data", "references", "reference_papers", "citation_map",
    "citation_registry", "citation_validation", "claim_verification",
    "generation_quality", "deliverable_validation", "final_review_integrity",
    "quality_gate", "generation_blocked", "quarantined_draft",
    # 写作就绪与章节诊断是本轮派生结果。若恢复时继续保留，状态不变量会把
    # 上一轮的 ready/失败章节误当作当前轮结论，导致尚未写作就被阻断。
    "generation_readiness", "deliverable_readiness", "evidence_quality_report",
    "search_report", "writer_diagnostics", "writer_section_diagnostics",
    "citation_allocation_plan", "claim_repairs",
    "claim_plans", "claim_evidence_gate", "claim_alignment",
    "claim_citation_consistency", "unique_cited_paper_count",
    "unique_valid_cited_paper_count", "final_requirement_met",
    # 写作计划与主题综合也需回滚：否则正文回滚后计划仍是修复轮的新版，
    # 主题序号错位会被完整性检查误报为“缺少计划章节”。
    "writing_plans", "citation_allocation_plans", "theme_synthesis",
    "route_merge_diagnostics",
)

_LOCAL_REWRITE_ISSUE_CODES = {
    "final_text_integrity_not_met",
    "deliverable_structure_invalid",
    "section_generation_failed",
    "claim_evidence_quality_not_met",
    "claim_citation_consistency_not_met",
    "invalid_citations",
    "minimum_cited_references_not_met",
}

# 用户选择“直接生成最佳草稿”是恢复策略，不是新的写作失败。它会被记录在
# quality_gate 供审计，但不能阻止随后针对真实章节/主张问题的局部重写。
_RECOVERY_STRATEGY_ISSUE_CODES = {
    "user_accepted_best_effort_generation",
}


def _derive_local_verification_targets(state: ResearchAgentState) -> Dict[str, Any]:
    """从上一轮修复记录和门禁诊断派生局部验证目标。

    目标只用于验证，不改变范围、路线或 renderer 的职责；无法定位具体句子时
    返回空目标，由调用方保留局部重写的中间产物复用，但验证器会安全回退全量。
    """
    claim_ids: set[str] = set()
    sentence_indices: set[int] = set()

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                key_name = str(key).lower()
                if key_name in {"claim_id", "claimid"} and item:
                    claim_ids.add(str(item))
                elif key_name in {"claim_ids", "removed_claim_ids", "rewritten_claim_ids"}:
                    if isinstance(item, (list, tuple, set)):
                        claim_ids.update(str(value) for value in item if value)
                    elif item:
                        claim_ids.add(str(item))
                elif key_name in {"sentence_index", "sentence_idx", "index"}:
                    try:
                        sentence_indices.add(int(item))
                    except (TypeError, ValueError):
                        pass
                elif key_name == "sentence" and item:
                    sentence = str(item)
                    for claim in (state.get("claim_verification") or {}).get("claims") or []:
                        if str(claim.get("sentence") or "") == sentence:
                            match = re.match(r"^c(\d+)", str(claim.get("claim_id") or ""))
                            if match:
                                sentence_indices.add(int(match.group(1)))
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(state.get("claim_repairs") or {})
    collect(state.get("claim_citation_consistency") or {})
    collect((state.get("quality_gate") or {}).get("blocking_issues") or [])
    section_diagnostics = state.get("writer_section_diagnostics") or []
    collect(section_diagnostics)

    # Section diagnostics often carry only section_id. Resolve that identifier
    # through the existing writing plan and map the section's sentences back to
    # the prior claim report; this remains diagnostic-only and does not alter
    # scope or rendering behavior.
    successful_section_statuses = {
        "success", "already_valid", "success_after_english_repair",
        "success_after_route_merge", "reused", "validated",
    }
    section_ids = {
        str(item.get("section_id") or "")
        for run in section_diagnostics if isinstance(run, dict)
        for item in (run.get("sections") or [])
        if isinstance(item, dict)
        and item.get("section_id")
        and (
            str(item.get("status") or "") not in successful_section_statuses
            or bool(item.get("errors"))
        )
    }
    prior_claims = (state.get("claim_verification") or {}).get("claims") or []
    review_text = str(state.get("review") or state.get("related_work") or state.get("introduction") or "")
    if section_ids and prior_claims and review_text:
        from app.tools.verify_claims import split_review_sentences

        for plan in state.get("writing_plans") or []:
            for section in plan.get("sections") or []:
                section_id = str(section.get("id") or section.get("section_id") or "")
                if section_id not in section_ids:
                    continue
                title = str(section.get("title") or "").strip()
                if not title:
                    continue
                heading_match = re.search(
                    rf"(?ms)^\s*#{1,6}\s+[^\n]*{re.escape(title)}[^\n]*\n(.*?)(?=^\s*#{1,6}\s+|\Z)",
                    review_text,
                )
                section_text = heading_match.group(1) if heading_match else review_text
                section_sentences = set(split_review_sentences(section_text))
                for claim in prior_claims:
                    if str(claim.get("sentence") or "") in section_sentences:
                        match = re.match(r"^c(\d+)", str(claim.get("claim_id") or ""))
                        if match:
                            sentence_indices.add(int(match.group(1)))
    for claim_id in list(claim_ids):
        match = re.match(r"^c(\d+)", claim_id)
        if match:
            sentence_indices.add(int(match.group(1)))
    return {
        "target_sentence_indices": sorted(index for index in sentence_indices if index > 0),
        "target_claim_ids": sorted(claim_ids),
        "target_section_ids": sorted(section_id for section_id in section_ids if section_id),
    }


def _build_regeneration_recovery_plan(state: ResearchAgentState) -> Dict[str, Any]:
    """根据上一轮门禁代码决定保守重写可复用的中间产物。"""
    issue_codes = {
        str(item.get("code") or "")
        for item in (state.get("quality_gate") or {}).get("blocking_issues") or []
        if isinstance(item, dict) and item.get("code")
    }
    issue_codes -= _RECOVERY_STRATEGY_ISSUE_CODES
    targets = _derive_local_verification_targets(state)
    targets["target_section_ids"] = list(dict.fromkeys([
        *[str(value) for value in state.get("target_section_ids") or [] if value],
        *targets["target_section_ids"],
    ]))
    same_evidence_local_rewrite = bool(
        state.get("conservative_regeneration")
        and issue_codes
        and issue_codes.issubset(_LOCAL_REWRITE_ISSUE_CODES)
        and state.get("validated_routes")
        and state.get("claim_plans")
        and not state.get("force_taxonomy_remediation")
        and not state.get("force_claim_plan_rebuild")
    )
    # 旧版本在章节失败路径没有保存 claim-citation 审计。此时继续复用旧
    # Claim Plan 会把新一轮正文的引用绑定到过期授权，表现为正文有引用但
    # 有效引用数骤降；先完整重建授权，才能安全决定是否回到局部重写。
    if (
        same_evidence_local_rewrite
        and "section_generation_failed" in issue_codes
        and not state.get("claim_citation_consistency")
    ):
        same_evidence_local_rewrite = False
    return {
        "issue_codes": sorted(issue_codes),
        "mode": "local_rewrite" if same_evidence_local_rewrite else "full_rebuild",
        "reuse_routes": same_evidence_local_rewrite,
        "reuse_claim_plans": same_evidence_local_rewrite,
        "reuse_global_evidence_gate": bool(
            same_evidence_local_rewrite and state.get("global_evidence_gate")
        ),
        "target_sentence_indices": targets["target_sentence_indices"],
        "target_claim_ids": targets["target_claim_ids"],
        "target_section_ids": targets["target_section_ids"],
        "previous_claim_verification": state.get("claim_verification") if same_evidence_local_rewrite else None,
    }


def _prepare_autonomous_regeneration(
    state: ResearchAgentState,
    *,
    should_cancel: Callable[[], bool] | None,
    progress_callback: Callable[[str, int, int], None] | None,
) -> Dict[str, Any]:
    """履行已选定的恢复准备，并交回唯一主控循环。"""
    recovery_plan = _build_regeneration_recovery_plan(state)
    append_step(
        state, "regeneration_recovery_plan", "success",
        input_data={"blocking_issue_codes": recovery_plan["issue_codes"]},
        output_data=recovery_plan, duration_ms=0,
    )
    state["target_section_ids"] = recovery_plan["target_section_ids"]
    state.pop("autonomous_verified_fingerprint", None)
    if state.pop("refresh_existing_evidence", False):
        # WHY: REFRESH_EVIDENCE 只允许刷新已选论文；原始候选池可能含已筛掉的
        # 论文，不能在恢复轮意外把它们重新送入详情和证据抽取。
        selected = list(state.get("paper_details") or state.get("candidate_papers") or [])
        state["candidate_papers"] = selected
        state["ranked_papers"] = selected
        _checkpoint(state, "refresh_existing_evidence", 0, 1, should_cancel, progress_callback)
        _run_search_stage(
            state, operation="fetch_metadata", objective="刷新现有论文的元数据与可访问证据",
            handler=lambda current: fetch_detail_node(current, should_cancel=should_cancel),
            expected_output=["paper_details", "source_diagnostics"], should_cancel=should_cancel,
        )
        card_llm = _role_llm(_get_llm(), "analysis") if get_settings().enable_llm_card_extraction else None
        _run_analysis_subagent(
            state, operation="extract_paper_cards", objective="刷新现有论文的证据卡",
            handler=lambda current: extract_card_node(current, llm=card_llm, should_cancel=should_cancel),
            expected_output=["paper_cards"], should_cancel=should_cancel,
        )
        recovery_plan["mode"] = "full_rebuild"
        recovery_plan["reuse_routes"] = False
        recovery_plan["reuse_claim_plans"] = False
    if not recovery_plan["reuse_routes"]:
        for key in ("validated_routes", "route_decisions", "route_validation_report",
                    "clusters", "dynamic_taxonomy", "taxonomy_validation"):
            state.pop(key, None)
    if not recovery_plan["reuse_claim_plans"]:
        # WHY: 全文重建可能来自证据刷新或论文排除。旧正文、引用表和授权
        # 同属上一证据版本，不能在候选失败时被写作回滚重新带入公开输出。
        _reset_generation_products(state)
        state.pop("global_evidence_gate", None)
    # 局部重写仍保留同证据旧稿供质量比较；finish 指纹已失效。
    return recovery_plan


def _snapshot_generation_products(state: ResearchAgentState) -> Dict[str, Any]:
    """保存当前写作产物快照（仅非 None 键），供修复退化时回滚。"""
    return {
        key: state.get(key)
        for key in _GENERATION_PRODUCT_KEYS
        if state.get(key) is not None
    }


def _restore_pre_repair_snapshot(
    state: ResearchAgentState,
    *,
    clear_incremental: bool = False,
) -> Dict[str, Any]:
    """恢复修复前写作产物快照；clear_incremental 同时清除增量检索残留。

    引用缺口修复轮可能中途抛异常（原稿已被 pop、快照还在），也可能产生
    退化草稿：两种情况都必须回到修复前快照，不能把半途污染的状态留给
    最终输出——引用表与正文不一致正是这种污染的直接后果。
    """
    snapshot = state.pop("_citation_gap_repair_snapshot", None) or {}
    for key, value in snapshot.items():
        state[key] = value
    if clear_incremental:
        state.pop("incremental_retrieval", None)
        state.pop("incremental_search_window", None)
        state["citation_shortfall_count"] = 0
    return snapshot


def _reset_generation_products(state: ResearchAgentState) -> None:
    """清除上一轮写作产物，防止陈旧计划/授权残留进新一轮写作。"""
    for key in _GENERATION_PRODUCT_KEYS:
        state.pop(key, None)


def _generate_deliverables_or_block(
    state: ResearchAgentState,
    should_cancel=None,
    llm=None,
) -> None:
    """三入口统一的写作调用：异常降级为 quality_gate 阻断而非任务崩溃。

    此前只有 run 主路径有降级包装，continue/regenerate 裸调用——同一类
    契约错误一个返回结构化阻断、一个直接抛未处理异常。协作式取消必须
    继续向上传播，不能被当成生成失败吞掉。
    """
    from app.agent.state_invariants import validate_research_state_invariants

    invariant_result = validate_research_state_invariants(state)
    state["state_invariant_check"] = invariant_result
    if invariant_result.get("blocking_issues"):
        # WHY: 写作前发现旧快照或时间窗口冲突时必须隔离草稿；继续生成会把
        # 陈旧诊断和当前证据混入同一交付物，之后再验证也无法恢复溯源边界。
        state["generation_blocked"] = True
        state["quality_gate"] = {
            "passed": False,
            "phase": "pre_generation",
            "blocking_issues": list(invariant_result["blocking_issues"]),
            "warnings": list(invariant_result.get("warnings") or []),
            "recovery_options": ["刷新当前证据快照和研究范围后重新生成"],
        }
        return

    try:
        generate_deliverables_node(
            state,
            llm=llm or _role_llm(_get_llm(), "writing"),
            should_cancel=should_cancel,
        )
    except AgentCancelledError:
        raise
    except Exception as deliverables_exc:  # noqa: BLE001 - 不让异常逃逸，走质量门禁
        state.setdefault("errors", []).append(
            f"generate_deliverables: {deliverables_exc}"
        )
        state["generation_blocked"] = True
        state["quality_gate"] = {
            "passed": False,
            "phase": "pre_generation",
            "blocking_issues": [{
                "code": "deliverable_generation_failed",
                "message": f"正文生成过程出现错误：{deliverables_exc}",
            }],
            "recovery_options": ["检查日志后重新提交研究请求"],
        }


def _check_claim_citation_consistency(state: ResearchAgentState) -> None:
    """Claim-Citation Consistency：引用的论文是否在 claim 允许的证据中。

    输入缺失时静默返回；写作后验证链（``_verify_generated_draft``）在写作与
    引用校验之后统一调用。
    """
    if not (
        state.get("claim_plans") and state.get("review") and state.get("citation_map")
    ):
        return
    from app.agent.claim_plan import validate_claim_citation_consistency

    ccc_result = validate_claim_citation_consistency(
        str(state.get("review") or ""),
        state.get("claim_plans") or [],
        citation_map=state.get("citation_map"),
    )
    state["claim_citation_consistency"] = ccc_result
    validly_authorized = set(ccc_result.get("validly_authorized_paper_ids") or [])
    state["unique_valid_cited_paper_count"] = len(validly_authorized)
    stats = dict(state.get("reference_coverage_stats") or {})
    stats["actual_cited"] = int(state.get("unique_cited_paper_count") or 0)
    stats["final_valid"] = len(validly_authorized)
    state["reference_coverage_stats"] = stats
    if ccc_result.get("inconsistent_sentences"):
        logger.warning(
            "Claim-Citation Consistency: %d/%d sentences have mismatched citations",
            ccc_result["inconsistent_sentences"],
            ccc_result["consistent_sentences"] + ccc_result["inconsistent_sentences"],
        )


def _verify_generated_draft(
    state: ResearchAgentState,
    *,
    checkpoint: Callable[[str], None] | None = None,
    verify_claims_kwargs: Dict[str, Any] | None = None,
    check_citation_authorization: bool = True,
) -> None:
    """写作后统一验证链：越权主张 → 逐句证据 → 参考文献 → 引用授权一致性。

    四个写作入口（首轮、增量、重生成、引用缺口修复）此前各自复制这段顺序，
    历史缺口正源于此：continue 路径曾漏掉引用授权一致性，修复轮曾漏掉逐句
    验证。这里只固定条件判断与调用顺序。

    WHY: 进度检查点仍由调用方通过 ``checkpoint`` 决定——各入口的步骤编号、
    钳制方式与取消粒度本就不同，把它们塞进本函数会改变已发布的进度语义。
    ``check_citation_authorization=False`` 供 run 主链使用：那里要等引用缺口
    修复轮结束后，再对最终留下的草稿做一次授权一致性判定。
    """
    def _mark(stage: str) -> None:
        if checkpoint is not None:
            checkpoint(stage)

    if state.get("claim_plans") and state.get("review"):
        _mark("claim_alignment")
        _claim_alignment_check(state)
    if state.get("writing_plans") and get_settings().enable_claim_verification:
        _mark("verify_claims")
        verify_claims_node(state, llm=_get_llm(), **(verify_claims_kwargs or {}))
    if state.get("writing_plans"):
        _mark("citation_check")
        # WHY: citation_check_node 的引用生成固定走本地校验，llm 形参不参与
        # 结果；四个入口此前传入的值不一致但对输出无影响，这里统一为 None。
        citation_check_node(state, llm=None)
    if check_citation_authorization:
        _check_claim_citation_consistency(state)


def _repair_citation_gap(
    state: ResearchAgentState,
    should_cancel: Callable[[], bool] | None,
    progress_callback: Callable[[str, int, int], None] | None,
    step_idx: int,
    total_steps: int,
) -> None:
    """引用缺口修复：增量检索定向扩召回后重新走证据与生成链路。

    复用 ``continue_research_agent`` 的增量模式：保留既有论文与卡片，
    只补新文献；refine 反馈携带 ``citation_shortfall`` 引导 LLM 给出
    扩大召回面的互补查询。进度索引用 ``min(step_idx, total_steps-1)``
    钳制，避免修复步骤超出原进度分母。
    """
    required = int(state.get("required_reference_count") or 0)
    cited_before = int(state.get("unique_cited_paper_count") or 0)
    state["citation_gap_repair_attempted"] = True
    state["citation_shortfall_count"] = max(required - cited_before, 0)
    state["_citation_gap_repair_previous_cited"] = cited_before
    state["_citation_gap_repair_snapshot"] = _snapshot_generation_products(state)
    _reset_generation_products(state)
    state["incremental_retrieval"] = True

    append_step(
        state,
        "citation_gap_repair",
        "started",
        input_data={"required_reference_count": required, "cited_before": cited_before},
        output_data={"shortfall": state["citation_shortfall_count"]},
        duration_ms=0,
    )

    settings = get_settings()
    counter = step_idx

    def _run(name: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        nonlocal counter
        _checkpoint(
            state, name, min(counter, max(total_steps - 1, 0)), total_steps,
            should_cancel, progress_callback,
        )
        fn(*args, **kwargs)
        counter += 1

    # 1. 补检索：refine 反馈携带缺口数量，定向扩召回。
    _run(
        "repair_search_and_rank",
        _search_rank_with_refinement,
        state,
        llm=_get_llm(),
        should_cancel=should_cancel,
        progress_callback=progress_callback,
        total_steps=total_steps,
    )
    # 2. 新候选详情补全与 PDF 解析（增量模式只处理新论文）。
    _run("repair_fetch_detail", fetch_detail_node, state, should_cancel=should_cancel)
    if settings.enable_pdf_pipeline:
        _run("repair_download_pdf", download_pdf_node, state, should_cancel=should_cancel)
        if should_parse_pdf(state):
            _run("repair_parse_pdf", parse_pdf_node, state, should_cancel=should_cancel)
    # 3. 卡片抽取（增量模式复用既有卡片）与路线重验。
    _run(
        "repair_extract_cards",
        extract_card_node,
        state,
        llm=_get_llm() if settings.enable_llm_card_extraction else None,
        should_cancel=should_cancel,
    )
    if {"research_status", "related_work", "narrative_review"}.intersection(
        state.get("core_deliverables") or []
    ):
        def _revalidate_routes() -> None:
            route_llm = _role_llm(_get_llm(), "analysis")
            validate_routes_node(state, llm=route_llm)
            # 修复场景不再嵌套证据恢复轮：主流程已跑过恢复，且 refine
            # 已压缩，再加一轮检索只拖长修复时长、边际收益极低。
            if not any(
                route.get("paper_ids") for route in (state.get("validated_routes") or [])
            ):
                cluster_llm = route_llm if settings.enable_llm_clustering else None
                cluster_node(state, llm=cluster_llm)

        _run("repair_revalidate_routes", _revalidate_routes)
    # 4. 重建 claim 授权，再重新评估证据充分性。
    # 顺序不可颠倒：全局门禁的主张强度指标读 claim_plans。
    if (
        any(route.get("paper_ids") for route in (state.get("validated_routes") or []))
        or state.get("dynamic_taxonomy")
    ):
        def _rebuild_claims() -> None:
            claim_plan_node(
                state, llm=_role_llm(_get_llm(), "analysis")
            )
            _run_claim_evidence_gate(state)

        _run("repair_claim_plan", _rebuild_claims)
    if state.get("paper_details") and settings.enable_global_evidence_gate:
        _run("repair_global_evidence_gate", global_evidence_gate_node, state)
    # 5. 基于扩充后的证据池重新生成（验证链由调用方在修复后重跑）。
    _run("repair_regenerate", generate_deliverables_node, state, llm=_get_llm())
    state.pop("incremental_retrieval", None)
    state.pop("incremental_search_window", None)
    # 缺口数量是单次修复轮内部的临时信号：成功路径在此清零，失败路径由
    # 调用方回滚时清零。残留值会让后续检索轮的 refine 收到过期的
    # “引用缺口”误导指引。
    state["citation_shortfall_count"] = 0


def _build_output(state: ResearchAgentState) -> Dict[str, Any]:
    """构造给调用方的输出字典，并执行最终草稿发布边界。"""
    from app.agent.context_builder import refresh_main_agent_context

    refresh_main_agent_context(state)
    output_status = derive_result_status(state)
    quality_gate = state.get("quality_gate") or {}
    draft_is_public = (
        quality_gate.get("passed") is True
        or quality_gate.get("draft_released") is True
    )
    if not draft_is_public and quality_gate.get("passed") is False:
        public_answer = state.get("answer") or "正式正文已被质量门禁阻止，未展示未经验证的正文。"
        public_body = ""
        public_related_work = None
        public_introduction = None
    else:
        public_answer = state.get("answer", state.get("review", state.get("related_work", state.get("introduction", ""))))
        public_body = state.get("body") or state.get("review") or state.get("related_work") or state.get("introduction") or ""
        public_related_work = state.get("related_work")
        public_introduction = state.get("introduction")
        if quality_gate.get("passed") is False and quality_gate.get("draft_released") is True:
            # WHY: API body 和类型化正文也可能被直接下载，不能只在 answer
            # 添加限制说明，否则同一 partial 草稿会出现有标识和无标识两版。
            public_body = public_answer
            if public_related_work:
                public_related_work = public_answer
            if public_introduction:
                public_introduction = public_answer
    if state.get("result_status") in {"waiting_user", "blocked", "failed", "cancelled"}:
        public_body, public_related_work, public_introduction = "", None, None
        if state.get("result_status") == "waiting_user":
            public_answer = (state.get("clarification") or {}).get("question") or "请补充研究要求。"
        else:
            from app.agent.public_errors import public_stop_reason
            base = {
                "blocked": "当前研究未满足交付要求。",
                "failed": "研究执行失败，未交付正文。",
                "cancelled": "研究已取消，未交付正文。",
            }[state["result_status"]]
            reason = public_stop_reason(state.get("errors") or [])
            public_answer = base + ("原因：" + reason + "。" if reason else "")
    output = {
        "status": output_status,
        "clarification": state.get("clarification"),
        "answer": public_answer,
        "body": public_body,
        # 类型稳定为标量：始终取主交付物（首个），无交付物时为 None；
        # 完整列表见 core_deliverables。旧实现单交付物返回 str、
        # 多交付物返回 list，消费方无法依赖统一类型。
        "deliverable_type": (state.get("core_deliverables") or [None])[0],
        "intent": state.get("intent"),
        "topic": state.get("topic"),
        "canonical_topic": state.get("canonical_topic") or state.get("topic"),
        "steps": state.get("steps", []),
        "references": state.get("references", []),
        "paper_cards": state.get("paper_cards", []),
        "clusters": state.get("clusters", []),
        "dynamic_taxonomy": state.get("dynamic_taxonomy"),
        "taxonomy_validation": state.get("taxonomy_validation"),
        "taxonomy_remediation": state.get("taxonomy_remediation"),
        "errors": state.get("errors", []),
        "citation_validation": state.get("citation_validation"),
        "claim_verification": state.get("claim_verification"),
        "generation_quality": state.get("generation_quality"),
        "evidence_quality_report": state.get("evidence_quality_report"),
        "evidence_scope": state.get("evidence_quality_report") or {},
        "related_work": public_related_work,
        "related_work_data": state.get("related_work_data") if draft_is_public else None,
        "introduction": public_introduction,
        "introduction_data": state.get("introduction_data") if draft_is_public else None,
        "citation_map": state.get("citation_map", {}),
        "citation_registry": state.get("citation_registry", {}),
        "literature_matrix": _build_literature_matrix(state),
        "research_plan": state.get("research_plan"),
        "research_semantic_frame": state.get("research_semantic_frame"),
        "search_branches": state.get("search_branches", []),
        "screening_protocol": state.get("screening_protocol"),
        "screening_report": state.get("screening_report"),
        "core_deliverables": state.get("core_deliverables", []),
        "user_paper_profile": state.get("user_paper_profile"),
        "search_report": state.get("search_report"),
        "search_result_quality": state.get("search_result_quality"),
        "theme_synthesis": state.get("theme_synthesis", []),
        "deliverable_readiness": state.get("deliverable_readiness", []),
        "writing_plans": state.get("writing_plans", []),
        "citation_allocation_plans": state.get("citation_allocation_plans", []),
        "deliverable_validation": state.get("deliverable_validation", []),
        "structure_validation": state.get("deliverable_validation", []),
        "planning_trace": {"writing_plans": state.get("writing_plans", [])},
        "writer_diagnostics": state.get("writer_diagnostics", []),
        "writer_section_diagnostics": state.get("writer_section_diagnostics", []),
        "deliverable_downgrades": state.get("deliverable_downgrades", []),
        "generation_readiness": state.get("generation_readiness"),
        "quality_gate": state.get("quality_gate"),
        "final_review_integrity": state.get("final_review_integrity"),
        "generation_blocked": state.get("generation_blocked", False),
        "draft_available": quality_gate.get("draft_available", bool(state.get("review"))),
        "draft_released": quality_gate.get("draft_released", draft_is_public),
        "draft_disposition": quality_gate.get("draft_disposition", "approved" if draft_is_public else "quarantined"),
        "unsupported_task_guard": state.get("unsupported_task_guard"),
        # Evaluation bundle fields
        "provisional_framework": state.get("provisional_framework", {}),
        "validated_routes": state.get("validated_routes", []),
        "route_decisions": state.get("route_decisions", []),
        "route_validation_report": state.get("route_validation_report", {}),
        "evidence_gap_report": state.get("evidence_gap_report", {}),
        "recovery_decision": state.get("recovery_decision", {}),
        "recovery_history": state.get("recovery_history", []),
        "recovery_statistics": state.get("recovery_statistics", {}),
        "route_recovery_progress": state.get("route_recovery_progress", {}),
        "route_evidence_deficits": state.get("route_evidence_deficits", []),
        "evidence_recovery_status": state.get("evidence_recovery_status"),
        "claim_plans": state.get("claim_plans", []),
        "claim_evidence_gate": state.get("claim_evidence_gate", {}),
        "claim_alignment": state.get("claim_alignment", {}),
        "claim_citation_consistency": state.get("claim_citation_consistency", {}),
        "global_evidence_gate": state.get("global_evidence_gate", {}),
        "reference_coverage_stats": state.get("reference_coverage_stats", {}),
        "quality_recovery_history": state.get("quality_recovery_history", []),
        "active_quality_recovery": state.get("active_quality_recovery"),
        "quality_recovery_decision": state.get("quality_recovery_decision"),
        "recovery_candidate_rejections": state.get("recovery_candidate_rejections", []),
    }
    # 只保存增量重生成真正需要的研究状态，避免下一轮重新检索和补全详情。
    output["research_state"] = {
        key: state.get(key)
        for key in (
            "user_query", "intent", "topic", "canonical_topic", "keywords", "core_keywords", "expanded_keywords", "keyword_batches", "scope_search_queries", "scope_query_roles", "required_concepts",
            "start_year", "end_year", "max_papers",
            "required_reference_count", "retrieval_target", "generation_limit",
            "max_papers_explicit", "year_range_explicit", "strict_year_range",
            "evidence_pool_target", "evidence_yield",
            "requested_sections", "language", "citation_style", "workflow",
            "core_deliverables", "user_paper_profile", "search_report", "search_result_quality",
            "research_request", "research_plan", "research_semantic_frame", "search_branches",
            "topic_interpretations", "selected_scope", "screening_protocol", "screening_report",
            "candidate_papers", "ranked_papers", "searched_keywords", "searched_query_windows",
            "source_diagnostics", "paper_details", "paper_cards", "pdf_paths", "parsed_papers",
            "clusters", "dynamic_taxonomy",
            "taxonomy_validation", "taxonomy_remediation", "theme_synthesis", "deliverable_readiness",
            "writing_plans", "citation_allocation_plan", "citation_allocation_plans", "deliverable_validation",
            "route_merge_diagnostics",
            "writer_diagnostics", "writer_section_diagnostics", "final_review_integrity",
            # WHY: 这些字段既是公开结果的一部分，也是同证据修复的事务快照。
            # 私有状态若不保存它们，候选退化后的回滚会丢失旧引用表和质量向量。
            "references", "reference_papers", "citation_map", "citation_registry",
            "citation_validation", "claim_verification", "generation_quality",
            "evidence_quality_report", "unique_cited_paper_count",
            "unique_valid_cited_paper_count", "final_requirement_met",
            "our_work", "background",
            "existing_limitations", "verified_results", "target_length",
            "unsupported_task_guard",
            "state_schema_version",
            "main_agent_context", "main_context_snapshot",
            "main_context_artifact_ref", "context_snapshot_version",
            "agent_task_results",
            "agent_orchestration_mode", "agent_operation_mode", "state_revision",
            "agent_mandatory_actions",
            "agent_execution_budget", "result_status", "clarification", "user_clarifications",
            "main_agent_decisions", "main_agent_rejections", "autonomous_verified_fingerprint",
            "session_id", "user_operation_sequence", "artifact_manifest",
            "generation_readiness", "quality_gate", "generation_blocked",
            "quality_recovery_attempts",
            "recovery_action_count", "quality_recovery_history",
            "active_quality_recovery", "quality_recovery_decision",
            "recovery_candidate_rejections", "allow_evidence_expansion",
            "reference_coverage_stats", "writing_version", "target_section_ids",
            "target_claim_ids",
            "section_checkpoints", "section_candidate_checkpoints",
            # 隔离正文和当前写作文本只进入可编辑研究状态；公开 body 仍由
            # quality_gate 的发布边界控制。
            "review", "quarantined_draft",
            "state_invariant_check",
            "best_effort_generation", "best_effort_on_failure",
            "best_effort_policy_source", "automatic_best_effort_attempted",
            "automatic_best_effort_generation", "allow_unvalidated_taxonomy",
            "draft_available", "draft_released", "draft_disposition",
            "forced_generation_issues",
            "contract_violations",
            # Evaluation
            "provisional_framework", "validated_routes", "route_decisions",
            "route_validation_report", "evidence_gap_report", "recovery_decision",
            "evidence_snapshot_version", "evidence_snapshot_fingerprint",
            "recovery_round", "route_recovery_attempts", "scope_revision_count",
            "route_recovery_progress", "route_evidence_deficits",
            "recovery_history", "recovery_statistics", "evidence_recovery_status",
            "claim_plans", "claim_evidence_gate", "claim_alignment",
            "claim_citation_consistency", "global_evidence_gate",
            "claim_verification_cache",
            # citation_shortfall_count 不导出：它是单次修复轮内部的临时
            # 信号，持久化后下一轮 continue 的 refine 会收到过期的
            # “引用缺口”误导指引。
            "citation_gap_repair_attempted",
        )
        if state.get(key) is not None
    }
    if output["research_state"].get("source_diagnostics"):
        output["research_state"]["source_diagnostics"] = [
            item.model_dump(mode="json") if hasattr(item, "model_dump") else item
            for item in output["research_state"]["source_diagnostics"]
        ]
    return output


def derive_result_status(state: Dict[str, Any]) -> str:
    """从实际执行状态集中推导公开结果状态，避免各入口各自误判。"""
    terminal = state.get("result_status")
    if terminal == "waiting_user":
        return "needs_clarification"
    if terminal == "partial":
        return "partial"
    if terminal in {"cancelled", "failed", "blocked"}:
        return terminal
    if state.get("planning_failed"):
        return "failed"
    if state.get("search_failed") and not state.get("candidate_papers"):
        return "failed"
    quality_gate = state.get("quality_gate") or {}
    if quality_gate.get("passed") is False:
        return "partial" if quality_gate.get("partial_success") else "blocked"
    if state.get("generation_blocked"):
        return "blocked"
    # 全局证据门：显式用户约束（引用数量/年份范围/同行评审）未满足 → partial；
    # 非显式的路线均衡缺口只记录提示，不改变结果状态。
    gate = state.get("global_evidence_gate") or {}
    if gate.get("status") == "EVALUATED" and gate.get("explicit_constraint_unmet"):
        return "partial"
    return "success"


def _restore_explicit_constraint_markers(state: ResearchAgentState) -> None:
    """从持久化研究请求恢复用户显式约束的标记。

    旧会话可能只保存了 ``required_reference_count``，却丢失
    ``max_papers_explicit``；数字仍在但门禁会把它当作默认值。研究请求是
    约束来源，恢复入口据此补回标记，不能从默认篇数反向猜测用户意图。
    """
    request = state.get("research_request") or {}
    # v2 增加恢复权限、共享预算和章节检查点。旧会话没有这些字段时采用
    # 保守且可继续的默认值；已有用户限制一律保留。
    if str(state.get("state_schema_version") or "1") != "2":
        legacy_recovery_count = (
            int(state.get("quality_recovery_attempts") or 0)
            + int(state.get("recovery_round") or 0)
        )
        state.setdefault("allow_evidence_expansion", True)
        state.setdefault("recovery_action_count", legacy_recovery_count)
        state.setdefault("quality_recovery_history", [])
        state.setdefault("section_checkpoints", {})
        state.setdefault("section_candidate_checkpoints", {})
        state.setdefault("writing_version", 0)
        state["state_schema_version"] = "2"
    explicit = request.get("max_papers_explicit")
    if explicit is not None:
        state["max_papers_explicit"] = bool(explicit)
    if explicit is True:
        # 旧版本输出可能只在 research_request 中保留了数值；保持顶层门禁
        # 与请求一致，避免恢复时因为字段缺失而跳过显式篇数检查。
        for key in ("required_reference_count", "max_papers"):
            if state.get(key) is None and request.get(key) is not None:
                state[key] = request[key]


@research_budget
def continue_research_agent(
    research_state: Dict[str, Any],
    should_cancel: Callable[[], bool] | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> Dict[str, Any]:
    """在既有证据池上增量检索并重生成，不重复规划和既有证据抽取。"""
    state: ResearchAgentState = dict(research_state)
    existing_details = list(state.get("paper_details") or [])
    existing_cards = list(state.get("paper_cards") or [])
    if not existing_details and not state.get("candidate_papers"):
        raise ValueError("当前会话没有可恢复的论文证据，请重新提交研究请求")

    state["candidate_papers"] = list(
        state.get("candidate_papers") or existing_details
    )
    state["ranked_papers"] = list(
        state.get("ranked_papers") or state["candidate_papers"]
    )
    state["paper_details"] = existing_details
    state["paper_cards"] = existing_cards
    state["incremental_retrieval"] = True
    state["incremental_search_new_candidates"] = 0
    state["steps"] = []
    state["errors"] = []
    _restore_explicit_constraint_markers(state)
    # 与 run 主路径共用同一清理清单：旧版列表缺 claim_plans/writing_plans 等，
    # 上一轮的失效授权会残留进 writer（见 _GENERATION_PRODUCT_KEYS 注释）。
    _reset_generation_products(state)

    normalize_orchestration_mode(state)
    state["agent_operation_mode"] = "incremental"
    from app.agent.generation_recovery import active_focus_recovery_targets

    if active_focus_recovery_targets(state):
        # WHY: 质量恢复阶梯已决定执行重点补检索；主 Agent 的路线检索动作
        # 不生成重点查询。先履行已选定动作，再交回主 Agent 决策。
        _run_search_subagent(
            state,
            objective="针对缺失的用户研究重点执行增量检索与重新排序",
            llm=_role_llm(_get_llm(), "search"),
            should_cancel=should_cancel,
            progress_callback=progress_callback,
            total_steps=max(1, int(get_settings().agent_main_max_rounds)),
        )
    return _run_autonomous_pipeline(
        state,
        should_cancel=should_cancel,
        progress_callback=progress_callback,
    )



def _build_literature_matrix(state: ResearchAgentState) -> List[Dict[str, Any]]:
    """构造前端可直接展示和导出的文献矩阵。"""
    citation_map = state.get("citation_map") or {}
    return [
        {
            "reference_number": citation_map.get(str(card.get("paper_id") or "")),
            **{
                key: card.get(key)
                for key in (
                    "paper_id", "authors", "year", "title", "venue", "doi", "url",
                    "publication_type", "peer_review_status", "evidence_level",
                    "research_problem", "study_design", "sample_size", "behavior_categories",
                    "data_modalities", "method", "dataset", "metrics", "results",
                    "limitations", "relation_type",
                    "evidence_state", "unsupported_fields", "quality_status", "quality_issues",
                )
            },
        }
        for card in (state.get("paper_cards") or [])
    ]


@research_budget
def regenerate_research_agent(
    research_state: Dict[str, Any],
    should_cancel: Callable[[], bool] | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> Dict[str, Any]:
    """基于编辑后的论文集合增量重生成，不重复检索、排序和详情补全。
    
    Args:
        research_state: 持久化的研究状态（来自 output["research_state"]）。
        should_cancel: 取消检查回调函数，返回 True 时停止执行。
        progress_callback: 进度更新回调函数，接收 (step_name, current, total)。
    
    Returns:
        包含 answer / steps / references / paper_cards 的结果字典。
        
    Raises:
        AgentCancelledError: 任务被取消时抛出。
        ValueError: 如果删除后没有剩余论文。
    """
    state: ResearchAgentState = dict(research_state)
    state["steps"] = []
    state["errors"] = []
    _restore_explicit_constraint_markers(state)
    verification_only_recovery = bool(state.pop("verification_only_recovery", False))
    normalize_orchestration_mode(state)
    state["agent_operation_mode"] = (
        "verification_only" if verification_only_recovery else "regeneration"
    )
    local_verification = None
    if not verification_only_recovery:
        recovery_plan = _prepare_autonomous_regeneration(
            state, should_cancel=should_cancel, progress_callback=progress_callback,
        )
        if recovery_plan["mode"] == "local_rewrite":
            local_verification = {
                "target_sentence_indices": recovery_plan["target_sentence_indices"],
                "target_claim_ids": recovery_plan["target_claim_ids"],
                "verification_scope": {
                    "mode": "local",
                    "previous_report": recovery_plan["previous_claim_verification"],
                },
            }
        if (state.get("active_quality_recovery") or state.get("best_effort_generation")
                or state.get("force_taxonomy_remediation")):
            # WHY: 质量恢复阶梯或用户已选定重写/最佳努力，主 Agent 不能先
            # 报告终止而跳过该动作。必做步骤仍由同一 MainAgentLoop 的注册
            # 动作、任务权限、预算和提交版本校验执行。
            mandatory: list[dict[str, Any]] = []
            if recovery_plan["mode"] == "full_rebuild" and state.get("paper_cards"):
                if state.get("force_taxonomy_remediation"):
                    # WHY: 用户已选“重新分类”时旧分类已被清除，必须先真实运行
                    # cluster_papers；模型直接结束会让本轮修复成为空动作。
                    mandatory.append({"action": "cluster_papers"})
                elif {"research_status", "related_work", "narrative_review"}.intersection(
                    state.get("core_deliverables") or []
                ):
                    mandatory.append({"action": "validate_routes"})
                mandatory.append({"action": "plan_claims"})
            if state.get("claim_plans") or any(item["action"] == "plan_claims" for item in mandatory):
                targets = list(state.get("target_section_ids") or [])
                if recovery_plan["mode"] == "local_rewrite" and targets and state.get("writing_plans") and state.get("review"):
                    mandatory.append({"action": "rewrite_sections", "arguments": {"section_ids": targets}})
                else:
                    mandatory.append({"action": "generate_deliverables"})
            if mandatory:
                state["agent_mandatory_actions"] = mandatory
    return _run_autonomous_pipeline(
        state,
        should_cancel=should_cancel,
        progress_callback=progress_callback,
        local_verification=local_verification,
    )
