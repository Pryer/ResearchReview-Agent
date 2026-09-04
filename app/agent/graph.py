"""Agent 工作流编排。

MVP 阶段使用顺序节点风格，不引入 LangGraph 依赖。
``run_research_agent`` 是主入口函数。
后续迁移到 LangGraph 时，只需重写本文件的编排逻辑，
节点函数保持不变。

执行原语（协作式取消、节点边界检查、LLM 工厂）在 ``app.agent.execution``，
检索精化循环在 ``app.agent.retrieval_loop``，证据恢复状态机在
``app.agent.recovery_loop``；本文件只保留编排顺序与分支路由。
"""

from __future__ import annotations

import os
import re

from typing import Any, Callable, Dict, List, Optional

from app.agent.execution import AgentCancelledError
from app.agent.execution import checkpoint as _checkpoint
from app.agent.execution import get_llm as _get_llm
from app.agent.nodes import (
    append_step,
    citation_check_node,
    claim_evidence_gate_node,
    claim_plan_node,
    cluster_node,
    diagnose_evidence_gaps_node,
    download_pdf_node,
    expand_search_year_node,
    fetch_detail_node,
    final_answer_node,
    extract_card_node,
    generate_deliverables_node,
    global_evidence_gate_node,
    parse_pdf_node,
    plan_node,
    rank_node,
    recovery_controller_node,
    refine_search_node,
    retrieval_shortfall_node,
    search_node,
    scope_revision_node,
    validate_routes_node,
    verify_claims_node,
)
from app.agent.recovery_loop import (
    run_route_evidence_recovery as _run_route_evidence_recovery,
    _record_recovery_statistics,
)
from app.agent.retrieval_loop import (
    search_rank_with_refinement as _search_rank_with_refinement,
)
from app.agent.router import should_parse_pdf
from app.agent.state import ResearchAgentState
from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)


def _compute_total_steps(state: ResearchAgentState) -> int:
    """根据当前已知的意图和配置，动态估算本次运行的总步骤数。

    步骤索引与 ``run_research_agent`` 中的 ``_checkpoint`` 调用保持一致：
    0 plan, 1 search_and_rank, 2 rank_papers, 3 refine_search,
    4 expand_search_year, 5 fetch_detail, 6 download_pdf, 7 parse_pdf,
    8 extract_paper_cards, 9 cluster_papers, 10 generate_deliverables,
    11 verify_claims, 12 citation_check, 13 final_answer。

    在 plan 节点执行前，``intent`` / ``core_deliverables`` 尚未知晓，本函数
    返回一个保守估计；plan 完成后应重新调用本函数以获得更准确的总数。
    这只影响进度条的分母展示，不影响任何执行分支逻辑。
    """
    settings = get_settings()
    intent = state.get("intent")

    # 基础步骤：plan + search_and_rank + rank_papers + refine_search
    #           + expand_search_year + final_answer
    total = 6

    if intent == "search_papers":
        # 只查论文：检索排序后直接输出，跳过详情补全和之后所有生成步骤。
        return total

    # fetch_detail 在非 search_papers 意图下总会执行。
    total += 1  # fetch_detail

    if settings.enable_pdf_pipeline:
        total += 1  # download_pdf
        # parse_pdf 是否运行取决于 should_parse_pdf(state)，规划阶段无法
        # 100% 确定；按"预计会解析"乐观计入，最坏情况下总数略偏高，
        # 不影响进度条单调递增。
        total += 1  # parse_pdf

    core_deliverables = state.get("core_deliverables") or []
    if core_deliverables:
        total += 1  # extract_paper_cards（乐观估计会检索到论文详情）

        taxonomy_deliverables = {"research_status", "related_work", "narrative_review"}
        if taxonomy_deliverables.intersection(core_deliverables):
            total += 1  # validate_routes（有路线候选时必经的验证检查点）
            total += 1  # cluster_papers（无证据路线时的回退聚类）
            if settings.enable_evidence_recovery:
                total += 1  # evidence_recovery（内部是有界复合步骤）

        total += 1  # global_evidence_gate（paper_details 为空时不实际执行）
        total += 2  # claim_plan + claim_evidence_gate（两个独立检查点）
        total += 1  # generate_deliverables
        total += 1  # claim_alignment（生成成功后的越权检查，乐观计入）

        if settings.enable_claim_verification:
            total += 1  # verify_claims
        total += 1  # citation_check

    return total


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
        "state_schema_version": "1",
        "steps": [],
        "errors": [],
    }
    
    # 合并用户传入的初始状态（如 our_work, background 等）
    if initial_state:
        state.update(initial_state)
        # 这些字段定义一次执行的身份与审计边界，不能被初始上下文覆盖。
        state["user_query"] = user_query
        state["state_schema_version"] = "1"
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

    # plan 节点执行前，intent/core_deliverables 未知，先给一个保守估计；
    # plan 成功后会重新计算更准确的总数（见下方）。
    total_steps = _compute_total_steps(state)

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
    plan_node(state, llm=_get_llm(), current_year=current_year)
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

    # plan 完成后 intent/core_deliverables 已知，重新计算总步骤数，
    # 让进度条分母更准确（仅影响展示，不影响执行分支）。
    total_steps = _compute_total_steps(state)

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

    # ---------- 2-7. ReAct 检索 → 排序 → 关键词修正 ----------
    _checkpoint(state, "search_and_rank", 1, total_steps, should_cancel, progress_callback)
    _search_rank_with_refinement(
        state,
        llm=_get_llm(),
        should_cancel=should_cancel,
        progress_callback=progress_callback,
        total_steps=total_steps,
    )
    if state.get("search_failed") and not state.get("candidate_papers"):
        state["answer"] = (
            "## 论文检索失败\n\n"
            "所有论文数据源均未返回结果，可能发生接口限流或网络异常。"
            "本次请求已停止，未据此判断相关论文数量，请稍后重试。"
        )
        append_step(
            state,
            "final_answer",
            "failed",
            error="paper_search_sources_unavailable",
            duration_ms=0,
        )
        return _build_output(state)
    # 检索精化循环内部已用了 0-3 号步骤位（plan/search_and_rank/rank_papers/
    # refine_search），expand_search_year 固定占第 4 号；后续步骤位不再硬编码
    # 绝对序号，改用递增计数器，避免 total_steps 动态变化后与固定索引脱节
    # （例如 search_papers 分支下 total_steps 远小于 14，若仍写死 13 会导致
    # current > total，进度条超过 100%）。
    step_idx = 4
    _checkpoint(state, "expand_search_year", step_idx, total_steps, should_cancel, progress_callback)
    expand_search_year_node(state, should_cancel=should_cancel)
    step_idx += 1

    # 只查论文：检索和排序后即可返回
    if state.get("intent") == "search_papers":
        _checkpoint(state, "final_answer", step_idx, total_steps, should_cancel, progress_callback)
        final_answer_node(state)
        return _build_output(state)

    _checkpoint(state, "fetch_detail", step_idx, total_steps, should_cancel, progress_callback)
    fetch_detail_node(state, should_cancel=should_cancel)
    step_idx += 1

    # ---------- 8-9. PDF 分支 ----------
    if get_settings().enable_pdf_pipeline:
        _checkpoint(state, "download_pdf", step_idx, total_steps, should_cancel, progress_callback)
        download_pdf_node(state, should_cancel=should_cancel)
        step_idx += 1
        if should_parse_pdf(state):
            _checkpoint(state, "parse_pdf", step_idx, total_steps, should_cancel, progress_callback)
            parse_pdf_node(state, should_cancel=should_cancel)
            step_idx += 1
    else:
        append_step(
            state,
            "download_pdf",
            "success",
            tool_name="download_pdf",
            input_data={
                "paper_details": len(state.get("paper_details") or []),
                "enable_pdf_pipeline": get_settings().enable_pdf_pipeline,
            },
            output_data={"skipped": True, "reason": "pdf_pipeline_explicitly_disabled"},
            duration_ms=0,
        )

    # ---------- 10. Evidence Card 与路线验证 ----------
    if state.get("paper_details"):
        _checkpoint(state, "extract_paper_cards", step_idx, total_steps, should_cancel, progress_callback)
        card_llm = _get_llm() if get_settings().enable_llm_card_extraction else None
        extract_card_node(state, llm=card_llm, should_cancel=should_cancel)
        step_idx += 1
        taxonomy_deliverables = {"research_status", "related_work", "narrative_review"}
        if taxonomy_deliverables.intersection(state.get("core_deliverables") or []):
            _checkpoint(state, "validate_routes", step_idx, total_steps, should_cancel, progress_callback)
            # 当有候选路线时走验证路径（KEEP/MERGE/SPLIT/DROP）；
            # 否则回退到原始无约束聚类
            validate_routes_node(state, llm=_get_llm())
            step_idx += 1
            # 候选路线验证后先执行有界证据恢复；不能先聚类回退，否则被 DROP
            # 的路线已经失去补搜机会。
            if state.get("provisional_framework") and get_settings().enable_evidence_recovery:
                _checkpoint(
                    state,
                    "evidence_recovery",
                    step_idx,
                    total_steps,
                    should_cancel,
                    progress_callback,
                )
                _run_route_evidence_recovery(
                    state,
                    should_cancel=should_cancel,
                )
                step_idx += 1
            # WEAK 路线会被保留给 Recovery，但没有任何已匹配证据时不能直接
            # 编译成正式写作结构；恢复耗尽后才回退到证据驱动分类。
            evidence_backed_routes = [
                route for route in (state.get("validated_routes") or [])
                if route.get("paper_ids")
            ]
            if not evidence_backed_routes:
                _checkpoint(state, "cluster_papers", step_idx, total_steps, should_cancel, progress_callback)
                cluster_llm = _get_llm() if get_settings().enable_llm_clustering else None
                cluster_node(state, llm=cluster_llm)
                step_idx += 1
        else:
            state["clusters"] = []
            state["dynamic_taxonomy"] = {}
            state["taxonomy_validation"] = {}
            state["validated_routes"] = []
            state["route_decisions"] = []

    # ---------- 10.4. Claim-Evidence Planning ----------
    # 必须早于 Global Evidence Gate：门禁的主张强度指标读 ``claim_plans``，
    # 反序会让它每次都落到路线体量回退值（实测 claim_support_proxy 恒为
    # 1.0，而同一次运行里 claim_plan 统计的是 141/149 条主张仅有单篇证据）。
    if (
        any(route.get("paper_ids") for route in (state.get("validated_routes") or []))
        or state.get("dynamic_taxonomy")
    ):
        _checkpoint(state, "claim_plan", step_idx, total_steps, should_cancel, progress_callback)
        claim_plan_node(state, llm=_get_llm())
        step_idx += 1
        _checkpoint(
            state,
            "claim_evidence_gate",
            step_idx,
            total_steps,
            should_cancel,
            progress_callback,
        )
        _run_claim_evidence_gate(state)
        step_idx += 1

    # ---------- 10.5. Global Evidence Gate（综述级证据充分性，只评估与推荐）----------
    if state.get("paper_details") and get_settings().enable_global_evidence_gate:
        _checkpoint(
            state,
            "global_evidence_gate",
            step_idx,
            total_steps,
            should_cancel,
            progress_callback,
        )
        global_evidence_gate_node(state)
        step_idx += 1

    # ---------- 11-13. 四交付物统一写作分支 ----------
    papers = state.get("paper_details") or []
    if state.get("core_deliverables"):
        if not papers:
            retrieval_shortfall_node(state)
        else:
            _checkpoint(state, "generate_deliverables", step_idx, total_steps, should_cancel, progress_callback)
            _generate_deliverables_or_block(state, should_cancel=should_cancel)
            step_idx += 1

            def _advance_verification(stage: str) -> None:
                nonlocal step_idx
                _checkpoint(state, stage, step_idx, total_steps, should_cancel, progress_callback)
                step_idx += 1

            # 引用授权一致性推迟到引用缺口修复之后：留在正文里的必须是最终草稿。
            _verify_generated_draft(
                state,
                checkpoint=_advance_verification,
                check_citation_authorization=False,
            )
            # ---------- 13.5. 引用缺口修复：成文引用数低于用户硬性要求时定向扩召回 ----------
            # “不少于 N 篇”是显式硬约束，不能只靠横幅提示兜底：以增量检索
            # 模式补一轮检索（refine 反馈携带缺口数量）并重新生成，修复后
            # 若引用数反而变少则回滚到修复前草稿。
            if _should_repair_citation_gap(state):
                cited_before = int(state.get("unique_cited_paper_count") or 0)
                required_refs = int(state.get("required_reference_count") or 0)
                support_before = float(
                    (state.get("claim_verification") or {}).get("support_rate") or 0.0
                )
                logger.info(
                    "Citation gap repair: cited=%d < required=%d, running incremental retrieval",
                    cited_before, required_refs,
                )
                repair_failed = False
                try:
                    _repair_citation_gap(
                        state, should_cancel, progress_callback, step_idx, total_steps
                    )
                except AgentCancelledError:
                    raise
                except Exception as repair_exc:  # noqa: BLE001
                    # 修复轮失败不能让整个任务崩掉：此时原稿已被 pop、快照还在，
                    # 必须恢复修复前草稿并正常收尾，否则连回滚分支都到不了。
                    repair_failed = True
                    state.setdefault("errors", []).append(
                        f"citation_gap_repair: {repair_exc}"
                    )
                    logger.warning(
                        "Citation gap repair failed; restoring pre-repair draft: %s",
                        repair_exc,
                    )
                    _restore_pre_repair_snapshot(state, clear_incremental=True)
                if repair_failed:
                    append_step(
                        state,
                        "citation_gap_repair",
                        "failed",
                        input_data={"required_reference_count": required_refs, "cited_before": cited_before},
                        output_data={"restored_pre_repair_draft": True},
                        error="citation_gap_repair_failed_restored_draft",
                        duration_ms=0,
                    )
                else:
                    # 修复产生的新草稿需要重新走写作后验证链。
                    def _mark_repair_verification(stage: str) -> None:
                        # 修复轮沿用原步骤位并钳制在分母内，也不为对齐步骤单独
                        # 报进度：进度条只前进不回跳，索引语义与首轮保持一致。
                        if stage == "claim_alignment":
                            return
                        _checkpoint(
                            state, stage, min(step_idx, max(total_steps - 1, 0)),
                            total_steps, should_cancel, progress_callback,
                        )

                    _verify_generated_draft(
                        state,
                        checkpoint=_mark_repair_verification,
                        check_citation_authorization=False,
                    )
                    cited_after = int(state.get("unique_cited_paper_count") or 0)
                    support_after = float(
                        (state.get("claim_verification") or {}).get("support_rate") or 0.0
                    )
                    append_step(
                        state,
                        "citation_gap_repair",
                        "success" if cited_after >= required_refs else "partial",
                        input_data={"required_reference_count": required_refs, "cited_before": cited_before},
                        output_data={"cited_after": cited_after, "repaired": cited_after > cited_before},
                        duration_ms=0,
                    )
                    # 回滚采用帕累托规则：只有引用数与支持率同时退化才回滚。
                    # 引用略减但主张支持率明显提升的修复版本仍是更好的草稿。
                    if cited_after < cited_before and support_after <= support_before:
                        _restore_pre_repair_snapshot(state)
                        logger.info(
                            "Citation gap repair regressed (cited %d -> %d, support %.1f%% -> %.1f%%); rolled back draft",
                            cited_before, cited_after, support_before * 100, support_after * 100,
                        )
                    else:
                        state.pop("_citation_gap_repair_snapshot", None)
                        logger.info(
                            "Citation gap repair kept (cited %d -> %d, support %.1f%% -> %.1f%%)",
                            cited_before, cited_after, support_before * 100, support_after * 100,
                        )
            # Claim-Citation Consistency: 引用的论文是否在 claim 的允许证据中
            _check_claim_citation_consistency(state)

    # ---------- 14. 最终输出 ----------
    _checkpoint(state, "final_answer", min(step_idx, total_steps - 1) if total_steps > 0 else step_idx, total_steps, should_cancel, progress_callback)
    final_answer_node(state)
    if progress_callback:
        progress_callback("completed", total_steps, total_steps)

    # 标准化诊断输出 + 自动导出 Evaluation Bundle
    try:
        from app.agent.diagnostics import format_diagnostics, export_evaluation_bundle
        diag_text = format_diagnostics(state)
        logger.info("Diagnostics:\n%s", diag_text)
        from datetime import datetime as _dt
        ts = _dt.now().strftime("%Y%m%d_%H%M%S")
        bundle_path = export_evaluation_bundle(
            state, output_dir=os.path.join("data", "eval_bundles", f"eval_bundle_{ts}")
        )
        logger.info("Evaluation bundle exported: %s", bundle_path)
    except Exception as exc:
        logger.debug("Diagnostics/export skipped: %s", exc)

    logger.info("Agent finished: %d steps, %d errors",
                len(state.get("steps", [])), len(state.get("errors", [])))

    return _build_output(state)


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
    section_ids = {
        str(item.get("section_id") or "")
        for run in section_diagnostics if isinstance(run, dict)
        for item in (run.get("sections") or [])
        if isinstance(item, dict) and item.get("section_id")
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
    }


def _build_regeneration_recovery_plan(state: ResearchAgentState) -> Dict[str, Any]:
    """根据上一轮门禁代码决定保守重写可复用的中间产物。"""
    issue_codes = {
        str(item.get("code") or "")
        for item in (state.get("quality_gate") or {}).get("blocking_issues") or []
        if isinstance(item, dict) and item.get("code")
    }
    targets = _derive_local_verification_targets(state)
    same_evidence_local_rewrite = bool(
        state.get("conservative_regeneration")
        and issue_codes
        and issue_codes.issubset(_LOCAL_REWRITE_ISSUE_CODES)
        and state.get("validated_routes")
        and state.get("claim_plans")
        and not state.get("force_taxonomy_remediation")
    )
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
        "previous_claim_verification": state.get("claim_verification") if same_evidence_local_rewrite else None,
    }


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
        generate_deliverables_node(state, llm=_get_llm(), should_cancel=should_cancel)
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


def _should_repair_citation_gap(state: ResearchAgentState) -> bool:
    """判断是否需要为引用缺口补一轮定向检索。

    仅当用户显式要求了引用数量、成文后实际引用数严格低于要求（且已有
    部分引用）、且尚未尝试过修复时触发；被阻断的生成或无证据池可补的
    会话不进入修复。
    """
    if state.get("generation_blocked"):
        return False
    if state.get("citation_gap_repair_attempted"):
        return False
    if not state.get("max_papers_explicit", False):
        return False
    required = int(state.get("required_reference_count") or 0)
    if required <= 0:
        return False
    cited = int(state.get("unique_cited_paper_count") or 0)
    if not (0 < cited < required):
        return False
    if not (state.get("candidate_papers") or state.get("paper_details")):
        return False
    return True


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
            route_llm = _get_llm()
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
            claim_plan_node(state, llm=_get_llm())
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
    output = {
        "status": output_status,
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
    }
    # 只保存增量重生成真正需要的研究状态，避免下一轮重新检索和补全详情。
    output["research_state"] = {
        key: state.get(key)
        for key in (
            "user_query", "intent", "topic", "canonical_topic", "keywords", "core_keywords", "expanded_keywords", "keyword_batches", "scope_search_queries", "scope_query_roles", "required_concepts",
            "start_year", "end_year", "max_papers",
            "required_reference_count", "retrieval_target", "generation_limit",
            "evidence_pool_target", "evidence_yield",
            "requested_sections", "language", "citation_style", "workflow",
            "core_deliverables", "user_paper_profile", "search_report",
            "research_request", "research_plan", "research_semantic_frame", "search_branches",
            "topic_interpretations", "selected_scope", "screening_protocol", "screening_report",
            "candidate_papers", "ranked_papers", "searched_keywords", "searched_query_windows",
            "source_diagnostics", "paper_details", "paper_cards", "pdf_paths",
            "clusters", "dynamic_taxonomy",
            "taxonomy_validation", "taxonomy_remediation", "theme_synthesis", "deliverable_readiness",
            "writing_plans", "citation_allocation_plans", "deliverable_validation",
            "route_merge_diagnostics",
            "writer_diagnostics", "writer_section_diagnostics", "final_review_integrity",
            "our_work", "background",
            "existing_limitations", "verified_results", "target_length",
            "unsupported_task_guard",
            "state_schema_version",
            "generation_readiness", "quality_gate", "quality_recovery_attempts",
            "state_invariant_check",
            "best_effort_generation", "allow_unvalidated_taxonomy",
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
    # 与 run 主路径共用同一清理清单：旧版列表缺 claim_plans/writing_plans 等，
    # 上一轮的失效授权会残留进 writer（见 _GENERATION_PRODUCT_KEYS 注释）。
    _reset_generation_products(state)

    settings = get_settings()
    total = 13 + (2 if settings.enable_pdf_pipeline else 0)
    step_idx = 0
    _checkpoint(state, "incremental_search", step_idx, total, should_cancel, progress_callback)
    _search_rank_with_refinement(
        state,
        llm=_get_llm(),
        should_cancel=should_cancel,
        progress_callback=progress_callback,
        total_steps=total,
    )
    # 检索闭环内部使用 1-3 号进度位；后续从 4 继续，保持进度单调。
    step_idx = 4
    if state.get("search_failed") and not state.get("candidate_papers"):
        state["answer"] = (
            "## 增量检索失败\n\n新增检索范围的数据源均未返回结果；"
            "已保存的前轮论文和证据未被删除。"
        )
        return _build_output(state)

    _checkpoint(state, "incremental_fetch_detail", step_idx, total, should_cancel, progress_callback)
    fetch_detail_node(state, should_cancel=should_cancel)
    step_idx += 1

    if settings.enable_pdf_pipeline:
        _checkpoint(state, "incremental_download_pdf", step_idx, total, should_cancel, progress_callback)
        download_pdf_node(state, should_cancel=should_cancel)
        step_idx += 1
        if should_parse_pdf(state):
            _checkpoint(state, "incremental_parse_pdf", step_idx, total, should_cancel, progress_callback)
            parse_pdf_node(state, should_cancel=should_cancel)
            step_idx += 1

    _checkpoint(state, "incremental_extract_cards", step_idx, total, should_cancel, progress_callback)
    card_llm = _get_llm() if settings.enable_llm_card_extraction else None
    extract_card_node(state, llm=card_llm, should_cancel=should_cancel)
    step_idx += 1

    if {"research_status", "related_work", "narrative_review"}.intersection(
        state.get("core_deliverables") or []
    ):
        _checkpoint(state, "revalidate_routes", step_idx, total, should_cancel, progress_callback)
        route_llm = _get_llm()
        validate_routes_node(state, llm=route_llm)
        if state.get("provisional_framework") and settings.enable_evidence_recovery:
            _run_route_evidence_recovery(state, should_cancel=should_cancel)
        if not any(
            route.get("paper_ids") for route in (state.get("validated_routes") or [])
        ):
            cluster_llm = route_llm if settings.enable_llm_clustering else None
            cluster_node(state, llm=cluster_llm)
        step_idx += 1

    if (
        any(route.get("paper_ids") for route in (state.get("validated_routes") or []))
        or state.get("dynamic_taxonomy")
    ):
        _checkpoint(state, "claim_plan", step_idx, total, should_cancel, progress_callback)
        claim_plan_node(state, llm=_get_llm())
        _run_claim_evidence_gate(state)
        step_idx += 1

    # 增量恢复后重新评估综述级证据充分性（覆盖前轮结果）。
    # 排在 claim_plan 之后：门禁的主张强度指标读 claim_plans。
    if state.get("paper_details") and settings.enable_global_evidence_gate:
        _checkpoint(state, "global_evidence_gate", step_idx, total, should_cancel, progress_callback)
        global_evidence_gate_node(state)
        step_idx += 1

    _checkpoint(state, "regenerate_content", step_idx, total, should_cancel, progress_callback)
    _generate_deliverables_or_block(state, should_cancel=should_cancel)
    step_idx += 1

    def _advance_verification(stage: str) -> None:
        nonlocal step_idx
        # 增量路径此前不为对齐步骤单独报进度，保持既有进度分母不变。
        if stage == "claim_alignment":
            return
        _checkpoint(state, stage, step_idx, total, should_cancel, progress_callback)
        step_idx += 1

    _verify_generated_draft(state, checkpoint=_advance_verification)
    _checkpoint(state, "final_answer", step_idx, total, should_cancel, progress_callback)
    final_answer_node(state)
    state.pop("incremental_retrieval", None)
    state.pop("incremental_search_window", None)
    if progress_callback:
        progress_callback("completed", total, total)
    return _build_output(state)


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
    recovery_llm_calls_before = None
    try:
        from app.core.metrics import get_metrics_collector
        recovery_llm_calls_before = get_metrics_collector().get_token_report().get("total_calls", 0)
    except Exception:
        recovery_llm_calls_before = None
    recovery_plan = _build_regeneration_recovery_plan(state)
    reusable_products = {
        key: state.get(key)
        for key in (
            "theme_synthesis", "claim_plans", "claim_evidence_gate",
            "global_evidence_gate",
        )
        if state.get(key) is not None
    }
    # 与 run/continue 共用同一清理清单：旧版列表缺 claim_plans/writing_plans、
    # related_work/introduction 等，上一轮产物会残留进输出与 writer。
    _reset_generation_products(state)
    if recovery_plan["reuse_claim_plans"]:
        # WHY: 这里仅复用同一证据池上的授权计划；写作计划、引用分配、正文和
        # 验证结果仍全部重建。证据池编辑或检索缺口会落入 full_rebuild。
        state.update(reusable_products)
    total = 5

    append_step(
        state,
        "regeneration_recovery_plan",
        "success",
        input_data={"blocking_issue_codes": recovery_plan["issue_codes"]},
        output_data=recovery_plan,
        duration_ms=0,
    )

    _checkpoint(state, "rebuild_routes", 0, total, should_cancel, progress_callback)
    if not recovery_plan["reuse_routes"] and {"research_status", "related_work", "narrative_review"}.intersection(
        state.get("core_deliverables") or []
    ):
        route_llm = _get_llm()
        validate_routes_node(state, llm=route_llm)
        if not any(
            route.get("paper_ids") for route in (state.get("validated_routes") or [])
        ):
            cluster_llm = route_llm if get_settings().enable_llm_clustering else None
            cluster_node(state, llm=cluster_llm)
    papers = state.get("paper_details") or []
    if not papers:
        raise ValueError("删除后没有剩余论文，无法重新生成")

    if not recovery_plan["reuse_claim_plans"] and (
        any(route.get("paper_ids") for route in (state.get("validated_routes") or []))
        or state.get("dynamic_taxonomy")
    ):
        claim_plan_node(state, llm=_get_llm())
        _run_claim_evidence_gate(state)

    # 基于编辑后的论文池重新评估综述级证据充分性（覆盖前轮结果）。
    # 排在 claim_plan 之后：门禁的主张强度指标读 claim_plans。
    if (
        state.get("paper_details")
        and get_settings().enable_global_evidence_gate
        and not recovery_plan["reuse_global_evidence_gate"]
    ):
        global_evidence_gate_node(state)

    _checkpoint(state, "regenerate_content", 1, total, should_cancel, progress_callback)
    _generate_deliverables_or_block(state, should_cancel=should_cancel)

    # 重生成入口的步骤位固定（总步数恒为 5），因此按阶段名查表而不是递增。
    _REGENERATION_STEP_INDEX = {"verify_claims": 2, "citation_check": 3}

    def _mark_verification(stage: str) -> None:
        index = _REGENERATION_STEP_INDEX.get(stage)
        if index is None:
            return
        _checkpoint(state, stage, index, total, should_cancel, progress_callback)

    _verify_generated_draft(
        state,
        checkpoint=_mark_verification,
        # 局部重写只重算受影响主张，其余句子由上一轮报告与指纹缓存复用。
        verify_claims_kwargs=(
            {
                "target_sentence_indices": recovery_plan["target_sentence_indices"],
                "target_claim_ids": recovery_plan["target_claim_ids"],
                "verification_scope": {
                    "mode": "local",
                    "previous_report": recovery_plan["previous_claim_verification"],
                },
            }
            if recovery_plan["mode"] == "local_rewrite"
            else None
        ),
    )
    _checkpoint(state, "final_answer", 4, total, should_cancel, progress_callback)
    final_answer_node(state)
    _record_recovery_statistics(
        state,
        reused_claims=(
            sum(int(plan.get("total_claims") or 0) for plan in (state.get("claim_plans") or []))
            if recovery_plan["reuse_claim_plans"] else 0
        ),
        recomputed_claims=0 if recovery_plan["reuse_claim_plans"] else sum(
            int(plan.get("total_claims") or 0) for plan in (state.get("claim_plans") or [])
        ),
        llm_calls_before=recovery_llm_calls_before,
    )
    if progress_callback:
        progress_callback("completed", total, total)
    return _build_output(state)
