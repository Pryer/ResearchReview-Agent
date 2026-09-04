"""规划与范围相关节点。"""

from __future__ import annotations

import json
import math
import re
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from app.agent.decorators import node, optional, provides, requires
from app.agent.nodes.base import (
    _compact_debug_value,
    _latest_step,
    _needs_current_time_tool,
    _paper_debug_item,
    _paper_identity_key,
    _preview_text,
    _select_branch_diverse_keywords,
    _select_search_keywords,
    _summarize_papers,
    append_step,
)
from app.core.config import get_settings
from app.core.logger import get_logger
from app.schemas.paper_schema import SourceDiagnostic

if TYPE_CHECKING:
    from app.agent.state import ResearchAgentState

logger = get_logger(__name__)

_YEAR_RANGE_RE = re.compile(
    r"(?<!\d)(?:19|20)\d{2}\s*(?:[-~—–]|到|至)\s*(?:19|20)\d{2}(?!\d)"
)


def _synchronize_semantic_frame_time_window(
    semantic_frame: Any,
    start_year: int,
    end_year: int,
) -> Any:
    """以最终槽位年份刷新语义帧中的叙述性年份快照。"""
    from app.schemas.research_plan_schema import ResearchSemanticFrame

    frame = ResearchSemanticFrame.model_validate(semantic_frame)
    data = frame.model_dump(mode="json")
    authoritative = f"{int(start_year)}-{int(end_year)}"

    def refresh(value: str) -> str:
        return _YEAR_RANGE_RE.sub(authoritative, str(value or ""))

    terminal = dict(data.get("terminal_goal") or {})
    terminal["description"] = refresh(terminal.get("description") or "")
    data["terminal_goal"] = terminal
    for key in ("secondary_goals", "task_chain", "assumptions"):
        data[key] = [refresh(item) for item in data.get(key) or []]
    return ResearchSemanticFrame.model_validate(data)


@node(name="plan", category="planning", description="解析用户需求，生成检索计划")
@requires("user_query")
@provides(
    "intent", "confidence", "topic", "canonical_topic", "keywords", 
    "core_keywords", "expanded_keywords",
    "search_branches", "scope_search_queries", "required_concepts", "topic_anchors",
    "excluded_title_terms", "start_year", "end_year", "max_papers",
    "required_reference_count", "retrieval_target", "generation_limit",
    "year_range_explicit", "strict_year_range", "max_papers_explicit",
    "requested_sections", "language", "citation_style", "workflow",
    "research_plan", "research_semantic_frame", "core_deliverables",
    "user_paper_profile", "screening_protocol"
)
@optional(
    "current_time", "selected_scope", "topic_interpretations",
    "research_request", "conversation_history"
)
def plan_node(state: "ResearchAgentState", llm=None, current_year: int | None = None) -> "ResearchAgentState":
    """解析用户需求，生成检索计划。"""
    t0 = time.time()
    try:
        from app.agent.intent import recognize_intent
        from app.agent.slot_extractor import extract_slots
        from app.agent.planner import build_screening_protocol, build_search_plan
        from app.agent.research_plan import build_research_request_plan
        from app.agent.research_semantic_parser import parse_research_semantics
        from app.core.config import get_settings
        from app.tools.get_current_time import get_current_time

        user_query = state["user_query"]
        settings = get_settings()
        if current_year is None and _needs_current_time_tool(user_query):
            current_time = get_current_time()
            state["current_time"] = current_time
            current_year = int(current_time["year"])
            append_step(
                state,
                "get_current_time",
                "success",
                tool_name="get_current_time",
                input_data={
                    "query": user_query,
                    "reason": "resolve_relative_or_default_year_range",
                },
                output_data=current_time,
                duration_ms=0,
            )
        elif current_year is None:
            # 闭区间年份不依赖当前时间；用用户给出的 end_year 作为内部占位。
            m = re.search(r"(?:19|20)\d{2}\s*(?:[-~—到至])\s*((?:19|20)\d{2})", user_query)
            current_year = int(m.group(1)) if m else 1900

        # 多轮会话恢复时，user_query 可能是原请求附加的澄清答案或范围说明。
        # 优先使用持久化的顶层任务意图，避免附加文本中偶然出现的词触发另一意图。
        persisted_request = state.get("research_request") or {}
        persisted_intent = persisted_request.get("task_type") or state.get("intent")
        # 会话服务在澄清恢复路径显式写入 ``intent_context_role``。不要仅凭
        # ``research_request`` 是否存在判断恢复：同一 session 开启新研究时
        # 也会携带旧状态，此时必须重新识别顶层意图。
        context_role = str(state.get("intent_context_role") or "").strip().lower()
        if context_role not in {"request", "clarification_answer", "working_query"}:
            history = state.get("conversation_history") or []
            latest_user_type = next(
                (
                    str(item.get("type") or "").strip().lower()
                    for item in reversed(history)
                    if isinstance(item, dict) and str(item.get("role") or "").lower() == "user"
                ),
                "",
            )
            context_role = (
                "working_query"
                if latest_user_type == "clarification_answer"
                else "request"
            )
        is_resumed_query = context_role in {"clarification_answer", "working_query"}
        intent_result = recognize_intent(
            user_query,
            llm,
            conversation_role=context_role,
            previous_intent=persisted_intent if is_resumed_query else None,
            original_query=persisted_request.get("original_query") if is_resumed_query else None,
        )
        state["intent"] = intent_result.intent
        state["confidence"] = intent_result.confidence

        # 槽位抽取
        slots = extract_slots(
            user_query,
            intent_result.intent,
            llm=llm,
            current_year=current_year,
        )
        # 澄清回答不能替换已确认的主题。范围缩小保存在 semantic_frame / selected_scope 中，
        # 不用将内部工作查询的整段文字再次作为主题抽取。
        persisted_topic = str(persisted_request.get("topic") or "").strip()
        if is_resumed_query and persisted_topic:
            slots = slots.model_copy(update={"topic": persisted_topic})

        semantic_frame_data = state.get("research_semantic_frame")
        if semantic_frame_data:
            from app.schemas.research_plan_schema import ResearchSemanticFrame

            # 澄清问答发生在两次规划之间时，缓存的语义帧是在范围收窄前解析的：
            # 其中 language_affinity 等判断看不到用户后来确认的研究范围。
            # 工作查询一旦变化（通常是追加了澄清原文与范围确认），必须重解析。
            frame_source = str(state.get("semantic_frame_source_query") or "")
            if not frame_source or frame_source != str(user_query):
                logger.info(
                    "Semantic frame stale (working query changed after earlier "
                    "parse); reparsing with clarified request"
                )
                semantic_frame = parse_research_semantics(
                    user_query=user_query,
                    topic=slots.topic or user_query,
                    deliverables=slots.requested_sections,
                    llm=llm,
                )
            else:
                semantic_frame = ResearchSemanticFrame.model_validate(
                    semantic_frame_data
                )
        else:
            semantic_frame = parse_research_semantics(
                user_query=user_query,
                topic=slots.topic or user_query,
                deliverables=slots.requested_sections,
                llm=llm,
            )
        if is_resumed_query and persisted_topic and semantic_frame.canonical_topic != persisted_topic:
            semantic_frame = semantic_frame.model_copy(update={
                "canonical_topic": persisted_topic,
                "validation_warnings": list(dict.fromkeys([
                    *semantic_frame.validation_warnings,
                    "canonical_topic_preserved_from_original_request",
                ])),
            })
        state["research_semantic_frame"] = semantic_frame.model_dump(mode="json")
        state["semantic_frame_source_query"] = str(user_query)
        state["canonical_topic"] = str(
            semantic_frame.canonical_topic or slots.topic or user_query
        ).strip()

        # 生成计划；语义分支优先于通用关键词，旧字段继续兼容下游节点。
        plan = build_search_plan(
            user_query, intent_result.intent, slots, llm,
            semantic_frame=semantic_frame,
        )
        state["workflow"] = plan["workflow"]
        state["topic"] = plan["topic"]
        state["keywords"] = plan["keywords"]
        state["core_keywords"] = plan.get("core_keywords") or [state["topic"]]
        state["expanded_keywords"] = plan.get("expanded_keywords") or []
        state["keyword_batches"] = plan.get("keyword_batches") or []
        state["search_branches"] = plan.get("search_branches") or []
        selected_scope = state.get("selected_scope") or {}
        scope_branches = selected_scope.get("branches") or []
        scope_queries = [
            str(query).strip()
            for branch in scope_branches
            for query in (branch.get("seed_queries") or [])[:1]
            if str(query).strip()
        ]
        if not scope_queries:
            scope_queries = [
                str(query).strip()
                for query in (selected_scope.get("seed_queries") or [])[:3]
                if str(query).strip()
            ]
        state["scope_search_queries"] = list(dict.fromkeys(scope_queries))
        state["scope_query_roles"] = {
            query.casefold(): "scope_precision" for query in state["scope_search_queries"]
        }
        if state["scope_search_queries"]:
            state["keywords"] = list(dict.fromkeys([
                *state["scope_search_queries"],
                *state["keywords"],
            ]))
        state["required_concepts"] = plan.get("topic_anchors") or plan.get("required_concepts") or []
        state["topic_anchors"] = (
            plan.get("semantic_topic_anchors")
            or plan.get("topic_anchors")
            or state.get("topic_anchors")
            or []
        )
        state["excluded_title_terms"] = []  # 不再自动生成排除词，避免误杀
        state["start_year"] = plan.get("start_year") or current_year - settings.default_year_lookback + 1
        state["end_year"] = plan.get("end_year") or current_year
        state["max_papers"] = plan.get("max_papers") or settings.default_max_papers
        state["required_reference_count"] = plan.get("required_reference_count") or state["max_papers"]
        state["retrieval_target"] = plan.get("retrieval_target") or state["required_reference_count"]
        state["generation_limit"] = plan.get("generation_limit") or state["required_reference_count"]
        state["year_range_explicit"] = bool(plan.get("year_range_explicit"))
        state["strict_year_range"] = bool(plan.get("strict_year_range"))
        state["max_papers_explicit"] = bool(plan.get("max_papers_explicit"))
        # 消歧续跑会给原请求附加英文种子检索式；输出约束必须以消歧前抽取的
        # research_request 为准，不能被这些内部检索文本二次污染。
        original_request = state.get("research_request") or {}
        # 多轮质量决策可能修改年份、篇数和检索目标。恢复执行时，这些持久化
        # 约束优先于重新解析原始自然语言得到的旧值。
        for key in (
            "start_year",
            "end_year",
            "max_papers",
            "required_reference_count",
            "retrieval_target",
            "generation_limit",
        ):
            if original_request.get(key) is not None:
                state[key] = int(original_request[key])
        for key in ("year_range_explicit", "strict_year_range", "max_papers_explicit"):
            if original_request.get(key) is not None:
                state[key] = bool(original_request[key])
        # WHY: semantic parser 可能把“近三年”解释成旧的绝对年份文本；真正的
        # 权威窗口要等槽位解析和多轮约束覆盖后才确定。此处统一刷新，避免后续
        # route/writer 同时看到 2022-2024 与 2024-2026 两套口径。
        semantic_frame = _synchronize_semantic_frame_time_window(
            semantic_frame,
            int(state["start_year"]),
            int(state["end_year"]),
        )
        state["research_semantic_frame"] = semantic_frame.model_dump(mode="json")
        state["requested_sections"] = (
            original_request.get("requested_sections")
            or plan.get("requested_sections")
            or ["related_work"]
        )
        state["language"] = original_request.get("language") or plan["language"]
        state["citation_style"] = (
            original_request.get("citation_style") or plan["citation_style"]
        )
        # 主题语言倾向决定的中文分支配额；缺省时 rank 节点回落到全局配置。
        if plan.get("language_branch_zh_ratio") is not None:
            state["language_branch_zh_ratio"] = float(plan["language_branch_zh_ratio"])
            state["language_branch_zh_ratio_reason"] = str(
                plan.get("language_branch_zh_ratio_reason") or ""
            )

        # 兼容迁移：旧字段继续供现有节点使用，同时生成新的统一研究计划。
        # 消歧恢复时以持久化的原始约束覆盖附加检索词后可能被污染的槽位。
        plan_slots = slots.model_copy(
            update={
                "topic": state["topic"],
                "start_year": state["start_year"],
                "end_year": state["end_year"],
                "required_reference_count": state["required_reference_count"],
                "retrieval_target": state["retrieval_target"],
                "generation_limit": state["generation_limit"],
                "requested_sections": state["requested_sections"],
                "language": state["language"],
                "citation_style": state["citation_style"],
            }
        )
        research_plan = build_research_request_plan(
            user_query=original_request.get("user_query") or user_query,
            intent_result=intent_result,
            slots=plan_slots,
            search_plan=plan,
            selected_scope=state.get("selected_scope"),
            topic_interpretations=state.get("topic_interpretations"),
            semantic_frame=semantic_frame,
        )
        state["research_plan"] = research_plan.model_dump(mode="json")
        state["screening_protocol"] = build_screening_protocol(
            original_query=str(
                original_request.get("original_query")
                or original_request.get("user_query")
                or user_query
            ),
            user_query=user_query,
            topic=state["topic"],
            conversation_history=state.get("conversation_history") or [],
            selected_scope=state.get("selected_scope") or {},
            semantic_frame=state.get("research_semantic_frame") or {},
            search_branches=state.get("search_branches") or [],
            topic_anchors=state.get("topic_anchors") or state.get("required_concepts") or [],
            llm=llm,
        )
        from app.agent.deliverable_router import (
            extract_user_paper_profile,
            resolve_core_deliverables,
        )
        core_deliverables = resolve_core_deliverables(
            intent_result.intent,
            state["requested_sections"],
        )
        state["core_deliverables"] = [item.value for item in core_deliverables]
        profile = extract_user_paper_profile(user_query, state)
        state["user_paper_profile"] = profile.model_dump(mode="json")

        has_english_keyword = any(
            re.search(r"[A-Za-z]{4,}", keyword)
            for keyword in state["keywords"]
        )
        planning_error = plan.get("planning_error")
        if planning_error or (
            re.search(r"[\u4e00-\u9fff]", state["topic"])
            and not has_english_keyword
        ):
            reason = planning_error or "未生成可用于国际论文库的英文检索词"
            # P1 集成：用 PlanningError 统一构造错误上下文，但仍只把字符串
            # 消息写入 state["errors"]（下游多处按 list[str] 消费该字段），
            # 保持向后兼容。异常对象本身用于结构化日志和 step 记录。
            from app.agent.exceptions import PlanningError

            error = PlanningError(
                reason,
                step="plan",
                context={
                    "keywords": state["keywords"][:5],
                    "required_concepts_count": len(state.get("required_concepts", [])),
                },
            )
            state["planning_failed"] = True
            state.setdefault("errors", []).append(f"plan: {reason}")
            logger.warning(
                "plan_node: planning_failed, keywords=%s concepts=%d error=%s",
                state["keywords"][:5],
                len(state.get("required_concepts", [])),
                reason,
            )
            append_step(
                state,
                "plan",
                "failed",
                input_data={"query": user_query},
                output_data={
                    "intent": intent_result.intent,
                    "topic": state["topic"],
                    "keywords": state["keywords"],
                    "task_chain": list(semantic_frame.task_chain),
                    "required_focuses": list(semantic_frame.required_focuses),
                    "required_concepts": state["required_concepts"],
                    "excluded_title_terms": state["excluded_title_terms"],
                    "start_year": state["start_year"],
                    "end_year": state["end_year"],
                    "max_papers": state["max_papers"],
                    "required_reference_count": state["required_reference_count"],
                    "retrieval_target": state["retrieval_target"],
                    "generation_limit": state["generation_limit"],
                    "requested_sections": state["requested_sections"],
                    "error_detail": error.to_dict(),
                },
                error=reason,
                duration_ms=int((time.time() - t0) * 1000),
            )
            return state

        append_step(
            state, "plan", "success",
            input_data={"query": user_query},
            output_data={
                "intent": intent_result.intent,
                "workflow": plan["workflow"],
                "topic": plan["topic"],
                "keywords": plan["keywords"],
                "scope_search_queries": state.get("scope_search_queries") or [],
                "research_mode": semantic_frame.research_mode.value,
                "method_roles": {
                    method.id: method.role.value for method in semantic_frame.methods
                },
                "task_chain": list(semantic_frame.task_chain),
                "required_focuses": list(semantic_frame.required_focuses),
                "search_branches": state.get("search_branches") or [],
                "screening_protocol": state.get("screening_protocol") or {},
                "required_concepts": state["required_concepts"],
                "excluded_title_terms": state["excluded_title_terms"],
                "start_year": state["start_year"],
                "end_year": state["end_year"],
                "max_papers": state["max_papers"],
                "required_reference_count": state["required_reference_count"],
                "retrieval_target": state["retrieval_target"],
                "generation_limit": state["generation_limit"],
                "year_range_explicit": state["year_range_explicit"],
                "strict_year_range": state["strict_year_range"],
                "max_papers_explicit": state["max_papers_explicit"],
                "requested_sections": state["requested_sections"],
                "language": state["language"],
            },
            duration_ms=int((time.time() - t0) * 1000),
        )
    except Exception as e:
        logger.error("plan_node failed: %s", e)
        # 通用异常同样必须置位：graph 在规划后检查该标志短路检索。
        # 缺失时会带空计划继续跑，把规划失败伪装成下游的"接口限流"。
        state["planning_failed"] = True
        state.setdefault("errors", []).append(f"plan: {e}")
        append_step(state, "plan", "failed", error=str(e))
    return state




# ============================================================
# Provisional Route 节点（搜索前概念规划）
# ============================================================
@node(name="provisional_routes", category="planning", description="搜索前生成候选研究路线框架")
@requires("canonical_topic", "research_semantic_frame")
@provides("provisional_framework")
@optional("user_query", "selected_scope", "core_deliverables")
def provisional_route_node(state: "ResearchAgentState", llm=None) -> "ResearchAgentState":
    """在检索之前生成候选研究路线（Layer 1: Conceptual Planning）。

    与 cluster_node 的关键区别：
    - cluster_node 在搜索后让论文数据自行聚类 → 容易出现 text/video/期刊论文
    - provisional_route_node 在搜索前根据用户意图生成路线 → 路线基于研究问题
    """
    t0 = time.time()
    try:
        from app.agent.provisional_routes import generate_provisional_routes

        framework = generate_provisional_routes(state, llm)
        if framework:
            state["provisional_framework"] = framework
            routes = framework.get("provisional_routes") or []

            # 注入全局召回检索式：防止路线导向搜索产生确认偏误
            from app.agent.provisional_routes import generate_global_recall_queries
            global_queries = generate_global_recall_queries(
                state.get("canonical_topic") or state.get("topic", ""),
                state.get("research_semantic_frame"),
            )
            existing_keywords = list(state.get("keywords") or [])
            state["keywords"] = list(dict.fromkeys(
                existing_keywords + global_queries
            ))
            state["global_recall_queries"] = global_queries

            # 注入分路线检索式到 search_branches
            from app.agent.provisional_routes import route_aware_search_queries
            route_branches = route_aware_search_queries(
                routes,
                global_topic=state.get("canonical_topic") or state.get("topic", ""),
            )
            existing_branches = list(state.get("search_branches") or [])
            state["search_branches"] = existing_branches + [
                {
                    "branch_type": f"route_{br['route_id']}",
                    "queries": br["queries"],
                    "required_concepts": br.get("core_concepts", []),
                    "rationale": f"候选路线「{br['route_name']}」的定向检索",
                    "constraint_level": "exploratory",
                }
                for br in route_branches
            ]

            append_step(
                state, "provisional_routes", "success",
                tool_name="generate_provisional_routes",
                input_data={
                    "topic": state.get("canonical_topic") or state.get("topic", ""),
                    "has_semantic_frame": bool(state.get("research_semantic_frame")),
                },
                output_data={
                    "route_count": len(routes),
                    "global_recall_queries": global_queries,
                    "route_branches": len(route_branches),
                    "routes": [
                        {"name": r.get("name"), "concepts": r.get("core_concepts", [])[:5]}
                        for r in routes
                    ],
                },
                duration_ms=int((time.time() - t0) * 1000),
            )
            logger.info(
                "Provisional routes: %d routes + %d global recall queries + %d route-aware branches",
                len(routes), len(global_queries), len(route_branches),
            )
        else:
            state["provisional_framework"] = {}
            append_step(
                state, "provisional_routes", "skipped",
                output_data={"reason": "LLM unavailable or generation failed"},
                duration_ms=int((time.time() - t0) * 1000),
            )
    except Exception as e:
        logger.warning("provisional_route_node failed: %s", e)
        state["provisional_framework"] = {}
        append_step(state, "provisional_routes", "failed", error=str(e))
    return state




@node(name="diagnose_evidence_gaps", category="generation", description="结合路线指标与 LLM 语义解释诊断证据缺口")
@requires("route_validation_report", "route_decisions")
@provides("evidence_gap_report")
@optional("provisional_framework", "searched_keywords", "source_diagnostics")
def diagnose_evidence_gaps_node(
    state: "ResearchAgentState",
    llm=None,
) -> "ResearchAgentState":
    """诊断缺口但不决定是否执行恢复动作。"""
    t0 = time.time()
    try:
        from app.agent.evidence_recovery import diagnose_evidence_gaps

        report = diagnose_evidence_gaps(state, llm=llm)
        state["evidence_gap_report"] = report.model_dump(mode="json")
        append_step(
            state,
            "diagnose_evidence_gaps",
            "success",
            tool_name="diagnose_evidence_gaps",
            input_data={
                "routes": len(
                    (state.get("provisional_framework") or {}).get("provisional_routes") or []
                ),
                "decisions": len(state.get("route_decisions") or []),
            },
            output_data=state["evidence_gap_report"],
            duration_ms=int((time.time() - t0) * 1000),
        )
    except Exception as exc:
        logger.warning("diagnose_evidence_gaps_node failed: %s", exc)
        state["evidence_gap_report"] = {
            "needs_recovery": False,
            "gaps": [],
            "affected_route_ids": [],
            "coverage_score": 0.0,
            "source_health": "unknown",
            "diagnosis_source": "failed",
            "notes": [str(exc)],
        }
        state.setdefault("errors", []).append(f"diagnose_evidence_gaps: {exc}")
        append_step(state, "diagnose_evidence_gaps", "failed", error=str(exc))
    return state


@node(name="recovery_controller", category="execution", description="按预算、数据源健康和查询新颖度确定恢复动作")
@requires("evidence_gap_report")
@provides("recovery_decision")
@optional("recovery_round", "route_recovery_attempts", "scope_revision_count")
def recovery_controller_node(state: "ResearchAgentState") -> "ResearchAgentState":
    """确定性恢复控制器；不调用 LLM。"""
    t0 = time.time()
    try:
        from app.agent.evidence_recovery import decide_recovery
        from app.core.config import get_settings
        from app.schemas.recovery_schema import EvidenceGapReport

        settings = get_settings()
        report = EvidenceGapReport.model_validate(state.get("evidence_gap_report") or {})
        if not settings.enable_evidence_recovery:
            from app.schemas.recovery_schema import (
                RecoveryAction,
                RecoveryDecision,
                RecoveryStatus,
            )
            decision = RecoveryDecision(
                action=RecoveryAction.CONTINUE,
                status=RecoveryStatus.NOT_REQUIRED,
                reason="证据恢复闭环已由配置关闭。",
            )
        else:
            decision = decide_recovery(
                state,
                report,
                max_rounds=settings.evidence_recovery_max_rounds,
                max_route_attempts=settings.evidence_recovery_max_route_attempts,
                min_query_novelty=settings.evidence_recovery_min_query_novelty,
                max_scope_revisions=settings.evidence_recovery_max_scope_revisions,
                max_queries=settings.max_search_keywords,
            )
        state["recovery_decision"] = decision.model_dump(mode="json")
        append_step(
            state,
            "recovery_controller",
            "success",
            tool_name="deterministic_recovery_controller",
            input_data={
                "recovery_round": int(state.get("recovery_round") or 0),
                "scope_revision_count": int(state.get("scope_revision_count") or 0),
            },
            output_data=state["recovery_decision"],
            duration_ms=int((time.time() - t0) * 1000),
        )
    except Exception as exc:
        logger.warning("recovery_controller_node failed: %s", exc)
        state["recovery_decision"] = {
            "action": "DEGRADE",
            "status": "DEGRADED",
            "reason": f"恢复控制器失败：{exc}",
            "affected_route_ids": [],
            "queries": [],
        }
        state.setdefault("errors", []).append(f"recovery_controller: {exc}")
        append_step(state, "recovery_controller", "failed", error=str(exc))
    return state


@node(name="scope_revision", category="planning", description="在用户显式边界内保守修订候选路线框架")
@requires("evidence_gap_report", "provisional_framework")
@provides("provisional_framework", "scope_revision_count")
def scope_revision_node(state: "ResearchAgentState", llm=None) -> "ResearchAgentState":
    """执行至多由控制器批准的范围修订，不修改用户 selected_scope。"""
    t0 = time.time()
    try:
        from app.agent.evidence_recovery import revise_scope_framework
        from app.schemas.recovery_schema import EvidenceGapReport

        report = EvidenceGapReport.model_validate(state.get("evidence_gap_report") or {})
        revised = revise_scope_framework(state, report, llm)
        if not revised:
            state["scope_revision_failed"] = True
            append_step(
                state,
                "scope_revision",
                "skipped",
                output_data={"reason": "LLM unavailable or revision did not satisfy route contract"},
                duration_ms=int((time.time() - t0) * 1000),
            )
            return state
        state["provisional_framework"] = revised
        state["scope_revision_count"] = int(state.get("scope_revision_count") or 0) + 1
        state["scope_revision_failed"] = False
        append_step(
            state,
            "scope_revision",
            "success",
            tool_name="revise_scope_framework",
            output_data={
                "scope_revision_count": state["scope_revision_count"],
                "route_count": len(revised.get("provisional_routes") or []),
            },
            duration_ms=int((time.time() - t0) * 1000),
        )
    except Exception as exc:
        logger.warning("scope_revision_node failed: %s", exc)
        state["scope_revision_failed"] = True
        state.setdefault("errors", []).append(f"scope_revision: {exc}")
        append_step(state, "scope_revision", "failed", error=str(exc))
    return state

