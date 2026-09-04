"""旧意图到四种核心交付物的兼容路由与前置检查。"""

from __future__ import annotations

import re
from typing import Any, Iterable

from app.deliverables.registry import get_deliverable_spec
from app.schemas.agent_schema import IntentType
from app.schemas.deliverable_schema import (
    CoreDeliverableType,
    DeliverableReadinessResult,
    GenerationReadinessResult,
    UserPaperProfile,
)
from app.tools.paper_rerank import RULE_SCREENED_RESERVE


def resolve_core_deliverables(
    intent: str,
    requested_sections: Iterable[str] | None,
) -> list[CoreDeliverableType]:
    if intent not in {
        IntentType.GENERATE_REVIEW.value,
        IntentType.GENERATE_RELATED_WORK.value,
        IntentType.GENERATE_INTRODUCTION.value,
    }:
        return []
    mapping = {
        "background": CoreDeliverableType.RESEARCH_BACKGROUND,
        "research_background": CoreDeliverableType.RESEARCH_BACKGROUND,
        "research_status": CoreDeliverableType.RESEARCH_STATUS,
        "related_work": CoreDeliverableType.RELATED_WORK,
        "review": CoreDeliverableType.NARRATIVE_REVIEW,
        "literature_review": CoreDeliverableType.NARRATIVE_REVIEW,
        "narrative_review": CoreDeliverableType.NARRATIVE_REVIEW,
        # 旧引言入口只保留兼容：四任务体系中归入不含本文贡献的研究背景。
        "introduction": CoreDeliverableType.RESEARCH_BACKGROUND,
    }
    values = [mapping[str(item)] for item in (requested_sections or []) if str(item) in mapping]
    if not values:
        if intent == IntentType.GENERATE_RELATED_WORK.value:
            values = [CoreDeliverableType.RELATED_WORK]
        elif intent == IntentType.GENERATE_INTRODUCTION.value:
            values = [CoreDeliverableType.RESEARCH_BACKGROUND]
        elif intent == IntentType.GENERATE_REVIEW.value:
            values = [CoreDeliverableType.NARRATIVE_REVIEW]
    return list(dict.fromkeys(values))


def extract_user_paper_profile(
    text: str,
    existing: dict[str, Any] | None = None,
) -> UserPaperProfile:
    existing = existing or {}
    legacy = existing.get("our_work") if "our_work" in existing else existing
    profile = UserPaperProfile(
        research_problem=str(legacy.get("research_problem") or "").strip(),
        proposed_method=str(
            legacy.get("proposed_method") or legacy.get("method_name") or legacy.get("method_summary") or ""
        ).strip() or None,
        research_direction=str(legacy.get("research_direction") or "").strip() or None,
        research_object=str(legacy.get("research_object") or "").strip() or None,
        target_task=str(legacy.get("target_task") or "").strip() or None,
        application_scenario=str(legacy.get("application_scenario") or "").strip() or None,
        claimed_contribution=str(legacy.get("claimed_contribution") or "").strip() or None,
        data_modalities=[str(x) for x in legacy.get("data_modalities") or []],
        main_contributions=[str(x) for x in legacy.get("main_contributions") or legacy.get("innovations") or []],
        comparison_targets=[str(x) for x in legacy.get("comparison_targets") or []],
    )
    content = str(text or "").strip()
    if not content:
        return profile
    self_research = bool(re.search(r"我的论文|我的研究|本文|本研究|我计划|我采用|我使用", content))
    if self_research and not profile.research_problem:
        problem_match = re.search(
            r"(?:重点|主要)?(?:解决|研究|面向|针对|检测|识别)([^，。；;]{2,80})",
            content,
        )
        profile.research_problem = (
            problem_match.group(1).strip() if problem_match else content[:180]
        )
    if self_research and not profile.proposed_method:
        method_match = re.search(
            r"(?:使用|采用|基于|通过|计划采用)([^，。；;]{2,60}?)(?:来|用于|解决|研究|检测|识别|$)",
            content,
        )
        if method_match:
            profile.proposed_method = method_match.group(1).strip()
    return profile


def check_deliverable_readiness(
    deliverable_type: CoreDeliverableType | str,
    state: dict[str, Any],
    phase: str = "post_evidence",
) -> DeliverableReadinessResult:
    requested = CoreDeliverableType(deliverable_type)
    missing: list[str] = []
    insufficient: list[str] = []
    question = None
    downgrade = None
    profile = UserPaperProfile.model_validate(state.get("user_paper_profile") or {})

    if not str(state.get("topic") or "").strip() and phase != "pre_retrieval":
        missing.append("topic")
    if requested == CoreDeliverableType.RELATED_WORK:
        if not profile.research_problem:
            missing.append("user_paper_profile.research_problem")
        if not (profile.proposed_method or profile.research_direction):
            missing.append("user_paper_profile.method_or_direction")
        if missing:
            question = "你的论文主要解决什么问题，并计划采用什么方法或研究路线？"

    if phase == "post_evidence":
        cards = state.get("paper_cards") or []
        usable = [
            card for card in cards
            if str(getattr(
                (card.get("evidence_state") or {}).get("access_level") or card.get("evidence_source"),
                "value",
                (card.get("evidence_state") or {}).get("access_level") or card.get("evidence_source"),
            ))
            in {"abstract", "partial_full_text", "full_text"}
            and card.get("quality_status") != "invalid"
        ]
        # PDF 下载/解析失败按论文逐篇降级为摘要证据。只要摘要卡片数量、
        # 动态分类和显式声明满足要求，仍可生成“摘要级研究现状”；
        # 详细结构、消融、数据划分等结论继续由 EvidenceState 禁止。
        taxonomy = state.get("dynamic_taxonomy") or {}
        validation = state.get("taxonomy_validation") or {}
        themes = [theme for theme in taxonomy.get("themes") or [] if theme]
        spec = get_deliverable_spec(requested)
        allow_unvalidated_taxonomy = bool(state.get("allow_unvalidated_taxonomy"))
        if len(usable) < int(spec.min_references or 1):
            insufficient.append(
                f"可用于事实性写作的摘要或全文论文只有 {len(usable)} 篇，少于建议的 {spec.min_references} 篇"
            )
        if spec.requires_dynamic_taxonomy:
            if not themes and not allow_unvalidated_taxonomy:
                insufficient.append("当前证据未形成可用的动态研究路线")
            elif not allow_unvalidated_taxonomy and validation and (
                (
                    not validation.get("valid", False)
                    and str(validation.get("status") or "") not in {"valid", "valid_with_warning"}
                )
                or str(validation.get("status") or "")
                in {"invalid", "refinement_required"}
            ):
                insufficient.append("动态分类未通过验证，需要修订分类后才能生成")
            fallback_theme_ids = {
                str(theme.get("theme_id") or "")
                for theme in themes
                if re.search(r"其他|其它|未分类|other|misc", str(theme.get("name") or ""), re.I)
            }
            non_fallback_theme_count = len(themes) - len(fallback_theme_ids)
            # 兜底主题是长尾证据的内部容器，WritingPlan 不会把它渲染为
            # 正文标题。只要仍有至少两条通过验证的正式路线，就不应因为
            # 一个任意比例阈值吞掉整份研究现状。
            if (
                not allow_unvalidated_taxonomy
                and fallback_theme_ids
                and non_fallback_theme_count < 2
            ):
                insufficient.append("动态分类包含兜底主题，无法直接用于生成研究现状")
                
        elif requested in {
            CoreDeliverableType.RESEARCH_STATUS,
            CoreDeliverableType.NARRATIVE_REVIEW,
        } and len(themes) < 2:
            insufficient.append("当前证据未形成至少两条可区分的研究路线")
            
        if requested == CoreDeliverableType.NARRATIVE_REVIEW and insufficient:
            downgrade = CoreDeliverableType.RESEARCH_STATUS
        elif requested == CoreDeliverableType.RESEARCH_STATUS and insufficient:
            downgrade = CoreDeliverableType.RESEARCH_BACKGROUND

    return DeliverableReadinessResult(
        ready=not missing and not insufficient,
        requested_type=requested,
        effective_type=downgrade or requested,
        missing_inputs=missing,
        insufficient_evidence=insufficient,
        clarification_question=question,
        downgrade_suggestion=downgrade,
    )


def unconfirmed_reference_ids(state: dict[str, Any]) -> set[str]:
    """返回仅经规则回填、未经 LLM 语义确认的论文 ID 集合。

    WHY: ``rule_screened_reserve`` 是重排阶段为凑够 ``minimum_required`` 而按
    规则分回填的论文，从未被 LLM 看过。它们此前可以自由成为证据卡片并被引用，
    于是"不少于 40 篇"的缺口被静默补齐、门禁记为达标。标记的唯一存活载体是
    ``paper_details``：卡片抽取只保留 ``PaperCard`` 的显式字段，下划线私有键
    在成卡时就被丢掉，所以这里必须回到详情列表读。
    """
    return {
        str(paper.get("paper_id") or "")
        for paper in state.get("paper_details") or []
        if paper.get("paper_id")
        and str(paper.get("_screening_decision") or "") == RULE_SCREENED_RESERVE
    }


def check_generation_readiness(state: dict[str, Any]) -> GenerationReadinessResult:
    """检查用户硬约束和分类质量；失败时不得进入 Writer。"""
    cards = state.get("paper_cards") or []
    unconfirmed_ids = unconfirmed_reference_ids(state)
    usable_ids = {
        str(card.get("paper_id") or "")
        for card in cards
        if card.get("paper_id")
        and str(card.get("paper_id")) not in unconfirmed_ids
        and card.get("quality_status") != "invalid"
        and str(getattr(
            (card.get("evidence_state") or {}).get("access_level") or card.get("evidence_source"),
            "value",
            (card.get("evidence_state") or {}).get("access_level") or card.get("evidence_source"),
        )) in {"abstract", "partial_full_text", "full_text"}
    }
    requested = int(state.get("required_reference_count") or state.get("max_papers") or 0)
    issues: list[dict[str, Any]] = []
    recovery: list[str] = []
    semantic_frame = state.get("research_semantic_frame") or {}
    from app.agent.evidence_roles import citation_eligible_paper_ids

    requested_deliverables = {
        str(value) for value in state.get("core_deliverables") or [] if str(value)
    }
    if requested_deliverables:
        eligible_ids = set().union(*(
            citation_eligible_paper_ids(
                semantic_frame, cards, deliverable_type=deliverable_type,
            )
            for deliverable_type in requested_deliverables
        ))
    else:
        eligible_ids = citation_eligible_paper_ids(semantic_frame, cards)
    eligible_ids &= usable_ids
    state["citation_eligible_paper_ids"] = sorted(eligible_ids)

    if state.get("max_papers_explicit", False) and requested and len(eligible_ids) < requested:
        issues.append({
            "code": "minimum_references_not_met",
            "message": f"要求至少引用 {requested} 篇，但只有 {len(eligible_ids)} 篇同时满足证据可用性与任务相关性",
            "requested": requested,
            "available": len(eligible_ids),
        })
        recovery.extend([
            "扩大检索年份范围",
            "在保持主题边界的前提下补充同义词和数据库",
            "明确允许纳入更多会议论文或预印本",
            f"确认接受少于 {requested} 篇后重新提交",
        ])

    if "research_status" in set(state.get("core_deliverables") or []):
        from app.agent.focus_coverage import required_focus_coverage

        coverage = required_focus_coverage(
            semantic_frame,
            cards,
        )
        state["focus_coverage"] = coverage
        if not coverage.get("ready", True):
            missing = coverage.get("missing_focuses") or []
            issues.append({
                "code": "required_focus_evidence_not_met",
                "message": "用户明确研究重点缺少足够直接证据：" + "、".join(missing),
                "missing_focuses": missing,
                "counts": coverage.get("counts") or {},
                "required_counts": coverage.get("required_counts") or {},
            })
            recovery.append("针对缺失重点执行专项补检索后再进入写作")

    # 分类质量由逐交付物 readiness 处理：研究现状/综述可安全降级为研究背景，
    # 相关工作等无可用降级路径的类型仍会被自身 readiness 阻断。这里不做全局
    # 阻断，避免一个无效分类连带阻止同批请求中不依赖分类的研究背景。

    return GenerationReadinessResult(
        ready=not issues,
        requested_minimum_references=requested,
        usable_reference_count=len(eligible_ids),
        blocking_issues=issues,
        recovery_options=list(dict.fromkeys(recovery)),
    )
