"""把交付物规格、动态分类和证据综合编译成可审计写作计划。"""

from __future__ import annotations

import json
import re
from typing import Any

from app.deliverables.registry import get_deliverable_spec
from app.schemas.deliverable_schema import (
    CoreDeliverableType,
    PlanningNode,
    UserPaperProfile,
    WritingPlan,
    WritingSection,
)
from app.tools.cluster_papers import semantic_route_label

_FALLBACK_THEME_RE = re.compile(r"其他|其它|未分类|兜底|other|misc", re.I)

# 独立成节所需的最低论文数：spec 要求每条路线做“路线内比较”而非逐篇
# 复述（planning.require_comparative_synthesis），两篇只能并列、三篇起
# 才谈得上比较。子节预算按此值随可用证据缩放，避免证据不足时把 spec
# 上限用满、产出一批两篇一节的碎片小节。
_MIN_PAPERS_PER_SUBSECTION = 3

# 正式研究路线小节的章节级最低唯一引用。spec 要求路线内比较而不是逐篇
# 复述，两篇是"能比较"的下界；该值只作用于章节，不参与全局
# required_reference_count 的核算。
_MIN_ROUTE_SECTION_REFERENCES = 2

# 章节级下限只对高引用量综述生效：小语料窄题的单路线证据边界属于事实，
# 硬性要求两篇会把证据边界误报成写作缺陷。
_SECTION_FLOOR_MIN_REVIEW_SIZE = 20


def build_writing_plan(
    deliverable_type: CoreDeliverableType | str,
    state: dict[str, Any],
    llm=None,
) -> WritingPlan:
    dtype = CoreDeliverableType(deliverable_type)
    spec = get_deliverable_spec(dtype)
    from app.agent.deliverable_router import unconfirmed_reference_ids

    unconfirmed_ids = unconfirmed_reference_ids(state)
    syntheses = [
        synthesis
        for synthesis in (state.get("theme_synthesis") or [])
        if not _is_fallback_theme(synthesis.get("theme_name"))
    ]
    usable_ids = [
        str(card.get("paper_id")) for card in state.get("paper_cards") or []
        if card.get("paper_id")
        and str(card.get("paper_id")) not in unconfirmed_ids
        and card.get("quality_status") != "invalid"
        and str(getattr(
            (card.get("evidence_state") or {}).get("access_level") or card.get("evidence_source"),
            "value",
            (card.get("evidence_state") or {}).get("access_level") or card.get("evidence_source"),
        ))
        in {"abstract", "partial_full_text", "full_text"}
    ]
    semantic_frame = state.get("research_semantic_frame") or {}
    from app.agent.evidence_roles import citation_eligible_paper_ids

    eligible_ids = citation_eligible_paper_ids(
        semantic_frame,
        state.get("paper_cards") or [],
        deliverable_type=dtype.value,
    )
    usable_ids = [paper_id for paper_id in usable_ids if paper_id in eligible_ids]
    state["citation_eligible_paper_ids"] = list(usable_ids)
    usable_set = set(usable_ids)
    syntheses = [
        {
            **synthesis,
            "paper_ids": [
                str(paper_id) for paper_id in synthesis.get("paper_ids") or []
                if str(paper_id) in usable_set
            ],
        }
        for synthesis in syntheses
        if any(str(paper_id) in usable_set for paper_id in synthesis.get("paper_ids") or [])
    ]
    if dtype == CoreDeliverableType.RESEARCH_BACKGROUND and llm is not None:
        state["_dynamic_background_outline"] = _induce_background_outline(
            state, usable_ids, llm,
        )

    builders = {
        CoreDeliverableType.RESEARCH_BACKGROUND: _build_background_plan,
        CoreDeliverableType.RESEARCH_STATUS: _build_research_status_plan,
        CoreDeliverableType.RELATED_WORK: _build_related_work_plan,
        CoreDeliverableType.NARRATIVE_REVIEW: _build_narrative_review_plan,
    }
    sections, hidden_nodes = builders[dtype](state, spec, syntheses, usable_ids)
    requested_references = int(state.get("required_reference_count") or 0)
    if dtype in {
        CoreDeliverableType.RESEARCH_BACKGROUND,
        CoreDeliverableType.RESEARCH_STATUS,
        CoreDeliverableType.NARRATIVE_REVIEW,
    }:
        # 用户给出的是整份正文的最低引用约束，不能因为同时请求多个
        # 交付物就在某一个 WritingPlan 内擅自折半。
        citation_target = max(spec.min_references, requested_references)
    else:
        citation_target = spec.min_references
    citation_target = min(citation_target, len(usable_ids))
    required_focuses = [
        str(item).strip()
        for item in (state.get("research_semantic_frame") or {}).get("required_focuses") or []
        if str(item).strip()
    ]
    covered_focuses, undercovered_focuses = _evaluate_focus_coverage(
        required_focuses, sections, semantic_frame
    )
    conservative_evidence_mode = bool(
        len([section for section in sections if section.id.startswith("theme_")]) < 3
        or undercovered_focuses
        or (
            requested_references
            and citation_target < requested_references
            and len(usable_ids) < requested_references
        )
    )

    return WritingPlan(
        deliverable_type=dtype,
        purpose=spec.purpose,
        organizing_strategy=(state.get("dynamic_taxonomy") or {}).get("organizing_principle") or "evidence_driven",
        sections=sections,
        hidden_planning_nodes=hidden_nodes,
        evidence_policy={"use_explicit_claims_only": True, "respect_unsupported_fields": True},
        citation_policy={
            "format": "[paper_id]",
            "critical_trend_min_sources": 2,
            "minimum_unique_references": citation_target,
        },
        style_constraints={
            "forbidden_patterns": spec.forbidden_patterns,
            "no_domain_template": True,
            "conservative_evidence_mode": conservative_evidence_mode,
            "conservative_evidence_reason": (
                "路线数量不足、重点覆盖不完整或引用上限受可用证据约束"
                if conservative_evidence_mode else ""
            ),
        },
        target_char_range=(spec.validation.target_char_range if spec.validation else None),
        required_focuses=required_focuses,
        covered_focuses=covered_focuses,
        undercovered_focuses=undercovered_focuses,
    )


def _build_background_plan(
    state: dict[str, Any],
    spec,
    syntheses: list[dict[str, Any]],
    usable_ids: list[str],
) -> tuple[list[WritingSection], list[PlanningNode]]:
    del syntheses
    topic = str(state.get("canonical_topic") or state.get("topic") or "当前研究主题")
    cards = state.get("paper_cards") or []
    claim_fields = {
        str(field)
        for card in cards
        if str(card.get("paper_id") or "") in set(usable_ids)
        for field, claims in (card.get("field_claims") or {}).items()
        if claims
    }
    roles = [
        ("context", "现实背景与问题场景", f"依据证据说明{topic}产生的现实、理论或应用背景"),
        ("limitations", "既有方式及其限制", f"概括与{topic}直接相关的既有方式及证据明确支持的限制"),
        ("development", "研究发展条件", f"说明推动{topic}发展的理论、方法、数据或技术条件"),
        ("importance", "研究对象与价值", f"说明{topic}的研究对象为何重要及其理论或实践价值"),
        ("challenges", "待解决问题", f"归纳现有证据能够支持的主要挑战，不套用通用研究空白"),
    ]
    outline = state.get("_dynamic_background_outline") or {}
    dynamic_goals = [
        (
            str(item.get("id") or f"dynamic_{index}"),
            str(item.get("label") or "背景论证"),
            str(item.get("writing_goal") or ""),
        )
        for index, item in enumerate(outline.get("paragraph_goals") or [], 1)
        if isinstance(item, dict) and str(item.get("writing_goal") or "").strip()
    ]
    if 3 <= len(dynamic_goals) <= 5:
        selected = dynamic_goals
    else:
        selected = [roles[0], roles[2], roles[3]]
        if "limitations" in claim_fields or "research_problem" in claim_fields:
            selected.insert(1, roles[1])
        if "limitations" in claim_fields and len(selected) < 5:
            selected.append(roles[4])
    evidence_ids = _all_claim_ids(cards, usable_ids)
    hidden = [
        PlanningNode(
            node_id=f"background_{node_id}",
            node_type="background_goal",
            label=label,
            writing_goal=goal,
            evidence_ids=evidence_ids,
        )
        for node_id, label, goal in selected[:5]
    ]
    visible_title = spec.structure.visible_title if spec.structure else "研究背景"
    section = WritingSection(
        id="background_body",
        title=visible_title,
        purpose=(
            f"围绕内部段落目标写成{len(hidden)}个递进的连续自然段；不生成小标题、列表、"
            "证据说明或系统提示；综合多篇证据，不逐篇介绍"
        ),
        claims_to_establish=[node.writing_goal for node in hidden],
        supporting_paper_ids=usable_ids,
        supporting_claim_ids=evidence_ids,
        comparison_dimensions=[
            str(value) for value in outline.get("comparison_dimensions") or []
            if str(value).strip()
        ],
        target_word_count=1500,
        heading_level=2,
    )
    return [section], hidden


def _induce_background_outline(
    state: dict[str, Any],
    usable_ids: list[str],
    llm,
) -> dict[str, Any]:
    """让 LLM 从当前主题与声明级证据归纳背景段落目标，不生成事实正文。"""
    usable = set(usable_ids)
    evidence_summary = []
    for card in state.get("paper_cards") or []:
        paper_id = str(card.get("paper_id") or "")
        if paper_id not in usable:
            continue
        claims = [
            {
                "field": field,
                "claim": str(claim.get("claim") or claim.get("text") or "")[:300],
            }
            for field, values in (card.get("field_claims") or {}).items()
            for claim in values or []
            if isinstance(claim, dict) and claim.get("explicitly_reported")
        ][:12]
        evidence_summary.append({
            "paper_id": paper_id,
            "title": str(card.get("title") or ""),
            "relation_type": str(card.get("relation_type") or ""),
            "claims": claims,
        })
    prompt = f"""你是学术写作规划器。请根据当前研究主题、用户语义框架和可引用的声明级证据，
动态规划“研究背景”的 3 至 5 个递进段落目标。你只规划论证结构，不写正文，不补充外部事实，
不得套用任何预设学科、方法分类或应用场景。

研究主题：{state.get('canonical_topic') or state.get('topic') or ''}
用户语义框架：{json.dumps(state.get('research_semantic_frame') or {}, ensure_ascii=False)}
可用证据：{json.dumps(evidence_summary, ensure_ascii=False)}

严格返回 JSON：
{{
  "paragraph_goals": [
    {{"id": "context", "label": "由当前证据归纳的短标签", "writing_goal": "该段需要建立的论断"}}
  ],
  "comparison_dimensions": ["仅填写当前证据确实支持的比较维度"],
  "rationale": "结构为何适合当前主题"
}}
"""
    try:
        response = llm.complete(
            prompt,
            response_format="json_object",
            temperature=0.0,
            operation="induce_background_outline",
        )
        from app.core.json_utils import parse_json_object

        data = parse_json_object(response if isinstance(response, str) else str(response))
        goals = data.get("paragraph_goals") or []
        if not isinstance(goals, list) or not 3 <= len(goals) <= 5:
            return {}
        return {
            "paragraph_goals": goals,
            "comparison_dimensions": list(data.get("comparison_dimensions") or []),
            "rationale": str(data.get("rationale") or ""),
        }
    except Exception:
        return {}


def _build_research_status_plan(
    state: dict[str, Any],
    spec,
    syntheses: list[dict[str, Any]],
    usable_ids: list[str],
) -> tuple[list[WritingSection], list[PlanningNode]]:
    max_routes = (spec.structure.max_subsections if spec.structure else 4) or 4
    routes = _merge_and_select_themes(
        syntheses,
        state,
        max_routes=max_routes,
        evidence_paper_count=len(usable_ids),
    )
    evidence_ids = _all_claim_ids(state.get("paper_cards") or [], usable_ids)
    from app.agent.geography import has_reliable_geographic_comparison

    geography_ready = has_reliable_geographic_comparison(
        state.get("paper_cards") or state.get("paper_details") or []
    )
    visible_title = (
        spec.structure.visible_title if spec.structure else "研究现状"
    )
    if not geography_ready:
        visible_title = "研究现状"
    overview_purpose = "用一个综合段概括总体进展和主要路线，不输出检索、范围或质量说明"
    overview_purpose += (
        "；可依据作者机构、研究地区或样本来源比较不同国家或地区"
        if geography_ready
        else "；不得划分国内外研究或用论文语言推断研究地域"
    )
    route_paper_ids = list(dict.fromkeys(
        str(paper_id) for route in routes for paper_id in route.get("paper_ids") or []
    ))
    has_stage_requirements = bool(
        (state.get("research_semantic_frame") or {}).get("evidence_requirements")
    )
    if has_stage_requirements and route_paper_ids:
        # 路线论文保持优先序，但主题综合未挂载的合格证据必须留在授权池：
        # 路线并集曾整个替换该支撑集合（2026-08 会话：路线 46 篇 vs 合格
        # 79 篇），成为引用分配与 M15 授权的共同天花板，"不少于 N 篇"
        # 在 N 超过路线并集时必然失败。主题章节支撑不变，分配层 fit
        # score 仍会把主题论文优先分给对应章节，概述收其余。
        overview_supporting = list(dict.fromkeys([*route_paper_ids, *usable_ids]))
    else:
        overview_supporting = usable_ids
    sections = [WritingSection(
        id="status_overview",
        title=visible_title,
        purpose=overview_purpose,
        supporting_paper_ids=overview_supporting,
        supporting_claim_ids=evidence_ids,
        target_word_count=450,
        heading_level=2,
    )]
    # 子节编号表必须覆盖 spec 的 max_subsections（研究现状放宽到 6），
    # 否则超过表长的子节会直接 IndexError。
    numerals = "一二三四五六七八九十"
    route_section_floor = (
        _MIN_ROUTE_SECTION_REFERENCES
        if int(state.get("required_reference_count") or 0) >= _SECTION_FLOOR_MIN_REVIEW_SIZE
        else 0
    )
    for index, synthesis in enumerate(routes):
        section = _theme_section(
            synthesis,
            CoreDeliverableType.RESEARCH_STATUS,
            title=f"（{numerals[index]}）{synthesis.get('theme_name') or '研究路线'}",
            heading_level=3,
            target_word_count=1000,
            minimum_unique_references=route_section_floor,
        )
        sections.append(section)
    if len(sections) > 1:
        sections[-1].purpose += (
            "；末段综合跨路线的共同进展、差异和有证据支持的不足，不另设研究空白标题"
        )
    hidden = [PlanningNode(
        node_id="status_final_synthesis",
        node_type="final_synthesis",
        label="跨路线综合",
        writing_goal="在研究现状最后一个自然段综合共同进展、路线差异、共性不足及尚未衔接的问题",
        evidence_ids=evidence_ids,
    )]
    return sections, hidden


def _build_related_work_plan(
    state: dict[str, Any],
    spec,
    syntheses: list[dict[str, Any]],
    usable_ids: list[str],
) -> tuple[list[WritingSection], list[PlanningNode]]:
    profile = UserPaperProfile.model_validate(state.get("user_paper_profile") or {})
    max_routes = max(1, (spec.structure.max_subsections if spec.structure else 4) - 1)
    routes = _select_related_themes(
        syntheses, profile, state.get("paper_cards") or [], max_routes=max_routes
    )
    evidence_ids = _all_claim_ids(state.get("paper_cards") or [], usable_ids)
    sections = [WritingSection(
        id="related_work_overview",
        title=spec.structure.visible_title if spec.structure else "相关工作",
        purpose="围绕用户论文问题和方法概括直接相关研究，不扩展为完整领域研究现状",
        supporting_paper_ids=usable_ids,
        supporting_claim_ids=evidence_ids,
        heading_level=2,
    )]
    numerals = "一二三四五六七八九十"
    for index, synthesis in enumerate(routes):
        sections.append(_theme_section(
            synthesis,
            CoreDeliverableType.RELATED_WORK,
            title=f"（{numerals[index]}）{synthesis.get('theme_name') or '相关研究路线'}",
            heading_level=3,
        ))
    if len(sections) < 5:
        sections.append(WritingSection(
            id="gap_and_positioning",
            title=f"（{numerals[len(sections) - 1]}）已有工作与本文研究的关系",
            purpose="根据已有证据说明尚未解决的问题，并仅依据用户资料定位本文研究",
            supporting_paper_ids=usable_ids,
            supporting_claim_ids=evidence_ids,
            heading_level=3,
        ))
    return sections, []


def _build_narrative_review_plan(
    state: dict[str, Any],
    spec,
    syntheses: list[dict[str, Any]],
    usable_ids: list[str],
) -> tuple[list[WritingSection], list[PlanningNode]]:
    routes = _merge_and_select_themes(syntheses, state, max_routes=3)
    evidence_ids = _all_claim_ids(state.get("paper_cards") or [], usable_ids)
    sections = [
        WritingSection(
            id="review_title", title=spec.structure.visible_title if spec.structure else "叙述性综述初稿",
            purpose="概括综述主题、范围与主要认识", supporting_paper_ids=usable_ids,
            supporting_claim_ids=evidence_ids, heading_level=2,
        ),
        WritingSection(
            id="introduction", title="研究背景与问题引入",
            purpose="说明主题价值、问题边界与综述目标", supporting_paper_ids=usable_ids,
            supporting_claim_ids=evidence_ids, heading_level=3,
        ),
    ]
    for synthesis in routes:
        sections.append(_theme_section(
            synthesis, CoreDeliverableType.NARRATIVE_REVIEW,
            title=str(synthesis.get("theme_name") or "研究路线"), heading_level=3,
        ))
    sections.extend([
        WritingSection(
            id="comparison_and_gaps", title="研究比较与证据支持的不足",
            purpose="按统一且有证据的维度比较路线，并归纳有来源的不足",
            supporting_paper_ids=usable_ids, supporting_claim_ids=evidence_ids,
            heading_level=3,
        ),
        WritingSection(
            id="conclusion", title="综合总结", purpose="总结主要认识而不新增事实",
            supporting_paper_ids=usable_ids, supporting_claim_ids=evidence_ids,
            heading_level=3,
        ),
    ])
    hidden = [PlanningNode(
        node_id="review_trajectory",
        node_type="research_trajectory",
        label="研究发展脉络",
        writing_goal="依据年份与显式证据说明问题、方法或证据如何变化，不强造因果演进故事",
        evidence_ids=evidence_ids,
    )]
    return sections[:8], hidden


def _theme_section(
    synthesis: dict[str, Any],
    dtype: CoreDeliverableType,
    *,
    title: str,
    heading_level: int,
    target_word_count: int | None = None,
    minimum_unique_references: int = 0,
) -> WritingSection:
    paper_ids = [str(x) for x in synthesis.get("paper_ids") or []]
    theme_name = str(synthesis.get("theme_name") or "研究路线")
    purpose = "综合该路线的共同问题、主要方法、代表性进展和证据明确支持的不足，不逐篇介绍"
    if dtype == CoreDeliverableType.RESEARCH_STATUS:
        purpose += "；仅在具有可靠地域证据时比较国内外研究，不根据论文语言猜测地域"
    focus_labels = [
        str(label).strip()
        for label in synthesis.get("protected_focus_labels") or []
        if str(label).strip()
    ]
    if focus_labels:
        # 被合并的单篇重点路线在此保留写作义务：证据仍在本节论文集合内，
        # 重点不能因为无法独立成节而从正文消失。
        purpose += "；本节需同时覆盖用户明确的研究重点：" + "、".join(
            dict.fromkeys(focus_labels)
        )
    problems = synthesis.get("reported_problems") or synthesis.get("common_problems") or []
    methods = synthesis.get("reported_methods") or synthesis.get("common_methods") or []
    findings = synthesis.get("reported_findings") or []
    limitations = synthesis.get("author_stated_limitations") or []
    return WritingSection(
        id=f"theme_{synthesis.get('theme_id')}",
        title=title,
        purpose=purpose,
        claims_to_establish=[
            str(item.get("claim"))
            for items in (problems, methods, findings)
            for item in items[:3]
            if item.get("claim")
        ],
        supporting_paper_ids=paper_ids,
        supporting_claim_ids=[
            str(item.get("claim_id"))
            for items in (problems, methods, findings, limitations)
            for item in items
            if item.get("claim_id")
        ],
        comparison_dimensions=synthesis.get("comparison_dimensions") or [],
        target_word_count=target_word_count,
        heading_level=heading_level,
        minimum_unique_references=minimum_unique_references,
    )


def _all_claim_ids(cards: list[dict[str, Any]], allowed_ids: list[str]) -> list[str]:
    allowed = set(allowed_ids)
    return list(dict.fromkeys(
        str(claim.get("evidence_id"))
        for card in cards if str(card.get("paper_id")) in allowed
        for claims in (card.get("field_claims") or {}).values()
        for claim in claims if claim.get("explicitly_reported") and claim.get("evidence_id")
    ))


def _select_related_themes(
    syntheses: list[dict[str, Any]],
    profile: UserPaperProfile,
    cards: list[dict[str, Any]],
    *,
    max_routes: int = 3,
) -> list[dict[str, Any]]:
    query = " ".join(filter(None, [
        profile.research_problem,
        profile.proposed_method,
        profile.research_direction,
        profile.research_object,
        profile.target_task,
        profile.application_scenario,
    ]))
    tokens = _tokens(query)
    cards_by_id = {str(card.get("paper_id")): card for card in cards}
    scored = []
    for synthesis in syntheses:
        text = " ".join([
            str(synthesis.get("theme_name") or ""),
            *[
                str(item.get("claim") or "")
                for items in (
                    synthesis.get("reported_problems") or synthesis.get("common_problems") or [],
                    synthesis.get("reported_methods") or synthesis.get("common_methods") or [],
                )
                for item in items
            ],
            *[
                str(cards_by_id.get(str(pid), {}).get("title") or "")
                for pid in synthesis.get("paper_ids") or []
            ],
        ])
        score = len(tokens & _tokens(text)) / max(1, len(tokens))
        scored.append((score, synthesis))
    selected = [
        item for score, item in sorted(scored, key=lambda pair: pair[0], reverse=True)
        if score > 0
    ]
    return (selected or [
        item for _, item in sorted(scored, key=lambda pair: pair[0], reverse=True)
    ])[:max_routes]


def _merge_and_select_themes(
    syntheses: list[dict[str, Any]],
    state: dict[str, Any],
    *,
    max_routes: int,
    evidence_paper_count: int = 0,
) -> list[dict[str, Any]]:
    """按用户重点、主题相似度和证据量选择并归并为有限可见路线。

    ``evidence_paper_count`` 给出时，可见路线数按可用证据自适应缩放：
    证据不足却用满 spec 预算只会产出“两篇一节”的碎片小节。缩放存在下界，
    不得挤掉用户明确要求的内容——既包括已编译成路线的证据要求
    （required_routes），也包括语义帧里的显式研究重点（required_focuses，
    其覆盖率由 plan.undercovered_focuses 对外承诺）。
    """
    if max_routes <= 0:
        return []
    required_focuses = [
        str(item)
        for item in (state.get("research_semantic_frame") or {}).get("required_focuses") or []
        if str(item).strip()
    ]
    merge_diagnostics: list[dict[str, Any]] = []

    def _finalize(routes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """统一收口：稀疏/溢出路线并入、语义化命名、并入诊断写回 state。"""
        result = _semanticize_routes(_rebalance_sparse_routes(
            routes,
            max_routes=max_routes,
            protected_focuses=required_focuses,
            semantic_frame=state.get("research_semantic_frame") or {},
            cards=state.get("paper_cards") or [],
            merge_diagnostics=merge_diagnostics,
        ), state)
        if merge_diagnostics:
            # 一次生成会为每份交付物各调用一次本函数，事件需累加而非覆盖。
            state["route_merge_diagnostics"] = [
                *(state.get("route_merge_diagnostics") or []),
                *merge_diagnostics,
            ]
        return result

    required_routes = _required_evidence_routes(state, syntheses)
    if evidence_paper_count > 0:
        explicit_focus_count = len([
            item for item in
            (state.get("research_semantic_frame") or {}).get("required_focuses") or []
            if str(item).strip()
        ])
        scaled = int(evidence_paper_count) // _MIN_PAPERS_PER_SUBSECTION
        max_routes = min(
            max_routes,
            max(1, len(required_routes), explicit_focus_count, scaled),
        )
    # 全局锚点守卫：required 路线描述的是"用户点名的任务/阶段"，覆盖全池
    # 是常态而非细分。原守卫只防"恰好 1 条"的情况——语义帧产出 2 条宽泛
    # 要求（如 research_object + method 各命中 89/90 篇）时会漏过，写作
    # 计划随之用两个 catch-all 小节替换掉全部证据验证路线，正文退化为
    # 两节同名复述（2026-08-23 会话实测）。判据泛化为：任一 required 路线
    # 覆盖 syntheses 论文的 75% 以上，或全部 required 路线合并后覆盖 75%
    # 以上且彼此高度重叠（并集/最大单线 ≥ 0.8）——都说明它们是全局锚点
    # 而非细分方向，syntheses（证据验证、互斥）应优先。
    total_syntheses_papers = len({
        str(p)
        for s in syntheses
        for p in s.get("paper_ids") or []
    })
    required_paper_counts = [
        len(route.get("paper_ids") or []) for route in required_routes
    ]
    required_union = len({
        str(paper_id)
        for route in required_routes
        for paper_id in route.get("paper_ids") or []
    })
    threshold = max(2, int(total_syntheses_papers * 0.75))
    any_dominant = any(count >= threshold for count in required_paper_counts)
    collectively_monolithic = bool(
        required_routes
        and len(required_routes) > 1
        and required_union >= threshold
        and required_union <= max(1, int(max(required_paper_counts, default=0) / 0.8))
    )
    is_monolithic_anchor = bool(
        required_routes
        and len(syntheses) >= 2
        and (any_dominant or collectively_monolithic)
    )
    if required_routes and not is_monolithic_anchor:
        # 证据要求是“覆盖约束”，不应直接替换动态主题。
        # 旧逻辑把每个 requirement 编译成独立路线，导致单篇证据路线占用所有子节名额，而真正的主题路线被丢弃。
        # 将稀疏要求附着到证据量充足的主题路线，仅保留能独立支撑比较的路线；不依赖任何领域词表。
        synthesis_pool = [dict(item) for item in syntheses]
        selected_routes: list[dict[str, Any]] = []
        deferred_routes: list[dict[str, Any]] = []

        def _paper_set(item: dict[str, Any]) -> set[str]:
            return {str(value) for value in item.get("paper_ids") or [] if str(value)}

        for required in required_routes:
            required_ids = _paper_set(required)
            if len(required_ids) >= _MIN_PAPERS_PER_SUBSECTION:
                selected_routes.append(dict(required))
                continue
            # 单篇/双篇证据不足以支撑独立综合，尝试附着到与其证据重叠最高的动态路线。
            candidates = [
                (len(required_ids & _paper_set(item)), index, item)
                for index, item in enumerate(synthesis_pool)
                if _paper_set(item)
            ]
            overlap, index, partner = max(candidates, default=(0, -1, None))
            if partner is not None and overlap > 0:
                merged = _merge_theme_records(partner, required)
                # 保留稀疏要求的语义标签，使附着后的路线仍能反映用户明确点名的研究重点。
                # 名称来自当前数据，不维护任何领域专属映射；名称过长时改记
                # protected_focus_labels，避免把标题截断成半个词。
                merged = _absorb_focus_label(
                    merged, str(required.get("theme_name") or "").strip()
                )
                synthesis_pool[index] = merged
            else:
                deferred_routes.append(dict(required))

        # 先保留证据实际支持的主题路线，再用剩余名额添加稀疏要求。
        already_ids = {
            str(value) for route in selected_routes for value in route.get("paper_ids") or []
        }
        ranked_pool = sorted(
            synthesis_pool,
            key=lambda item: len(_paper_set(item)),
            reverse=True,
        )
        overflow_routes: list[dict[str, Any]] = []
        for item in ranked_pool:
            if not (_paper_set(item) - already_ids):
                continue
            if len(selected_routes) >= max_routes:
                overflow_routes.append(item)
                continue
            selected_routes.append(item)
            already_ids.update(_paper_set(item))
        for item in deferred_routes:
            if len(selected_routes) >= max_routes:
                overflow_routes.append(item)
                continue
            selected_routes.append(item)
        # WHY: 名额已满时旧实现直接 break，未入选路线连同其独有 paper_ids 一起
        # 消失（实测 5 条 KEEP 路线在 max_routes=3 下只落地 2 个小节、45 篇证据
        # 只被引用 25 篇）。溢出路线并入语义最相近的存活路线：不再单独成节，
        # 但证据、论点与重点标签都保留。
        for item in overflow_routes:
            _merge_overflow_route(selected_routes, item, diagnostics=merge_diagnostics)
        return _finalize(selected_routes)
    if not syntheses:
        return []
    focus = " ".join([
        str(state.get("user_query") or ""),
        str(state.get("canonical_topic") or state.get("topic") or ""),
        str((state.get("selected_scope") or {}).get("description") or ""),
        *[str(item) for item in (state.get("research_semantic_frame") or {}).get("task_chain") or []],
        *[str(item) for item in (state.get("research_semantic_frame") or {}).get("required_focuses") or []],
    ])
    focus_tokens = _tokens(focus)

    ranked = sorted(
        enumerate(syntheses),
        key=lambda pair: (
            len(focus_tokens & _tokens(_route_semantic_text(pair[1]))),
            len(set(str(x) for x in pair[1].get("paper_ids") or [])),
            -pair[0],
        ),
        reverse=True,
    )
    selected_indices: list[int] = []
    # 先为明确重点保留最匹配路线，再按综合排序填满。多个重点可以由同一路线覆盖。
    for aliases in _focus_alias_groups(
        required_focuses, state.get("research_semantic_frame") or {}
    ):
        if len(selected_indices) >= max_routes:
            break
        candidates = [
            (len(aliases & _tokens(_route_semantic_text(item))), index)
            for index, item in enumerate(syntheses)
            if index not in selected_indices
        ]
        score, index = max(candidates, default=(0, -1))
        if score > 0:
            selected_indices.append(index)
    for index, _ in ranked:
        if len(selected_indices) >= max_routes:
            break
        if index not in selected_indices:
            selected_indices.append(index)

    seeds = [dict(syntheses[index]) for index in selected_indices]
    if len(syntheses) <= max_routes:
        return _finalize(seeds)

    seed_tokens = [_tokens(_route_semantic_text(item)) for item in seeds]
    for index, item in ranked:
        if index in selected_indices:
            continue
        tokens = _tokens(_route_semantic_text(item))
        similarities = [
            len(tokens & existing) / max(1, len(tokens | existing))
            for existing in seed_tokens
        ]
        if max(similarities, default=0.0) > 0:
            target_index = max(range(len(seeds)), key=lambda index: similarities[index])
        else:
            target_index = min(
                range(len(seeds)),
                key=lambda index: len(seeds[index].get("paper_ids") or []),
            )
        seeds[target_index] = _merge_theme_records(seeds[target_index], item)
        seed_tokens[target_index] |= tokens
        _record_route_merge(
            merge_diagnostics,
            source=item,
            target=seeds[target_index],
            reason="theme_similarity_merge",
        )
    return _finalize(seeds)


def _route_semantic_text(item: dict[str, Any]) -> str:
    """路线名与其问题/方法/发现文本，用于路线之间的相似度打分。"""
    return " ".join([
        str(item.get("theme_name") or ""),
        *[
            str(claim.get("claim") or claim.get("statement") or "")
            for claims in (
                item.get("reported_problems") or item.get("common_problems") or [],
                item.get("reported_methods") or item.get("common_methods") or [],
                item.get("reported_findings") or [],
            )
            for claim in claims
            if isinstance(claim, dict)
        ],
    ])


def _record_route_merge(
    diagnostics: list[dict[str, Any]] | None,
    *,
    source: dict[str, Any],
    target: dict[str, Any],
    reason: str,
) -> None:
    """记录一次路线并入事件。

    WHY: 并入/丢弃此前完全不可观测（步骤里只记 len(plans)），排查"证据被静默
    丢弃"只能靠 grep INFO 日志。
    """
    if diagnostics is None:
        return
    diagnostics.append({
        "reason": reason,
        "merged_route": str(source.get("theme_name") or source.get("theme_id") or ""),
        "target_route": str(target.get("theme_name") or ""),
        "migrated_paper_count": len(source.get("paper_ids") or []),
    })


def _merge_overflow_route(
    kept: list[dict[str, Any]],
    source: dict[str, Any],
    *,
    diagnostics: list[dict[str, Any]] | None = None,
    reason: str = "subsection_quota_overflow",
) -> None:
    """把一条超出小节名额的路线并入语义最相近的存活路线（就地修改 kept）。

    WHY: 旧实现按名额直接丢弃溢出路线，被丢路线的 paper_ids 随之消失，实测
    5 条 KEEP 路线在 max_routes=3 下只落地 2 个小节、45 篇证据只被引用 25 篇。
    并入后证据、论点与重点标签都保留，只是不再单独成节。
    """
    if not kept:
        kept.append(dict(source))
        return
    source_tokens = _tokens(_route_semantic_text(source))
    partner_index = max(
        range(len(kept)),
        key=lambda index: (
            len(source_tokens & _tokens(_route_semantic_text(kept[index]))),
            len(kept[index].get("paper_ids") or []),
            -index,
        ),
    )
    combined = _absorb_focus_label(
        _merge_theme_records(kept[partner_index], source),
        str(source.get("theme_name") or "").strip(),
    )
    kept[partner_index] = combined
    _record_route_merge(
        diagnostics, source=source, target=combined, reason=reason
    )


def _rebalance_sparse_routes(
    routes: list[dict[str, Any]],
    *,
    max_routes: int,
    protected_focuses: list[str] | None = None,
    semantic_frame: dict[str, Any] | None = None,
    cards: list[dict[str, Any]] | None = None,
    merge_diagnostics: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """把单篇论文路线尽量并入相邻路线，避免把单文献扩写成独立研究路线。"""
    if len(routes) <= 1:
        return routes
    if max_routes <= 0:
        return []

    merged = [dict(route) for route in routes]
    focus_groups = _focus_alias_groups(protected_focuses or [], semantic_frame or {})

    def is_protected(item: dict[str, Any]) -> bool:
        if not focus_groups:
            return False
        return any(group & _tokens(_route_semantic_text(item)) for group in focus_groups)

    while len(merged) > 1:
        sparse_indexes = [
            index for index, item in enumerate(merged)
            # A two-paper route already has independent corroboration.  Only a
            # singleton is structurally too sparse to become a formal section.
            #
            # WHY: 命中用户明确重点也不能豁免合并。单篇路线一旦独立成节，
            # 渲染层只能产出「本节纳入 1 篇文献」的书目式罗列（2026-08-30
            # 实测）。重点覆盖改由 _absorb_focus_label 迁移到合并后的路线：
            # 证据仍在本节，写作义务与覆盖判定都不丢。
            if len(item.get("paper_ids") or []) < 2
        ]
        if not sparse_indexes:
            break
        sparse_index = min(
            sparse_indexes,
            key=lambda index: (len(merged[index].get("paper_ids") or []), index),
        )
        if len(merged) <= 1:
            break
        source = merged[sparse_index]
        source_tokens = _tokens(_route_semantic_text(source))
        partner_index = max(
            (index for index in range(len(merged)) if index != sparse_index),
            key=lambda index: (
                len(source_tokens & _tokens(_route_semantic_text(merged[index]))),
                len(merged[index].get("paper_ids") or []),
                -index,
            ),
        )
        if partner_index == sparse_index:
            break
        combined = _merge_theme_records(merged[partner_index], source)
        if is_protected(source):
            combined = _absorb_focus_label(
                combined, str(source.get("theme_name") or "").strip()
            )
        merged[partner_index] = combined
        merged.pop(sparse_index)
        _record_route_merge(
            merge_diagnostics,
            source=source,
            target=combined,
            reason="sparse_single_paper_route",
        )
        if len(merged) <= max_routes and all(len(item.get("paper_ids") or []) >= 2 for item in merged):
            break

    # WHY: 旧实现此处是 merged[:max_routes] 裸切片，被丢路线的 paper_ids 随之
    # 消失（实测 5 条 KEEP 路线在 max_routes=3 下只落地 2 个小节）。溢出路线
    # 改为并入语义最相近的存活路线：不再单独成节，但证据与论点都保留。
    while len(merged) > max_routes:
        overflow_index = min(
            range(len(merged)),
            key=lambda index: (len(merged[index].get("paper_ids") or []), -index),
        )
        source = merged.pop(overflow_index)
        _merge_overflow_route(merged, source, diagnostics=merge_diagnostics)
    return _semanticize_routes(merged, {"paper_cards": cards or []})


def _required_evidence_routes(
    state: dict[str, Any],
    syntheses: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """把已满足的直接证据要求编译成动态路线，不预设领域标题。"""
    frame = state.get("research_semantic_frame") or {}
    requirements = [
        item for item in frame.get("evidence_requirements") or []
        if item.get("route_required", True)
    ]
    if not requirements:
        return []
    from app.agent.evidence_roles import evidence_coverage

    coverage = evidence_coverage(frame, state.get("paper_cards") or [])
    groups: dict[str, list[dict[str, Any]]] = {}
    for requirement in requirements:
        group = str(
            requirement.get("route_group")
            or requirement.get("evidence_role")
            or requirement.get("requirement_id")
        )
        groups.setdefault(group, []).append(requirement)

    routes: list[dict[str, Any]] = []
    for group, members in groups.items():
        labels = list(dict.fromkeys(
            str(item.get("label") or "").strip() for item in members
            if str(item.get("label") or "").strip()
        ))
        paper_ids = list(dict.fromkeys(
            paper_id
            for item in members
            for paper_id in coverage.get("matched_paper_ids", {}).get(
                str(item.get("requirement_id") or ""), []
            )
        ))
        if not paper_ids:
            continue
        overlapping = sorted(
            (
                item for item in syntheses
                if set(str(x) for x in item.get("paper_ids") or []) & set(paper_ids)
            ),
            key=lambda item: len(
                set(str(x) for x in item.get("paper_ids") or []) & set(paper_ids)
            ),
            reverse=True,
        )
        route = dict(overlapping[0]) if overlapping else {}
        semantic_name = _semantic_route_name(labels, members, route)
        route.update({
            "theme_id": f"required_{_safe_id(group)}",
            "theme_name": semantic_name,
            "paper_ids": paper_ids,
            "evidence_requirement_ids": [
                str(item.get("requirement_id") or "") for item in members
            ],
        })
        routes.append(route)
    return routes


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")
    return safe or "stage"


def _merge_theme_records(
    primary: dict[str, Any],
    secondary: dict[str, Any],
) -> dict[str, Any]:
    """归并动态主题但保留主主题名称和全部可追溯证据。"""
    merged = dict(primary)
    fields = (
        "reported_problems", "reported_methods", "shared_problems", "shared_methods",
        "common_problems", "common_methods", "reported_findings",
        "author_stated_limitations", "synthesized_gaps",
    )
    merged["theme_id"] = f"{primary.get('theme_id')}_merged"
    merged["paper_ids"] = list(dict.fromkeys([
        *[str(item) for item in primary.get("paper_ids") or []],
        *[str(item) for item in secondary.get("paper_ids") or []],
    ]))
    merged["comparison_dimensions"] = list(dict.fromkeys([
        *[str(item) for item in primary.get("comparison_dimensions") or []],
        *[str(item) for item in secondary.get("comparison_dimensions") or []],
    ]))
    merged["evidence_requirement_ids"] = list(dict.fromkeys([
        *[str(item) for item in primary.get("evidence_requirement_ids") or []],
        *[str(item) for item in secondary.get("evidence_requirement_ids") or []],
    ]))
    merged["protected_focus_labels"] = list(dict.fromkeys([
        *[str(item) for item in primary.get("protected_focus_labels") or []],
        *[str(item) for item in secondary.get("protected_focus_labels") or []],
    ]))
    for field in fields:
        merged[field] = [
            *(primary.get(field) or []),
            *(secondary.get(field) or []),
        ]
    return merged


def _absorb_focus_label(route: dict[str, Any], source_name: str) -> dict[str, Any]:
    """把被合并的用户重点路线名记入合并结果，保证重点不因合并而消失。

    WHY: 标题只在合并后仍不超过三个操作数、且长度仍可读时并入重点名；其余
    情况记入 ``protected_focus_labels``，由 ``_theme_section`` 写进章节写作目标，
    ``_evaluate_focus_coverage`` 也据此判定该重点已被覆盖。
    """
    if not source_name:
        return route
    absorbed = dict(route)
    current = str(absorbed.get("theme_name") or "").strip()
    existing_labels = [
        str(item) for item in absorbed.get("protected_focus_labels") or []
    ]
    absorbed["protected_focus_labels"] = list(dict.fromkeys([
        *existing_labels,
        source_name,
    ]))
    # 标题最多三个操作数：多级链式拼接（A与B与C与D）不是可读的学术小节名。
    # WHY: 旧护栏只数字数（≤30）并用 existing_labels 记"吸收过几次"，但
    # _semantic_route_name 自身就会产出"A与B"两段名且不计入 existing_labels，
    # 于是第一次吸收即可把两段名拼成四段（2026-09-01 实测 21 字标题放行）。
    # 判据改为直接数合并后的"与"分段数。
    if (
        current
        and source_name not in current
        and current.count("与") + source_name.count("与") <= 1
        and len(current) + len(source_name) + 1 <= 30
    ):
        absorbed["theme_name"] = f"{current}与{source_name}"
    return absorbed


def _semantic_route_name(
    labels: list[str],
    members: list[dict[str, Any]],
    route: dict[str, Any],
) -> str:
    """从路线标签、既有路线名称和成员数据提取规范学术路线名称。"""
    explicit_labels = [str(label).strip() for label in labels if str(label).strip()]
    if explicit_labels:
        # 清洗内部标签后缀（如"相关论文证据"、"相关文献"、"相关论文"、"相关证据"、
        # "相关研究"等），只保留核心概念作为用户可见章节标题。
        # WHY: 证据要求标签是内部检索/覆盖用语；直接进标题会产出
        # "课堂行为分析相关文献与师生行为相关文献"这类不像小节名的标题
        # （2026-08-30 实测）。"相关"前缀是判据，避免误伤"灰色文献"这类实名。
        _INTERNAL_SUFFIXES = re.compile(
            r"(?:相关(?:论文)?文献|(?:相关)?(?:论文证据|论文|证据|研究)+)$"
        )
        cleaned = []
        for label in explicit_labels:
            clean = _INTERNAL_SUFFIXES.sub("", label).strip()
            if clean and not _is_low_signal_theme_name(clean):
                cleaned.append(clean)
        if cleaned:
            unique = list(dict.fromkeys(cleaned))
            result = "与".join(unique[:3])
            if len(result) > 30:
                result = "与".join(unique[:2])
            if not _is_low_signal_theme_name(result):
                return result

    # 1. 优先使用 route 自身携带的高质量名称（如来自 validated_routes / provisional_framework）
    existing = str(route.get("theme_name") or route.get("name") or "").strip()
    if existing and not _is_low_signal_theme_name(existing):
        return existing

    # 从所有可用文本提取词项，优先长词
    evidence_text = " ".join(
        labels
        + [existing]
        + [
            str(item.get("label") or "") + " " + str(item.get("route_group") or "")
            for item in members
        ]
    )
    tokens = sorted(
        [t for t in _tokens(evidence_text) if len(t) > 1 and not _is_low_signal_theme_name(t)],
        key=lambda t: -len(t),
    )
    salient = tokens[:2]
    if len(salient) == 2:
        return f"{salient[0]}与{salient[1]}研究"
    if len(salient) == 1:
        return f"{salient[0]}研究"
    return existing or "研究路线"


def _semanticize_routes(
    routes: list[dict[str, Any]],
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    """纯数据驱动：从每个路线包含的论文中提取共同词项生成路线名称。"""
    cards_by_id = {
        str(card.get("paper_id") or ""): card
        for card in state.get("paper_cards") or []
        if card.get("paper_id")
    }
    semanticized: list[dict[str, Any]] = []
    for route in routes:
        item = dict(route)
        paper_ids = [str(pid) for pid in item.get("paper_ids") or [] if str(pid)]
        if str(item.get("theme_name") or "").strip() and not _is_low_signal_theme_name(
            str(item.get("theme_name") or "")
        ):
            semanticized.append(item)
            continue
        # 用该路线的全部论文数据驱动命名
        cluster_cards = [
            cards_by_id[paper_id] for paper_id in paper_ids
            if paper_id in cards_by_id
        ]
        item["theme_name"] = semantic_route_label(
            cluster_cards,
            str(item.get("theme_name") or ""),
        )
        semanticized.append(item)
    return semanticized


def _is_low_signal_theme_name(name: str) -> bool:
    cleaned = re.sub(r"\s+", " ", str(name or "")).strip()
    if not cleaned:
        return True
    lowered = cleaned.casefold()
    if lowered in {
        "image", "images", "video", "videos", "text", "audio", "speech", "sensor", "sensors",
        "signal", "signals", "pose", "skeleton", "rgb", "visual", "experiment", "dataset",
        "method", "methods", "task", "topic", "cluster", "route", "dna", "rna", "gene", "protein",
        "tabular", "molecule",
        "期刊论文", "会议论文", "会议短文", "预印本", "学位论文",
        "journal article", "conference paper", "preprint", "thesis",
        "journal_article", "conference_paper", "conference_short_paper",
    }:
        return True
    if re.fullmatch(r"[a-z_\- ]{1,18}", lowered) and len(lowered.split()) <= 2:
        return True
    if len(cleaned) <= 2 and not re.search(r"[\u4e00-\u9fff]{2,}", cleaned):
        return True
    return bool(re.search(r"待补充证据|未命名|未知|unknown|other|misc", lowered))


def _tokens(text: str) -> set[str]:
    lowered = str(text or "").lower()
    english = set(re.findall(r"[a-z][a-z0-9_-]{2,}", lowered))
    chinese = re.findall(r"[\u4e00-\u9fff]", lowered)
    return english | {"".join(chinese[i:i + 2]) for i in range(max(0, len(chinese) - 1))}


_SEMANTIC_ID_PREFIXES = {
    "obj", "object", "domain", "method", "action", "target", "ev", "req",
}


def _semantic_identifier_tokens(value: Any) -> set[str]:
    """拆分语义帧 ID；只用于 ID 关联，不改变路线排序使用的 ``_tokens``。"""
    parts = {
        item for item in re.split(r"[^a-z0-9]+", str(value or "").casefold())
        if len(item) >= 2
    }
    return parts - _SEMANTIC_ID_PREFIXES


def _focus_alias_tokens(values: list[str]) -> set[str]:
    tokens = _tokens(" ".join(values))
    for value in values:
        identifier_parts = _semantic_identifier_tokens(value)
        tokens.update(identifier_parts)
        if identifier_parts:
            tokens.add("_".join(sorted(identifier_parts)))
    return tokens


def _focus_alias_groups(
    focuses: list[str],
    semantic_frame: dict[str, Any] | None = None,
) -> list[set[str]]:
    """从本次 LLM 语义帧提取别名，不维护领域专属映射表。"""
    frame = semantic_frame or {}
    concepts = [
        item
        for field in (
            "application_domains", "research_objects", "methods",
            "research_actions", "analysis_targets",
        )
        for item in frame.get(field) or []
        if isinstance(item, dict)
    ]
    requirements = [
        item for item in frame.get("evidence_requirements") or []
        if isinstance(item, dict)
    ]
    groups: list[set[str]] = []
    for focus in focuses:
        aliases = [str(focus)]
        focus_id_tokens = _semantic_identifier_tokens(focus)
        for item in concepts:
            values = [
                str(item.get(key) or "")
                for key in ("id", "label", "surface_text")
            ]
            values.extend(str(value) for value in item.get("aliases") or [])
            concept_id_tokens = _semantic_identifier_tokens(item.get("id"))
            if focus_id_tokens and focus_id_tokens == concept_id_tokens:
                aliases.extend(values)
        for item in requirements:
            values = [
                str(item.get("requirement_id") or ""),
                str(item.get("label") or ""),
                *[str(value) for value in item.get("aliases") or []],
                *[str(value) for value in item.get("context_aliases") or []],
            ]
            source_ids = [str(value) for value in item.get("source_ids") or []]
            requirement_matches = any(
                focus_id_tokens
                and focus_id_tokens == _semantic_identifier_tokens(value)
                for value in [item.get("requirement_id"), *source_ids]
            )
            if requirement_matches:
                aliases.extend(values)
        groups.append(_focus_alias_tokens(aliases))
    return groups


def _evaluate_focus_coverage(
    focuses: list[str],
    sections: list[WritingSection],
    semantic_frame: dict[str, Any] | None = None,
) -> tuple[list[str], list[str]]:
    if not focuses:
        return [], []
    plan_text = " ".join([
        str(section.title or "") + " " + str(section.purpose or "") + " "
        + " ".join(section.claims_to_establish or [])
        for section in sections
    ])
    plan_tokens = _tokens(plan_text)
    groups = _focus_alias_groups(focuses, semantic_frame)
    covered = [focus for focus, aliases in zip(focuses, groups) if aliases & plan_tokens]
    return covered, [focus for focus in focuses if focus not in covered]


def _is_fallback_theme(name: Any) -> bool:
    """兜底分类只能用于内部诊断，不能编译成正式写作章节。"""
    return bool(_FALLBACK_THEME_RE.search(str(name or "")))
