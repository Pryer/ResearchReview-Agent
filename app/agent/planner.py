"""任务规划模块。

整合意图识别和槽位抽取的结果，生成执行计划
（包含工作流名称、检索关键词、检索参数等）。
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.core.json_utils import parse_json_object as _parse_json_object_robust

from app.core.logger import get_logger
from app.schemas.agent_schema import IntentType, SlotResult

logger = get_logger(__name__)

# ---------- 工作流映射 ----------
WORKFLOW_MAP: dict[str, str] = {
    IntentType.SEARCH_PAPERS.value: "paper_search_workflow",
    IntentType.GENERATE_REVIEW.value: "literature_review_workflow",
    IntentType.READ_PAPER.value: "single_paper_reading_workflow",
    IntentType.COMPARE_PAPERS.value: "paper_comparison_workflow",
    IntentType.GENERATE_REFERENCES.value: "citation_generation_workflow",
    IntentType.EXTRACT_PAPER_CARD.value: "paper_card_extraction_workflow",
    IntentType.FIND_DATASETS.value: "dataset_search_workflow",
    IntentType.FIND_TRENDS.value: "trend_analysis_workflow",
    IntentType.GENERAL_QA.value: "general_qa_workflow",
}


def resolve_language_branch_zh_ratio(semantic_frame: Any) -> tuple[float, str]:
    """把语义解析的语言倾向判断映射为中文分支配额。

    LLM 只输出 zh_dominant/balanced/en_dominant 枚举，比例由此处映射并钳制在
    ``[language_branch_zh_ratio_min, language_branch_zh_ratio_max]`` 内：直接
    让模型输出数值会出现 0.9/0.05 这类取值，把某一语种分支压到失效。

    Returns:
        ``(zh_ratio, reason)``；reason 仅用于诊断记录。
    """
    from app.core.config import get_settings

    settings = get_settings()
    default_ratio = float(settings.language_branch_zh_ratio)
    if not settings.language_branch_affinity_enabled or semantic_frame is None:
        return default_ratio, "affinity_disabled"

    if isinstance(semantic_frame, dict):
        raw_affinity = semantic_frame.get("language_affinity")
        reason = str(semantic_frame.get("language_affinity_reason") or "").strip()
    else:
        raw_affinity = getattr(semantic_frame, "language_affinity", None)
        reason = str(getattr(semantic_frame, "language_affinity_reason", "") or "").strip()
    affinity = str(getattr(raw_affinity, "value", raw_affinity) or "").strip().lower()

    mapping = {
        "zh_dominant": float(settings.language_branch_zh_ratio_zh_dominant),
        "balanced": default_ratio,
        "en_dominant": float(settings.language_branch_zh_ratio_en_dominant),
    }
    if affinity not in mapping:
        return default_ratio, f"unknown_affinity:{affinity or 'missing'}"

    ratio = min(
        float(settings.language_branch_zh_ratio_max),
        max(float(settings.language_branch_zh_ratio_min), mapping[affinity]),
    )
    return ratio, f"{affinity}: {reason}" if reason else affinity


def build_search_plan(
    user_query: str,
    intent: IntentType | str,
    slots: SlotResult,
    llm=None,
    semantic_frame=None,
) -> Dict[str, Any]:
    """整合意图和槽位，生成完整执行计划。

    Args:
        user_query: 原始用户请求。
        intent: 意图类型。
        slots: 槽位抽取结果。
        llm: 可选 LLM 客户端。

    Returns:
        执行计划字典，包含 workflow、keywords、params 等。
    """
    intent_str = intent.value if isinstance(intent, IntentType) else intent
    workflow = choose_workflow(intent_str)
    # 无领域内容的主题（"近三年综述"式残片）不进入检索规划，回退到
    # 用户原文，让关键词生成的 LLM 路径从完整请求里恢复主题。
    from app.agent.slot_extractor import has_topic_content

    plan_topic = slots.topic if has_topic_content(slots.topic or "") else None
    strategy = generate_search_strategy(
        plan_topic or user_query,
        llm,
        user_query=user_query,
    )
    keywords = strategy["keywords"]
    keyword_batches = strategy.get("keyword_batches") or []
    core_keywords = list(dict.fromkeys(
        [plan_topic or user_query]
        + [str(item).strip() for batch in keyword_batches if batch.get("type") == "exact" for item in batch.get("keywords") or [] if str(item).strip()]
    ))
    expanded_keywords = [
        str(item).strip() for batch in keyword_batches
        if batch.get("type") in {"broader", "variant"}
        for item in batch.get("keywords") or [] if str(item).strip()
    ]
    search_branches = []
    if semantic_frame is not None:
        from app.agent.search_plan_builder import (
            _HIGH_CITATION_TARGET,
            build_semantic_search_branches,
            prioritized_branch_queries,
        )
        from app.schemas.research_plan_schema import ResearchSemanticFrame

        frame = (
            semantic_frame
            if isinstance(semantic_frame, ResearchSemanticFrame)
            else ResearchSemanticFrame.model_validate(semantic_frame)
        )
        retrieval_target = slots.retrieval_target or slots.required_reference_count
        search_branches = build_semantic_search_branches(
            frame, retrieval_target=retrieval_target,
        )
        # 高引用目标下分支数会随方法学子方向扩展，同步放宽关键词预算，
        # 确保每个细分分支的首选查询都进入候选池（检索节点再按分支多样性选取）。
        keyword_limit = (
            max(10, min(24, len(search_branches) + 6))
            if retrieval_target and int(retrieval_target) >= _HIGH_CITATION_TARGET
            else 10
        )
        keywords = _deduplicate_keywords(
            [*prioritized_branch_queries(search_branches, limit=keyword_limit), *keywords],
            limit=keyword_limit,
        )
        if strategy.get("planning_error") and any(
            re.search(r"[A-Za-z]{4,}", keyword) for keyword in keywords
        ):
            strategy["planning_error"] = None

    plan: Dict[str, Any] = {
        "workflow": workflow,
        "intent": intent_str,
        "keywords": keywords,
        "core_keywords": core_keywords,
        "expanded_keywords": list(dict.fromkeys(expanded_keywords)),
        # 检索批次（exact→broader→variant）来自关键词生成工具的 type
        # 元数据，供检索层按"完整表达先检、外扩后检"的顺序派发。
        "keyword_batches": strategy.get("keyword_batches") or [],
        # 多分支查询的目标是并集召回，不能再用一组全局 AND 概念把领域基础或
        # 方法基础分支误删；各分支自己的概念约束保存在 search_branches 中。
        "topic_anchors": (
            []
            if (
                len(search_branches) > 1
                or any(branch.constraint_level == "exploratory" for branch in search_branches)
            )
            else strategy["topic_anchors"]
        ),
        # 多分支时不能把全局锚点用作逐篇硬过滤，但仍需保留其教育/主题
        # 语义，供查询排序和 search_drift 诊断使用。
        "semantic_topic_anchors": strategy["topic_anchors"],
        "planning_error": strategy["planning_error"],
        "search_branches": [branch.model_dump(mode="json") for branch in search_branches],
        "topic": plan_topic or user_query,
        "start_year": slots.start_year,
        "end_year": slots.end_year,
        "max_papers": slots.max_papers,
        "required_reference_count": slots.required_reference_count,
        "retrieval_target": slots.retrieval_target,
        "generation_limit": slots.generation_limit,
        "year_range_explicit": slots.year_range_explicit,
        "strict_year_range": slots.strict_year_range,
        "max_papers_explicit": slots.max_papers_explicit,
        "requested_sections": slots.requested_sections,
        "language": slots.language,
        "citation_style": slots.citation_style,
    }

    zh_ratio, zh_ratio_reason = resolve_language_branch_zh_ratio(semantic_frame)
    plan["language_branch_zh_ratio"] = zh_ratio
    plan["language_branch_zh_ratio_reason"] = zh_ratio_reason

    logger.info(
        "Plan built: workflow=%s, topic=%s, required_reference_count=%d, zh_ratio=%.2f (%s)",
        workflow, plan["topic"], slots.required_reference_count, zh_ratio, zh_ratio_reason,
    )
    return plan


def build_screening_protocol(
    *,
    original_query: str,
    user_query: str,
    topic: str,
    conversation_history: list[dict[str, Any]] | None,
    selected_scope: dict[str, Any] | None,
    semantic_frame: dict[str, Any] | None,
    search_branches: list[dict[str, Any]] | None,
    topic_anchors: list[list[str]] | None = None,
    llm=None,
) -> dict[str, Any]:
    """由多轮上下文生成筛选协议；失败时返回基于 topic 的最小保守协议。

    检索词用于召回，不再自动升级为逐篇硬条件。
    硬排除仅用于确定性条件（重复、年份、语言等），不做词法判断。
    """
    from app.schemas.screening_schema import ScreeningProtocol

    fallback = _enhanced_fallback_screening_protocol(
        topic=topic,
        search_branches=search_branches,
        semantic_frame=semantic_frame,
    )
    history = [
        {
            "role": str(item.get("role") or ""),
            "content": str(item.get("content") or "")[:2000],
            "type": str(item.get("type") or ""),
        }
        for item in (conversation_history or [])[-12:]
        if isinstance(item, dict) and str(item.get("content") or "").strip()
    ]
    # 单轮请求已由搜索策略 LLM 处理；筛选协议 LLM 专门用于多轮确认后的
    # 语义合并，避免所有请求无条件增加一次控制面调用。
    if llm is None or (not selected_scope and len(history) < 2):
        return fallback.model_dump(mode="json")

    try:
        from app.prompt.screening import SCREENING_PROTOCOL_GENERATION_PROMPT
        from app.core.config import get_settings

        prompt = SCREENING_PROTOCOL_GENERATION_PROMPT.format(
            original_query=original_query,
            user_query=user_query,
            topic=topic,
            conversation_json=json.dumps(history, ensure_ascii=False),
            selected_scope_json=json.dumps(selected_scope or {}, ensure_ascii=False),
            semantic_frame_json=json.dumps(semantic_frame or {}, ensure_ascii=False),
            search_branches_json=json.dumps(search_branches or [], ensure_ascii=False),
        )
        response = llm.complete(
            prompt,
            response_format="json_object",
            temperature=0.0,
            timeout=get_settings().llm_control_plane_timeout,
            retry_empty=False,
            operation="screening_protocol_planning",
        )
        raw = _safe_parse_json(response if isinstance(response, str) else str(response))
        raw = _normalize_screening_protocol_payload(raw)
        protocol = ScreeningProtocol.model_validate({
            **raw,
            "generated_by": "llm",
        })
        protocol = _sanitize_screening_protocol(
            protocol,
            original_query=original_query,
            conversation_history=history,
            selected_scope=selected_scope or {},
            fallback=fallback,
        )
        logger.info(
            "Screening protocol generated: hard=%d soft=%d routes=%d",
            len(protocol.hard_include_criteria),
            len(protocol.soft_include_criteria),
            len(protocol.routes),
        )
        return protocol.model_dump(mode="json")
    except Exception as exc:
        logger.warning(
            "Screening protocol generation failed; using deterministic fallback: %s",
            exc,
        )
        return fallback.model_dump(mode="json")


def _normalize_screening_protocol_payload(raw: dict[str, Any]) -> dict[str, Any]:
    """把 LLM 偶发返回的结构化对象收敛成 ScreeningProtocol 可接受的标量列表。"""
    data = dict(raw or {})

    def _stringify_list(values: Any) -> list[str]:
        items: list[str] = []
        for value in values if isinstance(values, list) else []:
            if isinstance(value, str):
                text = value.strip()
            elif isinstance(value, dict):
                text = str(
                    value.get("term")
                    or value.get("label")
                    or value.get("name")
                    or value.get("criterion_id")
                    or value.get("id")
                    or ""
                ).strip()
            else:
                text = str(value or "").strip()
            if text:
                items.append(text)
        return items

    exclusions = data.get("hard_exclude_title_terms")
    if isinstance(exclusions, list):
        data["hard_exclude_title_terms"] = _stringify_list(exclusions)

    for key in ("hard_include_criteria", "soft_include_criteria"):
        criteria = data.get(key)
        if not isinstance(criteria, list):
            continue
        normalized: list[dict[str, Any]] = []
        for item in criteria:
            if not isinstance(item, dict):
                continue
            cleaned = dict(item)
            cleaned["terms"] = _stringify_list(cleaned.get("terms"))
            cleaned["terms_zh"] = _stringify_list(cleaned.get("terms_zh"))
            cleaned["terms_en"] = _stringify_list(cleaned.get("terms_en"))
            normalized.append(cleaned)
        data[key] = normalized

    routes = data.get("routes")
    if isinstance(routes, list):
        normalized_routes: list[dict[str, Any]] = []
        for item in routes:
            if not isinstance(item, dict):
                continue
            cleaned = dict(item)
            cleaned["terms"] = _stringify_list(cleaned.get("terms"))
            normalized_routes.append(cleaned)
        data["routes"] = normalized_routes

    return data


def _enhanced_fallback_screening_protocol(
    *,
    topic: str,
    search_branches: list[dict[str, Any]] | None = None,
    semantic_frame: dict[str, Any] | None = None,
):
    """基于检索分支和语义框架构造增强版保守协议。

    当 LLM 筛选协议不可用（如单轮请求无需控制面二次调用）时，
    从 search_branches 提取 soft_include_criteria，从 semantic_frame 提取 routes，
    使 rank_papers 能够执行基础的主题软打分和路线覆盖计算。
    """
    from app.schemas.screening_schema import ScreeningProtocol, ScreeningCriterion, ScreeningRoute

    soft_criteria: list[ScreeningCriterion] = []
    routes: list[ScreeningRoute] = []

    # 从 search_branches 的 required_concepts 提取 soft 条件
    seen_criteria_keys: set[tuple[str, ...]] = set()
    for idx, branch in enumerate(search_branches or []):
        if not isinstance(branch, dict):
            continue
        branch_type = str(branch.get("branch_type") or f"branch_{idx}")
        for c_idx, concept_group in enumerate(branch.get("required_concepts") or []):
            if not isinstance(concept_group, list):
                continue
            terms = [str(t).strip() for t in concept_group if str(t).strip()]
            if not terms:
                continue
            key = tuple(sorted(t.lower() for t in terms))
            if key in seen_criteria_keys:
                continue
            seen_criteria_keys.add(key)
            terms_zh = [t for t in terms if re.search(r"[\u4e00-\u9fff]", t)]
            terms_en = [t for t in terms if not re.search(r"[\u4e00-\u9fff]", t)]
            soft_criteria.append(ScreeningCriterion(
                criterion_id=f"{branch_type}_{c_idx}",
                label=f"分支概念: {', '.join(terms[:3])}",
                terms=terms,
                terms_zh=terms_zh,
                terms_en=terms_en,
                source="inferred",
                applies_to_each_paper=False,
                rationale=str(branch.get("rationale") or "由检索分支生成的软匹配条件"),
            ))

    # 从 semantic_frame 的 method_roles 构建 routes
    frame = semantic_frame or {}
    method_roles = frame.get("method_roles") if isinstance(frame, dict) else getattr(frame, "method_roles", None) or {}
    if method_roles and isinstance(method_roles, dict):
        weight_per_route = round(1.0 / len(method_roles), 3)
        for role_name, role_type in method_roles.items():
            surface = str(role_name).replace("_", " ").strip()
            routes.append(ScreeningRoute(
                route_id=str(role_name),
                label=surface,
                terms=[surface],
                weight=weight_per_route,
                rationale=f"语义框架研究角色: {role_type}",
            ))

    generated_by = "enhanced_fallback" if (soft_criteria or routes) else "minimal_fallback"
    return ScreeningProtocol(
        corpus_goal=f"形成与「{topic}」直接相关的多路线证据池",
        hard_include_criteria=[],
        soft_include_criteria=soft_criteria,
        hard_exclude_title_terms=[],
        routes=routes,
        generated_by=generated_by,
        notes=["增强兜底：从检索分支和语义框架自动构建软条件，无需 LLM"] if (soft_criteria or routes) else ["最小兜底：不预设任何硬过滤条件，语义判断交由 LLM screening"],
    )


def _fallback_screening_protocol(
    *,
    topic: str,
):
    """基于 topic 构造最小保守协议（向后兼容）。"""
    return _enhanced_fallback_screening_protocol(topic=topic)


def _sanitize_screening_protocol(
    protocol,
    *,
    original_query: str,
    conversation_history: list[dict[str, Any]],
    selected_scope: dict[str, Any],
    fallback,
):
    """限制 LLM 权限：推断项不得升级为硬条件，硬排除必须可追溯。"""
    from app.schemas.screening_schema import ScreeningProtocol

    hard = []
    soft = list(protocol.soft_include_criteria)
    for criterion in protocol.hard_include_criteria:
        criterion_text = " ".join([
            str(criterion.criterion_id or ""),
            str(criterion.label or ""),
            *[str(term) for term in criterion.terms or []],
        ])
        # 年份、篇数等结构化约束由槽位和论文元数据校验，不能要求题名或
        # 摘要包含“近三年/40篇”等字样。
        if re.search(
            r"年份|发表时间|时间范围|近\s*[三四五六七八九十\d]+\s*年|"
            r"publication\s*(?:year|date)|time\s*window|recent\s+\w*\s*years?|"
            r"引用篇数|文献数量|reference\s*count|citation\s*count|"
            r"\b(?:19|20)\d{2}\s*[-–—]\s*(?:19|20)\d{2}\b",
            criterion_text,
            re.I,
        ):
            soft.append(criterion.model_copy(update={"applies_to_each_paper": False}))
            continue
        if (
            criterion.source in {"user_explicit", "confirmed_scope"}
            and criterion.applies_to_each_paper
            and criterion.terms
        ):
            hard.append(criterion)
        else:
            soft.append(criterion.model_copy(update={"applies_to_each_paper": False}))
    if not hard:
        hard = list(fallback.hard_include_criteria)

    explicit_scope_exclusions = {
        str(term).strip().lower()
        for term in selected_scope.get("exclude_terms") or []
        if str(term).strip()
    }
    user_text = " ".join([
        original_query,
        *[
            item["content"] for item in conversation_history
            if item.get("role") == "user"
        ],
    ]).lower()
    has_exclusion_cue = bool(re.search(r"排除|不包括|不要|不考虑|不纳入|exclude|without", user_text))
    exclusions = [
        term for term in protocol.hard_exclude_title_terms
        if term.lower() in explicit_scope_exclusions
        or (has_exclusion_cue and term.lower() in user_text)
    ]
    if not exclusions:
        exclusions = list(fallback.hard_exclude_title_terms)

    routes = list(protocol.routes) or list(fallback.routes)
    total_weight = sum(route.weight for route in routes)
    if routes and total_weight <= 0:
        equal_weight = 1.0 / len(routes)
        routes = [route.model_copy(update={"weight": equal_weight}) for route in routes]
    elif routes and total_weight > 0:
        routes = [
            route.model_copy(update={"weight": route.weight / total_weight})
            for route in routes
        ]

    return ScreeningProtocol(
        corpus_goal=protocol.corpus_goal or fallback.corpus_goal,
        hard_include_criteria=hard,
        soft_include_criteria=soft,
        hard_exclude_title_terms=exclusions,
        routes=routes,
        generated_by="llm",
        notes=list(protocol.notes),
    )


def choose_workflow(intent: str) -> str:
    """根据意图选择工作流名称。"""
    return WORKFLOW_MAP.get(intent, "general_qa_workflow")


def generate_search_keywords(
    topic: str,
    llm=None,
    user_query: str | None = None,
) -> List[str]:
    """把中文主题扩展为中英文检索关键词。

    主路径使用 LLM 生成领域相关的中英文检索词，避免为每个主题手写词表。
    LLM 不可用或返回异常时，保留原主题作为兜底查询。
    """
    return generate_search_strategy(topic, llm, user_query)["keywords"]


def _fallback_bilingual_anchor_group(
    topic: str,
    keywords: Sequence[str],
) -> List[List[str]]:
    """锚点组全部被门禁丢弃后，用主题与关键词机械重组双语兜底组。

    仅从主题与 LLM 已产出的关键词中按语言拆分取词，不做本地翻译
    猜测。兜底组让中文候选池在 rank 阶段仍能命中中文锚点词，
    避免锚点全空时退化为裸主题串精确匹配清空中文分支。
    """
    zh_terms: list[str] = []
    en_terms: list[str] = []
    for value in [topic, *(keywords or [])]:
        text = str(value or "").strip()
        if not text:
            continue
        if re.search(r"[\u4e00-\u9fff]", text):
            if text not in zh_terms:
                zh_terms.append(text)
        elif re.search(r"[A-Za-z]", text):
            if text.lower() not in {t.lower() for t in en_terms}:
                en_terms.append(text)
    if not zh_terms or not en_terms:
        return []
    return [(zh_terms + en_terms)[:8]]


def generate_search_strategy(
    topic: str,
    llm=None,
    user_query: str | None = None,
) -> Dict[str, Any]:
    """生成检索关键词和主题锚点（仅用于词法打分，不参与硬过滤）。"""
    topic = (topic or "").strip()
    user_query = (user_query or topic).strip()
    keywords = [topic] if topic else []
    topic_anchors: list[list[str]] = []
    dropped_anchor_groups: list[dict[str, str]] = []
    planning_error: str | None = None
    fallback = _fallback_search_strategy(topic) if topic else {
        "keywords": [],
        "topic_anchors": [],
    }

    if llm and topic:
        try:
            strategy = llm_generate_search_strategy(
                topic=topic,
                user_query=user_query,
                llm=llm,
            )
            llm_keywords = strategy.get("keywords", [])
            # 兼容旧字段名 required_concepts 和新字段名 topic_anchors
            llm_concepts = strategy.get("topic_anchors") or strategy.get("required_concepts") or []
            dropped_anchor_groups = strategy.get("dropped_monolingual_groups") or []

            # 校验 LLM 实际返回了可用的英文检索词
            english_keywords = [
                kw for kw in llm_keywords
                if re.search(r"[A-Za-z]{4,}", kw)
            ]
            if not english_keywords and not llm_concepts:
                planning_error = (
                    "LLM 返回的关键词中没有可用的英文检索词，"
                    "且 topic_anchors 为空。请检查 LLM 响应或网络连接。"
                )
                logger.warning("LLM returned no usable English keywords: %s", llm_keywords)
            else:
                keywords.extend(llm_keywords)
                # list 与 dict 两种概念组形态都归一为 list[str]，
                # 不再静默丢弃 list 形态导致概念组从未进入打分链路。
                topic_anchors = _normalize_topic_anchor_groups(llm_concepts)
        except Exception as e:
            logger.warning("LLM keyword generation failed: %s", e)
            planning_error = str(e)

    if topic and (planning_error or not any(re.search(r"[A-Za-z]{4,}", kw) for kw in keywords)):
        keywords.extend(fallback["keywords"])
        if not topic_anchors:
            topic_anchors = fallback.get("topic_anchors") or []

    final_keywords = _deduplicate_keywords(_clean_keyword_pool(keywords, topic), limit=6)
    # 通过专用检索关键词生成 tool 补全中英文关键词；include_metadata=True
    # 保留 exact/broader/variant 类型并组装为检索批次——"完整表达先检、
    # 外扩后检"的批次顺序是工具 type 的通用语义，不依赖任何领域词表。
    keyword_batches: list[dict[str, Any]] = []
    if llm:
        try:
            from app.tools.generate_search_keywords import generate_search_keywords
            generated = generate_search_keywords(topic, llm, include_metadata=True)
            typed_items = [*(generated.get("zh") or []), *(generated.get("en") or [])]
            keyword_batches = _keyword_batches_from_items(typed_items)
            merged = final_keywords + [str(item["keyword"]) for item in typed_items]
            final_keywords = _deduplicate_keywords(merged, limit=12)
        except Exception as exc:
            logger.warning("Keyword generation expansion skipped: %s", exc)
    if planning_error and (
        any(re.search(r"[A-Za-z]{4,}", kw) for kw in final_keywords)
        or topic_anchors
    ):
        planning_error = None

    if not topic_anchors and not dropped_anchor_groups:
        fallback_group = _fallback_bilingual_anchor_group(topic, final_keywords)
        if fallback_group:
            topic_anchors = fallback_group
            logger.info(
                "Topic anchors rebuilt from topic/keywords after bilingual gate: %s",
                topic_anchors,
            )

    return {
        "keywords": final_keywords,
        "keyword_batches": keyword_batches,
        "topic_anchors": topic_anchors,
        "planning_error": planning_error,
        "dropped_monolingual_groups": dropped_anchor_groups,
    }


# 批次顺序 = 关键词生成工具 type 的通用语义：完整表达先检，外扩后检。
_KEYWORD_BATCH_ORDER = ("exact", "broader", "variant")


def _keyword_batches_from_items(
    items: Sequence[dict[str, Any]],
    existing: Sequence[dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    """把带 type 的关键词项组装为 exact→broader→variant 的检索批次。

    批次划分完全来自工具返回的 type 元数据；existing 提供时按类型合并
    去重（先到先得，保序），供 refine 轮在不丢既有批次的前提下注入新词。
    """
    merged: Dict[str, List[str]] = {}
    for source in (existing or []):
        batch_type = str(source.get("type") or "").strip().lower()
        if batch_type not in _KEYWORD_BATCH_ORDER:
            continue
        bucket = merged.setdefault(batch_type, [])
        for keyword in source.get("keywords") or []:
            value = str(keyword).strip()
            if value and value.lower() not in {v.lower() for v in bucket}:
                bucket.append(value)
    for item in items or []:
        keyword = str(item.get("keyword") or "").strip()
        batch_type = str(item.get("type") or "").strip().lower()
        if not keyword or batch_type not in _KEYWORD_BATCH_ORDER:
            continue
        bucket = merged.setdefault(batch_type, [])
        if keyword.lower() not in {v.lower() for v in bucket}:
            bucket.append(keyword)
    return [
        {"type": batch_type, "keywords": merged[batch_type]}
        for batch_type in _KEYWORD_BATCH_ORDER
        if merged.get(batch_type)
    ]


def refine_search_strategy(
    topic: str,
    user_query: str,
    current_keywords: list[str],
    feedback: dict[str, Any],
    llm,
    existing_batches: list[dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """根据上一轮检索反馈，让 LLM 重新生成下一轮检索策略。"""
    strategy = llm_refine_search_strategy(
        topic=topic,
        user_query=user_query,
        current_keywords=current_keywords,
        feedback=feedback,
        llm=llm,
    )
    keywords = _deduplicate_keywords(
        _clean_keyword_pool(
            [topic] + strategy.get("keywords", []) + current_keywords, topic
        ),
        limit=10,
    )
    # 通过专用检索关键词生成 tool 补全中英文关键词；与既有批次合并，
    # 保证 refine 轮新增的 exact 词能进入对应批次而不是散落词池。
    keyword_batches: list[dict[str, Any]] = list(existing_batches or [])
    if llm:
        try:
            from app.tools.generate_search_keywords import generate_search_keywords
            generated = generate_search_keywords(topic, llm, include_metadata=True)
            typed_items = [*(generated.get("zh") or []), *(generated.get("en") or [])]
            keyword_batches = _keyword_batches_from_items(typed_items, existing=keyword_batches)
            keywords = keywords + [str(item["keyword"]) for item in typed_items]
            keywords = _deduplicate_keywords(keywords, limit=12)
        except Exception as exc:
            logger.warning("Keyword generation expansion skipped: %s", exc)
    return {
        "keywords": keywords,
        "keyword_batches": keyword_batches,
        "topic_anchors": strategy.get("topic_anchors") or strategy.get("required_concepts") or [],
        "dropped_monolingual_groups": strategy.get("dropped_monolingual_groups") or [],
    }


def llm_generate_search_keywords(topic: str, user_query: str, llm) -> list[str]:
    """调用 LLM 为任意研究主题生成检索关键词。"""
    return llm_generate_search_strategy(topic, user_query, llm)["keywords"]


def llm_generate_search_strategy(topic: str, user_query: str, llm) -> Dict[str, Any]:
    """调用 LLM 生成检索词及必需概念组（含双语对齐门禁）。"""
    from app.prompt.search import SEARCH_KEYWORD_GENERATION_PROMPT

    prompt = SEARCH_KEYWORD_GENERATION_PROMPT.format(
        user_query=user_query,
        topic=topic,
    )
    return _request_llm_strategy_with_anchor_gate(
        prompt, llm, topic, operation="initial_search_planning",
    )


def llm_refine_search_strategy(
    topic: str,
    user_query: str,
    current_keywords: list[str],
    feedback: dict[str, Any],
    llm,
) -> Dict[str, Any]:
    """调用 LLM 根据检索反馈修正关键词（含双语对齐门禁）。"""
    from app.prompt.search import SEARCH_KEYWORD_REFINEMENT_PROMPT

    prompt = SEARCH_KEYWORD_REFINEMENT_PROMPT.format(
        user_query=user_query,
        topic=topic,
        keywords_json=json.dumps(current_keywords, ensure_ascii=False),
        feedback_json=json.dumps(feedback, ensure_ascii=False, default=str),
    )
    return _request_llm_strategy_with_anchor_gate(
        prompt, llm, topic, operation="refined_search_planning",
    )


def _fallback_search_strategy(topic: str) -> dict[str, Any]:
    """LLM 不可用时只保留原主题，不猜测翻译、同义词或检索后缀。"""
    from app.agent.research_semantic_parser import parse_research_semantics
    from app.agent.search_plan_builder import (
        build_semantic_search_branches,
        prioritized_branch_queries,
    )

    frame = parse_research_semantics(topic, topic, llm=None)
    branches = build_semantic_search_branches(frame)
    return {
        "keywords": prioritized_branch_queries(branches),
        "topic_anchors": [],
    }


def _topic_requires_cjk(topic: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", topic or ""))


def _parse_llm_strategy_payload(data: Dict[str, Any]) -> Tuple[List[str], List[List[str]]]:
    """从 LLM 响应解析检索关键词与概念组，兼容 dict/list 两种概念组形态。"""
    raw_keywords = data.get("keywords", [])
    keywords = (
        [str(k).strip() for k in raw_keywords if str(k).strip()]
        if isinstance(raw_keywords, list)
        else []
    )
    topic_anchors: list[list[str]] = []
    raw_concepts = data.get("topic_anchors") or data.get("required_concepts") or []
    if isinstance(raw_concepts, list):
        for item in raw_concepts:
            terms = item.get("terms", []) if isinstance(item, dict) else item
            if not isinstance(terms, list):
                continue
            normalized = _deduplicate_keywords(
                [str(term).strip() for term in terms if str(term).strip()],
                limit=8,
            )
            if normalized:
                topic_anchors.append(normalized)
    return keywords, topic_anchors[:4]


def _validate_bilingual_topic_anchors(
    topic_anchors: List[List[str]],
    topic: str,
) -> Tuple[List[List[str]], List[Dict[str, str]]]:
    """校验概念组与主题语言对齐，返回 ``(合规组, 被丢弃组诊断)``。

    主题含中文时每个概念组须含至少一个中文词，纯英文主题反之亦然。
    单语概念组会让 rank 阶段的任务偏移惩罚静默失效（惩罚只对主题或
    同义词中出现的维度生效），因此不合规的组被显式丢弃并记录诊断，
    不在本地猜测补译——跨语言术语对齐必须由 LLM 显式给出。
    """
    requires_cjk = _topic_requires_cjk(topic)
    kept: List[List[str]] = []
    dropped: List[Dict[str, str]] = []
    for group in topic_anchors or []:
        terms = [str(t).strip() for t in group or [] if str(t).strip()]
        if not terms:
            continue
        aligned = any(
            bool(re.search(r"[\u4e00-\u9fff]", t)) == requires_cjk
            for t in terms
        )
        if aligned:
            kept.append(terms)
        else:
            dropped.append({
                "group": " / ".join(terms[:4]),
                "reason": "missing_topic_language_term",
            })
    if dropped:
        logger.warning(
            "Dropped %d monolingual topic anchor group(s) lacking %s terms: %s",
            len(dropped), "中文" if requires_cjk else "英文",
            [item["group"] for item in dropped],
        )
    return kept, dropped


def _request_llm_strategy_with_anchor_gate(
    prompt: str,
    llm,
    topic: str,
    operation: str,
) -> Dict[str, Any]:
    """调用 LLM 生成检索策略，并对概念组执行双语对齐门禁。

    概念组缺少与主题同语言的术语时，先带着校验错误定向再生成一次；
    再生成结果整体替换首轮（含关键词）；仍不合规的组被显式丢弃并
    记录在 ``dropped_monolingual_groups``，不会静默流入打分链路。
    """
    from app.core.config import get_settings

    settings = get_settings()

    def _complete(text: str) -> Dict[str, Any]:
        response = llm.complete(
            text,
            temperature=0.1,
            timeout=settings.llm_control_plane_timeout,
            retry_empty=False,
            operation=operation,
        )
        return _safe_parse_json(response if isinstance(response, str) else str(response))

    data = _complete(prompt)
    keywords, anchors = _parse_llm_strategy_payload(data)
    kept, dropped = _validate_bilingual_topic_anchors(anchors, topic)

    if dropped:
        lang_label = "中文" if _topic_requires_cjk(topic) else "英文"
        dropped_summary = "\n".join(f"- {item['group']}" for item in dropped)
        corrective = (
            f"\n\n注意：上次输出的以下 topic_anchors 概念组缺少{lang_label}术语，"
            f"已被系统判无效：\n{dropped_summary}\n"
            f"请重新输出完整结果，并确保每个概念组同时包含中文术语和英文术语。"
        )
        try:
            retry_data = _complete(prompt + corrective)
            retry_keywords, retry_anchors = _parse_llm_strategy_payload(retry_data)
            retry_kept, retry_dropped = _validate_bilingual_topic_anchors(
                retry_anchors, topic,
            )
            if retry_kept and retry_keywords:
                return {
                    "keywords": retry_keywords,
                    "topic_anchors": retry_kept,
                    "dropped_monolingual_groups": [*dropped, *retry_dropped],
                }
        except Exception as exc:
            logger.warning("Bilingual anchor regeneration failed: %s", exc)

    return {
        "keywords": keywords,
        "topic_anchors": kept,
        "dropped_monolingual_groups": dropped,
    }


def _normalize_topic_anchor_groups(groups: Any) -> List[List[str]]:
    """把概念组（LLM 已归一的 list 形态或历史 dict 形态）统一为 ``list[str]``。

    旧实现只接受 ``{"terms": [...]}`` dict，会静默丢弃 list 形态的
    概念组，导致规划产出的概念组从未进入打分链路。
    """
    normalized: List[List[str]] = []
    if not isinstance(groups, list):
        return normalized
    for group in groups:
        if isinstance(group, dict):
            terms = group.get("terms") or []
        elif isinstance(group, list):
            terms = group
        else:
            continue
        cleaned = [str(t).strip() for t in terms if str(t).strip()]
        if cleaned:
            normalized.append(cleaned)
    return normalized


def _split_bilingual_terms(terms: list[str]) -> dict[str, list[str]]:
    """将混合术语列表拆分为 terms_zh / terms_en。

    通过 CJK 字符判断每个 term 的语言归属。跨语言术语必须由上游 LLM
    在筛选协议中显式给出，避免本地领域词表悄悄扩大或改写用户范围。
    """
    zh: list[str] = []
    en: list[str] = []
    for term in terms:
        term_str = str(term).strip()
        if not term_str:
            continue
        # 判断语言：含 CJK → 中文，否则 → 英文
        if re.search(r"[一-鿿]", term_str):
            if term_str not in zh:
                zh.append(term_str)
        else:
            if term_str not in en:
                en.append(term_str)
    return {"terms_zh": zh, "terms_en": en}


def _clean_keyword_pool(keywords: list[str], topic: str) -> list[str]:
    """词池合并前的确定性清洗：中英混杂词拆出中文段，拆不出则丢弃。

    策略 LLM 偶尔产出"少样本学习 few-shot learning human action"这类
    混杂词。若留到派发层才清洗，它们先占用批次名额，且清洗后的
    中文段与池内既有词的重复无法被去重发现。含主题锚点的词保持
    原样，由派发层清洗，避免锚点子串判定失配。
    """
    from app.core.source_capabilities import sanitize_search_keyword

    anchor = str(topic or "").strip()
    cleaned: list[str] = []
    for keyword in keywords:
        text = str(keyword or "").strip()
        if not text:
            continue
        if anchor and anchor in text:
            cleaned.append(text)
            continue
        value = sanitize_search_keyword(text)
        if value:
            cleaned.append(value)
    return cleaned


def _deduplicate_keywords(keywords: list[str], limit: int = 10) -> list[str]:
    """去除重复、空串并限制数量。"""
    seen = set()
    unique: list[str] = []
    for k in keywords:
        k = k.strip()
        key = k.lower()
        if k and key not in seen:
            seen.add(key)
            unique.append(k)
    return unique[:limit]


from app.core.json_utils import parse_json_object as _safe_parse_json  # noqa: E402
