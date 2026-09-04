"""研究语义帧的一致性校验与保守修复。"""

from __future__ import annotations

import re
from typing import Iterable, TypeVar

from app.schemas.research_plan_schema import (
    ResearchMethod,
    ResearchMode,
    ResearchSemanticFrame,
    SemanticItem,
)

ItemT = TypeVar("ItemT", SemanticItem, ResearchMethod)


def ground_semantic_frame(
    frame: ResearchSemanticFrame,
    user_query: str,
) -> ResearchSemanticFrame:
    """移除没有用户文本或推断依据支持的实体，并规范 explicit/inferred。"""
    issues: list[str] = []
    warnings: list[str] = []
    evidence_aliases = _evidence_aliases_by_source(frame)

    domains = _ground_items(frame.application_domains, user_query, "application_domain", issues, warnings, evidence_aliases)
    objects = _ground_items(frame.research_objects, user_query, "research_object", issues, warnings, evidence_aliases)
    methods = _ground_items(frame.methods, user_query, "method", issues, warnings, evidence_aliases)
    actions = _ground_items(frame.research_actions, user_query, "research_action", issues, warnings, evidence_aliases)
    targets = _ground_items(frame.analysis_targets, user_query, "analysis_target", issues, warnings, evidence_aliases)

    return frame.model_copy(update={
        "application_domains": domains,
        "research_objects": objects,
        "methods": methods,
        "research_actions": actions,
        "analysis_targets": targets,
        "validation_issues": list(dict.fromkeys([*frame.validation_issues, *issues])),
        "validation_warnings": list(dict.fromkeys([*frame.validation_warnings, *warnings])),
    })


def validate_semantic_relations(frame: ResearchSemanticFrame) -> ResearchSemanticFrame:
    """校验派生模式、方法角色和终点目标之间的关系。"""
    issues = list(frame.validation_issues)
    warnings = list(frame.validation_warnings)
    technical = [method for method in frame.methods if method.category == "technical"]

    if frame.research_mode == ResearchMode.TECHNOLOGY_ORIENTED and not technical:
        issues.append("technology_oriented_without_technical_method")
    if frame.research_mode == ResearchMode.TECHNOLOGY_APPLIED_TO_DOMAIN:
        if not technical:
            issues.append("technology_application_without_technical_method")
        if not frame.application_domains:
            issues.append("technology_application_without_domain")
    if frame.research_mode == ResearchMode.TECHNOLOGY_ASSISTED_DOMAIN_ANALYSIS:
        if not technical:
            issues.append("assisted_analysis_without_upstream_technology")
        if not frame.application_domains:
            issues.append("assisted_analysis_without_domain")
        if frame.terminal_goal.type != "domain_analysis" and not frame.analysis_targets:
            issues.append("assisted_analysis_without_downstream_goal")

    for method in frame.methods:
        if method.inferred and method.confidence < 0.7:
            warnings.append(f"low_confidence_inferred_method:{method.id}")
        if method.inferred and not method.inference_basis:
            warnings.append(f"inferred_method_missing_basis:{method.id}")

    clarification_needed = frame.clarification_needed or bool(issues)
    question = frame.clarification_question
    if issues and not question:
        question = "我还不能可靠确定研究对象、方法与最终目标之间的关系，你希望最终解决什么问题？"
    return frame.model_copy(update={
        "validation_issues": list(dict.fromkeys(issues)),
        "validation_warnings": list(dict.fromkeys(warnings)),
        "clarification_needed": clarification_needed,
        "clarification_question": question,
    })


def _ground_items(
    items: Iterable[ItemT],
    user_query: str,
    field: str,
    issues: list[str],
    warnings: list[str],
    evidence_aliases: dict[str, list[str]],
) -> list[ItemT]:
    grounded: list[ItemT] = []
    for item in items:
        update = {}
        explicit = bool(item.explicit)
        inferred = bool(item.inferred)
        surface_grounded, matched_text = _item_grounding_match(
            item,
            user_query,
            evidence_aliases.get(item.id, []),
        )

        if explicit and not surface_grounded:
            if item.inference_basis and item.confidence >= 0.7:
                update.update({
                    "explicit": False,
                    "inferred": True,
                    "source": "llm_inference",
                })
                warnings.append(f"downgraded_ungrounded_explicit_{field}:{item.id}")
            else:
                issues.append(f"removed_ungrounded_{field}:{item.id}")
                continue
        elif explicit:
            if field == "method" and not _has_explicit_method_context(
                user_query,
                matched_text or item.surface_text or item.label,
            ):
                # 研究主题中的“行为分析/课堂行为分析”等短语经常被
                # LLM 误报为方法。仅凭主题词出现不能建立方法约束；
                # 方法必须出现在“采用/基于/使用/方法”等方法语境中，
                # 或带有明确的分析法/算法/模型表述。
                issues.append(f"removed_ungrounded_method:{item.id}")
                continue
            update.update({
                "inferred": False,
                "source": (
                    item.source if item.source not in {"", "unknown"} else "user_explicit"
                ),
            })
        elif inferred:
            if item.confidence < 0.5 and not item.inference_basis:
                warnings.append(f"removed_low_confidence_inferred_{field}:{item.id}")
                continue
            update["source"] = (
                item.source if item.source not in {"", "unknown"} else "llm_inference"
            )
        else:
            # 非明确、非推断的实体不能成为检索约束。
            update.update({"inferred": True, "source": "llm_inference"})
            warnings.append(f"implicit_item_marked_inferred_{field}:{item.id}")

        grounded.append(item.model_copy(update=update))
    return _dedupe(grounded)


def _has_explicit_method_context(user_query: str, surface_text: str) -> bool:
    """判断方法短语是否在用户原文中以方法语义出现。

    这是通用的语境校验，不维护主题或方法名称黑名单。它可以保留
    “基于YOLO”“采用S-T分析法”“使用行为识别”等显式方法，同时
    拒绝把“课堂行为分析”这类研究主题本身当作方法。
    """
    query = str(user_query or "")
    surface = str(surface_text or "").strip()
    if not query or not surface:
        return False
    normalized_query = re.sub(r"\s+", " ", query.lower())
    normalized_surface = re.sub(r"\s+", " ", surface.lower())
    # 方法短语自身带有通用方法/工具标记时，不要求标记再次出现在
    # 短语外部。这里只识别跨领域通用词，不维护具体方法名称表。
    if re.search(
        r"方法|技术|工具|仪器|分析法|算法|模型|网络|框架|系统|"
        r"(?:识别|检测|编码|估计|测量|测定|监测|试验|实验)|"
        r"\b(?:method|methodology|technique|instrument|analysis|recognition|"
        r"detection|coding|estimation|measurement|monitoring|algorithm|model|"
        r"network|framework|system)\b",
        normalized_surface,
        re.I,
    ):
        return True
    position = normalized_query.find(normalized_surface)
    if position < 0:
        return False
    left = normalized_query[max(0, position - 36):position]
    right = normalized_query[
        position + len(normalized_surface):
        position + len(normalized_surface) + 36
    ]
    context = left + " " + right
    return bool(re.search(
        r"采用|使用|运用|利用|基于|通过|引入|提出|结合|借助|依托|"
        r"算法|模型|网络|分析法|方法|识别|检测|编码|估计|"
        r"\b(?:use|using|based\s+on|adopt|adopting|method|algorithm|"
        r"model|network|analysis|recognition|detection|coding|estimation)\b",
        context,
        re.I,
    ))


def _evidence_aliases_by_source(frame: ResearchSemanticFrame) -> dict[str, list[str]]:
    """汇总本轮 LLM 证据要求中的动态表示，不引入本地领域词典。"""
    aliases: dict[str, list[str]] = {}
    for requirement in frame.evidence_requirements:
        values = [
            requirement.label,
            *requirement.aliases,
        ]
        for source_id in requirement.source_ids:
            aliases.setdefault(source_id, []).extend(
                str(value).strip() for value in values if str(value).strip()
            )
    return {
        source_id: list(dict.fromkeys(values))
        for source_id, values in aliases.items()
    }


def _item_grounding_match(
    item: SemanticItem,
    user_query: str,
    dynamic_aliases: Iterable[str],
) -> tuple[bool, str]:
    """用原词、标准名、ID、动态别名及其缩写寻找保守文本锚点。"""
    query = _normalize(user_query)
    compact_query = _compact(user_query)
    candidates = list(dict.fromkeys(
        str(value).strip()
        for value in (
            item.surface_text,
            item.label,
            item.id.replace("_", " "),
            *dynamic_aliases,
        )
        if str(value or "").strip()
    ))
    for candidate in candidates:
        normalized = _normalize(candidate)
        compact = _compact(candidate)
        if normalized and normalized in query:
            return True, candidate
        if len(compact) >= 4 and compact in compact_query:
            return True, candidate
        acronym = _acronym(candidate)
        if len(acronym) >= 2 and re.search(
            rf"(?<![a-z0-9]){re.escape(acronym)}(?![a-z0-9])",
            query,
            re.I,
        ):
            return True, acronym
    return False, ""


def _compact(value: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(value or "").lower())


def _acronym(value: str) -> str:
    words = re.findall(r"[a-z][a-z0-9]*", str(value or "").lower())
    return "".join(word[0] for word in words) if len(words) >= 2 else ""


def _dedupe(items: list[ItemT]) -> list[ItemT]:
    best: dict[str, ItemT] = {}
    for item in items:
        previous = best.get(item.id)
        if previous is None or (item.explicit, item.confidence) > (previous.explicit, previous.confidence):
            best[item.id] = item
    return list(best.values())


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())
