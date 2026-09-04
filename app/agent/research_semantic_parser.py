"""LLM 驱动的研究请求语义解析。

本模块只负责结构校验、原文落地检查和通用关系派生，不维护任何学科、
方法或研究对象词表。模型不可用时返回保守的空语义框架，避免把某个历史
项目的领域假设注入新主题。
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

from app.agent.evidence_roles import is_scope_only_text, is_temporal_qualifier_text
from app.core.config import get_settings
from app.core.json_utils import parse_json_object
from app.core.logger import get_logger
from app.schemas.research_plan_schema import (
    EvidenceRequirement,
    MethodRole,
    ResearchMethod,
    ResearchMode,
    ResearchSemanticFrame,
    TerminalGoal,
)

logger = get_logger(__name__)


def parse_research_semantics(
    user_query: str,
    topic: str,
    deliverables: Iterable[str] | None = None,
    llm=None,
) -> ResearchSemanticFrame:
    """调用 LLM 解析语义；失败时不猜测领域、对象或方法。"""
    deliverables = list(deliverables or [])
    frame = _empty_frame(topic or user_query)
    if llm is not None:
        try:
            from app.agent.example_retriever import retrieve_semantic_examples
            from app.prompt.research_semantics import RESEARCH_SEMANTIC_PARSER_PROMPT

            examples = retrieve_semantic_examples(user_query, top_k=3)
            response = llm.complete(
                RESEARCH_SEMANTIC_PARSER_PROMPT.format(
                    user_query=user_query,
                    topic=topic,
                    deliverables_json=json.dumps(list(deliverables or []), ensure_ascii=False),
                    retrieved_examples_json=json.dumps(examples, ensure_ascii=False),
                ),
                response_format="text",
                temperature=0.0,
                timeout=get_settings().llm_control_plane_timeout,
                retry_empty=False,
                operation="research_semantic_parsing",
            )
            frame = _validate_llm_frame(
                parse_json_object(response),
                topic or user_query,
                deliverables=deliverables,
            )
            frame = frame.model_copy(update={
                "retrieved_case_ids": [
                    str(item.get("case_id")) for item in examples if item.get("case_id")
                ]
            })
        except Exception as exc:
            logger.warning("Research semantic parsing failed; using empty semantic frame: %s", exc)

    from app.agent.semantic_consistency import ground_semantic_frame, validate_semantic_relations

    frame = _clamp_ungrounded_minimums(
        _drop_orphan_requirements(ground_semantic_frame(frame, user_query)),
        user_query,
    )
    frame = derive_research_semantics(frame)
    return validate_semantic_relations(frame)


# 用户原文里数量与概念的相邻窗口：中文表述里"教学互动至少15篇"这类限定
# 紧跟概念，跨句的数字属于另一处约束。窗口取字符数而非分词结果，中英文一致；
# 中间出现分句标点即视为跨越约束边界，无论距离多近。
_QUANTITY_PROXIMITY_WINDOW = 12
_QUANTITY_IN_QUERY_RE = re.compile(r"\d+")
_CLAUSE_BOUNDARY_RE = re.compile(r"[，。；、,.;\n]")


def _clamp_ungrounded_minimums(
    frame: ResearchSemanticFrame,
    user_query: str,
) -> ResearchSemanticFrame:
    """``minimum_direct_sources`` > 1 必须能在用户原文里溯源到该概念旁的数量。

    这是一条硬门禁：每条要求都要求 N 篇直接证据，达不到就阻断正文生成。
    实测模型会把"引用论文不少于40篇"这一条**整篇**引用下限拆成逐要求配额
    （2026-08-29 会话：40/20/20/10/10/5/10，合计 115 篇），于是 60 篇证据池
    必然七项全缺，写作被整体阻断——而整篇引用下限本就由
    ``required_reference_count`` 单独校验，不该在这里重复且加倍。

    判据与学科无关：只有当用户原文里某个数字**紧跟在**该要求自身的别名/标签
    之后（"教学互动研究至少15篇"）时，才承认它是用户为该概念指定的配额；
    出现在概念之前的数字属于整篇引用约束（"引用不少于40篇…课堂行为分析"），
    不得被逐要求配额重复消费。模型报得比原文更高时按原文取小。

    无法溯源时回落为 1，并在 validation_issues 留痕。
    """
    query = str(user_query or "")
    if not frame.evidence_requirements:
        return frame
    quantities = [
        (int(match.group()), match.start(), match.end())
        for match in _QUANTITY_IN_QUERY_RE.finditer(query)
    ]
    adjusted: list[EvidenceRequirement] = []
    issues: list[str] = []
    for requirement in frame.evidence_requirements:
        declared = int(requirement.minimum_direct_sources or 1)
        if declared <= 1:
            adjusted.append(requirement)
            continue
        grounded = min(declared, _grounded_quantity(requirement, query, quantities))
        if grounded == declared:
            adjusted.append(requirement)
            continue
        adjusted.append(requirement.model_copy(update={
            "minimum_direct_sources": grounded,
        }))
        issues.append(
            "clamped_ungrounded_minimum_sources:"
            f"{requirement.requirement_id}:{declared}->{grounded}"
        )
    if not issues:
        return frame
    return frame.model_copy(update={
        "evidence_requirements": adjusted,
        "validation_issues": list(dict.fromkeys([
            *(frame.validation_issues or []), *issues,
        ])),
    })


def _grounded_quantity(
    requirement: EvidenceRequirement,
    query: str,
    quantities: list[tuple[int, int, int]],
) -> int:
    """用户原文里紧跟该要求概念之后的数量；没有则返回 1。"""
    if not quantities:
        return 1
    probes = [
        text for text in (
            requirement.label,
            *(requirement.aliases or []),
            *(requirement.context_aliases or []),
        )
        if str(text or "").strip()
    ]
    best = 1
    for probe in probes:
        start = query.find(str(probe))
        while start >= 0:
            end = start + len(str(probe))
            for value, q_start, _q_end in quantities:
                if not 0 <= q_start - end <= _QUANTITY_PROXIMITY_WINDOW:
                    continue
                # 概念与数量之间出现分句标点说明二者属于不同约束子句
                # （"…课堂行为分析论文，并生成…，引用论文不少于40篇"）。
                if _CLAUSE_BOUNDARY_RE.search(query[end:q_start]):
                    continue
                best = max(best, value)
            start = query.find(str(probe), start + 1)
    return best


def _drop_orphan_requirements(frame: ResearchSemanticFrame) -> ResearchSemanticFrame:
    """证据要求必须溯源到落定后的实体。

    实体在 grounding 阶段被移除（如无原文依据的方法）后，其 requirement
    会成为孤儿：别名匹配不到任何论文，形成假门禁。孤儿要求随源实体一并
    移除，validation_issues 记录轨迹。
    """
    surviving_ids = {
        str(item.id)
        for items in (
            frame.research_objects, frame.methods,
            frame.research_actions, frame.analysis_targets,
        )
        for item in items
    }
    kept: list = []
    dropped: list = []
    for requirement in frame.evidence_requirements:
        if any(str(source_id) in surviving_ids for source_id in requirement.source_ids):
            kept.append(requirement)
        else:
            dropped.append(requirement)
    if not dropped:
        return frame
    return frame.model_copy(update={
        "evidence_requirements": kept,
        "validation_issues": list(dict.fromkeys([
            *(frame.validation_issues or []),
            *(f"dropped_orphan_requirement:{item.requirement_id}" for item in dropped),
        ])),
    })


def derive_research_semantics(frame: ResearchSemanticFrame) -> ResearchSemanticFrame:
    """仅依据结构化关系派生角色和模式，不读取领域关键词。"""
    has_domain = bool(frame.application_domains)
    technical = [method for method in frame.methods if method.category == "technical"]
    goal_type = str(frame.terminal_goal.type or "unspecified")
    # 是否存在下游分析必须由结构关系表达，不能根据任意 goal_type 字符串
    # 猜测。领域内的识别、监测或分类本身仍属于“技术应用”；只有模型明确
    # 给出分析目标，或把技术方法标为中间步骤时，才进入辅助领域分析模式。
    downstream = bool(frame.analysis_targets) or any(
        method.role == MethodRole.INTERMEDIATE_STEP for method in technical
    )
    algorithmic = goal_type in {"method_analysis", "method_comparison", "model_evaluation"}

    methods: list[ResearchMethod] = []
    for method in frame.methods:
        role = method.role
        if role == MethodRole.NOT_SPECIFIED:
            if method.category == "analytical":
                role = MethodRole.PRIMARY_ANALYSIS_METHOD
            elif method.category == "technical" and downstream:
                role = MethodRole.INTERMEDIATE_STEP
            elif method.category == "technical" and has_domain and not algorithmic:
                role = MethodRole.IMPLEMENTATION_METHOD
            elif method.category == "technical":
                role = MethodRole.PRIMARY_RESEARCH_TARGET
        methods.append(method.model_copy(update={"role": role}))

    if technical and downstream and has_domain:
        mode = ResearchMode.TECHNOLOGY_ASSISTED_DOMAIN_ANALYSIS
    elif technical and has_domain:
        mode = ResearchMode.TECHNOLOGY_APPLIED_TO_DOMAIN
    elif technical:
        mode = ResearchMode.TECHNOLOGY_ORIENTED
    elif has_domain:
        mode = ResearchMode.DOMAIN_ORIENTED
    else:
        mode = ResearchMode.AMBIGUOUS

    chain = list(dict.fromkeys(str(item) for item in frame.task_chain if str(item).strip()))
    if not chain:
        # 这是结构化字段的机械编译，不根据词义臆测阶段或顺序。
        chain = list(dict.fromkeys([
            *[item.id for item in frame.research_actions if item.explicit],
            *[item.id for item in methods if item.explicit],
            *[item.id for item in frame.analysis_targets if item.explicit],
        ]))

    requirements = frame.evidence_requirements or _generic_evidence_requirements(
        frame.model_copy(update={"methods": methods})
    )
    return frame.model_copy(update={
        "methods": methods,
        "research_mode": mode,
        "task_chain": chain,
        "required_focuses": _filter_quantity_focuses(
            frame.required_focuses,
            _domain_entity_aliases(
                frame.research_objects, frame.methods, frame.analysis_targets,
            ),
        ),
        "evidence_requirements": requirements,
    })


def _empty_frame(topic: str) -> ResearchSemanticFrame:
    return ResearchSemanticFrame(
        canonical_topic=str(topic or "").strip(),
        confidence={"overall": 0.0, "source": 0.0},
        validation_warnings=["semantic_parser_unavailable"],
    )


def _validate_llm_frame(
    raw: dict[str, Any],
    topic: str,
    deliverables: Iterable[str] | None = None,
) -> ResearchSemanticFrame:
    if not raw:
        raise ValueError("语义解析模型未返回 JSON 对象")
    data = dict(raw)
    data.pop("research_mode", None)
    data["canonical_topic"] = _remove_deliverable_terms(
        str(data.get("canonical_topic") or topic), deliverables
    ) or _remove_deliverable_terms(str(topic or ""), deliverables)
    for field in ("application_domains", "research_objects", "research_actions", "analysis_targets"):
        # 纯时间限定词（“近五年文献”）不是研究对象：留下它会让覆盖检查
        # 去论文正文里词面匹配时间词，必然误报直接证据缺失。
        data[field] = _filter_temporal_items(_normalize_items(data.get(field)))
    data["methods"] = _normalize_methods(data.get("methods"))
    for field in ("application_domains", "research_objects", "research_actions", "analysis_targets", "methods"):
        data[field] = _drop_deliverable_items(data[field])
    for field in ("assumptions", "scope_ambiguities", "secondary_goals", "task_chain", "required_focuses"):
        data[field] = _normalize_str_list(data.get(field))
    data["task_chain"] = [
        item for item in data["task_chain"] if not _is_deliverable_text(item)
    ]
    data["required_focuses"] = [
        item for item in data["required_focuses"] if not _is_deliverable_text(item)
    ]
    data["required_focuses"] = _filter_quantity_focuses(
        data["required_focuses"],
        _domain_entity_aliases(
            data.get("research_objects"),
            data.get("methods"),
            data.get("analysis_targets"),
        ),
    )
    data["confidence"] = _normalize_confidence(data.get("confidence"))
    data["evidence_requirements"] = _normalize_evidence_requirements(
        data.get("evidence_requirements"), data
    )
    frame = ResearchSemanticFrame.model_validate(data)
    confidence = dict(frame.confidence)
    confidence["source"] = 1.0
    return frame.model_copy(update={"confidence": confidence})


def _normalize_items(value: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for item in value if isinstance(value, list) else []:
        if isinstance(item, str):
            item = {"id": _snake(item), "label": item, "surface_text": item}
        if not isinstance(item, dict) or not item.get("id"):
            continue
        explicit = bool(item.get("explicit", False))
        items.append({
            "id": _snake(str(item["id"])),
            "label": item.get("label"),
            "surface_text": item.get("surface_text"),
            "explicit": explicit,
            "inferred": bool(item.get("inferred", not explicit)),
            "source": str(item.get("source") or ("user_explicit" if explicit else "llm_inference")),
            "inference_basis": item.get("inference_basis"),
            "confidence": _confidence(item.get("confidence")),
        })
    return items


def _normalize_methods(value: Any) -> list[dict[str, Any]]:
    roles = {role.value for role in MethodRole}
    result = _normalize_items(value)
    source_items = value if isinstance(value, list) else []
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(result):
        source = source_items[index] if index < len(source_items) and isinstance(source_items[index], dict) else {}
        role = str(source.get("role") or MethodRole.NOT_SPECIFIED.value)
        normalized.append({
            **item,
            "category": str(source.get("category") or "technical"),
            "role": role if role in roles else MethodRole.NOT_SPECIFIED.value,
        })
    return normalized


def _normalize_evidence_requirements(value: Any, frame_data: dict[str, Any]) -> list[dict[str, Any]]:
    valid_ids = {
        str(item.get("id"))
        for field in ("methods", "research_actions", "analysis_targets", "research_objects")
        for item in frame_data.get(field, [])
        if isinstance(item, dict) and item.get("id")
    }
    # 研究动作（调研、生成研究背景、生成研究现状…）描述的是本次任务要做
    # 什么，不是论文里必须写明的内容：没有任何论文会声明自己“生成研究
    # 背景”，把这类要求送进直接证据覆盖检查会让命中数恒为 0 而必然误拦
    # （2026-08-22 会话：少样本动作识别文献调研证据 / 研究背景生成证据 /
    # 研究现状生成证据）。按来源实体类型结构化排除，不靠标签词面枚举。
    action_ids = {
        str(item.get("id"))
        for item in frame_data.get("research_actions") or []
        if isinstance(item, dict) and item.get("id")
    }
    domain_aliases = _domain_entity_aliases(
        frame_data.get("research_objects"),
        frame_data.get("methods"),
        frame_data.get("analysis_targets"),
    )
    result: list[dict[str, Any]] = []
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, dict) or not item.get("requirement_id") or not item.get("label"):
            continue
        # 时间窗 + 动作壳的要求（“近五年文献调研证据”）按实体映射判为检索
        # 范围，由年份过滤和数量校验保证，不进入覆盖检查；旧会话兼容由
        # evidence_roles 的年份窗判定兜底。
        if is_scope_only_text(str(item.get("label")), domain_aliases):
            continue
        source_ids = [str(x) for x in item.get("source_ids") or [] if str(x) in valid_ids]
        if not source_ids:
            continue
        if action_ids and set(source_ids) <= action_ids:
            continue
        candidate = dict(item)
        candidate["source_ids"] = source_ids
        try:
            result.append(EvidenceRequirement.model_validate(candidate).model_dump(mode="json"))
        except Exception:
            continue
    return result


def _generic_evidence_requirements(frame: ResearchSemanticFrame) -> list[EvidenceRequirement]:
    requirements: list[EvidenceRequirement] = []
    objects = _semantic_aliases(frame.research_objects)
    for method in (item for item in frame.methods if item.explicit):
        role = "analytical_method" if method.category == "analytical" else f"{method.category}_method"
        requirements.append(EvidenceRequirement(
            requirement_id=f"{role}:{method.id}",
            label=str(method.surface_text or method.label or method.id),
            evidence_role=role,
            aliases=_semantic_aliases([method]),
            context_aliases=objects,
            source_ids=[method.id],
            exact_method_required=True,
            route_group=role,
        ))
    # 研究动作不派生证据要求：动作（调研、生成研究背景…）描述本次任务的
    # 产出流程，论文正文无法为其提供直接证据，作为覆盖判据只会恒不达标。
    # 动作仍保留在 task_chain / research_actions 中供规划与检索使用。
    targets = [item for item in frame.analysis_targets if item.explicit]
    if targets:
        requirements.append(EvidenceRequirement(
            requirement_id="analysis_target:" + "+".join(item.id for item in targets),
            label="、".join(str(item.surface_text or item.label or item.id) for item in targets),
            evidence_role="interpretation",
            aliases=_semantic_aliases(targets),
            context_aliases=objects,
            source_ids=[item.id for item in targets],
            route_group="analysis_targets",
        ))
    return requirements


def _semantic_aliases(items: Iterable[Any]) -> list[str]:
    return list(dict.fromkeys(
        value.strip()
        for item in items
        for value in (str(item.id), str(item.label or ""), str(item.surface_text or ""))
        if value.strip()
    ))


def _normalize_str_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        if isinstance(item, dict):
            item = item.get("description") or item.get("label") or item.get("id") or ""
        text = str(item).strip()
        if text:
            result.append(text)
    return result


def _normalize_confidence(value: Any) -> dict[str, float]:
    if isinstance(value, dict):
        return {str(key): _confidence(item) for key, item in value.items()}
    return {"overall": _confidence(value)}


_DELIVERABLE_TEXT_RE = re.compile(
    r"研究背景|背景与意义|研究意义|研究现状|国内外现状|相关工作|"
    r"文献综述|叙述性综述|综述|参考文献(?:列表)?|论文(?:清单|列表|表格)|"
    r"\b(?:research\s+background|research\s+status|state\s+of\s+the\s+art|"
    r"related\s+work|literature\s+review|narrative\s+review|reference\s+list|"
    r"paper\s+(?:list|table))\b|(?:research_background|research_status|related_work|"
    r"literature_review|narrative_review|reference_list|paper_(?:list|table)|"
    r"background_generation|status_generation|deliverable_generation)",
    re.I,
)


def _is_deliverable_text(value: Any) -> bool:
    return bool(_DELIVERABLE_TEXT_RE.search(str(value or "")))


def _drop_deliverable_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        item for item in items
        if not _is_deliverable_text(
            " ".join(str(item.get(key) or "") for key in ("id", "label", "surface_text"))
        )
    ]


def _remove_deliverable_terms(
    value: str,
    deliverables: Iterable[str] | None = None,
) -> str:
    text = _DELIVERABLE_TEXT_RE.sub(" ", str(value or ""))
    for deliverable in deliverables or []:
        text = re.sub(re.escape(str(deliverable).replace("_", " ")), " ", text, flags=re.I)
    return re.sub(r"[，,；;：:\-—]?\s+", " ", text).strip(" ，,；;：:-—")


_QUANTITY_FOCUS_RE = re.compile(r"(?:\d+\s*篇|\b\d+\s*(?:papers?|references?)\b)", re.I)


def _item_surfaces(item: Any) -> list[str]:
    keys = ("surface_text", "label")
    if isinstance(item, dict):
        return [str(item.get(key) or "") for key in keys]
    return [str(getattr(item, key, "") or "") for key in keys]


def _domain_entity_aliases(
    objects: Any,
    methods: Any,
    targets: Any,
) -> list[str]:
    """研究对象/方法/分析目标的表面词集合。

    research_actions 不算领域实体：调研、生成背景等动作描述的是综述任务
    本身，正是要被实体映射判掉的“动作壳”。
    """
    aliases = [
        surface
        for item in ([*objects, *methods, *targets] or [])
        for surface in _item_surfaces(item)
        if str(surface).strip()
    ]
    return list(dict.fromkeys(str(alias).strip() for alias in aliases))


def _filter_quantity_focuses(
    items: Iterable[str],
    domain_aliases: list[str] | None = None,
) -> list[str]:
    return list(dict.fromkeys(
        str(item).strip() for item in items
        if str(item).strip()
        and not _QUANTITY_FOCUS_RE.search(str(item))
        and not is_scope_only_text(str(item), domain_aliases or [])
    ))


def _filter_temporal_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """丢弃纯时间限定词派生的实体（研究对象/方法/动作/分析目标）。

    实体表条目若按自身表面词映射会恒命中（残差总包含自身），跨实体映射
    无法区分“近五年少样本动作识别”与“近五年文献调研”，此处保留词面
    剥除兜底；requirement/焦点层的实体映射才是主防线。
    """
    return [
        item for item in items
        if not is_temporal_qualifier_text(
            str(item.get("surface_text") or item.get("label") or "")
        )
    ]


def _confidence(value: Any) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _snake(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "concept"
