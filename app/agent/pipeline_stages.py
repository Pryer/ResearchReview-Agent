"""研究链阶段的领域中立表示与兼容判定。

综述里的一条研究链通常是「上游产物 → 下游解释」：先有感知/识别结果，
再有结构化编码产物，然后才有分析方法与下游解释。不同学科的词汇完全不同，
但这个偏序关系是通用的：**上游产物不能替代下游证据**。

本模块只做三件事：

1. 把系统里两套已有的角色词汇（路线侧 ``route_role``、证据侧
   ``evidence_role``）归一到同一条有序阶段轴；
2. 判定一篇论文实际产出了哪些阶段的证据；
3. 判定某篇论文能否支撑某条路线（阶段兼容性）。

这里不出现任何具体学科、对象、方法或应用场景名称。阶段判据本身由
``evidence_roles.direct_evidence_match`` 按语义帧动态给出，本模块只负责
排序与兼容规则。
"""

from __future__ import annotations

from typing import Any, Iterable

# 规范阶段轴：索引即上下游次序，越小越上游。
# 命名取自「产物类型」而非学科：感知产出、结构化产出、分析产出、解释产出。
STAGE_PERCEPTION = "perception"
STAGE_STRUCTURED = "structured_coding"
STAGE_ANALYSIS = "analytical_method"
STAGE_INTERPRETATION = "interpretation"

STAGE_ORDER: tuple[str, ...] = (
    STAGE_PERCEPTION,
    STAGE_STRUCTURED,
    STAGE_ANALYSIS,
    STAGE_INTERPRETATION,
)

# 与阶段轴无关的角色：综述类证据可以跨阶段概括，基准类证据只服务评测。
# 它们不参与偏序约束，避免把「综述」误判成某一阶段的上游或下游。
STAGE_AGNOSTIC_ROLES = frozenset({"survey", "benchmark", "synthesis"})

# 路线侧词汇（provisional_routes 的 route_role）→ 规范阶段。
# 这是纯词汇映射，不含领域内容；未收录的取值按「无阶段约束」处理。
_ROUTE_ROLE_TO_STAGE: dict[str, str] = {
    "sensing": STAGE_PERCEPTION,
    "perception": STAGE_PERCEPTION,
    "formalization": STAGE_STRUCTURED,
    "structured_coding": STAGE_STRUCTURED,
    "coding": STAGE_STRUCTURED,
    "analysis": STAGE_ANALYSIS,
    "analytical_method": STAGE_ANALYSIS,
    "interpretation": STAGE_INTERPRETATION,
    "application": STAGE_INTERPRETATION,
}


def normalize_stage(value: Any) -> str:
    """把任意角色词汇归一为规范阶段；无法归一时返回空串。"""
    token = str(value or "").strip().lower()
    if not token:
        return ""
    if token in STAGE_ORDER:
        return token
    return _ROUTE_ROLE_TO_STAGE.get(token, "")


def stage_rank(stage: str) -> int:
    """阶段在上下游轴上的位置；未知阶段返回 -1（不参与偏序约束）。"""
    normalized = normalize_stage(stage)
    return STAGE_ORDER.index(normalized) if normalized in STAGE_ORDER else -1


def route_stage(route: dict[str, Any]) -> str:
    """路线所处阶段；未声明也无法推导时返回空串。

    优先读 ``pipeline_stage``（由 :func:`annotate_route_stages` 推导写入），
    其次读 ``route_role``（候选路线生成时由 LLM 声明，实测经常缺失）。
    """
    return normalize_stage(
        route.get("pipeline_stage") or route.get("route_role")
    )


def _route_text(route: dict[str, Any]) -> str:
    """路线的可匹配文本：名称、研究问题与各类概念锚点。"""
    parts: list[str] = [
        str(route.get("name") or ""),
        str(route.get("research_question") or ""),
    ]
    for key in (
        "core_concepts", "semantic_anchors", "method_concepts",
        "task_anchors", "inclusion_criteria",
    ):
        parts.extend(str(item) for item in route.get(key) or [])
    return " ".join(parts).lower()


def infer_route_stage(
    route: dict[str, Any],
    semantic_frame: dict[str, Any] | None,
) -> str:
    """按语义帧的证据要求推导路线阶段。

    候选路线生成 prompt 虽然要求 LLM 输出 ``route_role``，但实测它经常缺失；
    缺失时阶段约束会整体休眠（``stage_compatible`` 无条件放行），下游路线因此
    仍会被上游论文填满。这里改为从路线自身文本推导：路线文本命中某条证据
    要求的别名，即继承该要求所处阶段。

    别名由语义解析按用户请求动态生成，不是领域词表。命中多个阶段时取最下游
    的那个——路线名通常同时含上游手段与下游目标（“借助X实现Y”），而路线要
    交付的是最下游的产物。
    """
    declared = normalize_stage(route.get("route_role"))
    if declared:
        return declared
    requirements = list((semantic_frame or {}).get("evidence_requirements") or [])
    if not requirements:
        return ""

    from app.agent.evidence_roles import _contains_concept  # noqa: PLC0415

    text = _route_text(route)
    if not text.strip():
        return ""
    matched: list[str] = []
    for requirement in requirements:
        stage = normalize_stage(requirement.get("evidence_role"))
        if not stage:
            continue
        aliases = [
            str(item) for item in requirement.get("aliases") or []
            if str(item).strip()
        ]
        label = str(requirement.get("label") or "").strip()
        probes = [*aliases, label] if label else aliases
        if any(_contains_concept(text, probe) for probe in probes if probe):
            matched.append(stage)
    if not matched:
        return ""
    return max(matched, key=stage_rank)


def annotate_route_stages(
    routes: Iterable[dict[str, Any]],
    semantic_frame: dict[str, Any] | None = None,
) -> None:
    """就地为路线写入 ``pipeline_stage``；已有非空值不覆盖。"""
    for route in routes:
        if route.get("pipeline_stage"):
            continue
        stage = infer_route_stage(route, semantic_frame)
        if stage:
            route["pipeline_stage"] = stage


def card_stages(card: dict[str, Any]) -> set[str]:
    """读取卡片上已写入的阶段集合。

    兼容三种历史形态：``pipeline_stages``（本模块写入）、``evidence_roles``
    （复数，早期读取方假定存在但从未被写入）以及 ``evidence_role``（单数，
    聚类阶段实际写入的字段）。
    """
    stages: set[str] = set()
    for key in ("pipeline_stages", "evidence_roles"):
        for item in card.get(key) or []:
            stage = normalize_stage(item)
            if stage:
                stages.add(stage)
    single = normalize_stage(card.get("evidence_role"))
    if single:
        stages.add(single)
    return stages


def card_is_stage_agnostic(card: dict[str, Any]) -> bool:
    """判断卡片是否属于跨阶段角色（综述/基准），不受偏序约束。"""
    values = [
        card.get("evidence_role"),
        *(card.get("pipeline_stages") or []),
        *(card.get("evidence_roles") or []),
    ]
    return any(
        str(value or "").strip().lower() in STAGE_AGNOSTIC_ROLES
        for value in values
    )


def derive_card_stages(
    card: dict[str, Any],
    semantic_frame: dict[str, Any] | None,
) -> set[str]:
    """按语义帧的证据要求推导论文实际产出的阶段。

    判据复用 ``evidence_roles.direct_evidence_match``——它按「论文是否报告了
    该阶段的产物」判定，本身与学科无关。语义帧缺失时退回卡片已有标注。
    """
    frame = semantic_frame or {}
    requirements = list(frame.get("evidence_requirements") or [])
    if not requirements:
        return card_stages(card)

    from app.agent.evidence_roles import direct_evidence_match

    stages: set[str] = set()
    for requirement in requirements:
        stage = normalize_stage(requirement.get("evidence_role"))
        if not stage:
            continue
        matched, _reason = direct_evidence_match(card, requirement)
        if matched:
            stages.add(stage)
    return stages or card_stages(card)


def annotate_card_stages(
    cards: Iterable[dict[str, Any]],
    semantic_frame: dict[str, Any] | None = None,
) -> None:
    """就地为卡片写入 ``pipeline_stages``；已有非空值不覆盖。"""
    for card in cards:
        if card.get("pipeline_stages"):
            continue
        stages = derive_card_stages(card, semantic_frame)
        if stages:
            card["pipeline_stages"] = sorted(
                stages, key=lambda item: stage_rank(item)
            )


def stage_compatible(route: dict[str, Any], card: dict[str, Any]) -> bool:
    """论文能否支撑该路线：上游产物不得替代下游证据。

    规则（与学科无关）：

    * 路线未声明阶段，或论文无可判定阶段 → 不施加约束（交由词面/锚点信号）。
    * 跨阶段角色（综述/基准）→ 不施加约束。
    * 否则要求论文至少有一个阶段达到路线所需阶段：``max(paper) >= route``。
      即下游路线可以引用同级或更下游的证据，但纯上游论文不行。
    """
    target = stage_rank(route_stage(route))
    if target < 0:
        return True
    if card_is_stage_agnostic(card):
        return True
    ranks = [stage_rank(stage) for stage in card_stages(card)]
    ranks = [rank for rank in ranks if rank >= 0]
    if not ranks:
        return True
    return max(ranks) >= target


def stage_gap_reason(route: dict[str, Any], card: dict[str, Any]) -> str:
    """阶段不兼容时的可读原因；兼容时返回空串。"""
    if stage_compatible(route, card):
        return ""
    target = route_stage(route)
    stages = sorted(card_stages(card), key=lambda item: stage_rank(item))
    return (
        f"论文证据止于上游阶段 {stages or ['unknown']}，"
        f"不能替代路线所需的 {target} 阶段产物"
    )


# 各阶段的产物特征词：用于把补检索从上游产物推向缺失的下游产物。
# 这些词描述「该阶段产出什么形态的东西」，跨学科通用——任何领域的结构化
# 阶段都会谈规则/粒度/一致性，任何领域的解释阶段都会谈机制/成效/启示。
# 不含任何具体学科、对象、方法或应用场景名称。
_STAGE_PROBE_TERMS: dict[str, dict[str, list[str]]] = {
    STAGE_PERCEPTION: {
        "zh": ["识别", "检测", "自动提取"],
        "en": ["recognition", "detection", "automatic extraction"],
    },
    STAGE_STRUCTURED: {
        "zh": ["编码体系", "标注规范", "时间粒度", "一致性信度"],
        "en": ["coding scheme", "annotation protocol", "inter-rater reliability"],
    },
    STAGE_ANALYSIS: {
        "zh": ["序列分析", "模式分析", "统计建模"],
        "en": ["sequence analysis", "pattern analysis", "statistical modeling"],
    },
    STAGE_INTERPRETATION: {
        "zh": ["作用机制", "成效评估", "实践启示"],
        "en": ["mechanism", "effectiveness evaluation", "practical implication"],
    },
}


def stage_probe_terms(role: Any) -> dict[str, list[str]]:
    """返回该阶段的产物特征词，按语言分组；未知阶段返回空表。

    调用方按别名语言选取同语言词项，避免拼出中英混杂的检索式。
    """
    stage = normalize_stage(role)
    if not stage:
        return {}
    return {
        key: list(values)
        for key, values in _STAGE_PROBE_TERMS.get(stage, {}).items()
    }


__all__ = (
    "STAGE_ORDER",
    "STAGE_AGNOSTIC_ROLES",
    "annotate_card_stages",
    "card_is_stage_agnostic",
    "card_stages",
    "derive_card_stages",
    "normalize_stage",
    "route_stage",
    "stage_compatible",
    "stage_gap_reason",
    "stage_probe_terms",
    "stage_rank",
)
