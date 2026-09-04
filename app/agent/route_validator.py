"""双语、可解释且证据充分性解耦的 Route Validator。"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Iterable

from app.core.json_utils import parse_json_object
from app.core.logger import get_logger
from app.schemas.route_validation_schema import (
    EvidenceSufficiencyAssessment,
    RouteAction,
    RoutePaperMatchFeatures,
    RouteStatus,
    RouteValidityAssessment,
)

logger = get_logger(__name__)

# 策略默认值集中于 app.core.config.ReviewThresholdPolicy；保留这些名称兼容旧模块导入。
from app.core.config import get_review_threshold_policy

_ANCHOR_TYPES = {"semantic", "method", "task"}
_MIN_SPLITTABLE_CORE = 6
_OVERSIZED_CORE_SHARE_FACTOR = 1.2
_MAX_SUB_ROUTES = 3


def _unique(values: Iterable[Any], *, limit: int = 24) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def _tokens(value: str) -> set[str]:
    text = str(value or "").casefold()
    latin = set(re.findall(r"[a-z][a-z0-9-]{1,}", text))
    chunks = re.findall(r"[\u4e00-\u9fff]+", text)
    cjk = {
        chunk[index:index + 2]
        for chunk in chunks
        for index in range(max(1, len(chunk) - 1))
        if chunk[index:index + 2]
    }
    return latin | cjk


def _compact(value: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(value or "").casefold())


def _term_match(term: str, text: str) -> float:
    term_compact = _compact(term)
    text_compact = _compact(text)
    if len(term_compact) >= 4 and term_compact in text_compact:
        return 1.0
    term_tokens = _tokens(term)
    text_tokens = _tokens(text)
    if not term_tokens:
        return 0.0
    return len(term_tokens & text_tokens) / len(term_tokens)


def _best_matches(terms: list[str], text: str) -> list[tuple[str, float]]:
    return sorted(
        ((term, _term_match(term, text)) for term in terms),
        key=lambda item: (-item[1], item[0]),
    )


def _top_coverage(terms: list[str], text: str, *, target_hits: int = 3) -> float:
    if not terms:
        return 0.0
    scores = [score for _, score in _best_matches(terms, text)]
    divisor = min(target_hits, len(scores))
    return min(1.0, sum(scores[:divisor]) / max(1, divisor))


def _claim_text(card: dict[str, Any]) -> str:
    return " ".join(
        str(claim.get("claim") or claim.get("text") or claim.get("statement") or "")
        for claims in (card.get("field_claims") or {}).values()
        for claim in claims or []
        if isinstance(claim, dict)
    )


def _paper_text(card: dict[str, Any]) -> str:
    keywords = card.get("keywords") or []
    if not isinstance(keywords, (list, tuple, set)):
        keywords = [keywords]
    return " ".join([
        str(card.get("title") or ""),
        str(card.get("abstract") or ""),
        str(card.get("research_problem") or ""),
        str(card.get("method") or ""),
        " ".join(str(item) for item in keywords),
        _claim_text(card),
    ])


def _anchor_inventory(route: dict[str, Any]) -> dict[str, list[str]]:
    return {
        "semantic": _unique([
            *(route.get("semantic_anchors") or []),
            *(route.get("core_concepts") or []),
        ]),
        "method": _unique(route.get("method_concepts") or []),
        "task": _unique(route.get("task_anchors") or []),
        # Exclusion criteria are full boundary sentences and commonly repeat
        # positive context (for example "exclude image-only few-shot action
        # recognition").  Treating the whole sentence as a negative anchor
        # creates a false positive/negative self-conflict.  Only the planner's
        # dedicated atomic negative anchors participate in matching.
        "negative": _unique(route.get("negative_anchors") or []),
    }


def _expansion_support_terms(route: dict[str, Any]) -> dict[str, set[str]]:
    shared = _unique([
        route.get("name"), route.get("research_question"),
        *(route.get("core_concepts") or []),
    ])
    return {
        "semantic": {_compact(item) for item in shared},
        "method": {_compact(item) for item in [*shared, *(route.get("method_concepts") or [])]},
        "task": {_compact(item) for item in [*shared, *(route.get("task_anchors") or [])]},
    }


def guard_anchor_expansions(route: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """校验扩展锚点的类型、来源绑定和负向边界，阻止 Route drift。"""
    normalized = dict(route)
    inventory = _anchor_inventory(normalized)
    support_terms = _expansion_support_terms(normalized)
    route_context = _unique([
        *(normalized.get("search_queries") or []),
        *inventory["semantic"], *inventory["method"], *inventory["task"],
    ])
    accepted: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []

    for item in normalized.get("anchor_expansions") or []:
        if not isinstance(item, dict):
            rejected.append({"text": str(item), "reason": "expansion_not_structured"})
            continue
        text = re.sub(r"\s+", " ", str(item.get("text") or "")).strip()
        anchor_type = str(item.get("anchor_type") or "").strip().lower()
        supports = re.sub(r"\s+", " ", str(item.get("supports") or "")).strip()
        record = {"text": text, "anchor_type": anchor_type, "supports": supports}
        if not text or len(text) > 100 or anchor_type not in _ANCHOR_TYPES:
            rejected.append({**record, "reason": "invalid_anchor_shape"})
            continue
        if _compact(supports) not in support_terms.get(anchor_type, set()):
            rejected.append({**record, "reason": "unsupported_route_expansion"})
            continue
        negative_conflict = max(
            (_term_match(text, negative) for negative in inventory["negative"]),
            default=0.0,
        )
        if negative_conflict >= 0.8:
            rejected.append({**record, "reason": "negative_boundary_conflict"})
            continue
        context_alignment = max(
            (_term_match(text, context) for context in route_context),
            default=0.0,
        )
        support_alignment = _term_match(text, supports)
        if max(context_alignment, support_alignment) < 0.34:
            rejected.append({**record, "reason": "route_drift_risk"})
            continue
        inventory[anchor_type].append(text)
        accepted.append(record)

    normalized["semantic_anchors"] = _unique(inventory["semantic"], limit=12)
    normalized["method_concepts"] = _unique(inventory["method"], limit=12)
    normalized["task_anchors"] = _unique(inventory["task"], limit=12)
    normalized["negative_anchors"] = _unique(inventory["negative"], limit=12)
    normalized["anchor_provenance"] = [
        *[item for item in normalized.get("anchor_provenance") or [] if isinstance(item, dict)],
        *[{**item, "source": "llm_expansion"} for item in accepted],
    ]
    normalized["anchor_expansions"] = []
    return normalized, {"accepted": accepted, "rejected": rejected}


def _needs_anchor_expansion(route: dict[str, Any]) -> bool:
    anchors = _anchor_inventory(route)
    latin_anchors = [
        item for item in [*anchors["semantic"], *anchors["method"], *anchors["task"]]
        if re.search(r"[a-zA-Z]", item)
    ]
    return len(latin_anchors) < 2


def _llm_anchor_expansions(routes: list[dict[str, Any]], llm) -> dict[str, list[dict[str, str]]]:
    if llm is None:
        return {}
    targets = [route for route in routes if _needs_anchor_expansion(route)]
    if not targets:
        return {}
    try:
        from app.prompt.route_validation import ROUTE_ANCHOR_EXPANSION_PROMPT

        payload = [{
            key: route.get(key)
            for key in (
                "route_id", "name", "research_question", "core_concepts",
                "method_concepts", "task_anchors", "negative_anchors",
                "inclusion_criteria", "exclusion_criteria",
            )
        } for route in targets]
        response = llm.complete(
            ROUTE_ANCHOR_EXPANSION_PROMPT.format(
                routes_json=json.dumps(payload, ensure_ascii=False)
            ),
            response_format="json",
            temperature=0.0,
            operation="expand_route_semantic_anchors",
        )
        data = parse_json_object(response if isinstance(response, str) else str(response))
        return {
            str(item.get("route_id") or ""): [
                expansion for expansion in item.get("anchor_expansions") or []
                if isinstance(expansion, dict)
            ]
            for item in data.get("routes") or []
            if isinstance(item, dict) and item.get("route_id")
        }
    except Exception as exc:
        logger.warning("Route anchor expansion failed; using planner anchors: %s", exc)
        return {}


def prepare_route_anchors(
    routes: list[dict[str, Any]],
    llm=None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """一次性扩展并校验路线锚点；Prompt 仅在锚点不足时按需加载。"""
    expansions = _llm_anchor_expansions(routes, llm)
    prepared: list[dict[str, Any]] = []
    reports: dict[str, dict[str, Any]] = {}
    for route in routes:
        candidate = dict(route)
        route_id = str(candidate.get("route_id") or "")
        candidate["anchor_expansions"] = [
            *(candidate.get("anchor_expansions") or []),
            *(expansions.get(route_id) or []),
        ]
        guarded, report = guard_anchor_expansions(candidate)
        prepared.append(guarded)
        reports[route_id] = report
    return prepared, reports


def extract_route_paper_features(
    route: dict[str, Any],
    card: dict[str, Any],
) -> RoutePaperMatchFeatures:
    """输出独立匹配特征；匹配策略稍后才将它们分为 core/supporting。"""
    anchors = _anchor_inventory(route)
    paper_text = _paper_text(card)
    claim_text = _claim_text(card)
    method_text = " ".join([
        str(card.get("method") or ""),
        " ".join(
            str(claim.get("claim") or claim.get("text") or "")
            for field, claims in (card.get("field_claims") or {}).items()
            if "method" in str(field).lower()
            for claim in claims or []
            if isinstance(claim, dict)
        ),
    ])
    positive_anchors = _unique([
        *anchors["semantic"], *anchors["method"], *anchors["task"]
    ])
    best = _best_matches(positive_anchors, paper_text)
    matched = [term for term, score in best if score >= 0.8]
    method_matches = [
        term for term, score in _best_matches(anchors["method"], method_text)
        if score >= 0.8
    ]
    lexical = min(1.0, len(matched) / max(1, min(3, len(positive_anchors))))
    concept = _top_coverage(positive_anchors, paper_text)
    claim_match = _top_coverage(positive_anchors, claim_text, target_hits=2)
    method_match = _top_coverage(anchors["method"], method_text, target_hits=2)
    task_match = _top_coverage(anchors["task"], paper_text, target_hits=2)
    route_text = " ".join([
        str(route.get("research_question") or ""), *positive_anchors
    ])
    route_tokens = _tokens(route_text)
    paper_tokens = _tokens(paper_text)
    semantic = (
        len(route_tokens & paper_tokens) / math.sqrt(len(route_tokens) * len(paper_tokens))
        if route_tokens and paper_tokens else 0.0
    )
    negative = max(
        (_term_match(term, paper_text) for term in anchors["negative"]),
        default=0.0,
    )
    # 阶段角色：此前读的是 ``evidence_roles``（复数），但全库只写入过单数
    # ``evidence_role``，因此该集合恒为空、``role_score`` 恒退化为 task_match，
    # 阶段校验从未生效——上游感知论文只靠词面命中就能成为下游路线的核心证据。
    # 现在统一由 pipeline_stages 归一两套词汇后比对。
    from app.agent.pipeline_stages import (
        route_stage as _route_stage,
        stage_compatible as _stage_compatible,
        stage_gap_reason as _stage_gap_reason,
        stage_rank as _stage_rank,
        card_stages as _card_stages,
    )

    declared_stage = _route_stage(route)
    paper_stages = _card_stages(card)
    role_score = (
        1.0
        if declared_stage and _stage_rank(declared_stage) in {
            _stage_rank(stage) for stage in paper_stages
        }
        else task_match
    )
    stage_ok = _stage_compatible(route, card)
    stage_reason = _stage_gap_reason(route, card)
    signals = [
        semantic >= 0.18,
        concept >= 0.34,
        lexical >= 0.34,
        claim_match >= 0.34,
        method_match >= 0.34,
        role_score >= 0.5,
    ]
    signal_count = sum(signals)
    anchor_backed = lexical >= 0.34 or claim_match >= 0.34 or method_match >= 0.34
    if negative < 0.8 and signal_count >= 2 and anchor_backed:
        match_level = "core"
    elif negative < 0.8 and signal_count >= 1:
        match_level = "supporting"
    else:
        match_level = "none"
    # 阶段不兼容：词面信号再强也不得升为核心证据。这是「上游产物不能替代
    # 下游证据」的执行点，与具体学科无关。
    if not stage_ok and match_level == "core":
        match_level = "supporting"
    return RoutePaperMatchFeatures(
        semantic_similarity=min(1.0, semantic),
        concept_coverage=concept,
        lexical_anchor_score=lexical,
        evidence_claim_match=claim_match,
        method_compatibility=method_match,
        evidence_role_score=role_score,
        negative_anchor_conflict=negative,
        matched_anchors=matched,
        matched_method_concepts=method_matches,
        positive_signal_count=signal_count,
        match_level=match_level,
        route_stage=declared_stage,
        paper_stages=sorted(paper_stages, key=_stage_rank),
        stage_compatible=stage_ok,
        stage_conflict_reason=stage_reason,
    )


def assess_route_validity(
    route: dict[str, Any],
    anchor_report: dict[str, Any] | None = None,
) -> RouteValidityAssessment:
    """只使用路线定义计算结构有效性，禁止读取 Evidence Card。"""
    report = anchor_report or {}
    anchors = _anchor_inventory(route)
    definition_checks = [
        bool(route.get("name")), bool(route.get("research_question")),
        len(route.get("core_concepts") or []) >= 2,
        bool(route.get("inclusion_criteria")), bool(route.get("route_role")),
    ]
    definition = sum(definition_checks) / len(definition_checks)
    boundary_checks = [
        bool(route.get("inclusion_criteria")), bool(route.get("exclusion_criteria")),
        bool(route.get("boundary_note")),
    ]
    boundary = sum(boundary_checks) / len(boundary_checks)
    positive = _unique([*anchors["semantic"], *anchors["method"], *anchors["task"]])
    anchor_grounding = min(1.0, len(positive) / 4) if positive else 0.0
    positive_negative_conflict = max(
        (
            _term_match(pos, neg)
            for pos in positive for neg in anchors["negative"]
        ),
        default=0.0,
    )
    consistency = max(0.0, 1.0 - positive_negative_conflict)
    role_clarity = 1.0 if route.get("route_role") else 0.0
    # The score is diagnostic only. Policy decisions below use the individual
    # features and explicit structural predicates, not a weighted match score.
    score = (definition + anchor_grounding + boundary + consistency + role_clarity) / 5
    structurally_valid = bool(
        definition >= 0.6
        and anchor_grounding >= 0.5
        and consistency >= 0.5
    )
    reasons: list[str] = []
    if definition < 0.6:
        reasons.append("route definition is incomplete")
    if anchor_grounding < 0.5:
        reasons.append("route lacks grounded semantic anchors")
    if consistency < 0.5:
        reasons.append("positive and negative route boundaries conflict")
    return RouteValidityAssessment(
        score=score,
        definition_completeness=definition,
        anchor_grounding=anchor_grounding,
        boundary_clarity=boundary,
        internal_consistency=consistency,
        role_clarity=role_clarity,
        structurally_valid=structurally_valid,
        rejected_anchor_expansions=list(report.get("rejected") or []),
        reasons=reasons,
    )


def assess_evidence_sufficiency(
    core_ids: list[str],
    supporting_ids: list[str],
    card_map: dict[str, dict[str, Any]],
    *,
    minimum_core_evidence: int,
) -> EvidenceSufficiencyAssessment:
    """只使用已匹配证据计算充分性，不读取路线结构完整度。"""
    evidence_ids = _unique([*core_ids, *supporting_ids], limit=10000)
    source_keys = {
        str(card_map.get(paper_id, {}).get("doi") or "").strip().lower()
        or paper_id
        for paper_id in evidence_ids
    }
    quality_count = sum(
        str(card_map.get(paper_id, {}).get("quality_status") or "").lower() != "invalid"
        for paper_id in evidence_ids
    )
    quality_rate = quality_count / len(evidence_ids) if evidence_ids else 0.0
    target = max(1, minimum_core_evidence)
    score = min(1.0, (len(core_ids) + 0.5 * len(supporting_ids)) / target)
    sufficient = len(core_ids) >= target and quality_rate >= 0.5
    reasons = [] if sufficient else [
        f"core evidence {len(core_ids)}/{target}; supporting evidence {len(supporting_ids)}"
    ]
    return EvidenceSufficiencyAssessment(
        score=score,
        sufficient=sufficient,
        core_evidence_count=len(core_ids),
        supporting_evidence_count=len(supporting_ids),
        independent_source_count=len(source_keys),
        evidence_quality_rate=quality_rate,
        core_paper_ids=core_ids,
        supporting_paper_ids=supporting_ids,
        reasons=reasons,
    )


def _decision(
    validity: RouteValidityAssessment,
    sufficiency: EvidenceSufficiencyAssessment,
    pair_features: list[RoutePaperMatchFeatures],
) -> tuple[RouteStatus, RouteAction, str]:
    if validity.structurally_valid and sufficiency.sufficient:
        return RouteStatus.KEEP, RouteAction.KEEP, "route is structurally valid and sufficiently supported"
    if validity.structurally_valid:
        return RouteStatus.WEAK, RouteAction.TARGETED_SEARCH, "valid route with insufficient evidence"
    if sufficiency.sufficient:
        return RouteStatus.WEAK, RouteAction.ROUTE_REVISION, "evidence exists but route definition requires revision"
    best_signal_count = max((item.positive_signal_count for item in pair_features), default=0)
    clearly_invalid = bool(
        validity.definition_completeness < 0.4
        and validity.anchor_grounding < 0.5
        and best_signal_count < 2
    )
    if clearly_invalid:
        return RouteStatus.DROP, RouteAction.DROP, "low validity, low compatibility, and insufficient evidence"
    return RouteStatus.WEAK, RouteAction.ROUTE_REVISION, "route is not yet valid enough for final use"


def route_fit_score(features: RoutePaperMatchFeatures) -> float:
    """论文与路线的综合契合度，用于在多条路线都判 core 时选出唯一归属。

    独占归属决定写作层每个小节的真实体量，因此不能按路线书写顺序取第一条：
    core 成员在路线间大面积重叠（实测 60 篇里 42 篇同时是 5 条路线的 core），
    取 ``core_routes[0]`` 会让首条路线独吞 42 篇，其余 4 条各剩 1-4 篇，正文
    退化为「本节纳入 1 篇文献」的罗列（2026-08-29 会话实测四节退化）。

    评分只累加已有的匹配特征，不引入任何领域词表；各特征本身已在
    ``extract_route_paper_features`` 内归一到 [0, 1]。
    """
    return sum((
        features.semantic_similarity,
        features.concept_coverage,
        features.lexical_anchor_score,
        features.evidence_claim_match,
        features.method_compatibility,
        features.evidence_role_score,
    ))


def _best_fit_route(
    paper_id: str,
    route_ids: list[str],
    per_route_features: dict[str, dict[str, RoutePaperMatchFeatures]],
    route_order: list[str],
) -> str:
    """在候选路线里选契合度最高的一条；同分时按路线顺序保持确定性。"""
    return max(
        route_ids,
        key=lambda route_id: (
            route_fit_score(per_route_features[route_id][paper_id]),
            -route_order.index(route_id),
        ),
    )


def merge_weak_routes_for_writing(
    routes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """在写作边界把 WEAK 路线并入最接近的可见路线。"""
    if len(routes) <= 1:
        return [dict(route) for route in routes]

    def route_tokens(route: dict[str, Any]) -> set[str]:
        return _tokens(" ".join([
            str(route.get("name") or ""),
            str(route.get("research_question") or ""),
            *[str(item) for item in route.get("core_concepts") or []],
        ]))

    strong = [dict(route) for route in routes if str(route.get("status") or "").upper() != "WEAK"]
    weak = [dict(route) for route in routes if str(route.get("status") or "").upper() == "WEAK"]
    if not weak:
        return [dict(route) for route in routes]
    if not strong:
        ranked = sorted(
            weak,
            key=lambda route: len(route.get("core_paper_ids") or route.get("paper_ids") or []),
            reverse=True,
        )
        strong, weak = ranked[:1], ranked[1:]

    for source in weak:
        source_tokens = route_tokens(source)
        target = max(
            strong,
            key=lambda route: (
                len(source_tokens & route_tokens(route)) / max(1, len(source_tokens | route_tokens(route))),
                len(route.get("core_paper_ids") or route.get("paper_ids") or []),
            ),
        )
        for field in ("paper_ids", "core_paper_ids", "supporting_paper_ids"):
            target[field] = _unique([
                *[str(item) for item in target.get(field) or []],
                *[str(item) for item in source.get(field) or []],
            ], limit=10000)
        target["merged_route_ids"] = _unique([
            *[str(item) for item in target.get("merged_route_ids") or []],
            str(source.get("route_id") or ""),
            *[str(item) for item in source.get("merged_route_ids") or []],
        ])
    return strong


def _split_oversized_routes(
    validated_routes: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    card_map: dict[str, dict[str, Any]],
    primary_owner: dict[str, str],
    llm=None,
    topic: str = "",
    policy=None,
) -> dict[str, str]:
    """把独占成员份额过高的路线拆成子路线，返回 论文 → 子路线 映射。

    判据用相对份额而非绝对篇数（避免把某次检索规模写死成领域常量），
    且度量的是**独占 primary 归属数**而不是 core 成员数——后者在路线间
    重叠，无法反映写作层每个小节的真实体量。原判据只有下限（core >= N
    即 STRONG_ROUTE），因此一条独占半数文献的路线也被判健康，写作层随之
    坍缩成巨型小节。

    子路线数按超额倍数推算并夹在 [2, ``_MAX_SUB_ROUTES``]，避免把一条
    路线打碎成远超写作层小节预算的碎片；拆不出足够子簇时保持原状。
    """
    if policy is None:
        policy = get_review_threshold_policy()
    if len(validated_routes) < 2:
        return {}
    exclusive: dict[str, list[str]] = {}
    for paper_id, route_id in primary_owner.items():
        exclusive.setdefault(str(route_id), []).append(str(paper_id))
    counts = [
        len(exclusive.get(str(route.get("route_id") or ""), []))
        for route in validated_routes
    ]
    even_share = sum(counts) / len(counts) if counts else 0.0
    if even_share <= 0:
        return {}
    token_cache = {
        paper_id: _tokens(_paper_text(card))
        for paper_id, card in card_map.items()
    }
    reassigned: dict[str, str] = {}
    for route in list(validated_routes):
        parent_id = str(route.get("route_id") or "")
        members = exclusive.get(parent_id) or []
        if (
            len(members) < policy.route_min_splittable_core
            or len(members) <= even_share * policy.route_oversized_share_factor
        ):
            continue
        target_count = min(
            policy.route_max_sub_routes,
            max(2, round(len(members) / even_share)),
        )
        clusters = _split_into_clusters(members, token_cache, target_count)
        if len(clusters) < 2:
            continue
        # 父路线的支撑成员保留给每条子路线用于引用授权：它们的主题归属
        # 由各自的 primary 决定，不会让子路线在主题层重新重叠。
        parent_supporting = [
            str(item) for item in route.get("supporting_paper_ids") or []
        ]
        sub_routes: list[dict[str, Any]] = []
        # 全局名称去重的预留清单：拆分子路线时，正文还会同时出现其余全部
        # 存活路线。候选名与其中任何一条共享概念（如"跨模态"对既有
        # "多模态与自监督"）都会让读者无法区分两个小节，必须避开。
        other_route_names = [
            str(item.get("name") or "").strip()
            for item in validated_routes
            if item is not route and str(item.get("name") or "").strip()
        ]
        deterministic_names = _distinctive_cluster_labels(
            clusters, card_map,
            parent_name=str(route.get("name") or parent_id),
            topic=topic,
            reserved_names=other_route_names,
        )
        if len(deterministic_names) < len(clusters):
            # 至少一个子簇取不出可读名称：放弃拆分，保留父路线。子路线名会
            # 成为正文小节标题，没有可区分的名字就没有拆分价值。
            logger.info(
                "Route %r split abandoned: no distinctive names for all %d clusters",
                route.get("name") or parent_id, len(clusters),
            )
            continue
        for index, cluster in enumerate(clusters, 1):
            sibling_names = [str(item.get("name") or "") for item in sub_routes]
            sub_routes.append({
                **route,
                "route_id": f"{parent_id}_S{index}",
                "parent_route_id": parent_id,
                "name": _sub_route_name(
                    cluster, card_map, llm,
                    fallback=deterministic_names[index - 1],
                    topic=topic,
                    parent_name=str(route.get("name") or ""),
                    sibling_names=sibling_names,
                    reserved_names=other_route_names,
                ),
                "core_paper_ids": list(cluster),
                "supporting_paper_ids": list(parent_supporting),
                "paper_ids": _unique([*cluster, *parent_supporting], limit=10000),
            })
        position = validated_routes.index(route)
        validated_routes[position:position + 1] = sub_routes
        for sub_route, cluster in zip(sub_routes, clusters):
            for paper_id in cluster:
                reassigned[str(paper_id)] = str(sub_route["route_id"])
        decisions.append({
            "route_id": parent_id,
            "route_name": route.get("name"),
            "status": RouteStatus.KEEP.value,
            "diagnosis": "OVERSIZED_ROUTE",
            "action": "SPLIT_INTO",
            "split_count": len(sub_routes),
            "sub_route_ids": [item["route_id"] for item in sub_routes],
            "reason": (
                f"独占成员 {len(members)} 篇超过路线均分份额 {even_share:.1f} 的 "
                f"{_OVERSIZED_CORE_SHARE_FACTOR} 倍，按证据子聚类拆为 "
                f"{len(sub_routes)} 条子路线"
            ),
        })
    return reassigned


def _sub_route_name(
    cluster: list[str],
    card_map: dict[str, dict[str, Any]],
    llm,
    *,
    fallback: str,
    topic: str = "",
    parent_name: str = "",
    sibling_names: list[str] | None = None,
    reserved_names: list[str] | None = None,
) -> str:
    """为拆出的子路线取语义名称；LLM 不可用或命名不合格时用确定性兜底名。

    子路线名会成为正文小节标题。它必须命名"本簇相对全部既有路线的区分点"，
    而不是复述研究主题：实测 LLM 在无上下文时会产出"跨模态小样本视频动作
    识别""时空对齐的零样本动作识别"这类名称——既把主题整句搬进小节标题，
    又擅自把主题的任务设定（少样本）改写成别的设定（零样本）；即使命名合格，
    也可能与既有路线指向同一概念（"跨模态匹配"对既有"多模态与自监督"）。
    因此把主题、父路线、同级与全部既有路线名称一并给出，并对结果做确定性
    校验（含共享概念子串判定，见 _name_overlaps）。
    """
    siblings = [str(item).strip() for item in sibling_names or [] if str(item).strip()]
    reserved = list(dict.fromkeys(
        str(item).strip()
        for item in [*(reserved_names or []), *siblings]
        if str(item).strip()
    ))
    if llm is None or not cluster:
        return fallback
    try:
        from app.agent.provisional_routes import _name_cluster

        cards = [card_map[pid] for pid in cluster if pid in card_map]
        name = _name_cluster(
            cards, llm,
            topic=topic, parent_name=parent_name, sibling_names=siblings,
            reserved_names=reserved,
        )
        if _is_valid_sub_route_name(
            name, topic=topic, sibling_names=siblings, reserved_names=reserved,
        ):
            return name
        logger.info(
            "Sub-route name %r rejected (restates topic or overlaps an existing "
            "route name); using deterministic name", name,
        )
        return fallback
    except Exception as exc:  # noqa: BLE001
        logger.warning("Sub-route naming failed, using deterministic name: %s", exc)
        return fallback


_NAME_CONNECTOR_RE = re.compile(r"[\s,，、;；/与和及]+")


def _name_words(name: str) -> list[str]:
    """把路线名按连接词切成概念词单元；纯英文名按词切。"""
    words = [
        item for item in _NAME_CONNECTOR_RE.split(str(name or ""))
        if item.strip()
    ]
    return words or [str(name or "").strip()]


def _name_overlaps(candidate: str, reserved: str) -> bool:
    """判断两个路线名是否指向同一概念而让读者无法区分。

    纯词法判据，不维护同义词表：
    1. 归一化相同（含英文名）；
    2. 概念词级包含（"数据增强" ⊂ "骨架数据增强"）；
    3. 概念词剥掉至多 1 个前导修饰字后公共前缀 ≥2 字——"跨模态/多模态"
       剥"跨/多"后同为"模态"判重叠；"度量学习/迁移学习"剥 1 字后无
       公共前缀，不误判（共用领域后缀但指向不同机制）；
    4. 任意位置共享 ≥3 字子串（兜住长概念词整体复用）。
    """
    left = _normalize_name(candidate)
    right = _normalize_name(reserved)
    if not left or not right:
        return False
    if left == right:
        return True

    def _word_cjk_grams(name: str) -> set[str]:
        # 3-gram 只在单个概念词内部取：CJK 提取若保留连接词（与/和/及），
        # 会拼出"学习与"这类跨词伪概念，把"度量学习"与"元学习"误判重叠。
        grams: set[str] = set()
        for word in _name_words(name):
            text = "".join(re.findall(r"[\u4e00-\u9fff]", word))
            for index in range(0, max(0, len(text) - 2)):
                grams.add(text[index:index + 3])
        return grams

    left_grams = _word_cjk_grams(candidate)
    if left_grams and (left_grams & _word_cjk_grams(reserved)):
        return True
    left_cjk = "".join(re.findall(r"[\u4e00-\u9fff]", str(candidate or "")))
    right_cjk = "".join(re.findall(r"[\u4e00-\u9fff]", str(reserved or "")))
    if len(left_cjk) >= 3 and len(right_cjk) >= 3:
        if left_cjk.find(right_cjk) >= 0 or right_cjk.find(left_cjk) >= 0:
            return True

    def _stems(word: str) -> list[str]:
        text = "".join(re.findall(r"[\u4e00-\u9fff]", word))
        return [text, text[1:]] if len(text) >= 3 else ([text] if text else [])

    for word_left in _name_words(candidate):
        for stem_left in _stems(word_left):
            if len(stem_left) < 2:
                continue
            for word_right in _name_words(reserved):
                for stem_right in _stems(word_right):
                    if len(stem_right) < 2:
                        continue
                    common = 0
                    for a, b in zip(stem_left, stem_right):
                        if a != b:
                            break
                        common += 1
                    if common >= 2:
                        return True
    return False


def _is_valid_sub_route_name(
    name: str,
    *,
    topic: str,
    sibling_names: list[str],
    reserved_names: list[str] | None = None,
) -> bool:
    """子路线名必须是区分点：不得复述主题核心词，也不得与既有路线指向同一概念。

    "不得包含主题核心词"这一条同时挡住了任务设定漂移——"零样本动作识别"
    含主题核心词"动作识别"，会被判不合格并退回确定性名，不需要维护
    任何领域词表或设定词对照表。reserved_names 覆盖全部既有路线（含
    同级与父级以外的路线）：候选名与任何预留名共享概念（如"跨模态"对
    "多模态"共享"模态"）同样拒收，否则正文会出现两个读者无法区分的
    同主题小节。
    """
    cleaned = str(name or "").strip()
    if not cleaned or len(cleaned) > 30:
        return False
    normalized = _normalize_name(cleaned)
    if not normalized:
        return False
    if any(_normalize_name(item) == normalized for item in sibling_names):
        return False
    for reserved in [item for item in (reserved_names or []) if str(item).strip()]:
        if _name_overlaps(cleaned, str(reserved)):
            return False
    for core in _topic_core_terms(topic):
        if core and core in normalized:
            return False
    return True


def _normalize_name(value: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", str(value or "").lower())


def _topic_core_terms(topic: str) -> list[str]:
    """从研究主题中取出核心词，用于判断子路线名是否只是复述主题。

    取主题去掉修饰后的完整串与其尾部实体片段（中文按 3-4 字尾缀、英文按
    末两词），不依赖任何领域词表。
    """
    normalized = _normalize_name(topic)
    if not normalized:
        return []
    terms = [normalized]
    chinese = re.findall(r"[\u4e00-\u9fff]+", str(topic or ""))
    if chinese:
        tail = chinese[-1]
        for size in (4, 3):
            if len(tail) >= size:
                terms.append(_normalize_name(tail[-size:]))
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9-]*", str(topic or ""))
    if len(words) >= 2:
        terms.append(_normalize_name(" ".join(words[-2:])))
    return list(dict.fromkeys(item for item in terms if len(item) >= 2))


def _phrase_candidates(card: dict[str, Any]) -> list[str]:
    """从卡片的结构化字段取**中文**短语级候选名。

    ``taxonomy_tokens`` 用 2/3/4 字滑窗切中文，会产出"学生课堂""堂行为分"
    这类跨词片段，且同一短语的各 n-gram 词频完全相同，片段可以和真实概念词
    同分胜出（实测兜底名"行为分析框架构建：学生课堂"）。结构化字段本身就是
    短语，优先用它们作候选可避免片段。

    只接受含中文的短语：子路线名会直接成为中文小节标题，结构化字段里的
    英文短标签（如 ``data_modalities`` 的 "audio"/"video"）拼进标题就成了
    实测出现的"课堂行为编码与分析框架：audio"。英文模态标签的区分工作
    交由 n-gram 层的英文词元处理，那里本来就按词边界对齐。
    """
    values: list[str] = []
    for key in ("data_modalities", "behavior_categories", "metrics"):
        values.extend(str(item).strip() for item in (card.get(key) or []))
    for key in ("study_design", "dataset"):
        values.append(str(card.get(key) or "").strip())
    return [
        value for value in values
        if 2 <= len(value) <= 20 and re.search(r"[\u4e00-\u9fff]", value)
    ]


def _is_boundary_aligned_gram(token: str, texts: list[str]) -> bool:
    """判断中文候选词是否在原文中至少一次出现在词组边界上。

    ``taxonomy_tokens`` 按 2/3/4 字滑窗切中文，同一短语的各 n-gram 词频完全
    相同，跨词片段因此能与真实概念词同分胜出（实测兜底名"…：学生课堂"）。
    中文没有空格，无法直接判定词边界，这里用可观测的近似：候选词只要在某个
    连续中文串的开头或结尾出现过，就认为它是一个可独立成词的单元。

    "多模态"是"多模态融合的课堂教学质量评估"的开头，判为对齐；"学生课堂"
    在"应用深度学习的学生课堂行为分析系统"中既非开头也非结尾，判为片段。
    """
    if not re.search(r"[\u4e00-\u9fff]", token):
        # 英文词元由 [a-z][a-z0-9_-]{2,} 正则切出，本身已按词边界对齐。
        return True
    for text in texts:
        for run in re.findall(r"[\u4e00-\u9fff]+", str(text or "")):
            if run.startswith(token) or run.endswith(token):
                return True
    return False


def _distinctive_cluster_labels(
    clusters: list[list[str]],
    card_map: dict[str, dict[str, Any]],
    *,
    parent_name: str,
    topic: str = "",
    reserved_names: list[str] | None = None,
) -> list[str]:
    """为每个子簇取一个"相对其他子簇最具区分度"的词项作为兜底名。

    单看簇内词频会选出各簇共有的泛词（如 recognition），三条子路线因此
    重名。这里用"簇内占比 − 簇外占比"排序，只保留真正区分本簇的词项，
    纯数据驱动且天然互不重复，不维护任何领域词表。reserved_names 提供
    既有路线名：与它们共享概念的词项同样不可用（如既有"多模态与自监督"
    路线时，modal/模态类词项会让兜底名与其他小节无法区分）。

    候选来源分两级：结构化字段短语优先，n-gram 词元次之且必须通过词边界
    校验——否则中文滑窗片段会成为小节标题。两级都无合格候选时用编号名，
    宁可平淡也不让读者看到截断词组。
    """
    topic_terms = _topic_core_terms(topic)
    reserved = [str(item).strip() for item in reserved_names or [] if str(item).strip()]

    def token_reserved(token: str) -> bool:
        return any(_name_overlaps(token, item) for item in reserved)

    def acceptable(token: str) -> bool:
        normalized = _normalize_name(token)
        if not normalized:
            return False
        if any(
            core and (core in normalized or normalized in core)
            for core in topic_terms
        ):
            return False
        return not token_reserved(token)

    def cluster_candidates(
        cluster: list[str],
    ) -> tuple[dict[str, int], dict[str, int], list[str]]:
        # 复用聚类的停用词表过滤 learning/approach 这类无区分度的学术
        # 泛词——直接用 _tokens 会把泛词选成兜底名（实测三条子路线分别
        # 叫 度量学习与特征对齐：learning/approach/multimodal）。
        from app.tools.cluster_papers import taxonomy_tokens

        phrase_counts: dict[str, int] = {}
        gram_counts: dict[str, int] = {}
        texts: list[str] = []
        for paper_id in cluster:
            card = card_map.get(paper_id) or {}
            text = " ".join([
                str(card.get("title") or ""),
                str(card.get("research_problem") or ""),
                str(card.get("method") or ""),
            ])
            texts.append(text)
            for phrase in _phrase_candidates(card):
                if acceptable(phrase):
                    phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
            for token in taxonomy_tokens(text):
                if len(token) < 3 or not acceptable(token):
                    continue
                gram_counts[token] = gram_counts.get(token, 0) + 1
        return phrase_counts, gram_counts, texts

    prepared = [cluster_candidates(cluster) for cluster in clusters]
    per_phrase = [item[0] for item in prepared]
    per_gram = [item[1] for item in prepared]
    texts_by_cluster = [item[2] for item in prepared]
    sizes = [max(1, len(cluster)) for cluster in clusters]
    labels: list[str] = []
    used: set[str] = set()

    def pick(index: int, per_cluster: list[dict[str, int]], require_boundary: bool) -> str:
        scored: list[tuple[float, str]] = []
        for token, count in per_cluster[index].items():
            if _normalize_name(token) in used:
                continue
            inside = count / sizes[index]
            outside_values = [
                other.get(token, 0) / sizes[position]
                for position, other in enumerate(per_cluster)
                if position != index
            ]
            outside = max(outside_values) if outside_values else 0.0
            # 只接受在本簇明显更常见的词项，避免选中各簇共有的泛词
            if inside >= 0.3 and inside - outside >= 0.2:
                if require_boundary and not _is_boundary_aligned_gram(
                    token, texts_by_cluster[index],
                ):
                    continue
                scored.append((inside - outside, token))
        scored.sort(key=lambda item: (-item[0], -len(item[1]), item[1]))
        return scored[0][1] if scored else ""

    for index in range(len(clusters)):
        label = pick(index, per_phrase, require_boundary=False)
        if not label:
            label = pick(index, per_gram, require_boundary=True)
        if not label:
            # 无合格候选：不再编造"（子路线N）"这类内部编号名。它会直接成为
            # 正文小节标题，对读者毫无信息量，还暴露了内部拆分机制。返回空
            # 列表让调用方放弃本次拆分——拆分的目的就是给读者可区分的子主题，
            # 命名不出来时保持父路线原样反而更诚实。
            logger.info(
                "Sub-route cluster %d has no distinctive label; abandoning split",
                index + 1,
            )
            return []
        used.add(_normalize_name(label))
        labels.append(f"{parent_name}：{label}" if parent_name else str(label))
    return labels


def _split_into_clusters(
    paper_ids: list[str],
    token_cache: dict[str, set[str]],
    target_count: int,
) -> list[list[str]]:
    """把论文按词元相似度确定性地划为 ``target_count`` 个规模相近的子簇。

    先取互相最不相似的成员作为种子，其余成员归入最相似种子；不依赖 LLM，
    同一输入恒得同一结果。
    """
    members = list(dict.fromkeys(paper_ids))
    if target_count < 2 or len(members) < target_count * 2:
        return []

    def similarity(left: str, right: str) -> float:
        left_tokens = token_cache.get(left, set())
        right_tokens = token_cache.get(right, set())
        union = left_tokens | right_tokens
        return len(left_tokens & right_tokens) / len(union) if union else 0.0

    seeds = [members[0]]
    while len(seeds) < target_count:
        candidate = min(
            (item for item in members if item not in seeds),
            key=lambda item: max(similarity(item, seed) for seed in seeds),
            default=None,
        )
        if candidate is None:
            break
        seeds.append(candidate)
    clusters: list[list[str]] = [[seed] for seed in seeds]
    capacity = -(-len(members) // len(seeds))  # 向上取整，保持规模相近
    for paper_id in members:
        if paper_id in seeds:
            continue
        order = sorted(
            range(len(seeds)),
            key=lambda index: -similarity(paper_id, seeds[index]),
        )
        target_index = next(
            (index for index in order if len(clusters[index]) < capacity),
            order[0],
        )
        clusters[target_index].append(paper_id)
    return [cluster for cluster in clusters if cluster]


def validate_route_evidence(
    provisional_routes: list[dict[str, Any]],
    paper_cards: list[dict[str, Any]],
    llm=None,
    topic: str = "",
    semantic_frame: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """验证路线并保留完整 feature matrix、Validity 与 Sufficiency 两条独立轴。"""
    from app.core.config import get_settings

    settings = get_settings()
    policy = get_review_threshold_policy()
    routes, anchor_reports = prepare_route_anchors(provisional_routes, llm=llm)
    # 阶段标注必须先于匹配：匹配阶段要读 ``pipeline_stages`` 才能执行
    # 「上游产物不能替代下游证据」的约束。判据由语义帧的证据要求给出，
    # 语义帧缺失时退回卡片已有角色，不引入任何领域词表。
    from app.agent.pipeline_stages import annotate_card_stages, annotate_route_stages

    annotate_card_stages(paper_cards, semantic_frame)
    # 路线阶段同样需要标注：候选路线生成虽要求 LLM 输出 ``route_role``，实测
    # 经常缺失（最新会话 5 条路线全为 None），此时阶段约束会整体休眠。改由
    # 路线文本命中证据要求别名来推导，别名由语义解析动态产出。
    annotate_route_stages(routes, semantic_frame)
    card_map = {
        str(card.get("paper_id") or ""): card
        for card in paper_cards if card.get("paper_id")
    }
    feature_matrix: dict[str, dict[str, dict[str, Any]]] = {}
    per_route_features: dict[str, dict[str, RoutePaperMatchFeatures]] = {}
    for route in routes:
        route_id = str(route.get("route_id") or "")
        per_route_features[route_id] = {
            paper_id: extract_route_paper_features(route, card)
            for paper_id, card in card_map.items()
        }
        feature_matrix[route_id] = {
            paper_id: features.model_dump(mode="json")
            for paper_id, features in per_route_features[route_id].items()
        }

    decisions: list[dict[str, Any]] = []
    validated_routes: list[dict[str, Any]] = []
    route_scores: list[dict[str, Any]] = []
    for route in routes:
        route_id = str(route.get("route_id") or "")
        route_features = per_route_features[route_id]
        core_ids = [pid for pid, features in route_features.items() if features.match_level == "core"]
        supporting_ids = [
            pid for pid, features in route_features.items()
            if features.match_level == "supporting"
        ]
        validity = assess_route_validity(route, anchor_reports.get(route_id))
        sufficiency = assess_evidence_sufficiency(
            core_ids, supporting_ids, card_map,
            minimum_core_evidence=policy.route_min_core_evidence,
        )
        status, action, reason = _decision(
            validity, sufficiency, list(route_features.values())
        )
        decision = {
            "route_id": route_id,
            "route_name": route.get("name"),
            "status": status.value,
            "diagnosis": (
                "STRONG_ROUTE" if status == RouteStatus.KEEP
                else "INSUFFICIENT_EVIDENCE" if action == RouteAction.TARGETED_SEARCH
                else "ROUTE_REVISION_REQUIRED" if action == RouteAction.ROUTE_REVISION
                else "INVALID_ROUTE"
            ),
            "action": action.value,
            "reason": reason,
            "route_validity": validity.model_dump(mode="json"),
            "evidence_sufficiency": sufficiency.model_dump(mode="json"),
            "scores": {
                "paper_count": len(core_ids) + len(supporting_ids),
                "core_paper_count": len(core_ids),
                "supporting_paper_count": len(supporting_ids),
                "route_validity": validity.score,
                "evidence_sufficiency": sufficiency.score,
            },
        }
        decisions.append(decision)
        route_scores.append({
            "route_id": route_id,
            "route_name": route.get("name"),
            "core_paper_ids": core_ids,
            "supporting_paper_ids": supporting_ids,
            **decision["scores"],
        })
        if status != RouteStatus.DROP:
            validated_routes.append({
                **route,
                "status": status.value,
                "paper_ids": _unique([*core_ids, *supporting_ids], limit=10000),
                "core_paper_ids": core_ids,
                "supporting_paper_ids": supporting_ids,
                "route_validity": decision["route_validity"],
                "evidence_sufficiency": decision["evidence_sufficiency"],
            })

    initial_drop_count = sum(item["status"] == RouteStatus.DROP.value for item in decisions)
    initial_keep_count = sum(item["status"] == RouteStatus.KEEP.value for item in decisions)
    initial_drop_ratio = initial_drop_count / len(decisions) if decisions else 0.0
    initial_keep_rate = initial_keep_count / len(decisions) if decisions else 0.0
    guard_triggered = bool(
        len(decisions) >= 3
        and (
            initial_drop_ratio > policy.route_drop_ratio_guard
            or initial_keep_rate < policy.route_min_keep_rate
        )
    )
    if guard_triggered:
        # A systemic all-DROP result is treated as validator uncertainty. Routes
        # with coherent definitions survive as WEAK so Recovery can inspect them.
        for decision, route in zip(decisions, routes):
            if decision["status"] != RouteStatus.DROP.value:
                continue
            validity = decision["route_validity"]
            if (
                float(validity.get("definition_completeness") or 0.0) >= 0.6
                and float(validity.get("internal_consistency") or 0.0) >= 0.5
            ):
                decision.update({
                    "status": RouteStatus.WEAK.value,
                    "action": RouteAction.TARGETED_SEARCH.value,
                    "diagnosis": "VALIDATOR_RECHECK_REQUIRED",
                    "reason": "systemic DROP guard preserved a coherent route for recovery",
                })
                score = next(item for item in route_scores if item["route_id"] == decision["route_id"])
                validated_routes.append({
                    **route,
                    "status": RouteStatus.WEAK.value,
                    "paper_ids": _unique([
                        *score["core_paper_ids"], *score["supporting_paper_ids"]
                    ], limit=10000),
                    "core_paper_ids": score["core_paper_ids"],
                    "supporting_paper_ids": score["supporting_paper_ids"],
                    "route_validity": decision["route_validity"],
                    "evidence_sufficiency": decision["evidence_sufficiency"],
                })

    # 独占 primary 归属决定写作层每个小节的真实体量，规模判据必须基于它
    # 而不是彼此重叠的 core 成员数。
    surviving_route_ids = {
        str(route.get("route_id") or "") for route in validated_routes
    }
    route_order = list(per_route_features.keys())
    primary_owner: dict[str, str] = {}
    for paper_id in card_map:
        core_routes = [
            route_id for route_id, features in per_route_features.items()
            if features[paper_id].match_level == "core"
            and route_id in surviving_route_ids
        ]
        if core_routes:
            primary_owner[paper_id] = _best_fit_route(
                paper_id, core_routes, per_route_features, route_order
            )
    # 规模失衡的路线在此拆分：必须早于 assignment_map 构造，否则被移到
    # 子路线的论文仍会把父路线 id 当作 primary_route，写作层拿不到子路线归属。
    reassigned_routes = _split_oversized_routes(
        validated_routes, decisions, card_map, primary_owner, llm=llm, topic=topic,
        policy=policy,
    )

    assignment_map: dict[str, dict[str, Any]] = {}
    for paper_id in card_map:
        core_routes = [
            route_id for route_id, features in per_route_features.items()
            if features[paper_id].match_level == "core"
        ]
        supporting_routes = [
            route_id for route_id, features in per_route_features.items()
            if features[paper_id].match_level == "supporting"
        ]
        sub_route_id = reassigned_routes.get(paper_id)
        if sub_route_id:
            assignment_map[paper_id] = {
                "type": "single_route",
                "primary_route": sub_route_id,
                "split_from": sub_route_id.rsplit("_S", 1)[0],
            }
        elif len(core_routes) == 1:
            assignment_map[paper_id] = {"type": "single_route", "primary_route": core_routes[0]}
        elif len(core_routes) > 1:
            # primary 必须是契合度最高的一条，而不是路线书写顺序里的第一条：
            # 下游 synthesize_themes 按 primary_theme_id 独占聚合，顺序取值会
            # 让首条路线吞掉全部重叠证据。
            primary = _best_fit_route(
                paper_id, core_routes, per_route_features, route_order
            )
            assignment_map[paper_id] = {
                "type": "cross_route", "primary_route": primary,
                "secondary_routes": [
                    route_id for route_id in core_routes if route_id != primary
                ],
            }
        elif supporting_routes:
            assignment_map[paper_id] = {
                "type": "ambiguous_uncertain",
                "best_route": _best_fit_route(
                    paper_id, supporting_routes, per_route_features, route_order
                ),
            }
        else:
            assignment_map[paper_id] = {"type": "unassigned"}

    understood = sum(
        item["type"] in {"single_route", "cross_route"}
        for item in assignment_map.values()
    )
    final_drop_count = sum(item["status"] == RouteStatus.DROP.value for item in decisions)
    route_count = len(decisions)
    survival_rate = (route_count - final_drop_count) / route_count if route_count else 0.0
    return {
        "validator_version": "route-evidence-v2",
        "threshold_policy": policy.snapshot(),
        "prepared_routes": routes,
        "validated_routes": validated_routes,
        "decisions": decisions,
        "route_scores": route_scores,
        "new_routes": [],
        "assignment_map": assignment_map,
        "feature_matrix": feature_matrix,
        "anchor_guard_reports": anchor_reports,
        "coverage": {
            "total_papers": len(card_map),
            "single_route_confident": sum(item["type"] == "single_route" for item in assignment_map.values()),
            "cross_route_confident": sum(item["type"] == "cross_route" for item in assignment_map.values()),
            "ambiguous_uncertain": sum(item["type"] == "ambiguous_uncertain" for item in assignment_map.values()),
            "unassigned": sum(item["type"] == "unassigned" for item in assignment_map.values()),
            "evidence_understood_rate": understood / len(card_map) if card_map else 0.0,
            "provisional_route_survival_rate": survival_rate,
            "initial_drop_ratio": initial_drop_ratio,
            "provisional_route_keep_rate": initial_keep_rate,
            "route_validator_recheck": guard_triggered,
        },
    }
