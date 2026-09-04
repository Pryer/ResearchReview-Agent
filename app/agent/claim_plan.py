"""Claim-Evidence Planning：在写作前为每条路线生成论证计划，绑定证据和语言强度。

这是 Layer 3（Evidence-grounded Writing）的核心组件。
从 Final Routes 出发，生成每条路线的 claim plan，约束 writer 只能写
有证据支撑的主张，并按证据数量决定语言强度。

目标：将主张—证据支持率从 ~54% 提升到 70%+。
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any

from app.core.config import get_review_threshold_policy
from app.core.logger import get_logger

logger = get_logger(__name__)

# 证据数量 → 语言强度映射（established 需要额外门禁，不只是 count）
_SUPPORT_LEVEL_MAP = {
    "single":      {"min": 1, "max": 1, "language": "一项研究/有研究尝试", "verbs": ["报告", "提出", "尝试"]},
    "moderate":    {"min": 2, "max": 3, "language": "部分研究/若干工作", "verbs": ["采用", "探索", "显示"]},
    "strong":      {"min": 4, "max": 999, "language": "多项研究/形成了较为明确的", "verbs": ["表明", "验证", "确认"]},
}

# established 是 strong 的升级版，需要额外满足：有综述支撑或多团队独立验证
_ESTABLISHED_GATE = {
    "min_evidence": 7,
    "requires_survey_or_independent_teams": True,
    "min_independent_teams": 3,
    "language": "已成为重要方向/广泛研究",
    "verbs": ["证实", "确立", "成为共识"],
    "warning": "established 级别需要综述论文或多团队独立验证，不能仅凭数量自动升级",
}

# 主张类型定义 + 每种类型的证据规则
_CLAIM_TYPES = {
    "problem": {
        "label": "研究问题/现实需求",
        "min_evidence": 1,
        "allow_single_source": True,
    },
    "method_progression": {
        "label": "方法演进/技术路线",
        "min_evidence": 2,
        "allow_single_source": False,
    },
    "finding": {
        "label": "研究发现/实验结果",
        "min_evidence": 1,
        "allow_single_source": True,  # 单个实验结果可以报告
        "requires_direct_report": True,  # 必须原文直接报告
    },
    "quantitative_result": {
        "label": "量化结果/性能数据",
        "min_evidence": 1,
        "allow_single_source": True,
        "requires_direct_report": True,
        "forbid_extrapolation": True,  # 不能跨论文推断数值
    },
    "limitation": {
        "label": "研究局限",
        "min_evidence": 1,
        "allow_single_source": True,
        "requires_author_stated": True,  # 必须作者明确报告
    },
    "trend": {
        "label": "领域趋势/发展方向",
        "min_evidence": 4,
        "allow_single_source": False,
        "requires_multiple_teams": True,
    },
    "comparison": {
        "label": "路线间比较",
        "min_evidence": 2,
        "allow_single_source": False,
    },
    "research_gap": {
        "label": "研究空白",
        "min_evidence": 3,
        "allow_single_source": False,
        "requires_author_stated": True,
        "forbid_absence_inference": True,  # 不能因为没搜到就说"尚缺乏"
        "allowed_phrasing": "在当前纳入证据中，…报道较少",  # 只能这样写
    },
    "background_fact": {
        "label": "背景事实",
        "min_evidence": 1,
        "allow_single_source": True,
    },
    "synthesis": {
        "label": "综合判断/跨路线总结",
        "min_evidence": 3,
        "allow_single_source": False,
        "requires_multiple_routes": True,
    },
}

# gap 类主张的措辞约束
_GAP_PHRASING_RULES = {
    "evidence_of_absence": "现有研究明确表明…不足/缺失",
    "absence_of_evidence": "在当前纳入证据中，…报道较少/尚不充分",
    "forbidden": [
        "该领域尚缺乏",
        "尚无研究涉及",
        "学界尚未关注",
        "目前没有研究",
        "至今未被研究",
        "所有研究均未",
    ],
}

# 弱访问级别：仅元数据/标题/摘要可见的论文。其卡片字段只能支撑
# “泛泛提及”，不能提取具体实验事实、数值或结论。
_WEAK_ACCESS_LEVELS = {"metadata_only", "title_and_keywords", "abstract"}


def _paper_id_from_evidence(evidence_id: str) -> str:
    """从 evidence_id 还原 paper_id。

    evidence_id 格式为 ``{paper_id}:eNNN``，而 paper_id 本身含冒号
    （如 ``s2:hash`` / ``doi:10.x/xx``），因此必须从最后一个冒号切分；
    ``split(":")[0]`` 只能拿到数据源前缀，会导致卡片查询全部落空。
    """
    text = str(evidence_id or "")
    if ":" not in text:
        return ""
    return text.rsplit(":", 1)[0]


def _card_access_level(card: dict[str, Any]) -> str:
    """归一化卡片的证据访问级别。"""
    access = str(
        (card.get("evidence_state") or {}).get("access_level")
        or {"metadata": "metadata_only", "abstract": "abstract", "full_text": "partial_full_text"}.get(
            str(card.get("evidence_source") or "metadata"), "metadata_only"
        )
    )
    if hasattr(card.get("evidence_state", {}).get("access_level"), "value"):
        access = str(card["evidence_state"]["access_level"].value)
    return access


def _is_weak_access_card(card: dict[str, Any]) -> bool:
    return _card_access_level(card) in _WEAK_ACCESS_LEVELS


def _can_contribute_factual_claims(card: dict[str, Any]) -> bool:
    """低置信放行（可能偏题）的论文只有在 LLM 语义筛选确认为直接/邻近
    相关时才能贡献事实主张；否则只能作为检索诊断，不能进入取证。"""
    if not card.get("anchor_low_confidence"):
        return True
    relation = str(card.get("relation_type") or "")
    return relation in {"direct", "near"}


def _apply_access_limit(claim_entry: dict[str, Any], evidence_ids: list[str], card_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """全部证据均来自摘要级及以下论文的主张，限制为泛泛提及强度。"""
    paper_ids = list(dict.fromkeys(
        _paper_id_from_evidence(eid) for eid in evidence_ids if _paper_id_from_evidence(eid)
    ))
    if not paper_ids:
        return claim_entry
    if all(_is_weak_access_card(card_map.get(pid) or {}) for pid in paper_ids):
        claim_entry["evidence_access_limit"] = "abstract_only"
        claim_entry["allowed_language"] = "仅可泛泛提及（另有研究涉及…/相关工作还包括…）"
        claim_entry["allowed_verbs"] = ["提及", "涉及", "关注"]
    return claim_entry


def build_claim_plans(
    validated_routes: list[dict[str, Any]],
    paper_cards: list[dict[str, Any]],
    llm=None,
    topic: str = "",
) -> list[dict[str, Any]]:
    """为每条已验证路线生成论证计划。

    Returns:
        [{route_id, route_name, claims: [{claim_id, claim_text, claim_type,
         evidence_ids, evidence_count, support_level, allowed_language}]}]
    """
    if not validated_routes or not paper_cards:
        return []

    card_map = {
        str(card.get("paper_id") or ""): card
        for card in paper_cards
        if card.get("paper_id")
    }

    plans: list[dict[str, Any]] = []
    for route in validated_routes:
        route_paper_ids = [
            str(pid) for pid in (route.get("core_paper_ids") or route.get("paper_ids") or [])
            if str(pid) in card_map
            and _can_contribute_factual_claims(card_map.get(str(pid)) or {})
        ]
        if not route_paper_ids:
            continue

        # 提取该路线所有论文的 claims
        raw_claims = _extract_route_claims(route_paper_ids, card_map)

        # 按主张内容去重 + 合并证据（字面层，只能消除逐字重复）
        merged_claims = _merge_similar_claims(raw_claims, route_paper_ids)

        # 语义层归并：把讲同一件事的主张聚成论点，让 support_level 反映真实的
        # 跨文献证据强度。字面层几乎从不触发合并（主张是各论文摘要整句），
        # 没有这一步 support_level 会恒为 single。
        merged_claims = cluster_claims_into_theses(
            merged_claims,
            route_name=str(route.get("name") or ""),
            topic=topic,
            card_map=card_map,
            llm=llm,
        )

        # 计算支持级别 + 类型特定规则
        claim_plan_claims = []
        for claim_text, evidence_data in merged_claims.items():
            evidence_ids = evidence_data["evidence_ids"]
            paper_ids_for_claim = list(evidence_data.get("paper_ids", []))
            # 支持级别按独立论文数判定，而非 evidence_id 数：同一篇论文的多个
            # 字段各有 evidence_id，用后者会把单篇证据误判成多篇支撑。
            count = max(len(paper_ids_for_claim), 1) if paper_ids_for_claim else len(evidence_ids)
            claim_type = evidence_data.get("claim_type", "finding")

            # 类型特定检查
            type_config = _CLAIM_TYPES.get(claim_type, {})
            if count < type_config.get("min_evidence", 1):
                continue  # 不满足该类型最低证据要求，跳过

            # 确定支持级别（含独立性检查）
            support_level = _determine_support_level(
                count, claim_type,
                paper_ids=paper_ids_for_claim,
                card_map=card_map,
            )
            allowed = _SUPPORT_LEVEL_MAP.get(support_level, _SUPPORT_LEVEL_MAP["single"])
            if support_level == "established":
                allowed = _ESTABLISHED_GATE

            claim_entry = {
                "claim_id": _claim_id(route.get("route_id", ""), claim_text),
                "claim_text": claim_text,
                "claim_type": claim_type,
                "evidence_ids": evidence_ids,
                "evidence_count": count,
                "independent_source_count": _count_independent_sources(paper_ids_for_claim, card_map),
                "support_level": support_level,
                "allowed_language": allowed["language"],
                "allowed_verbs": allowed["verbs"],
            }
            # 摘要级证据只能支撑泛泛提及，不能提取具体事实主张
            claim_entry = _apply_access_limit(claim_entry, evidence_ids, card_map)

            # research_gap 额外分类
            if claim_type == "research_gap":
                gap_info = _classify_gap_claim(claim_text, evidence_ids, card_map)
                claim_entry["gap_classification"] = gap_info["classification"]
                claim_entry["allowed_language"] = gap_info["allowed_phrasing"]
                claim_entry["forbidden_phrases"] = gap_info["forbidden_phrases"]

            # limitation 必须作者明确报告
            if type_config.get("requires_author_stated"):
                has_explicit = any(
                    _is_author_stated(eid, card_map)
                    for eid in evidence_ids
                )
                if not has_explicit:
                    continue  # 没有作者明确报告 → 跳过

            claim_plan_claims.append(claim_entry)

        # 按 support_level 排序（强证据在前）
        level_order = {"established": 0, "strong": 1, "moderate": 2, "single": 3}
        claim_plan_claims.sort(key=lambda c: level_order.get(c["support_level"], 99))

        # 可选：LLM 优化主张表述
        if llm and len(claim_plan_claims) >= 3:
            claim_plan_claims = _llm_refine_claims(
                claim_plan_claims,
                route.get("name", ""),
                route.get("research_question", ""),
                llm,
            )

        plans.append({
            "route_id": route.get("route_id", ""),
            "route_name": route.get("name", ""),
            "claims": claim_plan_claims,
            "total_evidence_papers": len(route_paper_ids),
            "total_claims": len(claim_plan_claims),
            "single_evidence_claims": sum(1 for c in claim_plan_claims if c["support_level"] == "single"),
            "strong_plus_claims": sum(1 for c in claim_plan_claims if c["support_level"] in ("strong", "established")),
        })

    return plans


def apply_claim_budget(
    claim_plans: list[dict[str, Any]],
    required_reference_count: int,
    *,
    minimum_per_route: int = 6,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """限制事实主张总量，同时优先保留强证据并维持路线覆盖。"""
    plans = [dict(plan) for plan in claim_plans]
    total_before = sum(len(plan.get("claims") or []) for plan in plans)
    budget = max(
        max(0, int(required_reference_count)),
        len(plans) * max(1, int(minimum_per_route)),
    )
    if total_before <= budget:
        return plans, {
            "total_before": total_before,
            "total_after": total_before,
            "budget": budget,
            "dropped": 0,
        }

    selected: dict[int, set[int]] = {index: set() for index in range(len(plans))}
    # 每条路线先保留一个最强主张，避免全局排序让小路线完全消失。
    for plan_index, plan in enumerate(plans):
        if plan.get("claims") and sum(len(items) for items in selected.values()) < budget:
            selected[plan_index].add(0)

    level_order = {"established": 0, "strong": 1, "moderate": 2, "single": 3}
    candidates = sorted(
        (
            level_order.get(str(claim.get("support_level") or "single"), 9),
            claim_index,
            plan_index,
        )
        for plan_index, plan in enumerate(plans)
        for claim_index, claim in enumerate(plan.get("claims") or [])
        if claim_index not in selected[plan_index]
    )
    remaining = budget - sum(len(items) for items in selected.values())
    for _, claim_index, plan_index in candidates[:remaining]:
        selected[plan_index].add(claim_index)

    for plan_index, plan in enumerate(plans):
        claims = [
            claim for claim_index, claim in enumerate(plan.get("claims") or [])
            if claim_index in selected[plan_index]
        ]
        plan["claims"] = claims
        plan["total_claims"] = len(claims)
        plan["single_evidence_claims"] = sum(
            1 for claim in claims if claim.get("support_level") == "single"
        )
        plan["strong_plus_claims"] = sum(
            1 for claim in claims
            if claim.get("support_level") in {"strong", "established"}
        )
    total_after = sum(len(plan.get("claims") or []) for plan in plans)
    return plans, {
        "total_before": total_before,
        "total_after": total_after,
        "budget": budget,
        "dropped": total_before - total_after,
    }


def build_background_claim_plan(
    background_outline: dict[str, Any],
    paper_cards: list[dict[str, Any]],
    llm=None,
) -> list[dict[str, Any]]:
    """为研究背景的每个段落目标生成 Claim Plan。

    背景不按路线组织，而是按 paragraph_goal 组织。
    """
    goals = background_outline.get("paragraph_goals") or []
    if not goals or not paper_cards:
        return []

    card_map = {
        str(card.get("paper_id") or ""): card
        for card in paper_cards
        if card.get("paper_id")
    }
    all_paper_ids = [
        pid for pid, card in card_map.items()
        if _can_contribute_factual_claims(card)
    ]

    plans: list[dict[str, Any]] = []
    for goal in goals:
        goal_id = str(goal.get("id") or f"bg_{len(plans)}")
        goal_label = str(goal.get("label") or "")
        goal_text = str(goal.get("goal") or "")

        # 从所有论文中提取与该目标相关的 claims
        # 使用 goal_text 的关键词做简单筛选
        goal_tokens = set(re.findall(r"[一-鿿]{2,}", goal_text.lower()))

        relevant_paper_ids: list[str] = []
        for pid, card in card_map.items():
            if not _can_contribute_factual_claims(card):
                continue
            card_text = " ".join(
                str(card.get(f) or "")
                for f in ("title", "research_problem", "abstract")
            ).lower()
            if goal_tokens & set(re.findall(r"[一-鿿]{2,}", card_text)):
                relevant_paper_ids.append(pid)
            elif not goal_tokens:
                relevant_paper_ids.append(pid)

        if not relevant_paper_ids:
            # 没有与段落目标直接匹配的证据时，不得用背景论文池凑数。
            relevant_paper_ids = []

        raw_claims = _extract_route_claims(relevant_paper_ids, card_map)
        merged_claims = _merge_similar_claims(raw_claims, relevant_paper_ids)

        bg_claims = []
        for claim_text, evidence_data in merged_claims.items():
            evidence_ids = evidence_data["evidence_ids"]
            paper_ids_for_claim = list(evidence_data.get("paper_ids", []))
            count = len(evidence_ids)
            claim_type = evidence_data.get("claim_type", "background_fact")

            if count < 1:
                continue

            support_level = _determine_support_level(
                count, "background_fact",
                paper_ids=paper_ids_for_claim,
                card_map=card_map,
            )
            allowed = _SUPPORT_LEVEL_MAP.get(support_level, _SUPPORT_LEVEL_MAP["single"])

            bg_claims.append(_apply_access_limit({
                "claim_id": f"bg_{goal_id}:{_normalize(claim_text)[:30]}",
                "claim_text": claim_text,
                "claim_type": "background_fact",
                "evidence_ids": evidence_ids,
                "evidence_count": count,
                "support_level": support_level,
                "allowed_language": allowed["language"],
                "allowed_verbs": allowed["verbs"],
            }, evidence_ids, card_map))

        # 最多 5 条主张/段落
        level_order = {"strong": 0, "moderate": 1, "single": 2}
        bg_claims.sort(key=lambda c: level_order.get(c["support_level"], 99))
        bg_claims = bg_claims[:5]

        plans.append({
            "route_id": f"background_{goal_id}",
            "route_name": goal_label or goal_text[:30],
            "claims": bg_claims,
            "total_evidence_papers": len(relevant_paper_ids),
            "total_claims": len(bg_claims),
            "strong_plus_claims": sum(1 for c in bg_claims if c["support_level"] == "strong"),
            "single_evidence_claims": sum(1 for c in bg_claims if c["support_level"] == "single"),
            "paragraph_goal": goal_text,
        })

    return plans


def enforce_claim_evidence_gate(
    claim_plans: list[dict[str, Any]],
    paper_cards: list[dict[str, Any]],
    llm=None,
    *,
    max_single_source_claims: int = 2,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """在写作前核验绑定、单篇证据预算和主张—证据真实蕴含关系。

    Claim Planner 原则上只从证据生成主张；本门禁是独立的契约防线，用于处理
    LLM 表述修订、旧状态恢复或上游卡片变更造成的失效证据 ID。零证据主张被
    删除，证据数量下降时只降低语言强度，不会为了保留主张而虚构证据。启用
    LLM 时，所有保留主张还必须通过声明级蕴含校验；未返回或未蕴含均不进入 Writer。
    """
    from app.schemas.recovery_schema import (
        ClaimEvidenceGap,
        ClaimEvidenceGateReport,
        ClaimGapType,
    )

    card_map = {
        str(card.get("paper_id") or ""): card
        for card in paper_cards
        if card.get("paper_id")
    }
    evidence_to_paper: dict[str, str] = {}
    for paper_id, card in card_map.items():
        for field, field_claims in (card.get("field_claims") or {}).items():
            for claim in field_claims or []:
                if not isinstance(claim, dict):
                    continue
                evidence_id = str(claim.get("evidence_id") or f"{paper_id}:{field}")
                evidence_to_paper[evidence_id] = paper_id
        for span in card.get("evidence_spans") or []:
            if isinstance(span, dict) and span.get("evidence_id"):
                evidence_to_paper[str(span["evidence_id"])] = paper_id

    repaired_plans: list[dict[str, Any]] = []
    gaps: list[ClaimEvidenceGap] = []
    searchable_core_gaps: list[ClaimEvidenceGap] = []
    total_claims = 0
    weakened = 0
    dropped = 0
    single_source_claims_dropped = 0
    entailment_checked_claims = 0
    entailment_failed_claims = 0
    level_order = {"single": 0, "moderate": 1, "strong": 2, "established": 3}
    core_types = {"problem", "method_progression", "background_fact"}
    optional_types = {"trend", "comparison", "research_gap", "synthesis"}

    for plan in claim_plans:
        repaired = dict(plan)
        retained_claims: list[dict[str, Any]] = []
        source_claim_counts: dict[str, int] = defaultdict(int)
        for raw_claim in plan.get("claims") or []:
            total_claims += 1
            claim = dict(raw_claim)
            claim_id = str(claim.get("claim_id") or "")
            claim_type = str(claim.get("claim_type") or "finding")
            valid_evidence_ids = list(dict.fromkeys(
                str(evidence_id)
                for evidence_id in claim.get("evidence_ids") or []
                if str(evidence_id) in evidence_to_paper
            ))
            paper_ids = list(dict.fromkeys(
                evidence_to_paper[evidence_id] for evidence_id in valid_evidence_ids
            ))

            # 单篇证据不能承载无限多个独立事实主张；按证据所属论文限额，
            # 超出的主张降级为可检索缺口，避免一篇摘要被扩写成整条路线。
            if len(paper_ids) == 1:
                source_id = paper_ids[0]
                if source_claim_counts[source_id] >= max_single_source_claims:
                    gaps.append(ClaimEvidenceGap(
                        claim_id=claim_id,
                        route_id=str(plan.get("route_id") or ""),
                        gap_type=ClaimGapType.OPTIONAL_UNSUPPORTED,
                        action="DROP",
                        reason=f"单篇证据 {source_id} 最多承载 {max_single_source_claims} 条独立主张。",
                    ))
                    dropped += 1
                    single_source_claims_dropped += 1
                    continue

            type_config = _CLAIM_TYPES.get(claim_type, _CLAIM_TYPES["finding"])
            minimum = int(type_config.get("min_evidence", 1))
            if len(valid_evidence_ids) < minimum:
                gap_type = (
                    ClaimGapType.OPTIONAL_UNSUPPORTED
                    if claim_type in optional_types else ClaimGapType.MISSING_SUPPORT
                )
                gap = ClaimEvidenceGap(
                    claim_id=claim_id,
                    route_id=str(plan.get("route_id") or ""),
                    gap_type=gap_type,
                    action="DROP",
                    reason=(
                        f"有效证据 {len(valid_evidence_ids)} 条，低于 {claim_type} "
                        f"主张所需的最低证据数 {minimum}。"
                    ),
                )
                gaps.append(gap)
                if claim_type in core_types:
                    searchable_core_gaps.append(gap.model_copy(update={"action": "SEARCH_OR_DROP"}))
                dropped += 1
                continue

            expected_level = _determine_support_level(
                len(valid_evidence_ids),
                claim_type,
                paper_ids=paper_ids,
                card_map=card_map,
            )
            current_level = str(claim.get("support_level") or "single")
            if level_order.get(current_level, 0) > level_order.get(expected_level, 0):
                allowed = (
                    _ESTABLISHED_GATE
                    if expected_level == "established"
                    else _SUPPORT_LEVEL_MAP.get(expected_level, _SUPPORT_LEVEL_MAP["single"])
                )
                claim["support_level"] = expected_level
                claim["allowed_language"] = allowed["language"]
                claim["allowed_verbs"] = allowed["verbs"]
                # 摘要级证据的泛泛提及限制不随证据数量变化而解除
                claim = _apply_access_limit(claim, valid_evidence_ids, card_map)
                gaps.append(ClaimEvidenceGap(
                    claim_id=claim_id,
                    route_id=str(plan.get("route_id") or ""),
                    gap_type=ClaimGapType.EXCESSIVE_STRENGTH,
                    action="WEAKEN",
                    reason=f"证据变化后语言强度由 {current_level} 降为 {expected_level}。",
                ))
                weakened += 1

            claim["evidence_ids"] = valid_evidence_ids
            claim["evidence_count"] = len(valid_evidence_ids)
            claim["independent_source_count"] = _count_independent_sources(
                paper_ids, card_map
            )
            for paper_id in paper_ids:
                source_claim_counts[paper_id] += 1
            retained_claims.append(claim)

        repaired["claims"] = retained_claims
        repaired["total_claims"] = len(retained_claims)
        repaired["single_evidence_claims"] = sum(
            1 for claim in retained_claims if claim.get("support_level") == "single"
        )
        repaired["strong_plus_claims"] = sum(
            1 for claim in retained_claims
            if claim.get("support_level") in {"strong", "established"}
        )
        repaired_plans.append(repaired)

    retained = sum(len(plan.get("claims") or []) for plan in repaired_plans)
    if llm is not None and retained:
        repaired_plans, checked, failed = _filter_non_entailed_claims(
            repaired_plans, card_map, llm
        )
        entailment_checked_claims = checked
        entailment_failed_claims = failed
        dropped += failed
        retained = sum(len(plan.get("claims") or []) for plan in repaired_plans)
    report = ClaimEvidenceGateReport(
        passed=not searchable_core_gaps and entailment_failed_claims == 0,
        total_claims=total_claims,
        retained_claims=retained,
        weakened_claims=weakened,
        dropped_claims=dropped,
        single_source_claim_limit=max_single_source_claims,
        single_source_claims_dropped=single_source_claims_dropped,
        entailment_checked_claims=entailment_checked_claims,
        entailment_failed_claims=entailment_failed_claims,
        searchable_core_gaps=searchable_core_gaps,
        gaps=gaps,
    )
    return repaired_plans, report.model_dump(mode="json")


def _filter_non_entailed_claims(
    plans: list[dict[str, Any]],
    card_map: dict[str, dict[str, Any]],
    llm,
) -> tuple[list[dict[str, Any]], int, int]:
    """写作前按证据片段执行真实蕴含校验，拒绝仅词面相似的主张。"""
    from app.tools.verify_claims import _card_evidence, _llm_entailment_results
    from app.schemas.verification_schema import ClaimEvidenceResult

    candidates: list[ClaimEvidenceResult] = []
    for plan in plans:
        for claim in plan.get("claims") or []:
            snippets = []
            for evidence_id in claim.get("evidence_ids") or []:
                paper_id = _paper_id_from_evidence(str(evidence_id))
                card = card_map.get(paper_id) or {}
                matched = [
                    span for span in _card_evidence(card)
                    if str(span.get("evidence_id") or "") == str(evidence_id)
                ]
                if not matched:
                    matched = _card_evidence(card)
                snippets.extend(
                    {"text": span.get("text"), "evidence_id": span.get("evidence_id")}
                    for span in matched[:2]
                    if span.get("text")
                )
            if not snippets:
                continue
            result = ClaimEvidenceResult(
                claim_id=str(claim.get("claim_id") or ""),
                sentence=str(claim.get("claim_text") or ""),
                citations=[_paper_id_from_evidence(str(eid)) for eid in claim.get("evidence_ids") or []],
                factual=True,
                support_status="partially_supported",
                evidence_snippets=snippets,
            )
            candidates.append(result)

    results = _llm_entailment_results(candidates, llm)
    failed_ids: set[str] = set()
    for result in candidates:
        verdict = results.get(result.claim_id)
        if not verdict or str(verdict.get("label") or "").lower() != "entailed":
            failed_ids.add(result.claim_id)

    filtered: list[dict[str, Any]] = []
    for plan in plans:
        updated = dict(plan)
        updated["claims"] = [
            claim for claim in plan.get("claims") or []
            if str(claim.get("claim_id") or "") not in failed_ids
        ]
        updated["total_claims"] = len(updated["claims"])
        updated["single_evidence_claims"] = sum(
            claim.get("support_level") == "single" for claim in updated["claims"]
        )
        updated["strong_plus_claims"] = sum(
            claim.get("support_level") in {"strong", "established"}
            for claim in updated["claims"]
        )
        filtered.append(updated)
    return filtered, len(candidates), len(failed_ids)


def claim_plan_to_writer_constraints(
    plans: list[dict[str, Any]],
) -> str:
    """将 claim plan 渲染为 writer 可用的约束文本。

    包含：每条路线的可写主张列表 + 每条主张的证据 ID + 语言强度限制。
    """
    if not plans:
        return ""

    sections = []
    for plan in plans:
        lines = [f"## {plan['route_name']}"]
        lines.append(f"（共 {plan['total_claims']} 条主张，{plan['total_evidence_papers']} 篇支撑论文）")
        lines.append("")

        for i, claim in enumerate(plan["claims"], 1):
            entry = (
                f"{i}. [{claim['support_level'].upper()}] {claim['claim_text']}\n"
                f"   证据：{', '.join(claim['evidence_ids'][:5])}"
                f"{' ...' if len(claim['evidence_ids']) > 5 else ''}\n"
                f"   允许的语言强度：{claim['allowed_language']}"
            )
            if claim.get("evidence_access_limit") == "abstract_only":
                entry += "（摘要级证据，仅可泛泛提及）"
            lines.append(entry)

        lines.append("")
        sections.append("\n".join(lines))

    return "\n".join(sections)


def validate_claim_support(
    review_text: str,
    claim_plans: list[dict[str, Any]],
) -> dict[str, Any]:
    """验证正文中的每个事实性主张是否得到 claim plan 的授权。

    逐句输出：sentence / mapped_claim_id / status / similarity / reason。

    Returns:
        除聚合指标外，还包含 per_sentence 数组供人工标注。
    """
    policy = get_review_threshold_policy()
    similarity_threshold = policy.claim_support_similarity
    min_claim_text_length = policy.claim_min_text_length
    # P0-1 fix: 趋势/演进/综合判断类语言标记，即使无引用也是事实性主张
    TREND_PATTERNS = [
        "从.*转向", "从.*走向", "从.*向.*扩展", "从.*拓展", "从.*演进",
        "形成了", "呈现出", "已成为", "逐步成为", "逐渐成为",
        "不再.*而是", "日益", "越来越", "持续.*趋势",
        "标志着", "意味着", "表明", "说明",
        "演进态势", "演进趋势", "发展脉络",
        "拓展至", "延伸到", "扩展到",
    ]
    # 解释性扩展语言：writer 在报告事实后追加自己的interpretation
    INTERPRETIVE_EXPANSION = [
        "验证了", "验证其", "被验证", "这说明", "这意味着", "证实了",
        "达到了.*水平", "具备了.*能力", "证明了",
        "能够达到", "可以被视为", "可以被认为",
        "为.*提供支撑", "为.*提供依据", "有助于", "能够促进",
    ]
    # 纯结构文本（标题、编号行）
    STRUCTURAL_PATTERNS = [
        r"^（[一二三四五六七八九十]+）", r"^[\(（][一二三四五六七八九十]+[\)）]",
        r"^[0-9]+\.[0-9]*\s", r"^第[一二三四五六七八九十]+[章节部分]",
    ]

    # 构建所有已授权主张的索引
    all_claims: list[dict[str, Any]] = []
    for plan in claim_plans:
        for claim in plan.get("claims") or []:
            all_claims.append({
                "claim_id": claim.get("claim_id", ""),
                "claim_text": claim.get("claim_text", ""),
                "claim_type": claim.get("claim_type", ""),
                "support_level": claim.get("support_level", ""),
                "allowed_language": claim.get("allowed_language", ""),
                "plan_route": plan.get("route_name", plan.get("route_id", "")),
            })

    # 预计算 normalized token set（跳过原文过短的噪声 claim）
    claim_tokens: dict[str, set[str]] = {}
    claim_full_text: dict[str, str] = {}
    for c in all_claims:
        if len(c["claim_text"]) >= min_claim_text_length:
            claim_tokens[c["claim_id"]] = set(_normalize(c["claim_text"]))
            claim_full_text[c["claim_id"]] = c["claim_text"]

    # 分句
    sentences = re.split(r"[。；\n]", review_text)
    candidates = [
        (i, s.strip()) for i, s in enumerate(sentences)
        if len(s.strip()) > 10
    ]

    per_sentence: list[dict[str, Any]] = []
    supported_count = 0
    unsupported_count = 0
    partial_count = 0
    non_factual_count = 0

    for idx, sentence in candidates:
        norm = _normalize(sentence)
        norm_set = set(norm)
        has_citations = bool(re.search(r"\[\d+\]", sentence))

        # 纯结构文本检测
        is_structural = any(re.match(p, sentence) for p in STRUCTURAL_PATTERNS)
        if is_structural and not has_citations:
            per_sentence.append(_make_entry(idx, sentence, False, "NON_FACTUAL",
                                            None, 0.0, "structural heading"))
            non_factual_count += 1
            continue

        # 找最佳匹配 claim
        best_claim_id = None
        best_similarity = 0.0
        for claim_id, ct in claim_tokens.items():
            sim = len(norm_set & ct) / max(1, len(ct))
            if sim > best_similarity:
                best_similarity = sim
                best_claim_id = claim_id

        # P0-1: 事实性判断基于内容而非引用
        has_trend_language = any(
            re.search(p, sentence) for p in TREND_PATTERNS
        )
        has_interpretive_expansion = any(
            re.search(p, sentence) for p in INTERPRETIVE_EXPANSION
        )
        is_factual_content = (
            has_citations or has_trend_language or len(norm) > 40
        )

        if not is_factual_content:
            per_sentence.append(_make_entry(idx, sentence, False, "NON_FACTUAL",
                                            None, best_similarity, "short, no factual markers"))
            non_factual_count += 1
            continue

        # P0-2: 有匹配时检查是否 overclaim
        if best_similarity >= similarity_threshold and best_claim_id:
            claim_text = claim_full_text.get(best_claim_id, "")
            claim_norm_len = len(_normalize(claim_text))
            overclaim_ratio = (len(norm) - claim_norm_len) / max(1, claim_norm_len)

            # 趋势/演进类主张的授权检查：
            # 1) 句子含趋势语言但 claim 类型非 trend/method_progression → 趋势判断未授权
            # 2) 句子含趋势语言但无引用 → writer 在无直接引用下做趋势判断
            claim_trend_authorized = any(
                c["claim_id"] == best_claim_id and c["claim_type"] in ("trend", "method_progression")
                for c in all_claims
            )
            has_overclaim_signal = (
                has_trend_language or has_interpretive_expansion or not has_citations
            )
            trend_overclaim = (
                has_trend_language
                and (not claim_trend_authorized or not has_citations)
            )
            # 解释性扩展：writer 在 claim 事实基础上加了 "验证了/说明了" 等解释
            interpretive_overclaim = (
                has_interpretive_expansion
                and overclaim_ratio > 0.25
            )

            if overclaim_ratio > 0.40 and has_overclaim_signal:
                status = "PARTIALLY_AUTHORIZED"
                reason = f"matched {best_claim_id} (sim={best_similarity:.2f}) but overclaim ratio={overclaim_ratio:.1f}"
                partial_count += 1
            elif trend_overclaim:
                status = "PARTIALLY_AUTHORIZED"
                reason = f"matched {best_claim_id} (sim={best_similarity:.2f}) but trend/progression language not authorized by claim type"
                partial_count += 1
            elif interpretive_overclaim:
                status = "PARTIALLY_AUTHORIZED"
                reason = f"matched {best_claim_id} (sim={best_similarity:.2f}) but interpretive expansion (ratio={overclaim_ratio:.1f})"
                partial_count += 1
            else:
                status = "SUPPORTED"
                reason = f"matched {best_claim_id} (sim={best_similarity:.2f})"
                supported_count += 1
        elif best_similarity >= similarity_threshold and best_claim_id:
            status = "SUPPORTED"
            reason = f"matched {best_claim_id} (sim={best_similarity:.2f})"
            supported_count += 1
        elif has_citations:
            status = "UNSUPPORTED"
            reason = f"no claim match (best={best_claim_id or 'none'} sim={best_similarity:.2f})"
            unsupported_count += 1
        else:
            # 无引用 + 趋势语言 → 标记为 UNSUPPORTED_TREND
            if has_trend_language:
                status = "UNSUPPORTED_TREND"
                reason = f"trend claim without citation or claim match (best={best_claim_id or 'none'} sim={best_similarity:.2f})"
            else:
                status = "UNSUPPORTED"
                reason = f"factual content without citation or claim match"
            unsupported_count += 1

        per_sentence.append(_make_entry(
            idx, sentence, has_citations, status,
            best_claim_id if status in ("SUPPORTED", "PARTIALLY_AUTHORIZED") else None,
            best_similarity, reason,
        ))

    total_factual = supported_count + unsupported_count + partial_count
    support_rate = supported_count / total_factual if total_factual > 0 else 1.0

    return {
        "total_factual_sentences": total_factual,
        "total_sentences_checked": len(per_sentence),
        "supported_sentences": supported_count,
        "unsupported_sentences": unsupported_count,
        "partial_authorized_sentences": partial_count,
        "non_factual_sentences": non_factual_count,
        "support_rate": support_rate,
        "unsupported_samples": [
            s["sentence"] for s in per_sentence
            if s["status"] in ("UNSUPPORTED", "UNSUPPORTED_TREND")
        ][:10],
        "overclaimed_samples": [
            s["sentence"] for s in per_sentence
            if s["status"] == "PARTIALLY_AUTHORIZED"
        ][:10],
        "per_sentence": per_sentence,
    }


def _make_entry(
    idx: int, sentence: str, has_citations: bool,
    status: str, mapped_claim_id: str | None,
    similarity: float, reason: str,
) -> dict[str, Any]:
    return {
        "sentence_index": idx,
        "sentence": sentence[:200],
        "has_citations": has_citations,
        "mapped_claim_id": mapped_claim_id,
        "status": status,
        "similarity": round(similarity, 4),
        "reason": reason,
    }


# ============================================================
# 内部
# ============================================================

def _extract_route_claims(
    paper_ids: list[str],
    card_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """提取路线内所有论文的声明级证据。"""
    claims: list[dict[str, Any]] = []
    for paper_id in paper_ids:
        card = card_map.get(paper_id, {})
        for field, field_claims in (card.get("field_claims") or {}).items():
            for claim in field_claims or []:
                if not isinstance(claim, dict):
                    continue
                claim_text = str(claim.get("claim") or claim.get("text") or "").strip()
                if not claim_text:
                    continue
                claims.append({
                    "paper_id": paper_id,
                    "claim_text": claim_text,
                    "claim_type": _field_to_claim_type(field),
                    "evidence_id": str(claim.get("evidence_id") or f"{paper_id}:{field}"),
                    "explicitly_reported": bool(claim.get("explicitly_reported")),
                })
    return claims


def _field_to_claim_type(field: str) -> str:
    mapping = {
        "research_problem": "problem",
        "method": "method_progression",
        "results": "finding",
        "limitations": "limitation",
    }
    return mapping.get(field, "finding")


def _merge_similar_claims(
    claims: list[dict[str, Any]],
    paper_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """合并在词面上相似的主张（normalized 前 80 字符相同视为同一主张）。

    注意这是**字面层**合并，只能消除近乎逐字重复的条目。原始主张是各论文
    摘要里的整句，措辞差异极大：实测同一路线内 873 对同类型主张的 3-gram
    相似度最大仅 0.125，因此本函数几乎从不触发合并，support_level 也就恒为
    single。跨文献的语义层归并由 :func:`cluster_claims_into_theses` 负责。
    """
    merged: dict[str, dict[str, Any]] = {}

    for claim in claims:
        text = claim["claim_text"]
        key = _normalize(text)[:80]
        if key not in merged:
            merged[key] = {
                "claim_text": text,  # 保留第一条的原文
                "claim_type": claim["claim_type"],
                "evidence_ids": [],
                "paper_ids": set(),
            }
        merged[key]["evidence_ids"].append(claim["evidence_id"])
        merged[key]["paper_ids"].add(claim["paper_id"])

    # 转换 set → list
    for key in merged:
        merged[key]["evidence_ids"] = list(dict.fromkeys(merged[key]["evidence_ids"]))
        merged[key]["paper_ids"] = list(merged[key]["paper_ids"])

    return merged


# 语义聚簇的确定性护栏上限。绝对上限挡住超大论点；相对上限挡住"整条路线
# 归为一个论点"——那等于凭空制造跨文献共识，而绝对上限对主张数少的路线无效
# （8 条以内的路线可以被整体折叠）。
_MAX_THESIS_MEMBERS = 8
_MAX_THESIS_MEMBER_RATIO = 0.5


def cluster_claims_into_theses(
    merged_claims: dict[str, dict[str, Any]],
    *,
    route_name: str,
    topic: str,
    card_map: dict[str, dict[str, Any]],
    llm,
) -> dict[str, dict[str, Any]]:
    """把字面层主张按语义归并为论点，使同一论点挂上多篇论文的证据。

    字面层合并（``_merge_similar_claims``）要求整句近乎逐字相同，实际几乎
    从不触发，导致每条主张只有一篇论文证据、support_level 恒为 single。这里
    请 LLM 判断哪些主张在学术含义上讲同一件事，把它们的 evidence_ids 与
    paper_ids 并入同一论点。

    LLM 只输出分组与概括文本，所有证据绑定与数量判定仍由代码完成。分组结果
    必须通过确定性护栏才被采纳，任一条不满足即整体回退到输入原样：

    1. 索引必须是输入的一个划分：不遗漏、不重复、不越界。
    2. 单个论点成员数不超过 ``_MAX_THESIS_MEMBERS``，且不超过输入总数的
       ``_MAX_THESIS_MEMBER_RATIO``——后者挡住"整条路线折叠成一个论点"。
    3. 多成员论点的成员必须来自不同论文——同篇论文的多个字段主张合并成
       "多篇文献支持"是伪造跨文献证据。
    4. 合并后的论点数必须真的少于输入数，否则本次调用无收益。

    Returns:
        与输入同构的 dict；LLM 不可用或护栏未通过时返回输入原样。
    """
    if llm is None or len(merged_claims) < 2:
        return merged_claims

    items = list(merged_claims.items())
    payload = []
    for index, (_key, data) in enumerate(items):
        payload.append({
            "claim_index": index,
            "claim_text": str(data.get("claim_text") or "")[:300],
            "claim_type": str(data.get("claim_type") or "finding"),
            "paper_ids": list(data.get("paper_ids") or []),
        })

    try:
        from app.prompt.claim_thesis import CLAIM_THESIS_CLUSTERING_PROMPT

        response = llm.complete(
            CLAIM_THESIS_CLUSTERING_PROMPT.format(
                route_name=route_name or "未命名路线",
                topic=topic or "",
                claims_json=json.dumps(payload, ensure_ascii=False, indent=1),
            ),
            response_format="json_object",
            temperature=0.0,
            operation="claim_thesis_clustering",
        )
        from app.core.json_utils import parse_json_object

        data = parse_json_object(response if isinstance(response, str) else str(response))
        theses = data.get("theses") or []
    except Exception as exc:  # noqa: BLE001
        logger.info("Claim thesis clustering skipped (%s); keeping literal merge", exc)
        return merged_claims

    grouped = _accept_thesis_grouping(theses, items, card_map)
    if grouped is None:
        return merged_claims
    logger.info(
        "Claim thesis clustering: route=%r %d literal claims -> %d theses",
        route_name, len(items), len(grouped),
    )
    return grouped


def _accept_thesis_grouping(
    theses: list[Any],
    items: list[tuple[str, dict[str, Any]]],
    card_map: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]] | None:
    """校验 LLM 分组并构造合并结果；不合格返回 None 表示拒绝采纳。"""
    if not isinstance(theses, list) or not theses:
        return None

    total = len(items)
    # 相对上限对小路线会收得过紧（5 条主张时只允许 2 成员，永远升不到
    # strong），故设下限 3；实测路线有 13~17 条主张，cap 落在 6~8。
    member_cap = min(
        _MAX_THESIS_MEMBERS,
        max(3, int(total * _MAX_THESIS_MEMBER_RATIO)),
    )
    seen: set[int] = set()
    normalized: list[tuple[str, str, list[int]]] = []
    for thesis in theses:
        if not isinstance(thesis, dict):
            return None
        raw_indices = thesis.get("member_indices")
        if not isinstance(raw_indices, list) or not raw_indices:
            return None
        indices: list[int] = []
        for value in raw_indices:
            if isinstance(value, bool) or not isinstance(value, int):
                return None
            if not 0 <= value < total or value in seen:
                # 越界或重复：分组不是一个合法划分。
                return None
            seen.add(value)
            indices.append(value)
        if len(indices) > member_cap:
            logger.info(
                "Thesis grouping rejected: %d members exceeds cap %d (total %d)",
                len(indices), member_cap, total,
            )
            return None
        text = str(thesis.get("thesis_text") or "").strip()
        claim_type = str(thesis.get("claim_type") or "").strip()
        normalized.append((text, claim_type, indices))

    if len(seen) != total:
        logger.info(
            "Thesis grouping rejected: covered %d/%d claims", len(seen), total,
        )
        return None
    if len(normalized) >= total:
        # 没有任何合并发生，用输入原样即可（避免用 LLM 改写替换原文）。
        return None

    result: dict[str, dict[str, Any]] = {}
    for text, claim_type, indices in normalized:
        members = [items[i][1] for i in indices]
        evidence_ids: list[str] = []
        paper_ids: list[str] = []
        for member in members:
            evidence_ids.extend(str(eid) for eid in member.get("evidence_ids") or [])
            paper_ids.extend(str(pid) for pid in member.get("paper_ids") or [])
        evidence_ids = list(dict.fromkeys(evidence_ids))
        paper_ids = list(dict.fromkeys(paper_ids))

        if len(indices) > 1 and len(paper_ids) < 2:
            # 同一篇论文的多条字段主张被并成一个"论点"：合并本身可以接受，
            # 但它不构成跨文献支撑，且原文表述比概括更可核验，故保留原样。
            for i in indices:
                key, data = items[i]
                result[key] = data
            continue

        # 多成员论点用 LLM 概括；成员类型不一致时按代码侧多数票定型，不采纳
        # LLM 给出的 mixed，避免下游拿不到已知的 claim_type 规则。
        merged_type = _majority_claim_type(members, fallback=claim_type)
        merged_text = text if len(indices) > 1 and text else str(
            members[0].get("claim_text") or ""
        )
        key = _normalize(merged_text)[:80] or f"thesis:{len(result)}"
        while key in result:
            key = f"{key}:{len(result)}"
        result[key] = {
            "claim_text": merged_text,
            "claim_type": merged_type,
            "evidence_ids": evidence_ids,
            "paper_ids": paper_ids,
            "thesis_member_count": len(indices),
        }
    return result


def _majority_claim_type(members: list[dict[str, Any]], fallback: str = "") -> str:
    """成员主张类型的多数票；平票时取 _CLAIM_TYPES 已知的第一个成员类型。"""
    counts: dict[str, int] = {}
    for member in members:
        value = str(member.get("claim_type") or "").strip()
        if value:
            counts[value] = counts.get(value, 0) + 1
    if counts:
        best = max(counts.items(), key=lambda item: (item[1], -len(item[0])))
        if best[1] * 2 > len(members):
            return best[0]
        for member in members:
            value = str(member.get("claim_type") or "").strip()
            if value in _CLAIM_TYPES:
                return value
    if fallback in _CLAIM_TYPES:
        return fallback
    return "finding"


def _determine_support_level(
    evidence_count: int,
    claim_type: str,
    paper_ids: list[str] | None = None,
    card_map: dict[str, dict[str, Any]] | None = None,
) -> str:
    """根据证据数量和类型规则确定支持级别。

    claim_type 不同规则不同：
    - trend: 至少 4 篇 + 多团队
    - research_gap: 至少 3 篇 + 不能从 absence 推断
    - limitation: 必须作者明确报告
    - quantitative_result: 禁止跨论文推断
    """
    policy = get_review_threshold_policy()
    type_config = _CLAIM_TYPES.get(claim_type, _CLAIM_TYPES["finding"])
    min_ev = type_config.get("min_evidence", 1)

    if evidence_count < min_ev:
        return "single"  # 不满足该类型最低要求 → 降级

    # 检查独立性（如果提供了 paper_ids）
    independent_count = evidence_count
    if paper_ids and card_map:
        independent_count = _count_independent_sources(paper_ids, card_map)

    # 确定基础级别
    for level in ["single", "moderate", "strong"]:
        info = _SUPPORT_LEVEL_MAP[level]
        if independent_count <= info["max"]:
            return level

    # 检查是否满足 established 门禁
    if independent_count >= policy.claim_established_min_evidence:
        if type_config.get("requires_multiple_teams"):
            teams = _count_unique_teams(paper_ids or [], card_map or {})
            if teams >= policy.claim_established_min_independent_teams:
                return "established"
        # 有综述支撑也可以升级
        if _has_survey_support(paper_ids or [], card_map or {}):
            return "established"
        # 否则保持 strong
        logger.debug(
            "Evidence count=%d but established gate not met (need %d teams or survey)",
            independent_count, policy.claim_established_min_independent_teams,
        )

    return "strong"


def _count_independent_sources(
    paper_ids: list[str],
    card_map: dict[str, dict[str, Any]],
) -> int:
    """计算独立证据来源数量（去除同团队/同数据集重复）。"""
    if not paper_ids or not card_map:
        return len(paper_ids)

    # 分组依据：相同 last author + 相同 dataset → 算一组
    groups: dict[str, set[str]] = {}
    for pid in paper_ids:
        card = card_map.get(pid, {})
        authors = card.get("authors") or []
        last_author = str(authors[-1]).strip() if authors else ""
        dataset = str(card.get("dataset") or "").strip()
        doi = str(card.get("doi") or "").strip().lower()

        # DOI 相同的论文必然是同组
        if doi:
            groups.setdefault(f"doi:{doi}", set()).add(pid)
        elif last_author and dataset:
            key = f"{last_author}:{dataset}"
            groups.setdefault(key, set()).add(pid)
        else:
            groups.setdefault(pid, set()).add(pid)

    return len(groups)


def _count_unique_teams(
    paper_ids: list[str],
    card_map: dict[str, dict[str, Any]],
) -> int:
    """统计独立研究团队数量（按 last author 去重）。"""
    last_authors: set[str] = set()
    for pid in paper_ids:
        card = card_map.get(pid, {})
        authors = card.get("authors") or []
        last = str(authors[-1]).strip().casefold() if authors else ""
        if last:
            last_authors.add(last)
    return max(1, len(last_authors))


def _has_survey_support(
    paper_ids: list[str],
    card_map: dict[str, dict[str, Any]],
) -> bool:
    """检查是否有综述/调查类论文支撑。"""
    survey_keywords = [
        "survey", "review", "comprehensive", "systematic", "bibliometric",
        "meta-analysis", "综述", "系统评价", "元分析", "文献计量",
    ]
    for pid in paper_ids:
        card = card_map.get(pid, {})
        title = str(card.get("title") or "").lower()
        if any(kw in title for kw in survey_keywords):
            return True
        if str(card.get("evidence_role") or "") == "survey":
            return True
    return False


def _classify_gap_claim(
    claim_text: str,
    evidence_ids: list[str],
    card_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """区分 evidence_of_absence 和 absence_of_evidence。

    只有作者明确报告的局限才能作为 evidence_of_absence。
    仅因检索未发现而推断的 gap 是 absence_of_evidence。
    """
    # 检查是否有任何证据直接来自 author-stated limitation
    has_explicit_limitation = False
    for eid in evidence_ids:
        paper_id = _paper_id_from_evidence(eid)
        card = card_map.get(paper_id, {})
        for field, claims in (card.get("field_claims") or {}).items():
            if field not in ("limitations", "limitation"):
                continue
            for claim in claims or []:
                if not isinstance(claim, dict):
                    continue
                if str(claim.get("evidence_id") or "") == eid:
                    if claim.get("explicitly_reported"):
                        has_explicit_limitation = True
                        break

    return {
        "is_evidence_of_absence": has_explicit_limitation,
        "classification": "evidence_of_absence" if has_explicit_limitation else "absence_of_evidence",
        "allowed_phrasing": (
            _GAP_PHRASING_RULES["evidence_of_absence"]
            if has_explicit_limitation
            else _GAP_PHRASING_RULES["absence_of_evidence"]
        ),
        "forbidden_phrases": _GAP_PHRASING_RULES["forbidden"],
    }


def _llm_refine_claims(
    claims: list[dict[str, Any]],
    route_name: str,
    research_question: str,
    llm,
) -> list[dict[str, Any]]:
    """用 LLM 优化主张表述，使其更学术化、更精确。不改变证据绑定。"""
    claims_json = json.dumps([
        {"text": c["claim_text"], "type": c["claim_type"], "level": c["support_level"]}
        for c in claims[:12]
    ], ensure_ascii=False)

    prompt = f"""你是学术编辑。请优化以下研究路线的论证主张表述，使其更精确、更学术化。

路线：{route_name}
研究问题：{research_question}

当前主张：
{claims_json}

要求：
1. 保持每条主张的证据强度不变（single/modrate/strong/established）
2. 不允许将 single-evidence 主张升级为趋势判断
3. 不允许添加原文没有的新事实
4. 返回优化后的主张列表

严格返回 JSON：
{{"refined_claims": [{{"claim_text": "...", "claim_type": "...", "support_level": "..."}}]}}
"""
    try:
        response = llm.complete(prompt, response_format="json_object", temperature=0.0)
        from app.core.json_utils import parse_json_object
        data = parse_json_object(response if isinstance(response, str) else str(response))
        refined = data.get("refined_claims") or []
        if len(refined) == len(claims):
            for i, ref in enumerate(refined):
                if str(ref.get("claim_text") or "").strip():
                    claims[i]["claim_text"] = str(ref["claim_text"]).strip()
    except Exception as exc:
        logger.debug("LLM claim refinement skipped: %s", exc)

    return claims


def validate_claim_citation_consistency(
    review_text: str,
    claim_plans: list[dict[str, Any]],
    citation_map: dict[str, int] | None = None,
) -> dict[str, Any]:
    """检查正文引用是否使用了 Claim Plan 授权的证据。

    两层检查：
    1. Coarse: 引用的 paper 是否出现在任何 claim 的 evidence 中？
    2. Fine: 引用的 paper 是否出现在最佳匹配 claim 的 evidence 中？

    Returns:
        {consistent_sentences, inconsistent_sentences, unmapped_papers, consistency_rate}
    """
    if not review_text or not claim_plans:
        return {"consistent_sentences": 0, "inconsistent_sentences": 0,
                "unmapped_papers": [], "consistency_rate": 1.0}

    # 构建反向映射：paper_id → {claim_ids}
    paper_to_claims: dict[str, set[str]] = {}
    all_authorized_paper_ids: set[str] = set()
    for plan in claim_plans:
        route_id = str(plan.get("route_id") or "")
        for claim_index, claim in enumerate(plan.get("claims") or []):
            claim_id = str(
                claim.get("claim_id")
                or claim.get("id")
                or f"{route_id}:claim_{claim_index}"
            )
            for eid in (claim.get("evidence_ids") or []):
                paper_id = _paper_id_from_evidence(eid)
                if paper_id:
                    paper_to_claims.setdefault(paper_id, set()).add(claim_id)
                    all_authorized_paper_ids.add(paper_id)

    # 反向 citation_map: citation_number → paper_id
    cited_number_to_paper: dict[int, str] = {}
    if citation_map:
        cited_number_to_paper = {
            num: str(paper_id)
            for paper_id, num in citation_map.items()
        }

    # 按句号分割，找包含引用的句子
    import re
    sentences = re.split(r"[。；\n]", review_text)
    factual_sentences = [
        (i, s.strip()) for i, s in enumerate(sentences)
        if re.search(r"\[\d+\]", s) and len(s.strip()) > 10
    ]

    consistent = 0
    inconsistent: list[dict[str, Any]] = []
    unmapped_papers: set[str] = set()
    validly_authorized_papers: set[str] = set()
    unauthorized_cited_papers: set[str] = set()

    for idx, sentence in factual_sentences:
        cited_numbers = [int(n) for n in re.findall(r"\[(\d+)\]", sentence)]
        cited_paper_ids = [
            cited_number_to_paper.get(n, "")
            for n in cited_numbers
        ]
        cited_paper_ids = [pid for pid in cited_paper_ids if pid]

        if not cited_paper_ids:
            continue

        # Coarse check: 每个 cited paper 是否在任意 claim 的 evidence 中
        unauthorized_papers = [
            pid for pid in cited_paper_ids
            if pid not in all_authorized_paper_ids
        ]
        unmapped_papers.update(unauthorized_papers)

        # Fine check: only the exact matched claim authorizes its evidence.
        best_claim_id = _find_best_matching_claim(sentence, claim_plans)
        if best_claim_id:
            matched_claim = next(
                (
                    claim
                    for plan in claim_plans
                    for claim in plan.get("claims") or []
                    if str(claim.get("claim_id") or claim.get("id") or "") == best_claim_id
                ),
                None,
            )
            if matched_claim:
                allowed_pids = {
                    paper_id
                    for eid in matched_claim.get("evidence_ids") or []
                    if (paper_id := _paper_id_from_evidence(eid))
                }
                fine_unauthorized = [
                    pid for pid in cited_paper_ids
                    if pid not in allowed_pids
                ]
                if fine_unauthorized:
                    unauthorized_cited_papers.update(fine_unauthorized)
                    validly_authorized_papers.update(
                        pid for pid in cited_paper_ids if pid in allowed_pids
                    )
                    inconsistent.append({
                        "sentence_index": idx,
                        "sentence": sentence[:150],
                        "cited_papers": cited_paper_ids,
                        "best_claim": best_claim_id,
                        "unauthorized_in_claim": fine_unauthorized,
                    })
                else:
                    consistent += 1
                    validly_authorized_papers.update(cited_paper_ids)
            else:
                consistent += 1
                validly_authorized_papers.update(cited_paper_ids)
        else:
            # 找不到匹配 claim（可能是背景句），只做 coarse 检查
            if not unauthorized_papers:
                consistent += 1
                validly_authorized_papers.update(cited_paper_ids)
            else:
                unauthorized_cited_papers.update(unauthorized_papers)
                inconsistent.append({
                    "sentence_index": idx,
                    "sentence": sentence[:150],
                    "cited_papers": cited_paper_ids,
                    "best_claim": "none",
                    "unauthorized_in_claim": unauthorized_papers,
                })

    total = consistent + len(inconsistent)
    return {
        "consistent_sentences": consistent,
        "inconsistent_sentences": len(inconsistent),
        "consistency_rate": consistent / total if total > 0 else 1.0,
        "unmapped_papers": sorted(unmapped_papers),
        "validly_authorized_paper_ids": sorted(validly_authorized_papers),
        "unauthorized_cited_paper_ids": sorted(unauthorized_cited_papers),
        "inconsistent_samples": inconsistent[:5],
    }


def _find_best_matching_claim(
    sentence: str,
    claim_plans: list[dict[str, Any]],
) -> str | None:
    """找与句子最匹配的 claim plan（按 token overlap）。"""
    import re
    sent_tokens = set(re.findall(r"[一-鿿a-z0-9]{2,}", sentence.lower()))
    if not sent_tokens:
        return None

    best_id = None
    best_score = 0.0
    for plan in claim_plans:
        route_id = str(plan.get("route_id") or "")
        for claim_index, claim in enumerate(plan.get("claims") or []):
            claim_id = str(
                claim.get("claim_id")
                or claim.get("id")
                or f"{route_id}:claim_{claim_index}"
            )
            claim_text = str(claim.get("claim_text") or "")
            claim_tokens = set(re.findall(r"[一-鿿a-z0-9]{2,}", claim_text.lower()))
            union = sent_tokens | claim_tokens
            if not union:
                continue
            score = len(sent_tokens & claim_tokens) / len(union)
            if score > best_score:
                best_score = score
                best_id = claim_id

    return best_id if best_score > 0.05 else None


def _is_author_stated(evidence_id: str, card_map: dict[str, dict[str, Any]]) -> bool:
    """检查某条证据是否是作者明确报告的。"""
    paper_id = _paper_id_from_evidence(evidence_id)
    card = card_map.get(paper_id, {})
    for field_claims in (card.get("field_claims") or {}).values():
        for claim in field_claims or []:
            if not isinstance(claim, dict):
                continue
            if str(claim.get("evidence_id") or "") == evidence_id:
                return bool(claim.get("explicitly_reported"))
    return False


def _claim_id(route_id: str, claim_text: str) -> str:
    safe = re.sub(r"[^a-z0-9一-鿿]+", "_", _normalize(claim_text)[:40]).strip("_")
    return f"{route_id}:{safe}" if safe else f"{route_id}:unknown"


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9一-鿿]+", "", str(text or "").casefold())
