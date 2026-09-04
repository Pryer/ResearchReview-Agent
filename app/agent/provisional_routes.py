"""搜索前的概念规划层：生成候选研究框架，指导后续定向检索。

在检索之前，LLM 根据用户意图、主题和视角生成：
1. Research Scope — 明确研究边界
2. Background Outline — 研究背景的论证结构
3. Provisional Routes — 候选研究路线（含每条路线的研究问题、核心概念、检索式）

检索完成后，这些候选路线会被证据验证和修正（KEEP/MERGE/SPLIT/DROP），
而不是让论文数据无约束地自行聚类。
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.core.logger import get_logger

logger = get_logger(__name__)

# 候选路线数量范围
_MIN_ROUTES = 1
_MAX_ROUTES = 5

# 每条路线的最低核心概念数
_MIN_CONCEPTS_PER_ROUTE = 3


def generate_provisional_routes(
    state: dict[str, Any],
    llm,
) -> dict[str, Any]:
    """根据用户意图和语义框架生成候选研究路线（搜索前执行）。

    返回包含 research_scope、background_outline、provisional_routes 的 dict。
    """
    topic = str(state.get("canonical_topic") or state.get("topic") or "")
    user_query = str(state.get("user_query") or "")
    semantic_frame = state.get("research_semantic_frame") or {}

    if not topic or not llm:
        return {}

    prompt = _build_provisional_route_prompt(topic, user_query, semantic_frame)

    try:
        response = llm.complete(
            prompt,
            response_format="json_object",
            temperature=0.0,
            operation="generate_provisional_routes",
        )
        from app.core.json_utils import parse_json_object

        data = parse_json_object(response if isinstance(response, str) else str(response))
        routes = data.get("provisional_routes") or []
        if not isinstance(routes, list) or len(routes) < _MIN_ROUTES:
            logger.warning(
                "Provisional routes under minimum (%d < %d), discarding",
                len(routes), _MIN_ROUTES,
            )
            return {}

        validated_routes = _validate_and_normalize_routes(routes)
        if len(validated_routes) < _MIN_ROUTES:
            return {}

        result = {
            "research_scope": data.get("research_scope") or {},
            "background_outline": data.get("background_outline") or {},
            "provisional_routes": validated_routes,
        }
        logger.info(
            "Provisional framework generated: %d routes, scope=%s",
            len(validated_routes),
            str(result["research_scope"].get("perspective") or "")[:60],
        )
        return result

    except Exception as exc:
        logger.warning("Provisional route generation failed: %s", exc)
        return {}


def route_aware_search_queries(
    provisional_routes: list[dict[str, Any]],
    global_topic: str = "",
) -> list[dict[str, Any]]:
    """从候选路线生成分路线检索式，每条路线产出中英文各 1-2 条。

    Returns:
        [{route_id, route_name, queries: [{text, language}]}, ...]
    """
    branches: list[dict[str, Any]] = []
    for route in provisional_routes:
        concepts = [
            str(item).strip()
            for item in route.get("core_concepts") or []
            if str(item).strip()
        ]
        route_queries = list(dict.fromkeys(
            str(item).strip()
            for item in route.get("search_queries") or []
            if str(item).strip()
        ))
        name = str(route.get("name") or "")
        route_id = str(route.get("route_id") or _safe_route_id(name))

        # 确保每条路线至少从 core_concepts 拼一个查询
        if not route_queries and concepts:
            route_queries = [" ".join(concepts[:3])]

        if not route_queries:
            continue

        branches.append({
            "route_id": route_id,
            "route_name": name,
            "research_question": str(route.get("research_question") or ""),
            "queries": route_queries[:4],
            "core_concepts": concepts,
        })

    return branches


def validate_routes_against_evidence(
    provisional_routes: list[dict[str, Any]],
    paper_cards: list[dict[str, Any]],
    llm=None,
    topic: str = "",
    semantic_frame: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """用 v2 特征矩阵分别评估路线有效性与证据充分性。"""
    # Route Validator v2 owns matching and decision policy.  The implementation
    # lives in a focused module so graph orchestration and route planning remain
    # independent from feature extraction.
    from app.agent.route_validator import validate_route_evidence

    return validate_route_evidence(
        provisional_routes,
        paper_cards,
        llm=llm,
        topic=topic,
        semantic_frame=semantic_frame,
    )


def _legacy_validate_routes_against_evidence(
    provisional_routes: list[dict[str, Any]],
    paper_cards: list[dict[str, Any]],
    llm=None,
) -> dict[str, Any]:
    """旧评分器，仅用于读取旧状态时的迁移对照，不参与当前主流程。"""
    from app.tools.cluster_papers import taxonomy_tokens, TOKEN_STOPWORDS

    if not provisional_routes or not paper_cards:
        return {"validated_routes": provisional_routes, "decisions": [], "new_routes": [],
                "coverage": {}, "assignment_map": {}}

    card_map = {
        str(card.get("paper_id") or ""): card
        for card in paper_cards
        if card.get("paper_id")
    }

    # 预处理每张卡片的词项
    card_tokens_cache: dict[str, set[str]] = {}
    for card in paper_cards:
        paper_id = str(card.get("paper_id") or "")
        card_text = " ".join(
            str(card.get(field) or "")
            for field in ("title", "research_problem", "method", "abstract")
        )
        card_tokens_cache[paper_id] = taxonomy_tokens(card_text)

    # 预计算每条路线的词项
    route_tokens_map: dict[str, set[str]] = {}
    for route in provisional_routes:
        route_text = " ".join([
            str(route.get("name") or ""),
            str(route.get("research_question") or ""),
            " ".join(
                re.sub(r"\W+", "", str(c).casefold())
                for c in (route.get("core_concepts") or [])
                if str(c).strip()
            ),
        ])
        route_tokens_map[route.get("route_id")] = taxonomy_tokens(route_text)

    # ===== Phase A: Per-paper route fit + multi-label assignment =====
    # 关键修改：不再用 fit>0 作为 matched 门槛，而是用自适应 supporting_threshold
    all_fits_global: list[float] = []
    paper_fits: dict[str, dict[str, float]] = {}
    for card in paper_cards:
        pid = str(card.get("paper_id") or "")
        paper_fits[pid] = {}
        for route in provisional_routes:
            rid = route.get("route_id")
            inter = card_tokens_cache[pid] & route_tokens_map[rid]
            union = card_tokens_cache[pid] | route_tokens_map[rid]
            fit = len(inter) / len(union) if union else 0.0
            paper_fits[pid][rid] = fit
            all_fits_global.append(fit)

    core_threshold, supporting_threshold = _adaptive_thresholds(all_fits_global)

    # 论文分类：single-route / cross-route / ambiguous / unassigned
    assignment_map: dict[str, dict[str, Any]] = {}
    for pid, fits in paper_fits.items():
        ranked = sorted(fits.items(), key=lambda x: -x[1])
        top1_rid, top1_fit = ranked[0]
        top2_fit = ranked[1][1] if len(ranked) >= 2 else 0.0
        margin = top1_fit - top2_fit

        if top1_fit >= core_threshold:
            # 检查是否有第二路线也超过 core 阈值
            cross_routes = [
                rid for rid, fit in ranked[1:]
                if fit >= core_threshold
            ]
            if cross_routes and margin < (core_threshold * 0.5):
                assignment_map[pid] = {
                    "type": "cross_route",
                    "primary_route": top1_rid,
                    "secondary_routes": cross_routes,
                    "top_fit": top1_fit,
                    "margin": margin,
                }
            else:
                assignment_map[pid] = {
                    "type": "single_route",
                    "primary_route": top1_rid,
                    "top_fit": top1_fit,
                    "margin": margin,
                }
        elif top1_fit >= supporting_threshold:
            assignment_map[pid] = {
                "type": "ambiguous_uncertain",
                "best_route": top1_rid,
                "top_fit": top1_fit,
                "margin": margin,
            }
        else:
            assignment_map[pid] = {
                "type": "unassigned",
                "top_fit": top1_fit,
            }

    # ===== Phase B: 路线评分（只用 confident assignments） =====
    route_scores: list[dict[str, Any]] = []
    for route in provisional_routes:
        rid = route.get("route_id")
        core_ids = [
            pid for pid, am in assignment_map.items()
            if am.get("primary_route") == rid and am["type"] == "single_route"
        ]
        cross_ids = [
            pid for pid, am in assignment_map.items()
            if am.get("primary_route") == rid and am["type"] == "cross_route"
        ] + [
            pid for pid, am in assignment_map.items()
            if rid in (am.get("secondary_routes") or [])
        ]
        supporting_ids = [
            pid for pid, am in assignment_map.items()
            if am.get("best_route") == rid and am["type"] == "ambiguous_uncertain"
        ]

        coherence = _compute_route_coherence(core_ids, card_tokens_cache)

        route_scores.append({
            "route_id": rid,
            "route_name": route.get("name"),
            "core_paper_count": len(core_ids),
            "cross_route_paper_count": len(cross_ids),
            "supporting_paper_count": len(supporting_ids),
            "paper_count": len(core_ids) + len(cross_ids) + len(supporting_ids),
            "mean_route_fit": (
                sum(paper_fits[pid][rid] for pid in core_ids) / len(core_ids)
                if core_ids else 0.0
            ),
            "semantic_coherence": coherence,
            "core_paper_ids": core_ids,
            "cross_route_paper_ids": cross_ids,
            "supporting_paper_ids": supporting_ids,
            "core_threshold": core_threshold,
            "supporting_threshold": supporting_threshold,
        })

    # ===== Phase C: 路线间重叠（只在 core papers 上计算） =====
    for i, ri in enumerate(route_scores):
        ri_core = set(ri["core_paper_ids"])
        overlaps = []
        for j, rj in enumerate(route_scores):
            if i == j:
                continue
            rj_core = set(rj["core_paper_ids"])
            union = ri_core | rj_core
            overlap = len(ri_core & rj_core) / len(union) if union else 0.0
            overlaps.append((rj["route_id"], overlap))
        ri["route_overlap"] = max((o for _, o in overlaps), default=0.0)
        ri["overlap_with"] = max(overlaps, key=lambda x: x[1], default=("", 0.0))[0]

        # 概念重叠：比较 research_question tokens
        ri_rq = route_tokens_map.get(ri["route_id"], set())
        ri["conceptual_overlap"] = 0.0
        for j, rj in enumerate(route_scores):
            if i == j:
                continue
            rj_rq = route_tokens_map.get(rj["route_id"], set())
            union = ri_rq | rj_rq
            if union:
                co = len(ri_rq & rj_rq) / len(union)
                ri["conceptual_overlap"] = max(ri["conceptual_overlap"], co)

    # ===== Phase D: 两阶段决策 (Diagnosis → Action) =====
    decisions: list[dict[str, Any]] = []
    kept_routes: list[dict[str, Any]] = []
    to_merge: list[dict[str, Any]] = []

    for scores in route_scores:
        core = scores["core_paper_count"]
        cross = scores["cross_route_paper_count"]
        coherence = scores["semantic_coherence"]
        overlap = scores["route_overlap"]
        conceptual_overlap = scores["conceptual_overlap"]
        mean_fit = scores["mean_route_fit"]

        # Stage A: Diagnosis
        if core >= 3 and coherence >= 0.7:
            if overlap >= 0.40 and conceptual_overlap >= 0.40:
                diagnosis = "OVERLAPPING_ROUTE"
            else:
                diagnosis = "STRONG_ROUTE"
        elif core >= 3 and coherence < 0.7:
            diagnosis = "BROAD_ROUTE"
        elif core >= 1 and cross >= 2 and coherence >= 0.6:
            diagnosis = "NICHE_ROUTE"
        elif core >= 1 and coherence >= 0.8:
            diagnosis = "NICHE_ROUTE"
        elif core >= 1:
            diagnosis = "WEAK_ROUTE"
        elif cross >= 3 and coherence >= 0.6:
            # 自身无法形成独立研究问题 → 作为另一路线的方法子集
            if core == 0 and cross >= 3:
                diagnosis = "DEPENDENT_SUBROUTE"
            else:
                # 研究链条存在输入输出依赖但各自独立 → 保留 + 标记依赖
                diagnosis = "PIPELINE_DEPENDENT_ROUTE"
        elif mean_fit < 0.02 and coherence >= 0.6 and core + cross >= 3:
            # 低 fit 但论文间一致 → 可能发现新路线
            diagnosis = "MISALIGNED_CLUSTER"
        else:
            diagnosis = "INSUFFICIENT_EVIDENCE"

        # Stage B: Action
        action_map = {
            "STRONG_ROUTE": ("KEEP", f"core={core} coh={coherence:.2f} ovlp={overlap:.2f}"),
            "OVERLAPPING_ROUTE": (
                "REDEFINE_BOUNDARY",
                f"core={core} paper_ovlp={overlap:.2f} concept_ovlp={conceptual_overlap:.2f}",
            ),
            "BROAD_ROUTE": ("OUTLIER_CHECK", f"core={core} coh={coherence:.2f} — denoise then re-evaluate"),
            "NICHE_ROUTE": ("KEEP", f"niche: core={core} cross={cross} coh={coherence:.2f}"),
            "WEAK_ROUTE": ("MERGE", f"core={core} cross={cross} coh={coherence:.2f}"),
            "DEPENDENT_SUBROUTE": ("MERGE", f"no independent research question, core={core} cross={cross}"),
            "PIPELINE_DEPENDENT_ROUTE": ("KEEP", f"pipeline dependency, core={core} cross={cross} coh={coherence:.2f} — mark link"),
            "MISALIGNED_CLUSTER": ("ADD_NEW_ROUTE_CANDIDATE", f"low fit ({mean_fit:.3f}) + coherence ({coherence:.2f}) → possible new route"),
            "INSUFFICIENT_EVIDENCE": ("DROP", f"core={core} cross={cross}"),
        }
        action, reason = action_map[diagnosis]

        decisions.append({
            "route_id": scores["route_id"],
            "route_name": scores["route_name"],
            "diagnosis": diagnosis,
            "action": action,
            "reason": reason,
            "scores": {
                k: v for k, v in scores.items()
                if k in ("paper_count", "core_paper_count", "cross_route_paper_count",
                          "semantic_coherence", "route_overlap", "conceptual_overlap",
                          "mean_route_fit", "core_threshold", "supporting_threshold")
            },
        })

        if action in ("KEEP", "REDEFINE_BOUNDARY", "OUTLIER_CHECK"):
            route = next(r for r in provisional_routes if r.get("route_id") == scores["route_id"])
            kept_routes.append({**route, "paper_ids": scores["core_paper_ids"] + scores["cross_route_paper_ids"],
                                "core_paper_ids": scores["core_paper_ids"], "route_scores": scores})
        elif action == "MERGE":
            route = next(r for r in provisional_routes if r.get("route_id") == scores["route_id"])
            to_merge.append({**route, "paper_ids": scores["core_paper_ids"] + scores["cross_route_paper_ids"],
                             "core_paper_count": core, "route_scores": scores})

    # 收集已分配论文 ID
    assigned_paper_ids: set[str] = set()
    for route in kept_routes:
        assigned_paper_ids.update(route.get("paper_ids", []))
    for route in to_merge:
        assigned_paper_ids.update(route.get("paper_ids", []))

    # BROAD_ROUTE 去噪
    for scores in route_scores:
        diag = next((d for d in decisions if d["route_id"] == scores["route_id"]), None)
        if diag and diag.get("diagnosis") == "BROAD_ROUTE":
            cleaned_core, cleaned_coherence = _remove_outliers_and_recompute(
                scores["core_paper_ids"] + scores["cross_route_paper_ids"],
                card_tokens_cache,
                core_ids=scores["core_paper_ids"],
            )
            if cleaned_coherence >= 0.7 and len(cleaned_core) >= 3:
                diag["action"] = "KEEP"
                diag["diagnosis"] = "STRONG_ROUTE"
                diag["reason"] = f"after denoise: core={len(cleaned_core)} coh={cleaned_coherence:.2f}"
                route = next(r for r in provisional_routes if r.get("route_id") == scores["route_id"])
                kept_routes.append({**route, "paper_ids": cleaned_core, "core_paper_ids": cleaned_core,
                                    "route_scores": {**scores, "core_paper_count": len(cleaned_core), "semantic_coherence": cleaned_coherence}})

    # 合并逻辑：弱路线合并到最相似强路线（用概念重叠决定）
    for weak_route in to_merge:
        best = _find_best_merge_target(weak_route, kept_routes)
        if best:
            best["paper_ids"] = list(dict.fromkeys(
                best.get("paper_ids", []) + weak_route.get("paper_ids", [])
            ))
            decisions.append({
                "route_id": weak_route.get("route_id"),
                "action": "MERGED_INTO",
                "target_route_id": best.get("route_id"),
                "reason": f"Merged into {best.get('name', '')}",
            })

    # SPLIT 逻辑：对内部一致性低的路线，尝试子聚类拆分
    for scores in route_scores:
        if scores.get("action") == "SPLIT" or any(
            d.get("route_id") == scores["route_id"] and d.get("action") == "SPLIT"
            for d in decisions
        ):
            split_routes = _try_split_route(
                scores, card_map, card_tokens_cache, llm
            )
            if split_routes and len(split_routes) >= 2:
                for sr in split_routes:
                    kept_routes.append(sr)
                    assigned_paper_ids.update(sr.get("paper_ids", []))
                decisions.append({
                    "route_id": scores["route_id"],
                    "action": "SPLIT_INTO",
                    "split_count": len(split_routes),
                    "reason": f"Split into {len(split_routes)} sub-routes",
                })

    # ADD_NEW_ROUTE: 未分配论文先做子聚类，再判断是否形成新路线
    all_assigned = assigned_paper_ids
    unassigned_ids = [
        str(card.get("paper_id") or "")
        for card in paper_cards
        if str(card.get("paper_id") or "") not in all_assigned
    ]
    new_routes: list[dict[str, Any]] = []
    if len(unassigned_ids) >= 3:
        unassigned_cards = [
            card_map[paper_id] for paper_id in unassigned_ids if paper_id in card_map
        ]
        # 先子聚类
        sub_clusters = _cluster_unassigned(unassigned_cards, card_tokens_cache)
        for cluster in sub_clusters:
            if len(cluster["paper_ids"]) >= 3 and cluster["coherence"] >= 0.6 and llm:
                llm_result = _detect_new_routes(
                    [card_map[pid] for pid in cluster["paper_ids"] if pid in card_map],
                    provisional_routes,
                    llm,
                )
                if llm_result:
                    for nr in llm_result:
                        nr["paper_ids"] = cluster["paper_ids"]
                    new_routes.extend(llm_result)
                    decisions.append({
                        "action": "ADD_NEW_ROUTE",
                        "reason": f"Cluster of {len(cluster['paper_ids'])} papers (coherence={cluster['coherence']:.2f}) forms new route",
                    })

    # 计算全局 coverage（用 assignment_map 的分类）
    single_route_ids = [
        pid for pid, am in assignment_map.items() if am["type"] == "single_route"
    ]
    cross_route_ids = [
        pid for pid, am in assignment_map.items() if am["type"] == "cross_route"
    ]
    ambiguous_ids = [
        pid for pid, am in assignment_map.items() if am["type"] == "ambiguous_uncertain"
    ]
    unassigned_ids = [
        pid for pid, am in assignment_map.items() if am["type"] == "unassigned"
    ]

    total = len(paper_cards)
    coverage = {
        "total_papers": total,
        "single_route_confident": len(single_route_ids),
        "cross_route_confident": len(cross_route_ids),
        "ambiguous_uncertain": len(ambiguous_ids),
        "unassigned": len(unassigned_ids),
        "evidence_understood_rate": (len(single_route_ids) + len(cross_route_ids)) / total if total else 0.0,
        "core_threshold": core_threshold,
        "supporting_threshold": supporting_threshold,
    }

    return {
        "validated_routes": kept_routes + new_routes,
        "decisions": decisions,
        "route_scores": route_scores,
        "new_routes": new_routes,
        "unassigned_paper_ids": unassigned_ids,
        "ambiguous_paper_ids": ambiguous_ids,
        "cross_route_paper_ids": cross_route_ids,
        "assignment_map": {
            pid: am for pid, am in assignment_map.items()
        },
        "coverage": coverage,
    }


def _remove_outliers_and_recompute(
    all_paper_ids: list[str],
    card_tokens_cache: dict[str, set[str]],
    core_ids: list[str] | None = None,
) -> tuple[list[str], float]:
    """移除离群论文后重新计算 coherence。使用 MAD-based 异常检测。"""
    target_ids = core_ids or all_paper_ids
    if len(target_ids) < 4:
        return (target_ids, _compute_route_coherence(target_ids, card_tokens_cache))

    # 计算每篇论文与其余论文的 pairwise 相似度中位数
    paper_scores: list[tuple[str, float]] = []
    for pid in target_ids:
        tokens = card_tokens_cache.get(pid, set())
        sims = []
        for other in target_ids:
            if other == pid:
                continue
            other_tokens = card_tokens_cache.get(other, set())
            union = tokens | other_tokens
            sims.append(len(tokens & other_tokens) / len(union) if union else 0.0)
        paper_scores.append((pid, sum(sims) / len(sims) if sims else 0.0))

    scores = [s for _, s in paper_scores]
    median = sorted(scores)[len(scores) // 2]
    mad = sorted(abs(s - median) for s in scores)[len(scores) // 2] if scores else 0.0

    # MAD=0 时所有论文高度一致，不需要去噪
    if mad == 0:
        return (target_ids, _compute_route_coherence(target_ids, card_tokens_cache))

    threshold = median - 1.5 * mad
    cleaned = [pid for pid, s in paper_scores if s >= threshold]

    # 不能去掉太多
    if len(cleaned) < max(3, len(target_ids) * 0.6):
        return (target_ids, _compute_route_coherence(target_ids, card_tokens_cache))

    return (cleaned, _compute_route_coherence(cleaned, card_tokens_cache))


# ============================================================
# 多维评分辅助函数
# ============================================================

def _adaptive_thresholds(fits: list[float]) -> tuple[float, float]:
    """根据实际 fit 分布计算自适应阈值，而非硬编码 0.08/0.04。

    core = P70, supporting = P40。无数据时回退保守默认值。
    """
    if not fits:
        return (0.15, 0.08)
    sorted_fits = sorted(fits)
    n = len(sorted_fits)

    def _percentile(p: float) -> float:
        idx = max(0, min(n - 1, int(n * p / 100)))
        return sorted_fits[idx]

    core = _percentile(70)
    supporting = _percentile(40)

    # 底线保护：阈值不能低于词项集随机碰撞水平
    core = max(core, 0.05)
    supporting = max(supporting, 0.02)
    # 上限保护：不能太高导致所有论文都被排除
    core = min(core, 0.40)
    supporting = min(supporting, 0.25)

    return (core, supporting)


def _compute_route_coherence(
    core_paper_ids: list[str],
    card_tokens_cache: dict[str, set[str]],
) -> float:
    """计算路线内部论文的 pairwise Jaccard 均值。

    需要至少 2 篇论文才能计算；< 2 篇返回 0.0（不是 1.0，
    因为单篇/零篇论文不能声称"完全一致"）。
    """
    if len(core_paper_ids) < 2:
        return 0.0
    tokens_list = [
        card_tokens_cache.get(paper_id, set())
        for paper_id in core_paper_ids
    ]
    scores: list[float] = []
    for i in range(len(tokens_list)):
        for j in range(i + 1, len(tokens_list)):
            union = tokens_list[i] | tokens_list[j]
            if not union:
                continue
            scores.append(len(tokens_list[i] & tokens_list[j]) / len(union))
    return sum(scores) / len(scores) if scores else 0.0


def _cluster_unassigned(
    unassigned_cards: list[dict[str, Any]],
    card_tokens_cache: dict[str, set[str]],
) -> list[dict[str, Any]]:
    """对未分配论文做子聚类，返回簇列表（每簇带 coherence 评分）。"""
    if len(unassigned_cards) < 3:
        return []

    paper_ids = [str(card.get("paper_id") or "") for card in unassigned_cards]
    # 简单贪心聚类：取第一篇，找最相似的 >=2 篇邻居形成簇，递归处理剩余
    remaining = set(paper_ids)
    clusters: list[dict[str, Any]] = []

    while len(remaining) >= 3:
        seed = next(iter(remaining))
        seed_tokens = card_tokens_cache.get(seed, set())
        if not seed_tokens:
            remaining.discard(seed)
            continue

        neighbors = []
        for other_id in list(remaining):
            if other_id == seed:
                continue
            other_tokens = card_tokens_cache.get(other_id, set())
            union = seed_tokens | other_tokens
            if not union:
                continue
            sim = len(seed_tokens & other_tokens) / len(union)
            if sim >= 0.05:
                neighbors.append((other_id, sim))

        neighbors.sort(key=lambda x: -x[1])
        cluster_ids = [seed] + [nid for nid, _ in neighbors[:2]]
        if len(cluster_ids) >= 3:
            coherence = _compute_route_coherence(cluster_ids, card_tokens_cache)
            clusters.append({
                "paper_ids": cluster_ids,
                "coherence": coherence,
            })

        for pid in cluster_ids:
            remaining.discard(pid)

    return clusters


def _try_split_route(
    scores: dict[str, Any],
    card_map: dict[str, dict[str, Any]],
    card_tokens_cache: dict[str, set[str]],
    llm=None,
) -> list[dict[str, Any]]:
    """尝试将内部不一致的路线拆分为子路线。"""
    core_ids = scores.get("core_paper_ids") or []
    if len(core_ids) < 6:
        return []

    # 对 core 论文做子聚类
    core_cards = [card_map[pid] for pid in core_ids if pid in card_map]
    sub_clusters = _cluster_unassigned(core_cards, card_tokens_cache)

    if len(sub_clusters) < 2:
        return []

    # 如果有 LLM，让它为子簇命名
    if llm:
        result = []
        for i, cluster in enumerate(sub_clusters, 1):
            cluster_cards = [card_map[pid] for pid in cluster["paper_ids"] if pid in card_map]
            name = _name_cluster(cluster_cards, llm)
            result.append({
                "route_id": f"{scores['route_id']}_S{i}",
                "name": name or f"{scores['route_name']}（子路线{i}）",
                "research_question": "",
                "core_concepts": [],
                "paper_ids": cluster["paper_ids"],
            })
        return result

    return []


def _name_cluster(
    cards: list[dict[str, Any]],
    llm,
    *,
    topic: str = "",
    parent_name: str = "",
    sibling_names: list[str] | None = None,
    reserved_names: list[str] | None = None,
) -> str:
    """用 LLM 为论文簇生成简短名称。

    给出主题、父路线、同级与全部既有路线名称作为约束上下文：缺少这些
    信息时模型会把整个研究主题搬进名称（"跨模态小样本视频动作识别"），
    擅自改写主题的任务设定（把少样本写成"零样本动作识别"），或与既有
    路线指向同一概念（"跨模态匹配"对既有"多模态与自监督"）。
    """
    titles = [str(card.get("title") or "")[:100] for card in cards[:5]]
    constraints = [
        "只用一个不超过 12 字的中文短语命名，只返回名称，不要解释。",
        "名称必须是这批论文相对其他路线的区分点（机制、模态或建模方式）。",
    ]
    if topic:
        constraints.append(
            f"研究主题是「{topic}」：名称不得包含或复述主题本身，"
            "也不得改写主题的任务设定或限定词。"
        )
    if parent_name:
        constraints.append(f"这批论文属于「{parent_name}」路线内部的一个细分方向。")
    siblings = [str(item).strip() for item in sibling_names or [] if str(item).strip()]
    if siblings:
        constraints.append(
            "已命名的同级子路线："
            + "、".join(siblings)
            + "；名称必须与它们明显不同，不得语义重叠。"
        )
    reserved = [
        str(item).strip() for item in reserved_names or [] if str(item).strip()
    ]
    if reserved:
        constraints.append(
            "正文将同时出现以下既有路线："
            + "、".join(reserved)
            + "；名称不得与它们中的任何一条指向同一概念或共享核心词"
            "（例如已有「多模态」路线时不得再起「跨模态」类名称）。"
        )
    prompt = (
        "为以下论文簇生成中文研究路线名称：\n"
        + chr(10).join(titles)
        + "\n\n要求：\n"
        + "\n".join(f"{index}. {item}" for index, item in enumerate(constraints, 1))
    )
    try:
        response = llm.complete(prompt, temperature=0.0, max_tokens=30)
        return str(response or "").strip()[:30]
    except Exception:
        return ""


def generate_global_recall_queries(
    topic: str,
    semantic_frame: dict[str, Any] | None = None,
) -> list[str]:
    """生成全局召回检索式，防止路线导向搜索产生确认偏误。

    这些查询不使用任何候选路线的概念，只用主题本身 + 语义框架的研究对象。
    """
    queries = [topic]
    frame = semantic_frame or {}
    objects = [
        str(item.get("surface_text") or item.get("label") or "")
        for item in frame.get("research_objects") or []
        if str(item.get("surface_text") or item.get("label") or "").strip()
    ]
    if objects:
        queries.append(" ".join(objects[:3]))
    # 主题 + 通用学术后缀
    queries.extend([
        f"{topic} 研究现状",
        f"{topic} 研究进展",
    ])
    return list(dict.fromkeys(q.strip() for q in queries if q.strip()))


def _find_best_merge_target(
    weak_route: dict[str, Any],
    strong_routes: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """为弱路线找最佳合并目标。"""
    from app.tools.cluster_papers import taxonomy_tokens

    if not strong_routes:
        return None

    weak_text = " ".join([
        str(weak_route.get("name") or ""),
        str(weak_route.get("research_question") or ""),
        " ".join(str(c) for c in weak_route.get("core_concepts") or []),
    ])
    weak_tokens = taxonomy_tokens(weak_text)

    best = None
    best_score = 0.0
    for route in strong_routes:
        strong_text = " ".join([
            str(route.get("name") or ""),
            str(route.get("research_question") or ""),
            " ".join(str(c) for c in route.get("core_concepts") or []),
        ])
        strong_tokens = taxonomy_tokens(strong_text)
        union = weak_tokens | strong_tokens
        if not union:
            continue
        score = len(weak_tokens & strong_tokens) / len(union)
        if score > best_score:
            best_score = score
            best = route

    return best if best_score > 0.15 else None


def _detect_new_routes(
    unassigned_cards: list[dict[str, Any]],
    existing_routes: list[dict[str, Any]],
    llm,
) -> list[dict[str, Any]]:
    """用 LLM 检查未分配论文是否构成新的研究路线。"""
    if len(unassigned_cards) < 3:
        return []

    summary = [
        {
            "title": str(card.get("title") or ""),
            "research_problem": str(card.get("research_problem") or ""),
            "method": str(card.get("method") or ""),
        }
        for card in unassigned_cards[:10]
    ]

    existing_names = [str(r.get("name") or "") for r in existing_routes]
    existing_questions = [str(r.get("research_question") or "") for r in existing_routes]
    prompt = f"""以下论文未被现有研究路线覆盖。请严格判断它们是否构成一条真正独立的、方法论层面不同的新研究路线。

现有路线名称：{json.dumps(existing_names, ensure_ascii=False)}
现有路线研究问题：{json.dumps(existing_questions, ensure_ascii=False)}
未分配论文：{json.dumps(summary, ensure_ascii=False)}

你必须回答三个问题：
1. 这些论文是否共享一个独立的研究问题，而不是现有路线的子集、变体或技术实例？
2. 如果独立，核心研究问题是什么？
3. 它与现有哪条路线最接近，本质区别是什么？

严格返回 JSON：
{{"new_routes": [
    {{"route_id": "new_1",
      "name": "≤12字中文名称",
      "research_question": "核心研究问题",
      "core_concepts": ["概念1", "概念2", "概念3"],
      "is_distinct_route": true,
      "difference_from_existing": "与现有路线X的本质区别是...",
      "educational_or_methodological_significance": "该路线的学术意义",
      "rationale": "判断依据"
    }}
]}}

如果这些论文只是现有路线的技术变体、应用场景、或不构成独立方法论路线，返回：
{{"new_routes": []}}
"""

    try:
        response = llm.complete(prompt, response_format="json_object", temperature=0.0)
        from app.core.json_utils import parse_json_object

        data = parse_json_object(response if isinstance(response, str) else str(response))
        new_routes = data.get("new_routes") or []
        if not isinstance(new_routes, list) or not new_routes:
            return []
        return [
            {
                "route_id": str(r.get("route_id") or f"new_{i}"),
                "name": str(r.get("name") or "未命名路线"),
                "research_question": str(r.get("research_question") or ""),
                "core_concepts": [
                    str(c).strip()
                    for c in r.get("core_concepts") or []
                    if str(c).strip()
                ],
                "paper_ids": [str(card.get("paper_id") or "") for card in unassigned_cards],
            }
            for i, r in enumerate(new_routes, 1)
        ]
    except Exception as exc:
        logger.debug("New route detection skipped: %s", exc)
        return []


# ============================================================
# 内部
# ============================================================

def _build_provisional_route_prompt(
    topic: str,
    user_query: str,
    semantic_frame: dict[str, Any],
) -> str:
    """构建候选路线生成 prompt。"""
    frame_json = json.dumps(semantic_frame, ensure_ascii=False)
    return f"""你是学术综述规划器。请根据用户的研究主题和视角，在检索之前生成候选研究框架。

**用户请求**：{user_query}
**研究主题**：{topic}
**语义框架**：{frame_json}

你需要完成三件事：

1. **Research Scope（研究范围）**：
   - 明确核心研究对象、目标、可包含和应排除的内容
   - 用语义框架中的对象、方法和分析目标给出边界；将实现手段与研究对象区分。不得引用其他领域示例，也不得自行补充用户未提及的应用场景、对象或排除项。

2. **Background Outline（研究背景框架）**：
   - 3-4 个递进的段落目标，每个目标说明该段需要论证什么
   - 不写正文，只写段落目标

3. **Provisional Routes（候选研究路线）**：
   - {_MIN_ROUTES}-{_MAX_ROUTES} 条互斥的候选路线；路线数量应由当前主题和证据边界决定，不要为满足固定数量虚构路线
   - 每条路线必须有：
     * name：简洁中文名称（≤15字）
     * research_question：该路线的核心研究问题（1句话）
     * route_role：该路线在研究链中的角色（formalization / sensing / interpretation / application / synthesis）
     * core_concepts：核心概念列表（≥{_MIN_CONCEPTS_PER_ROUTE}个，用于检索）
     * semantic_anchors：3-8个中英文语义锚点或文献常用同义表达
     * method_concepts：该路线的方法机制概念，不得混入研究对象或应用场景
     * task_anchors：该路线实际解决的研究任务或输入输出关系
     * negative_anchors：容易误召回、但明确不属于该路线的概念
     * search_queries：2-3条中英文检索式
     * inclusion_criteria：明确哪些论文属于该路线
     * exclusion_criteria：明确哪些论文不属于该路线（特别是指出与相邻路线的边界）
     * boundary_note：与哪条路线容易混淆，如何区分
   - 路线之间应有清晰界限，不要重叠
   - 路线分类维度应为**研究问题+方法机制**，不是数据类型、出版类型或文件格式
   - 路线名称禁止使用：期刊论文、会议论文、实验研究、图像/视频/文本等模态名

严格返回 JSON：
{{
  "research_scope": {{
    "core_objects": ["..."],
    "goals": ["..."],
    "includes": ["..."],
    "excludes": ["..."],
    "perspective": "..."
  }},
  "background_outline": {{
    "paragraph_goals": [
      {{"id": "bg_1", "label": "...", "goal": "...", "core_question": "..."}}
    ]
  }},
  "provisional_routes": [
    {{
      "route_id": "R1",
      "name": "...",
      "research_question": "...",
      "route_role": "formalization|sensing|interpretation|application|synthesis",
      "core_concepts": ["...", "..."],
      "semantic_anchors": ["...", "..."],
      "method_concepts": ["...", "..."],
      "task_anchors": ["...", "..."],
      "negative_anchors": ["..."],
      "search_queries": ["...", "..."],
      "inclusion_criteria": ["..."],
      "exclusion_criteria": ["..."],
      "boundary_note": "...",
      "rationale": "..."
    }}
  ]
}}
"""


def _validate_and_normalize_routes(
    routes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """验证和规范化候选路线。"""
    normalized: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    for i, route in enumerate(routes, 1):
        name = str(route.get("name") or "").strip()
        if not name or name in seen_names:
            continue
        seen_names.add(name)

        concepts = list(dict.fromkeys(
            str(c).strip()
            for c in route.get("core_concepts") or []
            if str(c).strip()
        ))

        if len(concepts) < _MIN_CONCEPTS_PER_ROUTE:
            logger.debug("Route %s has too few concepts (%d), skipping", name, len(concepts))
            continue

        normalized.append({
            "route_id": str(route.get("route_id") or f"R{i}"),
            "name": name,
            "research_question": str(route.get("research_question") or "").strip(),
            "route_role": str(route.get("route_role") or "").strip(),
            "core_concepts": concepts,
            "semantic_anchors": list(dict.fromkeys(
                str(value).strip()
                for value in route.get("semantic_anchors") or []
                if str(value).strip()
            ))[:12],
            "method_concepts": list(dict.fromkeys(
                str(value).strip()
                for value in route.get("method_concepts") or []
                if str(value).strip()
            ))[:12],
            "task_anchors": list(dict.fromkeys(
                str(value).strip()
                for value in route.get("task_anchors") or []
                if str(value).strip()
            ))[:12],
            "negative_anchors": list(dict.fromkeys(
                str(value).strip()
                for value in route.get("negative_anchors") or []
                if str(value).strip()
            ))[:12],
            "anchor_expansions": [
                dict(value) for value in route.get("anchor_expansions") or []
                if isinstance(value, dict)
            ][:24],
            "search_queries": list(dict.fromkeys(
                str(q).strip()
                for q in route.get("search_queries") or []
                if str(q).strip()
            )),
            "inclusion_criteria": [
                str(c).strip()
                for c in (route.get("inclusion_criteria") or [])
                if str(c).strip()
            ],
            "exclusion_criteria": [
                str(c).strip()
                for c in (route.get("exclusion_criteria") or [])
                if str(c).strip()
            ],
            "boundary_note": str(route.get("boundary_note") or "").strip(),
            "rationale": str(route.get("rationale") or "").strip(),
        })

    return normalized


def _safe_route_id(name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9一-鿿]+", "_", str(name or "")).strip("_")
    return safe or "route"
