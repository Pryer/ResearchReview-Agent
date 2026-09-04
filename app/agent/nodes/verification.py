"""主张规划、证据门禁与引用核查节点。"""

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


@node(name="claim_plan", category="generation", description="为每条路线生成论证计划，绑定证据和语言强度")
@requires("validated_routes", "paper_cards")
@provides("claim_plans")
@optional("dynamic_taxonomy")
def claim_plan_node(state: "ResearchAgentState", llm=None) -> "ResearchAgentState":
    """在 Route Validation 之后、写作之前，为每条路线生成 Claim-Evidence Plan。

    每条主张绑定：证据 IDs + 证据数量 + 支持级别 + 允许的语言强度。
    这直接约束 writer 只能写有证据支撑的主张。
    """
    t0 = time.time()
    try:
        validated = list(state.get("validated_routes") or [])
        if validated:
            from app.agent.route_validator import merge_weak_routes_for_writing

            validated = merge_weak_routes_for_writing(validated)
        # 如果 validated_routes 为空，从 dynamic_taxonomy 回退
        if not validated:
            taxonomy = state.get("dynamic_taxonomy") or {}
            themes = taxonomy.get("themes") or []
            assignments = taxonomy.get("assignments") or []
            for theme in themes:
                tid = theme.get("theme_id", "")
                pids = [
                    a.get("paper_id", "")
                    for a in assignments
                    if a.get("primary_theme_id") == tid
                ]
                validated.append({
                    "route_id": tid,
                    "name": theme.get("name", ""),
                    "core_paper_ids": pids,
                })

        if not validated:
            state["claim_plans"] = []
            append_step(state, "claim_plan", "skipped",
                        output_data={"reason": "no validated routes"},
                        duration_ms=int((time.time() - t0) * 1000))
            return state

        from app.agent.claim_plan import build_claim_plans

        from app.core.config import get_review_threshold_policy
        policy = get_review_threshold_policy()
        plans = build_claim_plans(
            validated,
            state.get("paper_cards") or [],
            llm=llm,
            topic=str(state.get("topic") or state.get("canonical_topic") or ""),
        )
        # 同时为研究背景生成 Claim Plan
        provisional = state.get("provisional_framework") or {}
        bg_outline = provisional.get("background_outline") or {}
        if bg_outline.get("paragraph_goals"):
            from app.agent.claim_plan import build_background_claim_plan
            bg_plans = build_background_claim_plan(
                bg_outline,
                state.get("paper_cards") or [],
                llm=llm,
            )
            if bg_plans:
                plans = list(plans) + list(bg_plans)

        from app.agent.claim_plan import apply_claim_budget

        plans, claim_budget = apply_claim_budget(
            list(plans),
            int(state.get("required_reference_count") or state.get("max_papers") or 0),
        )
        state["claim_plans"] = plans

        # 汇总统计
        total_claims = sum(p["total_claims"] for p in plans)
        strong_plus = sum(p["strong_plus_claims"] for p in plans)
        single_only = sum(p["single_evidence_claims"] for p in plans)

        append_step(
            state, "claim_plan", "success",
            tool_name="build_claim_plans",
            input_data={"validated_routes": len(validated)},
            output_data={
                "threshold_policy": policy.snapshot(),
                "plan_count": len(plans),
                "total_claims": total_claims,
                "strong_plus_claims": strong_plus,
                "single_evidence_claims": single_only,
                "claim_budget": claim_budget,
                "routes": [
                    {
                        "name": p["route_name"],
                        "claims": p["total_claims"],
                        "strong+": p["strong_plus_claims"],
                    }
                    for p in plans
                ],
            },
            duration_ms=int((time.time() - t0) * 1000),
        )
        logger.info(
            "Claim plans: %d routes, %d claims (%d strong+, %d single-evidence)",
            len(plans), total_claims, strong_plus, single_only,
        )
    except Exception as e:
        logger.warning("claim_plan_node failed: %s", e)
        state["claim_plans"] = []
        append_step(state, "claim_plan", "failed", error=str(e))

    return state


@node(name="claim_evidence_gate", category="generation", description="写作前核验 Claim-Evidence 绑定并降低过强主张")
@requires("claim_plans", "paper_cards")
@provides("claim_plans", "claim_evidence_gate")
def claim_evidence_gate_node(state: "ResearchAgentState", llm=None) -> "ResearchAgentState":
    """优先修改 Claim，不让证据无限追逐模型生成的主张。"""
    t0 = time.time()
    try:
        from app.agent.claim_plan import enforce_claim_evidence_gate

        plans, report = enforce_claim_evidence_gate(
            state.get("claim_plans") or [],
            state.get("paper_cards") or [],
            llm=llm,
        )
        state["claim_plans"] = plans
        state["claim_evidence_gate"] = report
        append_step(
            state,
            "claim_evidence_gate",
            "success" if report.get("passed") else "degraded",
            tool_name="enforce_claim_evidence_gate",
            input_data={"plan_count": len(plans)},
            output_data=report,
            duration_ms=int((time.time() - t0) * 1000),
        )
    except Exception as exc:
        logger.warning("claim_evidence_gate_node failed: %s", exc)
        state["claim_evidence_gate"] = {
            "passed": False,
            "gaps": [],
            "error": str(exc),
        }
        state.setdefault("errors", []).append(f"claim_evidence_gate: {exc}")
        append_step(state, "claim_evidence_gate", "failed", error=str(exc))
    return state


# ============================================================
# Global Evidence Gate 节点
# ============================================================
@node(
    name="global_evidence_gate",
    category="generation",
    description="综述级证据充分性门禁：评估全局证据是否满足用户显式要求（只评估与推荐，不执行恢复）",
)
@requires("user_query", "paper_details")
@optional("paper_cards", "validated_routes")
@provides("global_evidence_gate")
def global_evidence_gate_node(state: "ResearchAgentState") -> "ResearchAgentState":
    """RECORD AND CONTINUE：无论是否通过，流程都继续到 Claim 规划与写作。

    v1 冻结范围：只测量与推荐，不执行任何恢复动作。
    """
    t0 = time.time()
    try:
        from app.agent.global_evidence_gate import evaluate_global_sufficiency
        from app.core.config import get_settings

        settings = get_settings()
        result = evaluate_global_sufficiency(
            state,
            min_recency_ratio=settings.global_gate_min_recency_ratio,
            route_balance_min_ratio=settings.global_gate_route_balance_min_ratio,
            peer_review_ratio_threshold=settings.global_gate_peer_review_ratio,
        )
        state["global_evidence_gate"] = result
        append_step(
            state,
            "global_evidence_gate",
            "success" if result.get("passed") else "degraded",
            tool_name="evaluate_global_sufficiency",
            input_data={
                "paper_details": len(state.get("paper_details") or []),
                "validated_routes": len(state.get("validated_routes") or []),
            },
            output_data=result,
            duration_ms=int((time.time() - t0) * 1000),
        )
    except Exception as exc:
        logger.warning("global_evidence_gate_node failed: %s", exc)
        # 与 claim_evidence_gate 的失败 dict 不同：不写 passed=False，避免
        # 最终回答误报“证据不足”；derive_result_status 只认 status == EVALUATED。
        state["global_evidence_gate"] = {
            "status": "FAILED",
            "deficits": [],
            "error": str(exc),
        }
        state.setdefault("errors", []).append(f"global_evidence_gate: {exc}")
        append_step(state, "global_evidence_gate", "failed", error=str(exc))
    return state



# ============================================================
# Citation Check 节点
# ============================================================
@node(name="citation_check", category="generation", description="生成并校验参考文献")
@requires("writing_plans", "paper_details")
@provides(
    "references", "citation_map", "citation_registry", "citation_validation",
    "reference_papers", "unique_cited_paper_count", "final_requirement_met",
)
def citation_check_node(state: "ResearchAgentState", llm=None) -> "ResearchAgentState":
    """生成并校验参考文献。"""
    t0 = time.time()
    try:
        from app.agent.deliverable_router import unconfirmed_reference_ids
        from app.tools.generate_citation import generate_and_validate_citations
        from app.tools.citation_registry import build_citation_registry

        unconfirmed_ids = unconfirmed_reference_ids(state)
        # WHY: 写作前门禁排除 rule_screened_reserve 还不够；尽力生成会获准
        # 继续写作，若引用检查重新使用全部 paper_details，这些未经 LLM 语义
        # 确认的论文仍会生成参考文献并计入显式篇数要求。最终引用源必须沿用
        # 同一排除集合，命中的正文引用由校验器如实报告为缺失。
        details = [
            paper for paper in (state.get("paper_details") or [])
            if str(paper.get("paper_id") or "") not in unconfirmed_ids
        ]
        cards_by_id = {
            str(card.get("paper_id") or ""): card
            for card in (state.get("paper_cards") or [])
        }
        # WHY: 书目字段以检索层 paper_details 为权威。此前的 {**paper, **card}
        # 会让卡片（可能经过 LLM 抽取）反向覆盖作者、题名和 DOI。
        _AUTHORITATIVE_BIBLIOGRAPHIC_FIELDS = (
            "title", "authors", "year", "venue", "doi", "url", "arxiv_id",
            "publication_status", "source",
        )
        citation_sources = []
        for paper in details:
            card = cards_by_id.get(str(paper.get("paper_id") or ""), {})
            merged = {**paper, **card}
            for field in _AUTHORITATIVE_BIBLIOGRAPHIC_FIELDS:
                if paper.get(field) not in (None, "", []):
                    merged[field] = paper[field]
            citation_sources.append(merged)
        citation_sources = citation_sources or [
            card for card in (state.get("paper_cards") or [])
            if str(card.get("paper_id") or "") not in unconfirmed_ids
        ]
        generated_text = (
            state.get("review")
            or state.get("related_work")
            or state.get("introduction")
            or ""
        )
        result = generate_and_validate_citations(
            review_text=generated_text,
            paper_cards=citation_sources,
            citation_style=state.get("citation_style", "gbt7714"),
            # 本地引用校验已经能判断缺失引用和实际引用篇数。
            # 这里不再额外调用 LLM 生成建议，避免长任务在收尾阶段再次阻塞。
            llm=None,
        )
        state["references"] = result.get("references", [])
        state["citation_validation"] = result.get("validation", {})
        state["reference_papers"] = result.get("reference_papers", [])
        state["citation_map"] = result.get("citation_map", {})
        state["citation_registry"] = build_citation_registry(
            result.get("rendered_text") or generated_text,
            citation_sources,
        )
        rendered_text = result.get("rendered_text")
        if rendered_text:
            if state.get("review"):
                state["review"] = rendered_text
            elif state.get("related_work"):
                state["related_work"] = rendered_text
            elif state.get("introduction"):
                state["introduction"] = rendered_text
        cited_ids = set(state["citation_validation"].get("cited_ids", []))
        citable_ids = {
            str(paper.get("paper_id") or "")
            for paper in citation_sources
            if paper.get("paper_id")
        }
        state["unique_cited_paper_count"] = len(cited_ids & citable_ids)
        state["final_requirement_met"] = (
            state["unique_cited_paper_count"]
            >= int(state.get("required_reference_count") or state.get("max_papers") or 0)
        )
        # 成品率在此落盘：这是"可用参考文献数"与"最终引用数"同时确定的唯一时点。
        # WHY: 该字段不在 _GENERATION_PRODUCT_KEYS 里，增量检索/引用缺口修复重置
        # 写作产物后仍保留，下一轮 fetch_detail 才能按实测损耗倒推证据池目标。
        from app.agent.nodes.retrieval import evidence_yield_report

        observed_yield = evidence_yield_report(state)
        if observed_yield:
            state["evidence_yield"] = observed_yield
        append_step(
            state, "citation_check", "success",
            tool_name="generate_citation",
            input_data={
                "review_length": len(generated_text),
                "citation_style": state.get("citation_style", "gbt7714"),
                "source_count": len(citation_sources),
                "source_sample": _summarize_papers(citation_sources, limit=10),
            },
            output_data={
                "references": len(state["references"]),
                "unique_cited_paper_count": state["unique_cited_paper_count"],
                "required_reference_count": state.get("required_reference_count"),
                "final_requirement_met": state["final_requirement_met"],
                "reference_sample": state["references"][:10],
                "citation_map_sample": dict(list(state["citation_map"].items())[:10]),
                "citation_registry_unique_papers": len(
                    state["citation_registry"].get("unique_papers") or {}
                ),
                "validation": state.get("citation_validation", {}),
            },
            duration_ms=int((time.time() - t0) * 1000),
        )
    except Exception as e:
        from app.agent.exceptions import LLMGenerationError

        error = LLMGenerationError(str(e), step="citation_check", original_error=e)
        logger.error("citation_check_node failed: %s", error.message)
        state.setdefault("errors", []).append(f"citation_check: {e}")
        append_step(state, "citation_check", "failed", error=str(e))
    return state


# ============================================================
# Claim-Evidence Verification 节点
# ============================================================
@node(name="verify_claims", category="generation", description="逐句检查生成主张是否被所引论文的 Evidence Card 支持")
@requires("writing_plans", "paper_cards")
@provides("claim_verification", "generation_quality", "evidence_quality_report")
def verify_claims_node(
    state: "ResearchAgentState",
    llm=None,
    *,
    target_sentence_indices: list[int] | None = None,
    target_claim_ids: list[str] | None = None,
    verification_scope: dict[str, Any] | str | None = None,
) -> "ResearchAgentState":
    """逐句检查生成主张；可选地只重验证目标句并合并上一轮结果。"""
    t0 = time.time()
    try:
        from app.core.citation_syntax import normalize_citation_syntax
        from app.core.config import get_review_threshold_policy
        from app.tools.verify_claims import verify_review_claims

        generated_text = (
            state.get("review")
            or state.get("related_work")
            or state.get("introduction")
            or ""
        )
        citable_ids = {
            str(card.get("paper_id") or "")
            for card in state.get("paper_cards") or []
            if card.get("paper_id")
        }
        normalized_text = normalize_citation_syntax(generated_text, citable_ids)
        if normalized_text != generated_text:
            generated_text = normalized_text
            if state.get("review"):
                state["review"] = normalized_text
            elif state.get("related_work"):
                state["related_work"] = normalized_text
            elif state.get("introduction"):
                state["introduction"] = normalized_text
        dynamic_aliases: dict[str, list[str]] = {}
        for group in state.get("required_concepts") or []:
            chinese = [
                str(term) for term in group
                if re.search(r"[\u4e00-\u9fff]", str(term))
            ]
            english = [
                str(term) for term in group
                if re.search(r"[A-Za-z]", str(term))
            ]
            for term in chinese:
                if english:
                    dynamic_aliases[term] = english
        entailment_cache = dict(state.get("claim_verification_cache") or {})
        scope = verification_scope if isinstance(verification_scope, dict) else {}
        scope_mode = (
            verification_scope if isinstance(verification_scope, str)
            else scope.get("mode") or scope.get("scope")
        )
        previous_report = (
            scope.get("previous_report")
            or scope.get("prior_report")
            or state.get("claim_verification")
        )
        effective_scope: dict[str, Any] = dict(scope)
        if scope_mode:
            effective_scope["mode"] = scope_mode
        if previous_report:
            effective_scope["previous_report"] = previous_report
        effective_sentence_targets = target_sentence_indices
        if effective_sentence_targets is None:
            effective_sentence_targets = effective_scope.get("target_sentence_indices")
        effective_claim_targets = target_claim_ids
        if effective_claim_targets is None:
            effective_claim_targets = effective_scope.get("target_claim_ids")
        target_sentence_set = set()
        for index in effective_sentence_targets or []:
            try:
                target_sentence_set.add(int(index))
            except (TypeError, ValueError):
                continue
        for claim_id in effective_claim_targets or []:
            match = re.match(r"^c(\d+)", str(claim_id or ""))
            if match:
                target_sentence_set.add(int(match.group(1)))
        report = verify_review_claims(
            generated_text,
            state.get("paper_cards") or [],
            concept_aliases=dynamic_aliases,
            llm=llm,
            entailment_cache=entailment_cache,
            target_sentence_indices=target_sentence_indices,
            target_claim_ids=target_claim_ids,
            verification_scope=effective_scope or None,
        )
        state["claim_verification_cache"] = entailment_cache
        verification_scope_data = report.get("verification_scope") or {}
        local_verification = verification_scope_data.get("mode") == "local"
        reverify_kwargs = {
            "target_sentence_indices": effective_sentence_targets,
            "target_claim_ids": effective_claim_targets,
            "verification_scope": (
                {
                    "mode": "local",
                    "previous_report": report,
                }
                if local_verification else None
            ),
        }
        unsupported_before = [
            claim for claim in report.get("claims") or []
            if claim.get("factual")
            and claim.get("support_status") == "unsupported"
            and (
                not local_verification
                or claim.get("claim_id") in set(effective_claim_targets or [])
                or (
                    (claim_match := re.match(r"^c(\d+)", str(claim.get("claim_id") or "")))
                    and int(claim_match.group(1)) in target_sentence_set
                )
            )
        ]
        # 少量独立坏句可安全隔离；大量删除会破坏段落结构并把40篇引用静默
        # 删成34篇。超过保守阈值时保留原稿并让质量门禁阻断，交给章节重写。
        factual_count = max(1, int(report.get("factual_claims") or 0))
        safe_removal_limit = max(2, min(5, int(factual_count * 0.1)))
        if (
            unsupported_before
            and len(unsupported_before) <= safe_removal_limit
            and (state.get("review") or state.get("related_work") or state.get("introduction"))
        ):
            repaired_text, removed_claims = _remove_unsupported_claims(
                generated_text, unsupported_before
            )
            if removed_claims and repaired_text != generated_text:
                _update_state_review_text(state, repaired_text)
                generated_text = repaired_text
                state["claim_repairs"] = {
                    "strategy": "remove_unsupported_sentences",
                    "removed_count": len(removed_claims),
                    "removed_claim_ids": [
                        str(claim.get("claim_id") or "")
                        for claim in removed_claims
                    ],
                    "removed_samples": [
                        str(claim.get("sentence") or "")[:180]
                        for claim in removed_claims[:5]
                    ],
                }
                report = verify_review_claims(
                    repaired_text,
                    state.get("paper_cards") or [],
                    concept_aliases=dynamic_aliases,
                    entailment_cache=entailment_cache,
                    **reverify_kwargs,
                )
        elif unsupported_before and (state.get("review") or state.get("related_work") or state.get("introduction")):
            repaired_text, rewritten_records = _rewrite_and_weaken_unsupported_claims(
                generated_text,
                unsupported_before,
                state.get("paper_cards") or [],
                llm=llm,
            )
            if rewritten_records and repaired_text != generated_text:
                new_report = verify_review_claims(
                    repaired_text,
                    state.get("paper_cards") or [],
                    concept_aliases=dynamic_aliases,
                    llm=None,
                    entailment_cache=entailment_cache,
                    **reverify_kwargs,
                )
                # WHY: 旧条件是 OR，support_rate 只要不低于原值（哪怕零改善）
                # 就把整批改写落盘；实测 rewritten_count=7 而未支持主张仍是 22，
                # 支持率反而从 47.1% 掉到 36.1%。改写必须真正减少未支持主张，
                # 否则不落盘，改由 section_rewrite_required 如实报告缺口。
                if int(new_report.get("unsupported") or 0) < int(report.get("unsupported") or 0):
                    generated_text = repaired_text
                    _update_state_review_text(state, repaired_text)
                    report = new_report

                    remaining_unsupported = [
                        claim for claim in report.get("claims") or []
                        if claim.get("factual")
                        and claim.get("support_status") == "unsupported"
                    ]
                    if remaining_unsupported and len(remaining_unsupported) <= safe_removal_limit:
                        final_text, removed_claims = _remove_unsupported_claims(
                            generated_text, remaining_unsupported
                        )
                        if removed_claims and final_text != generated_text:
                            _update_state_review_text(state, final_text)
                            generated_text = final_text
                            report = verify_review_claims(
                                final_text,
                                state.get("paper_cards") or [],
                                concept_aliases=dynamic_aliases,
                            )

                    state["claim_repairs"] = {
                        "strategy": "rewrite_and_weaken_unsupported_claims",
                        "rewritten_count": len(rewritten_records),
                        "rewritten_samples": [
                            {
                                "original": r["original"][:120],
                                "revised": r["revised"][:120],
                                "method": r.get("method"),
                            }
                            for r in rewritten_records[:5]
                        ],
                        "unsupported_after_rewrite": int(report.get("unsupported") or 0),
                    }
                else:
                    state["claim_repairs"] = {
                        "strategy": "section_rewrite_required",
                        "removed_count": 0,
                        "unsupported_count": len(unsupported_before),
                        "safe_removal_limit": safe_removal_limit,
                        "unsupported_after_rewrite": int(new_report.get("unsupported") or 0),
                        "reason": "改写未减少未支持主张，且批量删除会破坏段落完整性和最低引用覆盖量",
                    }
            else:
                state["claim_repairs"] = {
                    "strategy": "section_rewrite_required",
                    "removed_count": 0,
                    "unsupported_count": len(unsupported_before),
                    "safe_removal_limit": safe_removal_limit,
                    "reason": "批量删除会破坏段落完整性和最低引用覆盖量",
                }
        state["claim_verification"] = report
        state["evidence_quality_report"] = {
            "evidence_summary": report.get("evidence_summary") or {},
            "limitations": report.get("evidence_limitations") or [],
        }
        support_rate = float(report.get("support_rate") or 0.0)
        unsupported = int(report.get("unsupported") or 0)

        paper_cards = state.get("paper_cards") or []
        abstract_or_meta_count = sum(
            1 for card in paper_cards
            if (card.get("evidence_state") or {}).get("access_level") in ("abstract", "metadata_only", "title_and_keywords")
            or str(card.get("evidence_source") or "metadata") in ("metadata", "abstract")
        )
        policy = get_review_threshold_policy()
        is_abstract_dominant = len(paper_cards) > 0 and (abstract_or_meta_count / len(paper_cards)) >= policy.synthesis_abstract_dominance
        quality_passed_threshold = (policy.synthesis_abstract_support_rate
                                    if is_abstract_dominant
                                    else policy.synthesis_fulltext_support_rate)

        state["generation_quality"] = {
            "passed": ((bool(report.get("valid")) or unsupported == 0) and support_rate >= quality_passed_threshold) or support_rate >= policy.synthesis_fulltext_support_rate,
            "support_rate": support_rate,
            "unsupported_claims": unsupported,
            "threshold": quality_passed_threshold,
            "reason": (
                "claim_evidence_check_passed"
                if (((bool(report.get("valid")) or unsupported == 0) and support_rate >= quality_passed_threshold) or support_rate >= policy.synthesis_fulltext_support_rate)
                else "unsupported_or_weakly_supported_claims"
            ),
        }
        append_step(
            state,
            "verify_claims",
            "success",
            tool_name="verify_claims",
            input_data={
                "text_length": len(generated_text),
                "evidence_cards": len(state.get("paper_cards") or []),
            },
            output_data={
                "threshold_policy": policy.snapshot(),
                **{
                    key: report.get(key)
                    for key in (
                        "valid",
                        "total_sentences",
                        "factual_claims",
                        "supported",
                        "partially_supported",
                        "unsupported",
                        "support_rate",
                    )
                },
                "claim_repairs": state.get("claim_repairs") or {},
                "entailment_cache_stats": report.get("entailment_cache_stats") or {},
                "verification_scope": report.get("verification_scope") or {},
            },
            duration_ms=int((time.time() - t0) * 1000),
        )
    except Exception as e:
        from app.agent.exceptions import DegradableAgentError

        error = DegradableAgentError(
            str(e), step="verify_claims", degraded_mode="skip_verification", original_error=e,
        )
        logger.error("verify_claims_node failed: %s", error.message)
        state.setdefault("errors", []).append(f"verify_claims: {e}")
        append_step(state, "verify_claims", "failed", error=str(e))
    return state


def _update_state_review_text(state: "ResearchAgentState", text: str) -> None:
    if state.get("review"):
        state["review"] = text
    elif state.get("related_work"):
        state["related_work"] = text
    elif state.get("introduction"):
        state["introduction"] = text
    state["body"] = text


def _rewrite_and_weaken_unsupported_claims(
    text: str,
    unsupported_claims: list[dict[str, Any]],
    paper_cards: list[dict[str, Any]],
    llm=None,
) -> tuple[str, list[dict[str, Any]]]:
    """对不受支持的事实句执行定向弱化与重写，使其符合可用证据范围并保留引用。"""
    cards_by_id = {
        str(card.get("paper_id") or ""): card
        for card in paper_cards
        if card.get("paper_id")
    }
    rendered = str(text or "")
    rewritten_records: list[dict[str, Any]] = []

    # 1. 尝试 LLM 批量改写
    if llm is not None and unsupported_claims:
        batch_payload = []
        for claim in unsupported_claims:
            sentence = str(claim.get("sentence") or "").strip()
            if not sentence or sentence not in rendered:
                continue
            citations = claim.get("citations") or []
            card_summaries = []
            for cid in citations:
                card = cards_by_id.get(cid) or {}
                card_summaries.append({
                    "paper_id": cid,
                    "title": card.get("title") or "",
                    "method": card.get("method") or "",
                    "research_problem": card.get("research_problem") or "",
                    "findings": card.get("results") or card.get("contributions") or "",
                })
            batch_payload.append({
                "claim_id": claim.get("claim_id"),
                "sentence": sentence,
                "citations": citations,
                "issues": claim.get("issues") or [],
                "evidence_context": card_summaries,
            })

        from app.prompt.claim_repair import UNSUPPORTED_CLAIM_REWRITE_PROMPT

        for start in range(0, len(batch_payload), 8):
            batch = batch_payload[start:start + 8]
            prompt = UNSUPPORTED_CLAIM_REWRITE_PROMPT.format(
                batch_json=json.dumps(batch, ensure_ascii=False, indent=2),
            )
            try:
                from app.core.json_utils import parse_json_object
                from app.core.config import get_settings
                from app.core.citation_syntax import extract_citation_ids
                from app.tools.validate_deliverable import find_similar_sentence

                response = llm.complete(
                    prompt,
                    response_format="json_object",
                    temperature=0.1,
                    timeout=get_settings().llm_control_plane_timeout,
                    operation="rewrite_unsupported_claims",
                )
                data = parse_json_object(response if isinstance(response, str) else str(response))
                for item in data.get("results") or []:
                    if not isinstance(item, dict):
                        continue
                    orig = str(item.get("original") or "").strip()
                    rev = str(item.get("revised") or "").strip()
                    claim_id = str(item.get("claim_id") or "")
                    matching_claim = next((c for c in batch if c["claim_id"] == claim_id or c["sentence"] == orig), None)
                    if not matching_claim:
                        continue
                    target_orig = matching_claim["sentence"]
                    if not (rev and target_orig in rendered):
                        continue
                    # WHY: 原判据 req_cits <= rev_cits 在原句没有引用时恒真，
                    # 改写可以凭空补 [1][2] 把无证据句伪装成有据句。引用集合
                    # 必须完全一致：不丢也不增。
                    req_cits = set(matching_claim["citations"])
                    rev_cits = set(extract_citation_ids(rev))
                    if req_cits != rev_cits:
                        logger.warning(
                            "claim rewrite rejected: citation set changed %s -> %s",
                            sorted(req_cits), sorted(rev_cits),
                        )
                        continue
                    # WHY: 改写曾把多句压成同一结论，正文出现逐字重复段落。
                    # 与终审 validate_final_review_integrity 共用同一判据，
                    # 在采纳前拦下，避免整批改写完才被终审否决。
                    if find_similar_sentence(rev, rendered, exclude=target_orig):
                        logger.warning(
                            "claim rewrite rejected: duplicates existing sentence (%s)",
                            claim_id or target_orig[:40],
                        )
                        continue
                    rendered = rendered.replace(target_orig, rev, 1)
                    rewritten_records.append({
                        "claim_id": claim_id,
                        "original": target_orig,
                        "revised": rev,
                        "method": "llm_rewrite",
                    })
            except Exception as exc:
                logger.warning("LLM claim rewrite batch failed: %s", exc)

    # 2. 规则级弱化兜底
    from app.tools.verify_claims import _weaken_strong_language
    for claim in unsupported_claims:
        sentence = str(claim.get("sentence") or "").strip()
        if not sentence or sentence not in rendered:
            continue
        if any(r["original"] == sentence for r in rewritten_records):
            continue
        weakened = _weaken_strong_language(sentence)
        if "numeric_value_not_found_in_evidence" in (claim.get("issues") or []):
            weakened = re.sub(r"\d+(?:\.\d+)?%", "一定幅度", weakened)
            weakened = re.sub(r"达到\s*\d+(?:\.\d+)?", "取得良好表现", weakened)
            weakened = re.sub(r"提升\s*\d+(?:\.\d+)?", "显著提升", weakened)
            weakened = _weaken_strong_language(weakened)
        if weakened != sentence and sentence in rendered:
            rendered = rendered.replace(sentence, weakened, 1)
            rewritten_records.append({
                "claim_id": str(claim.get("claim_id") or ""),
                "original": sentence,
                "revised": weakened,
                "method": "rule_weakening",
            })

    return rendered, rewritten_records


def _remove_unsupported_claims(
    text: str,
    unsupported_claims: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """隔离机器判定为不受支持的事实句，避免继续进入正式引用阶段。"""
    rendered = str(text or "")
    removed: list[dict[str, Any]] = []
    for claim in unsupported_claims:
        sentence = str(claim.get("sentence") or "").strip()
        if not sentence or sentence not in rendered:
            continue
        rendered = rendered.replace(sentence, "", 1)
        removed.append(claim)
    rendered = re.sub(r"[ \t]+\n", "\n", rendered)
    rendered = re.sub(r"\n{3,}", "\n\n", rendered).strip()
    # 若一节的全部事实句均被隔离，删除空章节；运行原因只进入
    # claim_repairs/quality_gate，不能以“证据不足”代理语言进入正文。
    rendered = re.sub(
        r"(?ms)(^#{2,3}\s+[^\n]+)\s*(?=^#{2,3}\s+|\Z)",
        "",
        rendered,
    ).strip()
    return rendered, removed

