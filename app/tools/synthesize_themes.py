"""基于声明级 PaperCard 生成跨论文主题综合，不在此阶段写正文。"""

from __future__ import annotations

import re
from typing import Any

from app.schemas.deliverable_schema import GapType, SynthesizedGap, ThemeSynthesis


def synthesize_themes(
    paper_cards: list[dict[str, Any]],
    taxonomy: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    taxonomy = taxonomy or {}
    cards = {str(card.get("paper_id") or ""): card for card in paper_cards if card.get("paper_id")}
    assignments: dict[str, list[str]] = {}
    for item in taxonomy.get("assignments") or []:
        assignments.setdefault(str(item.get("primary_theme_id") or ""), []).append(str(item.get("paper_id") or ""))
    themes = taxonomy.get("themes") or []
    if not themes and cards:
        themes = [{"theme_id": "T1", "name": "当前研究证据", "description": "尚未形成稳定的细分路线"}]
        assignments = {"T1": list(cards)}

    result: list[dict[str, Any]] = []
    for theme in themes:
        theme_id = str(theme.get("theme_id") or "")
        paper_ids = [paper_id for paper_id in assignments.get(theme_id, []) if paper_id in cards]
        if not paper_ids:
            continue
        field_claims = {
            field: _claims_for(field, paper_ids, cards)
            for field in ("research_problem", "method", "results", "limitations")
        }
        shared_problems = _shared_claims(field_claims["research_problem"])
        shared_methods = _shared_claims(field_claims["method"])
        dimensions = [
            field for field in ("research_problem", "method", "data_modalities", "dataset", "metrics", "study_design")
            if sum(bool(cards[paper_id].get(field)) for paper_id in paper_ids) >= min(2, len(paper_ids))
        ]
        gaps: list[SynthesizedGap] = []
        if sum("limitations" in (cards[paper_id].get("unsupported_fields") or []) for paper_id in paper_ids) > len(paper_ids) / 2:
            # 面向读者的表述：只陈述"文献未报告局限"这一文献事实，既不暴露
            # "作者局限字段不可访问"这类内部证据获取状态，也不使用
            # AGENT_PROCESS_LANGUAGE_RE 所禁止的"证据尚不足以"式元评价。
            gaps.append(SynthesizedGap(
                gap_type=GapType.EVIDENCE_ACCESS_LIMITATION,
                statement="该路线多数文献未明确报告方法的适用边界与局限，本节不对其局限性作跨文献综合。",
                supporting_paper_ids=paper_ids,
            ))
        result.append(ThemeSynthesis(
            theme_id=theme_id,
            theme_name=str(theme.get("name") or "未命名研究路线"),
            paper_ids=paper_ids,
            reported_problems=field_claims["research_problem"],
            reported_methods=field_claims["method"],
            shared_problems=shared_problems,
            shared_methods=shared_methods,
            # 旧字段仅保留可证明由多篇独立论文共同报告的声明。
            common_problems=shared_problems,
            common_methods=shared_methods,
            reported_findings=field_claims["results"],
            author_stated_limitations=field_claims["limitations"],
            synthesized_gaps=[item.model_dump(mode="json") for item in gaps],
            comparison_dimensions=dimensions,
        ).model_dump(mode="json"))
    return result


def _shared_claims(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """返回至少由两篇独立论文逐字/规范化一致报告的声明。

    更宽松的语义合并必须由上游 LLM 综合并再次验证；这里不使用词面相似度把
    相关但不同的主张冒充为“共同结论”。
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for claim in claims:
        text = str(claim.get("claim") or "").strip()
        key = re.sub(r"\W+", "", text.casefold())[:160]
        if key:
            grouped.setdefault(key, []).append(claim)
    shared: list[dict[str, Any]] = []
    for group in grouped.values():
        paper_ids = {str(item.get("paper_id") or "") for item in group}
        if len(paper_ids) >= 2:
            representative = dict(group[0])
            representative["supporting_paper_ids"] = sorted(paper_ids)
            shared.append(representative)
    return shared


def _claims_for(field: str, paper_ids: list[str], cards: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for paper_id in paper_ids:
        for claim in (cards[paper_id].get("field_claims") or {}).get(field) or []:
            if not claim.get("explicitly_reported"):
                continue
            text = str(claim.get("claim") or "").strip()
            key = re.sub(r"\W+", "", text.lower())[:160]
            if not text or key in seen:
                continue
            seen.add(key)
            output.append({
                "claim_id": str(claim.get("evidence_id") or f"{paper_id}:{field}"),
                "paper_id": paper_id,
                "claim": text,
                "source_text": str(claim.get("source_text") or ""),
                "source_section": claim.get("source_section"),
                "access_level": claim.get("evidence_level"),
            })
    return output


def build_search_report(state: dict[str, Any]) -> dict[str, Any]:
    """只报告运行时真实执行并记录的检索行为。"""
    existing = state.get("search_report") or {}
    if existing and not (state.get("searched_keywords") or state.get("source_diagnostics")):
        return {
            **existing,
            "writing_pool_count": len(state.get("paper_cards") or []),
        }
    diagnostics = [
        item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)
        for item in (state.get("source_diagnostics") or [])
    ]
    return {
        "queries": list(state.get("searched_keywords") or state.get("keywords") or []),
        "sources": [str(item.get("source") or "") for item in diagnostics if item.get("source")],
        "start_year": state.get("start_year"),
        "end_year": state.get("end_year"),
        "candidate_count": len(state.get("candidate_papers") or []),
        "screened_count": len(state.get("ranked_papers") or []),
        "writing_pool_count": len(state.get("paper_cards") or []),
        "source_diagnostics": diagnostics,
        "performed_steps": [str(step.get("step_name") or "") for step in state.get("steps") or []],
        "not_performed": [
            "双人独立筛选", "PRISMA流程", "偏倚风险评价", "人工全文质量评价"
        ],
    }
