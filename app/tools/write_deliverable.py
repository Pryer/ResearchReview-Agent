"""统一 SectionWriter 调度分发器：只消费 WritingPlan 中授权的论文与声明。"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.core.text_quality import (
    AGENT_PROCESS_LANGUAGE_RE,
    strip_evidence_meta_language,
)
from app.deliverables.renderers import (
    get_renderer,
    _allocated_paper_ids,
    _heading,
)
from app.schemas.deliverable_schema import WritingPlan

_AGENT_PROCESS_LANGUAGE_RE = AGENT_PROCESS_LANGUAGE_RE
_strip_evidence_meta_language = strip_evidence_meta_language

logger = logging.getLogger(__name__)


def __getattr__(name: str):
    """兼容旧的常量导入，同时保持工具模块本身不加载写作 Prompt。"""
    if name == "WRITER_PROMPT":
        from app.prompt.writing.deliverable import WRITER_PROMPT

        return WRITER_PROMPT
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _fallback_writer(
    plan: WritingPlan | dict[str, Any],
    state: dict[str, Any],
    cards: list[dict[str, Any]],
) -> str:
    """按交付物类型回退到确定性渲染器生成。"""
    plan_obj = plan if isinstance(plan, WritingPlan) else WritingPlan.model_validate(plan)
    renderer = get_renderer(plan_obj.deliverable_type)
    return renderer.render_fallback(plan_obj, state, cards)


def write_deliverable(
    plan: WritingPlan | dict[str, Any],
    state: dict[str, Any],
    llm=None,
) -> str:
    """交付物主写作入口，按 deliverable_type 委派给独立渲染器。"""
    plan = plan if isinstance(plan, WritingPlan) else WritingPlan.model_validate(plan)
    allowed_paper_ids = {
        paper_id for section in plan.sections for paper_id in section.supporting_paper_ids
    }
    allowed_claim_ids = {
        claim_id for section in plan.sections for claim_id in section.supporting_claim_ids
    }
    allowed_theme_names = {
        section.title for section in plan.sections if section.id.startswith("theme_")
    }
    allocated_paper_ids = set(_allocated_paper_ids(state))
    prompt_paper_ids = (
        allowed_paper_ids & allocated_paper_ids
        if allocated_paper_ids
        else allowed_paper_ids
    )
    cards = []
    for card in state.get("paper_cards") or []:
        paper_id = str(card.get("paper_id") or "")
        if paper_id not in prompt_paper_ids:
            continue
        claims = []
        for field, field_claims in (card.get("field_claims") or {}).items():
            for claim in field_claims:
                evidence_id = str(claim.get("evidence_id") or "")
                if claim.get("explicitly_reported") and (
                    not allowed_claim_ids or evidence_id in allowed_claim_ids
                ):
                    claims.append({"field": field, **claim})
        cards.append({
            "paper_id": paper_id,
            "title": card.get("title"),
            # WHY: 提示词要求点名第一作者，因此作者必须随卡片一起传入；否则
            # 模型只能从题名猜测，会写出 [56] 那类错误作者。
            "authors": [
                str(author).strip()
                for author in (card.get("authors") or [])
                if str(author).strip()
            ],
            "year": card.get("year"),
            "venue": card.get("venue"),
            "doi": card.get("doi"),
            "publication_type": card.get("publication_type"),
            "publication_status": card.get("publication_status") or "unknown",
            "peer_review_status": card.get("peer_review_status") or "unknown",
            "access_level": (card.get("evidence_state") or {}).get("access_level"),
            "unsupported_fields": card.get("unsupported_fields") or [],
            "evidence_role": card.get("evidence_role", "method"),
            "screening_decision": (
                card.get("_screening_decision")
                or card.get("screening_decision")
                or ""
            ),
            "claims": claims,
        })
    safe_synthesis = []
    for synthesis in state.get("theme_synthesis") or []:
        if str(synthesis.get("theme_name") or "") not in allowed_theme_names:
            continue
        sanitized = dict(synthesis)
        sanitized["paper_ids"] = [
            paper_id for paper_id in synthesis.get("paper_ids") or []
            if str(paper_id) in prompt_paper_ids
        ]
        for field in (
            "reported_problems", "reported_methods", "shared_problems", "shared_methods",
            "common_problems", "common_methods", "reported_findings", "author_stated_limitations"
        ):
            sanitized[field] = [
                claim for claim in synthesis.get(field) or []
                if str(claim.get("paper_id") or "") in prompt_paper_ids
                and (not allowed_claim_ids or str(claim.get("claim_id") or "") in allowed_claim_ids)
            ]
        safe_synthesis.append(sanitized)
    existing_safe_names = {
        str(item.get("theme_name") or "") for item in safe_synthesis
    }
    for section in plan.sections:
        if not section.id.startswith("theme_") or section.title in existing_safe_names:
            continue
        allowed_ids = set(section.supporting_paper_ids) & prompt_paper_ids
        members = [
            item for item in state.get("theme_synthesis") or []
            if allowed_ids & {str(paper_id) for paper_id in item.get("paper_ids") or []}
        ]
        if not members:
            continue
        combined: dict[str, Any] = {
            "theme_id": section.id.removeprefix("theme_"),
            "theme_name": section.title,
            "paper_ids": [
                paper_id for paper_id in section.supporting_paper_ids
                if paper_id in prompt_paper_ids
            ],
            "comparison_dimensions": list(dict.fromkeys(
                str(value)
                for item in members
                for value in item.get("comparison_dimensions") or []
                if value
            )),
        }
        for field in (
            "reported_problems", "reported_methods", "shared_problems", "shared_methods",
            "common_problems", "common_methods", "reported_findings",
            "author_stated_limitations", "synthesized_gaps",
        ):
            combined[field] = [
                claim
                for item in members
                for claim in item.get(field) or []
                if not isinstance(claim, dict)
                or not claim.get("paper_id")
                or str(claim.get("paper_id")) in allowed_ids
            ]
        safe_synthesis.append(combined)

    renderer = get_renderer(plan.deliverable_type)
    text = renderer.render(
        plan=plan,
        state=state,
        cards=cards,
        safe_synthesis=safe_synthesis,
        llm=llm,
    )
    # 交付物级不删空：渲染器全链失败返回空文本时回退确定性渲染器，
    # 保证用户要求的章节（如"二、研究现状"）不会整章消失。
    if not str(text or "").strip():
        # 降级渲染同样只消费授权数据：theme_synthesis 必须替换为过滤后的
        # safe_synthesis（与 base_renderer.render 内部的 safe_state 构造
        # 一致），否则 claim 门禁未放行的综合条目会经原始 state 泄漏进正文。
        text = renderer.render_fallback(
            plan,
            {**state, "theme_synthesis": safe_synthesis},
            cards,
        )
    # 最终防线：确定性拆散引用堆砌（如兜底模板遗留的 [pid][pid]… 连排），
    # 引用缺口由引用数量校验和最终质量门禁如实报告。
    from app.core.citation_density import break_citation_dumps

    return break_citation_dumps(str(text or ""))
