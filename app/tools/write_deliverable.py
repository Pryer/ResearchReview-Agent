"""统一 SectionWriter 调度分发器：只消费 WritingPlan 中授权的论文与声明。"""

from __future__ import annotations

import json
import hashlib
import logging
from typing import Any

from app.core.text_quality import (
    AGENT_PROCESS_LANGUAGE_RE,
    strip_evidence_meta_language,
)
from app.deliverables.renderers import (
    get_renderer,
    _allocated_paper_ids,
    _claim_constraints_for_title,
    _heading,
    _split_planned_sections,
    _survey_papers,
    _validate_rewritten_section,
)
from app.schemas.deliverable_schema import WritingPlan

_AGENT_PROCESS_LANGUAGE_RE = AGENT_PROCESS_LANGUAGE_RE
_strip_evidence_meta_language = strip_evidence_meta_language

logger = logging.getLogger(__name__)


def _section_checkpoint_key(plan: WritingPlan, section_id: str) -> str:
    return f"{plan.deliverable_type.value}:{section_id}"


def _section_input_fingerprint(
    plan: WritingPlan,
    section,
    state: dict[str, Any],
) -> str:
    allocation = next((
        item for item in (state.get("citation_allocation_plan") or {}).get("sections") or []
        if str(item.get("section_id") or "") == section.id
    ), {})
    section_paper_ids = {
        *[str(value) for value in section.supporting_paper_ids],
        *[str(value) for value in allocation.get("paper_ids") or []],
    }
    cards = {
        str(card.get("paper_id") or ""): {
            "access": (card.get("evidence_state") or {}).get("access_level")
            or card.get("evidence_source"),
            "quality": card.get("quality_status"),
            "evidence": {
                "field_evidence": card.get("field_evidence") or {},
                "field_claims": card.get("field_claims") or {},
                "evidence_spans": card.get("evidence_spans") or [],
            },
        }
        for card in state.get("paper_cards") or []
        if str(card.get("paper_id") or "") in section_paper_ids
    }
    projected, synthesis = _writer_inputs(plan, state)
    # WHY: 确定性证据草稿无需模型，复用前即可构建；授权/综合变化不能复用旧正文。
    draft = get_renderer(plan.deliverable_type).render_fallback(
        plan, {**state, "theme_synthesis": synthesis}, projected
    )
    from app.deliverables.renderers.base_renderer import _requires_cross_route_synthesis
    semantic = state.get("research_semantic_frame") or {}
    payload = {
        "deliverable": plan.deliverable_type.value,
        "citation_policy": plan.citation_policy or {},
        "section": section.model_dump(mode="json"),
        "allocation": allocation,
        "cards": cards,
        "screening": sorted(
            (
                str(item.get("paper_id") or ""),
                str(item.get("_screening_decision") or ""),
            )
            for item in state.get("paper_details") or []
            if str(item.get("paper_id") or "") in cards
        ),
        # WHY: 逐项列出真正进入本节提示词的输入，替代原先的全局证据快照。
        # 全局快照既过宽（新增一篇无关论文就废掉所有章节检查点），又覆盖不足
        # （claim 约束、主题、研究重点都不在其中）。收窄的同时必须补齐这些
        # 实际输入，否则会从"过度失效"变成"失效不足"，复用未重验的旧正文。
        "routes": sorted(
            (
                str(route.get("route_id") or ""),
                tuple(sorted(
                    paper_id
                    for paper_id in (
                        str(value) for value in route.get("core_paper_ids") or []
                    )
                    if paper_id in section_paper_ids
                )),
            )
            for route in state.get("validated_routes") or []
            if route.get("route_id")
            and {str(value) for value in route.get("core_paper_ids") or []}
            & section_paper_ids
        ),
        # 综述论文清单注入每一节提示词，是唯一的合法全局输入。
        "fingerprint_schema": 2,
        "survey_papers": _survey_papers(projected),
        "projected_cards": [card for card in projected if str(card.get("paper_id")) in section_paper_ids],
        "original": _split_planned_sections(draft, plan).get(section.id, ""),
        "require_cross_route_synthesis": _requires_cross_route_synthesis(plan, section.id),
        "topic": str(state.get("canonical_topic") or state.get("topic") or ""),
        "scope": str((state.get("selected_scope") or {}).get("description") or ""),
        "focuses": [str(item) for item in semantic.get("required_focuses") or []],
        "evidence_roles": [
            str(item.get("label") or "")
            for item in semantic.get("evidence_requirements") or []
            if str(item.get("label") or "").strip()
        ],
        "claim_constraints": _claim_constraints_for_title(
            state.get("claim_plans"), section.title
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:20]


def _reusable_checkpoint_sections(
    plan: WritingPlan,
    state: dict[str, Any],
) -> dict[str, str]:
    """返回本轮无需生成且依赖指纹仍匹配的已验证章节。"""
    target_ids = {str(value) for value in state.get("target_section_ids") or [] if value}
    if not target_ids:
        return {}
    verified = state.get("section_checkpoints") or {}
    reusable: dict[str, str] = {}
    for section in plan.sections:
        if section.id in target_ids:
            continue
        checkpoint = verified.get(_section_checkpoint_key(plan, section.id)) or {}
        if (
            checkpoint.get("status") not in {"validated", "writer_validated", "reused"}
            or checkpoint.get("input_fingerprint")
            != _section_input_fingerprint(plan, section, state)
            or not str(checkpoint.get("text") or "").strip()
        ):
            continue
        reusable[section.id] = str(checkpoint["text"])
    return reusable


def _reuse_and_checkpoint_sections(
    text: str,
    plan: WritingPlan,
    state: dict[str, Any],
) -> str:
    """复用未受影响的已验证章节，并保存本轮章节候选。"""
    sections = _split_planned_sections(text, plan)
    if len(sections) != len(plan.sections):
        return text
    target_ids = {str(value) for value in state.get("target_section_ids") or [] if value}
    verified = state.get("section_checkpoints") or {}
    candidates = dict(state.get("section_candidate_checkpoints") or {})
    merged: list[str] = []
    for section in plan.sections:
        key = _section_checkpoint_key(plan, section.id)
        fingerprint = _section_input_fingerprint(plan, section, state)
        current = sections.get(section.id, "")
        previous = verified.get(key) or {}
        reused = bool(
            target_ids
            and section.id not in target_ids
            and previous.get("status") in {"validated", "writer_validated", "reused"}
            and previous.get("input_fingerprint") == fingerprint
            and str(previous.get("text") or "").strip()
        )
        selected = str(previous.get("text")) if reused else current
        errors = _validate_rewritten_section(
            selected,
            section.title,
            [str(value) for value in (
                next((
                    item.get("paper_ids") or []
                    for item in (state.get("citation_allocation_plan") or {}).get("sections") or []
                    if str(item.get("section_id") or "") == section.id
                ), section.supporting_paper_ids)
            )],
            section.heading_level or 2,
        )
        candidates[key] = {
            "deliverable_type": plan.deliverable_type.value,
            "section_id": section.id,
            "text": selected,
            "input_fingerprint": fingerprint,
            "writing_version": int(state.get("writing_version") or 1),
            "status": "reused" if reused else "candidate",
            "local_valid": not errors,
            "errors": errors,
        }
        merged.append(selected)
    state["section_candidate_checkpoints"] = candidates
    return "\n\n".join(item for item in merged if item.strip())


def promote_section_checkpoints(
    state: dict[str, Any],
    plan: WritingPlan | dict[str, Any],
    validation: dict[str, Any],
) -> None:
    """仅提升同时通过局部、证据密度与跨句重复检查的章节候选。"""
    plan_obj = plan if isinstance(plan, WritingPlan) else WritingPlan.model_validate(plan)
    candidates = state.get("section_candidate_checkpoints") or {}
    verified = dict(state.get("section_checkpoints") or {})
    metrics = validation.get("metrics") or {}
    floor_by_section = {
        str(item.get("section_id") or ""): str(item.get("status") or "")
        for item in metrics.get("section_evidence_floors") or []
        if item.get("section_id")
    }
    duplicate_fragments = [
        str(sample.get(key) or "").strip()
        for sample in metrics.get("duplicate_sentence_samples") or []
        if isinstance(sample, dict)
        for key in ("first", "duplicate")
        if str(sample.get(key) or "").strip()
    ]
    duplicate_section_ids = {
        str(value) for value in metrics.get("duplicate_section_ids") or [] if value
    }
    for section in plan_obj.sections:
        key = _section_checkpoint_key(plan_obj, section.id)
        candidate = candidates.get(key)
        if not candidate or candidate.get("local_valid") is not True:
            continue
        if floor_by_section.get(section.id) not in {None, "ok"}:
            continue
        candidate_text = str(candidate.get("text") or "")
        if section.id in duplicate_section_ids or any(
            fragment in candidate_text for fragment in duplicate_fragments
        ):
            continue
        verified[key] = {
            **candidate,
            # WHY: 全篇可能因其他章节失败；当前章节已通过可归因的局部门禁时
            # 仍可事务式提交，后续复用后会再次执行完整交付物门禁。
            "status": "validated",
        }
    state["section_checkpoints"] = verified


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


def _writer_inputs(plan: WritingPlan, state: dict[str, Any]):
    """指纹和写作共用授权投影，禁止两套过滤逻辑漂移。"""
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
    # 引用分配可能把证据覆盖计划中的论文补到宽泛概述节；这些论文必须
    # 进入本轮 Writer 输入，否则分配层虽已授权，SectionWriter 却会因
    # WritingPlan 的原始 supporting 集合过窄而把它们过滤掉。
    allowed_paper_ids.update(allocated_paper_ids)
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

    return cards, safe_synthesis


def write_deliverable(
    plan: WritingPlan | dict[str, Any],
    state: dict[str, Any],
    llm=None,
) -> str:
    """交付物主写作入口，按 deliverable_type 委派给独立渲染器。"""
    plan = plan if isinstance(plan, WritingPlan) else WritingPlan.model_validate(plan)
    cards, safe_synthesis = _writer_inputs(plan, state)
    renderer = get_renderer(plan.deliverable_type)
    reusable_sections = _reusable_checkpoint_sections(plan, state)
    sentinel = object()
    previous_reusable = state.get("_reusable_section_texts", sentinel)
    if reusable_sections:
        # WHY: renderer 仍需完整 WritingPlan 维持章节顺序和跨路线语义，但逐节
        # 写手必须看到可复用集合，才能完全跳过未受影响章节的模型调用。
        state["_reusable_section_texts"] = reusable_sections
    try:
        text = renderer.render(
            plan=plan,
            state=state,
            cards=cards,
            safe_synthesis=safe_synthesis,
            llm=llm,
        )
    finally:
        if previous_reusable is sentinel:
            state.pop("_reusable_section_texts", None)
        else:
            state["_reusable_section_texts"] = previous_reusable
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

    final_text = break_citation_dumps(str(text or ""))
    return _reuse_and_checkpoint_sections(final_text, plan, state)
