"""Faithful rewrite of a research request after user clarification."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from app.agent.slot_extractor import extract_slots
from app.core.config import get_settings
from app.core.json_utils import parse_json_object
from app.core.logger import get_logger
from app.prompt.research_query_rewrite import RESEARCH_QUERY_REWRITE_PROMPT
from app.schemas.research_plan_schema import ResearchQueryRewrite

logger = get_logger(__name__)


def rewrite_research_query(
    original_query: str,
    question: str,
    answer: str,
    *,
    llm: Any = None,
    confirmed_scope: str = "",
    prior_answers: list[str] | None = None,
) -> dict[str, Any]:
    """Return a grounded working query and auditable user source spans."""
    original = str(original_query or "").strip()
    response = str(answer or "").strip()
    if not original or not response:
        raise ValueError("研究请求与澄清回答均不能为空")
    # WHY: 用户输入 scope_id 时，确认的是可见选项名称；候选描述和排除词仍非用户原话。
    effective_answer = confirmed_scope.strip() if confirmed_scope else response
    prior = [str(item).strip() for item in prior_answers or [] if str(item).strip()]
    anchors = [
        {"role": "user", "turn": "original", "text": original},
        *({"role": "user", "turn": f"clarification_{index}", "text": item}
          for index, item in enumerate(prior, start=1)),
        {"role": "user", "turn": "clarification", "text": response},
    ]
    fingerprint = hashlib.sha256(json.dumps(
        [original, question, *prior, response, effective_answer], ensure_ascii=False,
    ).encode("utf-8")).hexdigest()
    fallback = original.rstrip("。 .") + "；用户澄清：" + "；".join([*prior, effective_answer])
    rewritten = fallback
    source = "conservative_merge"
    attempts = 0
    validation_issues: list[str] = []
    if llm is not None:
        try:
            prompt = RESEARCH_QUERY_REWRITE_PROMPT.format(
                original_query=original, question=question, answer=effective_answer,
                prior_answers_json=json.dumps(prior, ensure_ascii=False),
            )
            for attempt in range(2):
                attempts += 1
                raw = llm.complete(
                    prompt, response_format="text", temperature=0.0,
                    timeout=get_settings().llm_control_plane_timeout,
                    retry_empty=False, operation="research_query_rewrite",
                )
                candidate = str(parse_json_object(raw).get("rewritten_query") or "").strip()
                if _preserves_constraints(candidate, original, effective_answer, prior_answers=prior):
                    rewritten, source = candidate, "llm_validated"
                    break
                validation_issues.append("rewrite_constraints_not_preserved")
                # WHY: 校验失败只允许一次带反馈的修正，避免澄清入口形成无界 LLM 重试。
                prompt += "\n上次改写未保留用户显式约束，或加入了未授权排除项。请重新返回忠实改写。"
        except Exception as exc:
            logger.warning("Research query rewrite failed; preserving user text: %s", exc)
            validation_issues.append("rewrite_invocation_failed")
    if source == "conservative_merge" and not _preserves_constraints(
        fallback, original, effective_answer, prior_answers=prior,
    ):
        # WHY: 原请求与后续轮次冲突时简单拼接会让下游槽位解析重新读到旧篇数或年份；
        # 无法安全构造有效工作查询就停止本轮规划，不能带着相互矛盾的硬约束检索。
        raise ValueError("澄清后的研究请求未能保留最新用户硬约束，请重试改写")
    return ResearchQueryRewrite.model_validate({
        "rewritten_query": rewritten,
        "source": source,
        "source_anchors": anchors,
        "preserved_constraints": _explicit_constraints(original, response, prior_answers=prior),
        "suggested_expansions": [],
        "unresolved_ambiguities": [],
        "input_fingerprint": fingerprint,
        "version": 1,
        "attempts": attempts,
        "validation_issues": validation_issues,
    }).model_dump(mode="json")


def _explicit_constraints(
    original: str, answer: str, *, prior_answers: list[str] | None = None,
) -> list[dict[str, Any]]:
    from app.agent.slot_extractor import extract_requested_sections

    turns = [("original", original)] + [
        (f"clarification_{index}", text)
        for index, text in enumerate(prior_answers or [], start=1)
    ] + [("clarification", answer)]
    selected: dict[str, dict[str, Any]] = {}
    for turn, text in turns:
        slots = extract_slots(text, "generate_review")

        def set_value(field: str, value: Any) -> None:
            selected[field] = {
                "field": field, "value": value,
                "source_turn": turn, "source_text": text,
            }

        if slots.max_papers_explicit:
            set_value("required_reference_count", slots.required_reference_count)
        if slots.year_range_explicit:
            set_value("year_range", [slots.start_year, slots.end_year])
        sections = extract_requested_sections(text, "lookup")
        if sections:
            previous = selected.get("requested_sections", {}).get("value") or []
            merged = (
                sections if re.search(r"改为|只要|仅需", text)
                else list(dict.fromkeys([*previous, *sections]))
            )
            set_value("requested_sections", merged)
    return list(selected.values())


def _preserves_constraints(
    candidate: str, original: str, answer: str, *, prior_answers: list[str] | None = None,
) -> bool:
    if not candidate or len(candidate) > 4000:
        return False
    # WHY: 数量、时间及交付物由用户原文确定；改写模型不能用伪造数值覆盖硬约束。
    resulting = extract_slots(candidate, "generate_review")
    expected = {
        item["field"]: item["value"]
        for item in _explicit_constraints(original, answer, prior_answers=prior_answers)
    }
    required_count = expected.get("required_reference_count")
    if required_count is not None and (
        not resulting.max_papers_explicit or resulting.required_reference_count != required_count
    ):
        return False
    if required_count is None and resulting.max_papers_explicit:
        return False
    years = expected.get("year_range")
    if years is not None and [resulting.start_year, resulting.end_year] != years:
        return False
    if years is None and resulting.year_range_explicit:
        return False
    from app.agent.slot_extractor import extract_requested_sections

    requested = set(expected.get("requested_sections") or [])
    if requested and not requested.issubset(resulting.requested_sections or []):
        return False
    # WHY: 模型新造的排除条件会直接缩窄检索，即使数量槽位完全正确也必须拒绝。
    exclusion_pattern = r"(?:排除|不包括|不纳入|剔除)[^，。；;,.\n]{1,40}"
    source = "\n".join([original, *(prior_answers or []), answer])
    for exclusion in re.findall(exclusion_pattern, candidate):
        if exclusion not in source:
            return False
    return True
