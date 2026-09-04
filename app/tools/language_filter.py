"""中英文分支独立硬过滤。

提供 ``evaluate_language_hard_filter``，根据论文语言选择对应的
协议术语（terms_zh / terms_en）进行硬条件匹配。
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from app.tools.language_router import detect_term_language
from app.tools.paper_matching import (
    compile_scope,
    excluded_term_matches_title,
    hard_anchor_matches_haystack,
)


def _get_criterion_terms(
    criterion: Dict[str, Any],
    language: str,
) -> list[str]:
    """从 criterion 中提取对应语言的 term 列表。

    优先取 ``terms_zh`` / ``terms_en``；
    回退到 ``terms`` 时**按语言过滤**，避免中文 term 误入英文分支。
    """
    # 1. 语言专用字段
    if language == "zh":
        specific = criterion.get("terms_zh") or []
        if specific:
            return [str(t).strip() for t in specific if str(t).strip()]
    elif language == "en":
        specific = criterion.get("terms_en") or []
        if specific:
            return [str(t).strip() for t in specific if str(t).strip()]

    # 2. 回退到通用 terms，但只取匹配目标语言的 term
    generic_terms = criterion.get("terms") or []
    if not generic_terms:
        return []

    matched: list[str] = []
    for term in generic_terms:
        term_str = str(term).strip()
        if not term_str:
            continue
        term_lang = detect_term_language(term_str)
        if term_lang == language or term_lang == "mixed":
            matched.append(term_str)

    return matched


def evaluate_language_hard_filter(
    paper: Dict[str, Any],
    screening_protocol: Dict[str, Any] | None,
    language: str,
    compiled_scope: Dict[str, Any] | None = None,
) -> Tuple[bool, str]:
    """对单篇论文执行对应语言分支的硬条件检查。

    与 ``evaluate_screening_protocol_hard_filter`` 逻辑一致，
    区别仅在于使用 ``terms_zh`` / ``terms_en`` 替代通用 ``terms``。

    Args:
        paper: 单篇论文字典。
        screening_protocol: 上下文筛选协议。
        language: ``"zh"`` 或 ``"en"``。

    Returns:
        ``(passed, reason)`` 元组。
    """
    if compiled_scope is None:
        compiled_scope = compile_scope(screening_protocol=screening_protocol)
    protocol = screening_protocol or {}
    if compiled_scope is not None:
        criteria = []
        for group in compiled_scope.get("groups") or []:
            if group.get("role") != "hard_include_criteria":
                continue
            terms_by_language = group.get("terms_by_language") or {}
            terms = terms_by_language.get(language) or group.get("languages", {}).get(language) or []
            if terms:
                criteria.append({
                    "terms": terms,
                    "source": group.get("source_name"),
                    "applies_to_each_paper": group.get("applies_to_each_paper", True),
                    "label": group.get("label"),
                })
        protocol = {
            "hard_exclude_title_terms": [
                str(value) for value in (compiled_scope.get("aliases") or {}).get("protocol_exclude") or []
            ],
            "hard_include_criteria": criteria,
        }
    if not protocol:
        return True, "未提供上下文筛选协议"

    title = str(paper.get("title") or "")
    haystack = " ".join([
        title,
        str(paper.get("abstract") or ""),
        str(paper.get("venue") or ""),
    ])

    # 硬排除：仅按标题严格短语匹配
    for term in protocol.get("hard_exclude_title_terms") or []:
        if excluded_term_matches_title(str(term), title):
            return False, f"命中用户明确排除词: {term}"

    enforced = 0
    for criterion in protocol.get("hard_include_criteria") or []:
        if not isinstance(criterion, dict):
            continue
        if not criterion.get("applies_to_each_paper"):
            continue
        if str(criterion.get("source") or "") not in {
            "user_explicit", "confirmed_scope",
        }:
            continue

        terms = _get_criterion_terms(criterion, language)
        if not terms:
            continue

        enforced += 1
        if not any(hard_anchor_matches_haystack(term, haystack) for term in terms):
            label = criterion.get("label") or criterion.get("criterion_id") or "硬条件"
            return False, f"[{language}] 未命中逐篇硬条件: {label}"

    return True, f"[{language}] 通过上下文硬条件 ({enforced})"
