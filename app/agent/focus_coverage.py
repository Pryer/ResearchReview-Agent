"""用户明确研究重点的直接证据覆盖与补检索。"""

from __future__ import annotations

import re
from typing import Any, Iterable

from app.agent.evidence_roles import evidence_coverage, paper_text
from app.agent.pipeline_stages import stage_probe_terms


def focus_aliases(focus: str) -> list[str]:
    """旧会话兼容：新会话优先使用语义帧内动态派生的 aliases。"""
    focus = str(focus or "").strip()
    return list(dict.fromkeys(filter(None, [focus, focus.replace("_", " ")])))


def required_focus_coverage(
    frame_or_focuses: dict[str, Any] | Iterable[str],
    papers: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    if isinstance(frame_or_focuses, dict) and frame_or_focuses.get("evidence_requirements"):
        return evidence_coverage(frame_or_focuses, papers)

    # 对没有 evidence_requirements 的旧会话保留保守兼容，不再为具体领域扩展别名。
    paper_list = list(papers)
    focuses = list(frame_or_focuses or []) if not isinstance(frame_or_focuses, dict) else list(
        frame_or_focuses.get("required_focuses") or []
    )
    counts: dict[str, int] = {}
    matched_ids: dict[str, list[str]] = {}
    required_counts: dict[str, int] = {}
    for raw_focus in focuses:
        focus = str(raw_focus or "").strip()
        if not focus:
            continue
        aliases = focus_aliases(focus)
        ids = [
            str(paper.get("paper_id") or paper.get("id") or "")
            for paper in paper_list
            if any(alias.lower() in paper_text(paper) for alias in aliases)
        ]
        ids = list(dict.fromkeys(item for item in ids if item))
        counts[focus] = len(ids)
        matched_ids[focus] = ids
        required_counts[focus] = 1
    missing = [focus for focus, count in counts.items() if count < required_counts[focus]]
    return {
        "ready": not missing,
        "counts": counts,
        "required_counts": required_counts,
        "matched_paper_ids": matched_ids,
        "requirement_labels": {focus: focus for focus in counts},
        "missing_requirement_ids": missing,
        "missing_focuses": missing,
    }


# 纯动词/过程性别名不能作为检索词：与主题拼接会生成“调研 近三年…综述”
# 这类动词前置的混杂查询，国际源返回 0 或噪声。综述/review 保留——
# “X 综述”是有效的检索式。
_GENERIC_FOCUS_ALIAS = {
    "调研", "检索", "搜索", "查询", "查找", "研究", "阅读", "梳理",
    "search", "research", "survey", "investigate", "find", "collect",
}


def supplemental_focus_queries(
    missing_focuses: Iterable[str],
    topic: str,
    semantic_frame: dict[str, Any] | None = None,
) -> list[str]:
    """为未满足的证据要求生成补检索式。

    每个缺口生成两类查询：

    1. ``主题 + 别名``：把缺口概念绑回主题，保证仍在用户范围内；
    2. ``主题 + 别名 + 阶段判据词``：阶段判据词取自该 requirement 所处研究链
       阶段的通用产物特征（如结构化阶段关注规则/粒度/一致性），用于把召回
       从上游产物推向缺失的下游产物。

    第 2 类是修复语义漂移的召回侧对应措施：仅靠 ``主题 + 别名`` 时，检索源
    往往仍返回上游感知论文——它们同样含有主题词。阶段判据词全部是跨学科的
    产物类型词，不含任何具体领域、方法或应用场景名称。
    """
    missing = {str(item) for item in missing_focuses if str(item).strip()}
    requirements = (semantic_frame or {}).get("evidence_requirements") or []
    topic_clean = str(topic or "").strip()
    queries: list[str] = []
    matched_any = False

    for requirement in requirements:
        requirement_id = str(requirement.get("requirement_id") or "")
        label = str(requirement.get("label") or "")
        if requirement_id not in missing and label not in missing:
            continue
        matched_any = True
        aliases = [
            str(item).strip() for item in requirement.get("aliases") or []
            if str(item).strip()
            and str(item).strip().lower() not in _GENERIC_FOCUS_ALIAS
        ]
        terms = aliases[:2] or (
            [label] if label and label.lower() not in _GENERIC_FOCUS_ALIAS else []
        )
        stage_terms = stage_probe_terms(requirement.get("evidence_role"))
        for term in terms:
            queries.append(" ".join(filter(None, (topic_clean, term))))
            # 别名与阶段判据词同为中文或同为英文时才拼接，避免生成
            # 中英混杂查询被 sanitize_search_keyword 丢弃。
            for probe in _stage_probes_matching_language(term, stage_terms):
                queries.append(" ".join(filter(None, (topic_clean, term, probe))))

    if not matched_any:
        queries.extend(
            " ".join(filter(None, (topic_clean, item)))
            for item in missing
            if item.lower() not in _GENERIC_FOCUS_ALIAS
        )
    return list(dict.fromkeys(query for query in queries if query.strip()))


def _stage_probes_matching_language(term: str, probes: dict[str, list[str]]) -> list[str]:
    """按别名语言选取同语言的阶段判据词，最多两个。

    与别名重复的判据词会被剔除：拼出「标注规范 标注规范」这类重复查询既无
    额外召回价值，也会挤掉一个有效查询名额。
    """
    is_cjk = bool(re.search(r"[\u4e00-\u9fff]", term))
    key = "zh" if is_cjk else "en"
    normalized_term = re.sub(r"\s+", "", term).casefold()
    candidates = [
        probe for probe in probes.get(key, [])
        if re.sub(r"\s+", "", probe).casefold() != normalized_term
    ]
    return candidates[:2]
