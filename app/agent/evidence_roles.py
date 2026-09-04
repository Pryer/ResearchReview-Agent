"""任务阶段的直接证据判定。

规则描述的是证据产物类型，而不是某个领域的固定章节：感知/识别结果、
结构化编码、用户点名的分析方法，以及下游解释分别判定，彼此不能替代。
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from app.utils.date_utils import current_year as get_current_year


# 时间限定词（近N年 / 近年来 / recent N years）只描述检索窗口，不是
# 研究对象或研究重点；把它们当实体会让覆盖检查去论文正文里词面匹配
# “近五年”，必然误报直接证据缺失（2026-08 少样本动作识别案例）。
_TEMPORAL_QUALIFIER_RE = re.compile(
    r"(?:最近?|近|过去|截至)\s*[一二三四五六七八九十百\d]+\s*(?:余|多)?\s*年"
    r"|近年(?:来|内)?"
    r"|(?:recent|last|past)\s+(?:\d+|five|ten|few)\s+years?"
    r"|recent\s+(?:research|studies|literature|progress)"
    r"|latest\s+(?:research|studies|literature)",
    re.I,
)
# 剥除时间词后再剥除的通用学术词：仅作无实体上下文时的兜底判据；
# 新会话的主判据是 is_scope_only_text 的实体映射，不依赖这份词表。
_GENERIC_ACADEMIC_WORD_RE = re.compile(
    r"[\s，。、,.;；:：的了的]+"
    r"|(?:文献|研究|调研|检索|搜索|论文|工作|成果|进展|综述|证据|资料)+"
    r"|\b(?:in|on|of|for|the|a|an|and|papers?|literature|studies|works?"
    r"|research|progress|review)\b",
    re.I,
)
# 从时间要求中解析窗口大小：默认“近年”= 5 年。
_TEMPORAL_WINDOW_RE = re.compile(
    r"(?:最近?|近|过去)\s*([一二三四五六七八九十\d]+)\s*年"
    r"|近年"
    r"|(?:recent|last|past)\s+(\d+|five|ten|few)\s+years?",
    re.I,
)
_CN_YEAR_NUM = {
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}


def is_temporal_qualifier_text(value: str) -> bool:
    """判定表面文本是否只是时间限定词：含时间词且剥除后无领域内容。

    “近五年文献证据”→ True；“近五年少样本动作识别”→ False（仍有领域内容）；
    “研究综述”不含时间词，恒为 False。

    这是无实体上下文（旧会话 requirement、实体表自指条目）时的词面兜底；
    语义帧可用时优先用 is_scope_only_text 的实体映射判据。
    """
    text = str(value or "").strip()
    if not text or not _TEMPORAL_QUALIFIER_RE.search(text):
        return False
    residual = _GENERIC_ACADEMIC_WORD_RE.sub("", _TEMPORAL_QUALIFIER_RE.sub("", text))
    return not residual


def has_temporal_qualifier(value: str) -> bool:
    return bool(_TEMPORAL_QUALIFIER_RE.search(str(value or "").strip()))


def strip_temporal_qualifiers(value: str) -> str:
    """移除文本中的时间限定表达，返回剩余文本（可能为空）。"""
    return _TEMPORAL_QUALIFIER_RE.sub("", str(value or "").strip())


def is_scope_only_text(value: str, domain_aliases: Iterable[str]) -> bool:
    """时间限定 + 残差不含任何已解析领域实体 → 检索范围/动作壳。

    判据是“label 去掉时间词后是否命中语义帧中的研究对象/方法/分析目标
    实体”，而不是枚举通用动词：残差没有领域实体支撑即按范围约束处理，
    因此“近五年文献调研证据 / 梳理证据 / 汇总证据”等任何动词变体都成立。
    无时间词的表述不属于本判据管辖（交由 source_ids 溯源校验）。
    """
    text = str(value or "").strip()
    if not text or not has_temporal_qualifier(text):
        return False
    residual = strip_temporal_qualifiers(text)
    if not residual:
        return True
    return not any(
        _contains_concept(residual, str(alias))
        for alias in domain_aliases
        if str(alias).strip()
    )


def temporal_requirement_window(
    requirement: dict[str, Any],
) -> tuple[int, int] | None:
    """把纯时间窗口要求解析为 ``(start_year, end_year)``；非时间要求返回 None。"""
    label = str(requirement.get("label") or "")
    if not is_temporal_qualifier_text(label):
        return None
    window_size = 5
    for source in (label, *(str(item) for item in requirement.get("aliases") or [])):
        match = _TEMPORAL_WINDOW_RE.search(str(source))
        if not match:
            continue
        cn_num, en_num = match.group(1), match.group(2)
        if cn_num:
            window_size = _CN_YEAR_NUM.get(cn_num) or (
                int(cn_num) if cn_num.isdigit() else 5
            )
        elif en_num:
            window_size = {"five": 5, "ten": 10, "few": 5}.get(
                en_num.lower(), int(en_num) if en_num.isdigit() else 5
            )
        break
    end_year = int(get_current_year())
    return (end_year - max(1, window_size) + 1, end_year)


_CODING_TERMS = (
    "behavior coding", "behaviour coding", "automated coding", "automatic coding",
    "行为编码", "自动编码", "观察编码",
)
_CODING_STRUCTURE_TERMS = (
    "coding scheme", "coding framework", "codebook", "coding rule",
    "annotation protocol", "annotation guideline", "temporal segmentation",
    "time window", "time unit", "event boundary", "onset", "offset",
    "inter-rater", "interrater", "agreement", "reliability", "validity",
    "structured observation", "behavior sequence", "behaviour sequence",
    "编码体系", "编码框架", "编码规则", "标注规范", "时间粒度", "时间窗口",
    "事件边界", "起止边界", "一致性", "信度", "效度", "结构化观察", "行为序列",
)
_PERCEPTION_ACTION_TERMS = (
    "recognition", "detection", "classification", "estimation", "tracking",
    "识别", "检测", "分类", "估计", "跟踪",
)


def paper_text(paper: dict[str, Any]) -> str:
    claims = " ".join(
        str(claim.get("claim") or claim.get("text") or "")
        for values in (paper.get("field_claims") or {}).values()
        for claim in values or []
        if isinstance(claim, dict)
    )
    return " ".join([
        str(paper.get("title") or ""),
        str(paper.get("abstract") or ""),
        " ".join(str(item) for item in paper.get("keywords") or []),
        str(paper.get("research_problem") or ""),
        str(paper.get("method") or ""),
        str(paper.get("study_design") or ""),
        claims,
    ]).lower()


def direct_evidence_match(
    paper: dict[str, Any],
    requirement: dict[str, Any],
) -> tuple[bool, str]:
    window = temporal_requirement_window(requirement)
    if window is not None:
        # 时间窗口要求按发表年份判定，而不是词面匹配“近五年”——
        # 年份约束已由检索层硬过滤执行，覆盖检查只需复核年份。
        return _match_publication_window(paper, window)
    text = paper_text(paper)
    role = str(requirement.get("evidence_role") or "")
    aliases = [str(item) for item in requirement.get("aliases") or [] if str(item).strip()]
    alias_hit = any(_contains_concept(text, alias) for alias in aliases)

    if role == "structured_coding":
        coding_hit = alias_hit or any(term in text for term in _CODING_TERMS)
        structure_hit = any(term in text for term in _CODING_STRUCTURE_TERMS)
        if coding_hit and structure_hit:
            return True, "报告了编码体系、时间/事件规则或编码效度等结构化编码证据"
        return False, "仅有行为类别或识别输出，未报告结构化编码规则、序列产物或效度"

    if role == "analytical_method":
        if alias_hit:
            return True, "正文或元数据直接出现用户指定的分析方法"
        if (
            str(requirement.get("selection_mode") or "all") == "open_any"
            and _matches_open_analytical_method(text, requirement)
        ):
            return True, "报告了与研究对象直接相关的其他明确分析方法"
        return False, "未直接出现用户指定方法，不能用一般时序或互动分析替代"

    if role == "perception":
        technical_aliases = [
            alias for alias in aliases
            if _normalize(alias) not in {"recognize", "detect", "识别", "检测"}
        ]
        method_hit = any(_contains_concept(text, alias) for alias in technical_aliases)
        action_hit = any(term in text for term in _PERCEPTION_ACTION_TERMS)
        if method_hit and action_hit:
            return True, "直接报告了感知、检测或识别任务及其方法"
        return False, "没有同时出现识别任务与相应技术方法"

    if role == "interpretation":
        if alias_hit:
            return True, "直接分析了用户指定的下游对象或解释目标"
        return False, "未直接覆盖用户指定的下游解释目标"

    return (alias_hit, "直接命中阶段概念" if alias_hit else "未直接命中阶段概念")


def _match_publication_window(
    paper: dict[str, Any],
    window: tuple[int, int],
) -> tuple[bool, str]:
    start_year, end_year = window
    raw_year = paper.get("year")
    try:
        year = int(str(raw_year or "")[:4])
    except (TypeError, ValueError):
        return False, "论文缺少年份，无法按用户指定时间窗判定"
    if start_year <= year <= end_year:
        return True, f"发表年份 {year} 在用户指定时间窗 {start_year}-{end_year} 内"
    return False, f"发表年份 {year} 不在用户指定时间窗 {start_year}-{end_year} 内"


def evidence_coverage(
    semantic_frame: dict[str, Any],
    papers: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    paper_list = list(papers)
    requirements = list(semantic_frame.get("evidence_requirements") or [])
    counts: dict[str, int] = {}
    required_counts: dict[str, int] = {}
    matched_ids: dict[str, list[str]] = {}
    labels: dict[str, str] = {}
    reasons: dict[str, dict[str, str]] = {}
    corpus = [paper_text(paper) for paper in paper_list]
    for requirement in requirements:
        requirement_id = str(requirement.get("requirement_id") or requirement.get("label") or "")
        if not requirement_id:
            continue
        ids: list[str] = []
        per_paper: dict[str, str] = {}
        for paper in paper_list:
            paper_id = str(paper.get("paper_id") or paper.get("id") or "")
            matched, reason = direct_evidence_match(paper, requirement)
            if paper_id:
                per_paper[paper_id] = reason
            if paper_id and matched:
                ids.append(paper_id)
        ids = list(dict.fromkeys(ids))
        if not ids:
            ids, per_paper = _decomposed_alias_matches(
                requirement, paper_list, corpus, per_paper
            )
        minimum = max(1, int(requirement.get("minimum_direct_sources") or 1))
        counts[requirement_id] = len(ids)
        required_counts[requirement_id] = minimum
        matched_ids[requirement_id] = ids
        labels[requirement_id] = str(requirement.get("label") or requirement_id)
        reasons[requirement_id] = per_paper
    missing_ids = [key for key, count in counts.items() if count < required_counts[key]]
    return {
        "ready": not missing_ids,
        "counts": counts,
        "required_counts": required_counts,
        "matched_paper_ids": matched_ids,
        "requirement_labels": labels,
        "missing_requirement_ids": missing_ids,
        "missing_focuses": [labels[key] for key in missing_ids],
        "match_reasons": reasons,
    }


# 复合别名的构件级回退阈值。
#
# 语义解析产出的别名常是"研究对象 + 动作"式复合词（实测「教学互动分析」），
# 而论文里写的是"教学互动行为""师生互动"——完整别名一次都不出现，逐篇字面
# 匹配因此恒为 0，覆盖门禁误报该重点缺失（2026-08-29 实测：用户澄清明确说
# "偏向于教学互动分析"，59 篇里该别名命中 0 次）。
#
# 回退判据全部由当前语料统计给出，不含任何领域词表：
# 1. 只在完整别名全语料零命中时启用，绝不放宽本来有效的要求；
# 2. 用语料自身做最长匹配切分，得到实际出现过的构件；
# 3. 要求论文同时含有全部构件（合取，不是析取）；
# 4. 构件覆盖率超过上限时放弃回退——说明构件太泛，合取已无区分度，
#    此时报告真实缺口比放行更诚实。
_DECOMPOSED_MIN_SEGMENTS = 2
_DECOMPOSED_MAX_MATCH_RATIO = 0.5


def _decomposed_alias_matches(
    requirement: dict[str, Any],
    papers: list[dict[str, Any]],
    corpus: list[str],
    per_paper: dict[str, str],
) -> tuple[list[str], dict[str, str]]:
    """完整别名零命中时，按语料切分出的构件做合取匹配。"""
    if not corpus or temporal_requirement_window(requirement) is not None:
        return [], per_paper
    segment_groups = [
        segments
        for alias in requirement.get("aliases") or []
        if (segments := _decompose_by_corpus(str(alias), corpus))
        and len(segments) >= _DECOMPOSED_MIN_SEGMENTS
    ]
    if not segment_groups:
        return [], per_paper
    updated = dict(per_paper)
    ids: list[str] = []
    for paper, text in zip(papers, corpus):
        paper_id = str(paper.get("paper_id") or paper.get("id") or "")
        if not paper_id:
            continue
        hit = next(
            (
                segments for segments in segment_groups
                if all(_contains_concept(text, segment) for segment in segments)
            ),
            None,
        )
        if hit:
            ids.append(paper_id)
            updated[paper_id] = (
                "完整别名未出现，但同时含有其全部构件：" + "、".join(hit)
            )
    if len(ids) > len(corpus) * _DECOMPOSED_MAX_MATCH_RATIO:
        # 构件过泛，合取失去区分度：保持原判并说明放弃回退的原因。
        return [], per_paper
    return list(dict.fromkeys(ids)), updated


def _decompose_by_corpus(alias: str, corpus: list[str]) -> list[str]:
    """把别名切成"在语料中真实出现过"的最长构件。

    中文没有词边界，因此不按词表切分，而是从左到右做最长匹配：每一步取
    仍能在语料里找到的最长前缀。拉丁字母别名按空白切分后同样只保留出现过
    的词。切不出（或只切出一段）时返回空列表，交由调用方放弃回退。
    """
    text = str(alias or "").strip()
    if not text:
        return []
    if not re.search(r"[\u4e00-\u9fff]", text):
        tokens = [token for token in re.split(r"\s+", text.lower()) if token]
        return [
            token for token in tokens
            if any(token in document for document in corpus)
        ]
    segments: list[str] = []
    start = 0
    while start < len(text):
        for end in range(len(text), start, -1):
            candidate = text[start:end]
            if len(candidate) >= 2 and any(
                _contains_concept(document, candidate) for document in corpus
            ):
                segments.append(candidate)
                start = end
                break
        else:
            start += 1
    return segments


def citation_eligible_paper_ids(
    semantic_frame: dict[str, Any],
    papers: Iterable[dict[str, Any]],
    deliverable_type: str | None = None,
) -> set[str]:
    """筛出可支撑指定交付物的论文；语义筛选结果优先于词法兜底。"""
    paper_list = list(papers)
    coverage = evidence_coverage(semantic_frame, paper_list)
    direct_ids = {
        paper_id
        for ids in coverage.get("matched_paper_ids", {}).values()
        for paper_id in ids
    }
    topic = str(semantic_frame.get("canonical_topic") or "")
    topic_tokens = _tokens(topic)
    object_aliases = _frame_aliases(semantic_frame, "research_objects")
    method_aliases = _frame_aliases(semantic_frame, "methods")
    target_aliases = _frame_aliases(semantic_frame, "analysis_targets")
    eligible = set(direct_ids)
    for paper in paper_list:
        paper_id = str(paper.get("paper_id") or paper.get("id") or "")
        if not paper_id:
            continue
        relation_type = str(
            paper.get("relation_type") or paper.get("_topic_relation") or ""
        ).strip().lower()
        relation_type = {
            "method_related": "near",
            "topic_related": "near",
            "background": "indirect",
            "analogy": "indirect",
        }.get(relation_type, relation_type)
        allowed_deliverables = {
            str(value) for value in (
                paper.get("eligible_deliverables")
                or paper.get("_eligible_deliverables")
                or []
            )
            if str(value).strip()
        }
        if relation_type in {"indirect", "unrelated"}:
            eligible.discard(paper_id)
            continue
        if allowed_deliverables:
            if not deliverable_type or deliverable_type in allowed_deliverables:
                eligible.add(paper_id)
            else:
                eligible.discard(paper_id)
            continue
        if relation_type in {"direct", "near"}:
            eligible.add(paper_id)
            continue

        text = paper_text(paper)
        token_overlap = len(topic_tokens & _tokens(text))
        object_hit = any(_contains_concept(text, alias) for alias in object_aliases)
        method_or_target_hit = any(
            _contains_concept(text, alias) for alias in [*method_aliases, *target_aliases]
        )
        # 用户明确研究对象本身就是直接主题锚点；并非每篇背景或教育分析
        # 论文都必须再次出现技术方法名。方法阶段是否充足由独立覆盖门禁负责。
        if object_hit:
            eligible.add(paper_id)
        elif not object_aliases and token_overlap >= max(2, min(4, len(topic_tokens))):
            eligible.add(paper_id)
        elif not topic_tokens and not object_aliases and not method_aliases and not target_aliases:
            # 兼容没有语义框架的离线/旧调用；生产链路会携带 relation_type。
            eligible.add(paper_id)
    return eligible


def _frame_aliases(frame: dict[str, Any], field: str) -> list[str]:
    return list(dict.fromkeys(
        str(value).strip()
        for item in frame.get(field) or []
        for value in (
            item.get("surface_text"), item.get("label"),
            str(item.get("id") or "").replace("_", " "),
        )
        if str(value or "").strip()
    ))


def _contains_concept(text: str, concept: str) -> bool:
    concept = str(concept or "").strip().lower()
    if not concept:
        return False
    if concept in text:
        return True
    compact_text = _normalize(text)
    compact_concept = _normalize(concept)
    return bool(compact_concept and compact_concept in compact_text)


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(value or "").lower())


def _matches_open_analytical_method(
    text: str,
    requirement: dict[str, Any],
) -> bool:
    """开放任选时接受其他明确方法，但仍要求命中用户研究上下文。"""
    context_aliases = [
        str(item) for item in requirement.get("context_aliases") or []
        if str(item).strip()
    ]
    context_hit = not context_aliases or any(
        _contains_concept(text, alias) for alias in context_aliases
    )
    if not context_hit:
        return False
    method_expression = bool(re.search(
        r"(?:采用|运用|使用|基于|通过).{1,36}(?:分析法|分析方法|分析模型|分析框架)|"
        r"(?:analysis|analytical)\s+(?:method|approach|framework|model)|"
        r"(?:using|applies?|employs?|based\s+on).{1,48}(?:analysis|analytical)",
        text,
        re.I,
    ))
    return method_expression


def _tokens(value: str) -> set[str]:
    lowered = str(value or "").lower()
    english = set(re.findall(r"[a-z][a-z0-9-]{2,}", lowered))
    chinese = re.findall(r"[\u4e00-\u9fff]", lowered)
    return english | {"".join(chinese[i:i + 2]) for i in range(max(0, len(chinese) - 1))}
