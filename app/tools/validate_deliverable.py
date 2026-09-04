"""四类交付物的确定性质量验证。"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from app.core.citation_syntax import (
    extract_citation_ids,
    malformed_citation_fragments,
    normalize_citation_syntax,
)
from app.deliverables.few_shot_blueprints import detect_blueprint_leakage
from app.deliverables.registry import get_deliverable_spec
from app.core.geography import has_reliable_geographic_comparison
from app.core.text_quality import (
    AGENT_PROCESS_LANGUAGE_RE,
    EDITORIAL_LEAKAGE_RE,
    content_sentences as _content_sentences,
    detect_english_sentences as _english_sentences,
    detect_incomplete_fragments as _incomplete_section_fragments,
)
from app.schemas.deliverable_schema import (
    CoreDeliverableType,
    DeliverableValidationResult,
    WritingPlan,
)

# 兼容旧的私有名称；新代码应从 app.core.text_quality 导入公共常量与函数。
_EDITORIAL_LEAKAGE_RE = EDITORIAL_LEAKAGE_RE
_AGENT_PROCESS_LANGUAGE_RE = AGENT_PROCESS_LANGUAGE_RE


def _planned_section_bodies(text: str) -> dict[str, str]:
    matches = list(re.finditer(r"^#{2,4}\s+(.+?)\s*$", str(text or ""), re.M))
    result: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text or "")
        result[match.group(1).strip()] = str(text or "")[match.end():end].strip()
    return result


# 章节标题的中文序号前缀（“（二）”/“(3)”/“二、”）。写作过程中主题章节
# 可能被重排序号，结构完整性检查应忽略序号只比对核心标题。
_SECTION_ORDINAL_PREFIX_RE = re.compile(
    r"^(?:[（(][一二三四五六七八九十0-9]+[）)]|[一二三四五六七八九十]+、)\s*"
)


def _strip_section_ordinal(title: str) -> str:
    """去除章节标题的序号前缀，返回核心标题。"""
    return _SECTION_ORDINAL_PREFIX_RE.sub("", str(title or "")).strip()


def _expected_heading(section) -> str:
    return f"{'#' * int(section.heading_level or 2)} {section.title}"


def _markdown_headings(text: str) -> list[tuple[int, str]]:
    return [
        (len(marks), title.strip())
        for marks, title in re.findall(r"^(#{1,4})\s+(.+?)\s*$", text or "", re.M)
    ]


# 正式研究路线小节（含跨路线比较与研究空白的兼容 ID）；只有这些章节需要
# 章节级证据密度，概述与结构性章节不设下限。
_ROUTE_SECTION_IDS = {"cross_route_comparison", "research_gaps"}

# 章节级正文长度下限：低于该值的小节无法承载路线内综合，只能是罗列。
_MIN_ROUTE_SECTION_PLAIN_CHARS = 80

_EVIDENCE_LIMITED_BODY_RE = re.compile(
    r"当前可访问证据不足|本节仅保留证据边界|无法同时满足主题相关性"
)


def _is_route_section_id(section_id: str) -> bool:
    section_id = str(section_id or "")
    return section_id.startswith("theme_") or section_id in _ROUTE_SECTION_IDS


def _section_authorized_ids(
    sections: list[dict[str, Any]],
    state: dict[str, Any],
) -> dict[str, set[str]]:
    """逐章节的授权论文集合 = 计划支撑 ∪ 该章节引用分配，并交卡片池。"""
    card_ids = {
        str(card.get("paper_id"))
        for card in state.get("paper_cards") or []
        if card.get("paper_id")
    }
    allocation_by_section: dict[str, set[str]] = {}
    for item in (state.get("citation_allocation_plan") or {}).get("sections") or []:
        section_id = str(item.get("section_id") or "")
        if not section_id:
            continue
        allocation_by_section.setdefault(section_id, set()).update(
            str(paper_id) for paper_id in item.get("paper_ids") or [] if paper_id
        )
    result: dict[str, set[str]] = {}
    for section in sections:
        section_id = str(section.get("id") or "")
        authorized = {
            str(paper_id)
            for paper_id in section.get("supporting_paper_ids") or []
            if paper_id
        } | allocation_by_section.get(section_id, set())
        result[section_id] = (authorized & card_ids) if card_ids else authorized
    return result


def _resolve_body_citation_ids(
    body: str,
    *,
    citation_map: dict[str, Any] | None,
    card_ids: set[str],
) -> tuple[set[str], bool]:
    """把章节正文的引用标记解析到 paper_id 空间。

    WHY: 最终成文在引用校验阶段已被渲染成顺序编码（``citation_map`` 是
    paper_id → 编号），而写作计划的授权集合始终是 paper_id。直接把两者取
    交集会让每个小节都算作零篇引用（2026-08-30 实测：六节全部误报）。
    没有映射可用时返回 ``False``，由调用方退回按唯一引用计数，并在诊断里
    标明所处标识空间，而不是静默放过或静默失败。
    """
    raw = {str(item) for item in extract_citation_ids(body)}
    if not raw:
        return set(), True
    number_to_paper = {
        str(index): str(paper_id)
        for paper_id, index in (citation_map or {}).items()
        if paper_id
    }
    resolved = {number_to_paper.get(item, item) for item in raw}
    if card_ids and not (resolved & card_ids):
        return resolved, False
    return resolved, True


def _section_evidence_floor_findings(
    text: str,
    sections: list[dict[str, Any]],
    state: dict[str, Any],
    *,
    fallback_minimum: int,
) -> list[dict[str, Any]]:
    """统一的章节级证据密度评估，供写作校验与最终成文审计共用。

    WHY: 两处曾各自硬编码「至少 2 篇引用、至少 80 字」，且只按原标题取
    正文；章节被重排序号后取不到正文即静默放过。这里按章节 ID 读取计划
    契约、按去序号标题定位正文，并只统计该章节授权范围内的有效引用，
    避免把其他章节或未授权论文算作本节达标。
    """
    bodies = _planned_section_bodies(text)
    bodies_by_core = {
        _strip_section_ordinal(title): body for title, body in bodies.items()
    }
    authorized_by_section = _section_authorized_ids(sections, state)
    card_ids = {
        str(card.get("paper_id"))
        for card in state.get("paper_cards") or []
        if card.get("paper_id")
    }
    citation_map = state.get("citation_map") or {}
    findings: list[dict[str, Any]] = []
    for section in sections:
        section_id = str(section.get("id") or "")
        title = str(section.get("title") or "")
        if not _is_route_section_id(section_id):
            continue
        planned_minimum = int(section.get("minimum_unique_references") or 0)
        minimum = planned_minimum or fallback_minimum
        if minimum <= 0:
            continue
        body = bodies.get(title) or bodies_by_core.get(_strip_section_ordinal(title), "")
        if not body:
            # 章节缺失由计划章节检查报告，这里不重复判定。
            continue
        authorized = authorized_by_section.get(section_id) or set()
        cited, in_paper_space = _resolve_body_citation_ids(
            body, citation_map=citation_map, card_ids=card_ids,
        )
        if authorized and in_paper_space:
            effective = cited & authorized
            identifier_space = "paper_id"
        else:
            effective = cited
            identifier_space = "paper_id" if in_paper_space else "rendered"
        plain_chars = len(re.sub(r"\[[^\]]+\]|\s+", "", body))
        if _EVIDENCE_LIMITED_BODY_RE.search(body):
            status = "evidence_limited"
        elif len(effective) < minimum:
            status = "sparse_citations"
        elif plain_chars < _MIN_ROUTE_SECTION_PLAIN_CHARS:
            status = "short_body"
        else:
            status = "ok"
        findings.append({
            "section_id": section_id,
            "title": title,
            "required_unique_references": minimum,
            "authorized_paper_count": len(authorized),
            "actual_unique_references": len(effective),
            "plain_char_count": plain_chars,
            "identifier_space": identifier_space,
            "status": status,
        })
    return findings


def _section_floor_error_messages(
    findings: list[dict[str, Any]],
    *,
    sparse_prefix: str,
    evidence_limited_prefix: str,
) -> list[str]:
    sparse = [
        item["title"] for item in findings
        if item["status"] in {"sparse_citations", "short_body"}
    ]
    evidence_limited = [
        item["title"] for item in findings if item["status"] == "evidence_limited"
    ]
    messages: list[str] = []
    if sparse:
        messages.append(sparse_prefix + "、".join(dict.fromkeys(sparse)))
    if evidence_limited:
        messages.append(
            evidence_limited_prefix + "、".join(dict.fromkeys(evidence_limited))
        )
    return messages



def validate_final_review_integrity(
    text: str,
    state: dict[str, Any],
) -> dict[str, Any]:
    """在主张删除/弱化之后再次审计最终成文，而非复用写作前结果。"""
    rendered = str(text or "")
    errors: list[str] = []
    bodies = _planned_section_bodies(rendered)
    bodies_by_core = {
        _strip_section_ordinal(title): body for title, body in bodies.items()
    }
    expected_sections = [
        section
        for plan in state.get("writing_plans") or []
        for section in plan.get("sections") or []
    ]
    missing = [
        str(section.get("title") or "")
        for section in expected_sections
        if section.get("title")
        and _strip_section_ordinal(section.get("title")) not in bodies_by_core
    ]
    if missing:
        errors.append("主张校验后缺少计划章节：" + "、".join(dict.fromkeys(missing)))

    editorial = _EDITORIAL_LEAKAGE_RE.findall(rendered)
    if editorial:
        errors.append("最终正文包含编辑提示或修改说明")
    agent_language = _AGENT_PROCESS_LANGUAGE_RE.findall(rendered)
    if agent_language:
        errors.append("最终正文包含检索、证据可用性或代理运行语言")
    incomplete = _incomplete_section_fragments(rendered)
    if incomplete:
        errors.append("最终正文包含截断句：" + "；".join(incomplete[:3]))
    abnormal = re.findall(
        r"(?:。{2,}|！{2,}|？{2,}|[.!?][。！？]|[。！？][.!?])",
        rendered,
    )
    if abnormal:
        errors.append("最终正文包含异常叠加标点")
    # WHY: 主张改写（verify_claims_node）会在本函数之前重写正文，而写作期的
    # validate_deliverable 审的是改写前文本；此处不查重句，改写产生的逐字重复
    # 段落就会直接进入成品。
    duplicate_sentences = _find_duplicate_sentences(rendered)
    if duplicate_sentences:
        errors.append("最终正文存在重复或高度相似句子")

    large_review = int(state.get("required_reference_count") or 0) >= 20
    # 未完成章节的检测覆盖全部计划章节（含概述与结构性章节），不受章节级
    # 证据下限的适用范围限制。
    evidence_limited = [
        str(section.get("title") or "")
        for section in expected_sections
        if _EVIDENCE_LIMITED_BODY_RE.search(
            bodies.get(str(section.get("title") or ""))
            or bodies_by_core.get(
                _strip_section_ordinal(str(section.get("title") or "")), ""
            )
        )
    ]
    if evidence_limited:
        errors.append("最终正文仍有未完成章节：" + "、".join(dict.fromkeys(evidence_limited)))
    floor_findings = _section_evidence_floor_findings(
        rendered,
        expected_sections,
        state,
        # 旧计划快照没有章节契约字段时，沿用"高引用量综述每节至少两篇"的既有口径。
        fallback_minimum=2 if large_review else 0,
    )
    sparse_messages = _section_floor_error_messages(
        floor_findings,
        sparse_prefix="研究路线章节证据或内容密度不足：",
        evidence_limited_prefix="",
    )
    errors.extend(
        message for message in sparse_messages
        if message.startswith("研究路线章节证据或内容密度不足：")
    )

    return {
        "valid": not errors,
        "errors": errors,
        "metrics": {
            "expected_section_count": len(expected_sections),
            "actual_section_count": len(bodies),
            "editorial_leakage_count": len(editorial),
            "agent_process_language_count": len(agent_language),
            "incomplete_fragment_count": len(incomplete),
            "abnormal_punctuation_count": len(abnormal),
            "duplicate_sentence_count": len(duplicate_sentences),
            "evidence_limited_section_count": len(set(evidence_limited)),
            "sparse_theme_section_count": len([
                item for item in floor_findings
                if item["status"] in {"sparse_citations", "short_body"}
            ]),
            "section_evidence_floors": floor_findings,
        },
    }


def _normalize_sentence(sentence: str) -> str:
    normalized = re.sub(r"\[[^\]]+\]", "", sentence)
    normalized = re.sub(r"[^\w\u4e00-\u9fff]+", "", normalized, flags=re.UNICODE)
    return normalized.casefold()


def _is_duplicate_pair(normalized: str, other_normalized: str) -> bool:
    """重复判据：归一化后完全相同，或长句相似度达到 0.92。"""
    if normalized == other_normalized:
        return True
    return (
        min(len(normalized), len(other_normalized)) >= 24
        and SequenceMatcher(None, normalized, other_normalized).ratio() >= 0.92
    )


def _find_duplicate_sentences(text: str) -> list[tuple[str, str]]:
    """检测完全重复和仅有轻微措辞差异的长句。"""
    candidates: list[tuple[str, str]] = []
    for sentence in _content_sentences(text):
        normalized = _normalize_sentence(sentence)
        if len(normalized) >= 16:
            candidates.append((sentence, normalized))

    duplicates: list[tuple[str, str]] = []
    for index, (sentence, normalized) in enumerate(candidates):
        for previous_sentence, previous_normalized in candidates[:index]:
            if _is_duplicate_pair(normalized, previous_normalized):
                duplicates.append((previous_sentence, sentence))
                break
    return duplicates


def find_similar_sentence(
    sentence: str,
    text: str,
    exclude: str | None = None,
) -> str | None:
    """返回 text 中与 sentence 构成重复的另一句，不存在则返回 None。

    与 _find_duplicate_sentences 共用同一判据，使改写阶段采纳单条改写前就能
    拦下重复，而不是等终审才发现整批白改。exclude 是被替换的原句自身。
    """
    normalized = _normalize_sentence(sentence)
    if len(normalized) < 16:
        return None
    excluded = _normalize_sentence(exclude) if exclude else ""
    for other in _content_sentences(text):
        if exclude and other == exclude:
            continue
        other_normalized = _normalize_sentence(other)
        if len(other_normalized) < 16 or other_normalized == excluded:
            continue
        if _is_duplicate_pair(normalized, other_normalized):
            return other
    return None


def _runtime_identifier_leakage(
    text: str,
    plan: WritingPlan,
    state: dict[str, Any],
) -> list[str]:
    """只检查本次运行真实产生的内部 ID，不维护领域词表。"""
    frame = state.get("research_semantic_frame") or {}
    values: list[str] = [
        *[str(item) for item in frame.get("task_chain") or []],
        *[
            str(item.get("id") or "")
            for field in ("methods", "research_actions", "analysis_targets")
            for item in frame.get(field) or []
        ],
        *[str(section.id) for section in plan.sections],
        *[str(node.node_id) for node in plan.hidden_planning_nodes],
        *[
            str(item.get("node_id") or "")
            for item in state.get("planning_nodes") or []
            if isinstance(item, dict)
        ],
    ]
    identifiers = sorted({
        value for value in values
        if "_" in value and re.fullmatch(r"[a-z][a-z0-9_]{2,}", value, re.I)
    })
    return [
        identifier for identifier in identifiers
        if re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(identifier)}(?![A-Za-z0-9_])",
            str(text or ""),
            re.I,
        )
    ]


def validate_deliverable(
    text: str,
    plan: WritingPlan | dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    plan = plan if isinstance(plan, WritingPlan) else WritingPlan.model_validate(plan)
    card_ids = {
        str(card.get("paper_id"))
        for card in state.get("paper_cards") or []
        if card.get("paper_id")
    }
    # 引用授权范围 = 写作计划各章节声明的证据 ∪ 当前交付物的引用分配计划。
    # 只用全量卡片池时，写作层越权引用其他交付物专属论文不会被拦截，
    # 全局唯一引用配额的核算也会随之失真。计划未携带任何授权 ID 时
    # 退回全量卡片池，保持旧调用方兼容。
    plan_authorized = {
        str(paper_id)
        for section in plan.sections
        for paper_id in (section.supporting_paper_ids or [])
        if paper_id
    } | {
        str(paper_id)
        for section in (state.get("citation_allocation_plan") or {}).get("sections") or []
        for paper_id in section.get("paper_ids") or []
        if paper_id
    }
    valid_ids = (plan_authorized & card_ids) if plan_authorized else card_ids
    text = normalize_citation_syntax(text, valid_ids)
    dtype = plan.deliverable_type
    errors: list[str] = []
    warnings: list[str] = []
    floor_findings: list[dict[str, Any]] = []
    missing = [
        section.title for section in plan.sections
        if _expected_heading(section) not in text
    ]
    if missing:
        errors.append("缺少计划章节：" + "、".join(missing))
    planned_headings = {
        (int(section.heading_level or 2), section.title)
        for section in plan.sections
    }
    actual_headings = {
        item for item in _markdown_headings(text) if item[0] >= 2
    }
    extra_titles = sorted(
        title for _, title in actual_headings - planned_headings
    )
    if extra_titles:
        errors.append("出现计划外章节：" + "、".join(extra_titles))
    if re.search(r"^#(?!#)\s+", text or "", re.M):
        errors.append("正文添加了计划外总标题")
    english_sentences = _english_sentences(text)
    if english_sentences:
        errors.append("正文包含完整英文句子，未按要求转述为中文")
    duplicate_sentences = _find_duplicate_sentences(text)
    if duplicate_sentences:
        errors.append("正文存在重复或高度相似句子")
    blueprint_leakage = detect_blueprint_leakage(text)
    if blueprint_leakage:
        errors.append("正文泄漏了写作少样本的占位内容或示例引用")
    editorial_leakage = _EDITORIAL_LEAKAGE_RE.findall(text or "")
    if editorial_leakage:
        errors.append("正文泄漏了编辑提示、修改说明或机器检查文本")
    agent_process_language = _AGENT_PROCESS_LANGUAGE_RE.findall(text or "")
    if agent_process_language:
        errors.append("正文包含检索、证据可用性或代理运行语言")
    leaked_runtime_ids = _runtime_identifier_leakage(text, plan, state)
    if leaked_runtime_ids:
        errors.append("正文泄漏内部任务链或规划节点标识：" + "、".join(leaked_runtime_ids))
    incomplete_fragments = _incomplete_section_fragments(text)
    if incomplete_fragments:
        errors.append("章节末尾存在疑似截断或未完成的句子")
    if "【章节生成未通过】" in text:
        errors.append("存在未能完成可靠中文综合的隔离章节")
    citations = extract_citation_ids(text, normalize_fullwidth=False)
    all_citations = set(citations)
    invented = sorted({citation for citation in all_citations if citation not in valid_ids})
    if invented:
        errors.append("正文引用了不存在或未授权的论文ID")
    minimum_unique = int(plan.citation_policy.get("minimum_unique_references") or 0)
    if len(all_citations & valid_ids) < minimum_unique:
        errors.append(
            f"正文唯一有效引用不足：要求{minimum_unique}篇，实际{len(all_citations & valid_ids)}篇"
        )
    malformed = malformed_citation_fragments(text)
    if malformed:
        errors.append("正文存在未闭合或格式损坏的引用标记")
    abnormal_punctuation = re.findall(r"(?:。{2,}|！{2,}|？{2,}|[.!?][。！？]|[。！？][.!?])", text or "")
    if abnormal_punctuation:
        errors.append("正文存在连续或叠加的异常标点")
    if re.search(r"Quantitative and Empirical|Qualitative and Observational|Theory and Concepts", text, re.I):
        errors.append("正文使用了内部通用研究设计标签")
    if re.search(r"^##\s*(?:Other|其他相关研究|其他)$", text, re.I | re.M):
        errors.append("正式章节使用了Other或其他兜底类别")
    profile = state.get("user_paper_profile") or {}
    has_user_objective = bool(
        profile.get("research_problem")
        and (profile.get("proposed_method") or profile.get("research_direction"))
    )
    if not has_user_objective and re.search(
        r"本文(?:提出|设计|开发|构建)|本文的?贡献|本研究(?:旨在|提出|设计|开发|构建)",
        text,
        re.I,
    ):
        errors.append("用户未提供研究目标，正文却生成了本文目标、方法或贡献")
    canonical_topic = str(state.get("canonical_topic") or state.get("topic") or "").strip()
    if canonical_topic and not _topic_tokens_present(canonical_topic, text):
        errors.append("正文未保持规范研究主题，可能发生主题漂移")
    deliverable_names = {
        CoreDeliverableType.RESEARCH_BACKGROUND: "研究背景",
        CoreDeliverableType.RESEARCH_STATUS: "研究现状",
        CoreDeliverableType.RELATED_WORK: "相关工作",
        CoreDeliverableType.NARRATIVE_REVIEW: "叙述性综述初稿",
    }
    # “相关工作”可能作为普通学术名词出现，只对明显的标题包装进行严格判断。
    if dtype != CoreDeliverableType.NARRATIVE_REVIEW and re.search(
        r"(?:^|[#：:])\s*[^\n]{0,80}叙述性综述初稿", text, re.M
    ):
        errors.append(f"{deliverable_names[dtype]}被错误包装为叙述性综述初稿")

    if dtype == CoreDeliverableType.RESEARCH_BACKGROUND:
        metric_count = len(re.findall(r"\b(?:accuracy|precision|recall|F1|mAP)\b|准确率|召回率|精确率", text, re.I))
        experimental_detail_sentences = [
            sentence for sentence in _content_sentences(text)
            if re.search(r"准确率|召回率|精确率|\bmAP\b|\bF1\b|提升\s*\d|降低\s*\d|\d+(?:\.\d+)?%", sentence, re.I)
        ]
        if metric_count > 2 or len(experimental_detail_sentences) > 2:
            errors.append("研究背景包含过多模型指标或单篇实验细节")
        if len(all_citations) < 2:
            errors.append("研究背景缺少至少两篇独立文献支持")
        h2 = [title for level, title in _markdown_headings(text) if level == 2]
        nested = [title for level, title in _markdown_headings(text) if level >= 3]
        if len(h2) != 1 or nested:
            errors.append("研究背景只能包含一个可见二级标题且不得设置内部小标题")
        if re.search(
            r"^(?:研究问题与场景|研究价值与必要性|主要挑战|进一步研究(?:的)?必要性|证据范围说明)\s*[：:]",
            text or "",
            re.M,
        ):
            errors.append("研究背景包含伪装成自然段的内部规划标题")
        body = _planned_section_bodies(text).get(plan.sections[0].title, "") if plan.sections else ""
        paragraphs = [part for part in re.split(r"\n\s*\n", body) if part.strip()]
        structure = get_deliverable_spec(dtype).structure
        if structure and not (
            (structure.min_paragraphs or 0) <= len(paragraphs)
            <= (structure.max_paragraphs or 10**6)
        ):
            errors.append(
                f"研究背景应写成{structure.min_paragraphs}—{structure.max_paragraphs}个连续自然段"
            )
    elif dtype == CoreDeliverableType.RESEARCH_STATUS:
        if plan.undercovered_focuses:
            errors.append(
                "研究现状写作计划未覆盖用户明确重点："
                + "、".join(plan.undercovered_focuses)
            )
        planned_theme_titles = [
            section.title for section in plan.sections
            if section.id.startswith("theme_")
        ]
        if planned_theme_titles and not any(
            title and _strip_section_ordinal(title) in text for title in planned_theme_titles
        ):
            errors.append("研究现状未使用动态主题组织正文")
        if len(state.get("theme_synthesis") or []) < 2:
            # 窄题/小语料的证据可能确实只支持单一研究路线；这是证据边界
            # 而不是写作缺陷，硬性否决会让生成在重试循环里耗尽恢复次数。
            warnings.append("当前证据仅归纳出单一研究路线，无法形成多路线对比")
        headings = _markdown_headings(text)
        h2_count = sum(level == 2 for level, _ in headings)
        h3_count = sum(level == 3 for level, _ in headings)
        # 子节数量上限必须读 spec，不能写死：spec 的 max_subsections 放宽到 6
        # 后，硬编码的 <=4 会让每份 5-6 节的正文都被判结构非法。
        structure = get_deliverable_spec(dtype).structure
        max_subsections = int((structure.max_subsections if structure else 0) or 4)
        min_subsections = max(1, int((structure.min_subsections if structure else 1) or 1))
        planned_route_count = sum(
            1 for section in plan.sections if section.id.startswith("theme_")
        )
        if planned_route_count:
            min_subsections = max_subsections = planned_route_count
        if h2_count != 1 or not min_subsections <= h3_count <= max_subsections:
            errors.append(
                f"研究现状应包含一个二级主标题和 {min_subsections} 至 "
                f"{max_subsections} 个动态三级研究路线"
            )
        forbidden = get_deliverable_spec(dtype).validation.forbidden_visible_sections
        leaked = [item for item in forbidden if any(item in title for _, title in headings)]
        if leaked:
            errors.append("研究现状包含禁止独立展示的内部章节：" + "、".join(leaked))
        theme_sections = [
            section for section in plan.sections if section.id.startswith("theme_")
        ]
        if theme_sections:
            final_bodies = _planned_section_bodies(text)
            final_core = _strip_section_ordinal(theme_sections[-1].title)
            final_body = final_bodies.get(theme_sections[-1].title) or next(
                (
                    body
                    for title, body in final_bodies.items()
                    if _strip_section_ordinal(title) == final_core
                ),
                "",
            )
            final_paragraph = next(
                (
                    part.strip()
                    for part in reversed(re.split(r"\n\s*\n", final_body))
                    if part.strip()
                ),
                "",
            )
            if not re.search(r"综合|总体|共同|差异|跨路线|上述证据", final_paragraph):
                errors.append("研究现状末段缺少跨路线综合")
        cards_by_id = {
            str(card.get("paper_id") or ""): card
            for card in state.get("paper_cards") or []
            if card.get("paper_id")
        }
        prose_only = re.sub(r"^#{1,4}.*$", "", text or "", flags=re.M)
        geographic_comparisons = [
            paragraph
            for paragraph in re.split(r"\n\s*\n", prose_only)
            if re.search(r"国内研究.{0,160}国外研究|国外研究.{0,160}国内研究", paragraph, re.S)
        ]
        unsupported_geographic = []
        for paragraph in geographic_comparisons:
            cited_cards = [
                cards_by_id[paper_id]
                for paper_id in set(extract_citation_ids(paragraph))
                if paper_id in cards_by_id
            ]
            if not has_reliable_geographic_comparison(cited_cards):
                unsupported_geographic.append(paragraph[:120])
        if unsupported_geographic:
            errors.append("国内外比较缺少同时覆盖两侧的可靠地域元数据")
        planned_sections = [
            section.model_dump(mode="json") for section in plan.sections
        ]
        floor_findings = _section_evidence_floor_findings(
            text,
            planned_sections,
            state,
            # 计划未携带章节契约时沿用既有口径：仅高引用量综述要求每节两篇。
            fallback_minimum=(
                2 if int(state.get("required_reference_count") or 0) >= 20 else 0
            ),
        )
        errors.extend(_section_floor_error_messages(
            floor_findings,
            sparse_prefix="研究路线章节证据过少或内容过短：",
            evidence_limited_prefix="正式研究现状仍包含未完成的证据边界章节：",
        ))
    elif dtype == CoreDeliverableType.RELATED_WORK:
        profile = state.get("user_paper_profile") or {}
        if not profile.get("research_problem") or not (profile.get("proposed_method") or profile.get("research_direction")):
            errors.append("相关工作缺少用户论文问题或方法路线")
        if re.search(r"本文方法(?:显著|全面)?优于|our method significantly outperforms", text, re.I):
            errors.append("正文无依据声称用户方法优于已有工作")
        h3_count = sum(level == 3 for level, _ in _markdown_headings(text))
        if not 2 <= h3_count <= 4:
            errors.append("相关工作应包含二至四个与用户论文直接相关的三级小节")
        alignment_terms = [
            str(profile.get(key) or "").strip()
            for key in (
                "research_problem", "proposed_method", "research_direction",
                "target_task", "application_scenario",
            )
            if profile.get(key)
        ]
        if alignment_terms and not any(term in text for term in alignment_terms):
            errors.append("相关工作未围绕用户论文的研究问题、任务或方法组织")
    elif dtype == CoreDeliverableType.NARRATIVE_REVIEW:
        if re.search(r"本系统综述|系统综述遵循|PRISMA流程|双人独立筛选", text, re.I):
            errors.append("叙述性综述冒用了未执行的系统综述流程")
        h3_count = sum(level == 3 for level, _ in _markdown_headings(text))
        if not 4 <= h3_count <= 7:
            errors.append("叙述性综述应包含四至七个动态三级小节")
        actual_section_ids = {
            section.id for section in plan.sections
            if _expected_heading(section) in text
        }
        if not any(section_id.startswith("theme_") for section_id in actual_section_ids):
            errors.append("叙述性综述缺少基于证据生成的动态分类路线")
        if not {"comparison_and_gaps", "conclusion"}.issubset(actual_section_ids):
            errors.append("叙述性综述缺少研究比较、证据支持的不足或综合总结")

    paragraphs = [
        part.strip() for part in re.split(r"\n\s*\n", text or "")
        if part.strip() and not part.lstrip().startswith("#")
    ]
    single_paper_paragraphs = sum(
        len(set(extract_citation_ids(part))) == 1 for part in paragraphs
    )
    author_led_paragraphs = sum(
        bool(re.match(r"(?:作者|[A-Za-z\u4e00-\u9fff]{2,20}等?(?:人)?(?:指出|提出|认为|发现))", part))
        for part in paragraphs
    )
    route_sections = [section for section in plan.sections if section.id.startswith("theme_")]
    route_bodies = _planned_section_bodies(text)
    covered_routes = sum(bool(route_bodies.get(section.title, "").strip()) for section in route_sections)
    plain_chars = len(re.sub(r"\s+|#+\s*|\[[^\]]+\]", "", text or ""))
    target_range = plan.target_char_range
    if target_range and not target_range[0] <= plain_chars <= target_range[1]:
        warnings.append(
            f"正文字符数{plain_chars}未落入建议范围{target_range[0]}—{target_range[1]}"
        )
    single_paper_ratio = single_paper_paragraphs / max(1, len(paragraphs))
    author_led_ratio = author_led_paragraphs / max(1, len(paragraphs))
    repeated_templates = _repeated_template_phrases(text)
    if repeated_templates:
        errors.append("正文重复使用模板化总结短语：" + "、".join(repeated_templates))
    if dtype == CoreDeliverableType.RESEARCH_BACKGROUND and len(paragraphs) >= 3 and single_paper_ratio > 0.75:
        errors.append("研究背景由过多单篇论文段落主导，缺少跨论文综合")
    if dtype == CoreDeliverableType.RESEARCH_STATUS:
        if len(paragraphs) >= 3 and single_paper_ratio > 0.65:
            errors.append("研究现状单篇论文主导段落比例过高，缺少跨论文综合")
        if len(paragraphs) >= 3 and author_led_ratio > 0.4:
            errors.append("研究现状按作者逐篇展开，未形成主题级综合")
    return DeliverableValidationResult(
        valid=not errors,
        deliverable_type=dtype,
        errors=errors,
        warnings=warnings,
        metrics={
            "citation_mentions": len(citations),
            "unique_cited_papers": len(set(citations)),
            "section_count": len(plan.sections),
            "english_sentence_count": len(english_sentences),
            "duplicate_sentence_count": len(duplicate_sentences),
            "duplicate_claim_ratio": len(duplicate_sentences) / max(1, len(_content_sentences(text))),
            "blueprint_leakage_count": len(blueprint_leakage),
            "blueprint_leakage_samples": blueprint_leakage[:5],
            "editorial_leakage_count": len(editorial_leakage),
            "agent_process_language_count": len(agent_process_language),
            "runtime_identifier_leakage_count": len(leaked_runtime_ids),
            "incomplete_fragment_count": len(incomplete_fragments),
            "incomplete_fragment_samples": incomplete_fragments[:5],
            "malformed_citation_count": len(malformed),
            "abnormal_punctuation_count": len(abnormal_punctuation),
            "duplicate_sentence_samples": [
                {"first": first[:160], "duplicate": duplicate[:160]}
                for first, duplicate in duplicate_sentences[:3]
            ],
            "author_led_ratio": author_led_ratio,
            "single_paper_paragraph_ratio": single_paper_ratio,
            "repeated_template_phrase_count": len(repeated_templates),
            "citation_density": len(citations) / max(1, plain_chars / 1000),
            "route_coverage_rate": covered_routes / max(1, len(route_sections)),
            "plain_char_count": plain_chars,
            "section_evidence_floors": floor_findings,
            "section_floor_failure_count": len([
                item for item in floor_findings if item["status"] != "ok"
            ]),
        },
    ).model_dump(mode="json")


# 规范研究主题中不携带实体语义的框架词；抽取主题实体词时先行剔除。
# canonical_topic 常为“少样本动作识别（few-shot action recognition）近五年研究综述”
# 这类完整句式，正文不会逐字复现，但必须保留其中的核心实体词。
_TOPIC_FRAME_WORDS = {
    "近五年", "近5年", "近年来", "近十年", "近十年间", "近年来研究",
    "研究", "综述", "系统综述", "研究综述", "文献综述", "回顾", "梳理",
    "分析", "探讨", "评述", "报告", "初稿", "展望", "调研", "考察",
    "方法", "基于", "相关", "本文", "领域", "现状", "现状分析", "问题",
    "主要", "重点", "其中", "以及", "对于", "关于", "针对", "围绕",
}

# 时间范围框架词（如“近三年”“近3年”“近两三年间”）数量组合不可枚举，
# 用正则统一剔除；否则“近三年少样本动作识别研究综述”会抽出
# “近三年少样本动作识别”这样的伪实体，导致正文永远无法匹配而误报主题漂移。
_TOPIC_TIMEFRAME_RE = re.compile(r"近[0-9一二两三四五六七八九十百]+\s*年(?:间|内|来)?")


def _topic_tokens_present(topic: str, text: str) -> bool:
    """规范研究主题的核心实体词是否在正文中出现。

    整句精确匹配对 canonical_topic 过于严格（正文以自然措辞表述同一主题时
    会被误报为主题漂移）；改为抽取主题中的中文实体段，要求至少一个实体段
    （或其 4 字片段）出现在正文中。中文实体缺失时，拉丁词不单独构成匹配，
    避免主题漂移后英文术语残留导致误放行。纯拉丁主题退化为拉丁词匹配。

    Args:
        topic: 规范研究主题（canonical_topic 或 topic）。
        text: 交付物正文。

    Returns:
        True 表示主题实体出现在正文中（或主题无实体词可检验）。
    """
    topic_l = topic.lower()
    text_l = str(text or "").lower()
    cjk_tokens: list[str] = []
    for run in re.findall(r"[一-鿿]+", topic_l):
        reduced = _TOPIC_TIMEFRAME_RE.sub(" ", run)
        for frame in sorted(_TOPIC_FRAME_WORDS, key=len, reverse=True):
            reduced = reduced.replace(frame, " ")
        for part in re.split(r"[\s，,、]+", reduced.strip()):
            part = part.strip("的与和及在对于是等、，,中")
            if len(part) >= 4:
                cjk_tokens.append(part)
    if cjk_tokens:
        if any(token in text_l for token in cjk_tokens):
            return True
        # 长复合主题（如“森林生态安全监控中的动作识别”）正文不会逐字复现整段，
        # 允许其 4 字片段命中（如“动作识别”）。短实体（如“少样本动作识别”）
        # 必须整体出现，避免主题漂移后仅凭通用片段（如“动作识别”）误放行。
        return any(
            fragment in text_l
            for token in cjk_tokens
            if len(token) >= 12
            for fragment in (
                token[index:index + 4] for index in range(len(token) - 3)
            )
        )
    latin_tokens = [
        word for word in re.findall(r"[a-z][a-z0-9_-]*", topic_l) if len(word) >= 5
    ]
    if latin_tokens:
        return any(word in text_l for word in latin_tokens)
    # 主题几乎全由框架词构成时无法抽实体，退回原精确匹配语义。
    return topic_l in text_l


def _repeated_template_phrases(text: str) -> list[str]:
    patterns = {
        "差异反映研究目标与数据条件": r"(?:这种|这些)差异反映(?:的)?是?研究目标与数据条件",
        "不能仅凭单项指标排序": r"不能仅凭单项指标(?:作|进行)?(?:统一)?排序",
        "需要指出的是": r"需要指出的是",
        "综合上述证据": r"综合上述证据",
        "不具备直接横向比较条件": r"不具备直接横向比较的?条件",
    }
    return [label for label, pattern in patterns.items() if len(re.findall(pattern, text or "")) > 1]
