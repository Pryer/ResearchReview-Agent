"""交付物渲染器基类与通用分段合成、清洗、校验工具。"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from app.core.citation_syntax import (
    extract_citation_ids as extract_normalized_citation_ids,
    normalize_citation_syntax,
)
from app.deliverables.few_shot_blueprints import (
    get_plan_blueprints,
    get_section_blueprint,
)
from app.deliverables.registry import get_deliverable_spec
from app.schemas.deliverable_schema import CoreDeliverableType, WritingPlan
from app.core.text_quality import (
    AGENT_PROCESS_LANGUAGE_RE,
    EDITORIAL_LEAKAGE_RE,
    detect_english_sentences,
    detect_incomplete_fragments,
    strip_evidence_meta_language,
)

_AGENT_PROCESS_LANGUAGE_RE = AGENT_PROCESS_LANGUAGE_RE
_EDITORIAL_LEAKAGE_RE = EDITORIAL_LEAKAGE_RE
_english_sentences = detect_english_sentences
_incomplete_section_fragments = detect_incomplete_fragments
_strip_evidence_meta_language = strip_evidence_meta_language

logger = logging.getLogger(__name__)

def _survey_papers(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """提取写作候选中的综述/调研类论文（evidence_role=survey）。"""
    return [
        {
            "paper_id": card.get("paper_id"),
            "title": card.get("title"),
            "year": card.get("year"),
        }
        for card in cards
        if str(card.get("evidence_role") or "") == "survey"
    ]


def _write_sections_in_chinese(
    draft: str,
    plan: WritingPlan,
    state: dict[str, Any],
    cards: list[dict[str, Any]],
    llm,
) -> str:
    """逐节把证据草稿改写为中文综合文本，并保持引用集合不变。"""
    sections = _split_planned_sections(draft, plan)
    if len(sections) != len(plan.sections):
        return draft

    max_workers = min(3, max(1, len(sections)))
    completed: dict[str, str] = {}
    diagnostics: list[dict[str, Any]] = []
    survey_papers = _survey_papers(cards)
    # 末条研究路线负责跨路线综合；多路线时才需要，单路线无“跨路线”可综合。
    theme_section_ids = [
        section.id for section in plan.sections if section.id.startswith("theme_")
    ]
    final_theme_section_id = theme_section_ids[-1] if len(theme_section_ids) > 1 else ""

    # 检查是否有 Claim Plan 约束
    claim_plans = state.get("claim_plans") or []
    _claim_plan_by_route = {
        str(plan_item.get("route_name") or ""): plan_item
        for plan_item in claim_plans
    }

    def _claim_constraints_for_section(section_title: str) -> str:
        """为指定章节生成 claim 约束文本。匹配路线名或背景段落标签。"""
        cleaned_title = str(section_title or "").strip()
        # 精确匹配路线名
        for route_name, plan_item in _claim_plan_by_route.items():
            if route_name in cleaned_title or cleaned_title in route_name:
                return _render_claim_constraints(plan_item)
        # 背景节：按标题关键词模糊匹配
        if not cleaned_title:
            return ""
        title_tokens = set(re.findall(r"[一-鿿]{2,}", cleaned_title.lower()))
        for route_name, plan_item in _claim_plan_by_route.items():
            plan_tokens = set(re.findall(r"[一-鿿]{2,}", route_name.lower()))
            if title_tokens & plan_tokens:
                return _render_claim_constraints(plan_item)
        return ""

    def rewrite(section) -> tuple[str, str, dict[str, Any]]:
        original = sections[section.id]
        required_ids = _citation_ids(original)
        if (
            not required_ids
            and not _english_sentences(original)
            and not re.search(r"论文明确报告|从其他纳入证据看", original)
            and len(re.sub(r"^#{2,4}[^\n]*", "", original).strip()) >= 20
        ):
            return section.id, original, {
                "section_id": section.id,
                "status": "already_valid",
                "attempts": 0,
                "required_citations": len(required_ids),
            }
        prompt = _section_rewrite_prompt(
            deliverable_type=plan.deliverable_type,
            section_id=section.id,
            title=section.title,
            heading_level=section.heading_level or 2,
            purpose=section.purpose,
            target_word_count=section.target_word_count,
            topic=str(state.get("canonical_topic") or state.get("topic") or ""),
            research_focus=" ".join(filter(None, [
                str((state.get("selected_scope") or {}).get("description") or ""),
                "必须覆盖：" + "；".join(
                    str(item)
                    for item in (state.get("research_semantic_frame") or {}).get("required_focuses") or []
                ),
                "证据角色：" + "；".join(
                    str(item.get("label") or "")
                    for item in (state.get("research_semantic_frame") or {}).get("evidence_requirements") or []
                    if str(item.get("label") or "").strip()
                ),
            ])),
            original=original,
            required_ids=required_ids,
            survey_papers=survey_papers,
            comparison_dimensions=section.comparison_dimensions,
            claim_constraints=_claim_constraints_for_section(section.title),
            require_cross_route_synthesis=(section.id == final_theme_section_id),
        )
        last_errors: list[str] = []
        previous_candidate = ""
        best_candidate = ""
        best_score = float("-inf")
        attempts_made = 0
        for attempt in range(3):
            retry_context = ""
            if last_errors:
                retry_context = (
                    "\n\n【上一次输出及机器检查结果】\n"
                    + previous_candidate
                    + "\n\n检查错误："
                    + "；".join(last_errors)
                    + "\n请在上一次中文输出基础上定向修复，不要从头自由发挥。"
                    + "重点补齐明确列出的缺失引用、删除多余引用和英文整句。"
                )
            try:
                response = llm.complete(
                    prompt + retry_context,
                    temperature=0.05,
                    # 统一使用 LLM_MAX_TOKENS，避免章节级小预算导致正文截断。
                    retry_empty=True,
                    thinking_enabled=True,
                    operation=(
                        f"write_section:{plan.deliverable_type.value}:"
                        f"{section.id}:attempt_{attempt + 1}"
                    ),
                )
                attempts_made = attempt + 1
            except Exception as exc:
                last_errors = [f"模型调用失败：{exc}"]
                continue
            candidate = _strip_agent_process_clauses(
                normalize_citation_syntax(
                    _clean_section_response(
                        response, section.title, section.heading_level or 2
                    ),
                    required_ids,
                )
            )
            last_errors = _validate_rewritten_section(
                candidate, section.title, required_ids, section.heading_level or 2
            )
            score = _section_candidate_score(
                candidate, section.title, required_ids, section.heading_level or 2
            )
            if score > best_score:
                best_candidate = candidate
                best_score = score
            previous_candidate = candidate
            if not last_errors:
                return section.id, candidate, {
                    "section_id": section.id,
                    "status": "success",
                    "attempts": attempt + 1,
                    "required_citations": len(required_ids),
                    "blueprint_applied": True,
                }
        # 整章只因少量英文残留失败时，给模型一个更小、更明确的局部修复任务。
        # 旧逻辑会直接丢弃数千字且引用完整的正文，造成有效引用瞬间归零。
        english_residue = _english_sentences(best_candidate)
        if best_candidate and english_residue:
            try:
                localized_prompt = _english_residue_repair_prompt(
                    text=best_candidate,
                    title=section.title,
                    heading_level=section.heading_level or 2,
                    required_ids=required_ids,
                    english_sentences=english_residue,
                )
                localized_response = llm.complete(
                    localized_prompt,
                    temperature=0.0,
                    retry_empty=True,
                    thinking_enabled=True,
                    operation=(
                        f"repair_english_fragments:{plan.deliverable_type.value}:"
                        f"{section.id}"
                    ),
                )
                attempts_made += 1
                localized = _strip_agent_process_clauses(
                    normalize_citation_syntax(
                        _clean_section_response(
                            localized_response,
                            section.title,
                            section.heading_level or 2,
                        ),
                        required_ids,
                    )
                )
                localized_errors = _validate_rewritten_section(
                    localized,
                    section.title,
                    required_ids,
                    section.heading_level or 2,
                )
                if not localized_errors:
                    return section.id, localized, {
                        "section_id": section.id,
                        "status": "success_after_english_repair",
                        "attempts": attempts_made,
                        "required_citations": len(required_ids),
                        "english_fragments_repaired": len(english_residue),
                        "blueprint_applied": True,
                    }
                localized_score = _section_candidate_score(
                    localized,
                    section.title,
                    required_ids,
                    section.heading_level or 2,
                )
                if localized_score > best_score:
                    best_candidate = localized
                    best_score = localized_score
                    last_errors = localized_errors
            except Exception as exc:
                last_errors = [*last_errors, f"英文残留局部修复失败：{exc}"]
        if _is_safe_partial_section(
            best_candidate, section.title, required_ids, section.heading_level or 2
        ):
            returned = best_candidate
            status = "partial_missing_citations"
        elif _only_english_residue(last_errors) and _is_safe_partial_section(
            best_candidate,
            section.title,
            required_ids,
            section.heading_level or 2,
            allow_english=True,
        ):
            # 局部修复仍未通过时保留引用完整、结构安全的最佳正文，让最终
            # 质量门禁准确报告英文残留；不能再用无引用占位句覆盖整章。
            returned = best_candidate
            status = "partial_english_residue"
        else:
            # 无法形成可靠正文时返回空值，稍后由路线合并器处理；禁止把
            # “证据边界/引用可核验性”等代理运行语言伪装成学术正文。
            returned = ""
            status = "evidence_limited"
        diagnostic = {
            "section_id": section.id,
            "status": status,
            "attempts": attempts_made,
            "required_citations": len(required_ids),
            "blueprint_applied": True,
            "errors": last_errors,
        }
        logger.warning(
            "Section writer did not fully pass: %s",
            json.dumps(diagnostic, ensure_ascii=False),
        )
        return section.id, returned, diagnostic

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(rewrite, section): section.id
            for section in plan.sections
        }
        for future in as_completed(futures):
            section_id = futures[future]
            try:
                result_id, text, diagnostic = future.result()
            except Exception as exc:
                result_id = section_id
                text = sections[section_id]
                diagnostic = {
                    "section_id": section_id,
                    "status": "fallback",
                    "errors": [str(exc)],
                }
            completed[result_id] = text
            diagnostics.append(diagnostic)

    _merge_failed_or_missing_sections(
        plan=plan,
        state=state,
        cards=cards,
        sections=sections,
        completed=completed,
        diagnostics=diagnostics,
        rewrite=rewrite,
    )

    # 章节级不删空：所有救援与迁移后仍为空的计划章节补保守证据段。
    # 计划章节（尤其是承载章节头的第一节，如研究现状的 status_overview）
    # 一旦被静默丢弃，整章会从最终正文中消失，而 best-effort 又会带病
    # 出厂——宁可以列表式保守段落保住结构，把质量问题留给门禁如实报告。
    for section in plan.sections:
        if not completed.get(section.id, "").strip():
            completed[section.id] = _conservative_evidence_section(section, cards)
            diagnostics.append({
                "section_id": section.id,
                "status": "conservative_evidence_fill",
            })

    state.setdefault("writer_section_diagnostics", []).append({
        "deliverable_type": plan.deliverable_type.value,
        "sections": sorted(diagnostics, key=lambda item: item["section_id"]),
    })
    ordered_sections = [
        (section.id, completed.get(section.id, ""))
        for section in plan.sections
        if completed.get(section.id, "").strip()
    ]
    return "\n\n".join(
        text for _sid, text in _deduplicate_boilerplate_clauses(ordered_sections)
    )


# 各章节独立改写、互不可见，few-shot 蓝本里的免责句式会被逐节复制。实测
# "不宜仅凭单项指标作统一排序"在一篇综述的四个小节中重复出现。这里做跨章节
# 去重：只保留首次出现，后续删除。
#
# 尾部通配一律排除逗号与顿号：允许跨过它们会让一个模式吞掉相邻的下一个分句
# （实测 pattern1 的尾部吞掉了紧随其后的"，并不构成简单的优劣"，只剩"关系。"）。
_BOILERPLATE_CLAUSE_PATTERNS = (
    # 蓝本 example 中的"不能仅凭单项指标作统一排序"及其常见改写。
    r"[，；。]?\s*(?:因此|故|所以)?\s*不(?:宜|能|应)(?:仅|只)?(?:凭|依据|根据|按)"
    r"[^，、；。\n]{0,12}统一排序[^，、；。\n]{0,10}[。；]?",
    # "不构成简单的优劣关系""并非简单的优劣排序"一类同义免责。
    r"[，；。]?\s*(?:二者|两者|各路线|它们)?(?:并)?不(?:构成|存在|意味着)"
    r"[^，、；。\n]{0,12}优劣(?:关系|排序|之分)[^，、；。\n]{0,10}[。；]?",
)


def _section_authorized_paper_ids(
    section_id: str,
    plan: WritingPlan,
    state: dict[str, Any],
) -> set[str]:
    """返回 fallback 当前章节的授权论文集合。"""
    section = next((item for item in plan.sections if item.id == section_id), None)
    allowed = {str(item) for item in (section.supporting_paper_ids if section else []) if item}
    for item in (state.get("citation_allocation_plan") or {}).get("sections") or []:
        if str(item.get("section_id") or "") == section_id:
            allowed.update(str(value) for value in item.get("paper_ids") or [] if value)
    return allowed


def _normalize_fallback_claim(text: str) -> str:
    """生成仅用于去重的规范文本，不改变实际交付物措辞。"""
    text = _neutralize_evidence_self_reference(text)
    text = re.sub(r"\[[^\]]+\]", "", text)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[，,。；;、！？!?：:（）()“”\"'「」]", "", text)
    return text.casefold()


def _deduplicate_fallback_claims(
    sections: list[tuple[str, str]],
    plan: WritingPlan,
    state: dict[str, Any],
) -> list[tuple[str, str]]:
    """按规范文本、引用集合和章节授权集去重 fallback 的重复句。

    只在同节且授权集一致时合并；跨节或授权集不同即使文本相同也保留，
    因而不会把一个章节的合法引用迁移到另一个章节。不同引用集合的同一
    主张会在原章节内合并引用并集，保证授权引用不丢失。
    """
    sentence_re = re.compile(r"[^。！？!?\n]*\[[^\]]+\][^。！？!?\n]*[。！？!?]?")
    result: list[tuple[str, str]] = []
    for section_id, text in sections:
        authorized = frozenset(_section_authorized_paper_ids(section_id, plan, state))
        matches = list(sentence_re.finditer(text))
        if not matches:
            result.append((section_id, text))
            continue
        # WHY: 分组键包含授权集合，确保相同措辞只在同一章节授权边界内合并。
        groups: dict[tuple[str, frozenset[str]], dict[str, Any]] = {}
        for match in matches:
            sentence = match.group(0)
            citation_ids = set(_citation_ids(sentence))
            if not citation_ids:
                continue
            key = (_normalize_fallback_claim(sentence), authorized)
            group = groups.setdefault(key, {
                "first": match,
                "citation_ids": set(),
            })
            group["citation_ids"].update(citation_ids)

        output_parts: list[str] = []
        cursor = 0
        emitted: set[tuple[str, frozenset[str]]] = set()
        for match in matches:
            output_parts.append(text[cursor:match.start()])
            sentence = match.group(0)
            citation_ids = set(_citation_ids(sentence))
            key = (_normalize_fallback_claim(sentence), authorized)
            if not citation_ids or key not in groups:
                output_parts.append(sentence)
            elif key in emitted:
                # 同节同授权的重复主张只删除正文重复项；引用已并入首次句。
                pass
            else:
                merged = re.sub(r"\[[^\]]+\]", "", sentence).rstrip()
                merged = merged.rstrip("。！？!? ") + "[" + "; ".join(sorted(groups[key]["citation_ids"])) + "]"
                if not merged.endswith(("。", "！", "？", "!", "?")):
                    merged += "。"
                output_parts.append(merged)
                emitted.add(key)
            cursor = match.end()
        output_parts.append(text[cursor:])
        output = re.sub(r"\n{3,}", "\n\n", "".join(output_parts)).strip()
        if output:
            result.append((section_id, output))
    return result



def _split_fallback_sections(text: str, plan: WritingPlan) -> list[tuple[str, str]]:
    """按 WritingPlan 标题恢复 fallback 的 section 记录。"""
    matches = list(re.finditer(r"^(#{2,4})\s+(.+?)\s*$", text or "", re.M))
    by_title: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        by_title[match.group(2).strip()] = text[match.start():end].strip()
    return [
        (section.id, by_title[section.title])
        for section in plan.sections
        if section.title in by_title
    ]


def _deduplicate_boilerplate_clauses(
    sections: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """跨章节删除重复的免责套话，只保留首次出现。

    逐节并发改写时每个章节都看不到其他章节，蓝本里的免责句式因此被反复复制。
    这类句子本身没错，但在同一篇综述里出现四次会暴露模板痕迹。删除后若句子
    所在段落变空，段落一并移除，避免留下空行。
    """
    compiled = [re.compile(pattern) for pattern in _BOILERPLATE_CLAUSE_PATTERNS]
    seen: set[int] = set()
    result: list[tuple[str, str]] = []
    for section_id, text in sections:
        cleaned = text
        for index, pattern in enumerate(compiled):
            if not pattern.search(cleaned):
                continue
            if index in seen:
                cleaned = pattern.sub("", cleaned)
            else:
                # 首次出现保留，仅记录，后续章节再遇到才删。
                seen.add(index)
        cleaned = _tidy_after_clause_removal(cleaned)
        if not _has_body_text(cleaned):
            # 整节正文只有这句套话：删掉会留下裸标题。保留原文，让门禁去报告
            # 该节证据不足，而不是在正文里留一个空章节。
            result.append((section_id, text))
            continue
        result.append((section_id, cleaned))
    return result


def _has_body_text(text: str) -> bool:
    """判断章节除标题外是否还有正文。"""
    return any(
        line.strip() and not line.lstrip().startswith("#")
        for line in text.split("\n")
    )


def _tidy_after_clause_removal(text: str) -> str:
    """删除套话后收拾标点与空行，避免出现"……，。"或缺失句末标点。"""
    lines = []
    for line in text.split("\n"):
        line = line.rstrip()
        # 悬空的分句标点：删掉句尾子句后原本的逗号会紧贴句末标点。
        line = re.sub(r"[，、；]\s*(?=[。；])", "", line)
        line = re.sub(r"[，、]{2,}", "，", line)
        if line and not line.startswith("#"):
            # 模式的前导标点会被一并删除，行尾可能只剩正文或悬空逗号；
            # 两种情况都补回句号，保持正文由完整句子构成。
            line = re.sub(r"[，、；]$", "。", line)
            if not re.search(r"[。；！？：)）”\"']$", line):
                line = f"{line}。"
        lines.append(line)
    # 折叠因整句删除而产生的空行，但保留标题与正文之间的单个换行结构。
    collapsed: list[str] = []
    for line in lines:
        if not line.strip() and (not collapsed or not collapsed[-1].strip()):
            continue
        collapsed.append(line)
    return "\n".join(collapsed).strip()


def _conservative_evidence_section(section, cards: list[dict[str, Any]]) -> str:
    """失败章节的保守证据段：只罗列分配给该节的文献事实，不做任何综合推断。

    此前只输出"《标题》（年份）"的清单，读者拿到的是书目而非综述文字。改为
    在同样不做推断的前提下，把卡片里已有的研究问题与方法逐篇如实陈述——这些
    都是各文献自身报告的内容，不构成跨文献综合，但足以让本节成为可读段落。
    """
    heading = _section_heading(section)
    pid_set = {str(p) for p in section.supporting_paper_ids or []}
    section_cards = (
        [card for card in cards if str(card.get("paper_id") or "") in pid_set]
        if pid_set
        else cards[:6]
    )
    entries: list[str] = []
    for card in section_cards[:60]:
        pid = str(card.get("paper_id") or "")
        title = str(card.get("title") or "").strip()
        if not pid or not title:
            continue
        year = card.get("year") or "年份未知"
        venue = str(card.get("venue") or "").strip()
        source_note = f"{year}，{venue}" if venue else str(year)
        # 只引用卡片中已有的原文级陈述，不做任何跨文献推断或补写。
        problem = _first_sentence(card.get("research_problem"))
        method = _first_sentence(card.get("method"))
        detail = ""
        if problem and method:
            detail = f"针对{problem}，采用{method}"
        elif problem:
            detail = f"针对{problem}展开研究"
        elif method:
            detail = f"采用{method}"
        if detail:
            entries.append(f"《{title}》（{source_note}）{detail}[{pid}]。")
        else:
            entries.append(f"《{title}》（{source_note}）[{pid}]。")

    if not entries:
        return "\n".join([heading, "当前证据池中没有分配给本节的论文。"])

    lead = (
        f"本节纳入 {len(entries)} 篇文献，以下按各文献自身报告的问题设定与方法"
        "如实列出，不作跨文献综合："
    )
    return "\n".join([heading, lead, *entries])


def _first_sentence(value: Any, limit: int = 80) -> str:
    """取字段首句并去掉句末标点，用于拼接为可读短语。"""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return ""
    text = re.split(r"[。；;！!？?\n]", text)[0].strip()
    text = text.rstrip("，,、").strip()
    return text[:limit]


def _heading(title: str, heading_level: int | None = 2) -> str:
    return f"{'#' * int(heading_level or 2)} {title}"


def _section_heading(section) -> str:
    return _heading(section.title, section.heading_level)


def _clean_route_title(title: str) -> str:
    return re.sub(
        r"^[（(][一二三四五六七八九十0-9]+[）)]\s*", "", str(title or "")
    ).strip()


def _route_titles(plan: WritingPlan) -> list[str]:
    return [
        title
        for section in plan.sections
        if section.id.startswith("theme_")
        and (title := _clean_route_title(section.title))
    ]


def _synthesis_aspects(items: list[dict[str, Any]]) -> list[str]:
    aspect_fields = (
        ("研究问题", ("reported_problems", "common_problems")),
        ("研究方法", ("reported_methods", "common_methods")),
        ("研究发现", ("reported_findings",)),
        ("适用边界与局限", ("synthesized_gaps", "author_stated_limitations")),
    )
    return [
        label
        for label, fields in aspect_fields
        if any(item.get(field) for item in items for field in fields)
    ]


def _status_overview_lead(topic: str, route_titles: list[str]) -> str:
    if route_titles:
        return f"围绕{topic}，现有研究主要从{'、'.join(route_titles)}等方向展开。"
    return f"围绕{topic}，现有文献从不同研究对象、分析视角与实践问题展开讨论。"


def _cross_route_summary(
    topic: str,
    route_titles: list[str],
    synthesis_items: list[dict[str, Any]],
) -> str:
    routes = "、".join(route_titles) if route_titles else "现有研究方向"
    aspects = "、".join(_synthesis_aspects(synthesis_items)) or "关注重点与主要结论"
    return f"综合来看，围绕{topic}的{routes}在{aspects}上呈现出各自的侧重与相互补充的认识。"


def _evidence_limited_section(
    section_id: str,
    title: str,
    heading_level: int = 2,
) -> str:
    """兼容旧调用：失败章节不再生成代理语言占位正文。"""
    del section_id, title, heading_level
    return ""


def _merge_failed_or_missing_sections(
    *,
    plan: WritingPlan,
    state: dict[str, Any],
    cards: list[dict[str, Any]],
    sections: dict[str, str],
    completed: dict[str, str],
    diagnostics: list[dict[str, Any]],
    rewrite,
) -> None:
    """将失败路线或漏引论文迁移到可写路线并仅重写受影响章节。"""
    diagnostic_by_id = {str(item.get("section_id") or ""): item for item in diagnostics}
    section_by_id = {section.id: section for section in plan.sections}
    surviving_routes = [
        section for section in plan.sections
        if section.id.startswith("theme_")
        and diagnostic_by_id.get(section.id, {}).get("status") not in {"evidence_limited", "fallback"}
        and completed.get(section.id, "").strip()
    ]
    if not surviving_routes:
        # 无主题路线幸存时，尝试回退到任意非主题幸存章节
        surviving_routes = [
            section for section in plan.sections
            if not section.id.startswith("theme_")
            and diagnostic_by_id.get(section.id, {}).get("status") not in {"evidence_limited", "fallback"}
            and completed.get(section.id, "").strip()
        ]
    if not surviving_routes:
        # 全线路失败兜底：若原始草稿经门禁剥离后仍可过基础校验则保留，
        # 否则沿用旧 quarantine 行为放空，避免英文碎片/零引用泄漏。
        for section in list(plan.sections):
            if section.id not in completed or not completed.get(section.id, "").strip():
                original = sections.get(section.id, "")
                if not original.strip():
                    continue
                rescued = _strip_agent_process_clauses(
                    _strip_evidence_meta_language(
                        _strip_stray_numeric_citations(original)
                    )
                )
                rescued_errors = _validate_rewritten_section(
                    rescued,
                    section.title,
                    _citation_ids(original) or [],
                    section.heading_level or 2,
                )
                # 只保留结构完整、非纯英文、至少有部分引用的草案
                fatal = [
                    err for err in rescued_errors
                    if "英文句子" in err or "缺少引用编号" in err or "章节标题" in err or "过短" in err
                ]
                if fatal:
                    continue
                completed[section.id] = rescued
                diagnostics.append({
                    "section_id": section.id,
                    "status": "original_draft_retained",
                    "attempts": diagnostic_by_id.get(section.id, {}).get("attempts", 0),
                    "retained_errors": rescued_errors,
                    "note": "所有改写均失败，原始草稿经门禁剥离后仍可过基础校验",
                })
        return

    allocation_sections = (state.get("citation_allocation_plan") or {}).get("sections") or []
    allocation_by_id = {
        str(item.get("section_id") or ""): item for item in allocation_sections
    }
    removed_ids: set[str] = set()

    for donor in list(plan.sections):
        diagnostic = diagnostic_by_id.get(donor.id, {})
        planned_ids = set(_citation_ids(sections.get(donor.id, "")))
        actual_ids = set(_citation_ids(completed.get(donor.id, "")))
        failed = diagnostic.get("status") in {"evidence_limited", "fallback"}
        missing_ids = sorted(planned_ids - actual_ids)
        if not failed and not missing_ids:
            continue
        target = max(
            surviving_routes,
            key=lambda section: _section_merge_score(donor, section),
        )
        moved_ids = sorted(planned_ids if failed else set(missing_ids))
        if not moved_ids:
            if failed:
                removed_ids.add(donor.id)
            continue

        target.supporting_paper_ids = list(dict.fromkeys([
            *target.supporting_paper_ids,
            *moved_ids,
        ]))
        recovery = _citation_recovery_paragraph(moved_ids, cards)
        if not recovery:
            continue
        base = completed.get(target.id) or sections.get(target.id, "")
        sections[target.id] = base.rstrip() + "\n\n" + recovery

        donor_allocation = allocation_by_id.get(donor.id)
        target_allocation = allocation_by_id.get(target.id)
        if donor_allocation is not None:
            donor_allocation["paper_ids"] = [
                item for item in donor_allocation.get("paper_ids") or []
                if str(item) not in set(moved_ids)
            ]
        if target_allocation is not None:
            target_allocation["paper_ids"] = list(dict.fromkeys([
                *[str(item) for item in target_allocation.get("paper_ids") or []],
                *moved_ids,
            ]))

        _, rewritten, target_diagnostic = rewrite(target)
        if rewritten.strip() and target_diagnostic.get("status") not in {"evidence_limited", "fallback"}:
            completed[target.id] = rewritten
            diagnostics.append({
                **target_diagnostic,
                "status": "success_after_route_merge",
                "merged_from": donor.id,
                "reassigned_paper_ids": moved_ids,
            })
            if failed:
                removed_ids.add(donor.id)
                completed.pop(donor.id, None)

    if removed_ids:
        plan.sections = [section for section in plan.sections if section.id not in removed_ids]
        if allocation_sections:
            state["citation_allocation_plan"]["sections"] = [
                item for item in allocation_sections
                if str(item.get("section_id") or "") not in removed_ids
            ]


def _section_merge_score(donor, target) -> int:
    def tokens(value: str) -> set[str]:
        value = str(value or "").lower()
        return set(re.findall(r"[a-z][a-z0-9_-]{2,}|[\u4e00-\u9fff]{2}", value))

    donor_tokens = tokens(f"{donor.title} {donor.purpose}")
    target_tokens = tokens(f"{target.title} {target.purpose}")
    return len(donor_tokens & target_tokens) - len(target.supporting_paper_ids) // 10


def _citation_recovery_paragraph(
    paper_ids: list[str],
    cards: list[dict[str, Any]],
) -> str:
    cards_by_id = {str(card.get("paper_id") or ""): card for card in cards}
    sentences: list[str] = []
    for paper_id in paper_ids:
        card = cards_by_id.get(paper_id) or {}
        claim = next((
            str(item.get("claim") or item.get("text") or "").strip()
            for field in ("research_problem", "method", "results", "contributions")
            for item in (card.get("field_claims") or {}).get(field) or []
            if isinstance(item, dict)
            and item.get("explicitly_reported", True)
            and str(item.get("claim") or item.get("text") or "").strip()
        ), "")
        if claim:
            claim = _neutralize_evidence_self_reference(claim)
            sentences.append(f"{claim.rstrip('。！？.!?')}[{paper_id}]。")
    return "".join(sentences)


def _neutralize_evidence_self_reference(text: str) -> str:
    """把从论文证据字段复制的作者自称改为有归属的第三人称。"""
    return str(text or "").replace("本研究", "该研究").replace("本文", "该研究")


def _render_claim_constraints(plan_item: dict[str, Any] | None) -> str:
    """将 claim plan 渲染为 writer 不可绕过的约束文本。"""
    if not plan_item:
        return ""
    claims = plan_item.get("claims") or []
    if not claims:
        return ""

    lines = ["【授权主张清单——只能写以下主张，不能创造新的事实性主张】"]
    abstract_only_papers: set[str] = set()
    for i, claim in enumerate(claims, 1):
        level = claim.get("support_level", "single")
        lang = claim.get("allowed_language", "")
        eids = ", ".join(claim.get("evidence_ids", [])[:5])
        entry = (
            f"  C{i}. [{level.upper()}] {claim['claim_text']}\n"
            f"      证据ID: {eids}\n"
            f"      允许措辞强度: {lang}"
        )
        if claim.get("evidence_access_limit") == "abstract_only":
            entry += "\n      ⚠ 摘要级证据：只能泛泛提及，禁止提取具体事实、数值或结论"
            # evidence_id 格式为 {paper_id}:eNNN，paper_id 自身含冒号，
            # 必须从最后一个冒号切分才能还原真实 paper_id。
            abstract_only_papers.update(
                str(eid).rsplit(":", 1)[0]
                for eid in claim.get("evidence_ids", [])
                if ":" in str(eid)
            )
        lines.append(entry)
    if abstract_only_papers:
        lines.append(
            "  【摘要级证据引用限制】论文 "
            + "、".join(sorted(abstract_only_papers))
            + " 仅有摘要级证据：引用它们时只能作背景性泛泛提及，"
            "不得从其提取具体方法细节、实验结果或性能数值。"
        )
    lines.append("  --- 以上清单外的任何事实性主张将被视为未经授权 ---")
    return "\n".join(lines)


def _section_rewrite_prompt(
    *,
    deliverable_type: CoreDeliverableType,
    section_id: str,
    title: str,
    heading_level: int = 2,
    topic: str,
    original: str,
    required_ids: list[str],
    purpose: str = "",
    target_word_count: int | None = None,
    research_focus: str = "",
    survey_papers: list[dict[str, Any]] | None = None,
    comparison_dimensions: list[str] | None = None,
    claim_constraints: str = "",
    require_cross_route_synthesis: bool = False,
) -> str:
    blueprint = get_section_blueprint(deliverable_type, section_id)
    survey_papers_json = json.dumps(survey_papers or [], ensure_ascii=False)
    dimensions = [str(value) for value in comparison_dimensions or [] if str(value).strip()]
    # 末条研究路线承担跨路线综合：该要求原先只写在 purpose 里，与"避免空泛
    # 固定结尾"的要求冲突而常被模型忽略，导致结构校验报"末段缺少跨路线综合"。
    cross_route_requirement = (
        "\n20. **本节是最后一条研究路线，必须以一个独立末段完成跨路线综合**："
        "概括各路线的共同进展、彼此差异与有证据支持的共性不足，"
        "并显式使用“综合”“总体”“共同”“差异”等表述；"
        "该段只综合本正文各路线已写明的判断，不得引入新事实，"
        "也不得另设“研究空白”“未来方向”之类的小标题。"
        if require_cross_route_synthesis
        else ""
    )
    if dimensions:
        comparison_instruction = (
            "仅在真实证据能够形成比较时，围绕 WritingPlan 动态给出的维度 "
            f"{json.dumps(dimensions, ensure_ascii=False)} 归纳差异、取舍或演进；"
            "比较结论必须由本节引用证据直接支持，不得套用预设方法分类。"
        )
    else:
        comparison_instruction = (
            "WritingPlan 未要求固定比较维度；根据章节任务和真实证据决定是否比较，"
            "不得为了形成趋势而发明方法类别、演进方向或适用条件。"
        )
    return f"""你是中文学术综述的章节编辑。请只改写下面这一个章节。

研究主题：{topic}
用户确认的分析重点：{research_focus or topic}
章节标题：{title}
章节任务：{purpose}
目标字数：{target_word_count or '按证据充分程度合理展开'}
必须原样保留且每个至少出现一次的引用编号：
{json.dumps(required_ids, ensure_ascii=False)}

【脱敏写作少样本——只示范修辞结构，不是事实证据】
建议修辞步骤：{json.dumps(blueprint["moves"], ensure_ascii=False)}
写法示例：{blueprint["example"]}

硬性要求：
1. 第一行必须且只能是“{_heading(title, heading_level)}”，不得增加其他标题、前言或修改说明。
2. 把英文证据忠实转述为自然、严谨的中文；模型名、缩写、数据集名可保留英文，但不得保留完整英文句子。
3. 只使用草稿已有事实，不补充常识、数字、结论或推测；摘要证据只按摘要可见范围表述。
4. 按共同研究问题、方法或发现进行综合，不逐篇列举，不使用“论文明确报告”“从其他纳入证据看”等机械句式。
5. 引用必须紧跟所支持的中文主张，并严格使用半角 ASCII 方括号 [paper_id]；禁止改成〔paper_id〕或其他括号。可把支持同一综合判断的多篇文献并列引用，但不能遗漏、新增或改写任何引用编号。
6. 删除残缺的英文片段、摘要页眉和关键词串；若片段无法形成完整事实，只保留其引用并与同节已有、确有证据的综合判断合并。
7. 避免空泛的固定结尾，避免与其他章节可能重复的通用句。
8. 只模仿少样本的组织方式；不得输出其中的〈占位内容〉、〔证据A〕或任何示例事实。
9. 每个“现有研究、多项工作、普遍、共同、形成趋势”类综合判断都必须紧跟支持它的引用，不能作为无引用的过渡句。
10. 必须按“用户确认的分析重点”区分感知或识别输出、结构化编码产物、指定分析方法与下游解释；不得用相邻阶段的证据替代当前章节要求的证据角色。
11. 对“该领域快速发展”“获得持续关注”“研究热点”“呈现X格局”等宏观判断，只能引用综述/调研类论文：{survey_papers_json}；不得用单篇方法论文支撑领域级断言。若名单为空，把宏观断言收窄到具体技术路线或子领域层面。
12. 与研究主题或当前任务阶段只有场景邻接关系的陈述不得作为方法证据；低相关论文可以不写，不得为了凑引用数量强行拼接。
13. 禁止连续句号、句号与逗号叠加等异常标点。
14. 严格完成”章节任务”规定的段落数量和组织方式，并在证据允许时接近目标字数；若要求连续自然段，不得自行增加内部小标题。
15. **动态比较要求**：{comparison_instruction}
16. **证据强度决定语言强度**（按草稿中支持同一主张的独立 paper_id 数量）：
    - 仅 1 篇 → 只能写"有研究尝试…""一项工作提出…""X 等人报告…"
    - 2–3 篇 → 可写"部分研究采用…""若干工作探索…""已有证据显示…"
    - 4–6 篇 → 可写"多项研究…""形成了较为明确的…""在…方面取得了可验证的进展"
    - 7+ 篇且有综述支撑 → 才可写"已成为重要研究方向""该领域形成了…格局"
    违反此映射表的"趋势""已成为""共同面临""普遍认为""主流"等宏观断言将被视为无引用支撑。
17. **授权主张清单**：只能写以下清单中的主张，使用对应措辞强度，引用对应证据ID。不能创造新的趋势判断、领域空白、性能声明或方法优劣评价。过渡句和结构连接可以自由生成，但不能包含新的事实性内容。
{claim_constraints}
18. **引用密度与点名引用**：单处引用建议 1~2 篇，最多不得超过 3 篇。绝对严禁在段落首尾一次性倾倒大段连排引用（如 [p1..p20]）。每次引用[paper_id]时，必须在同句中写明该论文的方法/模型名称缩写或第一作者姓；禁止"有研究提出[id]""有工作尝试[id]""另有研究指出[id]"等匿名引用句式。正确写法示例："Author 等提出的 Method 模型[paper_id]通过特定机制优化了核心任务表现"。
19. **章节文体与定位解耦**：
    - 若当前为【研究背景】：重点阐明现实应用痛点、理论与技术驱动力、数据与场景约束以及研究范式的现实需求，不罗列具体算法细节。
    - 若当前为【国内外研究现状】：直接聚焦于各方法学流派的核心网络机制、代表模型（包含创新点）、基准评测表现与技术边界；严禁在开头或子节中重复复述背景定义（如“本研究领域旨在...”）。{cross_route_requirement}

【真实证据草稿——正文事实与引用的唯一来源】
{original}
"""


def _english_residue_repair_prompt(
    *,
    text: str,
    title: str,
    heading_level: int,
    required_ids: list[str],
    english_sentences: list[str],
) -> str:
    return f"""你是中文学术正文的局部校订器。下面章节的结构、事实和引用已经通过，
现在只处理机器检测到的英文残留。

必须处理的英文片段：
{json.dumps(english_sentences[:10], ensure_ascii=False)}

硬性要求：
1. 完整输出修复后的章节，第一行必须是“{_heading(title, heading_level)}”。
2. 把上述完整英文句子忠实改写为中文；模型名、缩写、数据集名可以保留英文。
3. 不改动其他中文事实，不增加、删除或重新解释主张。
4. 以下引用编号必须全部原样保留，不能新增、删除或替换：
{json.dumps(required_ids, ensure_ascii=False)}
5. 不输出解释、修改说明、代码块或参考文献表。

待修复章节：
{text}
"""


def _split_planned_sections(
    text: str,
    plan: WritingPlan,
) -> dict[str, str]:
    matches = list(re.finditer(r"^(#{2,4})\s+(.+?)\s*$", text or "", re.M))
    by_title: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        by_title[match.group(2).strip()] = text[match.start():end].strip()
    return {
        section.id: by_title[section.title]
        for section in plan.sections
        if section.title in by_title
    }


def _citation_ids(text: str) -> list[str]:
    return extract_normalized_citation_ids(text)


def _clean_section_response(
    text: str,
    title: str,
    heading_level: int = 2,
) -> str:
    rendered = str(text or "").strip()
    rendered = re.sub(r"^```(?:markdown)?\s*", "", rendered, flags=re.I)
    rendered = re.sub(r"\s*```$", "", rendered)
    expected = _heading(title, heading_level)
    if expected in rendered and not rendered.startswith(expected):
        rendered = rendered[rendered.index(expected):]
    # 确定性拆散引用堆砌后再进入校验：模型输出的连续小组引用
    # （如 [p1, p2, p3][p4, p5]）直接修复，不再消耗重写轮次。
    from app.core.citation_density import break_citation_dumps

    rendered = break_citation_dumps(rendered)
    return rendered.strip()


def _validate_rewritten_section(
    text: str,
    title: str,
    required_ids: list[str],
    heading_level: int = 2,
) -> list[str]:
    from app.core.citation_density import detect_citation_dumps
    from app.deliverables.few_shot_blueprints import detect_blueprint_leakage

    errors: list[str] = []
    headings = re.findall(r"^(#{2,4})\s+(.+?)\s*$", text or "", re.M)
    if headings != [("#" * heading_level, title)]:
        errors.append("章节标题不唯一或不匹配")
    actual_ids = _citation_ids(text)
    missing_ids = [
        paper_id for paper_id in required_ids if paper_id not in set(actual_ids)
    ]
    extra_ids = [
        paper_id for paper_id in actual_ids if paper_id not in set(required_ids)
    ]
    if missing_ids:
        errors.append(
            "缺少引用编号：" + json.dumps(missing_ids, ensure_ascii=False)
        )
    if extra_ids:
        errors.append(
            "出现未授权引用编号：" + json.dumps(extra_ids, ensure_ascii=False)
        )
    dumps = detect_citation_dumps(text, max_per_group=3)
    if dumps:
        errors.append(f"单处引用堆砌超过3篇（检测到 {len(dumps)} 处）")
    english_sentences = _english_sentences(text)
    if english_sentences:
        errors.append(
            "仍含完整英文句子："
            + json.dumps(english_sentences[:3], ensure_ascii=False)
        )
    if re.search(r"论文明确报告|从其他纳入证据看", text or ""):
        errors.append("仍含机械证据罗列句式")
    if detect_blueprint_leakage(text):
        errors.append("泄漏了少样本占位内容或示例引用")
    if _EDITORIAL_LEAKAGE_RE.search(text or ""):
        errors.append("泄漏了编辑提示或修改说明")
    if _AGENT_PROCESS_LANGUAGE_RE.search(text or ""):
        errors.append("泄漏了检索、证据可用性或代理运行语言")
    if _incomplete_section_fragments(text):
        errors.append("章节末尾存在疑似截断句")
    if len(re.sub(r"^#{2,4}[^\n]*", "", text or "").strip()) < 20:
        errors.append("章节正文过短")
    return errors


def _section_candidate_score(
    text: str,
    title: str,
    required_ids: list[str],
    heading_level: int = 2,
) -> float:
    from app.core.citation_density import detect_citation_dumps
    from app.deliverables.few_shot_blueprints import detect_blueprint_leakage

    score = min(len(str(text or "")), 1200) / 1200
    headings = re.findall(r"^(#{2,4})\s+(.+?)\s*$", text or "", re.M)
    score += 10 if headings == [("#" * heading_level, title)] else -20
    actual = set(_citation_ids(text))
    required = set(required_ids)
    score += 2 * len(actual & required)
    score -= 4 * len(required - actual)
    score -= 8 * len(actual - required)
    dumps = detect_citation_dumps(text, max_per_group=3)
    score -= 10 * len(dumps)
    score -= 20 * len(_english_sentences(text))
    score -= 30 * len(detect_blueprint_leakage(text))
    score -= 30 if _EDITORIAL_LEAKAGE_RE.search(text or "") else 0
    score -= 30 if _AGENT_PROCESS_LANGUAGE_RE.search(text or "") else 0
    score -= 30 * len(_incomplete_section_fragments(text))
    if re.search(r"论文明确报告|从其他纳入证据看", text or ""):
        score -= 10
    return score


def _is_safe_partial_section(
    text: str,
    title: str,
    required_ids: list[str],
    heading_level: int = 2,
    allow_english: bool = False,
) -> bool:
    from app.deliverables.few_shot_blueprints import detect_blueprint_leakage

    if (
        not text
        or (_english_sentences(text) and not allow_english)
        or detect_blueprint_leakage(text)
        or _EDITORIAL_LEAKAGE_RE.search(text)
        or _AGENT_PROCESS_LANGUAGE_RE.search(text)
        or _incomplete_section_fragments(text)
    ):
        return False
    if re.findall(r"^(#{2,4})\s+(.+?)\s*$", text, re.M) != [
        ("#" * heading_level, title)
    ]:
        return False
    actual_ids = set(_citation_ids(text))
    required = set(required_ids)
    if actual_ids - required:
        return False
    if required:
        # 旧逻辑只检查“没有额外引用”，即便 required_ids 全部缺失也会放行。
        # 安全部分章节至少保留 80% 的计划引用，避免一节看似可读却把引用
        # 覆盖量从 40 篇静默压缩到个位数。
        from math import ceil

        minimum_coverage = max(1, ceil(len(required) * 0.8))
        if len(actual_ids & required) < minimum_coverage:
            return False
    if re.search(r"论文明确报告|从其他纳入证据看", text):
        return False
    return len(re.sub(r"^#{2,4}[^\n]*", "", text).strip()) >= 20


def _only_english_residue(errors: list[str]) -> bool:
    relevant = [
        error for error in errors
        if not error.startswith("英文残留局部修复失败：")
    ]
    return bool(relevant) and all(
        error.startswith("仍含完整英文句子") for error in relevant
    )


def _record_writer_diagnostic(
    state: dict[str, Any],
    plan: WritingPlan,
    *,
    strategy: str,
    validation: dict[str, Any],
) -> None:
    state.setdefault("writer_diagnostics", []).append({
        "deliverable_type": plan.deliverable_type.value,
        "strategy": strategy,
        "valid": bool(validation.get("valid")),
        "errors": list(validation.get("errors") or []),
        "metrics": dict(validation.get("metrics") or {}),
    })


_NUMERIC_CITATION_GROUP_RE = re.compile(
    r"\[(?:\s*\d+\s*[,，、;；\s]*)+\]"
    r"|〔(?:\s*\d+\s*[,，、;；\s]*)+\〕"
)


def _strip_stray_numeric_citations(text: str) -> str:
    """删除模型误写的纯数字引用。

    内部写作协议只接受 paper_id（如 ``[arxiv:2407.14744]``）；纯数字
    引用无法映射到任何论文，只会被总门禁判为“未授权的论文ID”。模型
    在逐节改写时偶尔把证据列表顺序误写成编号，这里在门禁前统一清除。
    """
    if not text:
        return text
    return _NUMERIC_CITATION_GROUP_RE.sub("", str(text))


_SENTENCE_BOUNDARY_RE = re.compile(r"[。！？!?；;\n]")


def _strip_agent_process_clauses(text: str) -> str:
    """删除章节候选中的门禁语言小句（“证据不足”等套话）。

    总门禁要求全文零出现 ``_AGENT_PROCESS_LANGUAGE_RE``。逐节改写时模型
    常以“现有证据尚不足以说明哪种建模策略普遍占优”这类自然措辞结尾，
    重试反馈也压不掉。这里在校验前确定性清除整句并清理残留标点——
    只删命中片段会留下无主语的残句，因此把匹配位置扩展到前后句界。
    """
    if not text:
        return text
    rendered = str(text)
    parts: list[str] = []
    last = 0
    for m in _AGENT_PROCESS_LANGUAGE_RE.finditer(rendered):
        # 从匹配点向前找上一句界（含“，然而，”这类引导语一起切掉）
        start = m.start()
        for pos in range(m.start() - 1, last - 1, -1):
            if rendered[pos] in "。！？!?；;\n":
                start = pos + 1
                break
        # 向后切到本句结束
        tail = _SENTENCE_BOUNDARY_RE.search(rendered, m.end())
        end = tail.end() if tail else len(rendered)
        parts.append(rendered[last:start])
        last = end
    parts.append(rendered[last:])
    cleaned = "".join(parts)
    cleaned = re.sub(r"[，,]{2,}", "，", cleaned)
    cleaned = re.sub(r"[。]{2,}", "。", cleaned)
    cleaned = re.sub(r"[，,]\s*[。]|[。]\s*[，,]", "。", cleaned)
    cleaned = re.sub(r"[^\S\n]{2,}", " ", cleaned)
    cleaned = re.sub(r" {2,}", " ", cleaned)
    return cleaned


def _normalize_evidence_citations(text: str, cards: list[dict[str, Any]]) -> str:
    """把模型误用的 evidence_id 引用确定性还原为所属 paper_id。"""
    evidence_to_paper: dict[str, str] = {}
    for card in cards:
        paper_id = str(card.get("paper_id") or "")
        for claim in card.get("claims") or []:
            evidence_id = str(claim.get("evidence_id") or "")
            if paper_id and evidence_id:
                evidence_to_paper[evidence_id] = paper_id
    rendered = str(text or "")
    for evidence_id, paper_id in sorted(
        evidence_to_paper.items(), key=lambda item: len(item[0]), reverse=True
    ):
        rendered = rendered.replace(f"[{evidence_id}]", f"[{paper_id}]")
    return normalize_citation_syntax(
        rendered,
        [str(card.get("paper_id") or "") for card in cards],
    )


def _citation_target_met(text: str, plan: WritingPlan) -> bool:
    target = int(plan.citation_policy.get("minimum_unique_references") or 0)
    if target <= 0:
        return True
    return len(set(_citation_ids(text))) >= target


def _style_guidance(plan: WritingPlan) -> str:
    style = plan.style_constraints or {}
    conservative = bool(style.get("conservative_evidence_mode"))
    reason = str(style.get("conservative_evidence_reason") or "").strip()
    guidance = [
        f"- conservative_evidence_mode: {str(conservative).lower()}",
    ]
    if reason:
        guidance.append(f"- conservative_evidence_reason: {reason}")
    if conservative:
        guidance.append(
            "- 仅使用收缩性表述：部分研究、已有证据显示、代表性研究、尚可支持的趋势。"
        )
        guidance.append(
            "- 禁止把单篇或少数论文扩写为领域整体结论；路线不足时优先压缩为更少的综合段。"
        )
    return "\n".join(guidance)


def _allocated_paper_ids(state: dict[str, Any]) -> list[str]:
    """按计划顺序返回去重后的 paper_id，兼容旧版 paper_indices。"""
    result: list[str] = []
    ranked = state.get("ranked_papers") or state.get("paper_cards") or []
    index_to_id = {
        index: str(paper.get("paper_id") or "")
        for index, paper in enumerate(ranked, start=1)
    }
    for section in (state.get("citation_allocation_plan") or {}).get("sections") or []:
        paper_ids = [str(item) for item in section.get("paper_ids") or []]
        paper_ids.extend(
            index_to_id.get(index, "")
            for index in section.get("paper_indices") or []
            if isinstance(index, int)
        )
        for paper_id in paper_ids:
            if paper_id and paper_id not in result:
                result.append(paper_id)
    return result


def _allocation_by_section(
    plan: WritingPlan,
    state: dict[str, Any],
    cards: list[dict[str, Any]],
) -> dict[str, list[str]]:
    """把引用计划解析为 WritingSection.id -> paper_ids。"""
    available_ids = {str(card.get("paper_id") or "") for card in cards}
    ranked = state.get("ranked_papers") or state.get("paper_cards") or []
    index_to_id = {
        index: str(paper.get("paper_id") or "")
        for index, paper in enumerate(ranked, start=1)
    }
    section_by_title = {section.title: section for section in plan.sections}
    section_by_id = {section.id: section for section in plan.sections}
    allocations: dict[str, list[str]] = {section.id: [] for section in plan.sections}
    for raw in (state.get("citation_allocation_plan") or {}).get("sections") or []:
        section = section_by_id.get(str(raw.get("section_id") or ""))
        if section is None:
            section = section_by_title.get(str(raw.get("section") or ""))
        if section is None:
            continue
        allowed = set(section.supporting_paper_ids) & available_ids
        raw_ids = [str(item) for item in raw.get("paper_ids") or []]
        raw_ids.extend(
            index_to_id.get(index, "")
            for index in raw.get("paper_indices") or []
            if isinstance(index, int)
        )
        for paper_id in raw_ids:
            if (
                paper_id
                and paper_id in allowed
                and paper_id not in allocations[section.id]
            ):
                allocations[section.id].append(paper_id)
    return allocations


def _ensure_citation_coverage(
    text: str,
    plan: WritingPlan,
    cards: list[dict[str, Any]],
) -> str:
    """不为满足数量要求追加脱离章节语境的引用填充句。

    引用覆盖不足交给引用验证和最终质量门禁处理，避免用机械句式把
    未综合的论文伪装成正文证据。
    """
    return text




class BaseRenderer:
    """交付物渲染器基类。"""

    def __init__(self, deliverable_type: CoreDeliverableType):
        self.deliverable_type = deliverable_type

    def render_fallback(
        self,
        plan: WritingPlan,
        state: dict[str, Any],
        cards: list[dict[str, Any]],
    ) -> str:
        """证据约束的降级写作器。用自然段落综合论文证据，不堆砌机械引用句式。"""
        claims_by_paper = {card["paper_id"]: card.get("claims") or [] for card in cards}
        synthesis_by_name = {
            str(item.get("theme_name")): item for item in state.get("theme_synthesis") or []
        }
        topic = str(state.get("canonical_topic") or state.get("topic") or "本研究主题")
        parts: list[str] = []
        seen_gap_statements: set[str] = set()
        theme_paper_ids = {
            str(paper_id)
            for item in plan.sections
            if item.id.startswith("theme_")
            for paper_id in item.supporting_paper_ids
        }

        for section_idx, section in enumerate(plan.sections):
            parts.append(_section_heading(section))

            # ── 固定章节 ──────────────────────────────────────────────────────────
            if section.id == "scope_definition":
                parts.append(
                    f"本节以“{topic}”作为规范研究主题；相邻概念仅在论文明确讨论其与该主题的关系时纳入，"
                    "不替换本次研究边界。"
                )
                continue
            if section.id == "search_scope":
                report = state.get("search_report") or {}
                sources = "、".join(report.get("sources") or []) or "已配置论文数据源"
                parts.append(
                    f"本次初稿检索范围为 {report.get('start_year') or '未限定'}—"
                    f"{report.get('end_year') or '未限定'} 年，"
                    f"使用{sources}进行召回，经去重与相关性筛选后形成 "
                    f"{report.get('writing_pool_count', 0)} 篇写作候选。"
                    "该过程不包含双人独立筛选、PRISMA流程或偏倚风险评价，"
                    "因此本文定位为叙述性综述初稿。"
                )
                continue
            if section.id == "evidence_statement":
                limitations = (state.get("evidence_quality_report") or {}).get("limitations") or []
                parts.append(
                    " ".join(limitations)
                    or "当前证据质量以系统实际获得的摘要和全文为准，未获得内容不作补全。"
                )
                continue
            if section.id == "status_overview":
                overview_claims = []
                for p_id in section.supporting_paper_ids:
                    # 主题节会承担这些论文的事实陈述；概述不再复述同一句证据。
                    if str(p_id) in theme_paper_ids:
                        continue
                    for c in claims_by_paper.get(p_id, []):
                        if c.get("field") in {"research_problem", "method", "results"} and c.get("claim"):
                            claim = _neutralize_evidence_self_reference(c["claim"])
                            overview_claims.append(f"{claim}[{p_id}]")
                            break
                lead = _status_overview_lead(topic, _route_titles(plan))
                if overview_claims:
                    body = "；".join(overview_claims) + "。"
                    parts.append(lead + f"代表性证据分别涉及：{body}")
                else:
                    parts.append(lead + "现有材料可从相关证据所覆盖的对象、问题与分析方式加以梳理。")
                continue

            # ── 主题综合章节（research_status / related_work 的分主题节） ─────────
            synthesis = synthesis_by_name.get(section.title)
            if synthesis:
                para_parts: list[str] = []
                seen_theme_pids: set[str] = set()

                # 研究问题：综合多篇
                problems = (synthesis.get("reported_problems") or []) + (synthesis.get("common_problems") or [])
                prob_sents = []
                for item in problems:
                    claim = _neutralize_evidence_self_reference(
                        item.get("claim_text") or item.get("claim") or item.get("statement") or item.get("reported_problem") or ""
                    ).strip()
                    pid = str(item.get("paper_id") or "")
                    if claim and pid and pid in section.supporting_paper_ids:
                        prob_sents.append(f"{claim}[{pid}]")
                        seen_theme_pids.add(pid)
                if prob_sents:
                    lead = "在研究问题方面，" + prob_sents[0] + "。"
                    extra = "此外，" + "；".join(prob_sents[1:]) + "。" if len(prob_sents) > 1 else ""
                    para_parts.append(lead + extra)

                # 方法路线：综合多篇
                methods = (synthesis.get("reported_methods") or []) + (synthesis.get("common_methods") or [])
                meth_sents = []
                for item in methods:
                    claim = _neutralize_evidence_self_reference(
                        item.get("claim_text") or item.get("claim") or item.get("statement") or item.get("method_name") or ""
                    ).strip()
                    pid = str(item.get("paper_id") or "")
                    if claim and pid and pid in section.supporting_paper_ids and pid not in seen_theme_pids:
                        meth_sents.append(f"{claim}[{pid}]")
                        seen_theme_pids.add(pid)
                if meth_sents:
                    lead = "在方法路线方面，" + meth_sents[0] + "。"
                    extra = "同时，" + "；".join(meth_sents[1:]) + "。" if len(meth_sents) > 1 else ""
                    para_parts.append(lead + extra)

                # 实验结果
                findings = synthesis.get("reported_findings") or []
                find_sents = [
                    f"{_neutralize_evidence_self_reference(item.get('claim', '')).strip()}[{item.get('paper_id', '')}]"
                    for item in findings
                    if item.get("claim") and item.get("paper_id") and item.get("paper_id") in section.supporting_paper_ids and item.get("paper_id") not in seen_theme_pids
                ]
                if find_sents:
                    para_parts.append("在实验结果方面，" + "；".join(find_sents) + "。")
                    for item in findings:
                        seen_theme_pids.add(item.get("paper_id"))

                # 覆盖本节剩余论文
                remaining_pids = [pid for pid in section.supporting_paper_ids if pid not in seen_theme_pids]
                if remaining_pids:
                    rem_sents = []
                    for pid in remaining_pids:
                        for c in claims_by_paper.get(pid, []):
                            ctext = _neutralize_evidence_self_reference(
                                c.get("claim") or c.get("statement") or ""
                            ).strip()
                            if ctext:
                                rem_sents.append(f"{ctext}[{pid}]")
                                seen_theme_pids.add(pid)
                                break
                    if rem_sents:
                        para_parts.append("在应用探索与相关实践方面，" + "；".join(rem_sents) + "。")

                # 研究空白（避免不同章节输出完全相同的文字）
                gaps = synthesis.get("synthesized_gaps") or []
                gap_labels = {
                    "author_reported": "作者指出",
                    "cross_paper_inference": "综合来看",
                    "evidence_access_limitation": "受限于证据获取",
                }
                gap_sents = []
                for item in gaps:
                    if isinstance(item, dict) and item.get("statement"):
                        stmt = str(item.get("statement") or "").strip()
                        if stmt not in seen_gap_statements:
                            seen_gap_statements.add(stmt)
                            gap_sents.append(f"{gap_labels.get(str(item.get('gap_type', '')), '综合来看')}，{stmt}")
                if gap_sents:
                    para_parts.append("在研究局限与空白方面，" + "；".join(gap_sents) + "。")

                # 如果是 research_status 的最后一个主题章节，追加跨路线综合
                is_last_theme = (section_idx == len(plan.sections) - 1) or (
                    section_idx < len(plan.sections) - 1 and not plan.sections[section_idx + 1].id.startswith("theme_")
                )
                if plan.deliverable_type == CoreDeliverableType.RESEARCH_STATUS and is_last_theme:
                    para_parts.append(
                        _cross_route_summary(
                            topic,
                            _route_titles(plan),
                            list(state.get("theme_synthesis") or []),
                        )
                    )

                parts.append(
                    "\n\n".join(para_parts)
                    if para_parts
                    else "当前可访问证据不足以对该研究路线作细粒度分析。"
                )
                continue

            # ── 通用章节：从 paper_cards 整合证据，生成连贯段落 ──────────────────
            preferred_fields = {
                "problem_context": {"research_problem"},
                "importance": {"research_problem", "contributions"},
                "existing_approaches": {"method", "results"},
                "research_need": {"research_problem", "method"},
                "comparison": {"method", "results", "metrics"},
                "research_gaps": {"research_problem", "limitations"},
            }.get(section.id) or {"research_problem", "method", "results"}

            field_claims: dict[str, list[tuple[str, str]]] = {}
            for paper_id in section.supporting_paper_ids:
                for claim in claims_by_paper.get(paper_id, []):
                    field = str(claim.get("field") or "")
                    claim_text = _neutralize_evidence_self_reference(
                        claim.get("claim") or ""
                    ).strip()
                    if field in preferred_fields and claim_text and claim.get("explicitly_reported"):
                        field_claims.setdefault(field, []).append((paper_id, claim_text))

            if not field_claims:
                for paper_id in section.supporting_paper_ids:
                    for claim in claims_by_paper.get(paper_id, []):
                        claim_text = _neutralize_evidence_self_reference(
                            claim.get("claim") or ""
                        ).strip()
                        if claim_text and claim.get("explicitly_reported"):
                            field_claims.setdefault("general", []).append((paper_id, claim_text))

            included_pids = {pid for items in field_claims.values() for pid, _ in items}
            for paper_id in section.supporting_paper_ids:
                if paper_id not in included_pids:
                    for claim in claims_by_paper.get(paper_id, []):
                        claim_text = _neutralize_evidence_self_reference(
                            claim.get("claim") or ""
                        ).strip()
                        if claim_text:
                            field_claims.setdefault("general", []).append((paper_id, claim_text))
                            break

            if field_claims:
                para_sents: list[str] = []
                for fld, items in list(field_claims.items()):
                    sents = [f"{c}[{pid}]" for pid, c in items]
                    para_sents.append("；".join(sents))

                if len(para_sents) == 1:
                    paragraph = para_sents[0] + "。"
                else:
                    paragraph = para_sents[0] + "。此外，" + "。另外，".join(para_sents[1:]) + "。"

                if (
                    plan.deliverable_type == CoreDeliverableType.RESEARCH_BACKGROUND
                    and section.id == "problem_context"
                ):
                    paragraph = f"在{topic}领域，" + paragraph

                parts.append(paragraph)
            else:
                parts.append(f"当前检索证据不足以支撑{section.title}的详细叙述，保留章节框架供后续补充。")

            if plan.deliverable_type == CoreDeliverableType.RELATED_WORK and section.id == "gap_and_positioning":
                profile = state.get("user_paper_profile") or {}
                method = profile.get("proposed_method") or profile.get("research_direction") or "（未提供）"
                prob = profile.get("research_problem") or "（未提供）"
                parts.append(
                    f"综上，现有研究在{prob}方面已取得一定进展，但仍存在上述不足。"
                    f"本研究采用{method}，旨在填补上述研究空白。"
                    "以上定位基于用户提供的信息，不推断未提供的贡献或性能优势。"
                )

        rendered = "\n\n".join(parts)
        section_records = _split_fallback_sections(rendered, plan)
        deduplicated = _deduplicate_fallback_claims(section_records, plan, state)
        return "\n\n".join(text for _section_id, text in deduplicated)

    def render(
        self,
        plan: WritingPlan,
        state: dict[str, Any],
        cards: list[dict[str, Any]],
        safe_synthesis: list[dict[str, Any]],
        llm=None,
    ) -> str:
        """执行统一交付物生成策略：尝试模型逐节/整篇生成，未通过时回退到确定性渲染器。"""
        spec = get_deliverable_spec(plan.deliverable_type)
        display_names = {
            CoreDeliverableType.RESEARCH_BACKGROUND: "研究背景",
            CoreDeliverableType.RESEARCH_STATUS: "研究现状",
            CoreDeliverableType.RELATED_WORK: "论文相关工作",
            CoreDeliverableType.NARRATIVE_REVIEW: "叙述性综述初稿",
        }
        forbidden_names = [
            name for dtype, name in display_names.items() if dtype != plan.deliverable_type
        ]
        deliverable_boundary = (
            f"只生成 {plan.deliverable_type.value}（{display_names[plan.deliverable_type]}）。"
            f"不得把正文命名或包装为：{'、'.join(forbidden_names)}。"
        )
        minimum_references = int(
            plan.citation_policy.get("minimum_unique_references") or 0
        )
        safe_state = {**state, "theme_synthesis": safe_synthesis}
        plan_blueprints = get_plan_blueprints(plan)
        style_guidance = _style_guidance(plan)

        global_reference_target = max(
            0,
            int(state.get("required_reference_count") or 0),
        )
        if llm is not None and (
            minimum_references >= 20 or global_reference_target >= 20
        ):
            fallback = self.render_fallback(plan, safe_state, cards)
            sectionwise = _write_sections_in_chinese(
                fallback, plan, state, cards, llm
            )
            sectionwise = _strip_evidence_meta_language(sectionwise)
            sectionwise = _strip_stray_numeric_citations(sectionwise)
            from app.tools.validate_deliverable import validate_deliverable

            sectionwise_validation = validate_deliverable(sectionwise, plan, state)
            _record_writer_diagnostic(
                state,
                plan,
                strategy="sectionwise_chinese_synthesis",
                validation=sectionwise_validation,
            )
            # 校验结果只作为诊断记录（_record_writer_diagnostic），不在这里
            # 静默丢弃已生成的分节文本；引用缺口与质量问题由下游校验和
            # 最终质量门禁如实报告。此前这里有一个两分支都 return 的死条件。
            return sectionwise

        llm_responded = False
        survey_papers = _survey_papers(cards)
        if llm is not None:
            try:
                from app.prompt.writing.deliverable import WRITER_PROMPT

                writer_prompt = WRITER_PROMPT.format(
                    spec_json=spec.model_dump_json(),
                    deliverable_boundary=deliverable_boundary,
                    plan_json=plan.model_dump_json(),
                    claims_json=json.dumps(cards, ensure_ascii=False),
                    synthesis_json=json.dumps(safe_synthesis, ensure_ascii=False),
                    search_report_json=json.dumps(state.get("search_report") or {}, ensure_ascii=False),
                    profile_json=json.dumps(state.get("user_paper_profile") or {}, ensure_ascii=False),
                    canonical_topic=str(state.get("canonical_topic") or state.get("topic") or ""),
                    evidence_report_json=json.dumps(state.get("evidence_quality_report") or {}, ensure_ascii=False),
                    citation_plan_json=json.dumps(state.get("citation_allocation_plan") or {}, ensure_ascii=False),
                    style_guidance=style_guidance,
                    blueprint_json=json.dumps(plan_blueprints, ensure_ascii=False),
                    survey_papers_json=json.dumps(survey_papers, ensure_ascii=False),
                )
                text = llm.complete(
                    writer_prompt,
                    temperature=0.1,
                    operation=f"write_deliverable:{plan.deliverable_type.value}",
                    thinking_enabled=True,
                )
                llm_responded = bool(text)
                if text and all(_section_heading(section) in text for section in plan.sections):
                    normalized = _strip_stray_numeric_citations(
                        _strip_evidence_meta_language(
                            _normalize_evidence_citations(text.strip(), cards)
                        )
                    )
                    from app.tools.validate_deliverable import validate_deliverable
                    validation = validate_deliverable(normalized, plan, state)
                    citation_ok = _citation_target_met(normalized, plan)

                    if validation.get("valid") and citation_ok:
                        return normalized

                    blocking_errors = [
                        e for e in (validation.get("errors") or [])
                        if not any(kw in e for kw in [
                            "计划外章节", "出现计划外章节", "过多实验指标"
                        ])
                    ]
                    if not blocking_errors and citation_ok and len(normalized) > 800:
                        return normalized

                    required_ids = _allocated_paper_ids(state)
                    cited_ids = set(re.findall(r"\[([^\]]+)\]", normalized))
                    missing_ids = [
                        paper_id for paper_id in required_ids if paper_id not in cited_ids
                    ]
                    repair_prompt = (
                        writer_prompt
                        + "\n\n【首稿机器校验未通过，请完整重写，不要解释】\n"
                        + f"校验错误：{json.dumps(validation.get('errors') or [], ensure_ascii=False)}\n"
                        + f"尚未引用的计划论文：{json.dumps(missing_ids, ensure_ascii=False)}\n"
                        + "必须重新输出全部计划章节；逐章落实 citation_allocation_plan 中的 paper_ids，"
                        + "保持中文综合表达，不复制英文原句，不得在文末追加引用凑数段。\n"
                        + "首稿仅供识别问题，不得直接沿用其不合格结构：\n"
                        + normalized[:12000]
                    )
                    repaired = llm.complete(
                        repair_prompt,
                        temperature=0.05,
                        operation=f"repair_deliverable:{plan.deliverable_type.value}",
                        thinking_enabled=True,
                    )
                    llm_responded = llm_responded or bool(repaired)
                    if repaired and all(
                        _section_heading(section) in repaired for section in plan.sections
                    ):
                        repaired = _strip_stray_numeric_citations(
                            _strip_evidence_meta_language(
                                _normalize_evidence_citations(repaired.strip(), cards)
                            )
                        )
                        repaired_validation = validate_deliverable(repaired, plan, state)
                        if repaired_validation.get("valid") and _citation_target_met(
                            repaired, plan
                        ):
                            return repaired
            except Exception:
                pass
        fallback = self.render_fallback(plan, safe_state, cards)
        fallback = _ensure_citation_coverage(fallback, plan, cards)
        if llm is not None and llm_responded:
            try:
                from app.tools.validate_deliverable import validate_deliverable

                fallback_validation = validate_deliverable(fallback, plan, state)
                if not fallback_validation.get("valid"):
                    required_ids = _allocated_paper_ids(state)
                    polish_prompt = f"""你是中文学术编辑。请将下面的证据约束草稿完整改写为流畅的中文综合段落。

硬性要求：
1. 保留全部计划标题、标题层级及其顺序，不增加总标题或计划外章节。
2. 英文证据必须忠实转述为中文；模型名、缩写和数据集名可保留英文。
3. 必须保留以下每一个引用编号，不能遗漏、替换或新增：
{json.dumps(required_ids, ensure_ascii=False)}
4. 不得逐篇罗列，不得出现“论文明确报告”或文末引用凑数段；应在每节内按共同问题、方法或发现综合组织。
5. 只使用草稿已有事实，不补充推测，不输出解释或修改说明。

以下脱敏少样本只示范写法，不是事实证据；不得输出其中的〈占位内容〉或〔证据A〕：
{json.dumps(plan_blueprints, ensure_ascii=False)}

草稿：
{fallback}
"""
                    polished = llm.complete(
                        polish_prompt,
                        temperature=0.05,
                        operation=f"polish_fallback:{plan.deliverable_type.value}",
                        thinking_enabled=True,
                    )
                    if polished and all(
                        _section_heading(section) in polished for section in plan.sections
                    ):
                        polished = _strip_stray_numeric_citations(
                            _strip_evidence_meta_language(
                                _normalize_evidence_citations(polished.strip(), cards)
                            )
                        )
                        polished_validation = validate_deliverable(polished, plan, state)
                        if polished_validation.get("valid") and _citation_target_met(
                            polished, plan
                        ):
                            return polished
            except Exception:
                pass
        return fallback
