"""国内外研究现状渲染器 (RESEARCH_STATUS)。"""

from __future__ import annotations

from typing import Any
from app.schemas.deliverable_schema import CoreDeliverableType, WritingPlan
from app.deliverables.renderers.base_renderer import (
    BaseRenderer,
    _allocated_paper_ids_for_section,
    _claim_plan_claims_by_paper,
    _clean_route_title,
    _cross_route_summary,
    _neutralize_evidence_self_reference,
    _safe_claim_for_body,
    _section_heading,
    _status_overview_lead,
)

class StatusRenderer(BaseRenderer):
    def __init__(self):
        super().__init__(CoreDeliverableType.RESEARCH_STATUS)

    def render_fallback(
        self,
        plan: WritingPlan,
        state: dict[str, Any],
        cards: list[dict[str, Any]],
        safe_synthesis: list[dict[str, Any]] | None = None,
    ) -> str:
        import re

        topic = str(state.get("canonical_topic") or state.get("topic") or "本研究主题")
        claims_by_paper = {card["paper_id"]: card.get("claims") or [] for card in cards}
        claim_plan_claims_by_paper = _claim_plan_claims_by_paper(state)
        reused_paper_ids = {
            str(paper_id)
            for paper_id in (state.get("citation_allocation_plan") or {}).get("reused_paper_ids") or []
        }

        def _first_safe_claim(paper_id: str) -> str:
            if str(paper_id) in reused_paper_ids:
                return ""
            for text in claim_plan_claims_by_paper.get(str(paper_id)) or []:
                safe_text = _safe_claim_for_body(text)
                if safe_text:
                    return safe_text
            return ""

        # 构建灵活索引：支持 theme_name、section.title、去除编号后的 title、theme_id 匹配
        synthesis_by_key: dict[str, Any] = {}
        synthesis_pool = safe_synthesis if safe_synthesis else (state.get("theme_synthesis") or [])
        for item in synthesis_pool:
            tname = str(item.get("theme_name") or "")
            tid = str(item.get("theme_id") or "")
            if tname:
                synthesis_by_key[tname] = item
                clean_tname = re.sub(r"^[（(][一二三四五六七八九十0-9]+[）)]\s*", "", tname).strip()
                if clean_tname:
                    synthesis_by_key[clean_tname] = item
            if tid:
                synthesis_by_key[tid] = item
                synthesis_by_key[f"theme_{tid}"] = item

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

            if section.id == "status_overview":
                theme_titles = [
                    _clean_route_title(s.title)
                    for s in plan.sections if s.id.startswith("theme_")
                ]
                lead = _status_overview_lead(topic, theme_titles)
                overview_claims = []
                allocated_overview_ids = _allocated_paper_ids_for_section(state, section.id)
                overview_ids = (
                    allocated_overview_ids
                    if state.get("conservative_regeneration") and allocated_overview_ids
                    else list(dict.fromkeys([
                        *(section.supporting_paper_ids or []),
                        *allocated_overview_ids,
                    ]))
                )
                for p_id in overview_ids:
                    if (
                        not (state.get("conservative_regeneration") and claim_plan_claims_by_paper)
                        and str(p_id) in theme_paper_ids
                    ):
                        continue
                    if state.get("conservative_regeneration") and claim_plan_claims_by_paper:
                        raw_claim = _first_safe_claim(str(p_id))
                        if raw_claim:
                            overview_claims.append(f"{raw_claim}[{p_id}]")
                    else:
                        for c in claims_by_paper.get(p_id, []):
                            if c.get("field") in {"research_problem", "method", "results"} and c.get("claim"):
                                raw_claim = _neutralize_evidence_self_reference(
                                    c["claim"]
                                ).strip().rstrip("。，；、,.; ")
                                if raw_claim:
                                    overview_claims.append(f"{raw_claim}[{p_id}]")
                                    break
                if overview_claims:
                    # 保守重写需要兑现当前分配层的全局覆盖；只在普通兜底
                    # 路径限制概述长度，避免将计划论文静默截掉。
                    visible_claims = (
                        overview_claims
                        if state.get("conservative_regeneration") and claim_plan_claims_by_paper
                        else overview_claims[:8]
                    )
                    if state.get("conservative_regeneration") and claim_plan_claims_by_paper:
                        parts.append("\n".join(f"{claim}。" for claim in visible_claims))
                    else:
                        body = "；".join(visible_claims) + "。"
                        parts.append(lead + f"从已有代表性成果看，相关研究在核心机制与性能边界上取得了持续进展：{body}")
                else:
                    parts.append(lead + "现有材料可按各研究路线的问题设定、分析方式与适用边界加以梳理。")
                continue

            # 保守重写只使用当前 Claim Plan 已授权的证据句，并合并分配层
            # 补充到本节的论文；不能复用上一轮 theme_synthesis 的旧主张。
            if state.get("conservative_regeneration") and claim_plan_claims_by_paper:
                allocated_ids = _allocated_paper_ids_for_section(state, section.id)
                section_paper_ids = (
                    allocated_ids
                    if allocated_ids
                    else list(dict.fromkeys(section.supporting_paper_ids or []))
                )
                authorized_points = [
                    (claim_text, str(paper_id))
                    for paper_id in section_paper_ids
                    for claim_text in [_first_safe_claim(str(paper_id))]
                    if claim_text
                    if claim_plan_claims_by_paper.get(str(paper_id))
                ]
                if authorized_points:
                    parts.append(
                        "\n".join(
                            f"{claim_text}[{paper_id}]。"
                            for claim_text, paper_id in authorized_points
                        )
                    )
                    if section_idx == len(plan.sections) - 1 or (
                        section_idx < len(plan.sections) - 1
                        and not plan.sections[section_idx + 1].id.startswith("theme_")
                    ):
                        # 跨路线总结必须位于最后一个主题段落末尾；否则后置的
                        # 证据边界句会成为校验器看到的“最后一段”，导致总结
                        # 已生成却仍被判定缺失。
                        if sum(len(text) for text, _paper_id in authorized_points) < 80:
                            parts.append(
                                f"本节围绕{_clean_route_title(section.title)}，仅依据已获授权论文明确报告的内容归纳，"
                                "不对未报告的信息作外推。"
                            )
                        theme_titles = [
                            _clean_route_title(s.title)
                            for s in plan.sections if s.id.startswith("theme_")
                        ]
                        parts.append(_cross_route_summary(topic, theme_titles, synthesis_pool))
                    # 短路线仍需说明证据边界，避免两篇授权证据被压缩成
                    # 不可读的单行清单；最后一节已在跨路线总结前追加。
                    elif sum(len(text) for text, _paper_id in authorized_points) < 80:
                        parts.append(
                            f"本节围绕{_clean_route_title(section.title)}，仅依据已获授权论文明确报告的内容归纳，"
                            "不对未报告的信息作外推。"
                        )
                    continue

            sec_clean_title = re.sub(r"^[（(][一二三四五六七八九十0-9]+[）)]\s*", "", section.title).strip()
            synthesis = (
                synthesis_by_key.get(section.id)
                or synthesis_by_key.get(section.id.removeprefix("theme_"))
                or synthesis_by_key.get(section.title)
                or synthesis_by_key.get(sec_clean_title)
            )

            if synthesis:
                para_parts: list[str] = []
                seen_theme_pids: set[str] = set()

                problems = (synthesis.get("reported_problems") or []) + (synthesis.get("common_problems") or [])
                methods = (synthesis.get("reported_methods") or []) + (synthesis.get("common_methods") or [])
                prob_sents = [
                    f"{_neutralize_evidence_self_reference(item.get('claim_text') or item.get('claim') or item.get('statement') or item.get('reported_problem') or '').strip().rstrip('。，；、,.; ')}[{str(item.get('paper_id') or '')}]"
                    for item in problems
                    if str(item.get('claim_text') or item.get('claim') or item.get('statement') or item.get('reported_problem') or '').strip()
                    and str(item.get('paper_id') or '') in section.supporting_paper_ids
                ]
                meth_sents = [
                    f"{_neutralize_evidence_self_reference(item.get('claim_text') or item.get('claim') or item.get('statement') or item.get('method_name') or '').strip().rstrip('。，；、,.; ')}[{str(item.get('paper_id') or '')}]"
                    for item in methods
                    if str(item.get('claim_text') or item.get('claim') or item.get('statement') or item.get('method_name') or '').strip()
                    and str(item.get('paper_id') or '') in section.supporting_paper_ids
                    and str(item.get('paper_id') or '') not in seen_theme_pids
                ]

                if prob_sents or meth_sents:
                    para1_chunks = []
                    if prob_sents:
                        for item in problems:
                            seen_theme_pids.add(str(item.get("paper_id") or ""))
                        para1_chunks.append("该路线主要聚焦于" + "，同时".join(prob_sents) + "。")
                    if meth_sents:
                        for item in methods:
                            seen_theme_pids.add(str(item.get("paper_id") or ""))
                        para1_chunks.append("在具体方法机制上，研究者通常采用" + "；另有工作提出".join(meth_sents) + "。")
                    para_parts.append("".join(para1_chunks))

                findings = synthesis.get("reported_findings") or []
                find_sents = [
                    f"{_neutralize_evidence_self_reference(item.get('claim', '')).strip().rstrip('。，；、,.; ')}[{item.get('paper_id', '')}]"
                    for item in findings
                    if item.get("claim") and item.get("paper_id") and item.get("paper_id") in section.supporting_paper_ids and item.get("paper_id") not in seen_theme_pids
                ]
                if find_sents:
                    for item in findings:
                        seen_theme_pids.add(item.get("paper_id"))
                    para_parts.append("在基准评测与实验验证层面，已有工作展示了显著的性能增益：" + "；".join(find_sents) + "。")

                remaining_pids = [pid for pid in section.supporting_paper_ids if pid not in seen_theme_pids]
                if remaining_pids:
                    rem_sents = []
                    for pid in remaining_pids:
                        for c in claims_by_paper.get(pid, []):
                            ctext = _neutralize_evidence_self_reference(
                                c.get("claim") or c.get("statement") or ""
                            ).strip().rstrip("。，；、,.; ")
                            if ctext:
                                rem_sents.append(f"{ctext}[{pid}]")
                                seen_theme_pids.add(pid)
                                break
                    if rem_sents:
                        para_parts.append("此外，相关探索还进一步延伸至具体场景与结构扩展：" + "；".join(rem_sents) + "。")

                gaps = synthesis.get("synthesized_gaps") or []
                gap_labels = {
                    "author_reported": "作者指出",
                    "cross_paper_inference": "综合来看",
                    "evidence_access_limitation": "受限于证据获取",
                }
                gap_sents = []
                for item in gaps:
                    if isinstance(item, dict) and item.get("statement"):
                        stmt = str(item.get("statement") or "").strip().rstrip("。，；、,.; ")
                        if stmt and stmt not in seen_gap_statements:
                            seen_gap_statements.add(stmt)
                            gap_sents.append(f"{gap_labels.get(str(item.get('gap_type', '')), '综合来看')}，{stmt}")
                if gap_sents:
                    para_parts.append("从适用边界与现存局限分析，" + "；".join(gap_sents) + "。")

                is_last_theme = (section_idx == len(plan.sections) - 1) or (
                    section_idx < len(plan.sections) - 1 and not plan.sections[section_idx + 1].id.startswith("theme_")
                )
                if is_last_theme:
                    theme_titles = [
                        _clean_route_title(s.title)
                        for s in plan.sections if s.id.startswith("theme_")
                    ]
                    route_syntheses = [
                        item for item in synthesis_pool
                        if str(item.get("theme_name") or "") in theme_titles
                        or str(item.get("theme_id") or "") in {
                            s.id.removeprefix("theme_")
                            for s in plan.sections if s.id.startswith("theme_")
                        }
                    ]
                    para_parts.append(_cross_route_summary(topic, theme_titles, route_syntheses))

                parts.append(
                    "\n\n".join(para_parts)
                    if para_parts
                    else "当前可访问证据不足以对该研究路线作细粒度分析。"
                )
            else:
                allocated = [c for c in cards if c.get("paper_id") in section.supporting_paper_ids]
                evidence_claims = []
                for card in allocated:
                    claim = next(
                        (
                            _neutralize_evidence_self_reference(
                                item.get("claim") or item.get("statement") or ""
                            ).strip()
                            for item in card.get("claims") or []
                            if str(item.get("claim") or item.get("statement") or "").strip()
                        ),
                        "",
                    )
                    if claim:
                        evidence_claims.append(f"{claim.rstrip('。！？.!?')}[{card['paper_id']}]。")
                if evidence_claims:
                    parts.append("；".join(evidence_claims))
                else:
                    parts.append(f"围绕{topic}，当前材料尚未提供可用于概括该方向的具体证据。")

        return "\n\n".join(parts)
