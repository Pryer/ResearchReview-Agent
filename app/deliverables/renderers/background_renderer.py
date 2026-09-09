"""研究背景渲染器 (RESEARCH_BACKGROUND)。"""

from __future__ import annotations

import re
from typing import Any
from app.schemas.deliverable_schema import CoreDeliverableType, WritingPlan
from app.deliverables.renderers.base_renderer import (
    BaseRenderer,
    _allocated_paper_ids_for_section,
    _claim_plan_claims_by_paper,
    _neutralize_evidence_self_reference,
    _safe_claim_for_body,
    _safe_title_for_body,
    _section_heading,
)

# 单处引用点最多携带的文献数；兜底文本同样遵守引用密度约束，
# 不允许把整组论文成排钉在通用句上（那只是把数量要求伪装成证据覆盖）。
_MAX_CITATIONS_PER_POINT = 3
_BACKGROUND_DETAIL_RE = re.compile(
    r"准确率|召回率|精确率|\bmAP\b|\bF1\b|提升\s*\d|降低\s*\d|\d+(?:\.\d+)?%",
    re.I,
)


class BackgroundRenderer(BaseRenderer):
    def __init__(self):
        super().__init__(CoreDeliverableType.RESEARCH_BACKGROUND)

    def render_fallback(
        self,
        plan: WritingPlan,
        state: dict[str, Any],
        cards: list[dict[str, Any]],
    ) -> str:
        topic = str(state.get("canonical_topic") or state.get("topic") or "本研究主题")
        cards_by_pid = {
            str(card.get("paper_id")): card for card in cards if card.get("paper_id")
        }
        claim_plan_claims_by_paper = _claim_plan_claims_by_paper(state)
        parts: list[str] = []

        def _cite(pids: list[str]) -> str:
            kept = [str(p) for p in pids[:_MAX_CITATIONS_PER_POINT]]
            return f"[{', '.join(kept)}]" if kept else ""

        def _authorized_background_claim(pid: str) -> str:
            """选择可放入背景的授权主张，过滤单篇实验指标细节。"""
            candidates = claim_plan_claims_by_paper.get(pid) or []
            for text in candidates:
                if _BACKGROUND_DETAIL_RE.search(text):
                    continue
                safe_text = _safe_claim_for_body(text)
                if safe_text:
                    return safe_text
            return ""

        for section in plan.sections:
            parts.append(_section_heading(section))
            all_pids = list(dict.fromkeys(
                str(p) for p in [
                    *(section.supporting_paper_ids or []),
                    *_allocated_paper_ids_for_section(state, section.id),
                ]
                if str(p or "")
            )) or [str(c.get("paper_id")) for c in cards]
            ordered_cards = (
                [cards_by_pid[p] for p in all_pids if p in cards_by_pid] or cards
            )

            first = ordered_cards[0] if ordered_cards else {}
            first_title = _safe_title_for_body(first.get("title"))
            p1 = ""
            authorized_points = [
                (claim_text, pid)
                for card in ordered_cards
                for pid in [str(card.get("paper_id") or "")]
                for claim_text in [_authorized_background_claim(pid)]
                if claim_text
            ]
            if not claim_plan_claims_by_paper and first_title:
                p1 = f"《{first_title}》{_cite([str(first.get('paper_id'))])} 直接讨论了{topic}。"

            # 逐点归因：每个陈述句只引用它自己的来源卡片。此前把前 6 张卡
            # 的文本拼成一句、再统一钉上前 3 张卡的 pid，甲论文的问题/局限
            # 会被挂到乙论文头上（引用错位）。
            problem_points: list[tuple[str, str]] = []
            for card in ordered_cards:
                pid = str(card.get("paper_id") or "")
                if claim_plan_claims_by_paper:
                    text = _authorized_background_claim(pid)
                else:
                    text = _neutralize_evidence_self_reference(
                        card.get("research_problem") or ""
                    ).strip()
                if text and pid:
                    problem_points.append((text, pid))
                if len(problem_points) >= _MAX_CITATIONS_PER_POINT:
                    break
            if problem_points:
                separator = "\n" if claim_plan_claims_by_paper else "；"
                p2 = (
                    f"围绕{topic}，已有文献报告的研究问题包括："
                    + separator.join(f"{text}{_cite([pid])}" for text, pid in problem_points)
                    + "。"
                )
            else:
                p2 = "各文献的具体问题设定与方法细节以其原文报告为准，本节不作外推。"

            limitation_points: list[tuple[str, str]] = []
            if not claim_plan_claims_by_paper:
                for card in ordered_cards:
                    pid = str(card.get("paper_id") or "")
                    for item in (card.get("limitations") or []):
                        text = _neutralize_evidence_self_reference(item).strip()
                        if text and pid:
                            limitation_points.append((text, pid))
                        if len(limitation_points) >= _MAX_CITATIONS_PER_POINT:
                            break
                    if len(limitation_points) >= _MAX_CITATIONS_PER_POINT:
                        break
            if limitation_points:
                p3 = (
                    "作者明确报告的局限包括："
                    + "；".join(f"{text}{_cite([pid])}" for text, pid in limitation_points)
                    + "。"
                )
            else:
                p3 = "现有证据的适用范围以各文献原文报告为准。"

            # 引用覆盖列表：逐篇一句、每句一个引用，保证最低引用数要求
            # 在兜底模式下也能达成，且不产生任何单点多篇的堆砌。
            listing_lines = []
            if claim_plan_claims_by_paper:
                used = {pid for _text, pid in problem_points}
                for text, pid in authorized_points:
                    if pid not in used:
                        listing_lines.append(f"{text}[{pid}]。")
            else:
                for card in ordered_cards:
                    pid = str(card.get("paper_id") or "")
                    title = _safe_title_for_body(card.get("title"))
                    if pid and title:
                        listing_lines.append(f"《{title}》[{pid}]。")
            if listing_lines:
                p4 = "本节纳入的文献包括：" + " ".join(listing_lines)
                parts.append("\n\n".join(item for item in (p1, p2, p3, p4) if item))
            else:
                parts.append("\n\n".join(item for item in (p1, p2, p3) if item))

        return "\n\n".join(parts)
