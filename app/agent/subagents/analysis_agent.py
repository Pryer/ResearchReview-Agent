"""证据分析专业 Agent。"""

from __future__ import annotations

from typing import Any

from app.agent.subagents.base import BaseSubAgent
from app.schemas.agent_task_schema import AgentRole


class AnalysisAgent(BaseSubAgent):
    role = AgentRole.ANALYSIS

    def summarize(
        self, state: dict[str, Any]
    ) -> tuple[list[str], list[str], list[dict[str, Any]]]:
        cards = state.get("paper_cards") or []
        routes = [item for item in state.get("validated_routes") or [] if item.get("paper_ids")]
        claim_count = sum(len(item.get("claims") or []) for item in state.get("claim_plans") or [])
        findings = [
            f"形成 {len(cards)} 张证据卡、{len(routes)} 条有证据路线、{claim_count} 条计划主张",
        ]
        refs = [
            f"state://paper-card/{card.get('paper_id')}"
            for card in cards[:20] if card.get("paper_id")
        ]
        questions = [
            {
                "category": "evidence_gap",
                "question": str(item.get("reason") or item.get("gap_type") or "证据缺口"),
                "blocking": bool((state.get("evidence_gap_report") or {}).get("needs_recovery")),
            }
            for item in (state.get("evidence_gap_report") or {}).get("gaps") or []
        ][:10]
        return findings, refs, questions
