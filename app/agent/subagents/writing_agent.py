"""写作专业 Agent。"""

from __future__ import annotations

from typing import Any

from app.agent.subagents.base import BaseSubAgent, _GOAL_FIELDS
from app.schemas.agent_task_schema import AgentRole


class WritingAgent(BaseSubAgent):
    role = AgentRole.WRITING
    # 写作者只能消费证据，不能改写检索结果、证据卡或主张授权。
    protected_fields = _GOAL_FIELDS | {
        "candidate_papers", "ranked_papers", "paper_details", "paper_cards",
        "source_diagnostics", "claim_plans", "evidence_gap_report",
        "evidence_snapshot_version", "evidence_snapshot_fingerprint",
    }

    def summarize(
        self, state: dict[str, Any]
    ) -> tuple[list[str], list[str], list[dict[str, Any]]]:
        findings = [
            f"生成 {len(state.get('writing_plans') or [])} 个写作计划，正文 {len(str(state.get('review') or ''))} 字符",
            f"生成状态：{'blocked' if state.get('generation_blocked') else 'available'}",
        ]
        refs = [
            f"state://writing-plan/{index}"
            for index, _ in enumerate(state.get("writing_plans") or [])
        ]
        questions = [
            {
                "category": "quality_gate",
                "question": str(item.get("message") or item.get("code") or "写作质量问题"),
                "blocking": True,
            }
            for item in (state.get("quality_gate") or {}).get("blocking_issues") or []
        ][:10]
        return findings, refs, questions
