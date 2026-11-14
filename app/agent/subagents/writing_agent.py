"""写作专业 Agent。"""

from __future__ import annotations

from typing import Any

from app.agent.subagents.base import BaseSubAgent, _GOAL_FIELDS
from app.schemas.agent_task_schema import AgentRole


class WritingAgent(BaseSubAgent):
    def _outcome(self, state, status, changed):
        from app.agent.quality_feedback import candidate_usability
        from app.schemas.agent_task_schema import AgentTaskOutcome, AgentTaskStatus
        if status == AgentTaskStatus.COMPLETED and not candidate_usability(state)["usable"]:
            return AgentTaskOutcome.BLOCKED
        return super()._outcome(state, status, changed)

    role = AgentRole.WRITING
    # 写作者只能消费证据，不能改写检索结果、证据卡或主张授权。
    protected_fields = _GOAL_FIELDS | {"check_report", "check_protocol_version", "autonomous_verified_fingerprint",
        "candidate_papers", "ranked_papers", "paper_details", "paper_cards",
        "source_diagnostics", "claim_plans", "evidence_gap_report",
        "evidence_snapshot_version", "evidence_snapshot_fingerprint",
    }

    def summarize(
        self, state: dict[str, Any]
    ) -> tuple[list[str], list[str], list[dict[str, Any]]]:
        from app.agent.quality_feedback import candidate_usability
        candidate = candidate_usability(state)
        findings = [
            f"生成 {len(state.get('writing_plans') or [])} 个写作计划，正文 {len(str(state.get('review') or ''))} 字符",
            f"生成状态：{'unusable' if not candidate['usable'] else 'candidate_requires_validation'}",
            *candidate["errors"],
        ]
        refs = [
            f"state://writing-plan/{index}"
            for index, _ in enumerate(state.get("writing_plans") or [])
        ]
        from app.services.writer_support import writer_issues
        questions = [{'category': i['code'], 'question': i['reason'], 'blocking': True}
                     for i in writer_issues(state)]
        refs.extend(f"state://writer-candidate/{r['candidate_id']}" for r in (state.get('writer_candidates') or [])[-4:])
        questions += [
            {
                "category": "quality_gate",
                "question": str(item.get("message") or item.get("code") or "写作质量问题"),
                "blocking": True,
            }
            for item in (state.get("quality_gate") or {}).get("blocking_issues") or []
        ][:10]
        return findings, refs, questions
