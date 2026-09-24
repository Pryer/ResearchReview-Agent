"""检索专业 Agent。"""

from __future__ import annotations

from typing import Any

from app.agent.subagents.base import BaseSubAgent
from app.schemas.agent_task_schema import AgentRole


class SearchAgent(BaseSubAgent):
    role = AgentRole.SEARCH

    def summarize(
        self, state: dict[str, Any]
    ) -> tuple[list[str], list[str], list[dict[str, Any]]]:
        candidates = state.get("candidate_papers") or []
        ranked = state.get("ranked_papers") or []
        findings = [
            f"检索候选 {len(candidates)} 篇，排序后保留 {len(ranked)} 篇",
            f"停止原因：{state.get('retrieval_stop_reason') or '未记录'}",
        ]
        refs = [
            f"state://paper/{paper.get('paper_id')}"
            for paper in ranked[:20] if paper.get("paper_id")
        ]
        questions = []
        if state.get("search_failed"):
            questions.append({
                "category": "source_failure",
                "question": "检索源失败，是否存在可重试或需要用户处理的数据源？",
                "blocking": not bool(candidates),
            })
        return findings, refs, questions
