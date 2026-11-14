"""只提交核验报告的独立专业 Agent。"""
from app.agent.subagents.base import BaseSubAgent, _GOAL_FIELDS
from app.schemas.agent_task_schema import AgentRole, AgentTaskOutcome, AgentTaskStatus


class CheckAgent(BaseSubAgent):
    role = AgentRole.CHECK
    protected_fields = _GOAL_FIELDS | {
        "review", "related_work", "introduction", "writing_plans", "paper_cards",
        "paper_details", "claim_plans", "candidate_papers", "ranked_papers",
        "citation_source_text", "citation_rendered_text", "writing_version",
    }

    def _outcome(self, state, status, changed):
        if status != AgentTaskStatus.COMPLETED:
            return AgentTaskOutcome.UNKNOWN
        report = state.get("check_report") or {}
        if report.get("execution_status") != "completed":
            return AgentTaskOutcome.BLOCKED
        return (AgentTaskOutcome.SUCCEEDED if report.get("verdict") == "passed"
                else AgentTaskOutcome.DEGRADED)

    def summarize(self, state):
        report = state.get("check_report") or {}
        return ([f"核验状态：{report.get('execution_status', 'unknown')}；结论：{report.get('verdict', 'unknown')}",
                 f"待处理问题 {len(report.get('issues') or [])} 项"],
                [f"state://check-report/{report['report_id']}"] if report.get("report_id") else [],
                [{"category": "check", "question": issue["reason"], "blocking": issue["severity"] == "blocking"}
                 for issue in (report.get("issues") or [])[:8]])
