"""不可变研究资料注册表；版本、角色与载荷边界由代码裁决。"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ArtifactSpec:
    roles: frozenset[str]
    schema_version: int = 1
    max_bytes: int = 262144


ALL = frozenset({"main", "search", "analysis", "writing", "check", "service"})
ARTIFACT_TYPES = {
    "main_agent_context": ArtifactSpec(ALL),
    "conversation_event": ArtifactSpec(frozenset({"service"})),
    "conversation_summary": ArtifactSpec(frozenset({"main", "service"})),
    "raw_search_result": ArtifactSpec(frozenset({"search", "analysis", "service"})),
    "paper_metadata": ArtifactSpec(ALL),
    "document_fragment": ArtifactSpec(frozenset({"search", "analysis", "service", "check"})),
    "paper_card": ArtifactSpec(ALL),
    "claim_plan": ArtifactSpec(frozenset({"analysis", "writing", "service", "check"})),
    "writing_plan": ArtifactSpec(frozenset({"analysis", "writing", "service", "check"})),
    "draft": ArtifactSpec(frozenset({"analysis", "writing", "service", "check"})),
    "quarantined_generation": ArtifactSpec(frozenset({"service"})),
    "task_result": ArtifactSpec(ALL),
    "context_trajectory": ArtifactSpec(frozenset({"service"})),
    "context_epoch": ArtifactSpec(frozenset({"service"})),
    "context_archive_index": ArtifactSpec(frozenset({"service"})),
    "tool_observation": ArtifactSpec(frozenset({"service"})),
    "evidence_bundle": ArtifactSpec(ALL),
    "writer_candidate": ArtifactSpec(frozenset({"main", "analysis", "writing", "check", "service"})),
    "published_delivery": ArtifactSpec(frozenset({"service"})),
    "execution_cache": ArtifactSpec(frozenset({"service"})),
    "execution_report": ArtifactSpec(frozenset({"service"})),
    "check_report": ArtifactSpec(frozenset({"main", "analysis", "writing", "check", "service"})),
}

# WHY: 按项目独立外置，正文按片段保存；旧节点仍在任务副本中使用原有字段。
STATE_ARTIFACT_FIELDS = {
    "published_versions": "published_delivery", "extraction_cache": "execution_cache",
    "search_query_ledger": "execution_cache",
    "candidate_papers": "raw_search_result", "ranked_papers": "raw_search_result",
    "paper_details": "paper_metadata", "paper_cards": "paper_card",
    "claim_plans": "claim_plan", "writing_plans": "writing_plan",
    "check_report": "check_report",
    "writer_candidates": "writer_candidate",
    "review": "draft", "agent_task_results": "task_result",
    "quarantined_generation_snapshot": "quarantined_generation",
    "pending_generation_previous": "quarantined_generation",
    "main_agent_trajectory": "context_trajectory",
    "subagent_sessions": "context_trajectory",
}
