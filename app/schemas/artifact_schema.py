"""不可变研究资料注册表；版本、角色与载荷边界由代码裁决。"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ArtifactSpec:
    roles: frozenset[str]
    schema_version: int = 1
    max_bytes: int = 262144


ALL = frozenset({"main", "search", "analysis", "writing", "service"})
ARTIFACT_TYPES = {
    "main_agent_context": ArtifactSpec(ALL),
    "conversation_event": ArtifactSpec(frozenset({"service"})),
    "conversation_summary": ArtifactSpec(frozenset({"main", "service"})),
    "raw_search_result": ArtifactSpec(frozenset({"search", "analysis", "service"})),
    "paper_metadata": ArtifactSpec(ALL),
    "document_fragment": ArtifactSpec(frozenset({"search", "analysis", "service"})),
    "paper_card": ArtifactSpec(ALL),
    "claim_plan": ArtifactSpec(frozenset({"analysis", "writing", "service"})),
    "writing_plan": ArtifactSpec(frozenset({"analysis", "writing", "service"})),
    "draft": ArtifactSpec(frozenset({"analysis", "writing", "service"})),
    "task_result": ArtifactSpec(ALL),
    "evidence_bundle": ArtifactSpec(ALL),
}

# WHY: 按项目独立外置，正文按片段保存；旧节点仍在任务副本中使用原有字段。
STATE_ARTIFACT_FIELDS = {
    "candidate_papers": "raw_search_result", "ranked_papers": "raw_search_result",
    "paper_details": "paper_metadata", "paper_cards": "paper_card",
    "claim_plans": "claim_plan", "writing_plans": "writing_plan",
    "review": "draft", "agent_task_results": "task_result",
}
