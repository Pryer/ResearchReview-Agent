"""静态能力模板；每次执行仍需动态状态与参数准入。"""
from dataclasses import dataclass
from app.schemas.agent_task_schema import AgentRole


@dataclass(frozen=True)
class CapabilityProfile:
    name: str
    role: AgentRole
    actions: tuple[str, ...]
    max_steps: int


PROFILES = {p.name: p for p in (
    CapabilityProfile("initial_research", AgentRole.SEARCH, ("search_and_rank", "fetch_metadata"), 6),
    CapabilityProfile("gap_recovery", AgentRole.SEARCH, ("targeted_search", "fetch_metadata"), 4),
    CapabilityProfile("evidence_build", AgentRole.ANALYSIS, ("extract_paper_cards", "validate_routes", "cluster_papers", "plan_claims"), 8),
    CapabilityProfile("gap_diagnosis", AgentRole.ANALYSIS, ("diagnose_evidence_gaps",), 1),
    CapabilityProfile("scope_revision", AgentRole.ANALYSIS, ("revise_scope",), 1),
    CapabilityProfile("draft_generation", AgentRole.WRITING, ("generate_deliverables",), 1),
    CapabilityProfile("section_revision", AgentRole.WRITING, ("rewrite_sections",), 1),
    CapabilityProfile("independent_check", AgentRole.CHECK, ("validate_result",), 1),
)}
