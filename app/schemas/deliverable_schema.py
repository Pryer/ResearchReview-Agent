"""四种核心学术交付物及其写作契约。"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class CoreDeliverableType(str, Enum):
    RESEARCH_BACKGROUND = "research_background"
    RESEARCH_STATUS = "research_status"
    RELATED_WORK = "related_work"
    NARRATIVE_REVIEW = "narrative_review"


class GapType(str, Enum):
    AUTHOR_REPORTED = "author_reported"
    CROSS_PAPER_INFERENCE = "cross_paper_inference"
    EVIDENCE_ACCESS_LIMITATION = "evidence_access_limitation"


class InputRequirement(BaseModel):
    field: str
    required: bool = True
    reason: str
    clarification_question: str | None = None


class SectionRequirement(BaseModel):
    id: str
    purpose: str
    required: bool = True
    allowed_evidence_levels: list[str] = Field(default_factory=list)


class StructureConstraints(BaseModel):
    """用户可见的交付物结构，不包含任何具体领域标题。"""

    visible_title: str
    allow_subsections: bool = False
    min_subsections: int = 0
    max_subsections: int = 0
    min_paragraphs: int = 1
    max_paragraphs: int | None = None
    output_form: str = "continuous_prose"


class PlanningConstraints(BaseModel):
    """内部规划策略；规划节点不等于可见章节。"""

    planning_strategy: str
    dynamic_topic_generation: bool = True
    allow_topic_merge: bool = True
    require_comparative_synthesis: bool = False
    require_final_synthesis: bool = False


class ValidationConstraints(BaseModel):
    """交付物结构与质量校验约束。"""

    required_sections: list[str] = Field(default_factory=list)
    forbidden_visible_sections: list[str] = Field(default_factory=list)
    target_char_range: tuple[int, int] | None = None


class DeliverableSpec(BaseModel):
    type: CoreDeliverableType
    purpose: str
    required_inputs: list[InputRequirement] = Field(default_factory=list)
    required_sections: list[SectionRequirement] = Field(default_factory=list)
    rhetorical_moves: list[str] = Field(default_factory=list)
    evidence_rules: list[str] = Field(default_factory=list)
    forbidden_patterns: list[str] = Field(default_factory=list)
    min_references: int | None = None
    requires_dynamic_taxonomy: bool = False
    requires_user_paper_profile: bool = False
    structure: StructureConstraints | None = None
    planning: PlanningConstraints | None = None
    validation: ValidationConstraints | None = None


class UserPaperProfile(BaseModel):
    research_problem: str = ""
    proposed_method: str | None = None
    research_direction: str | None = None
    research_object: str | None = None
    target_task: str | None = None
    application_scenario: str | None = None
    claimed_contribution: str | None = None
    data_modalities: list[str] = Field(default_factory=list)
    main_contributions: list[str] = Field(default_factory=list)
    comparison_targets: list[str] = Field(default_factory=list)


class DeliverableReadinessResult(BaseModel):
    ready: bool
    requested_type: CoreDeliverableType
    effective_type: CoreDeliverableType
    missing_inputs: list[str] = Field(default_factory=list)
    insufficient_evidence: list[str] = Field(default_factory=list)
    clarification_question: str | None = None
    downgrade_suggestion: CoreDeliverableType | None = None


class GenerationReadinessResult(BaseModel):
    """进入 Writer 前的全局硬约束检查。"""

    ready: bool = True
    requested_minimum_references: int = 0
    usable_reference_count: int = 0
    blocking_issues: list[dict[str, Any]] = Field(default_factory=list)
    recovery_options: list[str] = Field(default_factory=list)


class SynthesizedGap(BaseModel):
    gap_type: GapType
    statement: str
    supporting_paper_ids: list[str] = Field(default_factory=list)


class ThemeSynthesis(BaseModel):
    theme_id: str
    theme_name: str
    paper_ids: list[str] = Field(default_factory=list)
    reported_problems: list[dict[str, Any]] = Field(default_factory=list)
    reported_methods: list[dict[str, Any]] = Field(default_factory=list)
    shared_problems: list[dict[str, Any]] = Field(default_factory=list)
    shared_methods: list[dict[str, Any]] = Field(default_factory=list)
    common_problems: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Deprecated compatibility alias for shared_problems",
    )
    common_methods: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Deprecated compatibility alias for shared_methods",
    )
    reported_findings: list[dict[str, Any]] = Field(default_factory=list)
    author_stated_limitations: list[dict[str, Any]] = Field(default_factory=list)
    synthesized_gaps: list[SynthesizedGap] = Field(default_factory=list)
    comparison_dimensions: list[str] = Field(default_factory=list)


class WritingSection(BaseModel):
    id: str
    title: str
    purpose: str
    claims_to_establish: list[str] = Field(default_factory=list)
    supporting_paper_ids: list[str] = Field(default_factory=list)
    supporting_claim_ids: list[str] = Field(default_factory=list)
    comparison_dimensions: list[str] = Field(default_factory=list)
    target_word_count: int | None = None
    visible: bool = True
    heading_level: int | None = Field(default=2, ge=2, le=4)
    # WHY: 章节级最低唯一引用是独立契约，不是全局 required_reference_count
    # 的均分份额。0 表示该章节不设章节级下限（概述、结构性章节）；正式
    # 研究路线小节由规划器设为 2，用于保证路线内比较有独立证据。
    minimum_unique_references: int = Field(default=0, ge=0)


class PlanningNode(BaseModel):
    """只指导写作的内部节点；默认不渲染为正文标题。"""

    node_id: str
    node_type: str
    label: str
    visible: bool = False
    heading_level: int | None = Field(default=None, ge=2, le=4)
    writing_goal: str
    evidence_ids: list[str] = Field(default_factory=list)


class WritingPlan(BaseModel):
    deliverable_type: CoreDeliverableType
    purpose: str
    organizing_strategy: str
    sections: list[WritingSection] = Field(default_factory=list)
    hidden_planning_nodes: list[PlanningNode] = Field(default_factory=list)
    evidence_policy: dict[str, Any] = Field(default_factory=dict)
    citation_policy: dict[str, Any] = Field(default_factory=dict)
    style_constraints: dict[str, Any] = Field(default_factory=dict)
    target_char_range: tuple[int, int] | None = None
    required_focuses: list[str] = Field(default_factory=list)
    covered_focuses: list[str] = Field(default_factory=list)
    undercovered_focuses: list[str] = Field(default_factory=list)


class DeliverableValidationResult(BaseModel):
    valid: bool
    deliverable_type: CoreDeliverableType
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


class CitationOccurrence(BaseModel):
    paper_id: str
    section_title: str = ""
    paragraph_index: int = 0
    occurrence_index: int = 0


class CitationRegistry(BaseModel):
    """正文级唯一文献注册表，区分论文、引用出现和章节分配。"""

    unique_papers: dict[str, dict[str, Any]] = Field(default_factory=dict)
    citation_occurrences: list[CitationOccurrence] = Field(default_factory=list)
    section_allocations: dict[str, list[str]] = Field(default_factory=dict)
    detailed_introductions: dict[str, int] = Field(default_factory=dict)


class UnsupportedTaskGuardResult(BaseModel):
    allowed: bool = True
    supported_deliverables: list[CoreDeliverableType] = Field(default_factory=list)
    unsupported_requests: list[str] = Field(default_factory=list)
    message: str = ""
