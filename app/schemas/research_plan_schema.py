"""结构化研究请求、计划变更与受约束任务图 Schema。"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class TurnType(str, Enum):
    NEW_REQUEST = "new_request"
    MODIFICATION = "modification"
    CLARIFICATION_ANSWER = "clarification_answer"
    CORRECTION = "correction"
    CONTINUATION = "continuation"
    CANCEL = "cancel"


class ResearchOperation(str, Enum):
    SCOPE_DISAMBIGUATION = "scope_disambiguation"
    QUERY_EXPANSION = "query_expansion"
    SEARCH = "search"
    METADATA_VERIFICATION = "metadata_verification"
    DEDUPLICATE = "deduplicate"
    SCREEN = "screen"
    EXTRACT_PAPER_CARDS = "extract_paper_cards"
    CLASSIFY_PAPERS = "classify_papers"
    SYNTHESIZE = "synthesize"
    WRITE = "write"
    VALIDATE_CITATIONS = "validate_citations"
    REVISE = "revise"


class DeliverableType(str, Enum):
    RESEARCH_BACKGROUND = "research_background"
    RESEARCH_STATUS = "research_status"
    RELATED_WORK = "related_work"
    LITERATURE_REVIEW = "literature_review"
    NARRATIVE_REVIEW = "narrative_review"
    INTRODUCTION = "introduction"
    REFERENCE_LIST = "reference_list"
    PAPER_TABLE = "paper_table"
    PAPER_LIST = "paper_list"


class ResearchMode(str, Enum):
    """由对象、方法角色和终点目标派生的研究请求模式。"""

    DOMAIN_ORIENTED = "domain_oriented"
    TECHNOLOGY_ORIENTED = "technology_oriented"
    TECHNOLOGY_APPLIED_TO_DOMAIN = "technology_applied_to_domain"
    TECHNOLOGY_ASSISTED_DOMAIN_ANALYSIS = "technology_assisted_domain_analysis"
    AMBIGUOUS = "ambiguous"


class MethodRole(str, Enum):
    PRIMARY_RESEARCH_TARGET = "primary_research_target"
    IMPLEMENTATION_METHOD = "implementation_method"
    INTERMEDIATE_STEP = "intermediate_step"
    PRIMARY_ANALYSIS_METHOD = "primary_analysis_method"
    EVALUATION_OBJECT = "evaluation_object"
    NOT_SPECIFIED = "not_specified"


class SemanticItem(BaseModel):
    id: str
    label: Optional[str] = None
    surface_text: Optional[str] = None
    explicit: bool = False
    inferred: bool = False
    source: str = Field(
        default="unknown",
        description="user_explicit/term_normalizer/llm_inference/example_assisted/unknown",
    )
    inference_basis: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ResearchMethod(SemanticItem):
    """方法级角色，避免用一个全局 method_role 覆盖多种方法。"""

    category: str = Field(default="technical", description="technical/analytical/measurement")
    role: MethodRole = MethodRole.NOT_SPECIFIED


class TerminalGoal(BaseModel):
    type: str = Field(default="unspecified")
    target: Optional[str] = None
    description: str = ""


class SearchBranch(BaseModel):
    branch_type: str
    queries: list[str] = Field(default_factory=list)
    required_concepts: list[list[str]] = Field(default_factory=list)
    rationale: str = ""
    constraint_level: str = Field(
        default="soft", description="hard/soft/exploratory"
    )


class EvidenceRequirement(BaseModel):
    """由用户明确任务阶段派生的直接证据要求。"""

    requirement_id: str
    label: str
    evidence_role: str = Field(
        description="perception/structured_coding/analytical_method/interpretation"
    )
    aliases: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    minimum_direct_sources: int = Field(default=1, ge=1)
    exact_method_required: bool = False
    route_required: bool = True
    route_group: str = ""
    selection_mode: str = Field(
        default="all",
        description="all/any/open_any；控制并列方法是全部必需、命名任选或开放任选",
    )
    context_aliases: list[str] = Field(
        default_factory=list,
        description="开放方法判定时用于约束应用对象或分析目标的上下文锚点",
    )


class LanguageAffinity(str, Enum):
    """主题的文献语言倾向；仅为枚举判断，实际配额由代码映射并钳制。"""

    ZH_DOMINANT = "zh_dominant"
    BALANCED = "balanced"
    EN_DOMINANT = "en_dominant"


class ResearchSemanticFrame(BaseModel):
    """交付物之外的研究主题语义；research_mode 仅为派生标签。"""

    canonical_topic: str
    application_domains: list[SemanticItem] = Field(default_factory=list)
    research_objects: list[SemanticItem] = Field(default_factory=list)
    methods: list[ResearchMethod] = Field(default_factory=list)
    research_actions: list[SemanticItem] = Field(default_factory=list)
    analysis_targets: list[SemanticItem] = Field(default_factory=list)
    terminal_goal: TerminalGoal = Field(default_factory=TerminalGoal)
    secondary_goals: list[str] = Field(default_factory=list)
    task_chain: list[str] = Field(default_factory=list)
    required_focuses: list[str] = Field(
        default_factory=list,
        description="用户明确指定或由其指定分析方法直接决定、后续检索与写作不得静默丢弃的重点",
    )
    evidence_requirements: list[EvidenceRequirement] = Field(
        default_factory=list,
        description="按任务阶段区分的直接证据标准；不得用相邻阶段证据替代",
    )
    research_mode: ResearchMode = ResearchMode.AMBIGUOUS
    language_affinity: LanguageAffinity = Field(
        default=LanguageAffinity.BALANCED,
        description=(
            "该主题的高质量文献主要以哪种语言发表。仅输出枚举判断，"
            "中英文检索配额由代码映射并钳制在安全区间内"
        ),
    )
    language_affinity_reason: str = Field(
        default="",
        description="语言倾向判断依据；用于诊断，不参与检索",
    )
    search_branches: list[SearchBranch] = Field(default_factory=list)
    scope_ambiguities: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    confidence: dict[str, float] = Field(default_factory=dict)
    clarification_needed: bool = False
    clarification_question: Optional[str] = None
    retrieved_case_ids: list[str] = Field(default_factory=list)
    validation_issues: list[str] = Field(default_factory=list)
    validation_warnings: list[str] = Field(default_factory=list)


class TaskNodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INVALIDATED = "invalidated"
    SKIPPED = "skipped"


class ResearchScope(BaseModel):
    domain: Optional[str] = None
    included_perspectives: list[str] = Field(default_factory=list)
    excluded_perspectives: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


class TimeConstraint(BaseModel):
    """保留时间表达的语义，避免把“近三年”静默改成宽泛年份。"""

    raw_expression: Optional[str] = None
    mode: str = Field(default="unspecified", description="rolling/calendar_year/absolute/unspecified")
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    explicit: bool = False
    assumption: Optional[str] = None


class ResearchConstraints(BaseModel):
    time: TimeConstraint = Field(default_factory=TimeConstraint)
    minimum_references: Optional[int] = None
    maximum_references: Optional[int] = None
    retrieval_target: Optional[int] = None
    language: str = "zh"
    citation_style: Optional[str] = None
    source_types: list[str] = Field(default_factory=list)
    peer_reviewed_only: bool = False


class ClarificationState(BaseModel):
    needed: bool = False
    slot: Optional[str] = None
    question: Optional[str] = None
    options: list[str] = Field(default_factory=list)
    reason: Optional[str] = None


class TaskNode(BaseModel):
    """受约束任务节点；节点类型固定，由计划决定是否启用。"""

    id: str
    operation: ResearchOperation
    depends_on: list[str] = Field(default_factory=list)
    status: TaskNodeStatus = TaskNodeStatus.PENDING
    parameters: dict[str, Any] = Field(default_factory=dict)
    affected_deliverables: list[DeliverableType] = Field(default_factory=list)


class ResearchConfidence(BaseModel):
    """只保留目前能解释的置信度，避免伪精确的多维分数。"""

    overall: float = Field(default=0.0, ge=0.0, le=1.0)
    scope: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    uncertain_fields: list[str] = Field(default_factory=list)


class ResearchRequestPlan(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    turn_type: TurnType = TurnType.NEW_REQUEST
    summary: str
    topic: str
    scope: ResearchScope = Field(default_factory=ResearchScope)
    operations: list[ResearchOperation] = Field(default_factory=list)
    deliverables: list[DeliverableType] = Field(default_factory=list)
    constraints: ResearchConstraints = Field(default_factory=ResearchConstraints)
    excluded_paper_ids: list[str] = Field(default_factory=list)
    preferred_paper_ids: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    task_graph: list[TaskNode] = Field(default_factory=list)
    clarification: ClarificationState = Field(default_factory=ClarificationState)
    confidence: ResearchConfidence = Field(default_factory=ResearchConfidence)
    semantic_frame: Optional[ResearchSemanticFrame] = None
    legacy_intent: Optional[str] = None


class PlanChangeOperation(str, Enum):
    ADD = "add"
    REMOVE = "remove"
    REPLACE = "replace"
    CLEAR = "clear"


class PlanChange(BaseModel):
    field: str
    operation: PlanChangeOperation
    value: Any = None
    reason: Optional[str] = None


class ResearchPlanPatch(BaseModel):
    turn_type: TurnType
    changes: list[PlanChange] = Field(default_factory=list)
    affected_deliverables: list[DeliverableType] = Field(default_factory=list)
    requires_replanning: bool = True


class PaperIdentity(BaseModel):
    """论文的稳定身份；界面序号不得作为持久化标识。"""

    paper_id: str
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    semantic_scholar_id: Optional[str] = None
    openalex_id: Optional[str] = None
    normalized_title_hash: Optional[str] = None
