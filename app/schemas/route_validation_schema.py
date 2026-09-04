"""可解释的 Route–Evidence 验证数据契约。"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RouteStatus(str, Enum):
    """路线状态；状态与后续动作分离。"""

    KEEP = "KEEP"
    WEAK = "WEAK"
    DROP = "DROP"


class RouteAction(str, Enum):
    """路线验证后允许交给控制器的动作。"""

    KEEP = "KEEP"
    TARGETED_SEARCH = "TARGETED_SEARCH"
    ROUTE_REVISION = "ROUTE_REVISION"
    DROP = "DROP"


class RoutePaperMatchFeatures(BaseModel):
    """论文与路线之间的原始匹配证据，不预先压缩成单一相似度。"""

    semantic_similarity: float = Field(default=0.0, ge=0.0, le=1.0)
    concept_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    lexical_anchor_score: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_claim_match: float = Field(default=0.0, ge=0.0, le=1.0)
    method_compatibility: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_role_score: float = Field(default=0.0, ge=0.0, le=1.0)
    negative_anchor_conflict: float = Field(default=0.0, ge=0.0, le=1.0)
    matched_anchors: list[str] = Field(default_factory=list)
    matched_method_concepts: list[str] = Field(default_factory=list)
    positive_signal_count: int = 0
    match_level: str = "none"
    # 研究链阶段兼容性：上游产物不得替代下游证据。任一侧未声明阶段时
    # ``stage_compatible`` 保持 True，匹配仍由词面/锚点信号决定。
    route_stage: str = ""
    paper_stages: list[str] = Field(default_factory=list)
    stage_compatible: bool = True
    stage_conflict_reason: str = ""


class RouteValidityAssessment(BaseModel):
    """与当前证据数量无关的路线结构有效性。"""

    score: float = Field(default=0.0, ge=0.0, le=1.0)
    definition_completeness: float = Field(default=0.0, ge=0.0, le=1.0)
    anchor_grounding: float = Field(default=0.0, ge=0.0, le=1.0)
    boundary_clarity: float = Field(default=0.0, ge=0.0, le=1.0)
    internal_consistency: float = Field(default=0.0, ge=0.0, le=1.0)
    role_clarity: float = Field(default=0.0, ge=0.0, le=1.0)
    structurally_valid: bool = False
    rejected_anchor_expansions: list[dict[str, Any]] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class EvidenceSufficiencyAssessment(BaseModel):
    """仅由当前证据池决定的路线证据充分性。"""

    score: float = Field(default=0.0, ge=0.0, le=1.0)
    sufficient: bool = False
    core_evidence_count: int = 0
    supporting_evidence_count: int = 0
    independent_source_count: int = 0
    evidence_quality_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    core_paper_ids: list[str] = Field(default_factory=list)
    supporting_paper_ids: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)

