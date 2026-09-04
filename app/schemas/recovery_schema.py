"""证据恢复闭环的数据契约。"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RouteGapType(str, Enum):
    """路线级证据缺口的原因，而不是要执行的动作。"""

    SEARCH_COVERAGE_GAP = "SEARCH_COVERAGE_GAP"
    SEARCH_PRECISION_GAP = "SEARCH_PRECISION_GAP"
    ROUTE_STRUCTURE_GAP = "ROUTE_STRUCTURE_GAP"
    SCOPE_GAP = "SCOPE_GAP"


class RecoveryAction(str, Enum):
    """确定性控制器允许执行的恢复动作。"""

    CONTINUE = "CONTINUE"
    TARGETED_SEARCH = "TARGETED_SEARCH"
    QUERY_FILTER_REVISION = "QUERY_FILTER_REVISION"
    ROUTE_REVISION = "ROUTE_REVISION"
    SCOPE_REVISION = "SCOPE_REVISION"
    DEGRADE = "DEGRADE"


class RecoveryStatus(str, Enum):
    """恢复过程状态；与缺口原因分离。"""

    NOT_REQUIRED = "NOT_REQUIRED"
    RECOVERABLE = "RECOVERABLE"
    EXHAUSTED = "EXHAUSTED"
    DEGRADED = "DEGRADED"


class RouteEvidenceGap(BaseModel):
    route_id: str
    route_name: str = ""
    gap_type: RouteGapType
    reason: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    suggested_queries: list[str] = Field(default_factory=list)
    missing_constraints: list[str] = Field(default_factory=list)
    exclusion_candidates: list[str] = Field(default_factory=list)
    structurally_resolved: bool = False
    # 篇数目标与缺口：目标由交付物类型派生，不等于路线充分性判定阈值。
    core_evidence_count: int = 0
    target_core_evidence: int = 0
    core_evidence_deficit: int = 0
    # 多样性缺口是独立维度，补足篇数不代表脉络可写。
    diversity_deficit: dict[str, Any] | None = None


class EvidenceGapReport(BaseModel):
    evidence_snapshot_version: int = 0
    evidence_snapshot_fingerprint: str = ""
    needs_recovery: bool = False
    gaps: list[RouteEvidenceGap] = Field(default_factory=list)
    affected_route_ids: list[str] = Field(default_factory=list)
    coverage_score: float = Field(default=0.0, ge=0.0, le=1.0)
    source_health: str = "unknown"
    diagnosis_source: str = "deterministic"
    scope_revision_recommended: bool = False
    notes: list[str] = Field(default_factory=list)


class RecoveryDecision(BaseModel):
    action: RecoveryAction
    status: RecoveryStatus
    reason: str
    affected_route_ids: list[str] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)
    missing_constraints: list[str] = Field(default_factory=list)
    exclusion_candidates: list[str] = Field(default_factory=list)
    query_novelty: float = Field(default=0.0, ge=0.0, le=1.0)
    # 每条路线的目标篇数与本轮分到的查询；执行层据此做 per-route 收敛判定。
    route_targets: dict[str, int] = Field(default_factory=dict)
    route_query_allocation: dict[str, list[str]] = Field(default_factory=dict)


class RecoveryHistoryEntry(BaseModel):
    round: int = 0
    action: RecoveryAction
    status: RecoveryStatus
    affected_route_ids: list[str] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)
    new_evidence: int = 0
    new_relevant_evidence: int = 0
    coverage_before: float = 0.0
    coverage_after: float = 0.0
    coverage_gain: float = 0.0
    query_novelty: float = 0.0
    stop_reason: str = ""
    # 每条路线的 core_before / core_after / target，供边际收益按路线判定。
    route_progress: dict[str, dict[str, int]] = Field(default_factory=dict)


class ClaimGapType(str, Enum):
    MISSING_SUPPORT = "MISSING_SUPPORT"
    EXCESSIVE_STRENGTH = "EXCESSIVE_STRENGTH"
    OPTIONAL_UNSUPPORTED = "OPTIONAL_UNSUPPORTED"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"


class ClaimEvidenceGap(BaseModel):
    claim_id: str
    route_id: str = ""
    gap_type: ClaimGapType
    action: str
    reason: str


class ClaimEvidenceGateReport(BaseModel):
    passed: bool = True
    total_claims: int = 0
    retained_claims: int = 0
    weakened_claims: int = 0
    dropped_claims: int = 0
    single_source_claim_limit: int = 2
    single_source_claims_dropped: int = 0
    entailment_checked_claims: int = 0
    entailment_failed_claims: int = 0
    searchable_core_gaps: list[ClaimEvidenceGap] = Field(default_factory=list)
    gaps: list[ClaimEvidenceGap] = Field(default_factory=list)
