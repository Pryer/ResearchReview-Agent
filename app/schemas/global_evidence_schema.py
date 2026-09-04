"""综述级证据充分性（Global Evidence Gate）数据契约。

本门禁只测量与推荐，不执行任何恢复动作（v1 冻结范围）。
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SufficiencyDimension(str, Enum):
    """证据充分性评估维度。"""

    CITATION_COUNT = "citation_count"
    RECENCY = "recency"
    ROUTE_COVERAGE = "route_coverage"
    QUALITY = "quality"
    CLAIM_SUPPORT = "claim_support"


class DeficitSeverity(str, Enum):
    """缺口严重度；blocking 决定整体 passed，non_blocking 只作提示。"""

    BLOCKING = "blocking"
    NON_BLOCKING = "non_blocking"


class DimensionStatus(str, Enum):
    """单维度/整体评估状态。"""

    EVALUATED = "EVALUATED"
    NOT_REQUIRED = "NOT_REQUIRED"
    DEFERRED = "DEFERRED"
    FAILED = "FAILED"


class GlobalAction(str, Enum):
    """门禁推荐动作（v1 只推荐不执行）。"""

    REBALANCE_ROUTE = "REBALANCE_ROUTE"
    TARGETED_GLOBAL_SEARCH = "TARGETED_GLOBAL_SEARCH"
    ASK_USER = "ASK_USER"
    CONTINUE = "CONTINUE"


class EvidenceDeficit(BaseModel):
    """单个维度的证据缺口描述。"""

    type: SufficiencyDimension
    severity: DeficitSeverity
    required: int = 0  # 该维度需要达到的数量
    available: int = 0  # 实际可用数量
    missing: int = 0  # 缺口 = max(0, required - available)
    ratio: float = Field(default=0.0, ge=0.0, le=1.0)  # 实际达成比例
    reason: str = ""


class RouteEvidenceStat(BaseModel):
    """单条路线的证据统计（仅用于可审计输出，不参与决策）。"""

    route_id: str
    name: str = ""
    status: str = ""
    paper_count: int = 0
    core_paper_count: int = 0
    balanced: bool = True


class GlobalEvidenceMetrics(BaseModel):
    """门禁测量的全部指标。"""

    total_papers: int = 0
    citation_required: int = 0
    citation_available: int = 0
    in_window_papers: int = 0
    out_window_papers: int = 0
    recency_ratio: float = Field(default=1.0, ge=0.0, le=1.0)
    min_route_count: int = 0
    avg_route_count: float = 0.0
    route_balance_ratio: float = Field(default=1.0, ge=0.0, le=1.0)
    zero_paper_route_count: int = 0
    weak_route_count: int = 0
    peer_reviewed_count: int = 0
    peer_review_known_count: int = 0
    peer_review_unknown_count: int = 0
    peer_review_ratio: float = Field(default=1.0, ge=0.0, le=1.0)
    # 主张级证据强度：strong/established 主张占全部主张的比例。
    # 早期版本这里放的是"KEEP 路线证据体量均值"，由于分母是固定低阈值
    # （route_min_core_evidence=3），任何正常规模的路线都会被削平到 1.0，
    # 指标恒真、无区分度，且与 claim_plan 的实际统计矛盾。
    claim_support_proxy: float = Field(default=0.0, ge=0.0, le=1.0)
    claim_total_count: int = 0
    claim_strong_plus_count: int = 0
    claim_single_evidence_count: int = 0
    # 主张统计不可用时回退到路线证据体量，并在此标记，避免把回退值
    # 误读为真实主张强度。
    claim_support_source: str = ""
    evidence_recovery_status: str = ""
    route_stats: list[RouteEvidenceStat] = Field(default_factory=list)


class GlobalSufficiencyResult(BaseModel):
    """综述级证据充分性评估结果。"""

    passed: bool = True
    status: DimensionStatus = DimensionStatus.EVALUATED
    explicit_constraint_unmet: bool = False  # 存在阻断缺口且源于用户显式要求
    deficits: list[EvidenceDeficit] = Field(default_factory=list)
    evidence_debt: dict[str, int] = Field(default_factory=dict)  # {维度: 缺口数}
    recommended_actions: list[GlobalAction] = Field(default_factory=list)
    metrics: Optional[GlobalEvidenceMetrics] = None
    notes: list[str] = Field(default_factory=list)
