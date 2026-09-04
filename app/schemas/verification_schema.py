"""生成后主张—证据验证的数据结构。"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


SupportStatus = Literal["supported", "partially_supported", "unsupported", "not_applicable"]


class AtomicClaimEvidence(BaseModel):
    """句内一个原子事实主张及其局部证据。"""

    text: str
    citations: List[str] = Field(default_factory=list)
    support_status: SupportStatus
    support_score: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_ids: List[str] = Field(default_factory=list)
    evidence_snippets: List[dict] = Field(default_factory=list)
    issues: List[str] = Field(default_factory=list)


class ClaimEvidenceResult(BaseModel):
    """单个综述句子的引用支持性判断。"""

    claim_id: str
    sentence: str
    citations: List[str] = Field(default_factory=list)
    claim_type: str = "general"
    factual: bool = True
    support_status: SupportStatus
    support_score: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_ids: List[str] = Field(default_factory=list)
    evidence_snippets: List[dict] = Field(default_factory=list)
    atomic_claims: List[AtomicClaimEvidence] = Field(default_factory=list)
    issues: List[str] = Field(default_factory=list)
    suggested_revision: Optional[str] = None
    required_access_level: str = "abstract"
    actual_access_level: Optional[str] = None


class ClaimVerificationReport(BaseModel):
    """一篇生成文本的句子级验证报告。"""

    valid: bool
    total_sentences: int
    factual_claims: int
    supported: int
    partially_supported: int
    unsupported: int
    support_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    claims: List[ClaimEvidenceResult] = Field(default_factory=list)
    evidence_summary: dict = Field(default_factory=dict)
    evidence_limitations: List[str] = Field(default_factory=list)
    threshold_policy: dict = Field(default_factory=dict)
