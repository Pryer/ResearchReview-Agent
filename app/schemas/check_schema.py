"""版本绑定的核验报告与统一问题协议。"""
from typing import Literal
from uuid import uuid4
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ResultIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    issue_id: str = Field(default_factory=lambda: uuid4().hex)
    code: str
    severity: Literal["blocking", "warning"] = "blocking"
    reason: str
    claim_id: str = ""
    section_id: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    recoverable: bool = True
    issue_type: str = "quality"
    verification_status: str = "unverified"
    source_refs: list[str] = Field(default_factory=list)
    suggested_repair: str = ""
    blocks_draft: bool = True
    blocks_final: bool = True


class SuggestedAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: str
    target_issue_ids: list[str] = Field(default_factory=list)
    reason: str = ""


    @field_validator("action")
    @classmethod
    def registered_action(cls, value):
        from app.agent.action_registry import ACTION_REGISTRY
        if value not in ACTION_REGISTRY:
            raise ValueError("suggested action must be registered")
        return value


class CheckReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    report_id: str = Field(default_factory=lambda: uuid4().hex)
    schema_version: int = 1
    draft_version: int = 0
    verifier_profile: str = ""
    execution_identities: list[dict[str, str]] = Field(default_factory=list)
    input_fingerprint: str
    execution_status: Literal["completed", "incomplete", "failed"]
    verdict: Literal["passed", "needs_revision", "unable_to_verify"]
    coverage: dict[str, int | float | str | bool] = Field(default_factory=dict)
    issues: list[ResultIssue] = Field(default_factory=list)
    suggested_actions: list[SuggestedAction] = Field(default_factory=list)
