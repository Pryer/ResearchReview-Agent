"""主 Agent 与专业 Agent 之间的任务/结果协议。"""

from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class AgentRole(str, Enum):
    SEARCH = "search"
    ANALYSIS = "analysis"
    WRITING = "writing"


class AgentTaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STALE = "stale"
    CANCELLED = "cancelled"


class AgentTaskOutcome(str, Enum):
    SUCCEEDED = "succeeded"
    DEGRADED = "degraded"
    BLOCKED = "blocked"
    NO_CHANGE = "no_change"
    UNKNOWN = "unknown"


class AgentTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(default_factory=lambda: uuid4().hex)
    role: AgentRole
    operation: str
    objective: str
    constraints: dict[str, Any] = Field(default_factory=dict)
    input_artifact_refs: list[str] = Field(default_factory=list)
    expected_output: list[str] = Field(default_factory=list)
    budget: dict[str, int] = Field(default_factory=dict)
    source_state_version: str
    source_state_fingerprint: str
    idempotency_key: str = ""


class AgentTaskResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    role: AgentRole
    operation: str
    status: AgentTaskStatus
    outcome: AgentTaskOutcome = AgentTaskOutcome.UNKNOWN
    findings: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    proposed_decisions: list[dict[str, Any]] = Field(default_factory=list)
    open_questions: list[dict[str, Any]] = Field(default_factory=list)
    output_artifact_refs: list[str] = Field(default_factory=list)
    changed_state_fields: list[str] = Field(default_factory=list)
    usage: dict[str, int] = Field(default_factory=dict)
    input_state_fingerprint: str
    output_state_fingerprint: str = ""
    idempotency_key: str = ""
    # 仅供同进程 Controller 提交，序列化任务记录时不得泄漏整个状态补丁。
    state_patch: dict[str, Any] = Field(default_factory=dict, exclude=True)
    removed_state_fields: list[str] = Field(default_factory=list, exclude=True)
    error: str | None = None
