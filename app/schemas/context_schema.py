"""主 Agent 的最小工作上下文契约。

完整研究状态、原始文档和执行事件保存在外部状态/资料存储中。主 Agent
只消费这里的五个字段，避免把整个 ``ResearchAgentState`` 重复塞进模型。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictContextModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GoalContext(StrictContextModel):
    request: str = ""
    topic: str = ""
    deliverables: list[str] = Field(default_factory=list)
    scope: dict[str, Any] = Field(default_factory=dict)
    constraints: dict[str, Any] = Field(default_factory=dict)
    explicit_constraints: list[dict[str, Any]] = Field(default_factory=list)


class AgentStateContext(StrictContextModel):
    status: str = "created"
    execution_status: str = "idle"
    research_status: str = "created"
    current_stage: str = "created"
    next_actions: list[str] = Field(default_factory=list)
    allowed_actions: list[str] = Field(default_factory=list)
    completed_actions: list[str] = Field(default_factory=list)
    recent_trajectory: list[dict[str, Any]] = Field(default_factory=list)
    failure_reasons: list[str] = Field(default_factory=list)
    gate_status: dict[str, Any] = Field(default_factory=dict)
    artifact_summary: dict[str, int | bool | str | None] = Field(default_factory=dict)
    remaining_action_budget: int | None = None
    budget_remaining: dict[str, int] = Field(default_factory=dict)
    source_state_version: str = ""


class KeyEvidence(StrictContextModel):
    evidence_id: str
    finding: str
    paper_ids: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    evidence_level: str = "unknown"
    uncertainty: str | None = None
    supports: list[str] = Field(default_factory=list)


class DecisionRecord(StrictContextModel):
    decision_id: str
    category: str
    decision: str
    reason: str = ""
    source: Literal["user", "policy", "agent", "evidence"] = "agent"
    valid_for: str = "current_state"
    invalidated_by: list[str] = Field(default_factory=list)


class OpenQuestion(StrictContextModel):
    question_id: str
    category: str
    question: str
    blocking: bool = False
    needed_evidence: list[str] = Field(default_factory=list)
    next_action: str = ""


class MainAgentContext(StrictContextModel):
    """模型可见的完整顶层结构；严格限制为五个字段。"""

    goal: GoalContext = Field(default_factory=GoalContext)
    state: AgentStateContext = Field(default_factory=AgentStateContext)
    key_evidence: list[KeyEvidence] = Field(default_factory=list)
    decisions: list[DecisionRecord] = Field(default_factory=list)
    open_questions: list[OpenQuestion] = Field(default_factory=list)


class ContextSnapshot(StrictContextModel):
    """持久化元数据放在上下文外层，不污染模型可见的五字段协议。"""

    context: MainAgentContext
    source_fingerprint: str
    source_state_version: str
    schema_version: str = "main-context-v2"
    compacted: bool = True
    estimated_chars: int = 0
    max_chars: int = 0
    validation_errors: list[str] = Field(default_factory=list)
