"""Agent Status Bar 的结构化载荷契约。

Stable Prefix、ReAct Trajectory、状态栏及 Compaction 共同组成模型上下文。
这里的五个字段只定义最新状态栏；完整资料和归档历史保存在外部存储中。
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
    delegation: dict[str, Any] = Field(default_factory=dict)
    agent_messages: list[dict[str, Any]] = Field(default_factory=list)
    research_memory: list[dict[str, Any]] = Field(default_factory=list)
    memory_retrieval: dict[str, Any] = Field(default_factory=dict)
    failure_reasons: list[str] = Field(default_factory=list)
    gate_status: dict[str, Any] = Field(default_factory=dict)
    artifact_summary: dict[str, int | bool | str | None] = Field(default_factory=dict)
    remaining_action_budget: int | None = None
    budget_remaining: dict[str, int] = Field(default_factory=dict)
    context_budget: dict[str, int | str] = Field(default_factory=dict)
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
    """最新 Agent Status Bar；沿用五字段载荷，不代表整个模型上下文。"""

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
