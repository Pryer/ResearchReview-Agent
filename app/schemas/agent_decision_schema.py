"""主 Agent 单轮结构化动作协议。"""

from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MemoryFeedback(BaseModel):
    model_config = ConfigDict(extra="forbid")
    memory_id: str = Field(min_length=1, max_length=64)
    disposition: Literal["used", "irrelevant", "outdated"]
    reason: str = Field(min_length=1, max_length=240)


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: str = Field(min_length=1, max_length=64)
    objective: str = Field(min_length=1, max_length=240)


class ContextSelection(BaseModel):
    """只选择当前状态栏里的摘要 ID，不接受任意状态字段或原始文本。"""
    model_config = ConfigDict(extra="forbid")
    evidence_ids: list[str] = Field(default_factory=list, max_length=12)
    decision_ids: list[str] = Field(default_factory=list, max_length=8)
    question_ids: list[str] = Field(default_factory=list, max_length=8)
    memory_ids: list[str] = Field(default_factory=list, max_length=3)


class AgentDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_id: str = Field(default_factory=lambda: uuid4().hex)
    action: str = Field(min_length=1, max_length=64)
    arguments: dict[str, Any] = Field(default_factory=dict)
    reason: str = Field(default="", max_length=600)
    evidence_refs: list[str] = Field(default_factory=list, max_length=24)
    transport: Literal["native_tool", "json"] = "json"
    tool_call_id: str | None = None
    usage: dict[str, int] = Field(default_factory=dict)
    memory_feedback: list[MemoryFeedback] = Field(default_factory=list, max_length=5)
    plan: list[PlanStep] = Field(default_factory=list, max_length=6)
    context_selection: ContextSelection | None = None

    @field_validator("evidence_refs")
    @classmethod
    def validate_refs(cls, values: list[str]) -> list[str]:
        for value in values:
            if not str(value).startswith(("artifact://", "state://")):
                raise ValueError("evidence references must use artifact:// or state://")
        return values
