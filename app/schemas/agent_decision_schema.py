"""主 Agent 单轮结构化动作协议。"""

from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


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

    @field_validator("evidence_refs")
    @classmethod
    def validate_refs(cls, values: list[str]) -> list[str]:
        for value in values:
            if not str(value).startswith(("artifact://", "state://")):
                raise ValueError("evidence references must use artifact:// or state://")
        return values
