"""主从 Agent 通信是任务数据，不授予权限或交付授权。"""
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class SubagentReport(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    status: Literal["completed", "needs_help", "blocked"]
    message: str = Field(min_length=1, max_length=1200)
    suggested_actions: list[str] = Field(default_factory=list, max_length=5)


class AgentMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message_id: str = Field(default_factory=lambda: uuid4().hex)
    delegation_id: str
    sender: Literal["main", "search", "analysis", "writing"]
    recipient: Literal["main", "search", "analysis", "writing"]
    kind: Literal["task", "progress", "completed", "needs_help", "blocked"]
    content: str = Field(max_length=1200)
    suggested_actions: list[str] = Field(default_factory=list)
