"""服务端委派授权与单步调用身份；模型请求不携带执行权限。"""
from typing import Any, Literal
from uuid import uuid4
from pydantic import BaseModel, ConfigDict, Field, field_validator
from app.schemas.agent_task_schema import AgentRole

SCOPE_FIELDS = ('section_ids', 'gap_ids', 'paper_ids')


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DelegationRequest(StrictModel):
    agent: AgentRole
    profile: str
    goal: str = Field(min_length=1, max_length=600)
    scope: dict[str, Any] = Field(default_factory=dict,
        description='仅用section_ids、gap_ids、paper_ids收窄已知ID范围；普通委派用{}。主题、年份和篇数继承权威目标，不在scope重复定义。',
        json_schema_extra={'additionalProperties': False, 'properties': {
            key: {'type': 'array', 'items': {'type': 'string', 'minLength': 1}, 'uniqueItems': True}
            for key in SCOPE_FIELDS}})

    @field_validator('scope')
    @classmethod
    def validate_scope(cls, scope):
        if set(scope) - set(SCOPE_FIELDS):
            raise ValueError('scope只允许section_ids/gap_ids/paper_ids；主题、年份和篇数已在goal中，普通委派请传{}')
        for values in scope.values():
            if (not isinstance(values, list) or not all(isinstance(v, str) and v.strip() for v in values)
                    or len(set(values)) != len(values)):
                raise ValueError('scope字段必须是无重复的非空ID字符串数组')
        return scope


class DelegationGrant(StrictModel):
    delegation_id: str = Field(default_factory=lambda: uuid4().hex)
    session_id: str = Field(min_length=1)
    role: AgentRole
    profile: str
    goal: str
    goal_fingerprint: str
    allowed_actions: tuple[str, ...]
    scope: dict[str, Any] = Field(default_factory=dict)
    immutable_arguments: dict[str, Any] | None = None
    max_steps: int = Field(default=6, ge=1, le=32)
    revision: int = Field(default=1, ge=1)
    profile_version: int = 1
    architecture_version: int = 2


class ActionInvocation(StrictModel):
    invocation_id: str = Field(default_factory=lambda: uuid4().hex)
    delegation_id: str
    step_index: int = Field(ge=0)
    action: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    expected_version: int = Field(ge=0)
    source_fingerprint: str
    grant_revision: int = Field(default=1, ge=1)


class DelegationResult(StrictModel):
    delegation_id: str
    status: Literal["completed", "failed", "cancelled", "stale"]
    outcome: Literal["satisfied", "partial", "needs_help", "blocked"]
    action_result_refs: tuple[str, ...] = ()
    findings: tuple[str, ...] = ()
    unresolved: tuple[str, ...] = ()
