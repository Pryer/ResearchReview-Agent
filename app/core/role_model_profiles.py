"""角色模型参数只引用已配置的传输通道，不持久化凭证或连接地址。"""
import json
from functools import lru_cache
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator


class RoleModelProfile(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    provider_slot: Literal['primary', 'backup'] = 'primary'
    model: str | None = Field(default=None, min_length=1, max_length=200)
    fallback: Literal['inherit', 'disabled', 'primary', 'backup'] = 'inherit'
    fallback_model: str | None = Field(default=None, min_length=1, max_length=200)
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, ge=1)
    control_plane_max_tokens: int | None = Field(default=None, ge=1)
    context_window_tokens: int | None = Field(default=None, ge=1)
    native_tools_enabled: bool | None = None
    thinking_enabled: bool | None = None
    thinking_effort: Literal['low', 'high', 'max'] | None = None

    @model_validator(mode='after')
    def validate_fallback(self):
        if self.fallback == self.provider_slot:
            raise ValueError('fallback must use a different provider slot')
        if self.fallback == 'disabled' and self.fallback_model is not None:
            raise ValueError('disabled fallback cannot select a model')
        return self


@lru_cache(maxsize=32)
def parse_profiles(raw):
    return TypeAdapter(dict[Literal['main', 'search', 'analysis', 'writing', 'check'], RoleModelProfile]).validate_python(
        json.loads(raw))


def profile_for(role):
    from app.core.config import get_settings
    return parse_profiles(get_settings().agent_role_model_profiles).get(role)
