"""公共 Schema。

定义跨模块复用的基础数据结构和枚举。
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """任务执行状态。"""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    PARTIAL = "partial"
    BLOCKED = "blocked"
    FAILED = "failed"
    SKIPPED = "skipped"


class APIResponse(BaseModel):
    """统一 API 响应包装。"""

    code: int = Field(default=0, description="0 表示成功，非 0 表示错误")
    message: str = Field(default="success", description="响应消息")
    data: Optional[Any] = Field(default=None, description="响应数据")

    @classmethod
    def ok(cls, data: Any = None, message: str = "success") -> "APIResponse":
        return cls(code=0, message=message, data=data)

    @classmethod
    def error(cls, code: int, message: str) -> "APIResponse":
        return cls(code=code, message=message, data=None)


class PaginationParams(BaseModel):
    """分页参数。"""

    page: int = Field(default=1, ge=1, description="页码")
    page_size: int = Field(default=20, ge=1, le=100, description="每页数量")

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size


class PaginatedResponse(BaseModel):
    """分页响应。"""

    total: int = Field(default=0, description="总记录数")
    page: int = Field(default=1, description="当前页码")
    page_size: int = Field(default=20, description="每页数量")
    items: List[Dict[str, Any]] = Field(default_factory=list, description="数据列表")
