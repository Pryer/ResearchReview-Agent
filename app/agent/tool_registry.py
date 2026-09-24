"""稳定的专业 Agent 工具契约。

这里只描述主控允许委派的能力。实际外部请求仍由现有 clients/tools 层执行。
"""

from __future__ import annotations

from typing import Any


def tools_for_role(role: str) -> list[dict[str, Any]]:
    """返回稳定排序的只读副本，避免调用方污染固定工具前缀。"""
    from app.agent.action_registry import ACTION_REGISTRY
    if str(role) == "main":
        return [
            {"name": item.name, "description": item.description, "parameters": item.parameters}
            for item in ACTION_REGISTRY.values()
        ]
    return [
        {"name": item.name, "description": item.description}
        for item in ACTION_REGISTRY.values()
        if item.role is not None and item.role.value == str(role)
    ]


def operation_allowed(role: str, operation: str) -> bool:
    from app.agent.action_registry import ACTION_REGISTRY

    spec = ACTION_REGISTRY.get(str(operation))
    return bool(spec and spec.role is not None and spec.role.value == str(role))
