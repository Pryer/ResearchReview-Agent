"""Graph 使用的轻量条件判断。

业务路由由 :mod:`app.agent.graph` 与交付物注册表统一负责；本模块不再维护
第二份 intent→workflow 映射。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.agent.state import ResearchAgentState


def should_parse_pdf(state: "ResearchAgentState") -> bool:
    """判断是否需要解析 PDF。

    当存在成功下载的 PDF 路径时返回 True。
    """
    pdf_paths = state.get("pdf_paths") or {}
    return any(bool(path) for path in pdf_paths.values())
