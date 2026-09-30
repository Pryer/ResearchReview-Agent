"""Agent 执行原语：协作式取消异常、节点边界检查与 LLM 客户端工厂。

独立于 graph 编排层，供编排入口与检索/恢复循环共享。
``AgentCancelledError`` 的规范定义在本模块；graph 对外再导出，
兼容节点与服务层既有的 ``from app.agent.graph import AgentCancelledError``
导入路径。
"""

from __future__ import annotations

from typing import Any, Callable

from app.core.logger import get_logger

logger = get_logger(__name__)


class AgentCancelledError(RuntimeError):
    """任务收到协作式取消请求。"""


def checkpoint(
    state: Any,
    step: str,
    current: int,
    total: int,
    should_cancel: Callable[[], bool] | None,
    progress_callback: Callable[[str, int, int], None] | None,
) -> None:
    """在节点边界检查取消，并报告进度。"""
    # WHY: 节点边界是最密集的执行闸门；租约失效（owner 被接管/续约终止）
    # 必须在下一个昂贵步骤前被识别为 AgentExecutionStale 终止，而不是等
    # 远程调用层报错后被节点当普通故障降级、继续在失权会话里烧费。
    from app.agent.execution_budget import check_execution

    check_execution()
    if should_cancel and should_cancel():
        logger.info("Agent cancelled before step: %s", step)
        raise AgentCancelledError(f"任务已在 {step} 前取消")
    if progress_callback:
        progress_callback(step, current, total)


def get_llm() -> Any:
    """懒加载 LLM 客户端（只在需要时创建）。"""
    from app.services.llm_service import LLMService
    return LLMService()
