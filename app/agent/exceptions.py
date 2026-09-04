"""Agent 异常体系 - 统一错误处理。

错误分级：
- FATAL: 致命错误，立即停止，无法恢复
- ERROR: 严重错误，可能可以重试或降级
- WARNING: 警告，可以继续执行

使用方式：
    from app.agent.exceptions import FatalAgentError, RetryableAgentError
    
    # 致命错误
    if not keywords:
        raise FatalAgentError(
            "规划失败：未生成检索关键词",
            step="plan",
            should_stop=True
        )
    
    # 可重试错误
    try:
        papers = search_papers(...)
    except NetworkError as e:
        raise RetryableAgentError(
            "检索失败：网络错误",
            step="search",
            should_retry=True,
            max_retries=3,
            original_error=e
        )
"""

from __future__ import annotations

from typing import Any, Dict, Optional


# ============================================================
# 基础异常类
# ============================================================

class AgentError(Exception):
    """Agent 基础异常"""
    
    def __init__(
        self,
        message: str,
        step: Optional[str] = None,
        severity: str = "error",
        should_stop: bool = False,
        should_retry: bool = False,
        original_error: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.step = step
        self.severity = severity
        self.should_stop = should_stop
        self.should_retry = should_retry
        self.original_error = original_error
        self.context = context or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于日志和返回）"""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "step": self.step,
            "severity": self.severity,
            "should_stop": self.should_stop,
            "should_retry": self.should_retry,
            "original_error": str(self.original_error) if self.original_error else None,
            "context": self.context,
        }


# ============================================================
# 致命错误（立即停止）
# ============================================================

class FatalAgentError(AgentError):
    """致命错误，立即停止执行
    
    场景：
    - 规划失败（无法生成检索策略）
    - 必需输入缺失（如 Related Work 缺少 our_work）
    - 不支持的任务类型
    """
    
    def __init__(
        self,
        message: str,
        step: Optional[str] = None,
        original_error: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            step=step,
            severity="fatal",
            should_stop=True,
            should_retry=False,
            original_error=original_error,
            context=context,
        )


class PlanningError(FatalAgentError):
    """规划错误"""
    pass


class MissingRequiredInputError(FatalAgentError):
    """缺少必需输入"""
    pass


class UnsupportedTaskError(FatalAgentError):
    """不支持的任务类型"""
    pass


# ============================================================
# 可重试错误
# ============================================================

class RetryableAgentError(AgentError):
    """可重试错误
    
    场景：
    - 网络请求失败
    - API 限流
    - 临时性服务不可用
    """
    
    def __init__(
        self,
        message: str,
        step: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        original_error: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            step=step,
            severity="error",
            should_stop=False,
            should_retry=True,
            original_error=original_error,
            context=context or {},
        )
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.context["max_retries"] = max_retries
        self.context["retry_delay"] = retry_delay


class NetworkError(RetryableAgentError):
    """网络错误"""
    pass


class RateLimitError(RetryableAgentError):
    """API 限流"""
    
    def __init__(
        self,
        message: str,
        step: Optional[str] = None,
        retry_after: Optional[float] = None,
        original_error: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            step=step,
            max_retries=5,
            retry_delay=retry_after or 60.0,
            original_error=original_error,
            context=context,
        )


class ServiceUnavailableError(RetryableAgentError):
    """服务不可用"""
    pass


# ============================================================
# 可降级错误
# ============================================================

class DegradableAgentError(AgentError):
    """可降级错误
    
    场景：
    - 部分数据源失败（可以用其他源）
    - LLM 生成失败（可以用模板）
    - PDF 解析失败（可以只用摘要）
    """
    
    def __init__(
        self,
        message: str,
        step: Optional[str] = None,
        degraded_mode: Optional[str] = None,
        original_error: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            step=step,
            severity="warning",
            should_stop=False,
            should_retry=False,
            original_error=original_error,
            context=context or {},
        )
        self.degraded_mode = degraded_mode
        self.context["degraded_mode"] = degraded_mode


class PartialDataSourceError(DegradableAgentError):
    """部分数据源失败"""
    
    def __init__(
        self,
        message: str,
        failed_sources: list[str],
        successful_sources: list[str],
        step: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            step=step,
            degraded_mode="partial_sources",
            original_error=original_error,
            context={
                "failed_sources": failed_sources,
                "successful_sources": successful_sources,
            },
        )


class LLMGenerationError(DegradableAgentError):
    """LLM 生成失败"""
    
    def __init__(
        self,
        message: str,
        step: Optional[str] = None,
        fallback_available: bool = True,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            step=step,
            degraded_mode="template_fallback" if fallback_available else None,
            original_error=original_error,
            context={"fallback_available": fallback_available},
        )


# ============================================================
# 特殊异常
# ============================================================

# AgentCancelledError 的唯一定义在 app/agent/graph.py（RuntimeError 子类）。
# 节点与服务层统一从 graph 导入该异常；此处不再维护第二个不兼容定义，
# 避免两套异常类并存导致捕获遗漏。


class QualityGateError(AgentError):
    """质量门禁失败
    
    场景：
    - 检索结果不足目标数量
    - 引用验证失败率过高
    - 生成质量不达标
    """
    
    def __init__(
        self,
        message: str,
        gate_name: str,
        threshold: Any,
        actual: Any,
        step: Optional[str] = None,
        allow_override: bool = True,
    ):
        super().__init__(
            message=message,
            step=step,
            severity="warning" if allow_override else "error",
            should_stop=not allow_override,
            should_retry=False,
            context={
                "gate_name": gate_name,
                "threshold": threshold,
                "actual": actual,
                "allow_override": allow_override,
            },
        )


# 历史上的 handle_agent_error / should_continue_after_error / format_error_for_user /
# ErrorRecoveryStrategy 辅助层随 WorkflowExecutor 一并移除：该执行器从未接入
# 主链（graph.py 为顺序编排），这些函数没有生产调用方；异常本身的分级语义
# （should_stop/should_retry/degraded_mode）保留在各类的 to_dict() 输出中，
# 供步骤日志与诊断导出使用。
