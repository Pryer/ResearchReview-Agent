"""外部 API 熔断保护器 (Circuit Breaker)。

提供基于连续故障统计的状态机（CLOSED -> OPEN -> HALF_OPEN），
防止下游服务故障时无限重试引发级联雪崩或长时间卡死。
"""

from __future__ import annotations

import enum
import functools
import threading
import time
from typing import Any, Callable, Dict, Optional

from app.core.exceptions import AgentBaseError
from app.core.logger import get_logger

logger = get_logger(__name__)


class CircuitState(str, enum.Enum):
    CLOSED = "closed"        # 正常状态：允许请求
    OPEN = "open"            # 熔断状态：快速失败
    HALF_OPEN = "half_open"  # 半开状态：放行探测请求


class SourceCircuitBrokenError(AgentBaseError):
    """外部数据源处于熔断状态错误。"""

    def __init__(self, source_name: str, message: Optional[str] = None):
        msg = message or f"外部学术数据源 '{source_name}' 触发熔断保护，正在冷却中，已快速跳过"
        super().__init__(msg)
        self.source_name = source_name


class CircuitBreaker:
    """线程安全的熔断器。"""

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_success_threshold: int = 1,
    ):
        """初始化熔断器。

        Args:
            name: 熔断器名称（数据源标识）。
            failure_threshold: 触发熔断的连续失败次数阈值。
            recovery_timeout: 熔断开启后的冷却恢复等待时间（秒）。
            half_open_success_threshold: 半开状态下连续成功多少次后恢复为 CLOSED。
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_success_threshold = half_open_success_threshold

        self.state = CircuitState.CLOSED
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.last_state_change = time.monotonic()
        self._lock = threading.Lock()

    def allow_request(self) -> bool:
        """检查当前是否允许发起请求。"""
        with self._lock:
            now = time.monotonic()
            if self.state == CircuitState.CLOSED:
                return True

            if self.state == CircuitState.OPEN:
                if now - self.last_state_change >= self.recovery_timeout:
                    logger.info(
                        "CircuitBreaker [%s] recovery timeout passed, moving from OPEN to HALF_OPEN",
                        self.name,
                    )
                    self.state = CircuitState.HALF_OPEN
                    self.last_state_change = now
                    self.consecutive_successes = 0
                    return True
                return False

            if self.state == CircuitState.HALF_OPEN:
                # 半开状态下允许探测请求
                return True

            return False

    def record_success(self) -> None:
        """记录一次调用成功。"""
        with self._lock:
            self.consecutive_failures = 0
            if self.state == CircuitState.HALF_OPEN:
                self.consecutive_successes += 1
                if self.consecutive_successes >= self.half_open_success_threshold:
                    logger.info(
                        "CircuitBreaker [%s] probe succeeded, moving from HALF_OPEN to CLOSED",
                        self.name,
                    )
                    self.state = CircuitState.CLOSED
                    self.last_state_change = time.monotonic()

    def record_failure(self, exception: Optional[Exception] = None) -> None:
        """记录一次调用失败。"""
        with self._lock:
            self.consecutive_failures += 1
            now = time.monotonic()
            if self.state == CircuitState.CLOSED:
                if self.consecutive_failures >= self.failure_threshold:
                    logger.warning(
                        "CircuitBreaker [%s] reached %d consecutive failures, moving from CLOSED to OPEN (cooling down for %.1fs)",
                        self.name,
                        self.consecutive_failures,
                        self.recovery_timeout,
                    )
                    self.state = CircuitState.OPEN
                    self.last_state_change = now

            elif self.state == CircuitState.HALF_OPEN:
                logger.warning(
                    "CircuitBreaker [%s] probe failed, moving from HALF_OPEN back to OPEN",
                    self.name,
                )
                self.state = CircuitState.OPEN
                self.last_state_change = now

    def reset(self) -> None:
        """重置熔断器为初始正常状态 (CLOSED)。"""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.consecutive_failures = 0
            self.consecutive_successes = 0
            self.last_state_change = time.monotonic()


_CIRCUIT_BREAKERS: Dict[str, CircuitBreaker] = {}
_CB_LOCK = threading.Lock()


def get_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
) -> CircuitBreaker:
    """获取或创建指定名称的熔断器单例。"""
    cb_key = name.lower().strip()
    with _CB_LOCK:
        if cb_key not in _CIRCUIT_BREAKERS:
            _CIRCUIT_BREAKERS[cb_key] = CircuitBreaker(
                name=cb_key,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
            )
        return _CIRCUIT_BREAKERS[cb_key]


def circuit_breaker(name: str, fallback_return: Any = None):
    """熔断保护装饰器。触发熔断时快速返回 fallback_return 或抛出 SourceCircuitBrokenError。"""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cb = get_circuit_breaker(name)
            if not cb.allow_request():
                logger.warning("Call to %s blocked by CircuitBreaker [%s]", func.__name__, name)
                if fallback_return is not None or callable(fallback_return):
                    return fallback_return() if callable(fallback_return) else fallback_return
                raise SourceCircuitBrokenError(name)

            try:
                result = func(*args, **kwargs)
                cb.record_success()
                return result
            except Exception as exc:
                cb.record_failure(exc)
                raise
        return wrapper
    return decorator
