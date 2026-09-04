"""高精度、线程安全的令牌桶限流器 (Token Bucket Rate Limiter)。

支持按数据源域名/客户端独立配置 QPS 限制，平滑请求突发，避免触发学术 API 429 封禁。
"""

from __future__ import annotations

import functools
import threading
import time
from typing import Any, Callable, Dict, Optional

from app.core.logger import get_logger

logger = get_logger(__name__)


class TokenBucketRateLimiter:
    """线程安全的令牌桶算法限流器。"""

    def __init__(self, rate: float, capacity: Optional[float] = None, name: str = "default"):
        """初始化令牌桶。

        Args:
            rate: 每秒生成的令牌数 (QPS)。
            capacity: 桶容量（最大突发令牌数），默认等于 rate。
            name: 限流器名称（用于日志和指标监控）。
        """
        if rate <= 0:
            raise ValueError(f"Rate must be > 0, got {rate}")
        self.rate = float(rate)
        self.capacity = float(capacity if capacity is not None else max(1.0, rate))
        self.tokens = self.capacity
        self.last_refill = time.monotonic()
        self.name = name
        self._lock = threading.Lock()

    def _refill(self) -> None:
        """补充令牌。调用时必须已持有 _lock。"""
        now = time.monotonic()
        elapsed = now - self.last_refill
        if elapsed > 0:
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_refill = now

    def acquire(self, tokens: float = 1.0, timeout: Optional[float] = 30.0) -> bool:
        """阻塞直到获取指定数量的令牌或超时。

        Args:
            tokens: 所需令牌数。
            timeout: 最大等待秒数。None 表示无限等待。

        Returns:
            True 表示成功获取令牌，False 表示超时未获取到。
        """
        if tokens <= 0:
            return True

        deadline = (time.monotonic() + timeout) if timeout is not None else None

        while True:
            with self._lock:
                self._refill()
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True

                # 计算需要等待的时间
                deficit = tokens - self.tokens
                wait_time = deficit / self.rate

            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.warning(
                        "RateLimiter [%s] acquire timed out after %s seconds",
                        self.name,
                        timeout,
                    )
                    return False
                wait_time = min(wait_time, remaining)

            if wait_time > 0:
                time.sleep(max(0.001, wait_time))

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """非阻塞尝试获取令牌。"""
        with self._lock:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False


# 默认学术数据源 QPS 配置
DEFAULT_DOMAIN_RATES: Dict[str, float] = {
    "semanticscholar": 1.0,   # S2 未认证建议 1 req/s
    "openalex": 10.0,         # OpenAlex polite pool 支持 10 req/s
    "crossref": 5.0,          # Crossref polite 支持 5-10 req/s
    "arxiv": 3.0,             # arXiv API 建议每 3 秒不多于 1 次请求或 3 req/s
    "cnki": 0.5,              # 知网自动化爬取降频防封
    "default": 5.0,
}

_LIMITERS: Dict[str, TokenBucketRateLimiter] = {}
_REGISTRY_LOCK = threading.Lock()


def get_rate_limiter(domain: str, rate: Optional[float] = None) -> TokenBucketRateLimiter:
    """获取或创建指定域名的限流器单例。"""
    domain_key = domain.lower().strip()
    with _REGISTRY_LOCK:
        if domain_key not in _LIMITERS:
            effective_rate = rate or DEFAULT_DOMAIN_RATES.get(domain_key, DEFAULT_DOMAIN_RATES["default"])
            _LIMITERS[domain_key] = TokenBucketRateLimiter(
                rate=effective_rate,
                capacity=max(1.0, effective_rate),
                name=domain_key,
            )
        return _LIMITERS[domain_key]


def rate_limited(domain: str, tokens: float = 1.0, timeout: Optional[float] = 30.0):
    """限流装饰器。"""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            limiter = get_rate_limiter(domain)
            acquired = limiter.acquire(tokens=tokens, timeout=timeout)
            if not acquired:
                logger.warning("Rate limit exceeded for domain '%s' on %s", domain, func.__name__)
            return func(*args, **kwargs)
        return wrapper
    return decorator
