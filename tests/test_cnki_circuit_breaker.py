"""测试 CNKI 客户端的熔断保护机制。"""

from unittest.mock import patch
from app.clients.cnki_client import search_cnki
from app.core.circuit_breaker import get_circuit_breaker, CircuitState

def test_cnki_circuit_breaker_fast_skips_when_open(monkeypatch):
    cb = get_circuit_breaker("cnki", failure_threshold=2, recovery_timeout=60.0)
    cb.state = CircuitState.CLOSED
    cb.consecutive_failures = 0

    # 模拟 Selenium 启动连续抛出异常
    def fake_build_driver(*args, **kwargs):
        raise RuntimeError("Chrome driver not installed")

    monkeypatch.setattr("app.clients.cnki_client.build_driver", fake_build_driver)

    # 第一次失败
    res1 = search_cnki("课堂行为分析", 2023, 2025, max_results=10)
    assert res1 == []
    assert cb.state == CircuitState.CLOSED
    assert cb.consecutive_failures == 1

    # 第二次失败 -> 触发熔断 OPEN
    res2 = search_cnki("课堂行为分析", 2023, 2025, max_results=10)
    assert res2 == []
    assert cb.state == CircuitState.OPEN

    # 第三次调用在 OPEN 状态下，不应该甚至去调用 build_driver，而是立即快速返回空列表
    build_driver_called = False
    def spy_build_driver(*args, **kwargs):
        nonlocal build_driver_called
        build_driver_called = True
        raise RuntimeError("Should not be called")

    monkeypatch.setattr("app.clients.cnki_client.build_driver", spy_build_driver)
    res3 = search_cnki("课堂行为分析", 2023, 2025, max_results=10)
    assert res3 == []
    assert build_driver_called is False  # 验证确实被熔断拦截并未调用底层驱动
