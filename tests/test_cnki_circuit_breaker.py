"""测试 CNKI 客户端的熔断保护机制。"""

from unittest.mock import patch
import pytest
from app.clients import cnki_client
from app.clients.cnki_client import search_cnki
from app.core.circuit_breaker import get_circuit_breaker, CircuitState


@pytest.fixture(autouse=True)
def _reset_cnki_driver_state():
    """每个用例前后清除进程级驱动不兼容短路标志，避免跨用例泄漏。"""
    cnki_client._reset_driver_incompatibility()
    yield
    cnki_client._reset_driver_incompatibility()


def test_cnki_driver_version_mismatch_is_classified_and_not_relaunched(monkeypatch):
    """T14：驱动/浏览器版本不匹配被单独分类，且不再反复启动同一不兼容驱动。"""
    cb = get_circuit_breaker("cnki", failure_threshold=2, recovery_timeout=60.0)
    cb.state = CircuitState.CLOSED
    cb.consecutive_failures = 0

    calls = {"n": 0}

    def fake_build_driver(*args, **kwargs):
        calls["n"] += 1
        raise cnki_client.SessionNotCreatedException(
            "session not created: This version of ChromeDriver only supports "
            "Chrome version 152\nCurrent browser version is 154.0.8037.58"
        )

    monkeypatch.setattr("app.clients.cnki_client.build_driver", fake_build_driver)

    # 第一次：识别为驱动不兼容，记录进程级短路标志并给出版本修复信息。
    assert search_cnki("课堂行为分析", 2024, 2026, max_results=10) == []
    assert calls["n"] == 1
    assert cnki_client._DRIVER_INCOMPATIBLE
    reason = cnki_client._DRIVER_INCOMPATIBLE["reason"]
    assert "152" in reason and "154" in reason

    # 第二次：直接短路，绝不再次启动已知不兼容的驱动。
    assert search_cnki("课堂行为分析", 2024, 2026, max_results=10) == []
    assert calls["n"] == 1


def test_cnki_generic_driver_error_is_not_marked_incompatible(monkeypatch):
    """T14：普通驱动/网络错误不归为版本不兼容，仍走熔断而非永久短路。"""
    cb = get_circuit_breaker("cnki", failure_threshold=2, recovery_timeout=60.0)
    cb.state = CircuitState.CLOSED
    cb.consecutive_failures = 0

    def fake_build_driver(*args, **kwargs):
        raise RuntimeError("connection refused")

    monkeypatch.setattr("app.clients.cnki_client.build_driver", fake_build_driver)

    assert search_cnki("课堂行为分析", 2024, 2026, max_results=10) == []
    assert not cnki_client._DRIVER_INCOMPATIBLE
    assert cb.consecutive_failures == 1

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
