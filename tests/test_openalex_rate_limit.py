"""OpenAlex 429 处理：重试、冷却门、熔断隔离与 common pool 降速。"""

import time

import pytest
import requests

from app.clients import openalex_client
from app.core.circuit_breaker import CircuitState, get_circuit_breaker
from app.core.rate_limiter import get_rate_limiter


class _FakeResponse:
    def __init__(self, status_code: int, headers: dict | None = None, payload: dict | None = None):
        self.status_code = status_code
        self.headers = headers or {}
        self._payload = payload if payload is not None else {"results": []}
        self.url = "https://api.openalex.org/works?search=test"

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Client Error", response=self)


@pytest.fixture(autouse=True)
def _clean_state():
    """每个用例前重置熔断器、冷却门与限流器速率。"""
    cb = get_circuit_breaker("openalex")
    cb.reset()
    openalex_client._reset_cooldown()
    limiter = get_rate_limiter("openalex")
    limiter.rate = 10.0
    limiter.capacity = 10.0
    limiter.tokens = 10.0
    yield
    cb.reset()
    openalex_client._reset_cooldown()


def test_429_retries_then_enters_cooldown_without_tripping_breaker(monkeypatch):
    """429 重试用尽后进入冷却门，且不把限流计入熔断失败。"""
    monkeypatch.setattr(openalex_client.settings, "openalex_max_retries", 2)
    monkeypatch.setattr(openalex_client.settings, "openalex_max_retry_wait_seconds", 5.0)
    monkeypatch.setattr(openalex_client.settings, "openalex_cooldown_seconds", 60.0)

    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(params)
        return _FakeResponse(429, headers={"Retry-After": "0"})

    monkeypatch.setattr(openalex_client.requests, "get", fake_get)
    monkeypatch.setattr(openalex_client.time, "sleep", lambda _s: None)

    result = openalex_client.search_openalex("classroom behavior", 2024, 2026, max_results=10)

    assert result == []
    # 首次 + 2 次重试
    assert len(calls) == 3
    cb = get_circuit_breaker("openalex")
    assert cb.state == CircuitState.CLOSED
    assert cb.consecutive_failures == 0
    assert openalex_client._cooldown_remaining() > 0


def test_cooldown_short_circuits_subsequent_calls(monkeypatch):
    """冷却期内不再发请求，直接返回空列表。"""
    openalex_client._enter_cooldown(30.0)

    def forbidden_get(*args, **kwargs):
        raise AssertionError("cooldown 期内不应发起请求")

    monkeypatch.setattr(openalex_client.requests, "get", forbidden_get)

    assert openalex_client.search_openalex("q", 2024, 2026) == []
    assert openalex_client.get_openalex_detail("W123") is None


def test_429_recovers_within_retry_budget(monkeypatch):
    """首次 429、重试成功时应返回结果且不留冷却。"""
    monkeypatch.setattr(openalex_client.settings, "openalex_max_retries", 3)
    monkeypatch.setattr(openalex_client.settings, "openalex_max_retry_wait_seconds", 5.0)

    responses = [
        _FakeResponse(429, headers={"Retry-After": "0"}),
        _FakeResponse(200, payload={"results": []}),
    ]

    def fake_get(url, params=None, timeout=None):
        return responses.pop(0)

    monkeypatch.setattr(openalex_client.requests, "get", fake_get)
    monkeypatch.setattr(openalex_client.time, "sleep", lambda _s: None)

    result = openalex_client.search_openalex("q", 2024, 2026)

    assert result == []
    assert responses == []
    assert openalex_client._cooldown_remaining() <= 0
    assert get_circuit_breaker("openalex").state == CircuitState.CLOSED


def test_non_429_http_error_still_trips_breaker(monkeypatch):
    """真实故障（5xx）仍应计入熔断失败，行为不退化。"""
    monkeypatch.setattr(openalex_client.requests, "get", lambda *a, **k: _FakeResponse(503))

    assert openalex_client.search_openalex("q", 2024, 2026) == []
    assert get_circuit_breaker("openalex").consecutive_failures == 1


def test_missing_mailto_downgrades_rate_to_common_pool(monkeypatch):
    """未配置 mailto 时降到 common pool 速率，并压缩突发桶容量。"""
    monkeypatch.setattr(openalex_client.settings, "crossref_mailto", "")

    limiter = openalex_client._get_limiter()

    assert limiter.rate == openalex_client._COMMON_POOL_RATE
    assert limiter.capacity == 1.0


def test_configured_mailto_keeps_polite_pool_rate(monkeypatch):
    """配置真实 mailto 时保留 polite pool 速率。"""
    monkeypatch.setattr(openalex_client.settings, "crossref_mailto", "someone@lab.edu")

    limiter = openalex_client._get_limiter()

    assert limiter.rate == 10.0
    assert openalex_client._effective_mailto() == "someone@lab.edu"


def test_huge_retry_after_is_capped(monkeypatch, caplog):
    """日配额耗尽式的巨大 Retry-After 必须被钳制，不能锁死整天。"""
    monkeypatch.setattr(openalex_client.settings, "openalex_max_retries", 0)
    monkeypatch.setattr(openalex_client.settings, "openalex_cooldown_seconds", 90.0)
    monkeypatch.setattr(openalex_client.settings, "openalex_max_cooldown_seconds", 900.0)

    # 实测值：指向次日零点的 11.5 小时。
    monkeypatch.setattr(
        openalex_client.requests,
        "get",
        lambda *a, **k: _FakeResponse(429, headers={"Retry-After": "41559"}),
    )

    with caplog.at_level("WARNING"):
        assert openalex_client.search_openalex("q", 2024, 2026) == []

    remaining = openalex_client._cooldown_remaining()
    assert 0 < remaining <= 900.0
    assert any("exceeds cooldown ceiling" in r.message for r in caplog.records)


def test_normal_retry_after_is_not_capped(monkeypatch):
    """上限以内的 Retry-After 照常生效。"""
    monkeypatch.setattr(openalex_client.settings, "openalex_max_retries", 0)
    monkeypatch.setattr(openalex_client.settings, "openalex_cooldown_seconds", 30.0)
    monkeypatch.setattr(openalex_client.settings, "openalex_max_cooldown_seconds", 900.0)

    monkeypatch.setattr(
        openalex_client.requests,
        "get",
        lambda *a, **k: _FakeResponse(429, headers={"Retry-After": "120"}),
    )

    assert openalex_client.search_openalex("q", 2024, 2026) == []

    remaining = openalex_client._cooldown_remaining()
    assert 110.0 < remaining <= 120.0


def test_detail_huge_retry_after_is_capped(monkeypatch):
    """detail 接口同样受冷却上限约束。"""
    monkeypatch.setattr(openalex_client.settings, "openalex_max_cooldown_seconds", 600.0)
    monkeypatch.setattr(
        openalex_client.requests,
        "get",
        lambda *a, **k: _FakeResponse(429, headers={"Retry-After": "41559"}),
    )

    assert openalex_client.get_openalex_detail("W123") is None
    assert 0 < openalex_client._cooldown_remaining() <= 600.0


def test_cooldown_expires_and_allows_retry(monkeypatch):
    """冷却到期后应重新放行请求，而不是永久跳过。"""
    monkeypatch.setattr(openalex_client.settings, "openalex_max_cooldown_seconds", 900.0)
    openalex_client._enter_cooldown(41559.0, retry_after=41559.0)
    assert openalex_client._cooldown_remaining() > 0

    # 把时钟推过钳制后的冷却窗口。
    base = time.monotonic()
    monkeypatch.setattr(openalex_client.time, "monotonic", lambda: base + 901.0)
    assert openalex_client._cooldown_remaining() <= 0

    monkeypatch.setattr(
        openalex_client.requests,
        "get",
        lambda *a, **k: _FakeResponse(200, payload={"results": []}),
    )
    assert openalex_client.search_openalex("q", 2024, 2026) == []
