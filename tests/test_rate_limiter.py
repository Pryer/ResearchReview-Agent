"""Tests for TokenBucketRateLimiter."""

import time
from unittest.mock import MagicMock

from app.core.rate_limiter import TokenBucketRateLimiter, get_rate_limiter, rate_limited

def test_rate_limiter_acquires_tokens():
    limiter = TokenBucketRateLimiter(rate=10.0, capacity=10.0, name="test_limiter")
    assert limiter.acquire(tokens=5.0) is True
    assert limiter.try_acquire(tokens=5.0) is True
    assert limiter.try_acquire(tokens=1.0) is False

def test_rate_limiter_refills_over_time():
    limiter = TokenBucketRateLimiter(rate=20.0, capacity=2.0, name="test_refill")
    assert limiter.acquire(tokens=2.0) is True
    assert limiter.try_acquire(tokens=1.0) is False
    
    # Wait for refill (20 tokens/sec -> 0.1 sec gives 2 tokens)
    time.sleep(0.12)
    assert limiter.try_acquire(tokens=2.0) is True

def test_rate_limiter_timeout():
    limiter = TokenBucketRateLimiter(rate=1.0, capacity=1.0, name="test_timeout")
    assert limiter.acquire(tokens=1.0) is True
    # Next token requires 1 second, with timeout 0.05s it should return False
    assert limiter.acquire(tokens=1.0, timeout=0.05) is False

def test_rate_limited_decorator():
    calls = []

    @rate_limited("test_domain", tokens=1.0)
    def my_func(x):
        calls.append(x)
        return x * 2

    res = my_func(5)
    assert res == 10
    assert calls == [5]


def test_semantic_scholar_requests_pass_through_rate_limiter(monkeypatch):
    """M6 接线回归：Semantic Scholar 请求必须先获取令牌桶令牌。"""
    from unittest.mock import MagicMock
    import app.clients.semantic_scholar_client as s2

    acquired = []
    monkeypatch.setattr(
        "app.core.rate_limiter.get_rate_limiter",
        lambda domain, rate=None: MagicMock(
            acquire=lambda tokens=1.0, timeout=None: acquired.append(domain) or True,
        ),
    )
    resp = MagicMock(status_code=200)
    resp.json = lambda: {"data": []}
    monkeypatch.setattr(s2.requests, "get", lambda *a, **k: resp)
    monkeypatch.setattr(
        "app.core.circuit_breaker.get_circuit_breaker",
        lambda *a, **k: MagicMock(allow_request=lambda: True, record_success=lambda: None),
    )
    monkeypatch.setattr(s2, "_respect_rate_limit", lambda: None)

    s2._request_json("https://api.semanticscholar.org/x", {})

    assert acquired == ["semanticscholar"]


def test_arxiv_get_skips_request_when_token_times_out(monkeypatch):
    """M6 接线回归：arXiv 令牌等待超时应放弃请求而不是照发不误。"""
    import pytest
    import requests
    import app.clients.arxiv_client as arxiv

    monkeypatch.setattr(
        "app.core.rate_limiter.get_rate_limiter",
        lambda domain, rate=None: MagicMock(
            acquire=lambda tokens=1.0, timeout=None: False,
        ),
    )
    sent = []
    monkeypatch.setattr(arxiv.requests, "get", lambda *a, **k: sent.append(a) or MagicMock())

    with pytest.raises(requests.RequestException):
        arxiv._arxiv_get("https://export.arxiv.org/api/query", timeout=5)

    assert sent == []  # 令牌未获取时不得发出真实请求


def test_cnki_search_skips_when_token_times_out(monkeypatch):
    """M6 接线回归：CNKI 令牌等待超时应跳过本次爬取。"""
    import app.clients.cnki_client as cnki

    monkeypatch.setattr(
        "app.core.circuit_breaker.get_circuit_breaker",
        lambda *a, **k: MagicMock(allow_request=lambda: True),
    )
    monkeypatch.setattr(
        "app.core.rate_limiter.get_rate_limiter",
        lambda domain, rate=None: MagicMock(
            acquire=lambda tokens=1.0, timeout=None: False,
        ),
    )

    assert cnki.search_cnki("课堂行为分析", 2022, 2026, max_results=5) == []
