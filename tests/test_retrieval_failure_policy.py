"""Provider failures must not bypass rate limits, lose good results or spin forever."""

from __future__ import annotations

import time

import pytest
import requests

from app.agent.execution_budget import (
    AgentBudgetExceeded, AgentExecutionCancelled, budget_scope,
)
from app.clients import crossref_client, semantic_scholar_client
from app.core.circuit_breaker import get_circuit_breaker
from app.core.rate_limiter import (
    RateLimitAcquireTimeout, get_rate_limiter, rate_limited,
)
from app.core.retry_policy import (
    interruptible_sleep, retry_after_seconds,
)
from app.schemas.paper_schema import PaperMetadata
from app.tools.search_papers import search_papers


class Response:
    def __init__(self, status, *, headers=None, payload=None):
        self.status_code = status
        self.headers = headers or {}
        self._payload = payload or {"message": {"items": []}}
        self.url = "https://example.test/works"
        self.text = ""

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def clean_provider_state(monkeypatch):
    crossref_client._reset_cooldown()
    monkeypatch.setattr(semantic_scholar_client, "_S2_COOLDOWN_UNTIL", 0.0)
    monkeypatch.setattr(semantic_scholar_client, "_S2_NEXT_ALLOWED_AT", 0.0)
    for name, domain in (("crossref", "crossref"), ("semantic_scholar", "semanticscholar")):
        get_circuit_breaker(name).reset()
        limiter = get_rate_limiter(domain)
        limiter.tokens = limiter.capacity
    yield
    crossref_client._reset_cooldown()
    for name in ("crossref", "semantic_scholar"):
        get_circuit_breaker(name).reset()


def test_decorator_does_not_execute_without_token(monkeypatch):
    monkeypatch.setattr(
        "app.core.rate_limiter.get_rate_limiter",
        lambda domain: type("Denied", (), {"acquire": lambda self, **kw: False})(),
    )
    executed = []

    @rate_limited("crossref")
    def protected():
        executed.append(True)

    with pytest.raises(RateLimitAcquireTimeout):
        protected()
    assert executed == []


def test_retry_after_accepts_http_date_and_rejects_invalid_values():
    from datetime import datetime, timedelta, timezone
    from email.utils import format_datetime

    future = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=90))
    assert 80 <= retry_after_seconds({"Retry-After": future}) <= 90
    assert retry_after_seconds({"Retry-After": "2"}) == 2.0
    assert retry_after_seconds({"Retry-After": "-1"}) == 0.0
    assert retry_after_seconds({"Retry-After": "garbage"}) == 0.0


def test_retry_wait_obeys_cancellation(monkeypatch):
    flag = {"cancelled": False}
    monkeypatch.setattr(
        "app.core.retry_policy.time.sleep",
        lambda seconds: flag.__setitem__("cancelled", True),
    )
    with budget_scope({}, should_cancel=lambda: flag["cancelled"]):
        with pytest.raises(AgentExecutionCancelled):
            interruptible_sleep(1.0)


def test_retry_wait_obeys_deadline():
    with budget_scope({}) as active:
        active["deadline"] = time.monotonic() - 1.0
        with pytest.raises(AgentBudgetExceeded):
            interruptible_sleep(1.0)


def test_crossref_429_cools_down_both_search_and_detail_without_breaker(monkeypatch):
    monkeypatch.setattr(crossref_client.settings, "crossref_max_retries", 0)
    calls = []

    def get(*args, **kwargs):
        calls.append(args)
        return Response(429, headers={"Retry-After": "120"})

    monkeypatch.setattr(crossref_client.requests, "get", get)
    assert crossref_client.search_crossref("topic", 2024, 2026) == []
    assert crossref_client.get_crossref_detail("10.1/test") is None
    assert len(calls) == 1
    assert crossref_client._cooldown_remaining() > 0
    assert get_circuit_breaker("crossref").consecutive_failures == 0


def test_crossref_429_then_success_is_bounded_and_does_not_trip_breaker(monkeypatch):
    monkeypatch.setattr(crossref_client.settings, "crossref_max_retries", 1)
    monkeypatch.setattr(crossref_client.settings, "crossref_max_retry_wait_seconds", 5.0)
    monkeypatch.setattr("app.core.retry_policy.time.sleep", lambda seconds: None)
    monkeypatch.setattr(crossref_client.random, "uniform", lambda a, b: 0.0)
    calls = []

    def get(*args, **kwargs):
        calls.append(args)
        return (Response(429, headers={"Retry-After": "0"}) if len(calls) == 1
                else Response(200))

    monkeypatch.setattr(crossref_client.requests, "get", get)
    assert crossref_client.search_crossref("topic", 2024, 2026) == []
    assert len(calls) == 2
    assert crossref_client._cooldown_remaining() == 0
    assert get_circuit_breaker("crossref").consecutive_failures == 0


def test_crossref_timeout_retries_once_then_succeeds(monkeypatch):
    monkeypatch.setattr(crossref_client.settings, "crossref_max_retries", 1)
    monkeypatch.setattr("app.core.retry_policy.time.sleep", lambda seconds: None)
    attempts = []

    def get(*args, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise requests.Timeout("transient")
        return Response(200)

    monkeypatch.setattr(crossref_client.requests, "get", get)
    assert crossref_client.search_crossref("topic", 2024, 2026) == []
    assert len(attempts) == 2
    assert get_circuit_breaker("crossref").consecutive_failures == 0


def test_crossref_auth_failure_is_not_retried_or_counted_as_outage(monkeypatch):
    monkeypatch.setattr(crossref_client.settings, "crossref_max_retries", 3)
    attempts = []
    monkeypatch.setattr(crossref_client.requests, "get",
                        lambda *a, **kw: attempts.append(True) or Response(401))
    assert crossref_client.search_crossref("topic", 2024, 2026) == []
    assert len(attempts) == 1
    assert get_circuit_breaker("crossref").consecutive_failures == 0


def test_semantic_scholar_429_uses_cooldown_not_outage_breaker(monkeypatch):
    s2 = semantic_scholar_client
    monkeypatch.setattr(s2.settings, "semantic_scholar_max_retries", 0)
    monkeypatch.setattr(s2, "_respect_rate_limit", lambda: None)
    monkeypatch.setattr(s2.requests, "get",
                        lambda *a, **kw: Response(429, headers={"Retry-After": "60"}))
    with pytest.raises(s2.SemanticScholarRateLimitError):
        s2._request_json("https://api.semanticscholar.org/x", {})
    assert get_circuit_breaker("semantic_scholar").consecutive_failures == 0


def test_multi_source_keeps_successful_papers_when_another_provider_rate_limits(monkeypatch):
    import importlib
    tool = importlib.import_module("app.tools.search_papers")

    def arxiv(*args):
        return [PaperMetadata(
            paper_id="arxiv:123", title="A valid paper", authors=["A"],
            source="arxiv", year=2025,
        )]

    def crossref(*args):
        raise RuntimeError("HTTP 429 rate limited")

    monkeypatch.setattr(tool, "_search_arxiv", arxiv)
    monkeypatch.setattr(tool, "_search_crossref", crossref)
    diagnostics = []
    results = search_papers(
        "action recognition", 2024, 2026, 5,
        sources=["arxiv", "crossref"], diagnostics=diagnostics,
        enable_dual_channel=False,
    )
    assert [paper.paper_id for paper in results] == ["arxiv:123"]
    assert any(x.source == "crossref" and x.outcome == "rate_limited"
               for x in diagnostics)
