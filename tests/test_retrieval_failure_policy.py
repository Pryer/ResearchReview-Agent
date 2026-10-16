"""Provider failures must not bypass rate limits, lose good results or spin forever."""

from __future__ import annotations

import time

import pytest
import requests

from app.agent.execution_budget import (
    AgentBudgetExceeded,
    AgentExecutionCancelled,
    budget_scope,
)
from app.clients import crossref_client, semantic_scholar_client
from app.core.circuit_breaker import get_circuit_breaker
from app.core.rate_limiter import (
    RateLimitAcquireTimeout,
    get_rate_limiter,
    rate_limited,
)
from app.core.retry_policy import (
    interruptible_sleep,
    retry_after_seconds,
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

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


@pytest.fixture(autouse=True)
def clean_provider_state(monkeypatch):
    from app.clients import arxiv_client, openalex_client

    crossref_client._reset_cooldown()
    openalex_client._reset_cooldown()
    monkeypatch.setattr(arxiv_client, "_COOLDOWN_UNTIL", 0.0)
    monkeypatch.setattr(arxiv_client, "_last_request_time", 0.0)
    monkeypatch.setattr(semantic_scholar_client, "_S2_COOLDOWN_UNTIL", 0.0)
    monkeypatch.setattr(semantic_scholar_client, "_S2_NEXT_ALLOWED_AT", 0.0)
    for name, domain in (
        ("crossref", "crossref"),
        ("semantic_scholar", "semanticscholar"),
        ("openalex", "openalex"),
    ):
        get_circuit_breaker(name).reset()
        limiter = get_rate_limiter(domain)
        limiter.tokens = limiter.capacity
        limiter.last_refill = time.monotonic()
    yield
    crossref_client._reset_cooldown()
    openalex_client._reset_cooldown()
    for name, domain in (
        ("crossref", "crossref"),
        ("semantic_scholar", "semanticscholar"),
        ("openalex", "openalex"),
    ):
        get_circuit_breaker(name).reset()
        limiter = get_rate_limiter(domain)
        limiter.last_refill = time.monotonic()
        limiter.tokens = limiter.capacity


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


def test_crossref_429_then_success_is_bounded_and_does_not_trip_breaker(monkeypatch, retry_clock):
    monkeypatch.setattr(crossref_client.settings, "crossref_max_retries", 1)
    monkeypatch.setattr(crossref_client.settings, "crossref_max_retry_wait_seconds", 5.0)
    monkeypatch.setattr(crossref_client.random, "uniform", lambda a, b: 0.0)
    calls = []

    def get(*args, **kwargs):
        calls.append(args)
        return Response(429, headers={"Retry-After": "0"}) if len(calls) == 1 else Response(200)

    monkeypatch.setattr(crossref_client.requests, "get", get)
    assert crossref_client.search_crossref("topic", 2024, 2026) == []
    assert len(calls) == 2
    assert crossref_client._cooldown_remaining() == 0
    assert get_circuit_breaker("crossref").consecutive_failures == 0


def test_crossref_timeout_retries_once_then_succeeds(monkeypatch, retry_clock):
    monkeypatch.setattr(crossref_client.settings, "crossref_max_retries", 1)
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
    monkeypatch.setattr(
        crossref_client.requests, "get", lambda *a, **kw: attempts.append(True) or Response(401)
    )
    assert crossref_client.search_crossref("topic", 2024, 2026) == []
    assert len(attempts) == 1
    assert get_circuit_breaker("crossref").consecutive_failures == 0


def test_semantic_scholar_429_uses_cooldown_not_outage_breaker(monkeypatch):
    s2 = semantic_scholar_client
    monkeypatch.setattr(s2.settings, "semantic_scholar_max_retries", 0)
    monkeypatch.setattr(s2, "_respect_rate_limit", lambda: None)
    monkeypatch.setattr(
        s2.requests, "get", lambda *a, **kw: Response(429, headers={"Retry-After": "60"})
    )
    with pytest.raises(s2.SemanticScholarRateLimitError):
        s2._request_json("https://api.semanticscholar.org/x", {})
    assert get_circuit_breaker("semantic_scholar").consecutive_failures == 0


def test_multi_source_keeps_successful_papers_when_another_provider_rate_limits(monkeypatch):
    import importlib

    tool = importlib.import_module("app.tools.search_papers")

    def arxiv(*args):
        return [
            PaperMetadata(
                paper_id="arxiv:123",
                title="A valid paper",
                authors=["A"],
                source="arxiv",
                year=2025,
            )
        ]

    def crossref(*args):
        raise RuntimeError("HTTP 429 rate limited")

    monkeypatch.setattr(tool, "_search_arxiv", arxiv)
    monkeypatch.setattr(tool, "_search_crossref", crossref)
    diagnostics = []
    results = search_papers(
        "action recognition",
        2024,
        2026,
        5,
        sources=["arxiv", "crossref"],
        diagnostics=diagnostics,
        enable_dual_channel=False,
    )
    assert [paper.paper_id for paper in results] == ["arxiv:123"]
    assert any(x.source == "crossref" and x.outcome == "rate_limited" for x in diagnostics)


def test_semantic_scholar_timeout_retries_without_losing_result(monkeypatch, retry_clock):
    s2 = semantic_scholar_client
    monkeypatch.setattr(s2.settings, "semantic_scholar_max_retries", 1)
    monkeypatch.setattr(s2, "_respect_rate_limit", lambda: None)
    calls = []

    def get(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise requests.Timeout("temporary")
        return Response(200, payload={"data": []})

    monkeypatch.setattr(s2.requests, "get", get)
    assert s2._request_json("https://api.semanticscholar.org/x", {}) == {"data": []}
    assert len(calls) == 2
    assert get_circuit_breaker("semantic_scholar").consecutive_failures == 0


def test_openalex_timeout_retries_then_succeeds(monkeypatch, retry_clock):
    from app.clients import openalex_client as oa

    cb = get_circuit_breaker("openalex")
    cb.reset()
    oa._reset_cooldown()
    limiter = get_rate_limiter("openalex")
    limiter.tokens = limiter.capacity
    monkeypatch.setattr(oa.settings, "openalex_max_retries", 1)
    calls = []

    def get(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise requests.Timeout("temporary")
        return Response(200, payload={"results": []})

    monkeypatch.setattr(oa.requests, "get", get)
    assert oa.search_openalex("topic", 2024, 2026) == []
    assert len(calls) == 2
    assert cb.consecutive_failures == 0
    cb.reset()
    oa._reset_cooldown()


def test_arxiv_does_not_retry_when_retry_after_exceeds_budget(monkeypatch):
    from app.clients import arxiv_client as arxiv

    monkeypatch.setattr(arxiv, "_reserve_request_slot", lambda: None)
    limiter = get_rate_limiter("arxiv")
    limiter.tokens = limiter.capacity
    calls = []

    def get(*args, **kwargs):
        calls.append(1)
        return Response(429, headers={"Retry-After": "120"})

    monkeypatch.setattr(arxiv.requests, "get", get)
    with pytest.raises(requests.HTTPError):
        arxiv._arxiv_get("https://export.arxiv.org/api/query", timeout=5, max_retries=3)
    assert len(calls) == 1


def test_worker_budget_abort_is_not_misreported_as_source_failure(monkeypatch):
    import importlib

    tool = importlib.import_module("app.tools.search_papers")

    def fail_budget(*args):
        raise AgentBudgetExceeded("research execution deadline exceeded")

    monkeypatch.setattr(tool, "_search_crossref", fail_budget)
    diagnostics = []
    with pytest.raises(AgentBudgetExceeded):
        search_papers(
            "action recognition",
            2024,
            2026,
            5,
            sources=["crossref"],
            diagnostics=diagnostics,
            enable_dual_channel=False,
        )
    assert diagnostics == []


def provider_request(name):
    from app.clients import openalex_client

    if name == "crossref":
        return lambda: crossref_client._request_crossref("https://example.test/works")
    if name == "openalex":
        return lambda: openalex_client._request_openalex("https://example.test/works", params={})
    return lambda: semantic_scholar_client._request_json("https://example.test/works", {})


def provider_module(name):
    from app.clients import openalex_client, arxiv_client

    return {
        "crossref": crossref_client,
        "arxiv": arxiv_client,
        "openalex": openalex_client,
        "semantic_scholar": semantic_scholar_client,
    }[name]


def test_token_wait_checks_cancellation_without_waiting_for_token_timeout(monkeypatch):
    from app.core.rate_limiter import TokenBucketRateLimiter

    flag = {"cancel": False}
    limiter = TokenBucketRateLimiter(rate=0.001, capacity=1)
    assert limiter.try_acquire()
    monkeypatch.setattr(
        "app.core.retry_policy.time.sleep", lambda seconds: flag.update(cancel=True)
    )
    with budget_scope({}, should_cancel=lambda: flag["cancel"]):
        with pytest.raises(AgentExecutionCancelled):
            limiter.acquire(timeout=30)


def test_token_wait_obeys_execution_deadline(retry_clock):
    from app.core.rate_limiter import TokenBucketRateLimiter

    limiter = TokenBucketRateLimiter(rate=0.001, capacity=1)
    assert limiter.try_acquire()
    with budget_scope({}) as active:
        active["deadline"] = retry_clock.now + 0.1
        with pytest.raises(AgentBudgetExceeded):
            limiter.acquire(timeout=30)
    assert sum(retry_clock.waits) <= 0.25


def test_bounded_http_timeout_uses_remaining_execution_window(retry_clock):
    from app.core.retry_policy import bounded_request_timeout

    with budget_scope({}) as active:
        active["deadline"] = retry_clock.now + 0.125
        assert 0 < bounded_request_timeout(60) <= 0.125


@pytest.mark.parametrize("provider", ["crossref", "openalex", "semantic_scholar"])
def test_no_request_when_circuit_opens_during_token_wait(provider, monkeypatch):
    from types import SimpleNamespace

    module = provider_module(provider)
    cb = get_circuit_breaker(provider)
    monkeypatch.setattr(cb, "failure_threshold", 1)

    def acquire(*args, **kwargs):
        cb.record_failure()
        return True

    monkeypatch.setattr(
        "app.core.rate_limiter.get_rate_limiter",
        lambda *a, **k: SimpleNamespace(acquire=acquire, rate=1.0),
    )
    sent = []
    monkeypatch.setattr(module.requests, "get", lambda *a, **k: sent.append(1))
    assert provider_request(provider)() is None
    assert sent == []


@pytest.mark.parametrize("provider", ["crossref", "openalex", "semantic_scholar"])
def test_no_request_when_peer_enters_cooldown_during_token_wait(provider, monkeypatch):
    from types import SimpleNamespace

    module = provider_module(provider)

    def acquire(*args, **kwargs):
        module._enter_cooldown(30)
        return True

    monkeypatch.setattr(
        "app.core.rate_limiter.get_rate_limiter",
        lambda *a, **k: SimpleNamespace(acquire=acquire, rate=1.0),
    )
    sent = []
    monkeypatch.setattr(module.requests, "get", lambda *a, **k: sent.append(1))
    if provider == "semantic_scholar":
        with pytest.raises(semantic_scholar_client.SemanticScholarRateLimitError):
            provider_request(provider)()
    else:
        assert provider_request(provider)() is None
    assert sent == []


@pytest.mark.parametrize("provider", ["crossref", "openalex", "semantic_scholar"])
def test_retry_stops_when_transient_failure_opens_breaker(provider, monkeypatch, retry_clock):
    module = provider_module(provider)
    cb = get_circuit_breaker(provider)
    monkeypatch.setattr(cb, "failure_threshold", 1)
    monkeypatch.setattr(module.settings, provider + "_max_retries", 3)
    monkeypatch.setattr(semantic_scholar_client, "_respect_rate_limit", lambda: None)
    sent = []

    def get(*args, **kwargs):
        sent.append(1)
        raise requests.Timeout("transient")

    monkeypatch.setattr(module.requests, "get", get)
    assert provider_request(provider)() is None
    assert len(sent) == 1


@pytest.mark.parametrize("provider", ["crossref", "openalex", "semantic_scholar"])
def test_inline_retry_publishes_cooldown_to_peer_calls(provider, monkeypatch, retry_clock):
    module = provider_module(provider)
    monkeypatch.setattr(module.settings, provider + "_max_retries", 1)
    monkeypatch.setattr(semantic_scholar_client, "_respect_rate_limit", lambda: None)
    sent, peers = [], []

    def get(*args, **kwargs):
        sent.append(1)
        return Response(429, headers={"Retry-After": "2"}) if len(sent) == 1 else Response(200)

    monkeypatch.setattr(module.requests, "get", get)

    def sleep(seconds):
        if not peers:
            peers.append(True)
            if provider == "semantic_scholar":
                with pytest.raises(semantic_scholar_client.SemanticScholarRateLimitError):
                    provider_request(provider)()
            else:
                assert provider_request(provider)() is None
            assert len(sent) == 1
        retry_clock.now += seconds + 0.000001

    monkeypatch.setattr(module.time, "sleep", sleep)
    assert provider_request(provider)() is not None
    assert len(sent) == 2
    assert get_circuit_breaker(provider).consecutive_failures == 0


@pytest.mark.parametrize("provider", ["crossref", "openalex", "semantic_scholar"])
@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_auth_and_not_found_do_not_retry_or_trip_outage_breaker(provider, status, monkeypatch):
    module = provider_module(provider)
    monkeypatch.setattr(semantic_scholar_client, "_respect_rate_limit", lambda: None)
    sent = []
    monkeypatch.setattr(module.requests, "get", lambda *a, **k: sent.append(1) or Response(status))
    assert provider_request(provider)() is None
    assert len(sent) == 1
    assert get_circuit_breaker(provider).consecutive_failures == 0


def test_semantic_scholar_caps_huge_cooldown_without_shortening_existing_one(monkeypatch):
    s2 = semantic_scholar_client
    monkeypatch.setattr(s2.settings, "semantic_scholar_max_cooldown_seconds", 10.0)
    s2._enter_cooldown(1000000)
    first = s2._S2_COOLDOWN_UNTIL
    assert 0 < first - time.monotonic() <= 10
    s2._enter_cooldown(1)
    assert s2._S2_COOLDOWN_UNTIL == first


def test_openalex_detail_retries_transient_timeout(monkeypatch, retry_clock):
    from app.clients import openalex_client as oa

    monkeypatch.setattr(oa.settings, "openalex_max_retries", 1)
    sent = []

    def get(*args, **kwargs):
        sent.append(1)
        if len(sent) == 1:
            raise requests.Timeout("transient")
        return Response(200, payload={"id": "W1"})

    monkeypatch.setattr(oa.requests, "get", get)
    assert oa.get_openalex_detail("W1") == {"id": "W1"}
    assert len(sent) == 2


def test_arxiv_rate_limit_cooldown_protects_following_calls(monkeypatch):
    from app.clients import arxiv_client as arxiv

    monkeypatch.setattr(arxiv, "_reserve_request_slot", lambda: None)
    sent = []
    monkeypatch.setattr(
        arxiv.requests,
        "get",
        lambda *a, **k: sent.append(1) or Response(429, headers={"Retry-After": "120"}),
    )
    with pytest.raises(requests.HTTPError):
        arxiv._arxiv_get("https://example.test/", timeout=5)
    with pytest.raises(arxiv.ArxivRateLimitError):
        arxiv._arxiv_get("https://example.test/", timeout=5)
    assert len(sent) == 1


@pytest.mark.parametrize("failure_first", [False, True])
def test_dual_channel_failure_keeps_success_count_in_both_completion_orders(
    monkeypatch, failure_first
):
    import importlib
    from concurrent.futures import wait as real_wait

    tool = importlib.import_module("app.tools.search_papers")

    def client(*args):
        if args[-1] == "date":
            tool.record_client_diagnostic(
                "timeout", error_code="TIMEOUT", message="date channel timeout"
            )
            return []
        return [
            PaperMetadata(paper_id="doi:one", title="Verified result", source="crossref", year=2025)
        ]

    monkeypatch.setattr(tool, "_search_crossref", client)

    def ordered_wait(pending, **kwargs):
        done, _ = real_wait(pending)
        first = min(
            done, key=lambda f: bool(f.result()[0]) if failure_first else not bool(f.result()[0])
        )
        return {first}, pending - {first}

    monkeypatch.setattr(tool, "wait", ordered_wait)
    diagnostics = []
    papers = tool.search_papers(
        "topic", 2024, 2026, 5, sources=["crossref"], diagnostics=diagnostics
    )
    assert len(papers) == 1
    assert diagnostics[0].returned_count == 1
    assert diagnostics[0].outcome == "timeout"
    assert diagnostics[0].failure_category == "timeout"


def test_direct_cancellation_interrupts_workers_without_active_agent_budget(monkeypatch):
    import importlib
    from threading import Event

    tool = importlib.import_module("app.tools.search_papers")
    started, completed = Event(), []

    def client(*args):
        started.set()
        interruptible_sleep(60)
        completed.append(True)
        return []

    monkeypatch.setattr(tool, "_search_crossref", client)
    with pytest.raises(InterruptedError):
        tool.search_papers(
            "topic",
            2024,
            2026,
            5,
            sources=["crossref"],
            should_cancel=started.is_set,
            enable_dual_channel=False,
        )
    assert completed == []


def test_worker_budget_abort_stops_retrying_sibling_and_preserves_original_reason(monkeypatch):
    import importlib
    from threading import Barrier, Event

    tool = importlib.import_module("app.tools.search_papers")
    ready, stopped = Barrier(2), Event()

    def fail(*args):
        ready.wait(timeout=3)
        raise AgentBudgetExceeded("original-budget-stop")

    def waiting(*args):
        ready.wait(timeout=3)
        try:
            interruptible_sleep(60)
            pytest.fail("sibling must not continue")
        finally:
            stopped.set()

    monkeypatch.setattr(tool, "_search_arxiv", fail)
    monkeypatch.setattr(tool, "_search_crossref", waiting)
    with pytest.raises(AgentBudgetExceeded, match="original-budget-stop"):
        tool.search_papers(
            "topic", 2024, 2026, 5, sources=["arxiv", "crossref"], enable_dual_channel=False
        )
    assert stopped.is_set()


def test_worker_budget_abort_prevents_queued_provider_calls(monkeypatch):
    import importlib

    tool = importlib.import_module("app.tools.search_papers")
    monkeypatch.setattr(tool.get_settings(), "search_source_max_workers", 1)
    sent = []

    def fail(*args):
        raise AgentBudgetExceeded("original-budget-stop")

    monkeypatch.setattr(tool, "_search_arxiv", fail)
    monkeypatch.setattr(tool, "_search_crossref", lambda *a: sent.append(True) or [])
    with pytest.raises(AgentBudgetExceeded, match="original-budget-stop"):
        tool.search_papers(
            "topic", 2024, 2026, 5, sources=["arxiv", "crossref"], enable_dual_channel=False
        )
    assert sent == []


@pytest.mark.parametrize("provider", ["crossref", "openalex", "semantic_scholar", "arxiv"])
def test_server_retry_after_is_not_shortened_by_local_cooldown_ceiling(provider, monkeypatch):
    module = provider_module(provider)
    monkeypatch.setattr(module.settings, provider + "_max_cooldown_seconds", 5.0)
    module._enter_cooldown(100, retry_after=100)
    deadline = (
        getattr(module, "_S2_COOLDOWN_UNTIL", None)
        if provider == "semantic_scholar"
        else module._COOLDOWN_UNTIL
    )
    assert 99 < deadline - time.monotonic() <= 100


def test_arxiv_successful_retry_does_not_leave_failure_diagnostic(monkeypatch, retry_clock):
    from app.clients import arxiv_client as arxiv
    from app.tools.search_papers import _invoke_client

    monkeypatch.setattr(arxiv, "_reserve_request_slot", lambda: None)
    responses = [Response(429, headers={"Retry-After": "2"}), Response(200)]
    monkeypatch.setattr(arxiv.requests, "get", lambda *a, **k: responses.pop(0))

    def client(*args):
        arxiv._arxiv_get("https://example.test/", timeout=5)
        return []

    papers, events = _invoke_client(client, "topic", 2024, 2026, 5, "relevance")
    assert papers == []
    assert events == []
    assert responses == []


def test_local_token_timeout_is_reported_as_rate_limit_not_network_timeout():
    from app.tools.search_papers import _classify_search_exception

    assert _classify_search_exception(RateLimitAcquireTimeout("token wait expired")) == (
        "rate_limited",
        "RATE_LIMIT_WAIT_TIMEOUT",
    )


@pytest.mark.parametrize("status", [401, 403, 429, 503])
def test_arxiv_failed_phrase_does_not_retry_as_loose_query(status, monkeypatch):
    from app.clients import arxiv_client as arxiv

    monkeypatch.setattr(arxiv, "_reserve_request_slot", lambda: None)
    sent = []
    monkeypatch.setattr(
        arxiv.requests, "get",
        lambda *a, **k: sent.append(1) or Response(status, headers={"Retry-After": "120"}),
    )
    diagnostics = []
    assert search_papers(
        "action recognition", 2024, 2026, 5, sources=["arxiv"],
        diagnostics=diagnostics, enable_dual_channel=False,
    ) == []
    assert len(sent) == 1
    expected = "rate_limited" if status == 429 else (
        "authentication_failed" if status in {401, 403} else "api_failed"
    )
    assert diagnostics[0].outcome == expected


def test_arxiv_token_timeout_preserves_rate_limit_diagnostic(monkeypatch):
    from app.clients import arxiv_client as arxiv
    from types import SimpleNamespace

    acquired, sent = [], []
    monkeypatch.setattr(
        "app.core.rate_limiter.get_rate_limiter",
        lambda domain: SimpleNamespace(acquire=lambda *a, **k: acquired.append(1) or False),
    )
    monkeypatch.setattr(arxiv.requests, "get", lambda *a, **k: sent.append(1))
    diagnostics = []
    assert search_papers(
        "action recognition", 2024, 2026, 5, sources=["arxiv"],
        diagnostics=diagnostics, enable_dual_channel=False,
    ) == []
    assert sent == []
    assert diagnostics[0].outcome == "rate_limited"
    assert diagnostics[0].error_code == "RATE_LIMIT_WAIT_TIMEOUT"
    assert len(acquired) == 1


def test_arxiv_successful_empty_phrase_still_uses_loose_query(monkeypatch):
    from app.clients import arxiv_client as arxiv

    monkeypatch.setattr(arxiv, "_reserve_request_slot", lambda: None)
    monkeypatch.setattr(get_rate_limiter("arxiv"), "acquire", lambda *a, **k: True)
    sent = []
    monkeypatch.setattr(arxiv.requests, "get", lambda url, **k: sent.append(url) or Response(200))
    assert arxiv.search_arxiv("action recognition", 2024, 2026) == []
    assert len(sent) == 2


@pytest.mark.parametrize("error_type,attempts,outcome", [
    (requests.Timeout, 3, "timeout"),
    (requests.ConnectionError, 1, "api_failed"),
])
def test_arxiv_network_failure_does_not_restart_retry_budget(
    monkeypatch, retry_clock, error_type, attempts, outcome,
):
    from app.clients import arxiv_client as arxiv

    monkeypatch.setattr(arxiv, "_reserve_request_slot", lambda: None)
    monkeypatch.setattr(get_rate_limiter("arxiv"), "acquire", lambda *a, **k: True)
    sent = []

    def fail(*args, **kwargs):
        sent.append(1)
        raise error_type("network failure")

    monkeypatch.setattr(arxiv.requests, "get", fail)
    diagnostics = []
    assert search_papers(
        "action recognition", 2024, 2026, 5, sources=["arxiv"],
        diagnostics=diagnostics, enable_dual_channel=False,
    ) == []
    assert len(sent) == attempts
    assert diagnostics[0].outcome == outcome
