"""Shared, bounded retry primitives for external retrieval providers.

A retry is never allowed to bypass execution cancellation or the active
task deadline. Rate-limit (HTTP 429) is distinct from a provider outage.
"""

from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import math
import time
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Mapping


_RETRIEVAL_CANCEL = ContextVar("retrieval_cancel", default=None)


def check_retrieval_execution() -> None:
    from app.agent.execution_budget import check_execution

    check_execution()
    callback = _RETRIEVAL_CANCEL.get()
    if callback is not None:
        callback()


@contextmanager
def retrieval_cancellation(should_cancel=None):
    """Stop sibling retry/wait work without cancelling the parent research task."""
    stopped = threading.Event()
    failure = []
    lock = threading.Lock()
    parent = _RETRIEVAL_CANCEL.get()

    def stop(exc=None):
        with lock:
            if exc is not None and not failure:
                failure.append(exc)
            stopped.set()

    def check():
        if parent:
            parent()
        if stopped.is_set():
            if failure:
                raise failure[0]
            raise InterruptedError("论文检索已取消")
        if should_cancel and should_cancel():
            raise InterruptedError("论文检索已取消")

    token = _RETRIEVAL_CANCEL.set(check)
    try:
        yield stop
    finally:
        stop()
        _RETRIEVAL_CANCEL.reset(token)


def retry_after_seconds(headers: Mapping[str, str]) -> float:
    """Accept Retry-After delay-seconds or HTTP-date; invalid values mean unknown."""
    raw = (headers.get("Retry-After") or "").strip()
    if not raw:
        return 0.0
    try:
        value = float(raw)
        return max(0.0, value) if math.isfinite(value) else 0.0
    except (TypeError, ValueError):
        try:
            date = parsedate_to_datetime(raw)
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            return max(0.0, (date - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return 0.0


def bounded_request_timeout(configured_seconds: float) -> float:
    """Bound each HTTP attempt by the active execution's remaining deadline."""
    from app.agent.execution_budget import active_budget

    check_retrieval_execution()
    active = active_budget()
    remaining = (
        active["deadline"] - time.monotonic() if active is not None else float(configured_seconds)
    )
    return max(0.001, min(float(configured_seconds), remaining))


def interruptible_sleep(seconds: float, *, chunk: float = 0.25) -> None:
    """Recheck cancellation, ownership and deadline while waiting to retry.

    Fixed-step decrement also makes tests with a mocked sleep deterministic.
    """
    if not math.isfinite(seconds) or not math.isfinite(chunk) or chunk <= 0:
        raise ValueError("retry wait and chunk must be finite; chunk must be positive")
    remaining = max(0.0, float(seconds))
    check_retrieval_execution()
    while remaining > 0:
        check_retrieval_execution()
        step = min(remaining, chunk)
        time.sleep(step)
        remaining -= step
    check_retrieval_execution()
