"""Shared, bounded retry primitives for external retrieval providers.

A retry is never allowed to bypass execution cancellation or the active
task deadline. Rate-limit (HTTP 429) is distinct from a provider outage.
"""

from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import math
import time
from typing import Mapping


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
    from app.agent.execution_budget import active_budget, check_execution

    check_execution()
    active = active_budget()
    remaining = (
        active["deadline"] - time.monotonic()
        if active is not None else float(configured_seconds)
    )
    return max(0.001, min(float(configured_seconds), remaining))


def interruptible_sleep(seconds: float, *, chunk: float = 0.25) -> None:
    """Recheck cancellation, ownership and deadline while waiting to retry.

    Fixed-step decrement also makes tests with a mocked sleep deterministic.
    """
    from app.agent.execution_budget import check_execution

    remaining = max(0.0, float(seconds))
    check_execution()
    while remaining > 0:
        check_execution()
        step = min(remaining, chunk)
        time.sleep(step)
        remaining -= step
    check_execution()
