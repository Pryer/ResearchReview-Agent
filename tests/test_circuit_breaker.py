"""Tests for CircuitBreaker."""

import pytest
import time
from app.core.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    SourceCircuitBrokenError,
    circuit_breaker,
    get_circuit_breaker,
)

def test_circuit_breaker_transitions_to_open_after_failures():
    cb = CircuitBreaker("test_api", failure_threshold=3, recovery_timeout=0.1)
    assert cb.allow_request() is True
    assert cb.state == CircuitState.CLOSED

    cb.record_failure(ValueError("err1"))
    cb.record_failure(ValueError("err2"))
    assert cb.state == CircuitState.CLOSED
    assert cb.allow_request() is True

    # 3rd failure triggers OPEN
    cb.record_failure(ValueError("err3"))
    assert cb.state == CircuitState.OPEN
    assert cb.allow_request() is False

    # Wait for recovery timeout
    time.sleep(0.12)
    assert cb.allow_request() is True
    assert cb.state == CircuitState.HALF_OPEN

    # Success in half-open resets to CLOSED
    cb.record_success()
    assert cb.state == CircuitState.CLOSED
    assert cb.allow_request() is True

def test_circuit_breaker_decorator_fallback():
    cb = get_circuit_breaker("test_dec", failure_threshold=2, recovery_timeout=1.0)
    
    @circuit_breaker("test_dec", fallback_return=lambda: [])
    def failing_api():
        raise RuntimeError("API dead")

    with pytest.raises(RuntimeError):
        failing_api()

    with pytest.raises(RuntimeError):
        failing_api()

    # Now circuit is open (2 failures reached threshold), decorator returns fallback
    assert failing_api() == []
