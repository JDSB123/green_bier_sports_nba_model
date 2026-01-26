import time

import pytest

from src.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitState,
)


def test_circuit_breaker_opens_after_threshold_failures(monkeypatch):
    now = 1000.0

    def fake_time():
        return now

    monkeypatch.setattr(time, "time", fake_time)

    breaker = CircuitBreaker(
        "test",
        CircuitBreakerConfig(
            failure_threshold=2, success_threshold=1, timeout=10.0, expected_exception=ValueError
        ),
    )

    def bad():
        raise ValueError("nope")

    with pytest.raises(ValueError):
        breaker.call(bad)
    assert breaker.stats.state == CircuitState.CLOSED

    with pytest.raises(ValueError):
        breaker.call(bad)
    assert breaker.stats.state == CircuitState.OPEN

    with pytest.raises(CircuitBreakerError):
        breaker.call(lambda: 1)


def test_circuit_breaker_transitions_to_half_open_after_timeout(monkeypatch):
    now = 2000.0

    def fake_time():
        return now

    monkeypatch.setattr(time, "time", fake_time)

    breaker = CircuitBreaker(
        "test",
        CircuitBreakerConfig(
            failure_threshold=1, success_threshold=2, timeout=5.0, expected_exception=RuntimeError
        ),
    )

    def bad():
        raise RuntimeError("fail")

    with pytest.raises(RuntimeError):
        breaker.call(bad)
    assert breaker.stats.state == CircuitState.OPEN

    now += 4.9
    with pytest.raises(CircuitBreakerError):
        breaker.call(lambda: 123)

    now += 0.2
    assert breaker.call(lambda: 123) == 123
    assert breaker.stats.state == CircuitState.HALF_OPEN

    assert breaker.call(lambda: 456) == 456
    assert breaker.stats.state == CircuitState.CLOSED


def test_circuit_breaker_unexpected_exception_does_not_count(monkeypatch):
    now = 3000.0

    def fake_time():
        return now

    monkeypatch.setattr(time, "time", fake_time)

    breaker = CircuitBreaker(
        "test",
        CircuitBreakerConfig(
            failure_threshold=1, success_threshold=1, timeout=1.0, expected_exception=ValueError
        ),
    )

    def boom():
        raise RuntimeError("unexpected")

    with pytest.raises(RuntimeError):
        breaker.call(boom)

    assert breaker.stats.total_calls == 1
    assert breaker.stats.total_failures == 0
    assert breaker.stats.state == CircuitState.CLOSED
