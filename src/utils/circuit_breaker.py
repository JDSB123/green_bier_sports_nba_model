"""
Circuit breaker pattern for external API calls.

Prevents cascading failures by stopping requests when external APIs are failing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Any, Callable, Optional

from src.utils.logging import get_logger

logger = get_logger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5  # Open circuit after N failures
    success_threshold: int = 2  # Close circuit after N successes (half-open)
    timeout: float = 60.0  # Seconds to wait before trying half-open
    expected_exception: type[Exception] = Exception


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker."""

    failures: int = 0
    successes: int = 0
    last_failure_time: Optional[float] = None
    state: CircuitState = CircuitState.CLOSED
    total_calls: int = 0
    total_failures: int = 0
    total_successes: int = 0


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""

    pass


class CircuitBreaker:
    """
    Circuit breaker implementation for external API calls.

    Prevents cascading failures by:
    1. Opening circuit after threshold failures
    2. Rejecting requests when open
    3. Testing recovery in half-open state
    4. Closing circuit after successful recovery
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.stats = CircuitBreakerStats()
        self._lock = Lock()

    def _should_attempt_request(self) -> bool:
        """Check if request should be attempted based on circuit state."""
        with self._lock:
            if self.stats.state == CircuitState.CLOSED:
                return True

            if self.stats.state == CircuitState.OPEN:
                # Check if timeout has passed
                if self.stats.last_failure_time:
                    elapsed = time.time() - self.stats.last_failure_time
                    if elapsed >= self.config.timeout:
                        logger.info(f"Circuit breaker '{self.name}' transitioning to HALF_OPEN")
                        self.stats.state = CircuitState.HALF_OPEN
                        self.stats.successes = 0
                        return True
                return False

            # HALF_OPEN - allow limited requests
            return True

    def _record_success(self):
        """Record successful request."""
        with self._lock:
            self.stats.successes += 1
            self.stats.total_successes += 1
            self.stats.failures = 0

            if self.stats.state == CircuitState.HALF_OPEN:
                if self.stats.successes >= self.config.success_threshold:
                    logger.info(
                        f"Circuit breaker '{self.name}' recovered - transitioning to CLOSED"
                    )
                    self.stats.state = CircuitState.CLOSED
                    self.stats.successes = 0

    def _record_failure(self):
        """Record failed request."""
        with self._lock:
            self.stats.failures += 1
            self.stats.total_failures += 1
            self.stats.last_failure_time = time.time()
            self.stats.successes = 0

            if self.stats.failures >= self.config.failure_threshold:
                if self.stats.state != CircuitState.OPEN:
                    logger.warning(
                        f"Circuit breaker '{self.name}' opened after {self.stats.failures} failures"
                    )
                self.stats.state = CircuitState.OPEN

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is open
        """
        if not self._should_attempt_request():
            raise CircuitBreakerError(f"Circuit breaker '{self.name}' is OPEN - request rejected")

        self.stats.total_calls += 1

        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except self.config.expected_exception as e:
            self._record_failure()
            raise
        except Exception as e:
            # Unexpected exception - don't count as circuit breaker failure
            logger.error(f"Unexpected exception in circuit breaker '{self.name}': {e}")
            raise

    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute async function with circuit breaker protection.

        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is open
        """
        if not self._should_attempt_request():
            raise CircuitBreakerError(f"Circuit breaker '{self.name}' is OPEN - request rejected")

        self.stats.total_calls += 1

        try:
            result = await func(*args, **kwargs)
            self._record_success()
            return result
        except self.config.expected_exception as e:
            self._record_failure()
            raise
        except Exception as e:
            # Unexpected exception - don't count as circuit breaker failure
            logger.error(f"Unexpected exception in circuit breaker '{self.name}': {e}")
            raise

    def get_stats(self) -> dict:
        """Get circuit breaker statistics."""
        with self._lock:
            return {
                "name": self.name,
                "state": self.stats.state.value,
                "failures": self.stats.failures,
                "successes": self.stats.successes,
                "total_calls": self.stats.total_calls,
                "total_failures": self.stats.total_failures,
                "total_successes": self.stats.total_successes,
                "last_failure_time": self.stats.last_failure_time,
            }


# Global circuit breakers for external APIs
_odds_api_breaker = CircuitBreaker(
    "the_odds_api",
    CircuitBreakerConfig(
        failure_threshold=5,
        success_threshold=2,
        timeout=60.0,
    ),
)

_api_basketball_breaker = CircuitBreaker(
    "api_basketball",
    CircuitBreakerConfig(
        failure_threshold=5,
        success_threshold=2,
        timeout=60.0,
    ),
)


def get_odds_api_breaker() -> CircuitBreaker:
    """Get circuit breaker for The Odds API."""
    return _odds_api_breaker


def get_api_basketball_breaker() -> CircuitBreaker:
    """Get circuit breaker for API-Basketball."""
    return _api_basketball_breaker
