"""
API Rate Limiter with Request Queue.

Implements token bucket algorithm for rate limiting with:
- Per-API rate limits
- Request queuing during high traffic
- Automatic backoff on rate limit errors
- Statistics tracking
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, Optional, TypeVar
from dataclasses import dataclass, field
from threading import Lock
from collections import deque

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_second: float = 1.0  # Max RPS
    burst_size: int = 5  # Max burst above steady rate
    max_queue_size: int = 100  # Max pending requests
    timeout: float = 30.0  # Max wait time in queue


@dataclass
class RateLimitStats:
    """Statistics for rate limiter."""
    total_requests: int = 0
    requests_queued: int = 0
    requests_dropped: int = 0
    requests_succeeded: int = 0
    requests_failed: int = 0
    total_wait_time: float = 0.0
    max_wait_time: float = 0.0
    last_request_time: Optional[float] = None


class RateLimitExceeded(Exception):
    """Raised when rate limit queue is full."""
    pass


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter with request queuing.

    Allows smooth traffic shaping with burst handling.
    """

    def __init__(self, name: str, config: Optional[RateLimitConfig] = None):
        """
        Initialize rate limiter.

        Args:
            name: Name of the rate limiter (for logging)
            config: Rate limit configuration
        """
        self.name = name
        self.config = config or RateLimitConfig()

        self._lock = Lock()
        self._tokens = float(self.config.burst_size)
        self._last_refill = time.monotonic()
        self._queue: deque = deque()
        self._queue_event = asyncio.Event()

        self.stats = RateLimitStats()

    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        new_tokens = elapsed * self.config.requests_per_second
        self._tokens = min(self._tokens + new_tokens, self.config.burst_size)
        self._last_refill = now

    def _try_acquire(self) -> bool:
        """Try to acquire a token without waiting."""
        with self._lock:
            self._refill_tokens()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False

    async def acquire(self) -> float:
        """
        Acquire a token, waiting if necessary.

        Returns:
            Wait time in seconds

        Raises:
            RateLimitExceeded: If queue is full
            asyncio.TimeoutError: If wait time exceeds timeout
        """
        start_time = time.monotonic()

        # Try immediate acquisition
        if self._try_acquire():
            return 0.0

        # Check queue capacity
        with self._lock:
            if len(self._queue) >= self.config.max_queue_size:
                self.stats.requests_dropped += 1
                raise RateLimitExceeded(
                    f"Rate limiter '{self.name}' queue full ({self.config.max_queue_size} pending)"
                )

            # Add to queue
            future = asyncio.get_event_loop().create_future()
            self._queue.append(future)
            self.stats.requests_queued += 1

        try:
            # Wait for token
            await asyncio.wait_for(future, timeout=self.config.timeout)
        except asyncio.TimeoutError:
            self.stats.requests_dropped += 1
            raise

        wait_time = time.monotonic() - start_time
        self.stats.total_wait_time += wait_time
        self.stats.max_wait_time = max(self.stats.max_wait_time, wait_time)

        return wait_time

    async def _process_queue(self) -> None:
        """Background task to process queued requests."""
        while True:
            if not self._queue:
                await asyncio.sleep(0.1)
                continue

            # Wait for token
            while not self._try_acquire():
                await asyncio.sleep(1.0 / self.config.requests_per_second)

            # Release next queued request
            with self._lock:
                if self._queue:
                    future = self._queue.popleft()
                    if not future.done():
                        future.set_result(True)

    async def execute(
        self,
        func: Callable[..., Awaitable[T]],
        *args,
        **kwargs,
    ) -> T:
        """
        Execute a function with rate limiting.

        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result
        """
        self.stats.total_requests += 1
        self.stats.last_request_time = time.time()

        wait_time = await self.acquire()

        if wait_time > 0:
            logger.debug(f"Rate limiter '{self.name}': waited {wait_time:.2f}s")

        try:
            result = await func(*args, **kwargs)
            self.stats.requests_succeeded += 1
            return result
        except Exception as e:
            self.stats.requests_failed += 1
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        with self._lock:
            avg_wait = 0.0
            if self.stats.requests_queued > 0:
                avg_wait = self.stats.total_wait_time / self.stats.requests_queued

            return {
                "name": self.name,
                "config": {
                    "requests_per_second": self.config.requests_per_second,
                    "burst_size": self.config.burst_size,
                    "max_queue_size": self.config.max_queue_size,
                },
                "stats": {
                    "total_requests": self.stats.total_requests,
                    "requests_queued": self.stats.requests_queued,
                    "requests_dropped": self.stats.requests_dropped,
                    "requests_succeeded": self.stats.requests_succeeded,
                    "requests_failed": self.stats.requests_failed,
                    "avg_wait_time": round(avg_wait, 4),
                    "max_wait_time": round(self.stats.max_wait_time, 4),
                },
                "current_queue_size": len(self._queue),
                "current_tokens": round(self._tokens, 2),
            }

    def reset_stats(self) -> None:
        """Reset statistics."""
        with self._lock:
            self.stats = RateLimitStats()


# Pre-configured rate limiters for external APIs
# The Odds API: 500 requests per month = ~0.01 RPS, but allow bursts for slates
_the_odds_limiter = TokenBucketRateLimiter(
    "the_odds_api",
    RateLimitConfig(
        requests_per_second=2.0,  # 2 RPS sustained
        burst_size=10,  # Allow bursts of 10
        max_queue_size=50,
        timeout=30.0,
    )
)

# API-Basketball: 100 requests per day = ~0.001 RPS, allow small bursts
_api_basketball_limiter = TokenBucketRateLimiter(
    "api_basketball",
    RateLimitConfig(
        requests_per_second=1.0,  # 1 RPS sustained
        burst_size=5,  # Allow bursts of 5
        max_queue_size=30,
        timeout=30.0,
    )
)


def get_odds_api_limiter() -> TokenBucketRateLimiter:
    """Get rate limiter for The Odds API."""
    return _the_odds_limiter


def get_api_basketball_limiter() -> TokenBucketRateLimiter:
    """Get rate limiter for API-Basketball."""
    return _api_basketball_limiter


async def rate_limited_request(
    limiter: TokenBucketRateLimiter,
    func: Callable[..., Awaitable[T]],
    *args,
    **kwargs,
) -> T:
    """
    Execute an async function with rate limiting.

    Convenience wrapper for rate-limited API calls.

    Args:
        limiter: Rate limiter to use
        func: Async function to execute
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        Function result
    """
    return await limiter.execute(func, *args, **kwargs)
