from __future__ import annotations

import asyncio

import pytest


@pytest.mark.asyncio
async def test_rate_limiter_burst_allows_immediate_acquire():
    from src.utils.rate_limiter import RateLimitConfig, TokenBucketRateLimiter

    limiter = TokenBucketRateLimiter(
        "test",
        RateLimitConfig(requests_per_second=100.0, burst_size=2, max_queue_size=10, timeout=1.0),
    )

    assert await limiter.acquire() == 0.0
    assert await limiter.acquire() == 0.0


@pytest.mark.asyncio
async def test_rate_limiter_queue_is_processed():
    from src.utils.rate_limiter import RateLimitConfig, TokenBucketRateLimiter

    limiter = TokenBucketRateLimiter(
        "test",
        RateLimitConfig(requests_per_second=50.0, burst_size=1, max_queue_size=10, timeout=1.0),
    )

    # Consume the single burst token.
    assert await limiter.acquire() == 0.0

    worker = asyncio.create_task(limiter._process_queue())
    try:
        wait = await limiter.acquire()
        assert wait >= 0.0
        assert limiter.get_stats()["stats"]["requests_queued"] >= 1
    finally:
        worker.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker


@pytest.mark.asyncio
async def test_rate_limiter_queue_full_raises():
    from src.utils.rate_limiter import RateLimitConfig, RateLimitExceeded, TokenBucketRateLimiter

    # burst_size=0 => no tokens ever available (tokens capped at 0), so first acquire will queue.
    limiter = TokenBucketRateLimiter(
        "test",
        RateLimitConfig(requests_per_second=1.0, burst_size=0, max_queue_size=1, timeout=0.2),
    )

    first = asyncio.create_task(limiter.acquire())
    await asyncio.sleep(0)  # allow queue append

    with pytest.raises(RateLimitExceeded):
        await limiter.acquire()

    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first


@pytest.mark.asyncio
async def test_rate_limiter_execute_stats_success_and_failure():
    from src.utils.rate_limiter import RateLimitConfig, TokenBucketRateLimiter

    limiter = TokenBucketRateLimiter(
        "test",
        RateLimitConfig(requests_per_second=100.0, burst_size=5, max_queue_size=10, timeout=1.0),
    )

    async def ok(x: int) -> int:
        await asyncio.sleep(0)
        return x + 1

    async def boom() -> None:
        await asyncio.sleep(0)
        raise RuntimeError("fail")

    limiter.reset_stats()
    assert await limiter.execute(ok, 1) == 2

    with pytest.raises(RuntimeError):
        await limiter.execute(boom)

    stats = limiter.get_stats()["stats"]
    assert stats["total_requests"] == 2
    assert stats["requests_succeeded"] == 1
    assert stats["requests_failed"] == 1
