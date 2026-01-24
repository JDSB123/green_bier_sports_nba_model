from __future__ import annotations

import asyncio

import pytest


@pytest.mark.asyncio
async def test_api_cache_set_get_and_expire(monkeypatch):
    from src.utils.api_cache import APICache

    cache = APICache()

    # Freeze time for deterministic TTL tests.
    t = {"now": 1_000_000.0}

    def fake_time() -> float:
        return t["now"]

    monkeypatch.setattr("src.utils.api_cache.time.time", fake_time)

    cache.set("k1", {"v": 1}, ttl_hours=1.0, source="test", endpoint="/x")
    assert cache.get("k1") == {"v": 1}

    # Expire it.
    t["now"] += 3600.1
    assert cache.get("k1") is None


@pytest.mark.asyncio
async def test_api_cache_get_or_fetch_uses_cache(monkeypatch):
    from src.utils.api_cache import APICache

    cache = APICache()
    calls = {"n": 0}

    async def fetch_fn():
        calls["n"] += 1
        await asyncio.sleep(0)
        return {"data": calls["n"]}

    v1 = await cache.get_or_fetch("k2", fetch_fn, ttl_hours=2.0, source="test")
    v2 = await cache.get_or_fetch("k2", fetch_fn, ttl_hours=2.0, source="test")

    assert v1 == {"data": 1}
    assert v2 == {"data": 1}
    assert calls["n"] == 1

    v3 = await cache.get_or_fetch(
        "k2", fetch_fn, ttl_hours=2.0, source="test", force_refresh=True
    )
    assert v3 == {"data": 2}
    assert calls["n"] == 2


def test_api_cache_invalidate_and_stats():
    from src.utils.api_cache import APICache

    cache = APICache()
    cache.set("k3", 123, ttl_hours=1.0, source="the_odds", endpoint="/events")

    stats = cache.get_stats()
    assert stats["mode"] == "STRICT"
    assert stats["memory_count"] == 1
    assert stats["by_source"].get("the_odds") == 1

    assert cache.invalidate("k3") is True
    assert cache.get("k3") is None

    cache.set("k4", 1, ttl_hours=1.0, source="api_basketball")
    cache.set("k5", 2, ttl_hours=1.0, source="api_basketball")
    assert cache.invalidate_source("api_basketball") >= 2

    cleared = cache.clear_all()
    assert cleared >= 0
