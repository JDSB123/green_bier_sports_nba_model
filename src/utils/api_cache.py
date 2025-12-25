"""
API Response Caching Layer - v6.5 STRICT MODE.

v6.5 STRICT MODE: FILE CACHING DISABLED
Only session-memory caching for within-request deduplication.
All file caching operations are NO-OPs in STRICT MODE.

This ensures:
- Every request gets fresh data from APIs
- No stale data persists between requests
- ESPN standings are always current
- Team records are always accurate

Usage:
    from src.utils.api_cache import api_cache

    # In STRICT MODE, this always fetches fresh data
    data = await api_cache.get_or_fetch(
        key="the_odds_events_2024-01-15",
        fetch_fn=lambda: the_odds.fetch_events(),
        ttl_hours=2,  # Ignored in STRICT MODE
    )

    # Invalidate cache (clears memory cache only)
    api_cache.invalidate("the_odds_events_2024-01-15")
"""
from __future__ import annotations

import json
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Awaitable, TypeVar, Optional
from dataclasses import dataclass, asdict

from src.config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


@dataclass
class CacheEntry:
    """Metadata for a cached API response."""
    key: str
    data: Any
    created_at: float  # Unix timestamp
    ttl_seconds: int
    source: str  # API name (the_odds, api_basketball, etc.)
    endpoint: str  # Specific endpoint


class APICache:
    """
    API response cache - v6.5 STRICT MODE.

    STRICT MODE: File caching is DISABLED.
    Only session-memory caching for within-request deduplication.
    """

    # TTLs are ignored in STRICT MODE - always fetch fresh
    TTL_STATIC = 0  # DISABLED
    TTL_DAILY = 0   # DISABLED
    TTL_FREQUENT = 0  # DISABLED
    TTL_LIVE = 0  # DISABLED

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize the cache - STRICT MODE (memory only).

        Args:
            cache_dir: Ignored in STRICT MODE - no file caching
        """
        # STRICT MODE: No file cache directory needed
        self.cache_dir = cache_dir or Path(settings.data_processed_dir) / "cache" / "api"
        self._memory_cache: dict[str, CacheEntry] = {}
        self._strict_mode = True  # v6.5 STRICT MODE enabled
        logger.info("v6.5 STRICT MODE: File caching DISABLED - memory cache only")

    def _get_cache_path(self, key: str) -> Path:
        """Generate cache file path from key."""
        # Use hash for long keys
        safe_key = hashlib.md5(key.encode()).hexdigest()[:16] + "_" + key.replace("/", "_").replace(":", "_")[:50]
        return self.cache_dir / f"{safe_key}.json"

    def _is_valid(self, entry: CacheEntry) -> bool:
        """Check if cache entry is still valid."""
        age_seconds = time.time() - entry.created_at
        return age_seconds < entry.ttl_seconds

    def get(self, key: str) -> Optional[Any]:
        """Get cached data if valid.

        STRICT MODE: Only checks memory cache, never file cache.

        Args:
            key: Cache key

        Returns:
            Cached data or None if not found/expired
        """
        # STRICT MODE: Memory cache only (within-request deduplication)
        if key in self._memory_cache:
            entry = self._memory_cache[key]
            if self._is_valid(entry):
                logger.debug(f"Cache HIT (memory): {key}")
                return entry.data
            else:
                del self._memory_cache[key]

        # STRICT MODE: No file cache lookup
        return None

    def set(
        self,
        key: str,
        data: Any,
        ttl_hours: float,
        source: str = "unknown",
        endpoint: str = "",
    ) -> None:
        """Store data in cache.

        STRICT MODE: Only stores in memory cache, not file.

        Args:
            key: Cache key
            data: Data to cache
            ttl_hours: Time-to-live in hours (used for memory cache within request)
            source: API source name
            endpoint: API endpoint
        """
        entry = CacheEntry(
            key=key,
            data=data,
            created_at=time.time(),
            ttl_seconds=int(ttl_hours * 3600),
            source=source,
            endpoint=endpoint,
        )

        # STRICT MODE: Memory cache only (within-request deduplication)
        self._memory_cache[key] = entry
        logger.debug(f"Cache SET (memory only): {key}")

    async def get_or_fetch(
        self,
        key: str,
        fetch_fn: Callable[[], Awaitable[T]],
        ttl_hours: float,
        source: str = "unknown",
        endpoint: str = "",
        force_refresh: bool = False,
    ) -> T:
        """Get from cache or fetch and cache.

        Args:
            key: Cache key
            fetch_fn: Async function to fetch data if not cached
            ttl_hours: TTL in hours
            source: API source name
            endpoint: API endpoint
            force_refresh: If True, bypass cache and fetch fresh

        Returns:
            Cached or freshly fetched data
        """
        if not force_refresh:
            cached = self.get(key)
            if cached is not None:
                return cached

        # Fetch fresh data
        logger.info(f"Cache MISS: {key} - fetching fresh data")
        data = await fetch_fn()

        # Store in cache
        self.set(key, data, ttl_hours, source, endpoint)

        return data

    def invalidate(self, key: str) -> bool:
        """Invalidate a cache entry.

        Args:
            key: Cache key to invalidate

        Returns:
            True if entry was found and removed
        """
        removed = False

        # Remove from memory
        if key in self._memory_cache:
            del self._memory_cache[key]
            removed = True

        # Remove from file
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            cache_path.unlink()
            removed = True

        if removed:
            logger.debug(f"Cache INVALIDATED: {key}")

        return removed

    def invalidate_source(self, source: str) -> int:
        """Invalidate all cache entries from a source.

        Args:
            source: Source name (e.g., 'the_odds', 'api_basketball')

        Returns:
            Number of entries invalidated
        """
        count = 0

        # Check all cache files
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("source") == source:
                    cache_file.unlink()
                    count += 1
            except Exception:
                pass

        # Clear memory cache entries from source
        keys_to_remove = [
            k for k, v in self._memory_cache.items()
            if v.source == source
        ]
        for key in keys_to_remove:
            del self._memory_cache[key]
            count += 1

        logger.info(f"Invalidated {count} cache entries for source: {source}")
        return count

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        STRICT MODE: Only returns memory cache stats.

        Returns:
            Dict with cache stats
        """
        memory_count = len(self._memory_cache)

        # Count by source in memory
        by_source: dict[str, int] = {}
        for entry in self._memory_cache.values():
            source = entry.source
            by_source[source] = by_source.get(source, 0) + 1

        return {
            "mode": "STRICT",
            "file_caching": "DISABLED",
            "memory_count": memory_count,
            "by_source": by_source,
        }

    def clear_all(self) -> int:
        """Clear all cache entries.

        STRICT MODE: Only clears memory cache.

        Returns:
            Number of entries cleared
        """
        count = len(self._memory_cache)
        self._memory_cache.clear()

        logger.info(f"v6.5 STRICT MODE: Cleared {count} memory cache entries")
        return count


# Global cache instance
api_cache = APICache()


# Convenience functions for common cache operations
async def cached_fetch(
    key: str,
    fetch_fn: Callable[[], Awaitable[T]],
    ttl_hours: float = 2.0,
    source: str = "unknown",
) -> T:
    """Convenience wrapper for cached API fetch.

    Args:
        key: Cache key
        fetch_fn: Async function to fetch data
        ttl_hours: TTL in hours (default 2)
        source: API source name

    Returns:
        Cached or fetched data
    """
    return await api_cache.get_or_fetch(key, fetch_fn, ttl_hours, source)
