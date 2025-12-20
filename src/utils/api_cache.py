"""
API Response Caching Layer.

Provides file-based caching for API responses to minimize redundant API calls.
Each cache entry is stored as a JSON file with metadata (timestamp, TTL).

Cache TTL Guidelines:
- Static data (teams, rosters): 7 days
- Semi-static (standings, statistics): 24 hours
- Dynamic (odds, injuries): 2 hours
- Real-time (live odds): 15 minutes

Usage:
    from src.utils.api_cache import api_cache

    # Cached fetch
    data = await api_cache.get_or_fetch(
        key="the_odds_events_2024-01-15",
        fetch_fn=lambda: the_odds.fetch_events(),
        ttl_hours=2,
    )

    # Invalidate cache
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
    """File-based API response cache."""

    # Default TTLs by data type (in hours)
    TTL_STATIC = 7 * 24  # 7 days (teams, rosters)
    TTL_DAILY = 24  # 24 hours (standings, statistics, schedule)
    TTL_FREQUENT = 2  # 2 hours (injuries)
    TTL_LIVE = 0.25  # 15 minutes (live odds)

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize the cache.

        Args:
            cache_dir: Directory to store cache files. Defaults to data/cache/api/
        """
        self.cache_dir = cache_dir or Path(settings.data_processed_dir) / "cache" / "api"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: dict[str, CacheEntry] = {}

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

        Args:
            key: Cache key

        Returns:
            Cached data or None if not found/expired
        """
        # Check memory cache first
        if key in self._memory_cache:
            entry = self._memory_cache[key]
            if self._is_valid(entry):
                logger.debug(f"Cache HIT (memory): {key}")
                return entry.data
            else:
                del self._memory_cache[key]

        # Check file cache
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                entry = CacheEntry(**data)

                if self._is_valid(entry):
                    # Promote to memory cache
                    self._memory_cache[key] = entry
                    logger.debug(f"Cache HIT (file): {key}")
                    return entry.data
                else:
                    # Expired, delete file
                    cache_path.unlink()
                    logger.debug(f"Cache EXPIRED: {key}")
            except Exception as e:
                logger.warning(f"Failed to read cache {key}: {e}")

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

        Args:
            key: Cache key
            data: Data to cache
            ttl_hours: Time-to-live in hours
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

        # Store in memory
        self._memory_cache[key] = entry

        # Store in file
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(asdict(entry), f, ensure_ascii=False)
            logger.debug(f"Cache SET: {key} (TTL: {ttl_hours}h)")
        except Exception as e:
            logger.warning(f"Failed to write cache {key}: {e}")

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

        Returns:
            Dict with cache stats
        """
        file_count = len(list(self.cache_dir.glob("*.json")))
        memory_count = len(self._memory_cache)

        # Calculate total size
        total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.json"))

        # Count by source
        by_source: dict[str, int] = {}
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                source = data.get("source", "unknown")
                by_source[source] = by_source.get(source, 0) + 1
            except Exception:
                pass

        return {
            "file_count": file_count,
            "memory_count": memory_count,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "by_source": by_source,
            "cache_dir": str(self.cache_dir),
        }

    def clear_all(self) -> int:
        """Clear all cache entries.

        Returns:
            Number of entries cleared
        """
        count = 0

        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
            count += 1

        self._memory_cache.clear()

        logger.info(f"Cleared {count} cache entries")
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
