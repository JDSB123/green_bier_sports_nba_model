"""
GitHub Data Fetcher Module

Fetches data from GitHub repositories (raw.githubusercontent.com) for historical
NBA data, ELO ratings, and other open-source datasets.

This module provides:
- Async fetching from GitHub raw content URLs
- Support for CSV, JSON, and text files
- Predefined sources (FiveThirtyEight, etc.)
- Caching and local storage
- Retry logic with exponential backoff

Usage:
    from src.ingestion.github_data import GitHubDataFetcher, fetch_fivethirtyeight_elo

    # Fetch FiveThirtyEight ELO data
    df = await fetch_fivethirtyeight_elo()

    # Custom fetch
    fetcher = GitHubDataFetcher()
    data = await fetcher.fetch_csv("https://raw.githubusercontent.com/user/repo/branch/file.csv")
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Common GitHub data sources
FIVETHIRTYEIGHT_BASE = "https://raw.githubusercontent.com/fivethirtyeight/data/master"
FIVETHIRTYEIGHT_URLS = {
    "elo_historical": f"{FIVETHIRTYEIGHT_BASE}/nba-elo/nbaallelo.csv",
    "elo_latest": f"{FIVETHIRTYEIGHT_BASE}/nba-forecasts/nba_elo_latest.csv",
    "nba_forecasts": f"{FIVETHIRTYEIGHT_BASE}/nba-forecasts/nba_elo.csv",
}

TIMEOUT = 30.0
MAX_RETRIES = 3


@dataclass
class GitHubFetchResult:
    """Result from a GitHub data fetch operation."""

    url: str
    data: Any  # DataFrame, dict, str, or bytes
    format: str  # 'csv', 'json', 'text', 'bytes'
    cached: bool = False
    error: str | None = None


class GitHubDataFetcher:
    """Fetches data from GitHub repositories via raw.githubusercontent.com."""

    def __init__(self, cache_dir: str | None = None):
        """
        Initialize GitHub data fetcher.

        Args:
            cache_dir: Directory to cache fetched files. Defaults to data/raw/github/
        """
        self.cache_dir = Path(
            cache_dir or os.path.join(settings.data_raw_dir, "github")
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, url: str) -> Path:
        """Generate cache file path from URL."""
        # Convert URL to safe filename
        # e.g., "https://raw.githubusercontent.com/user/repo/branch/file.csv"
        # -> "user_repo_branch_file.csv"
        safe_name = url.replace("https://", "").replace("http://", "")
        safe_name = safe_name.replace("/", "_").replace(":", "_")
        return self.cache_dir / safe_name

    def _is_cached(self, url: str, max_age_hours: int = 24) -> bool:
        """Check if URL is cached and still fresh."""
        cache_path = self._get_cache_path(url)
        if not cache_path.exists():
            return False

        # Check file age
        import time
        age_seconds = time.time() - cache_path.stat().st_mtime
        return age_seconds < (max_age_hours * 3600)

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=8),
    )
    async def _fetch_raw(self, url: str, use_cache: bool = True) -> bytes:
        """
        Fetch raw bytes from GitHub URL.

        Args:
            url: GitHub raw content URL
            use_cache: Whether to use cached version if available

        Returns:
            Raw bytes content
        """
        # Check cache first
        if use_cache:
            cache_path = self._get_cache_path(url)
            if cache_path.exists() and self._is_cached(url):
                logger.info(f"Using cached version of {url}")
                return cache_path.read_bytes()

        # Fetch from GitHub
        logger.info(f"Fetching {url}")
        async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True) as client:
            response = await client.get(
                url,
                headers={
                    "User-Agent": "NBA-Prediction-System/5.0",
                    "Accept": "*/*",
                },
            )
            response.raise_for_status()
            content = response.content

            # Cache the result
            if use_cache:
                cache_path.write_bytes(content)
                logger.debug(f"Cached to {cache_path}")

            return content

    async def fetch_csv(
        self,
        url: str,
        use_cache: bool = True,
        **pandas_kwargs: Any,
    ) -> pd.DataFrame:
        """
        Fetch CSV file from GitHub and return as DataFrame.

        Args:
            url: GitHub raw CSV URL
            use_cache: Whether to use cached version
            **pandas_kwargs: Additional arguments passed to pd.read_csv()

        Returns:
            DataFrame with CSV data

        Raises:
            httpx.HTTPError: If fetch fails
            pd.errors.EmptyDataError: If CSV is empty
        """
        try:
            from io import StringIO
            content = await self._fetch_raw(url, use_cache=use_cache)
            content_str = content.decode("utf-8") if isinstance(content, bytes) else content
            df = pd.read_csv(
                StringIO(content_str),
                **pandas_kwargs,
            )
            logger.info(f"Fetched CSV: {len(df)} rows from {url}")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch CSV from {url}: {e}")
            raise

    async def fetch_json(
        self,
        url: str,
        use_cache: bool = True,
    ) -> dict[str, Any] | list[Any]:
        """
        Fetch JSON file from GitHub and return as dict/list.

        Args:
            url: GitHub raw JSON URL
            use_cache: Whether to use cached version

        Returns:
            Parsed JSON data (dict or list)

        Raises:
            httpx.HTTPError: If fetch fails
            json.JSONDecodeError: If JSON is invalid
        """
        import json

        try:
            content = await self._fetch_raw(url, use_cache=use_cache)
            data = json.loads(content.decode("utf-8"))
            logger.info(f"Fetched JSON from {url}")
            return data
        except Exception as e:
            logger.error(f"Failed to fetch JSON from {url}: {e}")
            raise

    async def fetch_text(
        self,
        url: str,
        use_cache: bool = True,
    ) -> str:
        """
        Fetch text file from GitHub and return as string.

        Args:
            url: GitHub raw text URL
            use_cache: Whether to use cached version

        Returns:
            Text content as string

        Raises:
            httpx.HTTPError: If fetch fails
        """
        try:
            content = await self._fetch_raw(url, use_cache=use_cache)
            text = content.decode("utf-8")
            logger.info(f"Fetched text from {url} ({len(text)} chars)")
            return text
        except Exception as e:
            logger.error(f"Failed to fetch text from {url}: {e}")
            raise

    async def fetch(
        self,
        url: str,
        format: str | None = None,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> GitHubFetchResult:
        """
        Fetch data from GitHub URL with automatic format detection.

        Args:
            url: GitHub raw content URL
            format: Force format ('csv', 'json', 'text', 'bytes'). Auto-detects if None
            use_cache: Whether to use cached version
            **kwargs: Additional arguments for format-specific parsers

        Returns:
            GitHubFetchResult with fetched data
        """
        # Auto-detect format from URL extension
        if format is None:
            if url.endswith(".csv"):
                format = "csv"
            elif url.endswith(".json"):
                format = "json"
            elif url.endswith((".txt", ".md", ".py", ".sql")):
                format = "text"
            else:
                format = "bytes"

        cached = use_cache and self._is_cached(url)

        try:
            if format == "csv":
                data = await self.fetch_csv(url, use_cache=use_cache, **kwargs)
            elif format == "json":
                data = await self.fetch_json(url, use_cache=use_cache)
            elif format == "text":
                data = await self.fetch_text(url, use_cache=use_cache)
            else:
                data = await self._fetch_raw(url, use_cache=use_cache)

            return GitHubFetchResult(
                url=url,
                data=data,
                format=format,
                cached=cached,
            )
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return GitHubFetchResult(
                url=url,
                data=None,
                format=format,
                error=str(e),
            )


# Convenience functions for common data sources


async def fetch_fivethirtyeight_elo(
    dataset: str = "elo_historical",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch FiveThirtyEight NBA ELO data.

    Args:
        dataset: Which dataset to fetch:
            - 'elo_historical': Historical ELO ratings (nbaallelo.csv)
            - 'elo_latest': Latest ELO forecasts (nba_elo_latest.csv)
            - 'nba_forecasts': NBA forecasts (nba_elo.csv)
        use_cache: Whether to use cached version

    Returns:
        DataFrame with ELO data

    Raises:
        KeyError: If dataset name is invalid
        httpx.HTTPError: If fetch fails
    """
    if dataset not in FIVETHIRTYEIGHT_URLS:
        raise KeyError(
            f"Invalid dataset '{dataset}'. Choose from: {list(FIVETHIRTYEIGHT_URLS.keys())}"
        )

    url = FIVETHIRTYEIGHT_URLS[dataset]
    fetcher = GitHubDataFetcher()
    return await fetcher.fetch_csv(url, use_cache=use_cache)


async def fetch_fivethirtyeight_all() -> dict[str, pd.DataFrame]:
    """
    Fetch all available FiveThirtyEight NBA datasets.

    Returns:
        Dictionary mapping dataset names to DataFrames
    """
    fetcher = GitHubDataFetcher()
    results = {}

    for dataset_name, url in FIVETHIRTYEIGHT_URLS.items():
        try:
            df = await fetcher.fetch_csv(url, use_cache=True)
            results[dataset_name] = df
            logger.info(f"Fetched {dataset_name}: {len(df)} rows")
        except Exception as e:
            logger.warning(f"Failed to fetch {dataset_name}: {e}")
            results[dataset_name] = pd.DataFrame()  # Empty DataFrame on failure

    return results


# Example usage for other common GitHub NBA data sources
COMMON_NBA_SOURCES = {
    "fivethirtyeight_elo_historical": FIVETHIRTYEIGHT_URLS["elo_historical"],
    "fivethirtyeight_elo_latest": FIVETHIRTYEIGHT_URLS["elo_latest"],
    # Add more sources as needed:
    # "swar_nba_api": "https://raw.githubusercontent.com/swar/nba_api/master/...",
    # "basketball_reference": "https://raw.githubusercontent.com/...",
}


async def fetch_common_source(source_name: str) -> pd.DataFrame | dict | str:
    """
    Fetch data from a common NBA data source on GitHub.

    Args:
        source_name: Name of the source (see COMMON_NBA_SOURCES)

    Returns:
        DataFrame, dict, or string depending on file format

    Raises:
        KeyError: If source name is invalid
    """
    if source_name not in COMMON_NBA_SOURCES:
        raise KeyError(
            f"Invalid source '{source_name}'. Choose from: {list(COMMON_NBA_SOURCES.keys())}"
        )

    url = COMMON_NBA_SOURCES[source_name]
    fetcher = GitHubDataFetcher()
    return await fetcher.fetch(url).data
