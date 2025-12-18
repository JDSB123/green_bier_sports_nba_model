#!/usr/bin/env python3
"""
Fetch historical odds snapshots (FG/1H/Q1 markets) from The Odds API.

Usage examples:
    python scripts/collect_historical_lines.py --start-date 2025-10-01 --end-date 2025-12-01
    python scripts/collect_historical_lines.py --start-date 2025-12-10 --days 3

The script stores each snapshot under data/raw/the_odds/historical/.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.ingestion import the_odds
from src.utils.logging import get_logger

logger = get_logger(__name__)

MARKETS = ",".join([
    "h2h",
    "spreads",
    "totals",
    "h2h_h1",
    "spreads_h1",
    "totals_h1",
    "h2h_q1",
    "spreads_q1",
    "totals_q1",
])


def _daterange(start: date, end: date) -> Iterable[date]:
    """Yield each date between the start and end dates (inclusive)."""
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _snapshot_datetime(day: date, hour: int = 12) -> datetime:
    """Convert a date to a timezone-aware datetime used for the snapshot query."""
    return datetime(day.year, day.month, day.day, hour, 0, 0, tzinfo=timezone.utc)


async def fetch_snapshot(snapshot_dt: datetime) -> Optional[Path]:
    """Fetch and persist a historical odds snapshot for the provided datetime."""
    iso_ts = snapshot_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    logger.info(f"Fetching historical odds for {iso_ts} (UTC)")
    try:
        data = await the_odds.fetch_historical_odds(
            date=iso_ts,
            markets=MARKETS,
        )
    except Exception as exc:
        logger.error(f"Failed to fetch historical odds for {iso_ts}: {exc}")
        return None

    out_dir = Path(settings.data_raw_dir) / "the_odds" / "historical"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"historical_{snapshot_dt.strftime('%Y%m%dT%H%M%SZ')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved historical odds snapshot to {out_path}")
    return out_path


async def run_collection(
    start_date: date,
    end_date: date,
    snapshot_hour: int,
) -> List[Path]:
    """Fetch snapshots for the inclusive date range."""
    saved_paths: List[Path] = []
    for day in _daterange(start_date, end_date):
        snapshot_dt = _snapshot_datetime(day, snapshot_hour)
        path = await fetch_snapshot(snapshot_dt)
        if path:
            saved_paths.append(path)
    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect historical odds snapshots (FG/1H/Q1) from The Odds API"
    )
    parser.add_argument("--start-date", type=str, required=True, help="YYYY-MM-DD")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--end-date", type=str, help="YYYY-MM-DD (inclusive)")
    group.add_argument("--days", type=int, help="Number of days to collect (starting at start-date)")
    parser.add_argument(
        "--snapshot-hour",
        type=int,
        default=12,
        help="UTC hour used for the historical snapshot (default: noon UTC)",
    )
    return parser.parse_args()


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


async def main_async() -> None:
    args = parse_args()
    start = _parse_date(args.start_date)
    if args.end_date:
        end = _parse_date(args.end_date)
    elif args.days:
        end = start + timedelta(days=max(0, args.days - 1))
    else:
        end = start

    if end < start:
        raise ValueError("end-date must be on or after start-date")

    logger.info(
        f"Collecting historical odds from {start.isoformat()} to {end.isoformat()} "
        f"(markets: {MARKETS})"
    )
    saved = await run_collection(start, end, args.snapshot_hour)
    logger.info(f"Completed collection: {len(saved)} snapshot(s) stored.")


def main() -> int:
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 1
    except Exception as exc:
        logger.error(f"Collection failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
