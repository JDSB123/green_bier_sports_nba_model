#!/usr/bin/env python3
"""
Aggregate FG/1H/Q1 betting lines from saved The Odds API payloads.

Reads JSON snapshots (current or historical) and produces a single CSV with
consensus lines per game and market. Consensus is computed via median across
all bookmakers to minimize outlier impact.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


def american_to_implied_prob(american_odds: int | float | None) -> Optional[float]:
    if american_odds is None:
        return None
    odds = float(american_odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    return None


def _load_events(payload: Any) -> List[Dict[str, Any]]:
    """
    Load events from various payload formats:
    - Direct list of events
    - {"data": [...]} format
    - {"events": [...]} format
    - Period odds format: {"data": [{"data": {...event...}, ...}, ...]}
    """
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        # Check for period_odds format (nested data -> data -> event)
        if "data" in payload and "markets" in payload:
            # This is the period_odds format
            data = payload.get("data") or []
            events = []
            for item in data:
                if isinstance(item, dict) and "data" in item:
                    # The actual event is nested inside item['data']
                    event = item.get("data")
                    if isinstance(event, dict):
                        events.append(event)
            return events
        # Standard format
        if "data" in payload:
            data = payload.get("data") or []
            if isinstance(data, list):
                return data
        if "events" in payload:
            data = payload.get("events") or []
            if isinstance(data, list):
                return data
    return []


def _median(values: Iterable[float]) -> Optional[float]:
    cleaned = [float(v) for v in values if v is not None]
    if not cleaned:
        return None
    return float(median(cleaned))


def summarize_event(
    event: Dict[str, Any],
    source_file: Path,
    query_ts: datetime,
) -> Dict[str, Any]:
    """Summarize median lines/prices for an event."""
    home = event.get("home_team") or event.get("teams", [None, None])[0]
    away = event.get("away_team") or event.get("teams", [None, None])[1]
    commence_time = event.get("commence_time") or event.get("start_time")
    event_id = event.get("id") or event.get("event_id")

    accumulator: Dict[str, List[float]] = defaultdict(list)

    bookmakers = event.get("bookmakers") or []
    for bookmaker in bookmakers:
        for market in bookmaker.get("markets") or []:
            key = (market.get("key") or market.get("market") or "").lower()
            outcomes = market.get("outcomes") or []

            # Full Game Moneyline (h2h)
            if key == "h2h":
                for outcome in outcomes:
                    if outcome.get("name") == home:
                        accumulator["fg_ml_home"].append(outcome.get("price"))
                    elif outcome.get("name") == away:
                        accumulator["fg_ml_away"].append(outcome.get("price"))

            # Full Game Spreads
            elif key == "spreads":
                for outcome in outcomes:
                    if outcome.get("name") == home:
                        accumulator["fg_spread_line"].append(outcome.get("point"))
                        accumulator["fg_spread_price"].append(outcome.get("price"))
                    elif outcome.get("name") == away:
                        accumulator["fg_spread_line_away"].append(outcome.get("point"))

            # Full Game Totals
            elif key == "totals":
                for outcome in outcomes:
                    if outcome.get("name") == "Over":
                        accumulator["fg_total_line"].append(outcome.get("point"))
                        accumulator["fg_total_price"].append(outcome.get("price"))

            # First Half Moneyline (h2h_h1)
            elif key == "h2h_h1":
                for outcome in outcomes:
                    if outcome.get("name") == home:
                        accumulator["fh_ml_home"].append(outcome.get("price"))
                    elif outcome.get("name") == away:
                        accumulator["fh_ml_away"].append(outcome.get("price"))

            # First Half Spreads
            elif key == "spreads_h1":
                for outcome in outcomes:
                    if outcome.get("name") == home:
                        accumulator["fh_spread_line"].append(outcome.get("point"))
                        accumulator["fh_spread_price"].append(outcome.get("price"))

            # First Half Totals
            elif key == "totals_h1":
                for outcome in outcomes:
                    if outcome.get("name") == "Over":
                        accumulator["fh_total_line"].append(outcome.get("point"))
                        accumulator["fh_total_price"].append(outcome.get("price"))

            # Q1 Spreads (keeping for backwards compatibility, but not used in model)
            elif key == "spreads_q1":
                for outcome in outcomes:
                    if outcome.get("name") == home:
                        accumulator["q1_spread_line"].append(outcome.get("point"))
                        accumulator["q1_spread_price"].append(outcome.get("price"))

            # Q1 Totals (keeping for backwards compatibility, but not used in model)
            elif key == "totals_q1":
                for outcome in outcomes:
                    if outcome.get("name") == "Over":
                        accumulator["q1_total_line"].append(outcome.get("point"))
                        accumulator["q1_total_price"].append(outcome.get("price"))

    row = {
        "event_id": event_id,
        "home_team": home,
        "away_team": away,
        "commence_time": commence_time,
        "query_timestamp": query_ts.isoformat(),
        "source_file": str(source_file),
    }

    for key, values in accumulator.items():
        row[key] = _median(values)
    return row


def parse_file(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    events = _load_events(payload)
    if not events:
        logger.warning(f"{path}: no events found")
    query_ts = datetime.utcfromtimestamp(path.stat().st_mtime)
    rows = [summarize_event(event, path, query_ts) for event in events]
    return rows


def gather_files(patterns: List[str]) -> List[Path]:
    files: List[Path] = []
    for pattern in patterns:
        files.extend(Path().glob(pattern))
    seen = set()
    unique_files: List[Path] = []
    for path in sorted(files):
        if path not in seen and path.is_file():
            seen.add(path)
            unique_files.append(path)
    return unique_files


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract consensus betting lines from odds snapshots")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "data/raw/the_odds/historical/*.json",
            "data/raw/the_odds/odds_*.json",
        ],
        help="Glob(s) pointing to odds payloads",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(settings.data_processed_dir) / "betting_lines.csv",
        help="Output CSV path",
    )
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    files = gather_files(args.inputs)
    if not files:
        logger.error("No odds payloads found. Provide at least one --inputs glob.")
        return 1

    logger.info(f"Processing {len(files)} odds snapshot(s)")
    rows: List[Dict[str, Any]] = []
    for path in files:
        rows.extend(parse_file(path))

    if not rows:
        logger.error("No betting lines extracted from provided files.")
        return 1

    df = pd.DataFrame(rows)
    df["commence_time"] = pd.to_datetime(df["commence_time"], errors="coerce")
    df = df.sort_values(["commence_time", "home_team", "away_team"])

    # Merge duplicate games by event_id, combining non-null values from multiple sources
    # This handles the case where FG data and 1H data come from different files
    # with slightly different commence_times for the same event
    numeric_cols = [c for c in df.columns if c not in
                    ["event_id", "home_team", "away_team", "commence_time", "query_timestamp", "source_file"]]

    # Custom aggregation that takes first non-null value
    def first_valid(series):
        valid = series.dropna()
        return valid.iloc[0] if len(valid) > 0 else None

    agg_dict = {
        "home_team": "first",
        "away_team": "first",
        "commence_time": "first",  # Take the first (usually the FG odds file's time)
        "query_timestamp": "last",
        "source_file": "last",
    }
    # For numeric betting columns, take the first non-null value
    for col in numeric_cols:
        if col in df.columns:
            agg_dict[col] = first_valid

    # Group by event_id to merge FG and period odds for same game
    df = df.groupby(["event_id"], as_index=False).agg(agg_dict)
    df = df.sort_values(["commence_time", "home_team", "away_team"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    logger.info(f"Wrote consensus betting lines to {args.output} ({len(df)} games)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
