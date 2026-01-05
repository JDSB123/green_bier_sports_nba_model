#!/usr/bin/env python3
"""
Cache consensus betting lines from committed The Odds API historical data.

Reads committed raw JSON under:
  - data/historical/the_odds/odds/<season>/odds_*_featured.json
  - data/historical/the_odds/period_odds/<season>/period_odds_1h.json (optional)

Outputs a single CSV with one row per event, including:
  - fg_spread_line, fg_total_line
  - fh_spread_line, fh_total_line (when available)

This is used to merge real historical lines into training_data.csv for
leakage-safe training/backtesting of the 4 independent market models.

Usage:
  python scripts/cache_theodds_lines.py
  python scripts/cache_theodds_lines.py --seasons 2023-2024 2024-2025
  python scripts/cache_theodds_lines.py --bookmaker draftkings
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent


def _median(values: List[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return float(statistics.median(vals))


def _iter_json_files(root: Path, pattern: str) -> Iterable[Path]:
    yield from sorted(root.glob(pattern))


def _extract_lines_from_event(
    event: Dict[str, Any],
    bookmaker_key: Optional[str] = None,
) -> Dict[str, Optional[float]]:
    """
    Extract consensus FG + 1H lines from a single The Odds event.

    Returns:
      { fg_spread_line, fg_total_line, fh_spread_line, fh_total_line }
    """
    home_team = event.get("home_team")
    bookmakers = event.get("bookmakers") or []

    fg_spreads: List[float] = []
    fg_totals: List[float] = []
    fh_spreads: List[float] = []
    fh_totals: List[float] = []

    for bm in bookmakers:
        if bookmaker_key and bm.get("key") != bookmaker_key:
            continue
        for market in bm.get("markets", []) or []:
            key = (market.get("key") or "").lower()
            outcomes = market.get("outcomes") or []

            if key == "spreads":
                for o in outcomes:
                    if o.get("name") == home_team and o.get("point") is not None:
                        fg_spreads.append(float(o["point"]))
            elif key == "totals":
                for o in outcomes:
                    if o.get("name") == "Over" and o.get("point") is not None:
                        fg_totals.append(float(o["point"]))
            elif key == "spreads_h1":
                for o in outcomes:
                    if o.get("name") == home_team and o.get("point") is not None:
                        fh_spreads.append(float(o["point"]))
            elif key == "totals_h1":
                for o in outcomes:
                    if o.get("name") == "Over" and o.get("point") is not None:
                        fh_totals.append(float(o["point"]))

    return {
        "fg_spread_line": _median(fg_spreads),
        "fg_total_line": _median(fg_totals),
        "fh_spread_line": _median(fh_spreads),
        "fh_total_line": _median(fh_totals),
    }


def load_fg_featured_odds(season_dir: Path, bookmaker_key: Optional[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for path in _iter_json_files(season_dir, "odds_*_featured.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        events = payload.get("data") or []
        snapshot_ts = payload.get("timestamp")
        for ev in events:
            lines = _extract_lines_from_event(ev, bookmaker_key=bookmaker_key)
            if not any(lines.values()):
                continue
            rows.append({
                "event_id": ev.get("id"),
                "commence_time": ev.get("commence_time"),
                "home_team": ev.get("home_team"),
                "away_team": ev.get("away_team"),
                "snapshot_timestamp": snapshot_ts,
                **lines,
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["commence_time"] = pd.to_datetime(df["commence_time"], utc=True, errors="coerce")
        df["snapshot_timestamp"] = pd.to_datetime(df["snapshot_timestamp"], utc=True, errors="coerce")
        df["line_date"] = df["commence_time"].dt.date
    return df


def load_1h_period_odds(period_file: Path, bookmaker_key: Optional[str]) -> pd.DataFrame:
    payload = json.loads(period_file.read_text(encoding="utf-8"))
    wrappers = payload.get("data") or []

    rows: List[Dict[str, Any]] = []
    for w in wrappers:
        ev = w.get("data") or {}
        if not ev:
            continue
        lines = _extract_lines_from_event(ev, bookmaker_key=bookmaker_key)
        # For period file we only expect 1H lines, but keep schema consistent
        if lines.get("fh_spread_line") is None and lines.get("fh_total_line") is None:
            continue
        rows.append({
            "event_id": ev.get("id"),
            "commence_time": ev.get("commence_time"),
            "home_team": ev.get("home_team"),
            "away_team": ev.get("away_team"),
            "snapshot_timestamp": w.get("timestamp") or payload.get("fetched_at"),
            "fh_spread_line": lines.get("fh_spread_line"),
            "fh_total_line": lines.get("fh_total_line"),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["commence_time"] = pd.to_datetime(df["commence_time"], utc=True, errors="coerce")
        df["snapshot_timestamp"] = pd.to_datetime(df["snapshot_timestamp"], utc=True, errors="coerce")
        df["line_date"] = df["commence_time"].dt.date
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cache The Odds historical lines to CSV")
    p.add_argument(
        "--seasons",
        nargs="*",
        default=[],
        help="Seasons to process (e.g. 2023-2024 2024-2025). Default: auto-discover.",
    )
    p.add_argument(
        "--bookmaker",
        type=str,
        default=None,
        help="Optional bookmaker key (e.g. draftkings). Default: median across all books.",
    )
    p.add_argument(
        "--out",
        type=str,
        default="data/historical/derived/theodds_lines.csv",
        help="Output CSV path",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    base = PROJECT_ROOT / "data" / "historical" / "the_odds"
    odds_root = base / "odds"
    period_root = base / "period_odds"

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seasons = args.seasons
    if not seasons:
        seasons = sorted([p.name for p in odds_root.iterdir() if p.is_dir()])

    all_rows: List[pd.DataFrame] = []
    for season in seasons:
        season_odds_dir = odds_root / season
        if season_odds_dir.exists():
            fg_df = load_fg_featured_odds(season_odds_dir, args.bookmaker)
        else:
            fg_df = pd.DataFrame()

        period_file = period_root / season / "period_odds_1h.json"
        if period_file.exists():
            fh_df = load_1h_period_odds(period_file, args.bookmaker)
        else:
            fh_df = pd.DataFrame()

        if fg_df.empty and fh_df.empty:
            continue

        # Merge FG + 1H per event (prefer exact event_id match)
        if not fg_df.empty and not fh_df.empty:
            merged = pd.merge(
                fg_df,
                fh_df[["event_id", "fh_spread_line", "fh_total_line"]],
                on="event_id",
                how="left",
                suffixes=("", "_fh"),
            )
            # Coalesce 1H columns (FG featured snapshots don't include 1H markets)
            if "fh_spread_line_fh" in merged.columns:
                merged["fh_spread_line"] = merged["fh_spread_line"].combine_first(merged["fh_spread_line_fh"])
                merged = merged.drop(columns=["fh_spread_line_fh"])
            if "fh_total_line_fh" in merged.columns:
                merged["fh_total_line"] = merged["fh_total_line"].combine_first(merged["fh_total_line_fh"])
                merged = merged.drop(columns=["fh_total_line_fh"])
        elif not fg_df.empty:
            merged = fg_df
        else:
            merged = fh_df

        merged["season"] = season
        all_rows.append(merged)

    if not all_rows:
        print("[WARN] No lines found.")
        return 1

    df = pd.concat(all_rows, ignore_index=True)
    df = df.drop_duplicates(subset=["event_id"], keep="last")
    df = df.sort_values(["commence_time", "event_id"]).reset_index(drop=True)
    df.to_csv(out_path, index=False)
    print(f"[OK] Wrote {len(df)} events to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

