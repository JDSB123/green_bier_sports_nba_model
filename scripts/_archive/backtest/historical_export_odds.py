#!/usr/bin/env python3
"""
Export and Normalize Historical Odds Data.

This script processes the raw JSON files from the historical odds ingestion
and exports them to normalized formats (CSV, Parquet) suitable for analysis.

The exported data can be used for:
- Backtesting betting strategies
- Model training (separate from live pipeline)
- Historical analysis and reporting

Output formats:
- CSV: Human-readable, easy import to Excel/Sheets
- Parquet: Efficient for large datasets, preserves types

Storage Structure:
    data/historical/exports/
    ├── {season}_events.csv
    ├── {season}_events.parquet
    ├── {season}_odds_featured.csv
    ├── {season}_odds_featured.parquet
    ├── {season}_odds_periods.csv
    ├── {season}_odds_periods.parquet
    └── manifest.json

Usage:
    # Export all ingested data
    python scripts/historical_export_odds.py

    # Export specific season
    python scripts/historical_export_odds.py --season 2023-2024

    # Export to CSV only
    python scripts/historical_export_odds.py --format csv

    # Export with team name standardization
    python scripts/historical_export_odds.py --standardize-teams
"""
from __future__ import annotations
from src.utils.logging import get_logger
from src.utils.historical_guard import resolve_historical_output_root, require_historical_mode, ensure_historical_path

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# flake8: noqa: E402

logger = get_logger(__name__)


# ==============================================================================
# DATA NORMALIZATION
# ==============================================================================


def normalize_events(events_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Normalize events data to a flat DataFrame.

    Args:
        events_data: List of event dictionaries from The Odds API

    Returns:
        DataFrame with normalized event data
    """
    rows = []
    for event in events_data:
        rows.append({
            "event_id": event.get("id"),
            "sport_key": event.get("sport_key"),
            "sport_title": event.get("sport_title"),
            "commence_time": event.get("commence_time"),
            "home_team": event.get("home_team"),
            "away_team": event.get("away_team"),
            "completed": event.get("completed", False),
        })

    df = pd.DataFrame(rows)

    if not df.empty:
        df["commence_time"] = pd.to_datetime(df["commence_time"], utc=True)
        df["game_date"] = df["commence_time"].dt.date

    return df


def normalize_odds(
    odds_data: Dict[str, Any],
    include_all_bookmakers: bool = False,
) -> pd.DataFrame:
    """
    Normalize odds data to a flat DataFrame.

    Args:
        odds_data: Odds response from The Odds API
        include_all_bookmakers: If False, only include primary bookmakers

    Returns:
        DataFrame with normalized odds data
    """
    primary_bookmakers = {
        "fanduel", "draftkings", "betmgm", "caesars", "pointsbetus",
        "bovada", "betonlineag", "betrivers", "unibet_us", "wynnbet",
        "superbook", "barstool", "twinspires", "betus"
    }

    rows = []

    # Handle both list and dict with 'data' key
    if isinstance(odds_data, list):
        events = odds_data
        timestamp = None
    else:
        events = odds_data.get("data", [])
        timestamp = odds_data.get("timestamp")

    for event in events:
        event_id = event.get("id")
        home_team = event.get("home_team")
        away_team = event.get("away_team")
        commence_time = event.get("commence_time")

        bookmakers = event.get("bookmakers", [])

        for bookmaker in bookmakers:
            bm_key = bookmaker.get("key")
            bm_title = bookmaker.get("title")

            # Filter to primary bookmakers if requested
            if not include_all_bookmakers and bm_key not in primary_bookmakers:
                continue

            last_update = bookmaker.get("last_update")

            markets = bookmaker.get("markets", [])

            for market in markets:
                market_key = market.get("key")
                outcomes = market.get("outcomes", [])

                for outcome in outcomes:
                    rows.append({
                        "snapshot_timestamp": timestamp,
                        "event_id": event_id,
                        "home_team": home_team,
                        "away_team": away_team,
                        "commence_time": commence_time,
                        "bookmaker_key": bm_key,
                        "bookmaker_title": bm_title,
                        "market_key": market_key,
                        "outcome_name": outcome.get("name"),
                        "outcome_price": outcome.get("price"),
                        "outcome_point": outcome.get("point"),
                        "last_update": last_update,
                    })

    df = pd.DataFrame(rows)

    if not df.empty:
        df["commence_time"] = pd.to_datetime(df["commence_time"], utc=True)
        df["last_update"] = pd.to_datetime(df["last_update"], utc=True)
        df["snapshot_timestamp"] = pd.to_datetime(
            df["snapshot_timestamp"], utc=True
        )
        df["game_date"] = df["commence_time"].dt.date

    return df


def normalize_player_props(props_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Normalize player props data to a flat DataFrame.

    Args:
        props_data: Props response from The Odds API

    Returns:
        DataFrame with normalized player props data
    """
    rows = []

    if isinstance(props_data, list):
        events = props_data
        timestamp = None
    else:
        events = props_data.get("data", [])
        timestamp = props_data.get("timestamp")

    for event in events:
        event_id = event.get("id")
        home_team = event.get("home_team")
        away_team = event.get("away_team")
        commence_time = event.get("commence_time")

        bookmakers = event.get("bookmakers", [])

        for bookmaker in bookmakers:
            bm_key = bookmaker.get("key")
            bm_title = bookmaker.get("title")
            last_update = bookmaker.get("last_update")

            markets = bookmaker.get("markets", [])

            for market in markets:
                market_key = market.get("key")

                # Skip non-player markets
                if not market_key.startswith("player_"):
                    continue

                outcomes = market.get("outcomes", [])

                for outcome in outcomes:
                    # Player props have description field for player name
                    player_name = outcome.get("description")

                    rows.append({
                        "snapshot_timestamp": timestamp,
                        "event_id": event_id,
                        "home_team": home_team,
                        "away_team": away_team,
                        "commence_time": commence_time,
                        "bookmaker_key": bm_key,
                        "bookmaker_title": bm_title,
                        "market_key": market_key,
                        "player_name": player_name,
                        "outcome_name": outcome.get("name"),
                        "outcome_price": outcome.get("price"),
                        "outcome_point": outcome.get("point"),
                        "last_update": last_update,
                    })

    df = pd.DataFrame(rows)

    if not df.empty:
        df["commence_time"] = pd.to_datetime(df["commence_time"], utc=True)
        df["last_update"] = pd.to_datetime(df["last_update"], utc=True)
        df["snapshot_timestamp"] = pd.to_datetime(
            df["snapshot_timestamp"], utc=True
        )
        df["game_date"] = df["commence_time"].dt.date

    return df


# ==============================================================================
# EXPORT PROCESSOR
# ==============================================================================


class HistoricalDataExporter:
    """Exports historical odds data to normalized formats."""

    def __init__(
        self,
        source_dir: str = "data/historical/the_odds",
        export_dir: str = "data/historical/exports",
    ):
        self.source_dir = Path(source_dir)
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)

    def _load_json_files(self, pattern: str) -> List[Dict[str, Any]]:
        """Load all JSON files matching pattern."""
        files = sorted(self.source_dir.glob(pattern))
        data = []
        for f in files:
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    data.append({
                        "file": str(f),
                        "data": json.load(fp)
                    self,
                    source_dir: str | None = None,
                    export_dir: str | None = None,
        return data
                    self.source_dir = Path(source_dir) if source_dir else resolve_historical_output_root("the_odds")
                    self.export_dir = Path(export_dir) if export_dir else resolve_historical_output_root("exports")
        self,
        season: str,
        output_format: str = "both",
    ) -> List[Path]:
        """
        Export events for a season.

        Args:
            season: Season string (e.g., "2023-2024")
            output_format: "csv", "parquet", or "both"

        Returns:
            List of exported file paths
        """
        logger.info(f"Exporting events for season {season}")

        # Load all event files for season
        pattern = f"events/{season}/events_*.json"
        file_data = self._load_json_files(pattern)

        if not file_data:
            logger.warning(f"No events data found for {season}")
            return []

        # Combine all events
        all_events = []
        for fd in file_data:
            data = fd["data"]
            events = data.get("data", []) if isinstance(data, dict) else data
            all_events.extend(events)

        logger.info(f"Found {len(all_events)} events")

        # Normalize
        df = normalize_events(all_events)

        # Deduplicate by event_id (keep latest)
        if not df.empty:
            df = df.drop_duplicates(subset=["event_id"], keep="last")
            df = df.sort_values("commence_time")

        # Export
        exported = []
        base_name = f"{season}_events"

        if output_format in ("csv", "both"):
            csv_path = self.export_dir / f"{base_name}.csv"
            df.to_csv(csv_path, index=False)
            exported.append(csv_path)
            logger.info(f"Exported {len(df)} events to {csv_path}")

        if output_format in ("parquet", "both"):
            parquet_path = self.export_dir / f"{base_name}.parquet"
            df.to_parquet(parquet_path, index=False)
            exported.append(parquet_path)
            logger.info(f"Exported {len(df)} events to {parquet_path}")

        return exported

    def export_season_odds(
        self,
        season: str,
        market_group: str="featured",
        output_format: str="both",
        include_all_bookmakers: bool=False,
    ) -> List[Path]:
        """
        Export odds for a season.

        Args:
            season: Season string
            market_group: Market group name
            output_format: "csv", "parquet", or "both"
            include_all_bookmakers: Include all bookmakers vs. primary only

        Returns:
            List of exported file paths
        """
        logger.info(f"Exporting {market_group} odds for season {season}")

        # Load odds files
        pattern = f"odds/{season}/odds_*_{market_group}.json"
        file_data = self._load_json_files(pattern)

        if not file_data:
            logger.warning(f"No {market_group} odds data found for {season}")
            return []

        # Combine and normalize
        all_dfs = []
        for fd in file_data:
            df = normalize_odds(fd["data"], include_all_bookmakers)
            if not df.empty:
                all_dfs.append(df)

        if not all_dfs:
            return []

        df = pd.concat(all_dfs, ignore_index=True)

        # Sort by time
        sort_cols = ["commence_time", "event_id",
                     "bookmaker_key", "market_key"]
        df = df.sort_values(sort_cols)

        logger.info(f"Processed {len(df)} odds records")

        # Export
        exported = []
        base_name = f"{season}_odds_{market_group}"

        if output_format in ("csv", "both"):
            csv_path = self.export_dir / f"{base_name}.csv"
            df.to_csv(csv_path, index=False)
            exported.append(csv_path)
            logger.info(f"Exported to {csv_path}")

        if output_format in ("parquet", "both"):
            parquet_path = self.export_dir / f"{base_name}.parquet"
            df.to_parquet(parquet_path, index=False)
            exported.append(parquet_path)
            logger.info(f"Exported to {parquet_path}")

        return exported

    def export_season_player_props(
        self,
        season: str,
        output_format: str="both",
    ) -> List[Path]:
        """
        Export player props for a season.

        Args:
            season: Season string
            output_format: "csv", "parquet", or "both"

        Returns:
            List of exported file paths
        """
        logger.info(f"Exporting player props for season {season}")

        # Load props files
        pattern = f"player_props/{season}/props_*.json"
        file_data = self._load_json_files(pattern)

        if not file_data:
            logger.warning(f"No player props data found for {season}")
            return []

        # Combine and normalize
        all_dfs = []
        for fd in file_data:
            df = normalize_player_props(fd["data"])
            if not df.empty:
                all_dfs.append(df)

        if not all_dfs:
            return []

        df = pd.concat(all_dfs, ignore_index=True)
        sort_cols = ["commence_time", "event_id", "player_name", "market_key"]
        df = df.sort_values(sort_cols)

        logger.info(f"Processed {len(df)} player prop records")

        # Export
        exported = []
        base_name = f"{season}_player_props"

        if output_format in ("csv", "both"):
            csv_path = self.export_dir / f"{base_name}.csv"
            df.to_csv(csv_path, index=False)
            exported.append(csv_path)
            logger.info(f"Exported to {csv_path}")

        if output_format in ("parquet", "both"):
            parquet_path = self.export_dir / f"{base_name}.parquet"
            df.to_parquet(parquet_path, index=False)
            exported.append(parquet_path)
            logger.info(f"Exported to {parquet_path}")

        return exported

    def export_all(
        self,
        season: Optional[str]=None,
        output_format: str="both",
        include_all_bookmakers: bool=False,
    ) -> Dict[str, List[Path]]:
        """
        Export all available data.

        Args:
            season: Optional specific season to export
            output_format: "csv", "parquet", or "both"
            include_all_bookmakers: Include all bookmakers

        Returns:
            Dict mapping data type to list of exported paths
        """
        results: Dict[str, List[Path]] = {
            "events": [],
            "odds": [],
            "player_props": [],
        }

        # Discover available seasons
        seasons: set = set()
        for subdir in ["events", "odds", "player_props"]:
            dir_path = self.source_dir / subdir
            if dir_path.exists():
                for s in dir_path.iterdir():
                    if s.is_dir():
                        seasons.add(s.name)

        if season:
            seasons = {season} if season in seasons else set()

        if not seasons:
            logger.warning("No seasons found to export")
            return results

        logger.info(f"Exporting seasons: {sorted(seasons)}")

        # Export each season
        for s in sorted(seasons):
            # Events
            paths = self.export_season_events(s, output_format)
            results["events"].extend(paths)

            # Odds by market group
            for mg in ["featured", "periods", "alternates", "team_totals"]:
                paths = self.export_season_odds(
                    s, mg, output_format, include_all_bookmakers
                )
                results["odds"].extend(paths)

            # Player props
            paths = self.export_season_player_props(s, output_format)
            results["player_props"].extend(paths)

        # Save manifest
        manifest = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "seasons": sorted(seasons),
            "files": {k: [str(p) for p in v] for k, v in results.items()},
            "total_files": sum(len(v) for v in results.values()),
        }

        manifest_path = self.export_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info(
            f"Exported {manifest['total_files']} files. Manifest: {manifest_path}"
        )

        return results

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of source data available."""
        summary: Dict[str, Any] = {
            "source_dir": str(self.source_dir),
            "export_dir": str(self.export_dir),
            "seasons": {},
        }

        for subdir in ["events", "odds", "player_props"]:
            dir_path = self.source_dir / subdir
            if dir_path.exists():
                for season_dir in dir_path.iterdir():
                    if season_dir.is_dir():
                        season = season_dir.name
                        if season not in summary["seasons"]:
                            summary["seasons"][season] = {
                                "events_files": 0,
                                "odds_files": 0,
                                "props_files": 0,
                            }

                        files = list(season_dir.glob("*.json"))
                        if subdir == "events":
                            summary["seasons"][season]["events_files"] = len(
                                files)
                        elif subdir == "odds":
                            summary["seasons"][season]["odds_files"] = len(
                                files)
                        elif subdir == "player_props":
                            summary["seasons"][season]["props_files"] = len(
                                files)

        return summary


# ==============================================================================
# CLI
# ==============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export historical odds data to CSV/Parquet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--season",
        type=str,
        help="Specific season to export (e.g., 2023-2024)",
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "parquet", "both"],
        default="both",
        help="Output format (default: both)",
    )

    parser.add_argument(
        "--source-dir",
        type=str,
        default=None,
        help="Source directory for raw JSON files (defaults to HISTORICAL_OUTPUT_ROOT/the_odds)",
    )

    parser.add_argument(
        "--export-dir",
        type=str,
        default=None,
        help="Output directory for exported files (defaults to HISTORICAL_OUTPUT_ROOT/exports)",
    )

    parser.add_argument(
        "--include-all-bookmakers",
        action="store_true",
        help="Include all bookmakers (default: primary US books only)",
    )

    parser.add_argument(
        "--show-summary",
        action="store_true",
        help="Show summary of available data and exit",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    require_historical_mode()
    args = parse_args()
    if args.source_dir:
        ensure_historical_path(Path(args.source_dir), "source-dir")
    if args.export_dir:
        ensure_historical_path(Path(args.export_dir), "export-dir")

    exporter = HistoricalDataExporter(
        source_dir=args.source_dir,
        export_dir=args.export_dir,
    )

    if args.show_summary:
        summary = exporter.get_summary()
        print("\n=== Historical Data Summary ===")
        print(f"Source: {summary['source_dir']}")
        print(f"Export: {summary['export_dir']}")
        print("\nSeasons:")
        for season, data in sorted(summary["seasons"].items()):
            print(f"  {season}:")
            print(f"    Events files: {data['events_files']}")
            print(f"    Odds files: {data['odds_files']}")
            print(f"    Props files: {data['props_files']}")
        return 0

    try:
        results = exporter.export_all(
            season=args.season,
            output_format=args.format,
            include_all_bookmakers=args.include_all_bookmakers,
        )

        print("\n=== Export Complete ===")
        for data_type, paths in results.items():
            if paths:
                print(f"\n{data_type.title()}:")
                for p in paths:
                    print(f"  - {p}")

        return 0

    except Exception as e:
        logger.error(f"Export failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
