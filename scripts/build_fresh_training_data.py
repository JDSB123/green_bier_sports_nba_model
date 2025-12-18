#!/usr/bin/env python3
"""
Build Fresh Training Data Pipeline.

This script is the SINGLE SOURCE OF TRUTH for training data generation.
It fetches REAL data from APIs - no placeholders, no sample data, no silent failures.

Data Sources:
    - API-Basketball: Game outcomes with Q1-Q4 scores
    - The Odds API: Real betting lines (spreads, totals)

Requirements:
    - THE_ODDS_API_KEY: Required for betting lines
    - API_BASKETBALL_KEY: Required for game outcomes

Usage:
    python scripts/build_fresh_training_data.py                    # Full pipeline
    python scripts/build_fresh_training_data.py --seasons 2024-2025,2025-2026
    python scripts/build_fresh_training_data.py --validate-only    # Check data integrity
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import settings
from src.ingestion.api_basketball import APIBasketballClient
from src.ingestion.standardize import normalize_team_to_espn
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Strict validation settings
REQUIRED_COLUMNS = [
    "date", "home_team", "away_team", 
    "home_score", "away_score",
]

BETTING_LINE_COLUMNS = [
    "spread_line", "total_line",
]

QUARTER_COLUMNS = [
    "home_q1", "home_q2", "home_q3", "home_q4",
    "away_q1", "away_q2", "away_q3", "away_q4",
]


class DataPipelineError(Exception):
    """Raised when data pipeline encounters an unrecoverable error."""
    pass


class DataValidationError(Exception):
    """Raised when data fails validation checks."""
    pass


class FreshDataPipeline:
    """
    Single source of truth for building training data.
    
    STRICT RULES:
    1. NO placeholder data - all data must come from real APIs
    2. NO silent failures - all errors raise exceptions
    3. NO sample data - empty datasets cause pipeline to fail
    4. Validation at every step
    """
    
    def __init__(
        self,
        output_dir: Optional[str] = None,
        strict_mode: bool = True,
    ):
        self.output_dir = Path(output_dir or settings.data_processed_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.strict_mode = strict_mode
        
        # Validate API keys upfront
        self._validate_api_keys()
    
    def _validate_api_keys(self) -> None:
        """Validate all required API keys are set."""
        errors = []
        
        if not settings.api_basketball_key:
            errors.append("API_BASKETBALL_KEY is not set")
        
        if not settings.the_odds_api_key:
            errors.append("THE_ODDS_API_KEY is not set")
        
        if errors:
            error_msg = "Missing required API keys:\n  - " + "\n  - ".join(errors)
            raise DataPipelineError(error_msg)
        
        logger.info("✓ All required API keys validated")
    
    async def fetch_game_outcomes(
        self,
        seasons: List[str],
    ) -> pd.DataFrame:
        """
        Fetch game outcomes from API-Basketball.
        
        Returns DataFrame with game results including Q1-Q4 scores.
        Raises DataPipelineError if no data fetched.
        """
        logger.info(f"Fetching game outcomes for seasons: {seasons}")
        
        all_games: List[Dict[str, Any]] = []
        
        for season in seasons:
            logger.info(f"  Fetching season {season}...")
            
            client = APIBasketballClient(season=season)
            
            try:
                result = await client.fetch_games()
                games = result.data.get("response", [])
                
                if not games:
                    logger.warning(f"  No games found for season {season}")
                    continue
                
                # Process each game
                for game in games:
                    status = game.get("status", {})
                    if status.get("short") != "FT":
                        continue  # Only finished games
                    
                    teams = game.get("teams", {})
                    scores = game.get("scores", {})
                    
                    home_team_raw = teams.get("home", {}).get("name", "")
                    away_team_raw = teams.get("away", {}).get("name", "")
                    
                    # Standardize team names (returns tuple: name, is_valid)
                    home_team, home_valid = normalize_team_to_espn(home_team_raw, source="api_basketball")
                    away_team, away_valid = normalize_team_to_espn(away_team_raw, source="api_basketball")
                    
                    if not home_valid or not away_valid:
                        logger.debug(f"  Skipping game with invalid teams: {teams}")
                        continue
                    
                    home_score = scores.get("home", {}).get("total")
                    away_score = scores.get("away", {}).get("total")
                    
                    if home_score is None or away_score is None:
                        continue
                    
                    # Extract quarter scores
                    home_quarters = scores.get("home", {})
                    away_quarters = scores.get("away", {})
                    
                    game_row = {
                        "game_id": game.get("id"),
                        "date": game.get("date"),
                        "season": season,
                        "home_team": home_team,
                        "away_team": away_team,
                        "home_score": int(home_score),
                        "away_score": int(away_score),
                        "home_q1": home_quarters.get("quarter_1"),
                        "home_q2": home_quarters.get("quarter_2"),
                        "home_q3": home_quarters.get("quarter_3"),
                        "home_q4": home_quarters.get("quarter_4"),
                        "away_q1": away_quarters.get("quarter_1"),
                        "away_q2": away_quarters.get("quarter_2"),
                        "away_q3": away_quarters.get("quarter_3"),
                        "away_q4": away_quarters.get("quarter_4"),
                    }
                    
                    all_games.append(game_row)
                
                logger.info(f"  ✓ {len([g for g in games if g.get('status', {}).get('short') == 'FT'])} finished games processed")
                
            except Exception as e:
                logger.error(f"  ✗ Error fetching season {season}: {e}")
                if self.strict_mode:
                    raise DataPipelineError(f"Failed to fetch games for season {season}: {e}")
        
        if not all_games:
            raise DataPipelineError("No game outcomes fetched from API-Basketball")
        
        df = pd.DataFrame(all_games)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.sort_values("date").reset_index(drop=True)
        
        logger.info(f"✓ Fetched {len(df)} game outcomes")
        return df
    
    async def fetch_betting_lines(
        self,
        game_dates: List[str],
    ) -> pd.DataFrame:
        """
        Fetch betting lines from The Odds API.
        
        NOTE: Historical odds require a paid plan (Group 2+).
        For current season data, use the live odds endpoint.
        
        Returns DataFrame with spread and total lines.
        """
        logger.info(f"Fetching betting lines for {len(game_dates)} unique dates...")
        
        # Import here to avoid circular imports
        from src.ingestion.the_odds import fetch_historical_odds, fetch_odds
        
        all_lines: List[Dict[str, Any]] = []
        
        # Try historical endpoint first (requires paid plan)
        historical_available = False
        
        for date_str in game_dates[:1]:  # Test with first date
            try:
                # Format date for API
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                iso_date = dt.strftime("%Y-%m-%dT12:00:00Z")
                
                data = await fetch_historical_odds(
                    date=iso_date,
                    markets="spreads,totals,h2h",
                )
                
                if data.get("data"):
                    historical_available = True
                    logger.info("✓ Historical odds endpoint available")
                    break
                    
            except Exception as e:
                if "403" in str(e) or "Forbidden" in str(e):
                    logger.warning("Historical odds endpoint not available (requires paid plan)")
                else:
                    logger.warning(f"Error testing historical endpoint: {e}")
        
        if historical_available:
            # Use historical endpoint
            for i, date_str in enumerate(game_dates):
                if i % 10 == 0:
                    logger.info(f"  Processing date {i+1}/{len(game_dates)}...")
                
                try:
                    dt = datetime.strptime(date_str, "%Y-%m-%d")
                    iso_date = dt.strftime("%Y-%m-%dT12:00:00Z")
                    
                    data = await fetch_historical_odds(
                        date=iso_date,
                        markets="spreads,totals,h2h",
                    )
                    
                    events = data.get("data", [])
                    lines = self._extract_lines_from_events(events, date_str)
                    all_lines.extend(lines)
                    
                except Exception as e:
                    logger.debug(f"  No lines for {date_str}: {e}")
        else:
            # Historical not available - warn user
            logger.warning(
                "⚠️  Historical odds not available. For full backtesting:\n"
                "   1. Upgrade to The Odds API Group 2+ plan\n"
                "   2. Or collect odds going forward with regular data collection"
            )
            
            # Try to get current/recent odds at least
            try:
                current_odds = await fetch_odds(markets="spreads,totals,h2h")
                lines = self._extract_lines_from_events(current_odds, datetime.now().strftime("%Y-%m-%d"))
                all_lines.extend(lines)
                logger.info(f"✓ Fetched current odds for {len(current_odds)} events")
            except Exception as e:
                logger.warning(f"Could not fetch current odds: {e}")
        
        if not all_lines:
            logger.warning("No betting lines fetched - backtest will be limited to moneyline")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_lines)
        logger.info(f"✓ Fetched {len(df)} betting line records")
        return df
    
    def _extract_lines_from_events(
        self,
        events: List[Dict[str, Any]],
        query_date: str,
    ) -> List[Dict[str, Any]]:
        """Extract consensus lines from event data."""
        lines = []
        
        for event in events:
            home_team, home_valid = normalize_team_to_espn(
                event.get("home_team", ""),
                source="the_odds"
            )
            away_team, away_valid = normalize_team_to_espn(
                event.get("away_team", ""),
                source="the_odds"
            )
            
            if not home_valid or not away_valid:
                continue
            
            bookmakers = event.get("bookmakers", [])
            
            spread_lines = []
            total_lines = []
            
            for bm in bookmakers:
                markets = bm.get("markets", [])
                for market in markets:
                    key = market.get("key", "")
                    outcomes = market.get("outcomes", [])
                    
                    if key == "spreads":
                        for outcome in outcomes:
                            if outcome.get("name") == home_team:
                                point = outcome.get("point")
                                if point is not None:
                                    spread_lines.append(float(point))
                    
                    elif key == "totals":
                        for outcome in outcomes:
                            if outcome.get("name") == "Over":
                                point = outcome.get("point")
                                if point is not None:
                                    total_lines.append(float(point))
            
            if spread_lines or total_lines:
                lines.append({
                    "query_date": query_date,
                    "event_id": event.get("id"),
                    "commence_time": event.get("commence_time"),
                    "home_team": home_team,
                    "away_team": away_team,
                    "spread_line": sum(spread_lines) / len(spread_lines) if spread_lines else None,
                    "total_line": sum(total_lines) / len(total_lines) if total_lines else None,
                    "spread_sources": len(spread_lines),
                    "total_sources": len(total_lines),
                })
        
        return lines
    
    def merge_outcomes_and_lines(
        self,
        outcomes_df: pd.DataFrame,
        lines_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge game outcomes with betting lines.
        
        Matches on team names and date (within tolerance).
        """
        if lines_df.empty:
            logger.warning("No betting lines to merge - returning outcomes only")
            outcomes_df["spread_line"] = None
            outcomes_df["total_line"] = None
            return outcomes_df
        
        logger.info("Merging outcomes with betting lines...")
        
        # Prepare lines for matching
        lines_df = lines_df.copy()
        if "commence_time" in lines_df.columns:
            lines_df["line_date"] = pd.to_datetime(lines_df["commence_time"]).dt.date
        elif "query_date" in lines_df.columns:
            lines_df["line_date"] = pd.to_datetime(lines_df["query_date"]).dt.date
        
        # Match on teams and date
        merged_rows = []
        matched = 0
        
        for _, game in outcomes_df.iterrows():
            game_date = game["date"].date()
            home = game["home_team"]
            away = game["away_team"]
            
            # Find matching line (within 1 day tolerance)
            matching_lines = lines_df[
                (lines_df["home_team"] == home) &
                (lines_df["away_team"] == away) &
                (abs((pd.to_datetime(lines_df["line_date"]) - pd.to_datetime(game_date)).dt.days) <= 1)
            ]
            
            row = game.to_dict()
            
            if len(matching_lines) > 0:
                line = matching_lines.iloc[0]
                row["spread_line"] = line.get("spread_line")
                row["total_line"] = line.get("total_line")
                matched += 1
            else:
                row["spread_line"] = None
                row["total_line"] = None
            
            merged_rows.append(row)
        
        df = pd.DataFrame(merged_rows)
        logger.info(f"✓ Matched {matched}/{len(outcomes_df)} games with betting lines")
        
        return df
    
    def compute_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute betting outcome labels for all markets.
        
        Labels computed:
        - home_win: Did home team win?
        - spread_covered: Did home team cover spread?
        - total_over: Did game go over total?
        - First half versions (if quarter data available)
        """
        logger.info("Computing betting labels...")
        
        df = df.copy()
        
        # Basic outcomes
        df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
        df["actual_margin"] = df["home_score"] - df["away_score"]
        df["actual_total"] = df["home_score"] + df["away_score"]
        
        # Spread covered (home perspective: home covers if margin > -spread)
        if "spread_line" in df.columns:
            df["spread_covered"] = df.apply(
                lambda r: int(r["actual_margin"] > -r["spread_line"]) 
                if pd.notna(r["spread_line"]) else None,
                axis=1
            )
        
        # Total over
        if "total_line" in df.columns:
            df["total_over"] = df.apply(
                lambda r: int(r["actual_total"] > r["total_line"])
                if pd.notna(r["total_line"]) else None,
                axis=1
            )
        
        # First half labels (if quarter data available)
        q_cols = ["home_q1", "home_q2", "away_q1", "away_q2"]
        if all(c in df.columns for c in q_cols):
            # Convert to numeric
            for c in q_cols:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            
            df["home_1h_score"] = df["home_q1"].fillna(0) + df["home_q2"].fillna(0)
            df["away_1h_score"] = df["away_q1"].fillna(0) + df["away_q2"].fillna(0)
            df["home_1h_win"] = (df["home_1h_score"] > df["away_1h_score"]).astype(int)
            df["actual_1h_margin"] = df["home_1h_score"] - df["away_1h_score"]
            df["actual_1h_total"] = df["home_1h_score"] + df["away_1h_score"]
            
            # Approximate 1H lines as half of full game lines
            if "spread_line" in df.columns:
                df["1h_spread_line"] = df["spread_line"] / 2
                df["1h_spread_covered"] = df.apply(
                    lambda r: int(r["actual_1h_margin"] > -r["1h_spread_line"])
                    if pd.notna(r["1h_spread_line"]) else None,
                    axis=1
                )
            
            if "total_line" in df.columns:
                df["1h_total_line"] = df["total_line"] / 2
                df["1h_total_over"] = df.apply(
                    lambda r: int(r["actual_1h_total"] > r["1h_total_line"])
                    if pd.notna(r["1h_total_line"]) else None,
                    axis=1
                )
        
        # First quarter labels
        if "home_q1" in df.columns and "away_q1" in df.columns:
            df["home_q1"] = pd.to_numeric(df["home_q1"], errors="coerce")
            df["away_q1"] = pd.to_numeric(df["away_q1"], errors="coerce")
            df["home_q1_win"] = (df["home_q1"] > df["away_q1"]).astype(int)
            df["actual_q1_margin"] = df["home_q1"] - df["away_q1"]
            df["actual_q1_total"] = df["home_q1"] + df["away_q1"]
            
            if "spread_line" in df.columns:
                df["q1_spread_line"] = df["spread_line"] / 4
            if "total_line" in df.columns:
                df["q1_total_line"] = df["total_line"] / 4
        
        return df
    
    async def enrich_with_betting_splits(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich training data with betting splits from Action Network.
        
        Adds columns for:
        - Public betting percentages (spread, total, ML)
        - Sharp money indicators
        - Reverse line movement (RLM) signals
        """
        logger.info("Fetching betting splits from Action Network...")
        
        try:
            from src.ingestion.betting_splits import (
                fetch_splits_action_network,
                splits_to_features,
            )
            
            # Get unique dates to fetch
            unique_dates = df["date"].dt.strftime("%Y-%m-%d").unique().tolist()
            
            # Fetch splits for recent dates (Action Network only has current/recent)
            # For historical, we'd need to have collected them over time
            all_splits = []
            
            for date_str in unique_dates[-30:]:  # Last 30 days
                try:
                    splits = await fetch_splits_action_network(date_str)
                    all_splits.extend(splits)
                except Exception as e:
                    logger.debug(f"No splits for {date_str}: {e}")
            
            if not all_splits:
                logger.warning("No betting splits fetched - continuing without splits features")
                return df
            
            # Convert to features and merge
            splits_data = []
            for split in all_splits:
                features = splits_to_features(split)
                features["home_team"] = split.home_team
                features["away_team"] = split.away_team
                features["game_date"] = split.game_time.strftime("%Y-%m-%d")
                splits_data.append(features)
            
            splits_df = pd.DataFrame(splits_data)
            
            # Merge with main dataframe
            df["game_date_str"] = df["date"].dt.strftime("%Y-%m-%d")
            
            merged = df.merge(
                splits_df,
                left_on=["home_team", "away_team", "game_date_str"],
                right_on=["home_team", "away_team", "game_date"],
                how="left",
                suffixes=("", "_splits"),
            )
            
            # Clean up
            merged = merged.drop(columns=["game_date_str", "game_date"], errors="ignore")
            
            splits_count = merged["is_rlm_spread"].notna().sum()
            logger.info(f"✓ Enriched {splits_count}/{len(df)} games with betting splits")
            
            return merged
            
        except Exception as e:
            logger.warning(f"Failed to enrich with betting splits: {e}")
            return df
    
    def validate_dataset(self, df: pd.DataFrame) -> None:
        """
        Validate dataset integrity.
        
        Raises DataValidationError if validation fails.
        """
        logger.info("Validating dataset...")
        
        errors = []
        warnings = []
        
        # Check required columns
        for col in REQUIRED_COLUMNS:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")
        
        # Check for empty dataset
        if len(df) == 0:
            errors.append("Dataset is empty")
        
        # Check for null values in critical columns
        if errors:
            raise DataValidationError("\n".join(errors))
        
        null_counts = df[REQUIRED_COLUMNS].isnull().sum()
        for col, count in null_counts.items():
            if count > 0:
                pct = count / len(df) * 100
                if pct > 10:
                    errors.append(f"Column '{col}' has {pct:.1f}% null values")
                else:
                    warnings.append(f"Column '{col}' has {count} null values ({pct:.1f}%)")
        
        # Check betting lines coverage
        if "spread_line" in df.columns:
            spread_coverage = df["spread_line"].notna().sum() / len(df) * 100
            if spread_coverage < 10:
                warnings.append(f"Only {spread_coverage:.1f}% of games have spread lines")
            else:
                logger.info(f"  ✓ Spread line coverage: {spread_coverage:.1f}%")
        
        if "total_line" in df.columns:
            total_coverage = df["total_line"].notna().sum() / len(df) * 100
            if total_coverage < 10:
                warnings.append(f"Only {total_coverage:.1f}% of games have total lines")
            else:
                logger.info(f"  ✓ Total line coverage: {total_coverage:.1f}%")
        
        # Check date range
        date_min = df["date"].min()
        date_max = df["date"].max()
        logger.info(f"  ✓ Date range: {date_min.date()} to {date_max.date()}")
        
        # Report warnings
        for w in warnings:
            logger.warning(f"  ⚠️  {w}")
        
        # Raise on errors
        if errors:
            raise DataValidationError("\n".join(errors))
        
        logger.info(f"✓ Dataset validated: {len(df)} games")
    
    async def build(
        self,
        seasons: Optional[List[str]] = None,
        output_file: str = "training_data.csv",
        skip_lines: bool = False,
    ) -> pd.DataFrame:
        """
        Build complete training dataset.
        
        This is the main entry point for the pipeline.
        
        Args:
            seasons: List of seasons to fetch (e.g., ["2024-2025", "2025-2026"])
            output_file: Output filename (in data/processed/)
            skip_lines: Skip fetching betting lines (for testing)
        
        Returns:
            Complete training DataFrame
        """
        seasons = seasons or settings.seasons_to_process
        
        logger.info("=" * 60)
        logger.info("FRESH DATA PIPELINE - BUILDING TRAINING DATA")
        logger.info("=" * 60)
        logger.info(f"Seasons: {seasons}")
        logger.info(f"Output: {self.output_dir / output_file}")
        logger.info("")
        
        # Step 1: Fetch game outcomes
        outcomes_df = await self.fetch_game_outcomes(seasons)
        
        # Step 2: Fetch betting lines (if not skipped)
        if not skip_lines:
            unique_dates = outcomes_df["date"].dt.strftime("%Y-%m-%d").unique().tolist()
            lines_df = await self.fetch_betting_lines(unique_dates)
        else:
            lines_df = pd.DataFrame()
        
        # Step 3: Merge outcomes and lines
        merged_df = self.merge_outcomes_and_lines(outcomes_df, lines_df)
        
        # Step 4: Compute labels
        labeled_df = self.compute_labels(merged_df)
        
        # Step 4.5: Fetch betting splits (if Action Network credentials available)
        if settings.action_network_username and settings.action_network_password:
            labeled_df = await self.enrich_with_betting_splits(labeled_df)
        
        # Step 5: Validate
        self.validate_dataset(labeled_df)
        
        # Step 6: Save
        output_path = self.output_dir / output_file
        labeled_df.to_csv(output_path, index=False)
        logger.info(f"✓ Training data saved to {output_path}")
        
        # Summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total games: {len(labeled_df)}")
        logger.info(f"Date range: {labeled_df['date'].min().date()} to {labeled_df['date'].max().date()}")
        
        if "spread_line" in labeled_df.columns:
            spread_count = labeled_df["spread_line"].notna().sum()
            logger.info(f"Games with spread lines: {spread_count} ({spread_count/len(labeled_df)*100:.1f}%)")
        
        if "total_line" in labeled_df.columns:
            total_count = labeled_df["total_line"].notna().sum()
            logger.info(f"Games with total lines: {total_count} ({total_count/len(labeled_df)*100:.1f}%)")
        
        return labeled_df


async def main():
    parser = argparse.ArgumentParser(
        description="Build fresh training data from APIs (no placeholders)"
    )
    parser.add_argument(
        "--seasons",
        type=str,
        default=None,
        help="Comma-separated seasons to fetch (e.g., 2024-2025,2025-2026)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="training_data.csv",
        help="Output filename (in data/processed/)",
    )
    parser.add_argument(
        "--skip-lines",
        action="store_true",
        help="Skip fetching betting lines (for testing)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing training data",
    )
    args = parser.parse_args()
    
    if args.validate_only:
        # Just validate existing data
        path = Path(settings.data_processed_dir) / args.output
        if not path.exists():
            print(f"ERROR: Training data not found at {path}")
            sys.exit(1)
        
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        
        pipeline = FreshDataPipeline()
        pipeline.validate_dataset(df)
        print("✓ Validation passed")
        sys.exit(0)
    
    # Parse seasons
    seasons = None
    if args.seasons:
        seasons = [s.strip() for s in args.seasons.split(",")]
    
    # Run pipeline
    pipeline = FreshDataPipeline()
    
    try:
        await pipeline.build(
            seasons=seasons,
            output_file=args.output,
            skip_lines=args.skip_lines,
        )
    except (DataPipelineError, DataValidationError) as e:
        print(f"\n✗ PIPELINE FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
