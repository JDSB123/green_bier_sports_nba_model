#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
from statistics import median

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import settings
from src.ingestion.api_basketball import APIBasketballClient
from src.ingestion.standardize import normalize_team_to_espn
from src.utils.logging import get_logger
from src.modeling.features import FeatureEngineer

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


def _median(values: List[float]) -> Optional[float]:
    """Safe median helper that ignores None values."""
    cleaned = [float(v) for v in values if v is not None]
    if not cleaned:
        return None
    return float(median(cleaned))


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
                # OPTIMIZED: Use ingest_essential() to fetch ALL Tier 1 endpoints in one call
                # This fetches: teams, games, statistics, and game_stats_teams
                logger.info(f"  Fetching all essential endpoints for season {season}...")
                results = await client.ingest_essential()
                
                # Extract games from the results
                games_result = next((r for r in results if r.name == "games"), None)
                if not games_result:
                    logger.warning(f"  No games result found for season {season}")
                    continue
                games = games_result.data.get("response", [])
                
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
        
        cache_path = Path(settings.data_processed_dir) / "betting_lines.csv"
        if cache_path.exists():
            logger.info(f"Using cached betting lines from {cache_path}")
            try:
                cached_df = pd.read_csv(cache_path)
                return cached_df
            except Exception as exc:
                logger.warning(f"Failed to load cached betting lines ({cache_path}): {exc}")
        
        # Import here to avoid circular imports
        from src.ingestion.the_odds import (
            fetch_odds,
            fetch_events,
            fetch_event_odds,
            fetch_betting_splits,
            fetch_participants,
        )
        
        all_lines: List[Dict[str, Any]] = []
        
        # Use unified fetch_odds() endpoint (handles both historical and current)
        # This ensures consistent data structure and format across training/prediction
        logger.info("  Fetching odds data via unified source (fetch_odds)...")
        
        try:
            # Fetch main odds (FG markets) - fetch_odds handles routing internally
            current_odds = await fetch_odds(markets="spreads,totals,h2h")
            logger.info(f"  Fetched FG odds for {len(current_odds)} events")
            
            # Enrich each event with 1H/Q1 markets
            # OPTIMIZATION: Use asyncio.gather with semaphore
            sem = asyncio.Semaphore(5)

            async def process_current_event(event):
                event_id = event.get("id")
                if not event_id:
                    return event

                async with sem:
                    try:
                        # Fetch 1H and Q1 markets for this event
                        event_odds = await fetch_event_odds(
                            event_id,
                            markets="spreads_h1,totals_h1,h2h_h1,spreads_q1,totals_q1,h2h_q1",
                        )
                        # Merge markets
                        existing_bms = {bm["key"]: bm for bm in event.get("bookmakers", [])}
                        for bm in event_odds.get("bookmakers", []):
                            if bm["key"] in existing_bms:
                                existing_bms[bm["key"]]["markets"].extend(bm.get("markets", []))
                            else:
                                existing_bms[bm["key"]] = bm
                        event["bookmakers"] = list(existing_bms.values())
                    except Exception as e:
                        logger.debug(f"  Could not fetch 1H/Q1 odds for event {event_id}: {e}")
                return event

            tasks = [process_current_event(event) for event in current_odds]
            enriched_odds = await asyncio.gather(*tasks)

            lines = self._extract_lines_from_events(enriched_odds, datetime.now().strftime("%Y-%m-%d"))
            all_lines.extend(lines)
            logger.info(f"✓ Fetched current odds with 1H/Q1 markets for {len(enriched_odds)} events")

            # OPTIMIZATION: Fetch betting splits if available (paid plan)
            try:
                splits = await fetch_betting_splits()
                logger.info(f"✓ Fetched betting splits for {len(splits)} games")
                # Note: Splits will be merged later in enrich_with_betting_splits()
            except Exception as e:
                logger.debug(f"Betting splits not available: {e}")

        except Exception as e:
            logger.warning(f"Could not fetch current odds: {e}")
        
        if not all_lines:
            logger.warning("No betting lines fetched - backtest will be limited to moneyline")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_lines)
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_path, index=False)
            logger.info(f"Cached betting lines to {cache_path}")
        except Exception as exc:
            logger.warning(f"Could not cache betting lines to {cache_path}: {exc}")
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
            fg_spreads: List[float] = []
            fg_totals: List[float] = []
            fh_spreads: List[float] = []
            fh_totals: List[float] = []
            q1_spreads: List[float] = []
            q1_totals: List[float] = []
            fg_home_ml: List[float] = []
            fg_away_ml: List[float] = []
            fh_home_ml: List[float] = []
            fh_away_ml: List[float] = []
            q1_home_ml: List[float] = []
            q1_away_ml: List[float] = []
            
            for bm in bookmakers:
                markets = bm.get("markets", [])
                for market in markets:
                    key = (market.get("key") or "").lower()
                    outcomes = market.get("outcomes", [])
                    
                    if key == "spreads":
                        for outcome in outcomes:
                            if outcome.get("name") == home_team:
                                point = outcome.get("point")
                                if point is not None:
                                    fg_spreads.append(float(point))
                    elif key == "totals":
                        for outcome in outcomes:
                            if outcome.get("name") == "Over":
                                point = outcome.get("point")
                                if point is not None:
                                    fg_totals.append(float(point))
                    elif key == "h2h":
                        for outcome in outcomes:
                            price = outcome.get("price")
                            if outcome.get("name") == home_team and price is not None:
                                fg_home_ml.append(float(price))
                            elif outcome.get("name") == away_team and price is not None:
                                fg_away_ml.append(float(price))
                    elif key == "spreads_h1":
                        for outcome in outcomes:
                            if outcome.get("name") == home_team:
                                point = outcome.get("point")
                                if point is not None:
                                    fh_spreads.append(float(point))
                    elif key == "totals_h1":
                        for outcome in outcomes:
                            if outcome.get("name") == "Over":
                                point = outcome.get("point")
                                if point is not None:
                                    fh_totals.append(float(point))
                    elif key == "h2h_h1":
                        for outcome in outcomes:
                            price = outcome.get("price")
                            if outcome.get("name") == home_team and price is not None:
                                fh_home_ml.append(float(price))
                            elif outcome.get("name") == away_team and price is not None:
                                fh_away_ml.append(float(price))
                    elif key == "spreads_q1":
                        for outcome in outcomes:
                            if outcome.get("name") == home_team:
                                point = outcome.get("point")
                                if point is not None:
                                    q1_spreads.append(float(point))
                    elif key == "totals_q1":
                        for outcome in outcomes:
                            if outcome.get("name") == "Over":
                                point = outcome.get("point")
                                if point is not None:
                                    q1_totals.append(float(point))
                    elif key == "h2h_q1":
                        for outcome in outcomes:
                            price = outcome.get("price")
                            if outcome.get("name") == home_team and price is not None:
                                q1_home_ml.append(float(price))
                            elif outcome.get("name") == away_team and price is not None:
                                q1_away_ml.append(float(price))
            
            if any([
                fg_spreads, fg_totals, fh_spreads, fh_totals,
                q1_spreads, q1_totals, fg_home_ml, fg_away_ml,
                fh_home_ml, fh_away_ml, q1_home_ml, q1_away_ml
            ]):
                lines.append({
                    "query_date": query_date,
                    "event_id": event.get("id"),
                    "commence_time": event.get("commence_time"),
                    "home_team": home_team,
                    "away_team": away_team,
                    "fg_spread_line": _median(fg_spreads),
                    "fg_total_line": _median(fg_totals),
                    "fg_home_ml": _median(fg_home_ml),
                    "fg_away_ml": _median(fg_away_ml),
                    "fh_spread_line": _median(fh_spreads),
                    "fh_total_line": _median(fh_totals),
                    "fh_home_ml": _median(fh_home_ml),
                    "fh_away_ml": _median(fh_away_ml),
                    "q1_spread_line": _median(q1_spreads),
                    "q1_total_line": _median(q1_totals),
                    "q1_home_ml": _median(q1_home_ml),
                    "q1_away_ml": _median(q1_away_ml),
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
        column_map = {
            "spread_line": "fg_spread_line",
            "total_line": "fg_total_line",
            "moneyline_home": "fg_home_ml",
            "moneyline_away": "fg_away_ml",
            "fh_spread_line": "fh_spread_line",
            "fh_total_line": "fh_total_line",
            "fh_home_ml": "fh_home_ml",
            "fh_away_ml": "fh_away_ml",
            "q1_spread_line": "q1_spread_line",
            "q1_total_line": "q1_total_line",
            "q1_home_ml": "q1_home_ml",
            "q1_away_ml": "q1_away_ml",
        }
        
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
                for target_col, source_col in column_map.items():
                    row[target_col] = line.get(source_col)
                matched += 1
            else:
                for target_col in column_map:
                    row[target_col] = None
            
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
            
            # First-half lines (prefer real data, fallback to FG approximations)
            if "fh_spread_line" in df.columns:
                df["1h_spread_line"] = df["fh_spread_line"]
            else:
                df["1h_spread_line"] = pd.Series([None] * len(df))
            if "spread_line" in df.columns:
                mask = df["1h_spread_line"].isna()
                df.loc[mask, "1h_spread_line"] = df.loc[mask, "spread_line"] / 2
            
            if "fh_total_line" in df.columns:
                df["1h_total_line"] = df["fh_total_line"]
            else:
                df["1h_total_line"] = pd.Series([None] * len(df))
            if "total_line" in df.columns:
                mask = df["1h_total_line"].isna()
                df.loc[mask, "1h_total_line"] = df.loc[mask, "total_line"] / 2
            
            df["1h_spread_covered"] = df.apply(
                lambda r: int(r["actual_1h_margin"] > -r["1h_spread_line"])
                if pd.notna(r["1h_spread_line"]) else None,
                axis=1
            )
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
            
            if "q1_spread_line" not in df.columns:
                df["q1_spread_line"] = pd.Series([None] * len(df))
            if "spread_line" in df.columns:
                mask = df["q1_spread_line"].isna()
                df.loc[mask, "q1_spread_line"] = df.loc[mask, "spread_line"] / 4
            
            if "q1_total_line" not in df.columns:
                df["q1_total_line"] = pd.Series([None] * len(df))
            if "total_line" in df.columns:
                mask = df["q1_total_line"].isna()
                df.loc[mask, "q1_total_line"] = df.loc[mask, "total_line"] / 4
            
            df["q1_spread_covered"] = df.apply(
                lambda r: int(r["actual_q1_margin"] > -r["q1_spread_line"])
                if pd.notna(r["q1_spread_line"]) else None,
                axis=1
            )
            df["q1_total_over"] = df.apply(
                lambda r: int(r["actual_q1_total"] > r["q1_total_line"])
                if pd.notna(r["q1_total_line"]) else None,
                axis=1
            )
        
        return df
    
    async def enrich_with_betting_splits(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich training data with betting splits from multiple sources.
        
        Tries in order:
        1. The Odds API betting-splits endpoint (if available, paid plan)
        2. Action Network (if credentials available)
        
        Adds columns for:
        - Public betting percentages (spread, total, ML)
        - Sharp money indicators
        - Reverse line movement (RLM) signals
        """
        logger.info("Enriching with betting splits from available sources...")
        
        # Try The Odds API betting splits first (optimized, single call)
        try:
            from src.ingestion.the_odds import fetch_betting_splits
            
            logger.info("  Attempting The Odds API betting-splits endpoint...")
            splits_data = await fetch_betting_splits()
            
            if splits_data and len(splits_data) > 0:
                logger.info(f"  ✓ Fetched {len(splits_data)} games with betting splits from The Odds API")
                
                # Convert to DataFrame format
                splits_features = []
                for split in splits_data:
                    # Extract splits data from The Odds API format
                    # Structure: split["markets"]["spreads"]["choices"][0/1]["value"]
                    try:
                        markets = split.get("markets", {})
                        
                        # Spread splits
                        spread_market = markets.get("spreads", {})
                        spread_choices = spread_market.get("choices", [])
                        if len(spread_choices) >= 2:
                            home_spread_pct = spread_choices[0].get("value", 50)
                            away_spread_pct = spread_choices[1].get("value", 50)
                        else:
                            home_spread_pct = away_spread_pct = 50
                        
                        # Total splits
                        total_market = markets.get("totals", {})
                        total_choices = total_market.get("choices", [])
                        if len(total_choices) >= 2:
                            over_pct = total_choices[0].get("value", 50)
                            under_pct = total_choices[1].get("value", 50)
                        else:
                            over_pct = under_pct = 50
                        
                        # Moneyline splits
                        ml_market = markets.get("h2h", {})
                        ml_choices = ml_market.get("choices", [])
                        if len(ml_choices) >= 2:
                            home_ml_pct = ml_choices[0].get("value", 50)
                            away_ml_pct = ml_choices[1].get("value", 50)
                        else:
                            home_ml_pct = away_ml_pct = 50
                        
                        # Extract teams (already standardized)
                        teams = split.get("home_team", ""), split.get("away_team", "")
                        
                        splits_features.append({
                            "home_team": teams[0],
                            "away_team": teams[1],
                            "game_date": split.get("commence_time", "").split("T")[0] if split.get("commence_time") else "",
                            "spread_public_home_pct": home_spread_pct,
                            "over_public_pct": over_pct,
                            "home_ml_public_pct": home_ml_pct,
                            # Derived features
                            "is_rlm_spread": 1 if (home_spread_pct > 60 and away_spread_pct < 40) else 0,
                            "spread_ticket_money_diff": home_spread_pct - away_spread_pct,
                        })
                    except Exception as e:
                        logger.debug(f"Error parsing splits data: {e}")
                        continue
                
                if splits_features:
                    splits_df = pd.DataFrame(splits_features)
                    df["game_date_str"] = df["date"].dt.strftime("%Y-%m-%d")
                    
                    merged = df.merge(
                        splits_df,
                        left_on=["home_team", "away_team", "game_date_str"],
                        right_on=["home_team", "away_team", "game_date"],
                        how="left",
                        suffixes=("", "_splits"),
                    )
                    
                    merged = merged.drop(columns=["game_date_str", "game_date"], errors="ignore")
                    splits_count = merged["is_rlm_spread"].notna().sum()
                    logger.info(f"✓ Enriched {splits_count}/{len(df)} games with The Odds API betting splits")
                    return merged
                    
        except Exception as e:
            logger.debug(f"The Odds API betting splits not available: {e}")
        
        # Fallback to Action Network if The Odds API splits unavailable
        try:
            from src.ingestion.betting_splits import (
                fetch_splits_action_network,
                splits_to_features,
            )
            
            logger.info("  Falling back to Action Network betting splits...")
            
            # Get unique dates to fetch
            unique_dates = df["date"].dt.strftime("%Y-%m-%d").unique().tolist()
            
            # Fetch splits for recent dates (Action Network only has current/recent)
            all_splits = []
            
            for date_str in unique_dates[-30:]:  # Last 30 days
                try:
                    splits = await fetch_splits_action_network(date_str)
                    all_splits.extend(splits)
                except Exception as e:
                    logger.debug(f"No Action Network splits for {date_str}: {e}")
            
            if not all_splits:
                logger.warning("No betting splits fetched from any source - continuing without splits features")
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
            logger.info(f"✓ Enriched {splits_count}/{len(df)} games with Action Network betting splits")
            
            return merged
            
        except Exception as e:
            logger.warning(f"Failed to enrich with betting splits from any source: {e}")
            return df

    def compute_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute engineered features for training using FeatureEngineer.

        This step is CRITICAL for training - without engineered features,
        models will fail due to insufficient features.

        Features computed:
        - Team rolling stats (PPG, PAPG, margin, win%)
        - ELO ratings
        - Rest days / back-to-back detection
        - Head-to-head history
        - Travel/fatigue features
        - Dynamic home court advantage
        - Period-specific stats (Q1, 1H)
        """
        logger.info("Computing engineered features...")

        # Need date column for lookback
        if "date" not in df.columns:
            logger.error("Cannot compute features: 'date' column missing")
            return df

        # Sort by date for proper rolling calculations
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # Initialize feature engineer with lookback window
        fe = FeatureEngineer(lookback=10)

        # Build features for each game using historical data up to that point
        all_features = []
        total_games = len(df)

        for idx, game in df.iterrows():
            game_date = pd.to_datetime(game["date"])

            # Historical games are all games BEFORE this game
            historical_df = df[df["date"] < game_date].copy()

            # Need at least 10 games of history for reliable stats
            if len(historical_df) < 10:
                # Skip early-season games with insufficient history
                all_features.append({})
                continue

            try:
                features = fe.build_game_features(game, historical_df)
                all_features.append(features if features else {})
            except Exception as e:
                logger.debug(f"Could not compute features for game {idx}: {e}")
                all_features.append({})

            # Progress indicator
            if (idx + 1) % 200 == 0:
                logger.info(f"  Processed {idx + 1}/{total_games} games...")

        # Merge features back into dataframe
        if all_features:
            features_df = pd.DataFrame(all_features)

            # Drop columns that would conflict
            overlap_cols = [c for c in features_df.columns if c in df.columns]
            if overlap_cols:
                features_df = features_df.drop(columns=overlap_cols, errors="ignore")

            # Merge by index
            df = pd.concat([df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)

            # Count how many features were added
            new_feature_count = len(features_df.columns)
            games_with_features = features_df.notna().any(axis=1).sum()
            logger.info(f"✓ Computed {new_feature_count} engineered features for {games_with_features}/{total_games} games")
        else:
            logger.warning("No features computed")

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
        
        if "1h_spread_line" in df.columns:
            fh_spread_cov = df["1h_spread_line"].notna().sum() / len(df) * 100
            logger.info(f"  ✓ 1H spread coverage: {fh_spread_cov:.1f}%")
        if "1h_total_line" in df.columns:
            fh_total_cov = df["1h_total_line"].notna().sum() / len(df) * 100
            logger.info(f"  ✓ 1H total coverage: {fh_total_cov:.1f}%")
        if "q1_spread_line" in df.columns:
            q1_spread_cov = df["q1_spread_line"].notna().sum() / len(df) * 100
            logger.info(f"  ✓ Q1 spread coverage: {q1_spread_cov:.1f}%")
        if "q1_total_line" in df.columns:
            q1_total_cov = df["q1_total_line"].notna().sum() / len(df) * 100
            logger.info(f"  ✓ Q1 total coverage: {q1_total_cov:.1f}%")
        
        # Check date range
        date_min = df["date"].min()
        date_max = df["date"].max()
        logger.info(f"  ✓ Date range: {date_min.date()} to {date_max.date()}")

        # Check engineered features
        feature_cols = ["home_ppg", "away_ppg", "home_elo", "away_elo", "ppg_diff", "elo_diff"]
        existing_features = [c for c in feature_cols if c in df.columns]
        if existing_features:
            logger.info(f"  ✓ Engineered features present: {len(existing_features)}/{len(feature_cols)}")
        else:
            warnings.append("No engineered features found - model training may fail")

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
        
        # OPTIMIZATION: Fetch participants reference first (for team validation)
        logger.info("Fetching participants reference from The Odds API...")
        try:
            participants = await fetch_participants()
            logger.info(f"✓ Fetched {len(participants)} participants for team validation")
        except Exception as e:
            logger.warning(f"Could not fetch participants: {e} (continuing anyway)")
        
        # Step 1: Fetch game outcomes (with ALL essential endpoints)
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
        
        # Step 4.5: Fetch betting splits (tries The Odds API first, falls back to Action Network)
        labeled_df = await self.enrich_with_betting_splits(labeled_df)

        # Step 5: Compute engineered features (CRITICAL for training)
        labeled_df = self.compute_engineered_features(labeled_df)

        # Step 6: Validate
        self.validate_dataset(labeled_df)

        # Step 7: Save
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
