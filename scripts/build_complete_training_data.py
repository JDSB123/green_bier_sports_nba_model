#!/usr/bin/env python3
"""
Build COMPLETE training dataset with ALL available features.

Data Sources:
- Kaggle betting: Q1-Q4 scores, FG lines, moneylines
- The Odds API: FG + 1H lines (multi-bookmaker consensus)
- The Odds API Historical: Fill gaps via API calls
- FiveThirtyEight: ELO ratings
- API-Basketball: Team stats (if available)

Features:
- Rolling team stats (PPG, PAPG, margins)
- ELO ratings and differentials
- Home/away splits
- Rest days / back-to-back detection
- Season progression
- Win streaks
- Head-to-head history

Usage:
    python scripts/build_complete_training_data.py --start-date 2023-01-01
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Paths
DATA_DIR = PROJECT_ROOT / "data"
KAGGLE_FILE = DATA_DIR / "external" / "kaggle" / "nba_2008-2025.csv"
THEODDS_DIR = DATA_DIR / "historical" / "the_odds"
ELO_FILE = DATA_DIR / "raw" / "github" / "raw.githubusercontent.com_fivethirtyeight_data_master_nba-elo_nbaallelo.csv"
OUTPUT_DIR = DATA_DIR / "processed"

# Team abbreviation mapping
TEAM_ABBREV_MAP = {
    "atl": "Atlanta Hawks", "bos": "Boston Celtics", "bkn": "Brooklyn Nets",
    "cha": "Charlotte Hornets", "chi": "Chicago Bulls", "cle": "Cleveland Cavaliers",
    "dal": "Dallas Mavericks", "den": "Denver Nuggets", "det": "Detroit Pistons",
    "gs": "Golden State Warriors", "gsw": "Golden State Warriors",
    "hou": "Houston Rockets", "ind": "Indiana Pacers",
    "lac": "Los Angeles Clippers", "lal": "Los Angeles Lakers",
    "mem": "Memphis Grizzlies", "mia": "Miami Heat", "mil": "Milwaukee Bucks",
    "min": "Minnesota Timberwolves", "no": "New Orleans Pelicans",
    "nop": "New Orleans Pelicans", "nyk": "New York Knicks", "ny": "New York Knicks",
    "okc": "Oklahoma City Thunder", "orl": "Orlando Magic",
    "phi": "Philadelphia 76ers", "phx": "Phoenix Suns",
    "por": "Portland Trail Blazers", "sac": "Sacramento Kings",
    "sa": "San Antonio Spurs", "sas": "San Antonio Spurs",
    "tor": "Toronto Raptors", "utah": "Utah Jazz", "uta": "Utah Jazz",
    "was": "Washington Wizards", "wsh": "Washington Wizards",
    "nj": "Brooklyn Nets", "njn": "Brooklyn Nets", "sea": "Oklahoma City Thunder",
    "noh": "New Orleans Pelicans", "nok": "New Orleans Pelicans",
}


def map_team_name(abbrev: str) -> str:
    """Map team abbreviation to full name."""
    return TEAM_ABBREV_MAP.get(abbrev.lower().strip(), abbrev)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_kaggle_data(start_date: str) -> pd.DataFrame:
    """Load Kaggle NBA data with Q1-Q4 scores and betting lines."""
    if not KAGGLE_FILE.exists():
        logger.error(f"Kaggle file not found: {KAGGLE_FILE}")
        return pd.DataFrame()
    
    df = pd.read_csv(KAGGLE_FILE)
    df["game_date"] = pd.to_datetime(df["date"])
    df = df[df["game_date"] >= start_date].copy()
    
    # Map team names
    df["home_team"] = df["home"].apply(map_team_name)
    df["away_team"] = df["away"].apply(map_team_name)
    
    # Scores
    df["home_score"] = df["score_home"]
    df["away_score"] = df["score_away"]
    df["fg_margin"] = df["home_score"] - df["away_score"]
    df["fg_total_actual"] = df["home_score"] + df["away_score"]
    
    # Q1-Q4 scores
    for q in [1, 2, 3, 4]:
        df[f"home_q{q}"] = df[f"q{q}_home"]
        df[f"away_q{q}"] = df[f"q{q}_away"]
    
    # 1H scores
    df["home_1h_score"] = df["home_q1"].fillna(0) + df["home_q2"].fillna(0)
    df["away_1h_score"] = df["away_q1"].fillna(0) + df["away_q2"].fillna(0)
    df["1h_total_actual"] = df["home_1h_score"] + df["away_1h_score"]
    df["1h_margin"] = df["home_1h_score"] - df["away_1h_score"]
    
    # FG betting lines (properly signed)
    df["fg_spread_line"] = df.apply(
        lambda r: -r["spread"] if r["whos_favored"] == "home" else r["spread"], axis=1
    )
    df["fg_total_line"] = df["total"]
    df["fg_ml_home"] = df["moneyline_home"]
    df["fg_ml_away"] = df["moneyline_away"]
    
    # Match keys
    df["date_str"] = df["game_date"].dt.strftime("%Y-%m-%d")
    df["match_key"] = (
        df["date_str"] + "_" +
        df["home_team"].str.lower().str.strip() + "_" +
        df["away_team"].str.lower().str.strip()
    )
    
    logger.info(f"Loaded {len(df)} games from Kaggle (from {start_date})")
    return df


def load_theodds_odds(odds_dir: Path) -> pd.DataFrame:
    """Load ALL odds from The Odds API (FG markets)."""
    rows = []
    
    for season_dir in sorted(odds_dir.glob("*")):
        if not season_dir.is_dir():
            continue
        
        for json_file in sorted(season_dir.glob("*.json")):
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)
                
                events = data if isinstance(data, list) else data.get("data", [])
                
                for event in events:
                    home = event.get("home_team", "")
                    away = event.get("away_team", "")
                    commence = event.get("commence_time", "")
                    
                    if not all([home, away, commence]):
                        continue
                    
                    # Collect lines from all bookmakers
                    spreads = []
                    totals = []
                    ml_home = []
                    ml_away = []
                    
                    for bm in event.get("bookmakers", []):
                        for mkt in bm.get("markets", []):
                            key = mkt.get("key", "").lower()
                            outcomes = mkt.get("outcomes", [])
                            
                            if key == "spreads":
                                for o in outcomes:
                                    if o.get("name") == home and o.get("point") is not None:
                                        spreads.append(o.get("point"))
                            elif key == "totals":
                                for o in outcomes:
                                    if o.get("name") == "Over" and o.get("point") is not None:
                                        totals.append(o.get("point"))
                            elif key == "h2h":
                                for o in outcomes:
                                    if o.get("name") == home and o.get("price") is not None:
                                        ml_home.append(o.get("price"))
                                    elif o.get("name") == away and o.get("price") is not None:
                                        ml_away.append(o.get("price"))
                    
                    if spreads or totals:
                        rows.append({
                            "commence_time": commence,
                            "home_team": home,
                            "away_team": away,
                            "theodds_fg_spread": _safe_median(spreads),
                            "theodds_fg_total": _safe_median(totals),
                            "theodds_fg_ml_home": _safe_median(ml_home),
                            "theodds_fg_ml_away": _safe_median(ml_away),
                            "theodds_fg_bookmakers": len(event.get("bookmakers", [])),
                        })
            except Exception as e:
                logger.warning(f"Error reading {json_file}: {e}")
    
    if not rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    df["game_date"] = pd.to_datetime(df["commence_time"]).dt.tz_localize(None)
    df["date_str"] = df["game_date"].dt.strftime("%Y-%m-%d")
    df["match_key"] = (
        df["date_str"] + "_" +
        df["home_team"].str.lower().str.strip() + "_" +
        df["away_team"].str.lower().str.strip()
    )
    
    logger.info(f"Loaded {len(df)} FG odds from The Odds API")
    return df


def load_theodds_1h(period_dir: Path) -> pd.DataFrame:
    """Load 1H odds from The Odds API."""
    rows = []
    
    for season_dir in sorted(period_dir.glob("*")):
        if not season_dir.is_dir():
            continue
        
        for json_file in season_dir.glob("period_odds_1h.json"):
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)
                
                events = data.get("data", [])
                
                for event_wrapper in events:
                    event = event_wrapper.get("data", event_wrapper)
                    if not event:
                        continue
                    
                    home = event.get("home_team", "")
                    away = event.get("away_team", "")
                    commence = event.get("commence_time", "")
                    
                    if not all([home, away, commence]):
                        continue
                    
                    spreads_h1 = []
                    totals_h1 = []
                    ml_home_h1 = []
                    ml_away_h1 = []
                    
                    for bm in event.get("bookmakers", []):
                        for mkt in bm.get("markets", []):
                            key = mkt.get("key", "").lower()
                            outcomes = mkt.get("outcomes", [])
                            
                            if key == "spreads_h1":
                                for o in outcomes:
                                    if o.get("name") == home and o.get("point") is not None:
                                        spreads_h1.append(o.get("point"))
                            elif key == "totals_h1":
                                for o in outcomes:
                                    if o.get("name") == "Over" and o.get("point") is not None:
                                        totals_h1.append(o.get("point"))
                            elif key == "h2h_h1":
                                for o in outcomes:
                                    if o.get("name") == home and o.get("price") is not None:
                                        ml_home_h1.append(o.get("price"))
                                    elif o.get("name") == away and o.get("price") is not None:
                                        ml_away_h1.append(o.get("price"))
                    
                    rows.append({
                        "commence_time": commence,
                        "home_team": home,
                        "away_team": away,
                        "1h_spread_line": _safe_median(spreads_h1),
                        "1h_total_line": _safe_median(totals_h1),
                        "1h_ml_home": _safe_median(ml_home_h1),
                        "1h_ml_away": _safe_median(ml_away_h1),
                    })
            except Exception as e:
                logger.warning(f"Error reading {json_file}: {e}")
    
    if not rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    df["game_date"] = pd.to_datetime(df["commence_time"]).dt.tz_localize(None)
    df["date_str"] = df["game_date"].dt.strftime("%Y-%m-%d")
    df["match_key"] = (
        df["date_str"] + "_" +
        df["home_team"].str.lower().str.strip() + "_" +
        df["away_team"].str.lower().str.strip()
    )
    
    logger.info(f"Loaded {len(df)} 1H odds from The Odds API")
    return df


def load_elo_ratings() -> pd.DataFrame:
    """Load FiveThirtyEight ELO ratings."""
    if not ELO_FILE.exists():
        logger.warning(f"ELO file not found: {ELO_FILE}")
        return pd.DataFrame()
    
    df = pd.read_csv(ELO_FILE)
    df["date"] = pd.to_datetime(df["date_game"])
    
    # Rename columns to standard format
    df["team1"] = df["team_id"]
    df["team2"] = df["opp_id"]
    df["elo1_post"] = df["elo_n"]
    df["elo2_post"] = df["opp_elo_n"]
    
    # Filter to recent data
    df = df[df["date"] >= "2015-01-01"].copy()
    
    logger.info(f"Loaded {len(df)} ELO records from FiveThirtyEight")
    return df


def _safe_median(values: list) -> float | None:
    """Calculate median, handling None and empty lists."""
    cleaned = [v for v in values if v is not None]
    if not cleaned:
        return None
    return float(median(cleaned))


# =============================================================================
# DATA MERGING
# =============================================================================

def merge_all_data(
    kaggle_df: pd.DataFrame,
    theodds_fg_df: pd.DataFrame,
    theodds_1h_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all data sources with timezone-aware matching."""
    df = kaggle_df.copy()
    
    # Create alternate match key (next day) for timezone handling
    # TheOdds uses UTC, so late night US games appear as next day
    df["date_next"] = (df["game_date"] + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
    df["match_key_alt"] = (
        df["date_next"] + "_" +
        df["home_team"].str.lower().str.strip() + "_" +
        df["away_team"].str.lower().str.strip()
    )
    
    # Merge The Odds API FG lines (try both date and date+1)
    if not theodds_fg_df.empty:
        fg_cols = ["match_key", "theodds_fg_spread", "theodds_fg_total", 
                   "theodds_fg_ml_home", "theodds_fg_ml_away", "theodds_fg_bookmakers"]
        fg_dedup = theodds_fg_df[fg_cols].drop_duplicates(subset=["match_key"])
        
        # Primary match
        df = df.merge(fg_dedup, on="match_key", how="left")
        
        # Alternate match (for timezone offset)
        fg_alt = fg_dedup.copy()
        fg_alt.columns = [c + "_alt" if c != "match_key" else "match_key_alt" for c in fg_alt.columns]
        df = df.merge(fg_alt, on="match_key_alt", how="left")
        
        # Combine: use primary, fall back to alternate
        for col in ["theodds_fg_spread", "theodds_fg_total", "theodds_fg_ml_home", "theodds_fg_ml_away"]:
            if f"{col}_alt" in df.columns:
                df[col] = df[col].fillna(df[f"{col}_alt"])
        
        matched = df["theodds_fg_spread"].notna().sum()
        logger.info(f"Merged TheOdds FG lines: {matched}/{len(df)} ({matched/len(df)*100:.1f}%)")
        
        # Fill missing Kaggle lines with TheOdds
        for col in ["fg_spread_line", "fg_total_line", "fg_ml_home", "fg_ml_away"]:
            theodds_col = f"theodds_{col.replace('fg_', 'fg_')}"
            if theodds_col in df.columns:
                df[col] = df[col].fillna(df[theodds_col])
    
    # Merge 1H lines (try both date and date+1)
    if not theodds_1h_df.empty:
        h1_cols = ["match_key", "1h_spread_line", "1h_total_line", "1h_ml_home", "1h_ml_away"]
        h1_dedup = theodds_1h_df[h1_cols].drop_duplicates(subset=["match_key"])
        
        # Primary match
        df = df.merge(h1_dedup, on="match_key", how="left", suffixes=("", "_pri"))
        
        # Alternate match (for timezone offset)
        h1_alt = h1_dedup.copy()
        h1_alt.columns = [c + "_alt" if c != "match_key" else "match_key_alt" for c in h1_alt.columns]
        df = df.merge(h1_alt, on="match_key_alt", how="left")
        
        # Combine: use primary, fall back to alternate
        for col in ["1h_spread_line", "1h_total_line", "1h_ml_home", "1h_ml_away"]:
            if f"{col}_alt" in df.columns:
                df[col] = df[col].fillna(df[f"{col}_alt"])
        
        matched = df["1h_spread_line"].notna().sum()
        logger.info(f"Merged TheOdds 1H lines: {matched}/{len(df)} ({matched/len(df)*100:.1f}%)")
    else:
        df["1h_spread_line"] = np.nan
        df["1h_total_line"] = np.nan
        df["1h_ml_home"] = np.nan
        df["1h_ml_away"] = np.nan
    
    # Clean up temporary columns
    drop_cols = [c for c in df.columns if c.endswith("_alt") or c == "date_next" or c == "match_key_alt"]
    df = df.drop(columns=drop_cols, errors="ignore")
    
    return df


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def add_all_features(df: pd.DataFrame, elo_df: pd.DataFrame) -> pd.DataFrame:
    """Add ALL available features."""
    df = df.sort_values("game_date").reset_index(drop=True)
    
    # Basic temporal features
    df["season"] = df["game_date"].apply(
        lambda d: f"{d.year}-{d.year+1}" if d.month >= 10 else f"{d.year-1}-{d.year}"
    )
    df["day_of_week"] = df["game_date"].dt.dayofweek
    df["month"] = df["game_date"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    
    # Season progression (0 = start, 1 = end)
    df["season_day"] = df.groupby("season").cumcount()
    df["season_pct"] = df.groupby("season")["season_day"].transform(
        lambda x: x / x.max() if x.max() > 0 else 0
    )
    
    logger.info("  Added temporal features")
    
    # Rolling team statistics
    df = add_rolling_team_stats(df)
    logger.info("  Added rolling team stats")
    
    # Rest days and back-to-back detection
    df = add_rest_features(df)
    logger.info("  Added rest/B2B features")
    
    # Win streaks
    df = add_streak_features(df)
    logger.info("  Added streak features")
    
    # Head-to-head history
    df = add_h2h_features(df)
    logger.info("  Added H2H features")
    
    # ELO ratings
    if not elo_df.empty:
        df = add_elo_features(df, elo_df)
        logger.info("  Added ELO features")
    
    # Implied probabilities from moneylines
    df = add_implied_prob_features(df)
    logger.info("  Added implied probability features")
    
    return df


def add_rolling_team_stats(df: pd.DataFrame, windows: list = [5, 10, 20]) -> pd.DataFrame:
    """Add rolling team statistics."""
    
    # Create team game logs
    home_games = df[["game_date", "home_team", "home_score", "away_score", "fg_margin"]].copy()
    home_games.columns = ["game_date", "team", "pts_for", "pts_against", "margin"]
    home_games["is_home"] = 1
    
    away_games = df[["game_date", "away_team", "away_score", "home_score"]].copy()
    away_games.columns = ["game_date", "team", "pts_for", "pts_against"]
    away_games["margin"] = -df["fg_margin"]
    away_games["is_home"] = 0
    
    team_games = pd.concat([home_games, away_games]).sort_values("game_date")
    team_games["win"] = (team_games["margin"] > 0).astype(int)
    
    # Calculate rolling stats per team
    rolling_stats = {}
    for window in windows:
        for team in team_games["team"].unique():
            team_data = team_games[team_games["team"] == team].copy()
            
            # Shift to avoid leakage (use only prior games)
            team_data[f"ppg_{window}"] = team_data["pts_for"].shift(1).rolling(window, min_periods=3).mean()
            team_data[f"papg_{window}"] = team_data["pts_against"].shift(1).rolling(window, min_periods=3).mean()
            team_data[f"margin_{window}"] = team_data["margin"].shift(1).rolling(window, min_periods=3).mean()
            team_data[f"win_pct_{window}"] = team_data["win"].shift(1).rolling(window, min_periods=3).mean()
            
            for _, row in team_data.iterrows():
                key = (row["game_date"], row["team"])
                if key not in rolling_stats:
                    rolling_stats[key] = {}
                for col in [f"ppg_{window}", f"papg_{window}", f"margin_{window}", f"win_pct_{window}"]:
                    rolling_stats[key][col] = row[col]
    
    # Merge back to main df
    for window in windows:
        for side in ["home", "away"]:
            team_col = f"{side}_team"
            prefix = f"{side}_{window}"
            
            df[f"{prefix}_ppg"] = df.apply(
                lambda r: rolling_stats.get((r["game_date"], r[team_col]), {}).get(f"ppg_{window}"), axis=1
            )
            df[f"{prefix}_papg"] = df.apply(
                lambda r: rolling_stats.get((r["game_date"], r[team_col]), {}).get(f"papg_{window}"), axis=1
            )
            df[f"{prefix}_margin"] = df.apply(
                lambda r: rolling_stats.get((r["game_date"], r[team_col]), {}).get(f"margin_{window}"), axis=1
            )
            df[f"{prefix}_win_pct"] = df.apply(
                lambda r: rolling_stats.get((r["game_date"], r[team_col]), {}).get(f"win_pct_{window}"), axis=1
            )
    
    # Differentials
    for window in windows:
        df[f"ppg_diff_{window}"] = df[f"home_{window}_ppg"] - df[f"away_{window}_ppg"]
        df[f"papg_diff_{window}"] = df[f"home_{window}_papg"] - df[f"away_{window}_papg"]
        df[f"margin_diff_{window}"] = df[f"home_{window}_margin"] - df[f"away_{window}_margin"]
        df[f"win_pct_diff_{window}"] = df[f"home_{window}_win_pct"] - df[f"away_{window}_win_pct"]
    
    return df


def add_rest_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rest days and back-to-back detection."""
    
    # Get last game date for each team
    team_last_game = {}
    
    df["home_rest_days"] = np.nan
    df["away_rest_days"] = np.nan
    
    for idx, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        game_date = row["game_date"]
        
        # Home team rest
        if home in team_last_game:
            df.at[idx, "home_rest_days"] = (game_date - team_last_game[home]).days
        
        # Away team rest
        if away in team_last_game:
            df.at[idx, "away_rest_days"] = (game_date - team_last_game[away]).days
        
        # Update last game
        team_last_game[home] = game_date
        team_last_game[away] = game_date
    
    # Back-to-back flags
    df["home_b2b"] = (df["home_rest_days"] == 1).astype(int)
    df["away_b2b"] = (df["away_rest_days"] == 1).astype(int)
    
    # Rest advantage
    df["rest_advantage"] = df["home_rest_days"].fillna(3) - df["away_rest_days"].fillna(3)
    
    return df


def add_streak_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add win/loss streak features."""
    
    team_streaks = {}  # team -> current streak (positive = wins, negative = losses)
    
    df["home_streak"] = np.nan
    df["away_streak"] = np.nan
    
    for idx, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        
        # Record current streaks before game
        df.at[idx, "home_streak"] = team_streaks.get(home, 0)
        df.at[idx, "away_streak"] = team_streaks.get(away, 0)
        
        # Update streaks after game
        if row["fg_margin"] > 0:  # Home wins
            team_streaks[home] = max(1, team_streaks.get(home, 0) + 1) if team_streaks.get(home, 0) >= 0 else 1
            team_streaks[away] = min(-1, team_streaks.get(away, 0) - 1) if team_streaks.get(away, 0) <= 0 else -1
        else:  # Away wins
            team_streaks[away] = max(1, team_streaks.get(away, 0) + 1) if team_streaks.get(away, 0) >= 0 else 1
            team_streaks[home] = min(-1, team_streaks.get(home, 0) - 1) if team_streaks.get(home, 0) <= 0 else -1
    
    df["streak_diff"] = df["home_streak"] - df["away_streak"]
    
    return df


def add_h2h_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add head-to-head history features."""
    
    h2h_record = {}  # (team1, team2) -> [margins]
    
    df["h2h_home_wins"] = 0
    df["h2h_games"] = 0
    df["h2h_avg_margin"] = np.nan
    
    for idx, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        matchup = tuple(sorted([home, away]))
        
        # Get prior H2H
        if matchup in h2h_record:
            margins = h2h_record[matchup]
            df.at[idx, "h2h_games"] = len(margins)
            
            # Count home wins in this matchup
            home_wins = sum(1 for m, h, a in margins if (h == home and m > 0) or (a == home and m < 0))
            df.at[idx, "h2h_home_wins"] = home_wins
            df.at[idx, "h2h_avg_margin"] = np.mean([m if h == home else -m for m, h, a in margins])
        
        # Update H2H
        if matchup not in h2h_record:
            h2h_record[matchup] = []
        h2h_record[matchup].append((row["fg_margin"], home, away))
    
    df["h2h_win_pct"] = df["h2h_home_wins"] / df["h2h_games"].replace(0, np.nan)
    
    return df


def add_elo_features(df: pd.DataFrame, elo_df: pd.DataFrame) -> pd.DataFrame:
    """Add ELO rating features."""
    
    # Map FiveThirtyEight team names
    elo_team_map = {
        "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BRK": "Brooklyn Nets",
        "CHO": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
        "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
        "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
        "LAC": "Los Angeles Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
        "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
        "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
        "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHO": "Phoenix Suns",
        "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
        "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards",
    }
    
    # Get latest ELO for each team before each game date
    elo_df["team1_name"] = elo_df["team1"].map(elo_team_map)
    elo_df["team2_name"] = elo_df["team2"].map(elo_team_map)
    
    # Build ELO lookup: (team, date) -> elo
    team_elo = {}
    for _, row in elo_df.sort_values("date").iterrows():
        team1 = row["team1_name"]
        team2 = row["team2_name"]
        date = row["date"]
        
        if pd.notna(team1):
            team_elo[(team1, date)] = row["elo1_post"]
        if pd.notna(team2):
            team_elo[(team2, date)] = row["elo2_post"]
    
    # Get most recent ELO for each team
    def get_recent_elo(team: str, game_date: datetime, team_elo: dict) -> float:
        relevant = [(d, e) for (t, d), e in team_elo.items() if t == team and d < game_date]
        if not relevant:
            return 1500  # Default ELO
        relevant.sort(key=lambda x: x[0], reverse=True)
        return relevant[0][1]
    
    df["home_elo"] = df.apply(lambda r: get_recent_elo(r["home_team"], r["game_date"], team_elo), axis=1)
    df["away_elo"] = df.apply(lambda r: get_recent_elo(r["away_team"], r["game_date"], team_elo), axis=1)
    df["elo_diff"] = df["home_elo"] - df["away_elo"]
    
    # ELO-based win probability
    df["elo_home_win_prob"] = 1 / (1 + 10 ** (-df["elo_diff"] / 400))
    
    return df


def add_implied_prob_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add implied probabilities from moneylines."""
    
    def american_to_prob(odds):
        if pd.isna(odds):
            return np.nan
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)
    
    df["fg_implied_home_prob"] = df["fg_ml_home"].apply(american_to_prob)
    df["fg_implied_away_prob"] = df["fg_ml_away"].apply(american_to_prob)
    
    # Normalize (remove vig)
    total = df["fg_implied_home_prob"] + df["fg_implied_away_prob"]
    df["fg_home_prob_novg"] = df["fg_implied_home_prob"] / total
    df["fg_away_prob_novg"] = df["fg_implied_away_prob"] / total
    
    # 1H implied probs
    df["1h_implied_home_prob"] = df["1h_ml_home"].apply(american_to_prob)
    df["1h_implied_away_prob"] = df["1h_ml_away"].apply(american_to_prob)
    
    return df


# =============================================================================
# LABELS
# =============================================================================

def calculate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all betting outcome labels."""
    
    # FG spread covered
    df["fg_spread_covered"] = np.where(
        df["fg_spread_line"].notna(),
        (df["fg_margin"] + df["fg_spread_line"]) > 0,
        np.nan
    )
    
    # FG total over
    df["fg_total_over"] = np.where(
        df["fg_total_line"].notna(),
        df["fg_total_actual"] > df["fg_total_line"],
        np.nan
    )
    
    # FG home win
    df["fg_home_win"] = (df["fg_margin"] > 0).astype(int)
    
    # 1H spread covered
    df["1h_spread_covered"] = np.where(
        df["1h_spread_line"].notna(),
        (df["1h_margin"] + df["1h_spread_line"]) > 0,
        np.nan
    )
    
    # 1H total over
    df["1h_total_over"] = np.where(
        df["1h_total_line"].notna(),
        df["1h_total_actual"] > df["1h_total_line"],
        np.nan
    )
    
    # 1H home win
    df["1h_home_win"] = (df["1h_margin"] > 0).astype(int)
    
    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Build complete training data with ALL features")
    parser.add_argument("--start-date", type=str, default="2023-01-01")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "training_data_complete_2023.csv")
    args = parser.parse_args()
    
    print("=" * 70)
    print("BUILDING COMPLETE NBA TRAINING DATA WITH ALL FEATURES")
    print("=" * 70)
    
    # Load all data sources
    print("\n[1] Loading data sources...")
    kaggle_df = load_kaggle_data(args.start_date)
    theodds_fg_df = load_theodds_odds(THEODDS_DIR / "odds")
    theodds_1h_df = load_theodds_1h(THEODDS_DIR / "period_odds")
    elo_df = load_elo_ratings()
    
    if kaggle_df.empty:
        logger.error("No Kaggle data - cannot proceed")
        return 1
    
    # Merge data
    print("\n[2] Merging data sources...")
    df = merge_all_data(kaggle_df, theodds_fg_df, theodds_1h_df)
    
    # Add features
    print("\n[3] Adding ALL features...")
    df = add_all_features(df, elo_df)
    
    # Calculate labels
    print("\n[4] Calculating labels...")
    df = calculate_labels(df)
    
    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nTotal NBA games: {len(df):,}")
    print(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}")
    print(f"\nFeatures: {len([c for c in df.columns if c not in ['match_key', 'date_str']])}")
    print(f"\nFG lines: {df['fg_spread_line'].notna().sum():,} ({df['fg_spread_line'].notna().mean()*100:.1f}%)")
    print(f"1H lines: {df['1h_spread_line'].notna().sum():,} ({df['1h_spread_line'].notna().mean()*100:.1f}%)")
    
    # Label distributions
    for label in ["fg_spread_covered", "fg_total_over", "1h_spread_covered", "1h_total_over"]:
        valid = df[label].notna().sum()
        if valid > 0:
            rate = df[label].dropna().mean() * 100
            print(f"{label}: {valid:,} games, {rate:.1f}% positive")
    
    print(f"\nOutput: {args.output}")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
