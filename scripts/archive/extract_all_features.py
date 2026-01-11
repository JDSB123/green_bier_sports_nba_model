#!/usr/bin/env python3
"""
COMPREHENSIVE FEATURE EXTRACTION

Extracts ALL available features from ALL data sources:
1. Box scores (FG%, 3P%, FT%, rebounds, assists, etc.)
2. Four Factors (eFG%, TOV%, OREB%, FT rate)
3. Pace/tempo estimates
4. Rolling advanced stats
5. Line movement (opening vs closing)
6. Q1/Q2/Q3/Q4 outcomes

This script maximizes use of historical data for backtesting.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.standardization import (
    standardize_team_name,
    generate_match_key,
    CST,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Paths
DATA_DIR = PROJECT_ROOT / "data"
BOX_SCORES_FILE = DATA_DIR / "external" / "nba_database" / "game.csv"
KAGGLE_FILE = DATA_DIR / "external" / "kaggle" / "nba_2008-2025.csv"
THEODDS_DIR = DATA_DIR / "historical" / "the_odds"
OUTPUT_DIR = DATA_DIR / "processed"


# =============================================================================
# 1. BOX SCORE FEATURES
# =============================================================================

def load_box_scores(start_date: str = "2023-01-01") -> pd.DataFrame:
    """Load and process box score data."""
    if not BOX_SCORES_FILE.exists():
        logger.warning(f"Box scores not found: {BOX_SCORES_FILE}")
        return pd.DataFrame()
    
    df = pd.read_csv(BOX_SCORES_FILE)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df[df["game_date"] >= start_date].copy()
    
    # Standardize team names
    df["home_team"] = df["team_name_home"].apply(standardize_team_name)
    df["away_team"] = df["team_name_away"].apply(standardize_team_name)
    
    # Generate match key
    df["match_key"] = df.apply(
        lambda r: generate_match_key(r["game_date"], r["home_team"], r["away_team"], source_is_utc=False),
        axis=1
    )
    
    logger.info(f"Loaded {len(df)} box scores from {start_date}")
    return df


def compute_four_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Four Factors for each team:
    1. eFG% = (FGM + 0.5 * 3PM) / FGA
    2. TOV% = TOV / (FGA + 0.44 * FTA + TOV)
    3. OREB% = OREB / (OREB + opp_DREB)
    4. FT Rate = FTM / FGA
    """
    for side in ["home", "away"]:
        opp = "away" if side == "home" else "home"
        
        # Effective Field Goal %
        df[f"{side}_efg_pct"] = (
            (df[f"fgm_{side}"] + 0.5 * df[f"fg3m_{side}"]) / 
            df[f"fga_{side}"].replace(0, np.nan)
        )
        
        # Turnover %
        possessions_approx = df[f"fga_{side}"] + 0.44 * df[f"fta_{side}"] + df[f"tov_{side}"]
        df[f"{side}_tov_pct"] = df[f"tov_{side}"] / possessions_approx.replace(0, np.nan)
        
        # Offensive Rebound %
        total_rebs = df[f"oreb_{side}"] + df[f"dreb_{opp}"]
        df[f"{side}_oreb_pct"] = df[f"oreb_{side}"] / total_rebs.replace(0, np.nan)
        
        # Free Throw Rate
        df[f"{side}_ft_rate"] = df[f"ftm_{side}"] / df[f"fga_{side}"].replace(0, np.nan)
        
        # Additional shooting splits
        df[f"{side}_fg_pct"] = df[f"fgm_{side}"] / df[f"fga_{side}"].replace(0, np.nan)
        df[f"{side}_3p_pct"] = df[f"fg3m_{side}"] / df[f"fg3a_{side}"].replace(0, np.nan)
        df[f"{side}_ft_pct"] = df[f"ftm_{side}"] / df[f"fta_{side}"].replace(0, np.nan)
        df[f"{side}_3p_rate"] = df[f"fg3a_{side}"] / df[f"fga_{side}"].replace(0, np.nan)
        
        # Assist/turnover ratio
        df[f"{side}_ast_tov_ratio"] = df[f"ast_{side}"] / df[f"tov_{side}"].replace(0, np.nan)
        
        # Rebounding
        df[f"{side}_total_reb"] = df[f"reb_{side}"]
        df[f"{side}_oreb"] = df[f"oreb_{side}"]
        df[f"{side}_dreb"] = df[f"dreb_{side}"]
    
    # Differentials
    df["efg_diff"] = df["home_efg_pct"] - df["away_efg_pct"]
    df["tov_diff"] = df["away_tov_pct"] - df["home_tov_pct"]  # Lower is better
    df["oreb_diff"] = df["home_oreb_pct"] - df["away_oreb_pct"]
    df["ft_rate_diff"] = df["home_ft_rate"] - df["away_ft_rate"]
    df["reb_diff"] = df["home_total_reb"] - df["away_total_reb"]
    
    logger.info("  Computed Four Factors")
    return df


def estimate_pace(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate pace (possessions per game).
    Possession estimate = FGA - OREB + TOV + 0.44*FTA
    """
    for side in ["home", "away"]:
        df[f"{side}_possessions"] = (
            df[f"fga_{side}"] - df[f"oreb_{side}"] + 
            df[f"tov_{side}"] + 0.44 * df[f"fta_{side}"]
        )
    
    df["pace"] = (df["home_possessions"] + df["away_possessions"]) / 2
    df["pace_diff"] = df["home_possessions"] - df["away_possessions"]
    
    logger.info("  Estimated pace")
    return df


def compute_offensive_defensive_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute offensive and defensive ratings (points per 100 possessions).
    """
    for side in ["home", "away"]:
        opp = "away" if side == "home" else "home"
        
        # Offensive rating = (Points / Possessions) * 100
        df[f"{side}_off_rtg"] = (
            df[f"pts_{side}"] / df[f"{side}_possessions"].replace(0, np.nan) * 100
        )
        
        # Defensive rating = (Opp Points / Opp Possessions) * 100
        df[f"{side}_def_rtg"] = (
            df[f"pts_{opp}"] / df[f"{opp}_possessions"].replace(0, np.nan) * 100
        )
        
        # Net rating
        df[f"{side}_net_rtg"] = df[f"{side}_off_rtg"] - df[f"{side}_def_rtg"]
    
    df["net_rtg_diff"] = df["home_net_rtg"] - df["away_net_rtg"]
    
    logger.info("  Computed offensive/defensive ratings")
    return df


# =============================================================================
# 2. ROLLING ADVANCED STATS
# =============================================================================

def compute_rolling_advanced_stats(df: pd.DataFrame, windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
    """
    Compute rolling averages of advanced stats for each team.
    """
    df = df.sort_values("game_date").copy()
    
    # Features to roll
    advanced_features = [
        "efg_pct", "tov_pct", "oreb_pct", "ft_rate",
        "fg_pct", "3p_pct", "ft_pct", "3p_rate",
        "ast_tov_ratio", "off_rtg", "def_rtg", "net_rtg",
        "possessions", "total_reb",
    ]
    
    for window in windows:
        for team_col, side in [("home_team", "home"), ("away_team", "away")]:
            for feature in advanced_features:
                col_name = f"{side}_{feature}"
                if col_name not in df.columns:
                    continue
                
                # Create rolling average per team
                roll_col = f"{side}_{window}g_{feature}"
                
                df[roll_col] = df.groupby(team_col)[col_name].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=3).mean()
                )
    
    logger.info(f"  Computed rolling advanced stats for windows {windows}")
    return df


# =============================================================================
# 3. QUARTER-BY-QUARTER OUTCOMES
# =============================================================================

def compute_quarter_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Q1, Q2, Q3, Q4 spread/total outcomes from Kaggle scores.
    """
    kaggle = pd.read_csv(KAGGLE_FILE)
    kaggle["game_date"] = pd.to_datetime(kaggle["date"])
    
    # Standardize and create match key
    kaggle["home_team"] = kaggle["home"].apply(standardize_team_name)
    kaggle["away_team"] = kaggle["away"].apply(standardize_team_name)
    kaggle["match_key"] = kaggle.apply(
        lambda r: generate_match_key(r["game_date"], r["home_team"], r["away_team"], source_is_utc=False),
        axis=1
    )
    
    # Compute quarter outcomes
    for q in [1, 2, 3, 4]:
        q_home = kaggle[f"q{q}_home"].fillna(0)
        q_away = kaggle[f"q{q}_away"].fillna(0)
        
        kaggle[f"q{q}_total"] = q_home + q_away
        kaggle[f"q{q}_margin"] = q_home - q_away  # Home perspective
        kaggle[f"q{q}_home_win"] = (q_home > q_away).astype(int)
    
    # 1H = Q1 + Q2
    kaggle["1h_home_score"] = kaggle["q1_home"].fillna(0) + kaggle["q2_home"].fillna(0)
    kaggle["1h_away_score"] = kaggle["q1_away"].fillna(0) + kaggle["q2_away"].fillna(0)
    kaggle["1h_total"] = kaggle["1h_home_score"] + kaggle["1h_away_score"]
    kaggle["1h_margin"] = kaggle["1h_home_score"] - kaggle["1h_away_score"]
    
    # 2H = Q3 + Q4
    kaggle["2h_home_score"] = kaggle["q3_home"].fillna(0) + kaggle["q4_home"].fillna(0)
    kaggle["2h_away_score"] = kaggle["q3_away"].fillna(0) + kaggle["q4_away"].fillna(0)
    kaggle["2h_total"] = kaggle["2h_home_score"] + kaggle["2h_away_score"]
    kaggle["2h_margin"] = kaggle["2h_home_score"] - kaggle["2h_away_score"]
    
    # Merge back
    quarter_cols = [
        "match_key",
        "q1_total", "q1_margin", "q1_home_win",
        "q2_total", "q2_margin", 
        "q3_total", "q3_margin",
        "q4_total", "q4_margin",
        "1h_total", "1h_margin",
        "2h_total", "2h_margin",
    ]
    quarter_data = kaggle[quarter_cols].drop_duplicates(subset=["match_key"])
    
    df = df.merge(quarter_data, on="match_key", how="left", suffixes=("", "_kaggle"))
    
    logger.info(f"  Added quarter-by-quarter outcomes")
    return df


# =============================================================================
# 4. LINE MOVEMENT
# =============================================================================

def extract_line_movement() -> pd.DataFrame:
    """
    Extract opening vs closing lines from TheOdds API historical data.
    Uses timestamps to identify line changes.
    """
    odds_dir = THEODDS_DIR / "odds"
    if not odds_dir.exists():
        return pd.DataFrame()
    
    movements = []
    
    for season_dir in sorted(odds_dir.glob("*")):
        if not season_dir.is_dir():
            continue
        
        for json_file in sorted(season_dir.glob("*.json")):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                
                events = data if isinstance(data, list) else data.get("data", [])
                
                for event in events:
                    home = standardize_team_name(event.get("home_team", ""))
                    away = standardize_team_name(event.get("away_team", ""))
                    commence = event.get("commence_time", "")
                    
                    if not all([home, away, commence]):
                        continue
                    
                    match_key = generate_match_key(commence, home, away)
                    
                    # Collect spreads from all bookmakers
                    spreads = []
                    totals = []
                    
                    for bm in event.get("bookmakers", []):
                        last_update = bm.get("last_update", "")
                        
                        for mkt in bm.get("markets", []):
                            if mkt.get("key") == "spreads":
                                for o in mkt.get("outcomes", []):
                                    if o.get("name") == event.get("home_team") and o.get("point") is not None:
                                        spreads.append({
                                            "point": o["point"],
                                            "update": last_update,
                                            "book": bm.get("key"),
                                        })
                            elif mkt.get("key") == "totals":
                                for o in mkt.get("outcomes", []):
                                    if o.get("name") == "Over" and o.get("point") is not None:
                                        totals.append({
                                            "point": o["point"],
                                            "update": last_update,
                                            "book": bm.get("key"),
                                        })
                    
                    if spreads:
                        # Sort by update time to get earliest and latest
                        spreads_sorted = sorted(spreads, key=lambda x: x.get("update", ""))
                        totals_sorted = sorted(totals, key=lambda x: x.get("update", ""))
                        
                        movements.append({
                            "match_key": match_key,
                            "spread_first": spreads_sorted[0]["point"] if spreads_sorted else None,
                            "spread_last": spreads_sorted[-1]["point"] if spreads_sorted else None,
                            "total_first": totals_sorted[0]["point"] if totals_sorted else None,
                            "total_last": totals_sorted[-1]["point"] if totals_sorted else None,
                            "num_books": len(set(s["book"] for s in spreads)),
                        })
                        
            except Exception as e:
                continue
    
    if not movements:
        return pd.DataFrame()
    
    df = pd.DataFrame(movements)
    
    # Compute movement
    df["spread_movement"] = df["spread_last"] - df["spread_first"]
    df["total_movement"] = df["total_last"] - df["total_first"]
    
    # Movement direction (1 = moved toward home, -1 = moved toward away, 0 = no move)
    df["spread_direction"] = np.sign(df["spread_movement"])
    df["total_direction"] = np.sign(df["total_movement"])
    
    logger.info(f"  Extracted line movement for {len(df)} games")
    return df


# =============================================================================
# 5. COMBINE ALL FEATURES
# =============================================================================

def build_comprehensive_features(start_date: str = "2023-01-01") -> pd.DataFrame:
    """Build dataset with ALL available features."""
    
    print("\n" + "=" * 70)
    print(" EXTRACTING ALL FEATURES FROM ALL SOURCES")
    print("=" * 70)
    
    # Load box scores
    print("\n[1] Loading box scores...")
    df = load_box_scores(start_date)
    
    if df.empty:
        logger.error("No box score data available")
        return pd.DataFrame()
    
    # Compute advanced stats
    print("\n[2] Computing advanced stats...")
    df = compute_four_factors(df)
    df = estimate_pace(df)
    df = compute_offensive_defensive_ratings(df)
    
    # Rolling stats
    print("\n[3] Computing rolling advanced stats...")
    df = compute_rolling_advanced_stats(df)
    
    # Quarter outcomes
    print("\n[4] Adding quarter-by-quarter outcomes...")
    df = compute_quarter_outcomes(df)
    
    # Line movement
    print("\n[5] Extracting line movement...")
    movement_df = extract_line_movement()
    if not movement_df.empty:
        df = df.merge(
            movement_df[["match_key", "spread_movement", "total_movement", 
                        "spread_direction", "total_direction"]],
            on="match_key",
            how="left"
        )
    
    # Summary
    print("\n" + "=" * 70)
    print(" FEATURE EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"\n  Total games: {len(df):,}")
    print(f"  Total features: {len(df.columns)}")
    
    # Count feature categories
    four_factors = [c for c in df.columns if any(x in c for x in ["efg", "tov_pct", "oreb_pct", "ft_rate"])]
    shooting = [c for c in df.columns if any(x in c for x in ["fg_pct", "3p_pct", "ft_pct", "3p_rate"])]
    ratings = [c for c in df.columns if any(x in c for x in ["off_rtg", "def_rtg", "net_rtg"])]
    rolling = [c for c in df.columns if any(x in c for x in ["5g_", "10g_", "20g_"])]
    quarter = [c for c in df.columns if c.startswith("q") and any(x in c for x in ["total", "margin"])]
    movement = [c for c in df.columns if "movement" in c or "direction" in c]
    
    print(f"\n  Feature breakdown:")
    print(f"    Four Factors: {len(four_factors)}")
    print(f"    Shooting splits: {len(shooting)}")
    print(f"    Ratings: {len(ratings)}")
    print(f"    Rolling stats: {len(rolling)}")
    print(f"    Quarter data: {len(quarter)}")
    print(f"    Line movement: {len(movement)}")
    
    # Save
    output_file = OUTPUT_DIR / "features_comprehensive.csv"
    df.to_csv(output_file, index=False)
    print(f"\n  Saved to: {output_file}")
    
    return df


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract comprehensive features")
    parser.add_argument("--start-date", default="2023-01-01", help="Start date (YYYY-MM-DD)")
    args = parser.parse_args()
    
    df = build_comprehensive_features(args.start_date)
    
    if not df.empty:
        print(f"\n  Sample features (first 5 columns):")
        print(df[df.columns[:10]].head(3).to_string())
