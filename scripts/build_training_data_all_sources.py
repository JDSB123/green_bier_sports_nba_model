#!/usr/bin/env python3
"""
BUILD TRAINING DATA FROM ALL SOURCES

Uses EVERY available data file:
1. Kaggle (base games + scores + FG lines)
2. TheOdds derived lines (FG + 1H pre-computed consensus)
3. TheOdds 1H odds exports (detailed 1H lines)
4. NBA Database game.csv (box scores through June 2023)
5. Compute ELO from game results (since historical ends 2015)

This script MAXIMIZES data utilization.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional

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

# Paths to ALL data files
DATA_DIR = PROJECT_ROOT / "data"
KAGGLE_FILE = DATA_DIR / "external" / "kaggle" / "nba_2008-2025.csv"
THEODDS_LINES = DATA_DIR / "historical" / "derived" / "theodds_lines.csv"
H1_ODDS_2324 = DATA_DIR / "historical" / "exports" / "2023-2024_odds_1h.csv"
H1_ODDS_2425 = DATA_DIR / "historical" / "exports" / "2024-2025_odds_1h.csv"
FEAT_ODDS_2324 = DATA_DIR / "historical" / "exports" / "2023-2024_odds_featured.csv"
FEAT_ODDS_2425 = DATA_DIR / "historical" / "exports" / "2024-2025_odds_featured.csv"
BOX_SCORES = DATA_DIR / "external" / "nba_database" / "game.csv"
OUTPUT_DIR = DATA_DIR / "processed"


def safe_median(values: list) -> Optional[float]:
    valid = [v for v in values if v is not None and pd.notna(v)]
    return median(valid) if valid else None


# =============================================================================
# LOAD ALL DATA SOURCES
# =============================================================================

def load_kaggle(start_date: str) -> pd.DataFrame:
    """Load Kaggle as base."""
    print("\n[1/7] Loading Kaggle (base games + scores)...")
    df = pd.read_csv(KAGGLE_FILE)
    df["game_date"] = pd.to_datetime(df["date"])
    df = df[df["game_date"] >= start_date].copy()
    
    df["home_team"] = df["home"].apply(standardize_team_name)
    df["away_team"] = df["away"].apply(standardize_team_name)
    df["match_key"] = df.apply(
        lambda r: generate_match_key(r["game_date"], r["home_team"], r["away_team"], source_is_utc=False),
        axis=1
    )
    
    # Scores
    df["home_score"] = df["score_home"]
    df["away_score"] = df["score_away"]
    df["fg_margin"] = df["home_score"] - df["away_score"]
    df["fg_total_actual"] = df["home_score"] + df["away_score"]
    
    # Q1-Q4
    for q in [1, 2, 3, 4]:
        df[f"home_q{q}"] = df[f"q{q}_home"]
        df[f"away_q{q}"] = df[f"q{q}_away"]
    
    # Half scores
    df["home_1h"] = df["home_q1"].fillna(0) + df["home_q2"].fillna(0)
    df["away_1h"] = df["away_q1"].fillna(0) + df["away_q2"].fillna(0)
    df["1h_total_actual"] = df["home_1h"] + df["away_1h"]
    df["1h_margin"] = df["home_1h"] - df["away_1h"]
    
    # FG lines from Kaggle
    df["kaggle_fg_spread"] = df.apply(
        lambda r: -r["spread"] if r.get("whos_favored") == "home" else r["spread"], axis=1
    )
    df["kaggle_fg_total"] = df["total"]
    df["kaggle_fg_ml_home"] = df["moneyline_home"]
    df["kaggle_fg_ml_away"] = df["moneyline_away"]
    
    print(f"       Loaded {len(df):,} games")
    return df


def load_theodds_derived() -> pd.DataFrame:
    """Load pre-computed TheOdds consensus lines."""
    print("\n[2/7] Loading TheOdds derived lines (pre-computed consensus)...")
    
    if not THEODDS_LINES.exists():
        print("       [SKIP] File not found")
        return pd.DataFrame()
    
    df = pd.read_csv(THEODDS_LINES)
    df["commence_time"] = pd.to_datetime(df["commence_time"])
    
    df["home_team"] = df["home_team"].apply(standardize_team_name)
    df["away_team"] = df["away_team"].apply(standardize_team_name)
    df["match_key"] = df.apply(
        lambda r: generate_match_key(r["commence_time"], r["home_team"], r["away_team"], source_is_utc=True),
        axis=1
    )
    
    # Rename columns
    df = df.rename(columns={
        "fg_spread_line": "to_fg_spread",
        "fg_total_line": "to_fg_total",
        "fg_ml_home": "to_fg_ml_home",
        "fg_ml_away": "to_fg_ml_away",
        "fh_spread_line": "to_1h_spread",
        "fh_total_line": "to_1h_total",
        "fh_ml_home": "to_1h_ml_home",
        "fh_ml_away": "to_1h_ml_away",
    })
    
    print(f"       Loaded {len(df):,} games with pre-computed lines")
    return df


def load_h1_odds_exports() -> pd.DataFrame:
    """Load detailed 1H odds from exports."""
    print("\n[3/7] Loading 1H odds exports...")
    
    dfs = []
    for file in [H1_ODDS_2324, H1_ODDS_2425]:
        if file.exists():
            df = pd.read_csv(file)
            dfs.append(df)
            print(f"       Loaded {len(df):,} rows from {file.name}")
    
    if not dfs:
        return pd.DataFrame()
    
    df = pd.concat(dfs, ignore_index=True)
    
    # Standardize
    df["home_team"] = df["home_team"].apply(standardize_team_name)
    df["away_team"] = df["away_team"].apply(standardize_team_name)
    df["commence_time"] = pd.to_datetime(df["commence_time"])
    df["match_key"] = df.apply(
        lambda r: generate_match_key(r["commence_time"], r["home_team"], r["away_team"], source_is_utc=True),
        axis=1
    )
    
    # Aggregate by game - compute consensus
    lines = []
    for match_key, group in df.groupby("match_key"):
        spreads_home = []
        totals = []
        ml_home = []
        ml_away = []
        
        for _, row in group.iterrows():
            market = row.get("market_key", "")
            outcome = row.get("outcome_name", "")
            point = row.get("outcome_point")
            price = row.get("outcome_price")
            
            if market == "spreads_h1" and outcome == row["home_team"] and pd.notna(point):
                spreads_home.append(point)
            elif market == "totals_h1" and outcome == "Over" and pd.notna(point):
                totals.append(point)
            elif market == "h2h_h1":
                if outcome == row["home_team"] and pd.notna(price):
                    ml_home.append(price)
                elif outcome == row["away_team"] and pd.notna(price):
                    ml_away.append(price)
        
        lines.append({
            "match_key": match_key,
            "export_1h_spread": safe_median(spreads_home),
            "export_1h_total": safe_median(totals),
            "export_1h_ml_home": safe_median(ml_home),
            "export_1h_ml_away": safe_median(ml_away),
        })
    
    result = pd.DataFrame(lines)
    print(f"       Aggregated to {len(result):,} games with 1H lines")
    return result


def load_featured_odds() -> pd.DataFrame:
    """Load featured FG odds with line movement."""
    print("\n[4/7] Loading featured odds exports (for line movement)...")
    
    dfs = []
    for file in [FEAT_ODDS_2324, FEAT_ODDS_2425]:
        if file.exists():
            df = pd.read_csv(file)
            dfs.append(df)
            print(f"       Loaded {len(df):,} rows from {file.name}")
    
    if not dfs:
        return pd.DataFrame()
    
    df = pd.concat(dfs, ignore_index=True)
    
    # Standardize
    df["home_team"] = df["home_team"].apply(standardize_team_name)
    df["away_team"] = df["away_team"].apply(standardize_team_name)
    df["commence_time"] = pd.to_datetime(df["commence_time"])
    df["snapshot_timestamp"] = pd.to_datetime(df["snapshot_timestamp"])
    df["match_key"] = df.apply(
        lambda r: generate_match_key(r["commence_time"], r["home_team"], r["away_team"], source_is_utc=True),
        axis=1
    )
    
    # Compute opening/closing lines and movement
    movements = []
    for match_key, group in df.groupby("match_key"):
        group = group.sort_values("snapshot_timestamp")
        
        first = group.iloc[0] if len(group) > 0 else None
        last = group.iloc[-1] if len(group) > 0 else None
        
        # Get spread lines
        spread_rows = group[(group["market_key"] == "spreads") & (group["outcome_name"] == group["home_team"])]
        if len(spread_rows) > 0:
            spread_rows = spread_rows.sort_values("snapshot_timestamp")
            open_spread = spread_rows.iloc[0]["outcome_point"] if len(spread_rows) > 0 else None
            close_spread = spread_rows.iloc[-1]["outcome_point"] if len(spread_rows) > 0 else None
        else:
            open_spread = close_spread = None
        
        # Get total lines
        total_rows = group[(group["market_key"] == "totals") & (group["outcome_name"] == "Over")]
        if len(total_rows) > 0:
            total_rows = total_rows.sort_values("snapshot_timestamp")
            open_total = total_rows.iloc[0]["outcome_point"] if len(total_rows) > 0 else None
            close_total = total_rows.iloc[-1]["outcome_point"] if len(total_rows) > 0 else None
        else:
            open_total = close_total = None
        
        movements.append({
            "match_key": match_key,
            "open_spread": open_spread,
            "close_spread": close_spread,
            "spread_movement": (close_spread - open_spread) if pd.notna(open_spread) and pd.notna(close_spread) else None,
            "open_total": open_total,
            "close_total": close_total,
            "total_movement": (close_total - open_total) if pd.notna(open_total) and pd.notna(close_total) else None,
        })
    
    result = pd.DataFrame(movements)
    print(f"       Computed line movement for {len(result):,} games")
    return result


def load_box_scores() -> pd.DataFrame:
    """Load box scores for advanced stats."""
    print("\n[5/7] Loading box scores (advanced stats)...")
    
    if not BOX_SCORES.exists():
        print("       [SKIP] File not found")
        return pd.DataFrame()
    
    df = pd.read_csv(BOX_SCORES)
    df["game_date"] = pd.to_datetime(df["game_date"])
    
    df["home_team"] = df["team_name_home"].apply(standardize_team_name)
    df["away_team"] = df["team_name_away"].apply(standardize_team_name)
    df["match_key"] = df.apply(
        lambda r: generate_match_key(r["game_date"], r["home_team"], r["away_team"], source_is_utc=False),
        axis=1
    )
    
    # Compute Four Factors and ratings
    for side in ["home", "away"]:
        opp = "away" if side == "home" else "home"
        
        df[f"{side}_efg_pct"] = (df[f"fgm_{side}"] + 0.5 * df[f"fg3m_{side}"]) / df[f"fga_{side}"].replace(0, np.nan)
        poss = df[f"fga_{side}"] + 0.44 * df[f"fta_{side}"] + df[f"tov_{side}"]
        df[f"{side}_tov_pct"] = df[f"tov_{side}"] / poss.replace(0, np.nan)
        total_reb = df[f"oreb_{side}"] + df[f"dreb_{opp}"]
        df[f"{side}_oreb_pct"] = df[f"oreb_{side}"] / total_reb.replace(0, np.nan)
        df[f"{side}_ft_rate"] = df[f"ftm_{side}"] / df[f"fga_{side}"].replace(0, np.nan)
        df[f"{side}_3p_rate"] = df[f"fg3a_{side}"] / df[f"fga_{side}"].replace(0, np.nan)
        df[f"{side}_poss"] = df[f"fga_{side}"] - df[f"oreb_{side}"] + df[f"tov_{side}"] + 0.44 * df[f"fta_{side}"]
    
    for side in ["home", "away"]:
        opp = "away" if side == "home" else "home"
        df[f"{side}_off_rtg"] = df[f"pts_{side}"] / df[f"{side}_poss"].replace(0, np.nan) * 100
        df[f"{side}_def_rtg"] = df[f"pts_{opp}"] / df[f"{opp}_poss"].replace(0, np.nan) * 100
        df[f"{side}_net_rtg"] = df[f"{side}_off_rtg"] - df[f"{side}_def_rtg"]
    
    df["pace"] = (df["home_poss"] + df["away_poss"]) / 2
    
    # Select columns
    cols = ["match_key", "pace"]
    for side in ["home", "away"]:
        cols.extend([
            f"{side}_efg_pct", f"{side}_tov_pct", f"{side}_oreb_pct", 
            f"{side}_ft_rate", f"{side}_3p_rate",
            f"{side}_off_rtg", f"{side}_def_rtg", f"{side}_net_rtg",
        ])
    
    result = df[cols].drop_duplicates(subset=["match_key"])
    print(f"       Loaded {len(result):,} games with box scores")
    print(f"       Date range: {df['game_date'].min().date()} to {df['game_date'].max().date()}")
    return result


def compute_elo_from_results(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ELO ratings from game results."""
    print("\n[6/7] Computing ELO from game results...")
    
    df = df.sort_values("game_date").copy()
    
    # Initialize ELO for all teams
    K = 20  # ELO K-factor
    HOME_ADV = 100  # Home court advantage
    elo = {team: 1500 for team in set(df["home_team"]) | set(df["away_team"])}
    
    home_elos = []
    away_elos = []
    
    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        
        home_elos.append(elo[home])
        away_elos.append(elo[away])
        
        # Expected scores
        home_exp = 1 / (1 + 10 ** ((elo[away] - elo[home] - HOME_ADV) / 400))
        away_exp = 1 - home_exp
        
        # Actual result
        home_win = 1 if row["home_score"] > row["away_score"] else 0
        away_win = 1 - home_win
        
        # Update ELO
        elo[home] += K * (home_win - home_exp)
        elo[away] += K * (away_win - away_exp)
    
    df["home_elo"] = home_elos
    df["away_elo"] = away_elos
    df["elo_diff"] = df["home_elo"] - df["away_elo"]
    df["elo_home_win_prob"] = 1 / (1 + 10 ** ((df["away_elo"] - df["home_elo"] - HOME_ADV) / 400))
    
    print(f"       Computed ELO for {len(df):,} games")
    return df


def compute_rolling_stats(df: pd.DataFrame, windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
    """Compute rolling stats."""
    print("\n[7/7] Computing rolling stats...")
    
    df = df.sort_values("game_date").copy()
    
    # Stats to roll
    stats = ["score", "1h", "fg_margin", "1h_margin"]
    advanced = ["efg_pct", "tov_pct", "oreb_pct", "ft_rate", "3p_rate", "off_rtg", "def_rtg", "net_rtg"]
    
    new_cols = {}
    
    for window in windows:
        for side in ["home", "away"]:
            team_col = f"{side}_team"
            
            for stat in stats:
                col = f"{side}_{stat}" if stat not in ["fg_margin", "1h_margin"] else stat
                if col not in df.columns:
                    continue
                roll_col = f"{side}_{window}g_{stat}"
                new_cols[roll_col] = df.groupby(team_col)[col].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=3).mean()
                )
            
            for stat in advanced:
                col = f"{side}_{stat}"
                if col not in df.columns:
                    continue
                roll_col = f"{side}_{window}g_{stat}"
                new_cols[roll_col] = df.groupby(team_col)[col].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=3).mean()
                )
    
    # Add all at once to avoid fragmentation
    for col, values in new_cols.items():
        df[col] = values
    
    print(f"       Added {len(new_cols)} rolling stat columns")
    return df


def merge_all(
    kaggle: pd.DataFrame,
    theodds_derived: pd.DataFrame,
    h1_exports: pd.DataFrame,
    featured: pd.DataFrame,
    box_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all data sources."""
    print("\n[MERGING] Combining all data sources...")
    
    df = kaggle.copy()
    original_len = len(df)
    
    # Merge TheOdds derived
    if not theodds_derived.empty:
        cols = ["match_key", "to_fg_spread", "to_fg_total", "to_fg_ml_home", "to_fg_ml_away",
                "to_1h_spread", "to_1h_total", "to_1h_ml_home", "to_1h_ml_away"]
        theodds_derived = theodds_derived[[c for c in cols if c in theodds_derived.columns]]
        df = df.merge(theodds_derived.drop_duplicates("match_key"), on="match_key", how="left")
        matched = df["to_fg_spread"].notna().sum()
        print(f"         TheOdds derived: {matched:,}/{original_len:,} ({matched/original_len*100:.1f}%)")
    
    # Merge 1H exports
    if not h1_exports.empty:
        df = df.merge(h1_exports.drop_duplicates("match_key"), on="match_key", how="left")
        matched = df["export_1h_spread"].notna().sum()
        print(f"         1H exports: {matched:,}/{original_len:,} ({matched/original_len*100:.1f}%)")
    
    # Merge featured (line movement)
    if not featured.empty:
        df = df.merge(featured.drop_duplicates("match_key"), on="match_key", how="left")
        matched = df["spread_movement"].notna().sum()
        print(f"         Line movement: {matched:,}/{original_len:,} ({matched/original_len*100:.1f}%)")
    
    # Merge box scores
    if not box_scores.empty:
        df = df.merge(box_scores.drop_duplicates("match_key"), on="match_key", how="left")
        matched = df["home_efg_pct"].notna().sum()
        print(f"         Box scores: {matched:,}/{original_len:,} ({matched/original_len*100:.1f}%)")
    
    # Consolidate lines
    df["fg_spread_line"] = df["to_fg_spread"].fillna(df["kaggle_fg_spread"])
    df["fg_total_line"] = df["to_fg_total"].fillna(df["kaggle_fg_total"])
    df["fg_ml_home"] = df["to_fg_ml_home"].fillna(df["kaggle_fg_ml_home"])
    df["fg_ml_away"] = df["to_fg_ml_away"].fillna(df["kaggle_fg_ml_away"])
    
    df["1h_spread_line"] = df["export_1h_spread"].fillna(df.get("to_1h_spread"))
    df["1h_total_line"] = df["export_1h_total"].fillna(df.get("to_1h_total"))
    df["1h_ml_home"] = df["export_1h_ml_home"].fillna(df.get("to_1h_ml_home"))
    df["1h_ml_away"] = df["export_1h_ml_away"].fillna(df.get("to_1h_ml_away"))
    
    return df


def compute_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Compute betting outcome labels."""
    print("\n[LABELS] Computing betting outcomes...")
    
    # FG
    df["fg_spread_covered"] = np.where(
        df["fg_spread_line"].notna(),
        (df["fg_margin"] + df["fg_spread_line"] > 0).astype(int),
        np.nan
    )
    df["fg_total_over"] = np.where(
        df["fg_total_line"].notna(),
        (df["fg_total_actual"] > df["fg_total_line"]).astype(int),
        np.nan
    )
    df["fg_home_win"] = (df["fg_margin"] > 0).astype(int)
    
    # 1H
    df["1h_spread_covered"] = np.where(
        df["1h_spread_line"].notna(),
        (df["1h_margin"] + df["1h_spread_line"] > 0).astype(int),
        np.nan
    )
    df["1h_total_over"] = np.where(
        df["1h_total_line"].notna(),
        (df["1h_total_actual"] > df["1h_total_line"]).astype(int),
        np.nan
    )
    df["1h_home_win"] = (df["1h_margin"] > 0).astype(int)
    
    return df


def print_summary(df: pd.DataFrame):
    """Print data summary."""
    print("\n" + "=" * 80)
    print(" FINAL TRAINING DATA SUMMARY")
    print("=" * 80)
    
    print(f"\n  Total games: {len(df):,}")
    print(f"  Date range: {df['game_date'].min().date()} to {df['game_date'].max().date()}")
    print(f"  Total columns: {len(df.columns)}")
    
    # Coverage
    print("\n  DATA COVERAGE:")
    coverage = [
        ("FG spread", "fg_spread_line"),
        ("FG total", "fg_total_line"),
        ("1H spread", "1h_spread_line"),
        ("1H total", "1h_total_line"),
        ("Line movement", "spread_movement"),
        ("Box scores (eFG%)", "home_efg_pct"),
        ("ELO ratings", "home_elo"),
    ]
    for name, col in coverage:
        if col in df.columns:
            pct = df[col].notna().mean() * 100
            print(f"    {name}: {pct:.1f}%")
    
    # Labels
    print("\n  LABEL COUNTS:")
    for label in ["fg_spread_covered", "fg_total_over", "1h_spread_covered", "1h_total_over"]:
        if label in df.columns:
            n = df[label].notna().sum()
            pct = df[label].mean() * 100 if n > 0 else 0
            print(f"    {label}: {n:,} games")


def main(start_date: str = "2023-01-01"):
    print("\n" + "=" * 80)
    print(" BUILDING TRAINING DATA FROM ALL SOURCES")
    print("=" * 80)
    
    # Load all sources
    kaggle = load_kaggle(start_date)
    theodds_derived = load_theodds_derived()
    h1_exports = load_h1_odds_exports()
    featured = load_featured_odds()
    box_scores = load_box_scores()
    
    # Merge
    df = merge_all(kaggle, theodds_derived, h1_exports, featured, box_scores)
    
    # Compute ELO
    df = compute_elo_from_results(df)
    
    # Rolling stats
    df = compute_rolling_stats(df)
    
    # Labels
    df = compute_labels(df)
    
    # Summary
    print_summary(df)
    
    # Save
    output_file = OUTPUT_DIR / f"training_data_all_sources_{start_date[:4]}.csv"
    df.to_csv(output_file, index=False)
    print(f"\n  Saved: {output_file}")
    
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2023-01-01")
    args = parser.parse_args()
    main(args.start_date)
