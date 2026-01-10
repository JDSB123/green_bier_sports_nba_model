#!/usr/bin/env python3
"""
BUILD COMPLETE TRAINING DATA

Uses ALL available data through June 2025:
1. Kaggle betting data (2007-2025): Scores, Q1-Q4, FG lines
2. TheOdds derived + exports (2021-2025): FG + 1H lines, line movement
3. wyattowalsh box scores (1946-2023): Advanced stats where available
4. Computed ELO from game results
5. All rolling stats from scores

This ensures we have data for ALL games 2023+.
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

# Paths
DATA_DIR = PROJECT_ROOT / "data"
KAGGLE_FILE = DATA_DIR / "external" / "kaggle" / "nba_2008-2025.csv"
THEODDS_LINES = DATA_DIR / "historical" / "derived" / "theodds_lines.csv"
H1_EXPORTS = [
    DATA_DIR / "historical" / "exports" / "2023-2024_odds_1h.csv",
    DATA_DIR / "historical" / "exports" / "2024-2025_odds_1h.csv",
]
FEAT_EXPORTS = [
    DATA_DIR / "historical" / "exports" / "2023-2024_odds_featured.csv",
    DATA_DIR / "historical" / "exports" / "2024-2025_odds_featured.csv",
]
THEODDS_2025_26 = DATA_DIR / "historical" / "the_odds" / "2025-2026" / "2025-2026_odds_fg.csv"
BOX_SCORES = DATA_DIR / "external" / "nba_database" / "game.csv"
NBA_API_BOX_SCORES = [
    DATA_DIR / "raw" / "nba_api" / "box_scores_2023_24.csv",
    DATA_DIR / "raw" / "nba_api" / "box_scores_2024_25.csv",
    DATA_DIR / "raw" / "nba_api" / "box_scores_2025_26.csv",
]
OUTPUT_DIR = DATA_DIR / "processed"


def safe_median(values: list) -> Optional[float]:
    valid = [v for v in values if v is not None and pd.notna(v)]
    return median(valid) if valid else None


# =============================================================================
# LOAD ALL SOURCES
# =============================================================================

def load_kaggle(start_date: str) -> pd.DataFrame:
    """Load Kaggle betting data - our primary source through 2025."""
    print("\n[1/8] Loading Kaggle betting data (2007-2025)...")
    
    df = pd.read_csv(KAGGLE_FILE)
    df["game_date"] = pd.to_datetime(df["date"])
    df = df[df["game_date"] >= start_date].copy()
    
    # Standardize team names
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
    
    # Q1-Q4 scores
    for q in [1, 2, 3, 4]:
        df[f"home_q{q}"] = df[f"q{q}_home"]
        df[f"away_q{q}"] = df[f"q{q}_away"]
    
    # Half scores
    df["home_1h"] = df["home_q1"].fillna(0) + df["home_q2"].fillna(0)
    df["away_1h"] = df["away_q1"].fillna(0) + df["away_q2"].fillna(0)
    df["1h_total_actual"] = df["home_1h"] + df["away_1h"]
    df["1h_margin"] = df["home_1h"] - df["away_1h"]
    
    df["home_2h"] = df["home_q3"].fillna(0) + df["home_q4"].fillna(0)
    df["away_2h"] = df["away_q3"].fillna(0) + df["away_q4"].fillna(0)
    
    # FG lines
    df["kaggle_fg_spread"] = df.apply(
        lambda r: -r["spread"] if r.get("whos_favored") == "home" else r["spread"], axis=1
    )
    df["kaggle_fg_total"] = df["total"]
    df["kaggle_fg_ml_home"] = df["moneyline_home"]
    df["kaggle_fg_ml_away"] = df["moneyline_away"]
    
    # Season info
    df["season"] = df["season"]
    df["is_playoffs"] = df["playoffs"].fillna(0).astype(int)
    
    print(f"       Games: {len(df):,}")
    print(f"       Date range: {df['game_date'].min().date()} to {df['game_date'].max().date()}")
    return df


def load_theodds_derived() -> pd.DataFrame:
    """Load pre-computed TheOdds consensus lines."""
    print("\n[2/8] Loading TheOdds derived lines...")
    
    if not THEODDS_LINES.exists():
        print("       [SKIP] Not found")
        return pd.DataFrame()
    
    df = pd.read_csv(THEODDS_LINES)
    df["commence_time"] = pd.to_datetime(df["commence_time"])
    df["home_team"] = df["home_team"].apply(standardize_team_name)
    df["away_team"] = df["away_team"].apply(standardize_team_name)
    df["match_key"] = df.apply(
        lambda r: generate_match_key(r["commence_time"], r["home_team"], r["away_team"], source_is_utc=True),
        axis=1
    )
    
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
    
    print(f"       Games: {len(df):,}")
    return df


def load_theodds_2025_26() -> pd.DataFrame:
    """Load 2025-26 FG odds from TheOdds API fetch."""
    print("\n[2b/8] Loading TheOdds 2025-26 odds...")
    
    if not THEODDS_2025_26.exists():
        print("       [SKIP] Not found")
        return pd.DataFrame()
    
    df = pd.read_csv(THEODDS_2025_26)
    df["commence_time"] = pd.to_datetime(df["commence_time"])
    df["home_team"] = df["home_team"].apply(standardize_team_name)
    df["away_team"] = df["away_team"].apply(standardize_team_name)
    df["match_key"] = df.apply(
        lambda r: generate_match_key(r["commence_time"], r["home_team"], r["away_team"], source_is_utc=True),
        axis=1
    )
    
    # Extract consensus lines from bookmakers (median across available)
    spread_cols = [c for c in df.columns if "spreads" in c and "point" in c and "Home" in c]
    total_cols = [c for c in df.columns if "totals" in c and "point" in c and "Over" in c]
    ml_home_cols = [c for c in df.columns if "h2h" in c and "price" in c and "Home" not in c]
    
    if spread_cols:
        df["to_fg_spread"] = df[spread_cols].median(axis=1)
    if total_cols:
        df["to_fg_total"] = df[total_cols].median(axis=1)
    
    print(f"       Games: {len(df):,}")
    return df


def load_h1_exports() -> pd.DataFrame:
    """Load detailed 1H odds exports."""
    print("\n[3/8] Loading 1H odds exports...")
    
    dfs = []
    for f in H1_EXPORTS:
        if f.exists():
            df = pd.read_csv(f)
            dfs.append(df)
            print(f"       {f.name}: {len(df):,} rows")
    
    if not dfs:
        return pd.DataFrame()
    
    df = pd.concat(dfs, ignore_index=True)
    df["home_team"] = df["home_team"].apply(standardize_team_name)
    df["away_team"] = df["away_team"].apply(standardize_team_name)
    df["commence_time"] = pd.to_datetime(df["commence_time"])
    df["match_key"] = df.apply(
        lambda r: generate_match_key(r["commence_time"], r["home_team"], r["away_team"], source_is_utc=True),
        axis=1
    )
    
    # Aggregate by game
    lines = []
    for match_key, group in df.groupby("match_key"):
        spreads, totals, ml_h, ml_a = [], [], [], []
        home_team = group.iloc[0]["home_team"]
        away_team = group.iloc[0]["away_team"]
        
        for _, row in group.iterrows():
            market = row.get("market_key", "")
            outcome = row.get("outcome_name", "")
            point = row.get("outcome_point")
            price = row.get("outcome_price")
            
            if market == "spreads_h1" and outcome == home_team:
                if pd.notna(point): spreads.append(point)
            elif market == "totals_h1" and outcome == "Over":
                if pd.notna(point): totals.append(point)
            elif market == "h2h_h1":
                if outcome == home_team and pd.notna(price): ml_h.append(price)
                elif outcome == away_team and pd.notna(price): ml_a.append(price)
        
        lines.append({
            "match_key": match_key,
            "exp_1h_spread": safe_median(spreads),
            "exp_1h_total": safe_median(totals),
            "exp_1h_ml_home": safe_median(ml_h),
            "exp_1h_ml_away": safe_median(ml_a),
        })
    
    result = pd.DataFrame(lines)
    print(f"       Aggregated: {len(result):,} games")
    return result


def load_line_movement() -> pd.DataFrame:
    """Load featured odds for line movement."""
    print("\n[4/8] Loading line movement data...")
    
    dfs = []
    for f in FEAT_EXPORTS:
        if f.exists():
            df = pd.read_csv(f)
            dfs.append(df)
            print(f"       {f.name}: {len(df):,} rows")
    
    if not dfs:
        return pd.DataFrame()
    
    df = pd.concat(dfs, ignore_index=True)
    df["home_team"] = df["home_team"].apply(standardize_team_name)
    df["away_team"] = df["away_team"].apply(standardize_team_name)
    df["commence_time"] = pd.to_datetime(df["commence_time"])
    df["snapshot_timestamp"] = pd.to_datetime(df["snapshot_timestamp"])
    df["match_key"] = df.apply(
        lambda r: generate_match_key(r["commence_time"], r["home_team"], r["away_team"], source_is_utc=True),
        axis=1
    )
    
    movements = []
    for match_key, group in df.groupby("match_key"):
        group = group.sort_values("snapshot_timestamp")
        home_team = group.iloc[0]["home_team"]
        
        # Spreads
        spread_rows = group[(group["market_key"] == "spreads") & (group["outcome_name"] == home_team)]
        if len(spread_rows) > 0:
            spread_rows = spread_rows.sort_values("snapshot_timestamp")
            open_spread = spread_rows.iloc[0]["outcome_point"]
            close_spread = spread_rows.iloc[-1]["outcome_point"]
        else:
            open_spread = close_spread = None
        
        # Totals
        total_rows = group[(group["market_key"] == "totals") & (group["outcome_name"] == "Over")]
        if len(total_rows) > 0:
            total_rows = total_rows.sort_values("snapshot_timestamp")
            open_total = total_rows.iloc[0]["outcome_point"]
            close_total = total_rows.iloc[-1]["outcome_point"]
        else:
            open_total = close_total = None
        
        movements.append({
            "match_key": match_key,
            "open_spread": open_spread,
            "close_spread": close_spread,
            "spread_move": (close_spread - open_spread) if pd.notna(open_spread) and pd.notna(close_spread) else None,
            "open_total": open_total,
            "close_total": close_total,
            "total_move": (close_total - open_total) if pd.notna(open_total) and pd.notna(close_total) else None,
        })
    
    result = pd.DataFrame(movements)
    print(f"       Aggregated: {len(result):,} games")
    return result


def load_box_scores() -> pd.DataFrame:
    """Load box scores (through June 2023)."""
    print("\n[5/8] Loading box scores...")
    
    if not BOX_SCORES.exists():
        print("       [SKIP] Not found")
        return pd.DataFrame()
    
    df = pd.read_csv(BOX_SCORES)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["home_team"] = df["team_name_home"].apply(standardize_team_name)
    df["away_team"] = df["team_name_away"].apply(standardize_team_name)
    df["match_key"] = df.apply(
        lambda r: generate_match_key(r["game_date"], r["home_team"], r["away_team"], source_is_utc=False),
        axis=1
    )
    
    # Compute advanced stats
    for side in ["home", "away"]:
        opp = "away" if side == "home" else "home"
        df[f"{side}_efg_pct"] = (df[f"fgm_{side}"] + 0.5 * df[f"fg3m_{side}"]) / df[f"fga_{side}"].replace(0, np.nan)
        df[f"{side}_3p_rate"] = df[f"fg3a_{side}"] / df[f"fga_{side}"].replace(0, np.nan)
        df[f"{side}_ft_rate"] = df[f"ftm_{side}"] / df[f"fga_{side}"].replace(0, np.nan)
        df[f"{side}_tov_pct"] = df[f"tov_{side}"] / (df[f"fga_{side}"] + 0.44 * df[f"fta_{side}"] + df[f"tov_{side}"]).replace(0, np.nan)
        df[f"{side}_oreb_pct"] = df[f"oreb_{side}"] / (df[f"oreb_{side}"] + df[f"dreb_{opp}"]).replace(0, np.nan)
        df[f"{side}_poss"] = df[f"fga_{side}"] - df[f"oreb_{side}"] + df[f"tov_{side}"] + 0.44 * df[f"fta_{side}"]
    
    for side in ["home", "away"]:
        opp = "away" if side == "home" else "home"
        df[f"{side}_off_rtg"] = df[f"pts_{side}"] / df[f"{side}_poss"].replace(0, np.nan) * 100
        df[f"{side}_def_rtg"] = df[f"pts_{opp}"] / df[f"{opp}_poss"].replace(0, np.nan) * 100
        df[f"{side}_net_rtg"] = df[f"{side}_off_rtg"] - df[f"{side}_def_rtg"]
    
    df["pace"] = (df["home_poss"] + df["away_poss"]) / 2
    
    cols = ["match_key", "pace"]
    for side in ["home", "away"]:
        cols.extend([f"{side}_efg_pct", f"{side}_3p_rate", f"{side}_ft_rate", 
                     f"{side}_tov_pct", f"{side}_oreb_pct",
                     f"{side}_off_rtg", f"{side}_def_rtg", f"{side}_net_rtg"])
    
    result = df[cols].drop_duplicates("match_key")
    print(f"       Games: {len(result):,}")
    print(f"       Date range: {df['game_date'].min().date()} to {df['game_date'].max().date()}")
    return result


# =============================================================================
# MERGE ALL SOURCES
# =============================================================================

def merge_all(kaggle, theodds, h1_exp, movement, box) -> pd.DataFrame:
    """Merge all data sources."""
    print("\n[6/8] Merging all sources...")
    
    df = kaggle.copy()
    n = len(df)
    
    # TheOdds derived
    if not theodds.empty:
        cols = ["match_key", "to_fg_spread", "to_fg_total", "to_fg_ml_home", "to_fg_ml_away",
                "to_1h_spread", "to_1h_total", "to_1h_ml_home", "to_1h_ml_away"]
        theodds = theodds[[c for c in cols if c in theodds.columns]].drop_duplicates("match_key")
        df = df.merge(theodds, on="match_key", how="left")
        m = df["to_fg_spread"].notna().sum()
        print(f"       TheOdds derived: {m:,}/{n:,} ({m/n*100:.1f}%)")
    
    # 1H exports
    if not h1_exp.empty:
        df = df.merge(h1_exp.drop_duplicates("match_key"), on="match_key", how="left")
        m = df["exp_1h_spread"].notna().sum()
        print(f"       1H exports: {m:,}/{n:,} ({m/n*100:.1f}%)")
    
    # Line movement
    if not movement.empty:
        df = df.merge(movement.drop_duplicates("match_key"), on="match_key", how="left")
        m = df["spread_move"].notna().sum()
        print(f"       Line movement: {m:,}/{n:,} ({m/n*100:.1f}%)")
    
    # Box scores
    if not box.empty:
        df = df.merge(box.drop_duplicates("match_key"), on="match_key", how="left")
        m = df["home_efg_pct"].notna().sum()
        print(f"       Box scores: {m:,}/{n:,} ({m/n*100:.1f}%)")
    
    # Consolidate lines
    df["fg_spread_line"] = df["to_fg_spread"].fillna(df["kaggle_fg_spread"])
    df["fg_total_line"] = df["to_fg_total"].fillna(df["kaggle_fg_total"])
    df["fg_ml_home"] = df["to_fg_ml_home"].fillna(df["kaggle_fg_ml_home"])
    df["fg_ml_away"] = df["to_fg_ml_away"].fillna(df["kaggle_fg_ml_away"])
    
    df["1h_spread_line"] = df.get("exp_1h_spread", df.get("to_1h_spread"))
    df["1h_total_line"] = df.get("exp_1h_total", df.get("to_1h_total"))
    df["1h_ml_home"] = df.get("exp_1h_ml_home", df.get("to_1h_ml_home"))
    df["1h_ml_away"] = df.get("exp_1h_ml_away", df.get("to_1h_ml_away"))
    
    return df


# =============================================================================
# COMPUTE FEATURES
# =============================================================================

def compute_elo(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ELO from results."""
    print("\n[7/8] Computing ELO ratings...")
    
    df = df.sort_values("game_date").copy()
    K, HOME_ADV = 20, 100
    elo = {t: 1500 for t in set(df["home_team"]) | set(df["away_team"])}
    
    h_elo, a_elo = [], []
    for _, r in df.iterrows():
        h, a = r["home_team"], r["away_team"]
        h_elo.append(elo[h])
        a_elo.append(elo[a])
        
        exp = 1 / (1 + 10 ** ((elo[a] - elo[h] - HOME_ADV) / 400))
        win = 1 if r["home_score"] > r["away_score"] else 0
        elo[h] += K * (win - exp)
        elo[a] += K * ((1-win) - (1-exp))
    
    df["home_elo"] = h_elo
    df["away_elo"] = a_elo
    df["elo_diff"] = df["home_elo"] - df["away_elo"]
    df["elo_prob"] = 1 / (1 + 10 ** ((df["away_elo"] - df["home_elo"] - HOME_ADV) / 400))
    
    print(f"       Computed for {len(df):,} games")
    return df


def compute_rolling(df: pd.DataFrame, windows=[5, 10, 20]) -> pd.DataFrame:
    """Compute rolling stats from scores."""
    print("\n[8/8] Computing rolling stats...")
    
    df = df.sort_values("game_date").copy()
    
    # Stats from scores (available for all games)
    score_stats = ["score", "1h", "q1", "q2", "q3", "q4"]
    # Advanced stats (only where box scores available)
    adv_stats = ["efg_pct", "3p_rate", "ft_rate", "tov_pct", "oreb_pct", "off_rtg", "def_rtg", "net_rtg"]
    
    new_cols = {}
    
    for w in windows:
        for side in ["home", "away"]:
            team = f"{side}_team"
            
            # Score-based stats (100% coverage)
            for stat in score_stats:
                col = f"{side}_{stat}"
                if col in df.columns:
                    new_cols[f"{side}_{w}g_{stat}"] = df.groupby(team)[col].transform(
                        lambda x: x.shift(1).rolling(w, min_periods=3).mean()
                    )
            
            # Advanced stats (22% coverage - only through June 2023)
            for stat in adv_stats:
                col = f"{side}_{stat}"
                if col in df.columns:
                    new_cols[f"{side}_{w}g_{stat}"] = df.groupby(team)[col].transform(
                        lambda x: x.shift(1).rolling(w, min_periods=3).mean()
                    )
    
    # Add differentials
    for w in windows:
        for stat in score_stats + adv_stats:
            h_col = f"home_{w}g_{stat}"
            a_col = f"away_{w}g_{stat}"
            if h_col in new_cols and a_col in new_cols:
                new_cols[f"diff_{w}g_{stat}"] = new_cols[h_col] - new_cols[a_col]
    
    # Add all at once
    for col, vals in new_cols.items():
        df[col] = vals
    
    print(f"       Added {len(new_cols)} rolling columns")
    return df


def compute_situational(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rest, B2B, streaks."""
    df = df.sort_values("game_date").copy()
    
    team_dates = {}
    rest_h, rest_a = [], []
    for _, r in df.iterrows():
        h, a = r["home_team"], r["away_team"]
        d = r["game_date"]
        rest_h.append((d - team_dates.get(h, d - timedelta(days=7))).days if h in team_dates else 7)
        rest_a.append((d - team_dates.get(a, d - timedelta(days=7))).days if a in team_dates else 7)
        team_dates[h] = team_dates[a] = d
    
    df["home_rest"] = rest_h
    df["away_rest"] = rest_a
    df["home_b2b"] = (df["home_rest"] == 1).astype(int)
    df["away_b2b"] = (df["away_rest"] == 1).astype(int)
    df["rest_adv"] = df["home_rest"] - df["away_rest"]
    
    # Streaks
    team_streaks = {}
    str_h, str_a = [], []
    for _, r in df.iterrows():
        h, a = r["home_team"], r["away_team"]
        str_h.append(team_streaks.get(h, 0))
        str_a.append(team_streaks.get(a, 0))
        win = r["home_score"] > r["away_score"]
        team_streaks[h] = max(1, team_streaks.get(h,0)+1) if win else min(-1, team_streaks.get(h,0)-1)
        team_streaks[a] = max(1, team_streaks.get(a,0)+1) if not win else min(-1, team_streaks.get(a,0)-1)
    
    df["home_streak"] = str_h
    df["away_streak"] = str_a
    df["streak_diff"] = df["home_streak"] - df["away_streak"]
    
    return df


def compute_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Compute betting outcomes."""
    df["fg_spread_covered"] = np.where(df["fg_spread_line"].notna(), (df["fg_margin"] + df["fg_spread_line"] > 0).astype(int), np.nan)
    df["fg_total_over"] = np.where(df["fg_total_line"].notna(), (df["fg_total_actual"] > df["fg_total_line"]).astype(int), np.nan)
    df["fg_home_win"] = (df["fg_margin"] > 0).astype(int)
    df["1h_spread_covered"] = np.where(df["1h_spread_line"].notna(), (df["1h_margin"] + df["1h_spread_line"] > 0).astype(int), np.nan)
    df["1h_total_over"] = np.where(df["1h_total_line"].notna(), (df["1h_total_actual"] > df["1h_total_line"]).astype(int), np.nan)
    df["1h_home_win"] = (df["1h_margin"] > 0).astype(int)
    return df


def print_summary(df: pd.DataFrame):
    print("\n" + "="*80)
    print(" FINAL TRAINING DATA")
    print("="*80)
    print(f"\n  Games: {len(df):,}")
    print(f"  Date range: {df['game_date'].min().date()} to {df['game_date'].max().date()}")
    print(f"  Columns: {len(df.columns)}")
    
    print("\n  COVERAGE:")
    for name, col in [("FG spread", "fg_spread_line"), ("FG total", "fg_total_line"),
                      ("1H spread", "1h_spread_line"), ("1H total", "1h_total_line"),
                      ("Line movement", "spread_move"), ("Box scores", "home_efg_pct"),
                      ("ELO", "home_elo")]:
        if col in df.columns:
            pct = df[col].notna().mean() * 100
            print(f"    {name}: {pct:.1f}%")
    
    print("\n  LABELS:")
    for lab in ["fg_spread_covered", "fg_total_over", "1h_spread_covered", "1h_total_over"]:
        if lab in df.columns:
            n = df[lab].notna().sum()
            print(f"    {lab}: {n:,} games")


def main(start_date: str = "2023-01-01"):
    print("\n" + "="*80)
    print(" BUILDING COMPLETE TRAINING DATA (2023-2026)")
    print("="*80)
    
    kaggle = load_kaggle(start_date)
    theodds = load_theodds_derived()
    theodds_2526 = load_theodds_2025_26()
    h1_exp = load_h1_exports()
    movement = load_line_movement()
    box = load_box_scores()
    
    # Combine theodds (2021-2025) with theodds_2526 (2025-26)
    if not theodds_2526.empty:
        theodds = pd.concat([theodds, theodds_2526], ignore_index=True).drop_duplicates(subset=["match_key"], keep="last")
        print(f"\n       Combined TheOdds: {len(theodds):,} games")
    
    df = merge_all(kaggle, theodds, h1_exp, movement, box)
    df = compute_elo(df)
    df = compute_rolling(df)
    df = compute_situational(df)
    df = compute_labels(df)
    
    print_summary(df)
    
    out = OUTPUT_DIR / f"training_data_complete_{start_date[:4]}.csv"
    df.to_csv(out, index=False)
    print(f"\n  Saved: {out}")
    
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2023-01-01")
    args = parser.parse_args()
    main(args.start_date)
