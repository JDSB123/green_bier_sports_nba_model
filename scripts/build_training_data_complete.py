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
THEODDS_2025_26 = DATA_DIR / "historical" / "the_odds" / "2025-2026" / "2025-2026_all_markets.csv"
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
    """Load 2025-26 ALL markets (FG + 1H + Q1 + alternates) from TheOdds API fetch."""
    print("\n[2b/8] Loading TheOdds 2025-26 odds (all markets)...")

    if not THEODDS_2025_26.exists():
        print("       [SKIP] Not found")
        return pd.DataFrame()

    df = pd.read_csv(THEODDS_2025_26)
    df["commence_time"] = pd.to_datetime(df["commence_time"])
    df["game_date"] = df["commence_time"].dt.tz_convert(CST).dt.date
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["home_team"] = df["home_team"].apply(standardize_team_name)
    df["away_team"] = df["away_team"].apply(standardize_team_name)
    df["match_key"] = df.apply(
        lambda r: generate_match_key(r["commence_time"], r["home_team"], r["away_team"], source_is_utc=True),
        axis=1
    )

    # The all_markets.csv already has extracted consensus lines
    # Rename to match expected column names
    rename_map = {
        "fg_spread": "to_fg_spread",
        "fg_total": "to_fg_total",
        "fg_ml_home": "to_fg_ml_home",
        "fg_ml_away": "to_fg_ml_away",
        "h1_spread": "to_1h_spread",
        "h1_total": "to_1h_total",
        "h1_ml_home": "to_1h_ml_home",
        "h1_ml_away": "to_1h_ml_away",
        "q1_spread": "to_q1_spread",
        "q1_total": "to_q1_total",
    }
    for old, new in rename_map.items():
        if old in df.columns:
            df[new] = df[old]

    fg_count = df["to_fg_spread"].notna().sum() if "to_fg_spread" in df.columns else 0
    h1_count = df["to_1h_spread"].notna().sum() if "to_1h_spread" in df.columns else 0
    q1_count = df["to_q1_spread"].notna().sum() if "to_q1_spread" in df.columns else 0
    print(f"       Games: {len(df):,} (FG: {fg_count}, 1H: {h1_count}, Q1: {q1_count})")
    
    # Print date range
    print(f"       Date range: {df['game_date'].min().date()} to {df['game_date'].max().date()}")
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
    """Load box scores (through June 2023 from wyattowalsh)."""
    print("\n[5/8] Loading wyattowalsh box scores (1946-2023)...")
    
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


def load_nba_api_box_scores() -> pd.DataFrame:
    """Load nba_api box scores for 2023-26 seasons."""
    print("\n[5b/8] Loading nba_api box scores (2023-2026)...")
    
    all_games = []
    for path in NBA_API_BOX_SCORES:
        if not path.exists():
            continue
        
        df = pd.read_csv(path)
        print(f"       {path.name}: {len(df)} rows")
        
        # Group by game_id
        for game_id, group in df.groupby("GAME_ID"):
            if len(group) != 2:
                continue
            
            row1, row2 = group.iloc[0], group.iloc[1]
            home_team = standardize_team_name(row2["TEAM_NAME"])  # Home is usually second
            away_team = standardize_team_name(row1["TEAM_NAME"])
            
            # Parse date from GAME_ID (format: 00XXYYYMMDD)
            game_date_str = str(game_id)[2:]  # Remove leading zeros
            
            game = {
                "game_id": game_id,
                "home_team": home_team,
                "away_team": away_team,
                "home_score": row2["PTS"],
                "away_score": row1["PTS"],
            }
            
            # All box score stats
            for stat in ["FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "DREB", "AST", "STL", "BLK", "TO", "PF"]:
                game[f"home_{stat.lower()}"] = row2[stat] if stat in row2 else None
                game[f"away_{stat.lower()}"] = row1[stat] if stat in row1 else None
            
            all_games.append(game)
    
    if not all_games:
        return pd.DataFrame()
    
    result = pd.DataFrame(all_games)
    
    # Compute advanced stats
    for side in ["home", "away"]:
        opp = "away" if side == "home" else "home"
        fga = result[f"{side}_fga"].replace(0, np.nan)
        result[f"{side}_efg_pct"] = (result[f"{side}_fgm"] + 0.5 * result[f"{side}_fg3m"]) / fga
        result[f"{side}_3p_rate"] = result[f"{side}_fg3a"] / fga
        result[f"{side}_ft_rate"] = result[f"{side}_ftm"] / fga
        poss = fga + 0.44 * result[f"{side}_fta"] + result[f"{side}_to"]
        result[f"{side}_tov_pct"] = result[f"{side}_to"] / poss.replace(0, np.nan)
        result[f"{side}_oreb_pct"] = result[f"{side}_oreb"] / (result[f"{side}_oreb"] + result[f"{opp}_dreb"]).replace(0, np.nan)
        result[f"{side}_poss"] = fga - result[f"{side}_oreb"] + result[f"{side}_to"] + 0.44 * result[f"{side}_fta"]
    
    for side in ["home", "away"]:
        opp = "away" if side == "home" else "home"
        result[f"{side}_off_rtg"] = result[f"{side}_score"] / result[f"{side}_poss"].replace(0, np.nan) * 100
        result[f"{side}_def_rtg"] = result[f"{opp}_score"] / result[f"{opp}_poss"].replace(0, np.nan) * 100
        result[f"{side}_net_rtg"] = result[f"{side}_off_rtg"] - result[f"{side}_def_rtg"]
    
    result["pace"] = (result["home_poss"] + result["away_poss"]) / 2
    
    print(f"       Total games: {len(result):,}")
    return result


# =============================================================================
# MERGE ALL SOURCES
# =============================================================================

def merge_all(kaggle, theodds, theodds_2526, h1_exp, movement, box_old, box_new) -> pd.DataFrame:
    """Merge all data sources including 2025-26 as new rows."""
    print("\n[6/8] Merging all sources...")
    
    df = kaggle.copy()
    kaggle_keys = set(df["match_key"])
    
    # ADD 2025-26 games as NEW ROWS (not just merge)
    if not theodds_2526.empty:
        # Get games not in Kaggle
        new_games = theodds_2526[~theodds_2526["match_key"].isin(kaggle_keys)].copy()
        
        if not new_games.empty:
            # Create base structure matching Kaggle
            new_rows = []
            for _, row in new_games.iterrows():
                new_row = {
                    "match_key": row["match_key"],
                    "game_date": pd.to_datetime(row.get("commence_time", row.get("game_date"))).tz_localize(None) if pd.notna(row.get("commence_time", row.get("game_date"))) else None,
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "season": "2025-26",
                    "to_fg_spread": row.get("to_fg_spread", row.get("fg_spread")),
                    "to_fg_total": row.get("to_fg_total", row.get("fg_total")),
                    "to_fg_ml_home": row.get("fg_ml_home"),
                    "to_fg_ml_away": row.get("fg_ml_away"),
                    "to_1h_spread": row.get("to_1h_spread", row.get("h1_spread")),
                    "to_1h_total": row.get("to_1h_total", row.get("h1_total")),
                    "to_1h_ml_home": row.get("h1_ml_home"),
                    "to_1h_ml_away": row.get("h1_ml_away"),
                }
                new_rows.append(new_row)
            
            new_df = pd.DataFrame(new_rows)
            df = pd.concat([df, new_df], ignore_index=True)
            print(f"       Added 2025-26 games: {len(new_rows):,}")
    
    n = len(df)
    
    # TheOdds derived (2021-2025)
    if not theodds.empty:
        cols = ["match_key", "to_fg_spread", "to_fg_total", "to_fg_ml_home", "to_fg_ml_away",
                "to_1h_spread", "to_1h_total", "to_1h_ml_home", "to_1h_ml_away"]
        theodds_merge = theodds[[c for c in cols if c in theodds.columns]].drop_duplicates("match_key")
        df = df.merge(theodds_merge, on="match_key", how="left", suffixes=("", "_dup"))
        # Fill missing with duplicate columns
        for col in ["to_fg_spread", "to_fg_total", "to_fg_ml_home", "to_fg_ml_away",
                    "to_1h_spread", "to_1h_total", "to_1h_ml_home", "to_1h_ml_away"]:
            if f"{col}_dup" in df.columns:
                df[col] = df[col].fillna(df[f"{col}_dup"])
                df = df.drop(columns=[f"{col}_dup"])
        m = df["to_fg_spread"].notna().sum()
        print(f"       TheOdds derived: {m:,}/{n:,} ({m/n*100:.1f}%)")

    # 1H exports
    if not h1_exp.empty:
        df = df.merge(h1_exp.drop_duplicates("match_key"), on="match_key", how="left")
        m = df["exp_1h_spread"].notna().sum() if "exp_1h_spread" in df.columns else 0
        print(f"       1H exports: {m:,}/{n:,} ({m/n*100:.1f}%)")

    # Line movement
    if not movement.empty:
        df = df.merge(movement.drop_duplicates("match_key"), on="match_key", how="left")
        m = df["spread_move"].notna().sum() if "spread_move" in df.columns else 0
        print(f"       Line movement: {m:,}/{n:,} ({m/n*100:.1f}%)")

    # Box scores (wyattowalsh - older)
    if not box_old.empty:
        df = df.merge(box_old.drop_duplicates("match_key"), on="match_key", how="left")
        m = df["home_efg_pct"].notna().sum() if "home_efg_pct" in df.columns else 0
        print(f"       Box scores (wyatt): {m:,}/{n:,} ({m/n*100:.1f}%)")

    # Box scores (nba_api - 2023-26)
    if not box_new.empty:
        # Match by teams since we don't have match_key
        for idx, row in df.iterrows():
            if pd.notna(df.at[idx, "home_efg_pct"]) if "home_efg_pct" in df.columns else False:
                continue  # Already have box score
            
            home, away = row["home_team"], row["away_team"]
            match = box_new[
                ((box_new["home_team"] == home) & (box_new["away_team"] == away)) |
                ((box_new["home_team"] == away) & (box_new["away_team"] == home))
            ]
            
            if len(match) > 0:
                m = match.iloc[0]
                is_flipped = m["home_team"] != home
                
                for stat in ["score", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", 
                            "oreb", "dreb", "ast", "stl", "blk", "to", "pf",
                            "efg_pct", "3p_rate", "ft_rate", "tov_pct", "oreb_pct",
                            "off_rtg", "def_rtg", "net_rtg"]:
                    h_col, a_col = f"home_{stat}", f"away_{stat}"
                    if h_col in m and a_col in m:
                        if is_flipped:
                            df.at[idx, h_col] = m[a_col]
                            df.at[idx, a_col] = m[h_col]
                        else:
                            df.at[idx, h_col] = m[h_col]
                            df.at[idx, a_col] = m[a_col]
                if "pace" in m:
                    df.at[idx, "pace"] = m["pace"]
        
        m = df["home_efg_pct"].notna().sum() if "home_efg_pct" in df.columns else 0
        print(f"       Box scores (nba_api): {m:,}/{n:,} ({m/n*100:.1f}%)")

    # Consolidate lines
    if "kaggle_fg_spread" in df.columns:
        df["fg_spread_line"] = df.get("to_fg_spread", pd.Series()).fillna(df.get("kaggle_fg_spread", pd.Series()))
        df["fg_total_line"] = df.get("to_fg_total", pd.Series()).fillna(df.get("kaggle_fg_total", pd.Series()))
        df["fg_ml_home"] = df.get("to_fg_ml_home", pd.Series()).fillna(df.get("kaggle_fg_ml_home", pd.Series()))
        df["fg_ml_away"] = df.get("to_fg_ml_away", pd.Series()).fillna(df.get("kaggle_fg_ml_away", pd.Series()))
    else:
        df["fg_spread_line"] = df.get("to_fg_spread")
        df["fg_total_line"] = df.get("to_fg_total")
        df["fg_ml_home"] = df.get("to_fg_ml_home")
        df["fg_ml_away"] = df.get("to_fg_ml_away")

    # 1H lines: prefer exports, fallback to TheOdds
    if "exp_1h_spread" in df.columns and "to_1h_spread" in df.columns:
        df["1h_spread_line"] = df["exp_1h_spread"].fillna(df["to_1h_spread"])
    elif "exp_1h_spread" in df.columns:
        df["1h_spread_line"] = df["exp_1h_spread"]
    elif "to_1h_spread" in df.columns:
        df["1h_spread_line"] = df["to_1h_spread"]
    
    if "exp_1h_total" in df.columns and "to_1h_total" in df.columns:
        df["1h_total_line"] = df["exp_1h_total"].fillna(df["to_1h_total"])
    elif "exp_1h_total" in df.columns:
        df["1h_total_line"] = df["exp_1h_total"]
    elif "to_1h_total" in df.columns:
        df["1h_total_line"] = df["to_1h_total"]
    
    if "exp_1h_ml_home" in df.columns and "to_1h_ml_home" in df.columns:
        df["1h_ml_home"] = df["exp_1h_ml_home"].fillna(df["to_1h_ml_home"])
    elif "to_1h_ml_home" in df.columns:
        df["1h_ml_home"] = df["to_1h_ml_home"]
    
    if "exp_1h_ml_away" in df.columns and "to_1h_ml_away" in df.columns:
        df["1h_ml_away"] = df["exp_1h_ml_away"].fillna(df["to_1h_ml_away"])
    elif "to_1h_ml_away" in df.columns:
        df["1h_ml_away"] = df["to_1h_ml_away"]

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
    box_old = load_box_scores()
    box_new = load_nba_api_box_scores()

    df = merge_all(kaggle, theodds, theodds_2526, h1_exp, movement, box_old, box_new)
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
