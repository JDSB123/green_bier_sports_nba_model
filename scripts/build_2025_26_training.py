#!/usr/bin/env python3
"""
Build training data specifically for 2025-26 season.

Uses:
1. TheOdds 2025-26 FG odds (578 games)
2. nba_api box scores (567 games)
3. Computed ELO and rolling stats
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.standardization import standardize_team_name, generate_match_key, CST

# Paths
DATA_DIR = PROJECT_ROOT / "data"
THEODDS_2526 = DATA_DIR / "historical" / "the_odds" / "2025-2026" / "2025-2026_all_markets.csv"
BOX_SCORES = DATA_DIR / "raw" / "nba_api" / "box_scores_2025_26.csv"
OUTPUT_DIR = DATA_DIR / "processed"


def load_theodds_2526():
    """Load 2025-26 odds (FG + 1H) with game info."""
    print("\n[1/5] Loading TheOdds 2025-26 odds (FG + 1H)...", flush=True)

    df = pd.read_csv(THEODDS_2526)
    df["game_date"] = pd.to_datetime(df["commence_time"]).dt.tz_convert(CST).dt.date
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["home_team"] = df["home_team"].apply(standardize_team_name)
    df["away_team"] = df["away_team"].apply(standardize_team_name)
    df["match_key"] = df.apply(
        lambda r: generate_match_key(r["game_date"], r["home_team"], r["away_team"], source_is_utc=False),
        axis=1
    )

    # all_markets.csv already has pre-computed consensus lines
    rename_map = {
        "fg_spread": "fg_spread_line",
        "fg_total": "fg_total_line",
        "h1_spread": "1h_spread_line",
        "h1_total": "1h_total_line",
    }
    for old, new in rename_map.items():
        if old in df.columns:
            df[new] = df[old]

    fg_count = df["fg_spread_line"].notna().sum() if "fg_spread_line" in df.columns else 0
    h1_count = df["1h_spread_line"].notna().sum() if "1h_spread_line" in df.columns else 0

    print(f"       Games: {len(df)}", flush=True)
    print(f"       FG: {fg_count}, 1H: {h1_count}", flush=True)
    print(f"       Date range: {df['game_date'].min().date()} to {df['game_date'].max().date()}", flush=True)

    cols = ["match_key", "game_date", "home_team", "away_team", "commence_time",
            "fg_spread_line", "fg_total_line"]
    if "1h_spread_line" in df.columns:
        cols.extend(["1h_spread_line", "1h_total_line"])
    
    return df[[c for c in cols if c in df.columns]].copy()


def load_box_scores():
    """Load 2025-26 box scores from nba_api."""
    print("\n[2/5] Loading 2025-26 box scores...", flush=True)
    
    if not BOX_SCORES.exists():
        print("       [SKIP] Not found", flush=True)
        return pd.DataFrame()
    
    df = pd.read_csv(BOX_SCORES)
    
    # Group by game_id to get home/away
    games = []
    for game_id, group in df.groupby("GAME_ID"):
        if len(group) != 2:
            continue
        
        # First team is away, second is home (NBA convention)
        row1, row2 = group.iloc[0], group.iloc[1]
        
        # Determine home/away by PLUS_MINUS (home usually listed second)
        # This is approximate - we'll match by team names to TheOdds data
        games.append({
            "game_id": game_id,
            "team1": standardize_team_name(row1["TEAM_NAME"]),
            "team1_pts": row1["PTS"],
            "team1_fgm": row1["FGM"], "team1_fga": row1["FGA"],
            "team1_fg3m": row1["FG3M"], "team1_fg3a": row1["FG3A"],
            "team1_ftm": row1["FTM"], "team1_fta": row1["FTA"],
            "team1_oreb": row1["OREB"], "team1_dreb": row1["DREB"],
            "team1_ast": row1["AST"], "team1_tov": row1["TO"],
            "team2": standardize_team_name(row2["TEAM_NAME"]),
            "team2_pts": row2["PTS"],
            "team2_fgm": row2["FGM"], "team2_fga": row2["FGA"],
            "team2_fg3m": row2["FG3M"], "team2_fg3a": row2["FG3A"],
            "team2_ftm": row2["FTM"], "team2_fta": row2["FTA"],
            "team2_oreb": row2["OREB"], "team2_dreb": row2["DREB"],
            "team2_ast": row2["AST"], "team2_tov": row2["TO"],
        })
    
    box_df = pd.DataFrame(games)
    print(f"       Games: {len(box_df)}", flush=True)
    return box_df


def merge_data(odds_df: pd.DataFrame, box_df: pd.DataFrame) -> pd.DataFrame:
    """Merge odds with box scores."""
    print("\n[3/5] Merging odds with box scores...", flush=True)
    
    if box_df.empty:
        print("       No box scores to merge", flush=True)
        return odds_df
    
    # Match by teams (ignoring home/away for now)
    merged = []
    for _, odds_row in odds_df.iterrows():
        home, away = odds_row["home_team"], odds_row["away_team"]
        
        # Find matching box score
        match = box_df[
            ((box_df["team1"] == home) & (box_df["team2"] == away)) |
            ((box_df["team1"] == away) & (box_df["team2"] == home))
        ]
        
        row = odds_row.to_dict()
        
        if len(match) > 0:
            m = match.iloc[0]
            # Determine which is home/away
            if m["team1"] == home:
                row["home_score"] = m["team1_pts"]
                row["away_score"] = m["team2_pts"]
                for stat in ["fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "ast", "tov"]:
                    row[f"home_{stat}"] = m[f"team1_{stat}"]
                    row[f"away_{stat}"] = m[f"team2_{stat}"]
            else:
                row["home_score"] = m["team2_pts"]
                row["away_score"] = m["team1_pts"]
                for stat in ["fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "ast", "tov"]:
                    row[f"home_{stat}"] = m[f"team2_{stat}"]
                    row[f"away_{stat}"] = m[f"team1_{stat}"]
        
        merged.append(row)
    
    df = pd.DataFrame(merged)
    matched = df["home_score"].notna().sum()
    print(f"       Matched: {matched}/{len(df)} ({100*matched/len(df):.1f}%)", flush=True)
    
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived features."""
    print("\n[4/5] Computing features...", flush=True)
    
    # Scores
    if "home_score" in df.columns:
        df["fg_margin"] = df["home_score"] - df["away_score"]
        df["fg_total_actual"] = df["home_score"] + df["away_score"]
    
    # Advanced stats
    for side in ["home", "away"]:
        if f"{side}_fga" in df.columns:
            fga = df[f"{side}_fga"].replace(0, np.nan)
            df[f"{side}_efg_pct"] = (df[f"{side}_fgm"] + 0.5 * df[f"{side}_fg3m"]) / fga
            df[f"{side}_3p_rate"] = df[f"{side}_fg3a"] / fga
            df[f"{side}_ft_rate"] = df[f"{side}_ftm"] / fga
            poss = fga + 0.44 * df[f"{side}_fta"] + df[f"{side}_tov"]
            df[f"{side}_tov_pct"] = df[f"{side}_tov"] / poss.replace(0, np.nan)
    
    # Labels
    if "fg_margin" in df.columns and "fg_spread_line" in df.columns:
        df["fg_spread_covered"] = (df["fg_margin"] + df["fg_spread_line"] > 0).astype(int)
    if "fg_total_actual" in df.columns and "fg_total_line" in df.columns:
        df["fg_total_over"] = (df["fg_total_actual"] > df["fg_total_line"]).astype(int)
    if "fg_margin" in df.columns:
        df["fg_home_win"] = (df["fg_margin"] > 0).astype(int)
    
    # Rest days (simplified - needs schedule data for accuracy)
    df = df.sort_values("game_date").reset_index(drop=True)
    
    print(f"       Features computed", flush=True)
    return df


def main():
    print("\n" + "=" * 60, flush=True)
    print(" BUILDING 2025-26 TRAINING DATA", flush=True)
    print("=" * 60, flush=True)
    
    odds = load_theodds_2526()
    box = load_box_scores()
    df = merge_data(odds, box)
    df = compute_features(df)
    
    print("\n[5/5] Summary...", flush=True)
    print(f"       Games: {len(df)}", flush=True)
    print(f"       Date range: {df['game_date'].min().date()} to {df['game_date'].max().date()}", flush=True)
    
    if "fg_spread_line" in df.columns:
        pct = df["fg_spread_line"].notna().mean() * 100
        print(f"       FG spread coverage: {pct:.1f}%", flush=True)
    
    if "home_score" in df.columns:
        pct = df["home_score"].notna().mean() * 100
        print(f"       Box score coverage: {pct:.1f}%", flush=True)
    
    output = OUTPUT_DIR / "training_data_2025_26.csv"
    df.to_csv(output, index=False)
    print(f"\n  Saved: {output}", flush=True)
    
    return df


if __name__ == "__main__":
    main()
