#!/usr/bin/env python3
"""
BUILD MAXIMUM TRAINING DATA

Uses EVERY available piece of historical data:
1. Kaggle (2008-2025): Games, scores, FG lines, Q1-Q4 scores
2. TheOdds API (2021-2025): FG lines, 1H lines (May 2023+)
3. wyattowalsh/basketball (1946-2023): Box scores, advanced stats
4. FiveThirtyEight ELO (1947-2015): Historical team strength

Data flow:
- Kaggle is the BASE (has most games with scores)
- Box scores merged where available (for advanced stats)
- TheOdds merged for betting lines
- ELO merged for team strength

Features computed:
- Basic: team names, scores, margins, totals
- Lines: FG spread/total, 1H spread/total, moneylines
- Advanced: Four Factors, pace, offensive/defensive ratings
- Rolling: 5/10/20 game windows of all stats
- Temporal: rest days, B2B, streaks, H2H
- Movement: Opening vs closing lines
"""
from __future__ import annotations

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

from src.data.standardization import (
    standardize_team_name,
    generate_match_key,
    to_cst_date,
    CST,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Paths
DATA_DIR = PROJECT_ROOT / "data"
KAGGLE_FILE = DATA_DIR / "external" / "kaggle" / "nba_2008-2025.csv"
BOX_SCORES_FILE = DATA_DIR / "external" / "nba_database" / "game.csv"
THEODDS_DIR = DATA_DIR / "historical" / "the_odds"
ELO_FILE = DATA_DIR / "raw" / "github" / "raw.githubusercontent.com_fivethirtyeight_data_master_nba-elo_nbaallelo.csv"
OUTPUT_DIR = DATA_DIR / "processed"


def safe_median(values: list) -> Optional[float]:
    """Compute median of non-null values."""
    valid = [v for v in values if v is not None and not np.isnan(v)]
    return median(valid) if valid else None


# =============================================================================
# 1. LOAD KAGGLE (BASE)
# =============================================================================

def load_kaggle(start_date: str) -> pd.DataFrame:
    """Load Kaggle as base dataset."""
    print("\n[1] Loading Kaggle data (BASE)...")
    
    df = pd.read_csv(KAGGLE_FILE)
    df["game_date"] = pd.to_datetime(df["date"])
    df = df[df["game_date"] >= start_date].copy()
    
    # Standardize
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
    df["2h_total_actual"] = df["home_2h"] + df["away_2h"]
    df["2h_margin"] = df["home_2h"] - df["away_2h"]
    
    # FG lines from Kaggle
    df["fg_spread_line"] = df.apply(
        lambda r: -r["spread"] if r.get("whos_favored") == "home" else r["spread"], 
        axis=1
    )
    df["fg_total_line"] = df["total"]
    df["fg_ml_home"] = df["moneyline_home"]
    df["fg_ml_away"] = df["moneyline_away"]
    
    print(f"    Loaded {len(df):,} games from Kaggle ({start_date} to {df['game_date'].max().date()})")
    
    return df


# =============================================================================
# 2. MERGE BOX SCORES (Advanced Stats)
# =============================================================================

def merge_box_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Merge box scores from wyattowalsh/basketball."""
    print("\n[2] Merging box scores (advanced stats)...")
    
    if not BOX_SCORES_FILE.exists():
        print("    [SKIP] Box scores file not found")
        return df
    
    box = pd.read_csv(BOX_SCORES_FILE)
    box["game_date"] = pd.to_datetime(box["game_date"])
    
    # Standardize
    box["home_team"] = box["team_name_home"].apply(standardize_team_name)
    box["away_team"] = box["team_name_away"].apply(standardize_team_name)
    box["match_key"] = box.apply(
        lambda r: generate_match_key(r["game_date"], r["home_team"], r["away_team"], source_is_utc=False),
        axis=1
    )
    
    # Compute advanced stats - first pass: everything except ratings
    for side in ["home", "away"]:
        opp = "away" if side == "home" else "home"
        
        # eFG% = (FGM + 0.5 * 3PM) / FGA
        box[f"{side}_efg_pct"] = (
            (box[f"fgm_{side}"] + 0.5 * box[f"fg3m_{side}"]) / 
            box[f"fga_{side}"].replace(0, np.nan)
        )
        
        # TOV% = TOV / (FGA + 0.44 * FTA + TOV)
        poss = box[f"fga_{side}"] + 0.44 * box[f"fta_{side}"] + box[f"tov_{side}"]
        box[f"{side}_tov_pct"] = box[f"tov_{side}"] / poss.replace(0, np.nan)
        
        # OREB% = OREB / (OREB + opp_DREB)
        total_reb = box[f"oreb_{side}"] + box[f"dreb_{opp}"]
        box[f"{side}_oreb_pct"] = box[f"oreb_{side}"] / total_reb.replace(0, np.nan)
        
        # FT rate
        box[f"{side}_ft_rate"] = box[f"ftm_{side}"] / box[f"fga_{side}"].replace(0, np.nan)
        
        # Shooting %
        box[f"{side}_fg_pct"] = box[f"fgm_{side}"] / box[f"fga_{side}"].replace(0, np.nan)
        box[f"{side}_3p_pct"] = box[f"fg3m_{side}"] / box[f"fg3a_{side}"].replace(0, np.nan)
        box[f"{side}_3p_rate"] = box[f"fg3a_{side}"] / box[f"fga_{side}"].replace(0, np.nan)
        
        # Pace (possessions) - compute for both sides first
        box[f"{side}_poss"] = box[f"fga_{side}"] - box[f"oreb_{side}"] + box[f"tov_{side}"] + 0.44 * box[f"fta_{side}"]
        
        # Other
        box[f"{side}_ast"] = box[f"ast_{side}"]
        box[f"{side}_tov"] = box[f"tov_{side}"]
        box[f"{side}_reb"] = box[f"reb_{side}"]
    
    # Second pass: ratings (need both poss columns)
    for side in ["home", "away"]:
        opp = "away" if side == "home" else "home"
        box[f"{side}_off_rtg"] = box[f"pts_{side}"] / box[f"{side}_poss"].replace(0, np.nan) * 100
        box[f"{side}_def_rtg"] = box[f"pts_{opp}"] / box[f"{opp}_poss"].replace(0, np.nan) * 100
        box[f"{side}_net_rtg"] = box[f"{side}_off_rtg"] - box[f"{side}_def_rtg"]
    
    # Differentials
    box["efg_diff"] = box["home_efg_pct"] - box["away_efg_pct"]
    box["tov_diff"] = box["away_tov_pct"] - box["home_tov_pct"]
    box["oreb_diff"] = box["home_oreb_pct"] - box["away_oreb_pct"]
    box["pace"] = (box["home_poss"] + box["away_poss"]) / 2
    box["net_rtg_diff"] = box["home_net_rtg"] - box["away_net_rtg"]
    
    # Select columns to merge
    box_cols = [
        "match_key",
        "home_efg_pct", "away_efg_pct", "efg_diff",
        "home_tov_pct", "away_tov_pct", "tov_diff",
        "home_oreb_pct", "away_oreb_pct", "oreb_diff",
        "home_ft_rate", "away_ft_rate",
        "home_fg_pct", "away_fg_pct",
        "home_3p_pct", "away_3p_pct",
        "home_3p_rate", "away_3p_rate",
        "home_off_rtg", "away_off_rtg",
        "home_def_rtg", "away_def_rtg",
        "home_net_rtg", "away_net_rtg", "net_rtg_diff",
        "pace",
        "home_ast", "away_ast",
        "home_tov", "away_tov",
        "home_reb", "away_reb",
    ]
    
    box_merge = box[box_cols].drop_duplicates(subset=["match_key"])
    
    # Merge
    before = len(df)
    df = df.merge(box_merge, on="match_key", how="left")
    matched = df["home_efg_pct"].notna().sum()
    
    print(f"    Merged {matched:,}/{before:,} games ({matched/before*100:.1f}%) with box score data")
    print(f"    Box score date range: {box['game_date'].min().date()} to {box['game_date'].max().date()}")
    
    return df


# =============================================================================
# 3. MERGE THEODDS (Betting Lines)
# =============================================================================

def merge_theodds(df: pd.DataFrame) -> pd.DataFrame:
    """Merge TheOdds API betting lines."""
    print("\n[3] Merging TheOdds API lines...")
    
    odds_dir = THEODDS_DIR / "odds"
    period_dir = THEODDS_DIR / "period_odds"
    
    # Load FG lines
    fg_rows = []
    for json_file in odds_dir.glob("*/*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
            events = data if isinstance(data, list) else data.get("data", [])
            
            for e in events:
                home = standardize_team_name(e.get("home_team", ""))
                away = standardize_team_name(e.get("away_team", ""))
                commence = e.get("commence_time", "")
                
                if not all([home, away, commence]):
                    continue
                
                spreads, totals, ml_h, ml_a = [], [], [], []
                for bm in e.get("bookmakers", []):
                    for mkt in bm.get("markets", []):
                        key = mkt.get("key", "")
                        for o in mkt.get("outcomes", []):
                            if key == "spreads" and o.get("name") == e.get("home_team"):
                                spreads.append(o.get("point"))
                            elif key == "totals" and o.get("name") == "Over":
                                totals.append(o.get("point"))
                            elif key == "h2h":
                                if o.get("name") == e.get("home_team"):
                                    ml_h.append(o.get("price"))
                                elif o.get("name") == e.get("away_team"):
                                    ml_a.append(o.get("price"))
                
                if spreads or totals:
                    fg_rows.append({
                        "match_key": generate_match_key(commence, home, away),
                        "to_fg_spread": safe_median(spreads),
                        "to_fg_total": safe_median(totals),
                        "to_fg_ml_home": safe_median(ml_h),
                        "to_fg_ml_away": safe_median(ml_a),
                    })
        except:
            continue
    
    if fg_rows:
        fg_df = pd.DataFrame(fg_rows).drop_duplicates(subset=["match_key"])
        df = df.merge(fg_df, on="match_key", how="left")
        matched = df["to_fg_spread"].notna().sum()
        print(f"    FG lines: {matched:,} games merged")
        
        # Fill missing Kaggle lines with TheOdds
        df["fg_spread_line"] = df["fg_spread_line"].fillna(df["to_fg_spread"])
        df["fg_total_line"] = df["fg_total_line"].fillna(df["to_fg_total"])
    
    # Load 1H lines
    h1_rows = []
    for json_file in period_dir.glob("*/period_odds_1h.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
            
            for wrapper in data.get("data", []):
                e = wrapper.get("data", wrapper)
                home = standardize_team_name(e.get("home_team", ""))
                away = standardize_team_name(e.get("away_team", ""))
                commence = e.get("commence_time", "")
                
                if not all([home, away, commence]):
                    continue
                
                spreads, totals, ml_h, ml_a = [], [], [], []
                for bm in e.get("bookmakers", []):
                    for mkt in bm.get("markets", []):
                        key = mkt.get("key", "")
                        for o in mkt.get("outcomes", []):
                            if key == "spreads_h1" and o.get("name") == e.get("home_team"):
                                spreads.append(o.get("point"))
                            elif key == "totals_h1" and o.get("name") == "Over":
                                totals.append(o.get("point"))
                            elif key == "h2h_h1":
                                if o.get("name") == e.get("home_team"):
                                    ml_h.append(o.get("price"))
                                elif o.get("name") == e.get("away_team"):
                                    ml_a.append(o.get("price"))
                
                h1_rows.append({
                    "match_key": generate_match_key(commence, home, away),
                    "1h_spread_line": safe_median(spreads),
                    "1h_total_line": safe_median(totals),
                    "1h_ml_home": safe_median(ml_h),
                    "1h_ml_away": safe_median(ml_a),
                })
        except:
            continue
    
    if h1_rows:
        h1_df = pd.DataFrame(h1_rows).drop_duplicates(subset=["match_key"])
        df = df.merge(h1_df, on="match_key", how="left")
        matched = df["1h_spread_line"].notna().sum()
        print(f"    1H lines: {matched:,} games merged")
    
    return df


# =============================================================================
# 4. COMPUTE ROLLING STATS
# =============================================================================

def compute_rolling_stats(df: pd.DataFrame, windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
    """Compute rolling averages for all numeric stats."""
    print("\n[4] Computing rolling stats...")
    
    df = df.sort_values("game_date").copy()
    
    # Stats to roll
    basic_stats = ["score", "1h", "2h"]
    for q in [1, 2, 3, 4]:
        basic_stats.append(f"q{q}")
    
    advanced_stats = [
        "efg_pct", "tov_pct", "oreb_pct", "ft_rate",
        "fg_pct", "3p_pct", "3p_rate",
        "off_rtg", "def_rtg", "net_rtg",
        "ast", "tov", "reb",
    ]
    
    all_stats = basic_stats + advanced_stats
    
    for window in windows:
        for side in ["home", "away"]:
            team_col = f"{side}_team"
            
            for stat in all_stats:
                col = f"{side}_{stat}"
                if col not in df.columns:
                    continue
                
                roll_col = f"{side}_{window}g_{stat}"
                df[roll_col] = df.groupby(team_col)[col].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=3).mean()
                )
        
        # Differentials
        for stat in all_stats:
            home_col = f"home_{window}g_{stat}"
            away_col = f"away_{window}g_{stat}"
            if home_col in df.columns and away_col in df.columns:
                df[f"diff_{window}g_{stat}"] = df[home_col] - df[away_col]
    
    roll_cols = [c for c in df.columns if any(f"{w}g_" in c for w in windows)]
    print(f"    Added {len(roll_cols)} rolling stat columns")
    
    return df


# =============================================================================
# 5. COMPUTE REST/STREAK FEATURES
# =============================================================================

def compute_situational_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rest days, B2B, streaks, etc."""
    print("\n[5] Computing situational features...")
    
    df = df.sort_values("game_date").copy()
    
    # Track last game date per team
    team_dates = {}
    rest_home, rest_away = [], []
    
    for _, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        date = row["game_date"]
        
        rest_h = (date - team_dates.get(home, date - timedelta(days=7))).days if home in team_dates else 7
        rest_a = (date - team_dates.get(away, date - timedelta(days=7))).days if away in team_dates else 7
        
        rest_home.append(rest_h)
        rest_away.append(rest_a)
        
        team_dates[home] = date
        team_dates[away] = date
    
    df["home_rest_days"] = rest_home
    df["away_rest_days"] = rest_away
    df["home_b2b"] = (df["home_rest_days"] == 1).astype(int)
    df["away_b2b"] = (df["away_rest_days"] == 1).astype(int)
    df["rest_advantage"] = df["home_rest_days"] - df["away_rest_days"]
    
    # Win streaks
    team_streaks = {}
    streaks_home, streaks_away = [], []
    
    for _, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        
        streaks_home.append(team_streaks.get(home, 0))
        streaks_away.append(team_streaks.get(away, 0))
        
        # Update streaks based on result
        home_won = row["home_score"] > row["away_score"]
        
        if home_won:
            team_streaks[home] = max(1, team_streaks.get(home, 0) + 1)
            team_streaks[away] = min(-1, team_streaks.get(away, 0) - 1)
        else:
            team_streaks[home] = min(-1, team_streaks.get(home, 0) - 1)
            team_streaks[away] = max(1, team_streaks.get(away, 0) + 1)
    
    df["home_streak"] = streaks_home
    df["away_streak"] = streaks_away
    df["streak_diff"] = df["home_streak"] - df["away_streak"]
    
    print(f"    Added rest, B2B, and streak features")
    
    return df


# =============================================================================
# 6. COMPUTE LABELS
# =============================================================================

def compute_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Compute betting outcome labels."""
    print("\n[6] Computing betting outcome labels...")
    
    # FG spread covered (home perspective)
    df["fg_spread_covered"] = np.where(
        df["fg_spread_line"].notna() & df["fg_margin"].notna(),
        (df["fg_margin"] + df["fg_spread_line"] > 0).astype(int),
        np.nan
    )
    
    # FG total over
    df["fg_total_over"] = np.where(
        df["fg_total_line"].notna() & df["fg_total_actual"].notna(),
        (df["fg_total_actual"] > df["fg_total_line"]).astype(int),
        np.nan
    )
    
    # FG home win
    df["fg_home_win"] = (df["fg_margin"] > 0).astype(int)
    
    # 1H spread covered
    df["1h_spread_covered"] = np.where(
        df["1h_spread_line"].notna() & df["1h_margin"].notna(),
        (df["1h_margin"] + df["1h_spread_line"] > 0).astype(int),
        np.nan
    )
    
    # 1H total over
    df["1h_total_over"] = np.where(
        df["1h_total_line"].notna() & df["1h_total_actual"].notna(),
        (df["1h_total_actual"] > df["1h_total_line"]).astype(int),
        np.nan
    )
    
    # 1H home win
    df["1h_home_win"] = (df["1h_margin"] > 0).astype(int)
    
    # Q1 outcomes (for future Q1 model)
    df["q1_total"] = df["home_q1"].fillna(0) + df["away_q1"].fillna(0)
    df["q1_margin"] = df["home_q1"].fillna(0) - df["away_q1"].fillna(0)
    df["q1_home_win"] = (df["q1_margin"] > 0).astype(int)
    
    print(f"    Added FG, 1H, Q1 outcome labels")
    
    return df


# =============================================================================
# 7. SUMMARY
# =============================================================================

def print_summary(df: pd.DataFrame):
    """Print comprehensive data summary."""
    print("\n" + "=" * 80)
    print(" DATA SUMMARY")
    print("=" * 80)
    
    print(f"\n  Total games: {len(df):,}")
    print(f"  Date range: {df['game_date'].min().date()} to {df['game_date'].max().date()}")
    print(f"  Total columns: {len(df.columns)}")
    
    # Data coverage
    print("\n  DATA COVERAGE:")
    coverage = {
        "FG spread line": df["fg_spread_line"].notna().mean() * 100,
        "FG total line": df["fg_total_line"].notna().mean() * 100,
        "1H spread line": df["1h_spread_line"].notna().mean() * 100 if "1h_spread_line" in df.columns else 0,
        "1H total line": df["1h_total_line"].notna().mean() * 100 if "1h_total_line" in df.columns else 0,
        "Box score stats": df["home_efg_pct"].notna().mean() * 100 if "home_efg_pct" in df.columns else 0,
    }
    for name, pct in coverage.items():
        status = "[OK]" if pct > 50 else "[!!]"
        print(f"    {status} {name}: {pct:.1f}%")
    
    # Feature categories
    print("\n  FEATURE CATEGORIES:")
    cats = {
        "Rolling stats": len([c for c in df.columns if "g_" in c]),
        "Advanced (Four Factors)": len([c for c in df.columns if any(x in c for x in ["efg", "tov_pct", "oreb_pct", "ft_rate"])]),
        "Ratings": len([c for c in df.columns if any(x in c for x in ["off_rtg", "def_rtg", "net_rtg"])]),
        "Situational": len([c for c in df.columns if any(x in c for x in ["rest", "b2b", "streak"])]),
        "Lines": len([c for c in df.columns if any(x in c for x in ["spread_line", "total_line", "ml_"])]),
        "Labels": len([c for c in df.columns if any(x in c for x in ["covered", "over", "win"])]),
    }
    for cat, count in cats.items():
        print(f"    {cat}: {count}")
    
    # Label distributions
    print("\n  LABEL DISTRIBUTIONS:")
    labels = ["fg_spread_covered", "fg_total_over", "fg_home_win", 
              "1h_spread_covered", "1h_total_over", "1h_home_win"]
    for label in labels:
        if label in df.columns:
            n = df[label].notna().sum()
            pct = df[label].mean() * 100 if n > 0 else 0
            print(f"    {label}: {n:,} games, {pct:.1f}% positive")


# =============================================================================
# MAIN
# =============================================================================

def main(start_date: str = "2023-01-01"):
    print("\n" + "=" * 80)
    print(" BUILDING MAXIMUM TRAINING DATA")
    print(" Using ALL available historical data")
    print("=" * 80)
    
    # Build dataset
    df = load_kaggle(start_date)
    df = merge_box_scores(df)
    df = merge_theodds(df)
    df = compute_rolling_stats(df)
    df = compute_situational_features(df)
    df = compute_labels(df)
    
    # Summary
    print_summary(df)
    
    # Save
    output_file = OUTPUT_DIR / f"training_data_maximum_{start_date[:4]}.csv"
    df.to_csv(output_file, index=False)
    print(f"\n  Saved to: {output_file}")
    
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2023-01-01")
    args = parser.parse_args()
    
    main(args.start_date)
