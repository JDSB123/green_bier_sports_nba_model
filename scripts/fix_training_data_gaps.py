#!/usr/bin/env python3
"""
Fix training data gaps for complete backtesting coverage.

Addresses:
1. fg_total_actual - compute from home_score + away_score
2. fg_spread_covered, fg_total_over, fg_home_win - add canonical column names
3. rest_days - compute from game schedule
4. Verify all labels are balanced and correct
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_FILE = PROJECT_ROOT / "data" / "processed" / "training_data_complete_2023.csv"


def main():
    print("=" * 70)
    print("FIXING TRAINING DATA GAPS")
    print("=" * 70)
    
    # Load data
    print("\n[1/6] Loading training data...")
    df = pd.read_csv(DATA_FILE, low_memory=False)
    df["game_date"] = pd.to_datetime(df["game_date"], format="mixed")
    print(f"      Loaded {len(df)} games")
    
    # ==== FIX 1: fg_total_actual ====
    print("\n[2/6] Fixing fg_total_actual...")
    before = df["fg_total_actual"].notna().sum()
    df["fg_total_actual"] = df["home_score"] + df["away_score"]
    after = df["fg_total_actual"].notna().sum()
    print(f"      Before: {before}, After: {after}")
    
    # ==== FIX 2: FG margin ====
    print("\n[3/6] Fixing fg_margin...")
    before = df["fg_margin"].notna().sum() if "fg_margin" in df.columns else 0
    df["fg_margin"] = df["home_score"] - df["away_score"]
    after = df["fg_margin"].notna().sum()
    print(f"      Before: {before}, After: {after}")
    
    # ==== FIX 3: FG Labels (canonical names) ====
    print("\n[4/6] Adding canonical FG label columns...")
    
    # fg_spread_covered: home margin + spread_line > 0 means home covered
    # Note: spread_line is typically negative for favorites
    if "spread_covered" in df.columns:
        df["fg_spread_covered"] = df["spread_covered"]
        print(f"      fg_spread_covered: copied from spread_covered ({df['fg_spread_covered'].notna().sum()})")
    else:
        df["fg_spread_covered"] = np.where(
            df["fg_spread_line"].notna() & df["fg_margin"].notna(),
            (df["fg_margin"] + df["fg_spread_line"] > 0).astype(float),
            np.nan
        )
        print(f"      fg_spread_covered: computed ({df['fg_spread_covered'].notna().sum()})")
    
    # fg_total_over: actual > line
    if "total_over" in df.columns:
        df["fg_total_over"] = df["total_over"]
        print(f"      fg_total_over: copied from total_over ({df['fg_total_over'].notna().sum()})")
    else:
        df["fg_total_over"] = np.where(
            df["fg_total_line"].notna() & df["fg_total_actual"].notna(),
            (df["fg_total_actual"] > df["fg_total_line"]).astype(float),
            np.nan
        )
        print(f"      fg_total_over: computed ({df['fg_total_over'].notna().sum()})")
    
    # fg_home_win: margin > 0
    if "home_win" in df.columns:
        df["fg_home_win"] = df["home_win"]
        print(f"      fg_home_win: copied from home_win ({df['fg_home_win'].notna().sum()})")
    else:
        df["fg_home_win"] = np.where(
            df["fg_margin"].notna(),
            (df["fg_margin"] > 0).astype(float),
            np.nan
        )
        print(f"      fg_home_win: computed ({df['fg_home_win'].notna().sum()})")
    
    # ==== FIX 4: Rest days ====
    print("\n[5/6] Computing rest days...")
    df = df.sort_values(["game_date"]).reset_index(drop=True)
    
    # Track last game date for each team
    last_game = {}
    home_rest = []
    away_rest = []
    
    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        date = row["game_date"]
        
        # Home rest
        if home in last_game:
            rest_h = (date - last_game[home]).days
        else:
            rest_h = np.nan  # First game of dataset
        home_rest.append(rest_h)
        
        # Away rest  
        if away in last_game:
            rest_a = (date - last_game[away]).days
        else:
            rest_a = np.nan
        away_rest.append(rest_a)
        
        # Update last game
        last_game[home] = date
        last_game[away] = date
    
    df["home_rest_days"] = home_rest
    df["away_rest_days"] = away_rest
    
    # Cap extreme values (start of season, etc.)
    df["home_rest_days"] = df["home_rest_days"].clip(upper=10)
    df["away_rest_days"] = df["away_rest_days"].clip(upper=10)
    
    print(f"      home_rest_days: {df['home_rest_days'].notna().sum()}")
    print(f"      away_rest_days: {df['away_rest_days'].notna().sum()}")
    
    # ==== SAVE ====
    print("\n[6/6] Saving...")
    df.to_csv(DATA_FILE, index=False)
    print(f"      Saved to {DATA_FILE}")
    
    # ==== VERIFICATION ====
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    
    print("\nFG Labels:")
    for col in ["fg_spread_covered", "fg_total_over", "fg_home_win"]:
        ct = df[col].notna().sum()
        dist = df[col].value_counts(dropna=True).to_dict()
        print(f"  {col}: {ct}/{len(df)} dist={dist}")
    
    print("\n1H Labels:")
    for col in ["1h_spread_covered", "1h_total_over", "1h_home_win"]:
        if col in df.columns:
            ct = df[col].notna().sum()
            dist = df[col].value_counts(dropna=True).to_dict()
            print(f"  {col}: {ct}/{len(df)} dist={dist}")
    
    print("\nRest Days:")
    print(f"  home_rest_days: mean={df['home_rest_days'].mean():.1f}, median={df['home_rest_days'].median():.1f}")
    print(f"  away_rest_days: mean={df['away_rest_days'].mean():.1f}, median={df['away_rest_days'].median():.1f}")
    
    print("\n2025-26 Check:")
    df26 = df[df["game_date"] >= "2025-10-01"]
    for col in ["fg_spread_covered", "fg_total_over", "1h_spread_covered", "1h_total_over"]:
        if col in df.columns:
            ct = df26[col].notna().sum()
            print(f"  {col}: {ct}/{len(df26)}")


if __name__ == "__main__":
    main()
