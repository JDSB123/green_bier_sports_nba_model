#!/usr/bin/env python3
"""
Merge quarter scores into training_data_complete_2023.csv

This merges the 2025-26 quarter scores we just fetched into the training data,
fixing the missing 1H scores for 2025-26 games.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.standardization import standardize_team_name, generate_match_key, CST

TRAINING_FILE = PROJECT_ROOT / "data" / "processed" / "training_data_complete_2023.csv"
QUARTER_SCORES = PROJECT_ROOT / "data" / "raw" / "nba_api" / "quarter_scores_2025_26.csv"


def main():
    print("="*60)
    print("MERGING 2025-26 QUARTER SCORES INTO TRAINING DATA")
    print("="*60)
    
    # Load training data
    print("\n[1/4] Loading training data...")
    df = pd.read_csv(TRAINING_FILE, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    print(f"      Total games: {len(df)}")
    
    # Load quarter scores
    print("\n[2/4] Loading quarter scores...")
    qs = pd.read_csv(QUARTER_SCORES)
    qs["game_date"] = pd.to_datetime(qs["game_date"])
    
    # Standardize team names in quarter scores
    qs["home_team"] = qs["home_team"].apply(standardize_team_name)
    qs["away_team"] = qs["away_team"].apply(standardize_team_name)
    
    # Generate match keys
    qs["match_key"] = qs.apply(
        lambda r: generate_match_key(r["game_date"], r["home_team"], r["away_team"], source_is_utc=True),
        axis=1
    )
    
    print(f"      Quarter scores: {len(qs)} games")
    
    # Check 2025-26 games in training data
    df_2526 = df[df["date"] >= "2025-10-01"]
    print(f"\n[3/4] 2025-26 games in training: {len(df_2526)}")
    print(f"      Current 1h_total_actual non-null: {df_2526['1h_total_actual'].notna().sum()}")
    
    # Generate match keys for training data
    if "match_key" not in df.columns:
        df["home_team_std"] = df["home_team"].apply(standardize_team_name)
        df["away_team_std"] = df["away_team"].apply(standardize_team_name)
        df["match_key"] = df.apply(
            lambda r: generate_match_key(r["date"], r["home_team_std"], r["away_team_std"], source_is_utc=False),
            axis=1
        )
    
    # Prepare quarter score data for merge
    qs_merge = qs[[
        "match_key",
        "home_q1", "home_q2", "home_q3", "home_q4", "home_score",
        "away_q1", "away_q2", "away_q3", "away_q4", "away_score",
    ]].copy()
    
    # Compute 1H values
    qs_merge["home_1h_new"] = qs_merge["home_q1"].fillna(0) + qs_merge["home_q2"].fillna(0)
    qs_merge["away_1h_new"] = qs_merge["away_q1"].fillna(0) + qs_merge["away_q2"].fillna(0)
    qs_merge["1h_total_actual_new"] = qs_merge["home_1h_new"] + qs_merge["away_1h_new"]
    qs_merge["1h_margin_new"] = qs_merge["home_1h_new"] - qs_merge["away_1h_new"]
    
    # Merge
    df = df.merge(
        qs_merge[["match_key", "home_q1", "home_q2", "home_q3", "home_q4",
                  "away_q1", "away_q2", "away_q3", "away_q4",
                  "home_1h_new", "away_1h_new", "1h_total_actual_new", "1h_margin_new"]],
        on="match_key",
        how="left",
        suffixes=("", "_qs")
    )
    
    # Fill missing values in original columns
    for col, new_col in [
        ("home_q1", "home_q1"),
        ("home_q2", "home_q2"),
        ("home_q3", "home_q3"),
        ("home_q4", "home_q4"),
        ("away_q1", "away_q1"),
        ("away_q2", "away_q2"),
        ("away_q3", "away_q3"),
        ("away_q4", "away_q4"),
        ("home_1h", "home_1h_new"),
        ("away_1h", "away_1h_new"),
        ("1h_total_actual", "1h_total_actual_new"),
        ("1h_margin", "1h_margin_new"),
    ]:
        if col in df.columns and new_col in df.columns:
            df[col] = df[col].fillna(df[new_col])
        elif new_col in df.columns:
            df[col] = df[new_col]
    
    # Also update quarter columns if they use different naming
    for orig, new in [("home_q1", "home_q1"), ("home_q2", "home_q2"),
                      ("away_q1", "away_q1"), ("away_q2", "away_q2")]:
        if f"{orig}_qs" in df.columns:
            df[orig] = df[orig].fillna(df[f"{orig}_qs"])
    
    # Recompute 1H labels now that we have actual values
    df["1h_total_over"] = np.where(
        df["1h_total_line"].notna() & df["1h_total_actual"].notna(),
        (df["1h_total_actual"] > df["1h_total_line"]).astype(int),
        np.nan
    )
    df["1h_spread_covered"] = np.where(
        df["1h_spread_line"].notna() & df["1h_margin"].notna(),
        (df["1h_margin"] + df["1h_spread_line"] > 0).astype(int),
        np.nan
    )
    df["1h_home_win"] = np.where(
        df["1h_margin"].notna(),
        (df["1h_margin"] > 0).astype(int),
        np.nan
    )
    
    # Drop temporary columns
    drop_cols = [c for c in df.columns if c.endswith("_new") or c.endswith("_qs")]
    df = df.drop(columns=drop_cols + ["home_team_std", "away_team_std"], errors="ignore")
    
    # Verify
    df_2526_after = df[df["date"] >= "2025-10-01"]
    print(f"\n[4/4] After merge:")
    print(f"      2025-26 1h_total_actual non-null: {df_2526_after['1h_total_actual'].notna().sum()}/{len(df_2526_after)}")
    print(f"      2025-26 1h_total_over non-null: {df_2526_after['1h_total_over'].notna().sum()}/{len(df_2526_after)}")
    print(f"      2025-26 1h_total_over distribution:")
    print(df_2526_after["1h_total_over"].value_counts(dropna=False))
    
    # Save
    df.to_csv(TRAINING_FILE, index=False)
    print(f"\n      Saved to {TRAINING_FILE}")
    
    # Summary
    print("\n" + "="*60)
    print("MERGE COMPLETE")
    print("="*60)
    print(f"  Total games: {len(df)}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  1H Total coverage: {df['1h_total_actual'].notna().sum()}/{len(df)} ({df['1h_total_actual'].notna().mean()*100:.1f}%)")


if __name__ == "__main__":
    main()
