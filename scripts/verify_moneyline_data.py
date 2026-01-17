#!/usr/bin/env python3
"""
Verify that training data has all required columns for moneyline optimization.

This script checks the data structure and reports what's available.
"""
import os
import sys
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import settings


def verify_data():
    """Verify training data has required columns for moneyline optimization."""

    data_path = os.path.join(settings.data_processed_dir, "training_data.csv")

    if not os.path.exists(data_path):
        print(f"ERROR: Training data not found at {data_path}")
        return False

    print("="*70)
    print("Moneyline Data Verification")
    print("="*70)
    print(f"\nReading: {data_path}")

    df = pd.read_csv(data_path, low_memory=False, nrows=100)  # Sample first 100 rows

    print(f"Total columns: {len(df.columns)}")
    print(f"\nChecking required columns...\n")

    # Required columns for FG moneyline
    fg_required = {
        "predicted_margin": "FG predicted margin (from spread model)",
        "fg_ml_home": "FG home moneyline odds",
        "fg_ml_away": "FG away moneyline odds",
        "home_score": "Actual home score",
        "away_score": "Actual away score",
    }

    # Required columns for 1H moneyline
    h1_required = {
        "predicted_margin_1h": "1H predicted margin (from spread model)",
        "1h_ml_home": "1H home moneyline odds",
        "1h_ml_away": "1H away moneyline odds",
    }

    # 1H scores (flexible - need one of these combinations)
    h1_scores = [
        ["home_1h", "away_1h"],
        ["home_q1", "home_q2", "away_q1", "away_q2"],
    ]

    # Date column (flexible)
    date_cols = ["date", "game_date"]

    # Check FG columns
    print("FG MONEYLINE REQUIREMENTS:")
    print("-" * 50)
    fg_ready = True
    for col, desc in fg_required.items():
        if col in df.columns:
            non_null = df[col].notna().sum()
            print(f"✓ {col:25s} - {desc} ({non_null}/100 non-null)")
        else:
            print(f"✗ {col:25s} - {desc} (MISSING)")
            fg_ready = False

    # Check date column
    date_found = False
    for col in date_cols:
        if col in df.columns:
            print(f"✓ {col:25s} - Date column")
            date_found = True
            break
    if not date_found:
        print(f"✗ date/game_date          - MISSING")
        fg_ready = False

    # Check 1H columns
    print(f"\n1H MONEYLINE REQUIREMENTS:")
    print("-" * 50)
    h1_ready = True
    for col, desc in h1_required.items():
        if col in df.columns:
            non_null = df[col].notna().sum()
            print(f"✓ {col:25s} - {desc} ({non_null}/100 non-null)")
        else:
            print(f"✗ {col:25s} - {desc} (MISSING)")
            h1_ready = False

    # Check 1H scores
    h1_score_found = False
    for score_combo in h1_scores:
        if all(col in df.columns for col in score_combo):
            print(f"✓ {', '.join(score_combo):25s} - 1H scores available")
            h1_score_found = True
            break

    if not h1_score_found:
        print(f"✗ 1H scores               - MISSING (need home_1h/away_1h OR quarters)")
        h1_ready = False

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("="*70)

    if fg_ready:
        print("✓ FG Moneyline: READY - All required columns present")
    else:
        print("✗ FG Moneyline: NOT READY - Missing required columns")

    if h1_ready:
        print("✓ 1H Moneyline: READY - All required columns present")
    else:
        print("✗ 1H Moneyline: NOT READY - Missing required columns")

    print(f"\n{'='*70}")

    # Show sample of moneyline odds
    if "fg_ml_home" in df.columns and "fg_ml_away" in df.columns:
        print("\nSample FG Moneyline Odds:")
        print("-" * 50)
        sample = df[df["fg_ml_home"].notna()].head(5)
        if not sample.empty:
            for _, row in sample.iterrows():
                print(f"  Home: {row['fg_ml_home']:+6.0f}  |  Away: {row['fg_ml_away']:+6.0f}")
        else:
            print("  No valid FG moneyline data in sample")

    if "1h_ml_home" in df.columns and "1h_ml_away" in df.columns:
        print("\nSample 1H Moneyline Odds:")
        print("-" * 50)
        sample = df[df["1h_ml_home"].notna()].head(5)
        if not sample.empty:
            for _, row in sample.iterrows():
                print(f"  Home: {row['1h_ml_home']:+6.0f}  |  Away: {row['1h_ml_away']:+6.0f}")
        else:
            print("  No valid 1H moneyline data in sample")

    print(f"\n{'='*70}")

    if fg_ready or h1_ready:
        print("\n✓ You can proceed with moneyline optimization!")
        print("\nRun:")
        if fg_ready and h1_ready:
            print("  python scripts/train_moneyline_models.py --market all")
        elif fg_ready:
            print("  python scripts/train_moneyline_models.py --market fg")
        elif h1_ready:
            print("  python scripts/train_moneyline_models.py --market 1h")
    else:
        print("\n✗ Cannot proceed - missing required data columns")
        print("\nEnsure your training_data.csv includes:")
        print("  1. Predicted margins from trained spread models")
        print("  2. Historical moneyline odds (fg_ml_home, fg_ml_away, etc.)")
        print("  3. Actual game scores for outcomes")

    print()

    return fg_ready or h1_ready


if __name__ == "__main__":
    success = verify_data()
    sys.exit(0 if success else 1)
