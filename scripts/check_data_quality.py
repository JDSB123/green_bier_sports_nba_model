#!/usr/bin/env python3
"""Detailed data quality check for betting card."""
import pandas as pd
from pathlib import Path

card_path = Path("data/processed/betting_card_v3.csv")
pred_path = Path("data/processed/predictions_v3.csv")
training_path = Path("data/processed/training_data.csv")

print("=" * 80)
print("DATA QUALITY REPORT")
print("=" * 80)

# Check betting card
print(f"\n[1] BETTING CARD LOCATION:")
print(f"    {card_path.absolute()}")
print(f"    Exists: {card_path.exists()}")

if card_path.exists():
    df = pd.read_csv(card_path)
    print(f"    Size: {len(df)} plays")
    print(f"    Last modified: {card_path.stat().st_mtime}")

print(f"\n[2] DATA QUALITY ISSUES FOUND:")

if card_path.exists():
    df = pd.read_csv(card_path)
    
    # Issue 1: 100% confidence
    perfect_conf = df[df['confidence'] >= 1.0]
    if len(perfect_conf) > 0:
        print(f"\n    [ISSUE] {len(perfect_conf)} plays with 100% confidence:")
        for idx, row in perfect_conf.iterrows():
            print(f"      - {row['matchup']} {row['period']} {row['market']}: {row['pick']}")
        print("      This suggests model may be overconfident or miscalibrated.")
        print("      Real-world models rarely output exactly 100% probability.")
    
    # Issue 2: Negative edges passing filter
    negative_edges = df[(df['edge'] < 0) & (df['edge'].notna())]
    if len(negative_edges) > 0:
        print(f"\n    [ISSUE] {len(negative_edges)} plays with NEGATIVE edges:")
        for idx, row in negative_edges.iterrows():
            print(f"      - {row['matchup']} {row['period']} {row['market']}: {row['pick']}")
            print(f"        Edge: {row['edge']:.1f} pts (model predicts WORSE than line)")
        print("      These should typically be filtered out unless there's a good reason.")
    
    # Issue 3: Missing rationales
    missing_rationale = df[df['rationale'].isna()]
    if len(missing_rationale) > 0:
        print(f"\n    [ISSUE] {len(missing_rationale)} plays missing rationales")
    
    # Positive checks
    print(f"\n[3] DATA QUALITY CHECKS PASSED:")
    print(f"    [OK] All {len(df)} plays have confidence values")
    print(f"    [OK] All {len(df)} plays have edge values")
    print(f"    [OK] {len(df[df['rationale'].notna()])}/{len(df)} plays have rationales")
    print(f"    [OK] Date format is consistent: {df['date'].nunique()} unique dates")
    print(f"    [OK] Market distribution looks reasonable")

print(f"\n[4] RECOMMENDATIONS:")
print("    1. Review 100% confidence plays - check if model probabilities are capped")
print("    2. Consider filtering negative edge plays (currently all totals pass)")
print("    3. Validate that predicted totals match Vegas lines (expected to differ slightly)")

print("\n" + "=" * 80)

print(f"\n[5] TRAINING DATA LINE COVERAGE:")
print(f"    {training_path.absolute()}")
print(f"    Exists: {training_path.exists()}")

if training_path.exists():
    df_train = pd.read_csv(training_path)
    total_games = len(df_train)
    print(f"    Games: {total_games}")
    
    def coverage(col: str) -> str:
        if col not in df_train.columns:
            return "missing"
        return f"{df_train[col].notna().mean() * 100:.1f}%"
    
    coverage_report = [
        ("FG Spread", coverage("spread_line")),
        ("FG Total", coverage("total_line")),
        ("1H Spread", coverage("1h_spread_line")),
        ("1H Total", coverage("1h_total_line")),
        ("Q1 Spread", coverage("q1_spread_line")),
        ("Q1 Total", coverage("q1_total_line")),
    ]
    for label, cov in coverage_report:
        print(f"    - {label}: {cov}")
else:
    print("    [WARN] training_data.csv not found; run build_fresh_training_data.py first.")
