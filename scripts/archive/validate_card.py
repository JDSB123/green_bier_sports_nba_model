#!/usr/bin/env python3
"""Quick validation script for betting card data quality."""
import pandas as pd
from pathlib import Path

card_path = Path("data/processed/betting_card_v3.csv")

if not card_path.exists():
    print(f"❌ Betting card not found at {card_path}")
    exit(1)

df = pd.read_csv(card_path)

print("=" * 80)
print("BETTING CARD VALIDATION")
print("=" * 80)
print(f"\nLocation: {card_path.absolute()}")
print(f"Total Plays: {len(df)}")
print(f"Date: {df['date'].iloc[0] if len(df) > 0 else 'N/A'}")

print("\n" + "=" * 80)
print("DATA QUALITY CHECKS")
print("=" * 80)

# Check confidence values
print(f"\n[CONFIDENCE] Range:")
print(f"   Min: {df['confidence'].min():.1%}")
print(f"   Max: {df['confidence'].max():.1%}")
print(f"   Mean: {df['confidence'].mean():.1%}")

high_conf_count = len(df[df['confidence'] >= 0.9])
print(f"\n[WARN] High Confidence Plays (>=90%): {high_conf_count}")
if high_conf_count > 0:
    print("   These may be overconfident - review model calibration")

# Check edge values
print(f"\n[EDGE] Statistics:")
print(f"   Positive edges: {len(df[df['edge'] > 0])}")
print(f"   Negative edges: {len(df[df['edge'] < 0])}")
print(f"   Null edges: {df['edge'].isna().sum()}")

# Breakdown by market
print(f"\n[MARKETS] Breakdown:")
breakdown = df.groupby(['period', 'market']).size()
for (period, market), count in breakdown.items():
    print(f"   {period} {market}: {count} plays")

# Show rationale quality
print(f"\n[RATIONALE] Quality:")
rationale_count = df['rationale'].notna().sum()
print(f"   Plays with rationale: {rationale_count}/{len(df)}")
if rationale_count < len(df):
    print("   [WARN] Some plays missing rationales")

# Show sample plays
print("\n" + "=" * 80)
print("SAMPLE PLAYS (Top 3 by Confidence)")
print("=" * 80)
for idx, row in df.nlargest(3, 'confidence').iterrows():
    print(f"\n{row['matchup']} ({row['period']} {row['market']})")
    print(f"  Pick: {row['pick']}")
    print(f"  Confidence: {row['confidence']:.1%}")
    print(f"  Edge: {row['edge']}")
    if pd.notna(row['rationale']):
        print(f"  Rationale: {row['rationale'][:100]}...")

print("\n" + "=" * 80)
print("VALIDATION COMPLETE")
print("=" * 80)
