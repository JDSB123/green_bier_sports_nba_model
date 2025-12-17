#!/usr/bin/env python3
"""Analyze backtest results and display performance metrics."""
import pandas as pd
from pathlib import Path

RESULTS_FILE = Path("data/processed/all_markets_backtest_results.csv")

if not RESULTS_FILE.exists():
    print(f"[ERROR] Backtest results not found: {RESULTS_FILE}")
    exit(1)

df = pd.read_csv(RESULTS_FILE)
print("=" * 70)
print("BACKTEST RESULTS SUMMARY")
print("=" * 70)
print(f"\nTotal Predictions: {len(df)}")
print(f"Date Range: {df['date'].min()} to {df['date'].max()}")

print("\n" + "=" * 70)
print("MARKET PERFORMANCE")
print("=" * 70)
print(f"{'Market':<20} | {'Bets':<6} | {'Accuracy':<10} | {'ROI':<10} | {'Total Profit':<12}")
print("-" * 70)

for market in sorted(df['market'].unique()):
    mdf = df[df['market'] == market]
    acc = mdf['correct'].mean()
    roi = mdf['profit'].sum() / len(mdf)
    total_profit = mdf['profit'].sum()
    print(f"{market:<20} | {len(mdf):<6} | {acc:>9.1%} | {roi:>9.1%} | ${total_profit:>10.2f}")

print("\n" + "=" * 70)
print("OVERALL PERFORMANCE")
print("=" * 70)
overall_acc = df['correct'].mean()
overall_roi = df['profit'].sum() / len(df)
total_profit = df['profit'].sum()
print(f"Overall Accuracy: {overall_acc:.1%}")
print(f"Overall ROI: {overall_roi:.1%}")
print(f"Total Profit: ${total_profit:.2f}")

# High confidence analysis
if 'confidence' in df.columns:
    high_conf = df[df['confidence'] >= 0.6]
    if len(high_conf) > 0:
        print("\n" + "=" * 70)
        print("HIGH CONFIDENCE (>= 60%) PERFORMANCE")
        print("=" * 70)
        hc_acc = high_conf['correct'].mean()
        hc_roi = high_conf['profit'].sum() / len(high_conf)
        print(f"Bets: {len(high_conf)}")
        print(f"Accuracy: {hc_acc:.1%}")
        print(f"ROI: {hc_roi:.1%}")

print("\n" + "=" * 70)

