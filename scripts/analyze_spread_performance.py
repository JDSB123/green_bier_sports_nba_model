"""Analyze spread prediction performance to identify improvement areas."""
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import settings

def analyze_spreads():
    # Load gradient boosting results (better performer)
    df = pd.read_csv(
        os.path.join(settings.data_processed_dir, "backtest_current_season_gradient_boosting.csv")
    )

    spread_df = df[df['spread_actual'].notna()].copy()

    print("=" * 70)
    print("SPREAD PREDICTION DEEP DIVE ANALYSIS")
    print("=" * 70)

    # Basic stats
    print(f"\n1. OVERALL PERFORMANCE")
    print(f"   Total predictions: {len(spread_df)}")
    win_rate = (spread_df["spread_pred"] == spread_df["spread_actual"]).mean()
    print(f"   Win rate: {win_rate:.1%}")
    print(f"   Break-even needed: 52.4% (at -110 odds)")
    print(f"   Current edge: {(win_rate - 0.524) * 100:+.1f} percentage points")

    # Prediction error analysis
    spread_df['prediction_error'] = spread_df['home_margin'] - (-spread_df['spread_line'])
    print(f"\n2. PREDICTION ERROR STATS")
    print(f"   Mean absolute error: {abs(spread_df['prediction_error']).mean():.2f} points")
    print(f"   Std dev: {spread_df['prediction_error'].std():.2f} points")
    print(f"   Median error: {spread_df['prediction_error'].median():.2f} points")

    # Performance by confidence
    print(f"\n3. PERFORMANCE BY CONFIDENCE LEVEL")
    for threshold in [0.55, 0.60, 0.65, 0.70]:
        high_conf = spread_df[spread_df['spread_prob'].apply(
            lambda x: x > threshold or x < (1-threshold)
        )]
        if len(high_conf) > 0:
            acc = (high_conf['spread_pred'] == high_conf['spread_actual']).mean()
            correct = (high_conf['spread_pred'] == high_conf['spread_actual']).sum()
            total = len(high_conf)
            roi = (correct * (100/110) - (total - correct)) / total
            print(f"   >{int(threshold*100)}% confidence: {acc:.1%} ({correct}/{total}) | ROI: {roi:+.1%}")

    # Analyze by predicted side
    print(f"\n4. PERFORMANCE BY PREDICTED SIDE")
    home_bets = spread_df[spread_df['spread_pred'] == 1]  # Predicting home covers
    away_bets = spread_df[spread_df['spread_pred'] == 0]  # Predicting away covers

    home_acc = (home_bets['spread_pred'] == home_bets['spread_actual']).mean()
    away_acc = (away_bets['spread_pred'] == away_bets['spread_actual']).mean()

    print(f"   Home covers: {home_acc:.1%} accuracy ({len(home_bets)} bets)")
    print(f"   Away covers: {away_acc:.1%} accuracy ({len(away_bets)} bets)")

    # Analyze by spread size
    print(f"\n5. PERFORMANCE BY SPREAD SIZE")
    spread_df['spread_abs'] = abs(spread_df['spread_line'])

    bins = [0, 3, 6, 9, 100]
    labels = ['Pick em (0-3)', 'Small (3-6)', 'Medium (6-9)', 'Large (9+)']
    spread_df['spread_category'] = pd.cut(spread_df['spread_abs'], bins=bins, labels=labels)

    for cat in labels:
        cat_data = spread_df[spread_df['spread_category'] == cat]
        if len(cat_data) > 0:
            acc = (cat_data['spread_pred'] == cat_data['spread_actual']).mean()
            print(f"   {cat}: {acc:.1%} accuracy ({len(cat_data)} bets)")

    # Model calibration analysis
    print(f"\n6. MODEL CALIBRATION (Are probabilities accurate?)")
    prob_bins = [0, 0.45, 0.50, 0.55, 0.60, 1.0]
    prob_labels = ['<45%', '45-50%', '50-55%', '55-60%', '>60%']
    spread_df['prob_category'] = pd.cut(spread_df['spread_prob'], bins=prob_bins, labels=prob_labels)

    for cat in prob_labels:
        cat_data = spread_df[spread_df['prob_category'] == cat]
        if len(cat_data) > 0:
            actual_rate = cat_data['spread_actual'].mean()
            avg_prob = cat_data['spread_prob'].mean()
            print(f"   {cat}: Predicted {avg_prob:.1%}, Actual {actual_rate:.1%} ({len(cat_data)} bets)")

    # Recent vs early season
    print(f"\n7. PERFORMANCE OVER TIME")
    spread_df['date'] = pd.to_datetime(spread_df['date'])
    spread_df = spread_df.sort_values('date')

    early = spread_df.iloc[:len(spread_df)//2]
    late = spread_df.iloc[len(spread_df)//2:]

    early_acc = (early['spread_pred'] == early['spread_actual']).mean()
    late_acc = (late['spread_pred'] == late['spread_actual']).mean()

    print(f"   First half of season: {early_acc:.1%} accuracy ({len(early)} bets)")
    print(f"   Second half of season: {late_acc:.1%} accuracy ({len(late)} bets)")

    # Worst predictions (for learning)
    print(f"\n8. BIGGEST MISSES (Top 10 prediction errors)")
    worst = spread_df.nlargest(10, 'prediction_error')[
        ['date', 'home_team', 'away_team', 'spread_line', 'home_margin', 'prediction_error', 'spread_prob']
    ]
    print(worst.to_string(index=False))

    print("\n" + "=" * 70)

if __name__ == "__main__":
    analyze_spreads()
