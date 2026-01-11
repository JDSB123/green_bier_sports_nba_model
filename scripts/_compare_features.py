#!/usr/bin/env python3
"""Compare production model features vs training data columns."""
import pandas as pd
import json

# Load training data
df = pd.read_csv('data/processed/training_data_complete_2023.csv', low_memory=False, nrows=5)

# Production model features (from feature_importance.json)
production_features = [
    "home_ppg", "away_ppg", "home_papg", "away_papg", "home_ortg", "away_ortg",
    "home_drtg", "away_drtg", "home_net_rtg", "away_net_rtg", "home_pace_factor",
    "away_pace_factor", "expected_pace_factor", "home_expected_pts", "away_expected_pts",
    "predicted_margin", "predicted_total", "home_avg_margin", "away_avg_margin",
    "home_total_ppg", "away_total_ppg", "ppg_diff", "home_win_pct", "away_win_pct",
    "win_pct_diff", "home_position", "away_position", "position_diff", "h2h_win_rate",
    "home_l5_win_pct", "away_l5_win_pct", "home_l5_margin", "away_l5_margin",
    "home_l10_margin", "away_l10_margin", "home_days_rest", "away_days_rest",
    "home_rest_adj", "away_rest_adj", "rest_margin_adj", "home_form_adj", "away_form_adj",
    "form_margin_adj", "home_elo", "away_elo", "home_injury_impact_ppg", "away_injury_impact_ppg",
    "injury_margin_adj", "elo_diff", "elo_prob_home", "away_travel_distance",
    "away_timezone_change", "away_travel_fatigue", "is_away_long_trip", "is_away_cross_country",
    "away_b2b_travel_penalty", "travel_advantage", "home_court_advantage"
]

data_cols = set(df.columns)
prod_features = set(production_features)

in_both = prod_features & data_cols
in_prod_not_data = prod_features - data_cols
in_data_not_prod = data_cols - prod_features

print("=" * 60)
print("FEATURE COMPARISON: Production Model vs Training Data")
print("=" * 60)

print(f"\nProduction model features: {len(prod_features)}")
print(f"Training data columns: {len(data_cols)}")

print(f"\n[OK] Features in BOTH ({len(in_both)}):")
for f in sorted(in_both):
    print(f"   {f}")

print(f"\n[MISSING] Production features MISSING from training data ({len(in_prod_not_data)}):")
for f in sorted(in_prod_not_data):
    print(f"   {f}")

print(f"\n[EXTRA] Training data columns NOT in production model ({len(in_data_not_prod)}):")
# Just show a sample
sample = sorted(in_data_not_prod)[:30]
for f in sample:
    print(f"   {f}")
if len(in_data_not_prod) > 30:
    print(f"   ... and {len(in_data_not_prod) - 30} more")
