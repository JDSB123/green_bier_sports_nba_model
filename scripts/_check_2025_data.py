#!/usr/bin/env python3
"""Check if 2025-26 training data has production features."""
import pandas as pd

# Production model features
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

# Check 2025-26 data
print("Checking training_data_2025_26.csv...")
df = pd.read_csv('data/processed/training_data_2025_26.csv', low_memory=False, nrows=5)
print(f"Columns: {len(df.columns)}")

data_cols = set(df.columns)
prod_features = set(production_features)

in_both = prod_features & data_cols
missing = prod_features - data_cols

print(f"Production features found: {len(in_both)}/{len(prod_features)}")
print(f"Missing: {len(missing)}")

if missing:
    print("\nMissing features:")
    for f in sorted(missing)[:20]:
        print(f"  - {f}")
    if len(missing) > 20:
        print(f"  ... and {len(missing) - 20} more")
else:
    print("\nAll production features found!")

# Show sample of columns
print("\nSample columns in data:")
for col in sorted(df.columns)[:30]:
    print(f"  - {col}")
