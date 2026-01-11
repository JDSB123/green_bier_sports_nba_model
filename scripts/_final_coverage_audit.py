#!/usr/bin/env python3
"""Final coverage audit of training data with new injury features."""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"

# Load the updated training data
df = pd.read_csv(DATA_DIR / 'training_data_complete_2023_with_injuries.csv', low_memory=False)
print('=' * 60)
print('FINAL TRAINING DATA COVERAGE AUDIT')
print('=' * 60)
print(f'Total games: {len(df):,}')

# Extract date from game_id
min_date = df['game_id'].min()[:10] if pd.notna(df['game_id'].min()) else 'N/A'
max_date = df['game_id'].max()[:10] if pd.notna(df['game_id'].max()) else 'N/A'
print(f'Date range: {min_date} to {max_date}')
print()

# Check injury features
print('=== INJURY FEATURE COVERAGE ===')
home_injury_coverage = (df['home_injury_impact'] > 0).sum()
away_injury_coverage = (df['away_injury_impact'] > 0).sum()
any_injury = ((df['home_injury_impact'] > 0) | (df['away_injury_impact'] > 0)).sum()
print(f'Home injury impact > 0: {home_injury_coverage:,} ({100*home_injury_coverage/len(df):.1f}%)')
print(f'Away injury impact > 0: {away_injury_coverage:,} ({100*away_injury_coverage/len(df):.1f}%)')
print(f'Any injury data: {any_injury:,} ({100*any_injury/len(df):.1f}%)')
print(f'Avg home impact: {df["home_injury_impact"].mean():.2f}')
print(f'Avg away impact: {df["away_injury_impact"].mean():.2f}')
print()

# Check all model features
print('=== ALL MODEL FEATURES COVERAGE ===')
MODEL_FEATURES = [
    'home_pts_rolling', 'away_pts_rolling', 'home_opp_pts_rolling', 'away_opp_pts_rolling',
    'home_spread_covered_pct', 'away_spread_covered_pct', 'home_ou_over_pct', 'away_ou_over_pct',
    'home_elo', 'away_elo', 'elo_diff', 'home_rest_days', 'away_rest_days',
    'home_streak', 'away_streak', 'home_injury_impact', 'away_injury_impact',
    'home_fg_pct_rolling', 'away_fg_pct_rolling', 'home_fg3_pct_rolling', 'away_fg3_pct_rolling',
    'home_ft_pct_rolling', 'away_ft_pct_rolling', 'home_reb_rolling', 'away_reb_rolling',
    'home_ast_rolling', 'away_ast_rolling', 'home_tov_rolling', 'away_tov_rolling',
    'home_pace_rolling', 'away_pace_rolling', 'home_off_rating_rolling', 'away_off_rating_rolling',
    'home_def_rating_rolling', 'away_def_rating_rolling', 'home_net_rating_rolling', 'away_net_rating_rolling',
    'home_true_shooting_rolling', 'away_true_shooting_rolling', 'home_efg_pct_rolling', 'away_efg_pct_rolling',
    'home_oreb_pct_rolling', 'away_oreb_pct_rolling', 'home_dreb_pct_rolling', 'away_dreb_pct_rolling',
    'home_ast_ratio_rolling', 'away_ast_ratio_rolling', 'home_tov_ratio_rolling', 'away_tov_ratio_rolling',
    'home_pts_paint_rolling', 'away_pts_paint_rolling', 'home_pts_fb_rolling', 'away_pts_fb_rolling',
    'home_pts_2nd_rolling', 'away_pts_2nd_rolling',
]

present = 0
low_coverage = []
missing = []

for f in MODEL_FEATURES:
    if f in df.columns:
        non_null = df[f].notna().sum()
        coverage = 100 * non_null / len(df)
        if coverage >= 99:
            present += 1
        else:
            low_coverage.append((f, coverage))
    else:
        missing.append(f)

print(f'Features with >99% coverage: {present}/{len(MODEL_FEATURES)}')

if low_coverage:
    print(f'\nLow coverage features:')
    for f, cov in low_coverage:
        print(f'  {f}: {cov:.1f}%')

if missing:
    print(f'\nMissing features: {missing}')

# Year breakdown
print('\n=== COVERAGE BY YEAR ===')
df['year'] = df['game_id'].apply(lambda x: int(str(x)[:4]) if pd.notna(x) else 0)
for year in sorted(df['year'].unique()):
    if year == 0:
        continue
    year_df = df[df['year'] == year]
    year_injury = ((year_df['home_injury_impact'] > 0) | (year_df['away_injury_impact'] > 0)).sum()
    print(f'  {year}: {len(year_df)} games, {year_injury} with injury data ({100*year_injury/len(year_df):.1f}%)')

# Summary
print('\n' + '=' * 60)
if present >= 50 and any_injury >= 3500:
    print('✓ SUCCESS! Training data is ready for model training.')
    print(f'  - {present}/55+ model features at >99% coverage')
    print(f'  - {any_injury}/{len(df)} games with injury data ({100*any_injury/len(df):.1f}%)')
else:
    print('⚠️ Some features need attention.')
print('=' * 60)
