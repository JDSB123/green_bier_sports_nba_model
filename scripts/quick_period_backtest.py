#!/usr/bin/env python3
"""Quick period-based moneyline backtests for 1H and Q1."""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Load training data
df = pd.read_csv(DATA_DIR / "training_data.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# Create period-specific outcomes
df['home_1h_score'] = df['home_q1'].fillna(0) + df['home_q2'].fillna(0)
df['away_1h_score'] = df['away_q1'].fillna(0) + df['away_q2'].fillna(0)
df['home_1h_win'] = (df['home_1h_score'] > df['away_1h_score']).astype(int)
df['home_q1_win'] = (df['home_q1'].fillna(0) > df['away_q1'].fillna(0)).astype(int)

# Pre-compute rolling features for each period
teams = set(df['home_team'].unique()) | set(df['away_team'].unique())

# Team game history with period data
team_games = {team: [] for team in teams}
for idx, row in df.iterrows():
    home = row['home_team']
    away = row['away_team']
    date = row['date']

    team_games[home].append({
        'date': date,
        'fg_scored': row['home_score'],
        'fg_allowed': row['away_score'],
        'fg_won': 1 if row['home_score'] > row['away_score'] else 0,
        '1h_scored': row['home_1h_score'],
        '1h_allowed': row['away_1h_score'],
        '1h_won': 1 if row['home_1h_score'] > row['away_1h_score'] else 0,
        'q1_scored': row['home_q1'] if pd.notna(row['home_q1']) else 0,
        'q1_allowed': row['away_q1'] if pd.notna(row['away_q1']) else 0,
        'q1_won': 1 if (row['home_q1'] or 0) > (row['away_q1'] or 0) else 0,
    })
    team_games[away].append({
        'date': date,
        'fg_scored': row['away_score'],
        'fg_allowed': row['home_score'],
        'fg_won': 1 if row['away_score'] > row['home_score'] else 0,
        '1h_scored': row['away_1h_score'],
        '1h_allowed': row['home_1h_score'],
        '1h_won': 1 if row['away_1h_score'] > row['home_1h_score'] else 0,
        'q1_scored': row['away_q1'] if pd.notna(row['away_q1']) else 0,
        'q1_allowed': row['home_q1'] if pd.notna(row['home_q1']) else 0,
        'q1_won': 1 if (row['away_q1'] or 0) > (row['home_q1'] or 0) else 0,
    })


def get_team_period_stats(team, before_date, period='fg', lookback=10):
    """Get rolling stats for a team in a specific period."""
    games = [g for g in team_games[team] if g['date'] < before_date]
    if len(games) < 3:
        return None
    recent = games[-lookback:] if len(games) >= lookback else games

    ppg = np.mean([g[f'{period}_scored'] for g in recent])
    papg = np.mean([g[f'{period}_allowed'] for g in recent])
    margin = ppg - papg
    win_pct = np.mean([g[f'{period}_won'] for g in recent])

    return {'ppg': ppg, 'papg': papg, 'margin': margin, 'win_pct': win_pct}


def run_period_backtest(period, label_col):
    """Run walk-forward backtest for a period."""
    print(f"\n{'='*50}")
    print(f"BACKTEST: {period.upper()} Moneyline")
    print(f"{'='*50}")

    features_list = []
    for idx, row in df.iterrows():
        home_stats = get_team_period_stats(row['home_team'], row['date'], period)
        away_stats = get_team_period_stats(row['away_team'], row['date'], period)

        if home_stats is None or away_stats is None:
            features_list.append(None)
            continue

        features = {
            'game_idx': idx,
            'home_ppg': home_stats['ppg'],
            'home_papg': home_stats['papg'],
            'home_margin': home_stats['margin'],
            'home_win_pct': home_stats['win_pct'],
            'away_ppg': away_stats['ppg'],
            'away_papg': away_stats['papg'],
            'away_margin': away_stats['margin'],
            'away_win_pct': away_stats['win_pct'],
            'ppg_diff': home_stats['ppg'] - away_stats['ppg'],
            'margin_diff': home_stats['margin'] - away_stats['margin'],
            'win_pct_diff': home_stats['win_pct'] - away_stats['win_pct'],
            'label': row[label_col]
        }
        features_list.append(features)

    valid_indices = [i for i, f in enumerate(features_list) if f is not None]
    features_df = pd.DataFrame([features_list[i] for i in valid_indices])

    # Walk-forward
    min_train = 150
    results = []
    feature_cols = ['home_ppg', 'home_papg', 'home_margin', 'home_win_pct',
                    'away_ppg', 'away_papg', 'away_margin', 'away_win_pct',
                    'ppg_diff', 'margin_diff', 'win_pct_diff']

    for i in range(min_train, len(features_df)):
        train_df = features_df.iloc[:i]
        test_row = features_df.iloc[i]

        X_train = train_df[feature_cols].values
        y_train = train_df['label'].values
        X_test = test_row[feature_cols].values.reshape(1, -1)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train_scaled, y_train)

        proba = model.predict_proba(X_test_scaled)[0, 1]
        pred = 1 if proba >= 0.5 else 0
        actual = int(test_row['label'])
        profit = 100/110 if pred == actual else -1.0

        game_idx = int(test_row['game_idx'])
        results.append({
            'date': df.iloc[game_idx]['date'],
            'home_team': df.iloc[game_idx]['home_team'],
            'away_team': df.iloc[game_idx]['away_team'],
            'market': f'{period}_moneyline',
            'predicted': pred,
            'actual': actual,
            'confidence': proba if pred == 1 else 1 - proba,
            'profit': profit,
            'correct': 1 if pred == actual else 0
        })

    results_df = pd.DataFrame(results)
    acc = results_df["correct"].mean()
    roi = results_df["profit"].sum() / len(results_df)
    print(f"Bets: {len(results_df)} | Accuracy: {acc:.1%} | ROI: {roi:+.1%}")

    high_conf = results_df[results_df['confidence'] >= 0.6]
    if len(high_conf) > 0:
        hc_acc = high_conf["correct"].mean()
        hc_roi = high_conf["profit"].sum() / len(high_conf)
        print(f"High Conf: {len(high_conf)} bets | Accuracy: {hc_acc:.1%} | ROI: {hc_roi:+.1%}")

    return results_df


if __name__ == "__main__":
    # Run period backtests
    results_1h = run_period_backtest('1h', 'home_1h_win')
    results_q1 = run_period_backtest('q1', 'home_q1_win')

    # Load FG results
    fg_results = pd.read_csv(DATA_DIR / "quick_backtest_results.csv")

    # Combine all results
    all_results = pd.concat([fg_results, results_1h, results_q1], ignore_index=True)
    all_results.to_csv(DATA_DIR / "all_moneyline_backtest_results.csv", index=False)

    print(f"\n{'='*50}")
    print("OVERALL MONEYLINE BACKTEST SUMMARY")
    print(f"{'='*50}")
    for market in all_results['market'].unique():
        mkt_df = all_results[all_results['market'] == market]
        acc = mkt_df["correct"].mean()
        roi = mkt_df["profit"].sum() / len(mkt_df)
        print(f"{market}: {len(mkt_df)} bets | {acc:.1%} acc | {roi:+.1%} ROI")

    print(f"\nTotal: {len(all_results)} predictions saved to all_moneyline_backtest_results.csv")
