"""
Generate 1H training data using ONLY historical game data (no API calls).

Much faster than the API-based version - builds features from rolling statistics.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import settings

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"


def load_first_half_data() -> pd.DataFrame:
    """Load 1H historical data."""
    fh_path = PROCESSED_DIR / "first_half_historical_data.csv"
    if not fh_path.exists():
        raise FileNotFoundError(
            f"First half data not found. Run: python scripts/collect_first_half_data.py"
        )

    df = pd.read_csv(fh_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df


def calculate_team_rolling_stats(games_df: pd.DataFrame, lookback: int = 10):
    """
    Calculate rolling statistics for each team.

    For each game, calculates team stats from previous N games.
    """
    stats = []

    for idx, game in games_df.iterrows():
        game_date = game['date']
        home_team = game['home_team']
        away_team = game['away_team']

        # Get previous games for each team
        prev_games = games_df[games_df['date'] < game_date]

        # Home team previous games
        home_prev = prev_games[
            (prev_games['home_team'] == home_team) | (prev_games['away_team'] == home_team)
        ].tail(lookback)

        # Away team previous games
        away_prev = prev_games[
            (prev_games['home_team'] == away_team) | (prev_games['away_team'] == away_team)
        ].tail(lookback)

        # Calculate home team stats
        home_stats = calculate_team_stats(home_prev, home_team, prefix='home')

        # Calculate away team stats
        away_stats = calculate_team_stats(away_prev, away_team, prefix='away')

        # Combine
        game_stats = {
            'game_id': game['game_id'],
            'date': game['date'],
            'home_team': home_team,
            'away_team': away_team,
            **home_stats,
            **away_stats,
        }

        # Add actual results as targets
        game_stats.update({
            '1h_home_score': game['home_1h'],
            '1h_away_score': game['away_1h'],
            '1h_spread': game['1h_spread'],
            '1h_total': game['1h_total'],
            'fg_home_score': game['home_final'],
            'fg_away_score': game['away_final'],
            'fg_spread': game['fg_spread'],
            'fg_total': game['fg_total'],
        })

        stats.append(game_stats)

        if idx % 50 == 0:
            print(f"  Progress: {idx}/{len(games_df)} games")

    return pd.DataFrame(stats)


def calculate_team_stats(team_games: pd.DataFrame, team_name: str, prefix: str) -> dict:
    """Calculate stats for a team from their recent games."""
    if len(team_games) == 0:
        # No historical data - return zeros
        return {
            f'{prefix}_ppg_1h': 0,
            f'{prefix}_papg_1h': 0,
            f'{prefix}_ppg_fg': 0,
            f'{prefix}_papg_fg': 0,
            f'{prefix}_win_pct': 0.5,
            f'{prefix}_spread_margin_1h': 0,
            f'{prefix}_spread_margin_fg': 0,
            f'{prefix}_games_played': 0,
        }

    # Separate home/away games for this team
    home_games = team_games[team_games['home_team'] == team_name]
    away_games = team_games[team_games['away_team'] == team_name]

    # Points scored/allowed in 1H
    home_1h_scored = home_games['home_1h'].values if len(home_games) > 0 else []
    home_1h_allowed = home_games['away_1h'].values if len(home_games) > 0 else []
    away_1h_scored = away_games['away_1h'].values if len(away_games) > 0 else []
    away_1h_allowed = away_games['home_1h'].values if len(away_games) > 0 else []

    pts_1h = np.concatenate([home_1h_scored, away_1h_scored]) if len(home_1h_scored) + len(away_1h_scored) > 0 else [0]
    allowed_1h = np.concatenate([home_1h_allowed, away_1h_allowed]) if len(home_1h_allowed) + len(away_1h_allowed) > 0 else [0]

    # Points scored/allowed in FG
    home_fg_scored = home_games['home_final'].values if len(home_games) > 0 else []
    home_fg_allowed = home_games['away_final'].values if len(home_games) > 0 else []
    away_fg_scored = away_games['away_final'].values if len(away_games) > 0 else []
    away_fg_allowed = away_games['home_final'].values if len(away_games) > 0 else []

    pts_fg = np.concatenate([home_fg_scored, away_fg_scored]) if len(home_fg_scored) + len(away_fg_scored) > 0 else [0]
    allowed_fg = np.concatenate([home_fg_allowed, away_fg_allowed]) if len(home_fg_allowed) + len(away_fg_allowed) > 0 else [0]

    # Win %
    home_wins = (home_games['home_final'] > home_games['away_final']).sum()
    away_wins = (away_games['away_final'] > away_games['home_final']).sum()
    total_games = len(team_games)
    win_pct = (home_wins + away_wins) / total_games if total_games > 0 else 0.5

    # Spread margins
    home_margins_1h = home_games['1h_spread'].values if len(home_games) > 0 else []
    away_margins_1h = -away_games['1h_spread'].values if len(away_games) > 0 else []
    margins_1h = np.concatenate([home_margins_1h, away_margins_1h]) if len(home_margins_1h) + len(away_margins_1h) > 0 else [0]

    home_margins_fg = home_games['fg_spread'].values if len(home_games) > 0 else []
    away_margins_fg = -away_games['fg_spread'].values if len(away_games) > 0 else []
    margins_fg = np.concatenate([home_margins_fg, away_margins_fg]) if len(home_margins_fg) + len(away_margins_fg) > 0 else [0]

    return {
        f'{prefix}_ppg_1h': float(np.mean(pts_1h)),
        f'{prefix}_papg_1h': float(np.mean(allowed_1h)),
        f'{prefix}_ppg_fg': float(np.mean(pts_fg)),
        f'{prefix}_papg_fg': float(np.mean(allowed_fg)),
        f'{prefix}_win_pct': float(win_pct),
        f'{prefix}_spread_margin_1h': float(np.mean(margins_1h)),
        f'{prefix}_spread_margin_fg': float(np.mean(margins_fg)),
        f'{prefix}_games_played': int(total_games),
    }


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features (differentials, predicted values, etc.)."""
    # Offensive/defensive differentials for 1H
    df['ppg_diff_1h'] = df['home_ppg_1h'] - df['away_ppg_1h']
    df['papg_diff_1h'] = df['away_papg_1h'] - df['home_papg_1h']  # Lower is better

    # Offensive/defensive differentials for FG
    df['ppg_diff_fg'] = df['home_ppg_fg'] - df['away_ppg_fg']
    df['papg_diff_fg'] = df['away_papg_fg'] - df['home_papg_fg']

    # Win % differential
    df['win_pct_diff'] = df['home_win_pct'] - df['away_win_pct']

    # Predicted margin (using FG stats as baseline)
    df['predicted_margin'] = (
        (df['home_ppg_fg'] - df['home_papg_fg']) -
        (df['away_ppg_fg'] - df['away_papg_fg'])
    )

    # Predicted total
    df['predicted_total'] = df['home_ppg_fg'] + df['away_ppg_fg']

    # Predicted 1H margin/total (scaled from FG)
    df['predicted_margin_1h'] = df['predicted_margin'] * 0.5
    df['predicted_total_1h'] = df['predicted_total'] * 0.5

    return df


def add_training_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add 1H prediction targets."""
    # Simulate 1H lines (in production, use real lines from The Odds API)
    np.random.seed(42)

    # 1H spread line ~ predicted margin * 0.5 + noise
    df['1h_spread_line'] = -df['predicted_margin_1h'] + np.random.normal(0, 1.0, len(df))
    df['1h_spread_line'] = (df['1h_spread_line'] * 2).round() / 2  # Round to 0.5

    # 1H total line ~ predicted total * 0.5 + noise
    df['1h_total_line'] = df['predicted_total_1h'] + np.random.normal(0, 3.0, len(df))
    df['1h_total_line'] = df['1h_total_line'].round()

    # Targets
    df['1h_spread_covered'] = (df['1h_spread'] > -df['1h_spread_line']).astype(int)
    df['1h_total_over'] = (df['1h_total'] > df['1h_total_line']).astype(int)

    return df


def main():
    """Main execution."""
    print("="*80)
    print("FAST 1H TRAINING DATA GENERATION (No API calls)")
    print("="*80)

    # Load 1H data
    fh_games = load_first_half_data()
    print(f"\nLoaded {len(fh_games)} 1H games")

    # Calculate rolling stats
    print("\nCalculating rolling statistics...")
    training_df = calculate_team_rolling_stats(fh_games, lookback=10)

    print(f"\n[OK] Generated stats for {len(training_df)} games")

    # Add derived features
    print("\nAdding derived features...")
    training_df = add_derived_features(training_df)

    # Add targets
    print("Adding training targets...")
    training_df = add_training_targets(training_df)

    # Save
    output_path = PROCESSED_DIR / "first_half_training_data.csv"
    training_df.to_csv(output_path, index=False)

    print(f"\n{'='*80}")
    print(f"[OK] Saved training data to:")
    print(f"     {output_path}")
    print(f"{'='*80}")

    # Summary
    print(f"\nTraining Data Summary:")
    print(f"  Total games: {len(training_df)}")
    print(f"  Date range: {training_df['date'].min()} to {training_df['date'].max()}")
    print(f"  Features: {len([c for c in training_df.columns if c not in ['game_id', 'date', 'home_team', 'away_team']])}")

    print(f"\n1H Spread:")
    print(f"  Home covered: {training_df['1h_spread_covered'].sum()} ({training_df['1h_spread_covered'].mean():.1%})")
    print(f"  Mean spread: {training_df['1h_spread'].mean():+.1f}")

    print(f"\n1H Total:")
    print(f"  Over: {training_df['1h_total_over'].sum()} ({training_df['1h_total_over'].mean():.1%})")
    print(f"  Mean total: {training_df['1h_total'].mean():.1f}")

    print(f"\nNext steps:")
    print(f"  1. python scripts/train_first_half_models.py")
    print(f"  2. python scripts/backtest_first_half.py")


if __name__ == "__main__":
    main()
