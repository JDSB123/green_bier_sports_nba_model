"""
Collect historical first half data for model training.

Extracts Q1 + Q2 scores from historical games to create 1H training targets.
"""
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import settings

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "api_basketball"
OUTPUT_DIR = DATA_DIR / "processed"


def load_historical_games(season: str = None) -> pd.DataFrame:
    """
    Load historical games with quarter scores from API-Basketball data.

    Returns DataFrame with:
    - game_id
    - date
    - home_team, away_team
    - home_q1, home_q2, away_q1, away_q2
    - home_1h, away_1h (calculated)
    - home_final, away_final
    """
    season = season or settings.current_season

    # Find largest games file (most complete data)
    games_files = list(RAW_DIR.glob("games_*.json"))
    if not games_files:
        raise FileNotFoundError(f"No games data found in {RAW_DIR}")

    # Use largest file
    games_file = max(games_files, key=lambda f: f.stat().st_size)
    print(f"Loading games from: {games_file.name} ({games_file.stat().st_size / 1024 / 1024:.1f} MB)")

    with open(games_file, 'r') as f:
        data = json.load(f)

    games = []
    for game in data.get("response", []):
        # Only include finished games with scores
        if game['status']['short'] != 'FT':
            continue

        scores = game.get('scores', {})
        home_scores = scores.get('home', {})
        away_scores = scores.get('away', {})

        # Skip if missing quarter scores (but allow 0 as valid score)
        if home_scores.get('quarter_1') is None or home_scores.get('quarter_2') is None:
            continue
        if away_scores.get('quarter_1') is None or away_scores.get('quarter_2') is None:
            continue

        # Calculate 1H scores
        home_1h = home_scores['quarter_1'] + home_scores['quarter_2']
        away_1h = away_scores['quarter_1'] + away_scores['quarter_2']

        games.append({
            'game_id': game['id'],
            'date': game['date'],
            'home_team': game['teams']['home']['name'],
            'away_team': game['teams']['away']['name'],
            # Quarter scores
            'home_q1': home_scores['quarter_1'],
            'home_q2': home_scores['quarter_2'],
            'away_q1': away_scores['quarter_1'],
            'away_q2': away_scores['quarter_2'],
            # First half totals
            'home_1h': home_1h,
            'away_1h': away_1h,
            # Final scores
            'home_final': home_scores['total'],
            'away_final': away_scores['total'],
            # 1H targets
            '1h_spread': home_1h - away_1h,
            '1h_total': home_1h + away_1h,
            # FG targets (for comparison)
            'fg_spread': home_scores['total'] - away_scores['total'],
            'fg_total': home_scores['total'] + away_scores['total'],
        })

    df = pd.DataFrame(games)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    return df


def analyze_first_half_distribution(df: pd.DataFrame):
    """Analyze 1H vs FG scoring patterns."""
    print("\n" + "="*80)
    print("FIRST HALF SCORING ANALYSIS")
    print("="*80)

    # Calculate 1H percentage of FG
    df['1h_pct_of_fg'] = df['1h_total'] / df['fg_total']

    print(f"\nTotal games with 1H data: {len(df)}")
    print(f"\n1H Total as % of FG Total:")
    print(f"  Mean: {df['1h_pct_of_fg'].mean():.1%}")
    print(f"  Median: {df['1h_pct_of_fg'].median():.1%}")
    print(f"  Std: {df['1h_pct_of_fg'].std():.1%}")
    print(f"  Min: {df['1h_pct_of_fg'].min():.1%}")
    print(f"  Max: {df['1h_pct_of_fg'].max():.1%}")

    # Spread analysis
    print(f"\n1H Spread vs FG Spread:")
    print(f"  1H Spread Mean: {df['1h_spread'].mean():+.1f}")
    print(f"  FG Spread Mean: {df['fg_spread'].mean():+.1f}")
    print(f"  1H Spread Std: {df['1h_spread'].std():.1f}")
    print(f"  FG Spread Std: {df['fg_spread'].std():.1f}")

    # Total analysis
    print(f"\n1H Total vs FG Total:")
    print(f"  1H Total Mean: {df['1h_total'].mean():.1f}")
    print(f"  FG Total Mean: {df['fg_total'].mean():.1f}")
    print(f"  1H Total Std: {df['1h_total'].std():.1f}")
    print(f"  FG Total Std: {df['fg_total'].std():.1f}")

    # Check if 50% scaling is reasonable
    actual_1h_avg = df['1h_total'].mean()
    predicted_1h_from_50pct = df['fg_total'].mean() * 0.5
    error = abs(actual_1h_avg - predicted_1h_from_50pct)

    print(f"\n50% Scaling Validation:")
    print(f"  Actual 1H Average: {actual_1h_avg:.1f}")
    print(f"  Predicted (FG * 0.5): {predicted_1h_from_50pct:.1f}")
    print(f"  Error: {error:.1f} pts ({error/actual_1h_avg:.1%})")

    if error / actual_1h_avg > 0.05:  # >5% error
        print(f"  [WARNING] 50% scaling has {error/actual_1h_avg:.1%} error!")
    else:
        print(f"  [OK] 50% scaling is reasonable (within 5% error)")


def save_first_half_data(df: pd.DataFrame):
    """Save 1H data for training."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "first_half_historical_data.csv"
    df.to_csv(output_path, index=False)

    print(f"\n{'='*80}")
    print(f"[OK] Saved {len(df)} games with 1H data to:")
    print(f"     {output_path}")
    print(f"{'='*80}")

    return output_path


def main():
    """Main execution."""
    print("="*80)
    print("FIRST HALF DATA COLLECTION")
    print("="*80)

    # Load historical games
    df = load_historical_games()

    # Analyze 1H vs FG patterns
    analyze_first_half_distribution(df)

    # Save for training
    save_first_half_data(df)

    print(f"\nNext steps:")
    print(f"  1. Run: python scripts/generate_training_data.py --market first_half_spread")
    print(f"  2. Run: python scripts/generate_training_data.py --market first_half_total")
    print(f"  3. Run: python scripts/train_models.py --market first_half_spread")
    print(f"  4. Run: python scripts/train_models.py --market first_half_total")


if __name__ == "__main__":
    main()
