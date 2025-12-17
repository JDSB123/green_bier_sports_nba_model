"""
Generate training data for NBA spreads/totals models.

Uses FiveThirtyEight's public NBA ELO dataset which includes historical
game scores - perfect for training labels.

Usage: python scripts/generate_training_data.py
"""
from __future__ import annotations
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime

from src.config import settings
from src.modeling.dataset import DatasetBuilder


# FiveThirtyEight ELO datasets (public, no API key needed)
FIVETHIRTYEIGHT_URLS = [
    "https://raw.githubusercontent.com/fivethirtyeight/data/master/nba-elo/nbaallelo.csv",
    "https://raw.githubusercontent.com/fivethirtyeight/data/master/nba-forecasts/nba_elo_latest.csv",
]


def download_elo_data() -> pd.DataFrame:
    """Download FiveThirtyEight's NBA ELO dataset."""
    for url in FIVETHIRTYEIGHT_URLS:
        print(f"Trying: {url}")
        try:
            df = pd.read_csv(url)
            print(f"Downloaded {len(df)} historical games")
            return df
        except Exception as e:
            print(f"  Error: {e}")
            continue

    print("All sources failed. Creating synthetic dataset...")
    return create_synthetic_training_data()


def create_synthetic_training_data() -> pd.DataFrame:
    """Create synthetic training data when APIs are unavailable."""
    np.random.seed(42)

    # NBA teams
    teams = [
        "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets",
        "Chicago Bulls", "Cleveland Cavaliers", "Dallas Mavericks",
        "Denver Nuggets", "Detroit Pistons", "Golden State Warriors",
        "Houston Rockets", "Indiana Pacers", "Los Angeles Clippers",
        "Los Angeles Lakers", "Memphis Grizzlies", "Miami Heat",
        "Milwaukee Bucks", "Minnesota Timberwolves", "New Orleans Pelicans",
        "New York Knicks", "Oklahoma City Thunder", "Orlando Magic",
        "Philadelphia 76ers", "Phoenix Suns", "Portland Trail Blazers",
        "Sacramento Kings", "San Antonio Spurs", "Toronto Raptors",
        "Utah Jazz", "Washington Wizards"
    ]

    # Team strengths (ELO-like)
    team_elos = {team: 1500 + np.random.randn() * 100 for team in teams}

    # Generate ~2000 games
    n_games = 2000
    rows = []

    base_date = pd.Timestamp("2023-10-01")
    for i in range(n_games):
        home, away = np.random.choice(teams, 2, replace=False)
        date = base_date + pd.Timedelta(days=i // 10)

        home_elo = team_elos[home]
        away_elo = team_elos[away]

        # Expected margin based on ELO (home team gets ~3 pt advantage)
        expected_margin = (home_elo - away_elo) / 28 + 3

        # Actual scores with variance
        home_base = 110 + (home_elo - 1500) / 20
        away_base = 110 + (away_elo - 1500) / 20

        home_score = int(home_base + np.random.randn() * 12)
        away_score = int(away_base + np.random.randn() * 12)

        rows.append({
            "gameday": date.strftime("%Y-%m-%d"),
            "year_id": 2024,
            "fran_id": home,
            "opp_fran": away,
            "pts": home_score,
            "opp_pts": away_score,
            "elo_i": home_elo,
            "opp_elo_i": away_elo,
            "game_location": "H",
        })

    return pd.DataFrame(rows)


def process_elo_to_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert FiveThirtyEight ELO data to training format.

    Handles multiple formats:
    - nbaallelo.csv: gameday, year_id, fran_id, opp_fran, pts, opp_pts, elo_i
    - nba_elo_latest.csv: date, season, team1, team2, score1, score2, elo1_pre
    - synthetic: same as nbaallelo format
    """
    if df.empty:
        return pd.DataFrame()

    # Detect format and normalize column names
    if "date_game" in df.columns:
        # nbaallelo.csv format (actual column names)
        # Only home games to avoid duplicates
        df = df[df["game_location"] == "H"].copy()
        df = df.rename(columns={
            "date_game": "date",
            "year_id": "season",
            "fran_id": "team1",
            "opp_fran": "team2",
            "pts": "score1",
            "opp_pts": "score2",
            "elo_i": "elo1_pre",
            "opp_elo_i": "elo2_pre",
        })
    elif "gameday" in df.columns:
        # synthetic data format
        df = df[df["game_location"] == "H"].copy()
        df = df.rename(columns={
            "gameday": "date",
            "year_id": "season",
            "fran_id": "team1",
            "opp_fran": "team2",
            "pts": "score1",
            "opp_pts": "score2",
            "elo_i": "elo1_pre",
            "opp_elo_i": "elo2_pre",
        })

    # Filter to recent seasons with complete data
    if "season" in df.columns:
        # Use last 5 seasons available in dataset
        max_season = df["season"].max()
        df = df[df["season"] >= max_season - 4].copy()
        print(f"  Using seasons {max_season - 4} to {max_season}")

    # Filter to games that have been played (have scores)
    if "score1" in df.columns:
        df = df[df["score1"].notna() & df["score2"].notna()].copy()

    if df.empty:
        return pd.DataFrame()

    # Convert scores to integers
    df["score1"] = df["score1"].astype(int)
    df["score2"] = df["score2"].astype(int)

    # Compute training targets
    df["home_margin"] = df["score1"] - df["score2"]
    df["total_score"] = df["score1"] + df["score2"]

    # Handle ELO columns
    if "elo1_pre" not in df.columns:
        df["elo1_pre"] = 1500
    if "elo2_pre" not in df.columns:
        df["elo2_pre"] = 1500

    # Simulate spread lines based on ELO difference
    elo_diff = df["elo1_pre"] - df["elo2_pre"]
    df["spread_line"] = -(elo_diff / 28).round(1)

    # Simulate total lines based on rolling average
    avg_total = df["total_score"].rolling(window=50, min_periods=10).mean()
    df["total_line"] = avg_total.fillna(df["total_score"].mean()).round(1)

    # Compute target variables
    df["spread_covered"] = (df["home_margin"] > -df["spread_line"]).astype(int)
    df["spread_push"] = (df["home_margin"] == -df["spread_line"]).astype(int)
    df["went_over"] = (df["total_score"] > df["total_line"]).astype(int)
    df["total_push"] = (df["total_score"] == df["total_line"]).astype(int)

    # Handle missing elo_prob1
    if "elo_prob1" not in df.columns:
        df["elo_prob1"] = 1 / (1 + 10 ** (-elo_diff / 400))

    # Rename columns to match our schema
    training_df = pd.DataFrame({
        "source": "fivethirtyeight",
        "game_id": range(len(df)),
        "date": pd.to_datetime(df["date"]),
        "season": df["season"],
        "home_team": df["team1"],
        "away_team": df["team2"],
        "home_score": df["score1"],
        "away_score": df["score2"],
        "total_score": df["total_score"].values,
        "home_margin": df["home_margin"].values,
        "spread_line": df["spread_line"].values,
        "spread_covered": df["spread_covered"].values,
        "spread_push": df["spread_push"].values,
        "total_line": df["total_line"].values,
        "went_over": df["went_over"].values,
        "total_push": df["total_push"].values,
        "home_elo": df["elo1_pre"].values,
        "away_elo": df["elo2_pre"].values,
        "elo_diff": elo_diff.values,
        "elo_prob_home": df["elo_prob1"].values,
    })

    return training_df


def compute_rolling_features(df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    """Add rolling team statistics as features."""
    df = df.sort_values("date").copy()

    # We'll compute these per-team
    teams = pd.concat([df["home_team"], df["away_team"]]).unique()

    team_stats = {}
    for team in teams:
        # Games where this team played
        home_games = df[df["home_team"] == team].copy()
        away_games = df[df["away_team"] == team].copy()

        # Combine and sort
        home_games["team_score"] = home_games["home_score"]
        home_games["opp_score"] = home_games["away_score"]
        away_games["team_score"] = away_games["away_score"]
        away_games["opp_score"] = away_games["home_score"]

        all_games = pd.concat([
            home_games[["date", "team_score", "opp_score"]],
            away_games[["date", "team_score", "opp_score"]]
        ]).sort_values("date")

        # Rolling stats
        all_games["ppg"] = all_games["team_score"].rolling(lookback, min_periods=3).mean()
        all_games["papg"] = all_games["opp_score"].rolling(lookback, min_periods=3).mean()
        all_games["margin"] = all_games["team_score"] - all_games["opp_score"]
        all_games["avg_margin"] = all_games["margin"].rolling(lookback, min_periods=3).mean()

        team_stats[team] = all_games.set_index("date")[["ppg", "papg", "avg_margin"]].to_dict("index")

    # Now add features to main dataframe
    home_ppg, home_papg, home_margin = [], [], []
    away_ppg, away_papg, away_margin = [], [], []

    for _, row in df.iterrows():
        date = row["date"]
        home = row["home_team"]
        away = row["away_team"]

        # Get most recent stats before this game
        home_stats = team_stats.get(home, {})
        away_stats = team_stats.get(away, {})

        # Find the most recent date before this game
        home_dates = [d for d in home_stats.keys() if d < date]
        away_dates = [d for d in away_stats.keys() if d < date]

        if home_dates:
            latest_home = max(home_dates)
            home_ppg.append(home_stats[latest_home]["ppg"])
            home_papg.append(home_stats[latest_home]["papg"])
            home_margin.append(home_stats[latest_home]["avg_margin"])
        else:
            home_ppg.append(np.nan)
            home_papg.append(np.nan)
            home_margin.append(np.nan)

        if away_dates:
            latest_away = max(away_dates)
            away_ppg.append(away_stats[latest_away]["ppg"])
            away_papg.append(away_stats[latest_away]["papg"])
            away_margin.append(away_stats[latest_away]["avg_margin"])
        else:
            away_ppg.append(np.nan)
            away_papg.append(np.nan)
            away_margin.append(np.nan)

    df["home_ppg"] = home_ppg
    df["home_papg"] = home_papg
    df["home_avg_margin"] = home_margin
    df["away_ppg"] = away_ppg
    df["away_papg"] = away_papg
    df["away_avg_margin"] = away_margin

    # Derived features
    df["home_total_ppg"] = df["home_ppg"] + df["home_papg"]
    df["away_total_ppg"] = df["away_ppg"] + df["away_papg"]
    df["predicted_margin"] = (df["home_avg_margin"] - df["away_avg_margin"]) / 2 + 3
    df["predicted_total"] = (df["home_total_ppg"] + df["away_total_ppg"]) / 2
    df["win_pct_diff"] = (df["home_elo"] - df["away_elo"]) / 200  # Proxy for win pct
    df["ppg_diff"] = df["home_ppg"] - df["away_ppg"]

    return df


def main():
    print("=" * 60)
    print("Generating Training Data for NBA Models")
    print("=" * 60)

    output_dir = settings.data_processed_dir
    os.makedirs(output_dir, exist_ok=True)
    training_path = os.path.join(output_dir, "training_data.csv")
    historical_path = os.path.join(output_dir, "historical_games.csv")

    # ------------------------------------------------------------------
    # 1) Preferred path: build training data from REAL odds + outcomes
    #    using DatasetBuilder + FeatureEngineer so the feature space
    #    mirrors live predictions.
    # ------------------------------------------------------------------
    print("\nAttempting to build training data from real odds + outcomes...")
    builder = DatasetBuilder()

    real_training_df = builder.build_training_dataset(
        odds_path=os.path.join(output_dir, "odds_the_odds.csv"),
        outcomes_path=os.path.join(output_dir, "game_outcomes.csv"),
        output_path=training_path,
    )

    if not real_training_df.empty:
        # Ensure a canonical date column exists for downstream scripts
        if "date" not in real_training_df.columns and "game_date" in real_training_df.columns:
            real_training_df["date"] = pd.to_datetime(real_training_df["game_date"])

        # Also save as historical games for feature computation
        real_training_df.to_csv(historical_path, index=False)
        print(f"Saved historical games to {historical_path}")

        # Summary
        print("\n" + "=" * 60)
        print("Real Odds-Based Training Data Summary")
        print("=" * 60)
        print(f"Total games: {len(real_training_df)}")
        if "date" in real_training_df.columns:
            print(f"Date range: {real_training_df['date'].min()} to {real_training_df['date'].max()}")
        if "spread_covered" in real_training_df.columns:
            print(f"Spread cover rate: {real_training_df['spread_covered'].mean():.1%}")
        if "went_over" in real_training_df.columns:
            print(f"Over rate: {real_training_df['went_over'].mean():.1%}")

        print("\nFeatures available (head):")
        print(", ".join(list(real_training_df.columns)[:30]))
        return

    print("Real odds-based training data could not be built "
          "(missing or insufficient odds/outcomes). Falling back to FiveThirtyEight ELO data.\n")

    # ------------------------------------------------------------------
    # 2) Fallback path: original FiveThirtyEight-based synthetic lines.
    #    This preserves the old behaviour when real data isn't present.
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Generating Training Data from FiveThirtyEight (Fallback)")
    print("=" * 60)

    # Download data
    elo_df = download_elo_data()
    if elo_df.empty:
        print("Failed to download data. Check network connection.")
        return

    # Process to training format
    print("\nProcessing to training format...")
    training_df = process_elo_to_training_data(elo_df)
    print(f"Created {len(training_df)} training samples")

    # Add rolling features
    print("\nComputing rolling features...")
    training_df = compute_rolling_features(training_df)

    # Drop rows without features (first few games per team)
    initial_count = len(training_df)
    training_df = training_df.dropna(subset=["home_ppg", "away_ppg"])
    print(f"Dropped {initial_count - len(training_df)} rows without sufficient history")

    # Save fallback training data
    training_df.to_csv(training_path, index=False)
    print(f"\nSaved training data to {training_path}")

    # Also save as historical games for feature computation
    training_df.to_csv(historical_path, index=False)
    print(f"Saved historical games to {historical_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Training Data Summary (FiveThirtyEight Fallback)")
    print("=" * 60)
    print(f"Total games: {len(training_df)}")
    print(f"Seasons: {training_df['season'].min()} - {training_df['season'].max()}")
    print(f"Date range: {training_df['date'].min()} to {training_df['date'].max()}")
    print(f"\nSpread cover rate: {training_df['spread_covered'].mean():.1%}")
    print(f"Over rate: {training_df['went_over'].mean():.1%}")
    print("\nFeatures available:")
    print("  - ELO ratings (home_elo, away_elo, elo_diff)")
    print("  - Rolling stats (ppg, papg, avg_margin)")
    print("  - Derived (predicted_margin, predicted_total)")


if __name__ == "__main__":
    main()
