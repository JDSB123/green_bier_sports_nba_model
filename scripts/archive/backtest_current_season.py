"""
Backtest spreads/totals models on current 2025-26 season with NO LEAKAGE.

Uses walk-forward validation: train on past games, predict next game.
Uses game_outcomes.csv for results and simulates betting lines based on team stats.

Usage: python scripts/backtest_current_season.py
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Tuple

import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.config import settings  # noqa: E402
from src.modeling.features import FeatureEngineer  # noqa: E402
from src.modeling.models import find_value_bets  # noqa: E402
from src.modeling.betting import annotate_value_bets  # noqa: E402


# ============================================================
# BACKTEST ENGINE
# ============================================================

class WalkForwardBacktest:
    """Walk-forward backtest with no lookahead bias."""

    def __init__(
        self,
        min_training_games: int = 50,
        retrain_frequency: int = 20,
    ):
        self.min_training_games = min_training_games
        self.retrain_frequency = retrain_frequency
        self.feature_engineer = FeatureEngineer(lookback=10)

    def prepare_training_data(
        self,
        games_df: pd.DataFrame,
        cutoff_date: pd.Timestamp,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Prepare training data using only games before cutoff."""
        training_games = games_df[games_df["date"] < cutoff_date].copy()

        if len(training_games) < self.min_training_games:
            return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float)

        features_list = []
        spread_targets = []
        total_targets = []

        for _, game in training_games.iterrows():
            features = self.feature_engineer.build_game_features(
                game, training_games
            )
            if features:
                features_list.append(features)

                # Target: did home team cover spread?
                spread_line = game.get("spread_line")
                home_margin = game.get("home_margin")
                if pd.notna(spread_line) and pd.notna(home_margin):
                    covered = 1 if home_margin > -spread_line else 0
                    spread_targets.append(covered)
                else:
                    spread_targets.append(np.nan)

                # Target: did game go over?
                total_line = game.get("total_line")
                total_score = game.get("total_score")
                if pd.notna(total_line) and pd.notna(total_score):
                    over = 1 if total_score > total_line else 0
                    total_targets.append(over)
                else:
                    total_targets.append(np.nan)

        if not features_list:
            return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float)

        return (
            pd.DataFrame(features_list),
            pd.Series(spread_targets),
            pd.Series(total_targets),
        )

    def run_backtest(
        self,
        games_df: pd.DataFrame,
        model_type: str = "logistic",
    ) -> pd.DataFrame:
        """Run walk-forward backtest."""
        games_df = games_df.sort_values("date").reset_index(drop=True)

        results = []
        last_train_idx = 0
        spreads_model = None
        totals_model = None
        scaler = None
        feature_cols = None

        print(f"\nRunning walk-forward backtest on {len(games_df)} games...")
        print(f"Model type: {model_type}")
        print(f"Min training games: {self.min_training_games}")
        print(f"Retrain frequency: {self.retrain_frequency}\n")

        for idx, game in games_df.iterrows():
            game_date = pd.to_datetime(game["date"])

            # Skip if not enough history
            games_before = len(games_df[games_df["date"] < game_date])
            if games_before < self.min_training_games:
                continue

            # Retrain model periodically
            should_retrain = (
                spreads_model is None or
                idx - last_train_idx >= self.retrain_frequency
            )

            if should_retrain:
                X_train, y_spread, y_total = self.prepare_training_data(
                    games_df, game_date
                )

                if X_train.empty:
                    continue

                # Remove rows with NaN targets
                valid_spread = ~y_spread.isna()
                valid_total = ~y_total.isna()

                if valid_spread.sum() < 30 or valid_total.sum() < 30:
                    continue

                # Get feature columns
                feature_cols = X_train.columns.tolist()

                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_train.fillna(0))

                # Train spreads model
                if model_type == "logistic":
                    spreads_model = LogisticRegression(max_iter=1000, C=0.1)
                    totals_model = LogisticRegression(max_iter=1000, C=0.1)
                else:
                    spreads_model = GradientBoostingClassifier(
                        n_estimators=50, max_depth=3, learning_rate=0.1
                    )
                    totals_model = GradientBoostingClassifier(
                        n_estimators=50, max_depth=3, learning_rate=0.1
                    )

                spreads_model.fit(
                    X_scaled[valid_spread], y_spread[valid_spread]
                )
                totals_model.fit(X_scaled[valid_total], y_total[valid_total])
                last_train_idx = idx

            # Make prediction for current game
            features = self.feature_engineer.build_game_features(game, games_df)
            if not features or scaler is None:
                continue

            # Ensure same features (handle missing columns)
            X_pred = pd.DataFrame([features])
            for col in feature_cols:
                if col not in X_pred.columns:
                    X_pred[col] = 0
            X_pred = X_pred[feature_cols].fillna(0)
            X_pred_scaled = scaler.transform(X_pred)

            # Predictions
            spread_prob = spreads_model.predict_proba(X_pred_scaled)[0][1]
            total_prob = totals_model.predict_proba(X_pred_scaled)[0][1]

            # Actual results
            actual_margin = game.get("home_margin", np.nan)
            actual_total = game.get("total_score", np.nan)
            spread_line = game.get("spread_line", np.nan)
            total_line = game.get("total_line", np.nan)

            if pd.notna(spread_line) and pd.notna(actual_margin):
                spread_covered = 1 if actual_margin > -spread_line else 0
            else:
                spread_covered = np.nan

            if pd.notna(total_line) and pd.notna(actual_total):
                went_over = 1 if actual_total > total_line else 0
            else:
                went_over = np.nan

            results.append({
                "date": game_date,
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "spread_line": spread_line,
                "spread_prob": spread_prob,
                "spread_pred": 1 if spread_prob > 0.5 else 0,
                "spread_actual": spread_covered,
                "total_line": total_line,
                "total_prob": total_prob,
                "total_pred": 1 if total_prob > 0.5 else 0,
                "total_actual": went_over,
                "home_margin": actual_margin,
                "total_score": actual_total,
            })

        return pd.DataFrame(results)


# ============================================================
# LOAD CURRENT SEASON DATA
# ============================================================

def load_current_season_data() -> pd.DataFrame:
    """
    Load 2025-26 season data with simulated spread/total lines.
    """
    print("Loading current season game outcomes...")

    game_outcomes_path = os.path.join(settings.data_processed_dir, "game_outcomes.csv")
    df = pd.read_csv(game_outcomes_path)
    print(f"Loaded {len(df)} games from game_outcomes.csv")

    # Ensure required columns
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # Calculate margins and totals if not present
    if "home_margin" not in df.columns:
        df["home_margin"] = df["home_score"] - df["away_score"]
    if "total_score" not in df.columns:
        df["total_score"] = df["home_score"] + df["away_score"]

    # Simulate realistic spread and total lines based on rolling team stats
    print("Simulating spread and total lines based on team performance...")

    # Calculate rolling team stats for each team
    team_stats = {}

    for idx, game in df.iterrows():
        home_team = game["home_team"]
        away_team = game["away_team"]
        game_date = game["date"]

        # Get historical games for both teams before this game
        hist_home = df[(df["date"] < game_date) &
                       ((df["home_team"] == home_team) | (df["away_team"] == home_team))]
        hist_away = df[(df["date"] < game_date) &
                       ((df["home_team"] == away_team) | (df["away_team"] == away_team))]

        # Calculate home team stats
        if len(hist_home) >= 5:
            home_margins = []
            home_scores = []
            for _, h in hist_home.tail(10).iterrows():
                if h["home_team"] == home_team:
                    home_margins.append(h["home_score"] - h["away_score"])
                    home_scores.append(h["home_score"])
                else:
                    home_margins.append(h["away_score"] - h["home_score"])
                    home_scores.append(h["away_score"])
            home_avg_margin = np.mean(home_margins)
            home_avg_score = np.mean(home_scores)
        else:
            home_avg_margin = 0
            home_avg_score = 110

        # Calculate away team stats
        if len(hist_away) >= 5:
            away_margins = []
            away_scores = []
            for _, a in hist_away.tail(10).iterrows():
                if a["home_team"] == away_team:
                    away_margins.append(a["home_score"] - a["away_score"])
                    away_scores.append(a["home_score"])
                else:
                    away_margins.append(a["away_score"] - a["home_score"])
                    away_scores.append(a["away_score"])
            away_avg_margin = np.mean(away_margins)
            away_avg_score = np.mean(away_scores)
        else:
            away_avg_margin = 0
            away_avg_score = 110

        # Simulate spread line (home team perspective)
        # Margin diff + home court advantage (3 points) + noise
        expected_margin = (home_avg_margin - away_avg_margin) / 2 + 3
        noise = np.random.normal(0, 2)
        spread_line = -round((expected_margin + noise) * 2) / 2  # Round to 0.5

        # Simulate total line
        expected_total = (home_avg_score + away_avg_score)
        total_noise = np.random.normal(0, 4)
        total_line = round((expected_total + total_noise) * 2) / 2  # Round to 0.5

        df.at[idx, "spread_line"] = spread_line
        df.at[idx, "total_line"] = total_line

    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    return df


def evaluate_backtest(results_df: pd.DataFrame) -> Dict[str, float]:
    """Evaluate backtest results."""
    metrics: Dict[str, float] = {}

    # Spread accuracy
    valid_spread = results_df["spread_actual"].notna()
    if valid_spread.sum() > 0:
        spread_results = results_df[valid_spread]
        metrics["spread_accuracy"] = (
            spread_results["spread_pred"] == spread_results["spread_actual"]
        ).mean()
        metrics["spread_n_bets"] = len(spread_results)

        # Confidence buckets
        high_conf = spread_results[spread_results["spread_prob"].apply(
            lambda x: x > 0.6 or x < 0.4
        )]
        if len(high_conf) > 0:
            metrics["spread_high_conf_acc"] = (
                high_conf["spread_pred"] == high_conf["spread_actual"]
            ).mean()
            metrics["spread_high_conf_n"] = len(high_conf)

        # Brier score for spreads
        probs_spread = spread_results["spread_prob"].values
        actual_spread = spread_results["spread_actual"].values
        metrics["spread_brier"] = float(np.mean((probs_spread - actual_spread) ** 2))

    # Totals accuracy
    valid_total = results_df["total_actual"].notna()
    if valid_total.sum() > 0:
        total_results = results_df[valid_total]
        metrics["total_accuracy"] = (
            total_results["total_pred"] == total_results["total_actual"]
        ).mean()
        metrics["total_n_bets"] = len(total_results)

        # High confidence
        high_conf = total_results[total_results["total_prob"].apply(
            lambda x: x > 0.6 or x < 0.4
        )]
        if len(high_conf) > 0:
            metrics["total_high_conf_acc"] = (
                high_conf["total_pred"] == high_conf["total_actual"]
            ).mean()
            metrics["total_high_conf_n"] = len(high_conf)

    # ROI simulation (assuming -110 odds)
    def calculate_roi(preds, actuals):
        correct = (preds == actuals).sum()
        total = len(preds)
        # Win: +100/110, Lose: -1
        profit = correct * (100/110) - (total - correct)
        return profit / total if total > 0 else 0

    if valid_spread.sum() > 0:
        metrics["spread_roi"] = calculate_roi(
            spread_results["spread_pred"].values,
            spread_results["spread_actual"].values
        )
        high_conf = spread_results[spread_results["spread_prob"].apply(
            lambda x: x > 0.6 or x < 0.4
        )]
        if len(high_conf) > 0:
            metrics["spread_high_conf_roi"] = calculate_roi(
                high_conf["spread_pred"].values,
                high_conf["spread_actual"].values,
            )

    if valid_total.sum() > 0:
        metrics["total_roi"] = calculate_roi(
            total_results["total_pred"].values,
            total_results["total_actual"].values
        )
        high_conf = total_results[total_results["total_prob"].apply(
            lambda x: x > 0.6 or x < 0.4
        )]
        if len(high_conf) > 0:
            metrics["total_high_conf_roi"] = calculate_roi(
                high_conf["total_pred"].values,
                high_conf["total_actual"].values,
            )

    return metrics


def main():
    print("=" * 70)
    print("NBA CURRENT SEASON BACKTEST - NO LEAKAGE")
    print("2025-2026 Season")
    print("=" * 70)

    # Set random seed for reproducible line simulation
    np.random.seed(42)

    # Load data
    games_df = load_current_season_data()
    if games_df.empty:
        print("No data available for backtest")
        return

    print(f"\nTotal games: {len(games_df)}")
    print(f"Date range: {games_df['date'].min()} to {games_df['date'].max()}")

    # Run backtest with both model types
    backtest = WalkForwardBacktest(
        min_training_games=50,
        retrain_frequency=20,
    )

    for model_type in ["logistic", "gradient_boosting"]:
        print("\n" + "=" * 70)
        print(f"MODEL: {model_type.upper()}")
        print("=" * 70)

        results = backtest.run_backtest(games_df, model_type=model_type)

        if results.empty:
            print("No results generated")
            continue

        metrics = evaluate_backtest(results)

        print("\n--- SPREADS ---")
        print(f"Accuracy: {metrics.get('spread_accuracy', 0):.1%}")
        print(f"N bets: {metrics.get('spread_n_bets', 0)}")
        print(f"ROI: {metrics.get('spread_roi', 0):+.1%}")
        if "spread_brier" in metrics:
            print(f"Brier: {metrics['spread_brier']:.4f}")
        if "spread_high_conf_acc" in metrics:
            acc = metrics["spread_high_conf_acc"]
            n = metrics["spread_high_conf_n"]
            print(f"High conf (>60%) accuracy: {acc:.1%} (n={n})")
        if "spread_high_conf_roi" in metrics:
            print(f"High conf (>60%) ROI: {metrics['spread_high_conf_roi']:+.1%}")

        print("\n--- TOTALS ---")
        print(f"Accuracy: {metrics.get('total_accuracy', 0):.1%}")
        print(f"N bets: {metrics.get('total_n_bets', 0)}")
        print(f"ROI: {metrics.get('total_roi', 0):+.1%}")
        if "total_high_conf_acc" in metrics:
            acc = metrics["total_high_conf_acc"]
            n = metrics["total_high_conf_n"]
            print(f"High conf (>60%) accuracy: {acc:.1%} (n={n})")
        if "total_high_conf_roi" in metrics:
            print(f"High conf (>60%) ROI: {metrics['total_high_conf_roi']:+.1%}")

        # Save results
        output_path = os.path.join(
            settings.data_processed_dir,
            f"backtest_current_season_{model_type}.csv",
        )
        results.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")

        # Generate value bet report
        value_input = results.copy()
        std_implied = 110.0 / (110.0 + 100.0)
        value_input["spread_implied_prob"] = std_implied
        value_input["total_implied_prob"] = std_implied

        value_bets = find_value_bets(value_input)
        if not value_bets.empty:
            value_bets = annotate_value_bets(
                value_bets,
                prob_col="model_prob",
                implied_col="implied_prob",
            )
            vb_path = os.path.join(
                settings.data_processed_dir,
                f"backtest_current_season_value_bets_{model_type}.csv",
            )
            value_bets.to_csv(vb_path, index=False)
            print(f"Value bets saved to {vb_path}")

    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
