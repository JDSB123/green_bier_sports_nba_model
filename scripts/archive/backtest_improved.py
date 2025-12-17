"""
IMPROVED backtest with calibration, filtering, and better bet selection.

Improvements over baseline:
1. Probability calibration (CalibratedClassifierCV)
2. Filter out 3-6 point spreads (poor performance zone)
3. Edge-based bet selection (only bet when edge > threshold)
4. Separate reporting for filtered vs all predictions

Usage: python scripts/backtest_improved.py
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
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402

from src.config import settings  # noqa: E402
from src.modeling.features import FeatureEngineer  # noqa: E402
from src.modeling.models import find_value_bets  # noqa: E402
from src.modeling.betting import annotate_value_bets  # noqa: E402


class ImprovedWalkForwardBacktest:
    """Walk-forward backtest with calibration and smart filtering."""

    def __init__(
        self,
        min_training_games: int = 50,
        retrain_frequency: int = 20,
        use_calibration: bool = True,
        filter_small_spreads: bool = True,
        min_edge: float = 0.05,
    ):
        self.min_training_games = min_training_games
        self.retrain_frequency = retrain_frequency
        self.use_calibration = use_calibration
        self.filter_small_spreads = filter_small_spreads
        self.min_edge = min_edge
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

    def create_model(self, model_type: str, use_calibration: bool):
        """Create base model with optional calibration."""
        if model_type == "logistic":
            base_model = LogisticRegression(max_iter=1000, C=0.1)
        else:
            base_model = GradientBoostingClassifier(
                n_estimators=50, max_depth=3, learning_rate=0.1
            )

        if use_calibration:
            # Wrap with isotonic calibration
            return CalibratedClassifierCV(base_model, method='isotonic', cv=3)
        else:
            return base_model

    def run_backtest(
        self,
        games_df: pd.DataFrame,
        model_type: str = "logistic",
    ) -> pd.DataFrame:
        """Run walk-forward backtest with improvements."""
        games_df = games_df.sort_values("date").reset_index(drop=True)

        results = []
        last_train_idx = 0
        spreads_model = None
        totals_model = None
        scaler = None
        feature_cols = None

        print(f"\n{'='*70}")
        print(f"IMPROVED BACKTEST - {model_type.upper()}")
        print(f"{'='*70}")
        print(f"Calibration: {'ON' if self.use_calibration else 'OFF'}")
        print(f"Filter small spreads (3-6): {'ON' if self.filter_small_spreads else 'OFF'}")
        print(f"Minimum edge: {self.min_edge:.1%}")
        print(f"{'='*70}\n")

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

                # Train models with calibration
                spreads_model = self.create_model(model_type, self.use_calibration)
                totals_model = self.create_model(model_type, self.use_calibration)

                spreads_model.fit(
                    X_scaled[valid_spread], y_spread[valid_spread]
                )
                totals_model.fit(X_scaled[valid_total], y_total[valid_total])
                last_train_idx = idx

            # Make prediction for current game
            features = self.feature_engineer.build_game_features(game, games_df)
            if not features or scaler is None:
                continue

            # Ensure same features
            X_pred = pd.DataFrame([features])
            for col in feature_cols:
                if col not in X_pred.columns:
                    X_pred[col] = 0
            X_pred = X_pred[feature_cols].fillna(0)
            X_pred_scaled = scaler.transform(X_pred)

            # Predictions
            spread_prob = spreads_model.predict_proba(X_pred_scaled)[0][1]
            total_prob = totals_model.predict_proba(X_pred_scaled)[0][1]

            # Calculate edge
            spread_edge = abs(spread_prob - 0.5)
            total_edge = abs(total_prob - 0.5)

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

            # Determine if bet passes filters
            spread_abs = abs(spread_line) if pd.notna(spread_line) else 0
            is_small_spread = 3.0 <= spread_abs <= 6.0

            # Apply filters for "should_bet"
            spread_should_bet = True
            if self.filter_small_spreads and is_small_spread:
                spread_should_bet = False
            if spread_edge < self.min_edge:
                spread_should_bet = False

            total_should_bet = total_edge >= self.min_edge

            results.append({
                "date": game_date,
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "spread_line": spread_line,
                "spread_prob": spread_prob,
                "spread_pred": 1 if spread_prob > 0.5 else 0,
                "spread_actual": spread_covered,
                "spread_edge": spread_edge,
                "spread_should_bet": spread_should_bet,
                "is_small_spread": is_small_spread,
                "total_line": total_line,
                "total_prob": total_prob,
                "total_pred": 1 if total_prob > 0.5 else 0,
                "total_actual": went_over,
                "total_edge": total_edge,
                "total_should_bet": total_should_bet,
                "home_margin": actual_margin,
                "total_score": actual_total,
            })

        return pd.DataFrame(results)


def load_current_season_data() -> pd.DataFrame:
    """Load and prepare current season data."""
    print("Loading current season game outcomes...")

    game_outcomes_path = os.path.join(settings.data_processed_dir, "game_outcomes.csv")
    df = pd.read_csv(game_outcomes_path)
    print(f"Loaded {len(df)} games from game_outcomes.csv")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    if "home_margin" not in df.columns:
        df["home_margin"] = df["home_score"] - df["away_score"]
    if "total_score" not in df.columns:
        df["total_score"] = df["home_score"] + df["away_score"]

    print("Simulating spread and total lines based on team performance...")

    for idx, game in df.iterrows():
        home_team = game["home_team"]
        away_team = game["away_team"]
        game_date = game["date"]

        hist_home = df[(df["date"] < game_date) &
                       ((df["home_team"] == home_team) | (df["away_team"] == home_team))]
        hist_away = df[(df["date"] < game_date) &
                       ((df["home_team"] == away_team) | (df["away_team"] == away_team))]

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

        expected_margin = (home_avg_margin - away_avg_margin) / 2 + 3
        noise = np.random.normal(0, 2)
        spread_line = -round((expected_margin + noise) * 2) / 2

        expected_total = (home_avg_score + away_avg_score)
        total_noise = np.random.normal(0, 4)
        total_line = round((expected_total + total_noise) * 2) / 2

        df.at[idx, "spread_line"] = spread_line
        df.at[idx, "total_line"] = total_line

    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    return df


def evaluate_backtest(results_df: pd.DataFrame, filter_name: str = "All") -> Dict[str, float]:
    """Evaluate backtest results with detailed metrics."""
    metrics: Dict[str, float] = {}

    # Spread accuracy
    valid_spread = results_df["spread_actual"].notna()
    if valid_spread.sum() > 0:
        spread_results = results_df[valid_spread]
        metrics["spread_accuracy"] = (
            spread_results["spread_pred"] == spread_results["spread_actual"]
        ).mean()
        metrics["spread_n_bets"] = len(spread_results)

        # High confidence
        high_conf = spread_results[spread_results["spread_prob"].apply(
            lambda x: x > 0.6 or x < 0.4
        )]
        if len(high_conf) > 0:
            metrics["spread_high_conf_acc"] = (
                high_conf["spread_pred"] == high_conf["spread_actual"]
            ).mean()
            metrics["spread_high_conf_n"] = len(high_conf)

        # Brier score
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

        high_conf = total_results[total_results["total_prob"].apply(
            lambda x: x > 0.6 or x < 0.4
        )]
        if len(high_conf) > 0:
            metrics["total_high_conf_acc"] = (
                high_conf["total_pred"] == high_conf["total_actual"]
            ).mean()
            metrics["total_high_conf_n"] = len(high_conf)

    # ROI calculation
    def calculate_roi(preds, actuals):
        correct = (preds == actuals).sum()
        total = len(preds)
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
    print("NBA IMPROVED BACKTEST - 2025-2026 Season")
    print("=" * 70)

    np.random.seed(42)

    # Load data
    games_df = load_current_season_data()
    if games_df.empty:
        print("No data available for backtest")
        return

    print(f"\nTotal games: {len(games_df)}")

    # Run IMPROVED backtest
    backtest = ImprovedWalkForwardBacktest(
        min_training_games=50,
        retrain_frequency=20,
        use_calibration=True,
        filter_small_spreads=True,
        min_edge=0.05,
    )

    for model_type in ["logistic", "gradient_boosting"]:
        results = backtest.run_backtest(games_df, model_type=model_type)

        if results.empty:
            print("No results generated")
            continue

        # Evaluate ALL predictions
        all_metrics = evaluate_backtest(results, "All Predictions")

        # Evaluate FILTERED predictions (should_bet = True)
        filtered_spreads = results[results["spread_should_bet"] == True]
        filtered_totals = results[results["total_should_bet"] == True]

        print(f"\n{'='*70}")
        print(f"RESULTS COMPARISON - {model_type.upper()}")
        print(f"{'='*70}")

        print("\n--- SPREADS (ALL PREDICTIONS) ---")
        print(f"Accuracy: {all_metrics.get('spread_accuracy', 0):.1%}")
        print(f"N bets: {all_metrics.get('spread_n_bets', 0)}")
        print(f"ROI: {all_metrics.get('spread_roi', 0):+.1%}")
        print(f"Brier: {all_metrics.get('spread_brier', 0):.4f}")

        print("\n--- SPREADS (FILTERED BETS ONLY) ---")
        if len(filtered_spreads) > 0:
            filt_spread_acc = (filtered_spreads["spread_pred"] == filtered_spreads["spread_actual"]).mean()
            correct = (filtered_spreads["spread_pred"] == filtered_spreads["spread_actual"]).sum()
            total = len(filtered_spreads)
            filt_roi = (correct * (100/110) - (total - correct)) / total
            print(f"Accuracy: {filt_spread_acc:.1%}")
            print(f"N bets: {len(filtered_spreads)}")
            print(f"ROI: {filt_roi:+.1%}")
            print(f"Improvement: {(filt_spread_acc - all_metrics.get('spread_accuracy', 0))*100:+.1f} pts")

            # Show what was filtered out
            small_spread_filtered = results[results["is_small_spread"] == True]
            low_edge_filtered = results[results["spread_edge"] < backtest.min_edge]
            print(f"\nFiltered out:")
            print(f"  - Small spreads (3-6): {len(small_spread_filtered)} games")
            print(f"  - Low edge (<{backtest.min_edge:.0%}): {len(low_edge_filtered)} games")
        else:
            print("No bets passed filter")

        print("\n--- TOTALS (ALL PREDICTIONS) ---")
        print(f"Accuracy: {all_metrics.get('total_accuracy', 0):.1%}")
        print(f"N bets: {all_metrics.get('total_n_bets', 0)}")
        print(f"ROI: {all_metrics.get('total_roi', 0):+.1%}")

        print("\n--- TOTALS (FILTERED BETS ONLY) ---")
        if len(filtered_totals) > 0:
            filt_total_acc = (filtered_totals["total_pred"] == filtered_totals["total_actual"]).mean()
            correct = (filtered_totals["total_pred"] == filtered_totals["total_actual"]).sum()
            total = len(filtered_totals)
            filt_roi = (correct * (100/110) - (total - correct)) / total
            print(f"Accuracy: {filt_total_acc:.1%}")
            print(f"N bets: {len(filtered_totals)}")
            print(f"ROI: {filt_roi:+.1%}")
            print(f"Improvement: {(filt_total_acc - all_metrics.get('total_accuracy', 0))*100:+.1f} pts")
        else:
            print("No bets passed filter")

        # Save results
        output_path = os.path.join(
            settings.data_processed_dir,
            f"backtest_improved_{model_type}.csv",
        )
        results.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")

    print("\n" + "=" * 70)
    print("IMPROVED BACKTEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
