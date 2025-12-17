"""
ULTIMATE BACKTEST with all improvements:
- Enhanced features (clutch, opponent-adjusted, dynamic HCA)
- XGBoost ensemble
- Smart filtering (no calibration)
- Separate home/away models (optional)

Usage: python scripts/backtest_ultimate.py
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Tuple, Optional

import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

# Try to import XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    print("Warning: XGBoost not installed. Using GradientBoosting only.")
    HAS_XGBOOST = False

from src.config import settings  # noqa: E402
from src.modeling.features import FeatureEngineer  # noqa: E402


class UltimateBacktest:
    """Ultimate backtest with all improvements."""

    def __init__(
        self,
        min_training_games: int = 50,
        retrain_frequency: int = 20,
        use_ensemble: bool = True,
        separate_home_away: bool = False,
        filter_small_spreads: bool = True,
        min_edge: float = 0.05,
    ):
        self.min_training_games = min_training_games
        self.retrain_frequency = retrain_frequency
        self.use_ensemble = use_ensemble and HAS_XGBOOST
        self.separate_home_away = separate_home_away
        self.filter_small_spreads = filter_small_spreads
        self.min_edge = min_edge
        self.feature_engineer = FeatureEngineer(lookback=10)

        print(f"\n{'='*70}")
        print(f"ULTIMATE BACKTEST CONFIGURATION")
        print(f"{'='*70}")
        print(f"Enhanced Features: ON (clutch, opponent-adj, dynamic HCA)")
        print(f"Ensemble (XGBoost): {'ON' if self.use_ensemble else 'OFF'}")
        print(f"Separate Home/Away Models: {'ON' if separate_home_away else 'OFF'}")
        print(f"Filter small spreads (3-6): {'ON' if filter_small_spreads else 'OFF'}")
        print(f"Minimum edge: {min_edge:.1%}")
        print(f"{'='*70}\n")

    def prepare_training_data(
        self,
        games_df: pd.DataFrame,
        cutoff_date: pd.Timestamp,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Prepare training data with enhanced features."""
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

    def create_ensemble_model(self):
        """Create ensemble model with multiple algorithms."""
        if self.use_ensemble:
            estimators = [
                ('lr', LogisticRegression(max_iter=1000, C=0.1)),
                ('gb', GradientBoostingClassifier(n_estimators=50, max_depth=3, learning_rate=0.1)),
                ('xgb', xgb.XGBClassifier(
                    n_estimators=50,
                    max_depth=3,
                    learning_rate=0.1,
                    eval_metric='logloss',
                    random_state=42
                ))
            ]
            return VotingClassifier(
                estimators=estimators,
                voting='soft',
                weights=[1, 2, 2]  # Weight tree models more
            )
        else:
            return GradientBoostingClassifier(
                n_estimators=50, max_depth=3, learning_rate=0.1
            )

    def run_backtest(
        self,
        games_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Run ultimate backtest with all improvements."""
        games_df = games_df.sort_values("date").reset_index(drop=True)

        results = []
        last_train_idx = 0

        # Models for spread and totals
        spreads_model = None
        totals_model = None

        # Separate home/away models (optional)
        spreads_home_model = None
        spreads_away_model = None

        scaler = None
        feature_cols = None

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

                # Train models
                spreads_model = self.create_ensemble_model()
                totals_model = self.create_ensemble_model()

                spreads_model.fit(
                    X_scaled[valid_spread], y_spread[valid_spread]
                )
                totals_model.fit(X_scaled[valid_total], y_total[valid_total])

                # Train separate home/away models if requested
                if self.separate_home_away:
                    # Split by whether prediction is for home or away
                    # Home cover = y_spread == 1, Away cover = y_spread == 0
                    home_predictions_mask = y_spread == 1
                    away_predictions_mask = y_spread == 0

                    home_valid = valid_spread & home_predictions_mask
                    away_valid = valid_spread & away_predictions_mask

                    if home_valid.sum() >= 20 and away_valid.sum() >= 20:
                        spreads_home_model = self.create_ensemble_model()
                        spreads_away_model = self.create_ensemble_model()

                        spreads_home_model.fit(
                            X_scaled[home_valid], y_spread[home_valid]
                        )
                        spreads_away_model.fit(
                            X_scaled[away_valid], y_spread[away_valid]
                        )

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

            # If using separate models, get alternative prediction
            spread_prob_alt = None
            if self.separate_home_away and spreads_home_model and spreads_away_model:
                if spread_prob > 0.5:  # Predicting home cover
                    spread_prob_alt = spreads_home_model.predict_proba(X_pred_scaled)[0][1]
                else:  # Predicting away cover
                    spread_prob_alt = spreads_away_model.predict_proba(X_pred_scaled)[0][1]

                # Use average of both models
                spread_prob = (spread_prob + spread_prob_alt) / 2

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

            # Apply filters
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
    print(f"Loaded {len(df)} games")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    if "home_margin" not in df.columns:
        df["home_margin"] = df["home_score"] - df["away_score"]
    if "total_score" not in df.columns:
        df["total_score"] = df["home_score"] + df["away_score"]

    print("Simulating betting lines...")

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


def evaluate_results(results_df: pd.DataFrame, name: str = ""):
    """Evaluate and print results."""
    print(f"\n{'='*70}")
    print(f"RESULTS{' - ' + name if name else ''}")
    print(f"{'='*70}")

    # All predictions
    valid_spread = results_df["spread_actual"].notna()
    valid_total = results_df["total_actual"].notna()

    if valid_spread.sum() > 0:
        all_spread = results_df[valid_spread]
        all_acc = (all_spread["spread_pred"] == all_spread["spread_actual"]).mean()
        all_correct = (all_spread["spread_pred"] == all_spread["spread_actual"]).sum()
        all_total = len(all_spread)
        all_roi = (all_correct * (100/110) - (all_total - all_correct)) / all_total

        print(f"\nSPREADS (ALL {all_total} PREDICTIONS):")
        print(f"  Accuracy: {all_acc:.1%}")
        print(f"  ROI: {all_roi:+.1%}")

    # Filtered predictions
    filtered_spread = results_df[results_df["spread_should_bet"] == True]
    if len(filtered_spread) > 0:
        filt_acc = (filtered_spread["spread_pred"] == filtered_spread["spread_actual"]).mean()
        filt_correct = (filtered_spread["spread_pred"] == filtered_spread["spread_actual"]).sum()
        filt_total = len(filtered_spread)
        filt_roi = (filt_correct * (100/110) - (filt_total - filt_correct)) / filt_total

        print(f"\nSPREADS (FILTERED {filt_total} BETS):")
        print(f"  Accuracy: {filt_acc:.1%}")
        print(f"  ROI: {filt_roi:+.1%}")
        if valid_spread.sum() > 0:
            print(f"  Improvement: {(filt_acc - all_acc)*100:+.1f} pts accuracy")

        # Show what was filtered
        small_filtered = len(results_df[results_df["is_small_spread"] == True])
        low_edge_filtered = len(results_df[results_df["spread_edge"] < results_df["spread_edge"].quantile(0.5)])

        print(f"\n  Filtered out:")
        print(f"    - Small spreads (3-6): {small_filtered} games")
        print(f"    - Low confidence bets: {len(results_df) - filt_total - small_filtered} games")

    # Totals
    if valid_total.sum() > 0:
        all_total = results_df[valid_total]
        tot_acc = (all_total["total_pred"] == all_total["total_actual"]).mean()
        tot_correct = (all_total["total_pred"] == all_total["total_actual"]).sum()
        tot_n = len(all_total)
        tot_roi = (tot_correct * (100/110) - (tot_n - tot_correct)) / tot_n

        print(f"\nTOTALS (ALL {tot_n} PREDICTIONS):")
        print(f"  Accuracy: {tot_acc:.1%}")
        print(f"  ROI: {tot_roi:+.1%}")

    # Filtered totals
    filtered_total = results_df[results_df["total_should_bet"] == True]
    if len(filtered_total) > 0:
        tot_filt_acc = (filtered_total["total_pred"] == filtered_total["total_actual"]).mean()
        tot_filt_correct = (filtered_total["total_pred"] == filtered_total["total_actual"]).sum()
        tot_filt_n = len(filtered_total)
        tot_filt_roi = (tot_filt_correct * (100/110) - (tot_filt_n - tot_filt_correct)) / tot_filt_n

        print(f"\nTOTALS (FILTERED {tot_filt_n} BETS):")
        print(f"  Accuracy: {tot_filt_acc:.1%}")
        print(f"  ROI: {tot_filt_roi:+.1%}")


def main():
    print("=" * 70)
    print("ULTIMATE NBA BACKTEST - ALL IMPROVEMENTS")
    print("=" * 70)

    np.random.seed(42)

    # Load data
    games_df = load_current_season_data()
    if games_df.empty:
        print("No data available")
        return

    print(f"\nTotal games: {len(games_df)}")

    # Run ultimate backtest
    backtest = UltimateBacktest(
        min_training_games=50,
        retrain_frequency=20,
        use_ensemble=True,
        separate_home_away=False,  # Can test True later
        filter_small_spreads=True,
        min_edge=0.05,
    )

    results = backtest.run_backtest(games_df)

    if results.empty:
        print("No results generated")
        return

    # Evaluate
    evaluate_results(results, "ULTIMATE MODEL")

    # Save results
    output_path = os.path.join(
        settings.data_processed_dir,
        "backtest_ultimate_results.csv",
    )
    results.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
