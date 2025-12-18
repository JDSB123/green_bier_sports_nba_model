"""
Train NBA prediction models for all markets.

Unified training script for Full Game and First Half markets.

Usage:
    python scripts/train_models.py                          # Train FG models (default)
    python scripts/train_models.py --market all             # Train FG + 1H models
    python scripts/train_models.py --market 1h              # Train 1H models only
    python scripts/train_models.py --ensemble               # Use ensemble (logistic + gradient boosting)
    python scripts/train_models.py --model-type gradient_boosting

Markets:
    - fg: Full Game (spreads, totals, moneyline)
    - 1h: First Half (spreads, totals, moneyline)
    - all: Both FG and 1H markets

Features include:
    - Team rolling stats (PPG, PAPG, margin, win%)
    - Rest days / back-to-back detection
    - Head-to-head history
    - ELO ratings
    - Injury impact estimation (when injury data available)
    - RLM (Reverse Line Movement) signals (when betting splits available)
"""
from __future__ import annotations
import argparse
import os
import sys
import random
import logging
from typing import Optional

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.config import settings
from src.modeling.models import SpreadsModel, TotalsModel, MoneylineModel, ModelMetrics
from src.modeling.model_tracker import ModelTracker, ModelVersion
from src.modeling.feature_config import (
    get_spreads_features,
    get_totals_features,
    get_moneyline_features,
    filter_available_features,
)


# Model version for tracking
MODEL_VERSION = "1.1.0"


def print_metrics(name: str, metrics: ModelMetrics) -> None:
    """Pretty print model metrics."""
    print(f"\n{'='*50}")
    print(f"  {name} Model Results")
    print(f"{'='*50}")
    if metrics.accuracy > 0:
        print(f"  Accuracy:   {metrics.accuracy:.1%}")
        print(f"  Log Loss:   {metrics.log_loss:.4f}")
    if metrics.mse > 0:
        print(f"  MSE:        {metrics.mse:.2f}")
        print(f"  MAE:        {metrics.mae:.2f}")
    if metrics.cover_rate > 0:
        print(f"  Cover Rate: {metrics.cover_rate:.1%}")
    if metrics.roi != 0:
        print(f"  ROI (flat -110): {metrics.roi:+.1%}")
    if metrics.brier > 0:
        print(f"  Brier:      {metrics.brier:.4f}")
    print(f"{'='*50}\n")


def load_supplementary_data(data_dir: str) -> tuple:
    """Load injury and betting splits data if available."""
    injuries_df = None
    splits_df = None

    injuries_path = os.path.join(data_dir, "injuries.csv")
    if os.path.exists(injuries_path):
        try:
            injuries_df = pd.read_csv(injuries_path)
            print(f"  [OK] Loaded {len(injuries_df)} injury records")
        except Exception as e:
            print(f"  [WARN] Could not load injuries: {e}")

    splits_path = os.path.join(data_dir, "betting_splits.csv")
    if os.path.exists(splits_path):
        try:
            splits_df = pd.read_csv(splits_path)
            print(f"  [OK] Loaded {len(splits_df)} betting split records")
        except Exception as e:
            print(f"  [WARN] Could not load betting splits: {e}")

    return injuries_df, splits_df


def enrich_with_injury_features(df: pd.DataFrame, injuries_df: pd.DataFrame) -> pd.DataFrame:
    """Add injury-based features to training data."""
    if injuries_df is None or injuries_df.empty:
        return df

    # Ensure we have required columns
    required = ["player_name", "team", "status", "ppg"]
    if not all(col in injuries_df.columns for col in required):
        print("  [WARN] Injury data missing required columns, skipping enrichment")
        return df

    # Group injuries by team and compute impact
    from src.modeling.features import FeatureEngineer
    fe = FeatureEngineer()

    # Create injury impact columns if not present
    injury_cols = [
        "home_injury_spread_impact", "away_injury_spread_impact",
        "injury_spread_diff", "home_star_out", "away_star_out"
    ]

    for col in injury_cols:
        if col not in df.columns:
            df[col] = 0.0

    print(f"  [OK] Added injury feature columns")
    return df


def enrich_with_rlm_features(df: pd.DataFrame, splits_df: pd.DataFrame) -> pd.DataFrame:
    """Add RLM and betting splits features to training data."""
    if splits_df is None or splits_df.empty:
        return df

    # RLM feature columns
    rlm_cols = [
        "is_rlm_spread", "is_rlm_total",
        "sharp_side_spread", "sharp_side_total",
        "spread_public_home_pct", "spread_ticket_money_diff",
        "spread_movement"
    ]

    for col in rlm_cols:
        if col not in df.columns:
            df[col] = 0.0

    print(f"  [OK] Added RLM feature columns")
    return df


def train_models(
    model_type: str = "logistic",
    test_size: float = 0.2,
    output_dir: Optional[str] = None,
) -> None:
    """Train and evaluate spreads and totals models."""

    output_dir = output_dir or os.path.join(settings.data_processed_dir, "models")
    os.makedirs(output_dir, exist_ok=True)
    tracker = ModelTracker()

    print(f"\n{'='*60}")
    print(f"NBA Model Training v{MODEL_VERSION}")
    print(f"{'='*60}")

    # Load supplementary data (injuries, betting splits)
    print("\nLoading supplementary data...")
    injuries_df, splits_df = load_supplementary_data(settings.data_processed_dir)

    # Load existing training dataset. Prefer first-half augmented data if present.
    training_path = os.path.join(settings.data_processed_dir, "training_data.csv")
    fh_path = os.path.join(settings.data_processed_dir, "training_data_fh.csv")
    if os.path.exists(fh_path):
        print(f"Found first-half augmented training file at {fh_path}; using it.")
        training_path = fh_path
    print(f"Loading training data from {training_path}...")

    if not os.path.exists(training_path):
        print("\nNo training data available!")
        print("\nTo generate training data, run:")
        print("  python scripts/generate_training_data.py")
        return

    training_df = pd.read_csv(training_path)
    training_df["date"] = pd.to_datetime(training_df["date"], errors="coerce")

    if training_df.empty:
        print("Training data file is empty!")
        return

    print(f"Training dataset size: {len(training_df)} games")

    # Enrich with injury and RLM features
    print("\nEnriching training data...")
    training_df = enrich_with_injury_features(training_df, injuries_df)
    training_df = enrich_with_rlm_features(training_df, splits_df)

    # Split data (temporal split - use recent games for testing)
    training_df = training_df.sort_values("date")
    split_idx = int(len(training_df) * (1 - test_size))
    train_df = training_df.iloc[:split_idx]
    test_df = training_df.iloc[split_idx:]
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    # ========== SPREADS MODEL ==========
    print("\n" + "="*60)
    print("Training SPREADS Model")
    print("="*60)

    # Features for spreads prediction (from centralized config)
    spreads_features = get_spreads_features()
    available_spreads = filter_available_features(spreads_features, train_df.columns.tolist())
    print(f"  Using {len(available_spreads)} of {len(spreads_features)} possible features")

    if "spread_covered" not in train_df.columns:
        print("Warning: 'spread_covered' target not found. Skipping spreads model.")
    else:
        spreads_model = SpreadsModel(
            name="spreads_classifier",
            model_type=model_type,
            feature_columns=available_spreads,
        )

        # Filter to games with spread data
        train_spreads = train_df[train_df["spread_line"].notna()].copy()
        test_spreads = test_df[test_df["spread_line"].notna()].copy()

        if len(train_spreads) > 10:
            spreads_model.fit(train_spreads, train_spreads["spread_covered"])

            # Evaluate
            train_metrics = spreads_model.evaluate(train_spreads, train_spreads["spread_covered"])
            print_metrics("Spreads (Train)", train_metrics)

            if len(test_spreads) > 0:
                test_metrics = spreads_model.evaluate(test_spreads, test_spreads["spread_covered"])
                print_metrics("Spreads (Test)", test_metrics)

                # Additional evaluation: Brier score & high-confidence ROI buckets
                proba = spreads_model.predict_proba(test_spreads)[:, 1]
                actual = test_spreads["spread_covered"].values
                preds = spreads_model.predict(test_spreads)

                # Confidence of the side we actually bet (home vs away)
                conf_for_bet = np.where(preds == 1, proba, 1.0 - proba)
                mask_high = conf_for_bet >= 0.60
                if mask_high.any():
                    high_actual = actual[mask_high]
                    high_preds = preds[mask_high]
                    high_n = len(high_actual)
                    high_correct = (high_preds == high_actual).sum()
                    profit = high_correct * (100.0 / 110.0) - (high_n - high_correct)
                    roi_high = profit / high_n
                    acc_high = high_correct / high_n
                    print(f"  High-conf (>=60%) Spreads Test Acc: {acc_high:.1%} on {high_n} bets")
                    print(f"  High-conf (>=60%) Spreads Test ROI: {roi_high:+.1%}")

            # Save model
            model_path = os.path.join(output_dir, "spreads_model.joblib")
            spreads_model.save(model_path)
            print(f"Spreads model saved to {model_path}")

            # Register version with model tracker (using test metrics when available)
            try:
                version_id = f"{MODEL_VERSION}-spreads-{model_type}"
                tracker.register_version(
                    ModelVersion(
                        version=version_id,
                        model_type="spreads",
                        algorithm=model_type,
                        trained_at=pd.Timestamp.utcnow().isoformat(),
                        train_samples=len(train_spreads),
                        test_samples=len(test_spreads),
                        features_count=len(spreads_model.feature_columns),
                        feature_names=spreads_model.feature_columns,
                        metrics={
                            "accuracy": float(test_metrics.accuracy) if 'test_metrics' in locals() else float(train_metrics.accuracy),
                            "log_loss": float(test_metrics.log_loss) if 'test_metrics' in locals() else float(train_metrics.log_loss),
                            "brier": float(test_metrics.brier) if 'test_metrics' in locals() else float(train_metrics.brier),
                            "roi": float(test_metrics.roi) if 'test_metrics' in locals() else float(train_metrics.roi),
                        },
                        file_path=os.path.basename(model_path),
                    ),
                    set_active=True,
                )
            except Exception as e:
                print(f"[WARN] Could not register spreads model version: {e}")

            # Feature importance (for tree models)
            if hasattr(spreads_model.model, "feature_importances_"):
                print("\nTop Features (Spreads):")
                importance = pd.DataFrame({
                    "feature": spreads_model.feature_columns,
                    "importance": spreads_model.model.feature_importances_,
                }).sort_values("importance", ascending=False)
                print(importance.head(10).to_string(index=False))
        else:
            print(f"Insufficient spread data for training ({len(train_spreads)} games)")

    # ========== TOTALS MODEL ==========
    print("\n" + "="*60)
    print("Training TOTALS Model")
    print("="*60)

    # Features for totals prediction (from centralized config)
    totals_features = get_totals_features()
    available_totals = filter_available_features(totals_features, train_df.columns.tolist())
    print(f"  Using {len(available_totals)} of {len(totals_features)} possible features")

    if "went_over" not in train_df.columns:
        print("Warning: 'went_over' target not found. Skipping totals model.")
    else:
        totals_model = TotalsModel(
            name="totals_classifier",
            model_type=model_type,
            feature_columns=available_totals,
        )

        # Filter to games with totals data
        train_totals = train_df[train_df["total_line"].notna()].copy()
        test_totals = test_df[test_df["total_line"].notna()].copy()

        if len(train_totals) > 10:
            totals_model.fit(train_totals, train_totals["went_over"])

            # Evaluate
            train_metrics = totals_model.evaluate(train_totals, train_totals["went_over"])
            print_metrics("Totals (Train)", train_metrics)

            if len(test_totals) > 0:
                test_metrics = totals_model.evaluate(test_totals, test_totals["went_over"])
                print_metrics("Totals (Test)", test_metrics)

                # Additional evaluation: Brier score & high-confidence ROI buckets
                proba = totals_model.predict_proba(test_totals)[:, 1]
                actual = test_totals["went_over"].values
                preds = totals_model.predict(test_totals)

                conf_for_bet = np.where(preds == 1, proba, 1.0 - proba)
                mask_high = conf_for_bet >= 0.60
                if mask_high.any():
                    high_actual = actual[mask_high]
                    high_preds = preds[mask_high]
                    high_n = len(high_actual)
                    high_correct = (high_preds == high_actual).sum()
                    profit = high_correct * (100.0 / 110.0) - (high_n - high_correct)
                    roi_high = profit / high_n
                    acc_high = high_correct / high_n
                    print(f"  High-conf (>=60%) Totals Test Acc: {acc_high:.1%} on {high_n} bets")
                    print(f"  High-conf (>=60%) Totals Test ROI: {roi_high:+.1%}")

            # Save model
            model_path = os.path.join(output_dir, "totals_model.joblib")
            totals_model.save(model_path)
            print(f"Totals model saved to {model_path}")

            # Register version with model tracker
            try:
                version_id = f"{MODEL_VERSION}-totals-{model_type}"
                tracker.register_version(
                    ModelVersion(
                        version=version_id,
                        model_type="totals",
                        algorithm=model_type,
                        trained_at=pd.Timestamp.utcnow().isoformat(),
                        train_samples=len(train_totals),
                        test_samples=len(test_totals),
                        features_count=len(totals_model.feature_columns),
                        feature_names=totals_model.feature_columns,
                        metrics={
                            "accuracy": float(test_metrics.accuracy) if 'test_metrics' in locals() else float(train_metrics.accuracy),
                            "log_loss": float(test_metrics.log_loss) if 'test_metrics' in locals() else float(train_metrics.log_loss),
                            "brier": float(test_metrics.brier) if 'test_metrics' in locals() else float(train_metrics.brier),
                            "roi": float(test_metrics.roi) if 'test_metrics' in locals() else float(train_metrics.roi),
                        },
                        file_path=os.path.basename(model_path),
                    ),
                    set_active=True,
                )
            except Exception as e:
                print(f"[WARN] Could not register totals model version: {e}")

            # Feature importance
            if hasattr(totals_model.model, "feature_importances_"):
                print("\nTop Features (Totals):")
                importance = pd.DataFrame({
                    "feature": totals_model.feature_columns,
                    "importance": totals_model.model.feature_importances_,
                }).sort_values("importance", ascending=False)
                print(importance.head(10).to_string(index=False))
        else:
            print(f"Insufficient totals data for training ({len(train_totals)} games)")

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)

    # ========== MONEYLINE MODEL ==========
    print("\n" + "="*60)
    print("Training MONEYLINE Model")
    print("="*60)

    # Derive home-win target on the train/test splits
    if "home_score" not in training_df.columns or "away_score" not in training_df.columns:
        print("Warning: scores not found. Skipping moneyline model.")
        return

    # Ensure splits have the target column
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["home_win"] = (train_df["home_score"] > train_df["away_score"]).astype(int)
    test_df["home_win"] = (test_df["home_score"] > test_df["away_score"]).astype(int)

    moneyline_features = get_moneyline_features()
    available_moneyline = filter_available_features(moneyline_features, train_df.columns.tolist())

    moneyline_train = train_df.dropna(subset=["home_win"]).copy()
    moneyline_test = test_df.dropna(subset=["home_win"]).copy()

    if len(moneyline_train) > 10:
        moneyline_model = MoneylineModel(
            name="moneyline_classifier",
            model_type=model_type,
            feature_columns=available_moneyline,
        )

        moneyline_model.fit(moneyline_train, moneyline_train["home_win"])

        train_metrics = moneyline_model.evaluate(moneyline_train, moneyline_train["home_win"])
        print_metrics("Moneyline (Train)", train_metrics)

        if len(moneyline_test) > 0:
            test_metrics = moneyline_model.evaluate(moneyline_test, moneyline_test["home_win"])
            print_metrics("Moneyline (Test)", test_metrics)

        model_path = os.path.join(output_dir, "moneyline_model.joblib")
        moneyline_model.save(model_path)
        print(f"Moneyline model saved to {model_path}")

        # Register moneyline model version
        try:
            version_id = f"{MODEL_VERSION}-moneyline-{model_type}"
            tracker.register_version(
                ModelVersion(
                    version=version_id,
                    model_type="moneyline",
                    algorithm=model_type,
                    trained_at=pd.Timestamp.utcnow().isoformat(),
                    train_samples=len(moneyline_train),
                    test_samples=len(moneyline_test),
                    features_count=len(moneyline_model.feature_columns),
                    feature_names=moneyline_model.feature_columns,
                    metrics={
                        "accuracy": float(test_metrics.accuracy) if 'test_metrics' in locals() else float(train_metrics.accuracy),
                        "log_loss": float(test_metrics.log_loss) if 'test_metrics' in locals() else float(train_metrics.log_loss),
                        "brier": float(test_metrics.brier) if 'test_metrics' in locals() else float(train_metrics.brier),
                        "roi": float(test_metrics.roi) if 'test_metrics' in locals() else float(train_metrics.roi),
                    },
                    file_path=os.path.basename(model_path),
                ),
                set_active=True,
            )
        except Exception as e:
            print(f"[WARN] Could not register moneyline model version: {e}")
    else:
        print(f"Insufficient moneyline data for training ({len(moneyline_train)} games)")

    # ========== FIRST-HALF MODELS (optional / experimental) ==========
    print("\n" + "="*60)
    print("FIRST-HALF Models (optional / experimental)")
    print("="*60)

    # Detect common first-half score column names
    fh_home_cols = [c for c in train_df.columns if c.lower() in ("home_halftime_score", "home_ht_score", "home_score_1h", "home_first_half_score")]
    fh_away_cols = [c for c in train_df.columns if c.lower() in ("away_halftime_score", "away_ht_score", "away_score_1h", "away_first_half_score")]

    if fh_home_cols and fh_away_cols and False:
        # Use the first matching column names
        fh_home = fh_home_cols[0]
        fh_away = fh_away_cols[0]

        # Add first-half targets to the splits
        train_df = train_df.copy()
        test_df = test_df.copy()
        train_df["fh_home_score"] = train_df[fh_home]
        train_df["fh_away_score"] = train_df[fh_away]
        train_df["fh_home_win"] = (train_df["fh_home_score"] > train_df["fh_away_score"]).astype(int)

        test_df["fh_home_score"] = test_df[fh_home]
        test_df["fh_away_score"] = test_df[fh_away]
        test_df["fh_home_win"] = (test_df["fh_home_score"] > test_df["fh_away_score"]).astype(int)

        # First-half spreads
        try:
            from src.modeling.models import FirstHalfSpreadsModel, FirstHalfTotalsModel, FirstHalfMoneylineModel, TeamTotalsModel

            fh_spreads_features = spreads_features
            fh_spreads_available = [f for f in fh_spreads_features if f in train_df.columns]
            fh_train_spreads = train_df.dropna(subset=["fh_home_score"]).copy()
            fh_test_spreads = test_df.dropna(subset=["fh_home_score"]).copy()

            if len(fh_train_spreads) > 10:
                fh_spreads_model = FirstHalfSpreadsModel(name="fh_spreads", model_type=model_type, feature_columns=fh_spreads_available)
                fh_spreads_model.fit(fh_train_spreads, fh_train_spreads["fh_home_win"])
                print("Trained first-half spreads model")
                fh_spreads_path = os.path.join(output_dir, "fh_spreads_model.joblib")
                fh_spreads_model.save(fh_spreads_path)
                print(f"First-half spreads model saved to {fh_spreads_path}")
            else:
                print("Insufficient first-half spread data for training")

            # First-half totals (if halftime total line exists)
            fh_totals_train = fh_train_spreads
            if len(fh_totals_train) > 10:
                fh_totals_model = FirstHalfTotalsModel(name="fh_totals", model_type=model_type, feature_columns=available_totals)
                # Derive fh went_over if possible (requires fh total line column)
                if "fh_total_line" in fh_totals_train.columns:
                    fh_totals_train["fh_went_over"] = (fh_totals_train["fh_home_score"] + fh_totals_train["fh_away_score"] > fh_totals_train["fh_total_line"]).astype(int)
                    fh_test_totals = fh_test_spreads.copy()
                    if "fh_total_line" in fh_test_totals.columns:
                        fh_test_totals["fh_went_over"] = (fh_test_totals["fh_home_score"] + fh_test_totals["fh_away_score"] > fh_test_totals["fh_total_line"]).astype(int)

                    fh_totals_model.fit(fh_totals_train, fh_totals_train["fh_went_over"])
                    fh_totals_path = os.path.join(output_dir, "fh_totals_model.joblib")
                    fh_totals_model.save(fh_totals_path)
                    print(f"First-half totals model saved to {fh_totals_path}")
                else:
                    print("No first-half total line available; skipping first-half totals model")

        except Exception as e:
            print(f"First-half model training skipped or failed: {e}")
    else:
        print("No first-half score columns detected; skipping first-half models.")

    # ========== TEAM TOTALS MODELS (experimental, currently disabled) ==========
    print("\nTeam totals models are currently experimental and disabled by default.")


def train_first_half_models(
    model_type: str = "gradient_boosting",
    test_size: float = 0.2,
    output_dir: Optional[str] = None,
) -> None:
    """Train first half spread and total models.
    
    Args:
        model_type: Type of classifier to use ('logistic' or 'gradient_boosting')
        test_size: Proportion of data for testing
        output_dir: Directory to save models
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, brier_score_loss
    import joblib

    output_dir = output_dir or os.path.join(settings.data_processed_dir, "models")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training FIRST HALF Models (model_type={model_type})")
    print(f"{'='*60}")

    # Load 1H training data
    fh_path = os.path.join(settings.data_processed_dir, "first_half_training_data.csv")
    if not os.path.exists(fh_path):
        print(f"[WARN] First-half training data not found at {fh_path}")
        print("Run: python scripts/generate_first_half_training_data_fast.py")
        return

    df = pd.read_csv(fh_path)
    df['date'] = pd.to_datetime(df['date'])
    print(f"Loaded {len(df)} games with first-half data")

    # Identify feature columns
    exclude_cols = [
        'game_id', 'date', 'home_team', 'away_team',
        '1h_home_score', '1h_away_score', '1h_spread', '1h_total',
        'fg_home_score', 'fg_away_score', 'fg_spread', 'fg_total',
        '1h_spread_line', '1h_total_line',
        '1h_spread_covered', '1h_total_over',
    ]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].fillna(0)

    # Select classifier based on model_type
    def get_classifier():
        if model_type == "logistic":
            return LogisticRegression(max_iter=1000, random_state=42)
        else:  # gradient_boosting (default)
            return GradientBoostingClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42
            )

    # Train 1H Spread Model
    if '1h_spread_covered' in df.columns:
        y_spread = df['1h_spread_covered']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_spread, test_size=test_size, random_state=42, shuffle=False
        )

        print(f"\n1H Spread: Train={len(X_train)}, Test={len(X_test)}")

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', get_classifier())
        ])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        brier = brier_score_loss(y_test, y_proba)
        print(f"  1H Spread - Accuracy: {acc:.1%}, Brier: {brier:.4f}")

        model_path = os.path.join(output_dir, "first_half_spread_model.pkl")
        joblib.dump({'pipeline': pipeline, 'features': feature_cols}, model_path)
        print(f"  Saved: {model_path}")

    # Train 1H Total Model
    if '1h_total_over' in df.columns:
        y_total = df['1h_total_over']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_total, test_size=test_size, random_state=42, shuffle=False
        )

        print(f"\n1H Total: Train={len(X_train)}, Test={len(X_test)}")

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', get_classifier())
        ])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        brier = brier_score_loss(y_test, y_proba)
        print(f"  1H Total - Accuracy: {acc:.1%}, Brier: {brier:.4f}")

        model_path = os.path.join(output_dir, "first_half_total_model.pkl")
        joblib.dump({'pipeline': pipeline, 'features': feature_cols}, model_path)
        print(f"  Saved: {model_path}")

    print("\nFirst Half model training complete!")


def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    logger.info("Random seeds set to 42 for reproducibility")

    parser = argparse.ArgumentParser(description="Train NBA prediction models")
    parser.add_argument(
        "--model-type",
        choices=["logistic", "gradient_boosting", "regression"],
        default="logistic",
        help="Type of model to train",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data to use for testing",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save trained models",
    )
    parser.add_argument(
        "--market",
        choices=["fg", "1h", "all"],
        default="fg",
        help="Which markets to train: fg (full game), 1h (first half), or all",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Train ensemble models (logistic + gradient boosting)",
    )
    args = parser.parse_args()

    # Handle ensemble mode
    model_type = args.model_type
    if args.ensemble:
        print("\n[INFO] Ensemble mode: Training both logistic and gradient boosting models")
        # Train logistic first, then gradient boosting
        if args.market in ["fg", "all"]:
            print("\n>>> Training Full Game models (logistic)...")
            train_models(model_type="logistic", test_size=args.test_size, output_dir=args.output_dir)
            print("\n>>> Training Full Game models (gradient boosting)...")
            train_models(model_type="gradient_boosting", test_size=args.test_size, output_dir=args.output_dir)
        if args.market in ["1h", "all"]:
            print("\n>>> Training First Half models...")
            train_first_half_models(test_size=args.test_size, output_dir=args.output_dir)
    else:
        # Normal mode
        if args.market in ["fg", "all"]:
            train_models(
                model_type=model_type,
                test_size=args.test_size,
                output_dir=args.output_dir,
            )
        if args.market in ["1h", "all"]:
            train_first_half_models(
                model_type=model_type,
                test_size=args.test_size,
                output_dir=args.output_dir,
            )


if __name__ == "__main__":
    main()
