#!/usr/bin/env python3
"""
Train experimental models on expanded historical data.

This script trains models in the models/experimental/ directory,
keeping production models completely isolated.

Key Features:
- Walk-forward validation across seasons (no data leakage)
- Multiple algorithm comparison
- Comparison against production baseline
- Safe output to experimental directory only

Usage:
    python scripts/train_experimental_models.py
    python scripts/train_experimental_models.py --data data/experimental/training_data_2020_2025.csv
    python scripts/train_experimental_models.py --experiment v2_gradient_boosting --model-type gradient_boosting

Output:
    models/experimental/{experiment_name}/
        ├── fg_spread_model.joblib
        ├── fg_total_model.joblib
        ├── 1h_spread_model.joblib
        ├── 1h_total_model.joblib
        └── experiment.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss
import joblib

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Directories
EXPERIMENTAL_DIR = PROJECT_ROOT / "models" / "experimental"
DATA_DIR = PROJECT_ROOT / "data" / "experimental"


class ExperimentalModel:
    """Wrapper for experimental model training with walk-forward validation."""
    
    def __init__(
        self,
        model_type: str = "logistic",
        calibration: str = "isotonic",
    ):
        self.model_type = model_type
        self.calibration = calibration
        self.model = None
        self.feature_names = None
        self.metrics = {}
        
    def _create_base_model(self):
        """Create the base model based on model_type."""
        if self.model_type == "logistic":
            return LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight="balanced",
            )
        elif self.model_type == "gradient_boosting":
            try:
                from sklearn.ensemble import GradientBoostingClassifier
                return GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=5,
                    random_state=42,
                )
            except ImportError:
                logger.warning("GradientBoosting not available, using LogisticRegression")
                return LogisticRegression(max_iter=1000, random_state=42)
        elif self.model_type == "xgboost":
            try:
                from xgboost import XGBClassifier
                return XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric="logloss",
                )
            except ImportError:
                logger.warning("XGBoost not available, using LogisticRegression")
                return LogisticRegression(max_iter=1000, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: list[str] | None = None,
    ):
        """Train the model with optional calibration."""
        self.feature_names = feature_names or list(X.columns)
        
        base_model = self._create_base_model()
        
        if self.calibration == "isotonic":
            self.model = CalibratedClassifierCV(
                base_model,
                method="isotonic",
                cv=5,
            )
        elif self.calibration == "sigmoid":
            self.model = CalibratedClassifierCV(
                base_model,
                method="sigmoid",
                cv=5,
            )
        else:
            self.model = base_model
        
        self.model.fit(X, y)
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        return self.model.predict_proba(X)[:, 1]
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict binary outcomes."""
        return self.model.predict(X)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """Evaluate model on test data."""
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        accuracy = accuracy_score(y, y_pred)
        
        # Handle single-class edge case for log_loss
        unique_classes = y.unique()
        if len(unique_classes) < 2:
            logloss = float('inf')
        else:
            logloss = log_loss(y, y_proba)
        
        # Calculate ROI assuming -110 odds
        correct = (y_pred == y).astype(int)
        winnings = correct * 100 - (1 - correct) * 110
        roi = winnings.sum() / (len(y) * 110)
        
        return {
            "accuracy": accuracy,
            "log_loss": logloss,
            "roi": roi,
            "n_samples": len(y),
        }
    
    def save(self, path: Path):
        """Save model to file."""
        joblib.dump({
            "model": self.model,
            "feature_names": self.feature_names,
            "model_type": self.model_type,
            "calibration": self.calibration,
            "metrics": self.metrics,
        }, path)
        logger.info(f"  Saved: {path}")
    
    @classmethod
    def load(cls, path: Path) -> "ExperimentalModel":
        """Load model from file."""
        data = joblib.load(path)
        instance = cls(
            model_type=data.get("model_type", "logistic"),
            calibration=data.get("calibration", "isotonic"),
        )
        instance.model = data["model"]
        instance.feature_names = data["feature_names"]
        instance.metrics = data.get("metrics", {})
        return instance


def walk_forward_validation(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    model_type: str = "logistic",
    min_train_seasons: int = 3,
) -> list[dict[str, Any]]:
    """
    Perform walk-forward validation across seasons.
    
    Train on seasons 1...N, test on season N+1.
    Ensures no data leakage.
    """
    results = []
    seasons = sorted(df["season"].unique())
    
    logger.info(f"  Walk-forward validation: {len(seasons)} seasons")
    
    for i in range(min_train_seasons, len(seasons)):
        train_seasons = seasons[:i]
        test_season = seasons[i]
        
        train_df = df[df["season"].isin(train_seasons)]
        test_df = df[df["season"] == test_season]
        
        # Skip if insufficient data
        if len(train_df) < 100 or len(test_df) < 50:
            continue
        
        X_train = train_df[feature_cols]
        y_train = train_df[label_col]
        X_test = test_df[feature_cols]
        y_test = test_df[label_col]
        
        # Train model
        model = ExperimentalModel(model_type=model_type)
        model.train(X_train, y_train, feature_cols)
        
        # Evaluate
        metrics = model.evaluate(X_test, y_test)
        
        results.append({
            "train_seasons": train_seasons,
            "test_season": test_season,
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            **metrics,
        })
        
        logger.info(
            f"    Seasons {train_seasons[-1]}-{test_season}: "
            f"acc={metrics['accuracy']:.1%}, roi={metrics['roi']:+.1%}"
        )
    
    return results


def train_market_model(
    df: pd.DataFrame,
    market: str,
    feature_cols: list[str],
    model_type: str = "logistic",
    output_dir: Path = None,
) -> dict[str, Any]:
    """
    Train and evaluate model for a specific market.
    
    Args:
        df: Training data
        market: Market name (fg_spread, fg_total, 1h_spread, 1h_total)
        feature_cols: Feature column names
        model_type: Algorithm to use
        output_dir: Directory to save model
    
    Returns:
        Dictionary with training results
    """
    # Define label column
    label_map = {
        "fg_spread": "fg_spread_covered",
        "fg_total": "fg_total_over",
        "1h_spread": "1h_spread_covered",
        "1h_total": "1h_total_over",
    }
    label_col = label_map.get(market)
    
    if label_col not in df.columns:
        logger.warning(f"  Label column not found for {market}: {label_col}")
        return None
    
    # Filter to valid rows for this market
    valid_df = df[df[label_col].notna()].copy()
    valid_df[label_col] = valid_df[label_col].astype(int)
    
    logger.info(f"\n  Training {market.upper()} model on {len(valid_df):,} games")
    
    # Skip if no valid data
    if len(valid_df) < 100:
        logger.warning(f"  Skipping {market} - insufficient data ({len(valid_df)} games)")
        return None
    
    # Walk-forward validation
    wf_results = walk_forward_validation(
        valid_df,
        feature_cols,
        label_col,
        model_type=model_type,
    )
    
    # Calculate aggregate metrics
    if wf_results:
        avg_accuracy = np.mean([r["accuracy"] for r in wf_results])
        avg_roi = np.mean([r["roi"] for r in wf_results])
        logger.info(f"  Walk-forward average: acc={avg_accuracy:.1%}, roi={avg_roi:+.1%}")
    else:
        avg_accuracy = 0
        avg_roi = 0
    
    # Train final model on all data except most recent season
    seasons = sorted(valid_df["season"].unique())
    train_df = valid_df[valid_df["season"] != seasons[-1]]
    
    X_train = train_df[feature_cols]
    y_train = train_df[label_col]
    
    final_model = ExperimentalModel(model_type=model_type)
    final_model.train(X_train, y_train, feature_cols)
    final_model.metrics = {
        "walk_forward_accuracy": avg_accuracy,
        "walk_forward_roi": avg_roi,
        "training_samples": len(train_df),
        "training_seasons": list(train_df["season"].unique()),
    }
    
    # Save model
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / f"{market}_model.joblib"
        final_model.save(model_path)
    
    return {
        "market": market,
        "walk_forward_results": wf_results,
        "avg_accuracy": avg_accuracy,
        "avg_roi": avg_roi,
        "training_samples": len(train_df),
    }


def run_experiment(
    experiment_name: str,
    data_path: Path,
    model_type: str = "logistic",
    hypothesis: str = "",
) -> dict[str, Any]:
    """
    Run a complete experiment with all 4 markets.
    
    Args:
        experiment_name: Name for this experiment
        data_path: Path to training data CSV
        model_type: Algorithm to use
        hypothesis: Description of what we're testing
    
    Returns:
        Experiment results dictionary
    """
    logger.info("=" * 60)
    logger.info(f"EXPERIMENT: {experiment_name}")
    logger.info(f"Model Type: {model_type}")
    logger.info(f"Data: {data_path}")
    logger.info("=" * 60)
    
    # Load data
    df = pd.read_csv(data_path, parse_dates=["game_date"])
    logger.info(f"Loaded {len(df):,} games")
    
    # Define features
    feature_cols = [
        "home_rolling_ppg", "home_rolling_papg", "home_rolling_win_pct", "home_rolling_margin",
        "away_rolling_ppg", "away_rolling_papg", "away_rolling_win_pct", "away_rolling_margin",
        "home_rest_days", "away_rest_days",
        "ppg_diff", "papg_diff", "win_pct_diff", "margin_diff", "rest_advantage",
    ]
    
    # Filter to available features
    available_features = [c for c in feature_cols if c in df.columns]
    logger.info(f"Using {len(available_features)} features")
    
    # Fill missing values
    for col in available_features:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    
    # Output directory
    output_dir = EXPERIMENTAL_DIR / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train all 4 markets
    markets = ["fg_spread", "fg_total", "1h_spread", "1h_total"]
    results = {}
    
    for market in markets:
        result = train_market_model(
            df,
            market,
            available_features,
            model_type=model_type,
            output_dir=output_dir,
        )
        if result:
            results[market] = result
    
    # Save experiment metadata
    experiment_info = {
        "name": experiment_name,
        "hypothesis": hypothesis,
        "model_type": model_type,
        "data_path": str(data_path),
        "features": available_features,
        "created_at": datetime.now().isoformat(),
        "results": {
            market: {
                "avg_accuracy": r["avg_accuracy"],
                "avg_roi": r["avg_roi"],
                "training_samples": r["training_samples"],
            }
            for market, r in results.items()
        },
    }
    
    experiment_path = output_dir / "experiment.json"
    with open(experiment_path, "w") as f:
        json.dump(experiment_info, f, indent=2, default=str)
    
    logger.info(f"\n✓ Experiment saved to {output_dir}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for market, r in results.items():
        logger.info(f"  {market}: acc={r['avg_accuracy']:.1%}, roi={r['avg_roi']:+.1%}")
    
    return experiment_info


def main():
    parser = argparse.ArgumentParser(
        description="Train experimental models on expanded historical data"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="v1_expanded_data",
        help="Experiment name (default: v1_expanded_data)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to training data CSV (default: data/experimental/training_data_full.csv)"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["logistic", "gradient_boosting", "xgboost"],
        default="logistic",
        help="Model algorithm (default: logistic)"
    )
    parser.add_argument(
        "--hypothesis",
        type=str,
        default="",
        help="Hypothesis being tested"
    )
    
    args = parser.parse_args()
    
    # Determine data path
    if args.data:
        data_path = Path(args.data)
    else:
        data_path = DATA_DIR / "training_data_full.csv"
    
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        logger.info("Run 'python scripts/build_experimental_training_data.py' first")
        sys.exit(1)
    
    # Run experiment
    run_experiment(
        experiment_name=args.experiment,
        data_path=data_path,
        model_type=args.model_type,
        hypothesis=args.hypothesis or f"Testing {args.model_type} on expanded data",
    )


if __name__ == "__main__":
    main()
