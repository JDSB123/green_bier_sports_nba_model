#!/usr/bin/env python3
"""
Train all 9 independent NBA betting models.

NBA v6.0: 9 Models Architecture
===============================
Q1:  Spread, Total, Moneyline
1H:  Spread, Total, Moneyline
FG:  Spread, Total, Moneyline

Each model is trained INDEPENDENTLY on period-specific features computed
from historical data for that period. No cross-period dependencies.

Usage:
    python scripts/train_all_models.py                    # Train all 9 models
    python scripts/train_all_models.py --periods fg,1h    # Train specific periods
    python scripts/train_all_models.py --markets spread   # Train specific market
    python scripts/train_all_models.py --model-type gradient_boosting
"""
import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import joblib

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.features import FeatureEngineer
from src.modeling.period_features import MODEL_CONFIGS, PERIOD_SCALING
from src.modeling.models import (
    SpreadsModel,
    TotalsModel,
    MoneylineModel,
    FirstHalfSpreadsModel,
    FirstHalfTotalsModel,
    FirstHalfMoneylineModel,
    FirstQuarterSpreadsModel,
    FirstQuarterTotalsModel,
    FirstQuarterMoneylineModel,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROCESSED_DIR / "models"


# Model class mapping
MODEL_CLASSES = {
    "q1_spread": FirstQuarterSpreadsModel,
    "q1_total": FirstQuarterTotalsModel,
    "q1_moneyline": FirstQuarterMoneylineModel,
    "1h_spread": FirstHalfSpreadsModel,
    "1h_total": FirstHalfTotalsModel,
    "1h_moneyline": FirstHalfMoneylineModel,
    "fg_spread": SpreadsModel,
    "fg_total": TotalsModel,
    "fg_moneyline": MoneylineModel,
}


def load_training_data(data_file: Optional[str] = None) -> pd.DataFrame:
    """Load and prepare training data with all period labels."""
    if data_file:
        training_path = Path(data_file) if Path(data_file).is_absolute() else PROCESSED_DIR / data_file
    else:
        training_path = PROCESSED_DIR / "training_data.csv"

    if not training_path.exists():
        raise FileNotFoundError(f"Training data not found: {training_path}")

    logger.info(f"Loading training data from {training_path}")
    df = pd.read_csv(training_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Verify required columns
    required = ["date", "home_team", "away_team", "home_score", "away_score"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Create FG labels
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)

    if "spread_line" in df.columns:
        df["actual_margin"] = df["home_score"] - df["away_score"]
        df["spread_covered"] = (df["actual_margin"] > -df["spread_line"]).astype(int)
        df.loc[df["spread_line"].isna(), "spread_covered"] = None

    if "total_line" in df.columns:
        df["actual_total"] = df["home_score"] + df["away_score"]
        df["total_over"] = (df["actual_total"] > df["total_line"]).astype(int)
        df.loc[df["total_line"].isna(), "total_over"] = None

    # Create 1H labels from quarter data
    quarter_cols = ["home_q1", "home_q2", "away_q1", "away_q2"]
    if all(col in df.columns for col in quarter_cols):
        df["home_1h_score"] = df["home_q1"].fillna(0) + df["home_q2"].fillna(0)
        df["away_1h_score"] = df["away_q1"].fillna(0) + df["away_q2"].fillna(0)
        df["home_1h_win"] = (df["home_1h_score"] > df["away_1h_score"]).astype(int)
        df["actual_1h_margin"] = df["home_1h_score"] - df["away_1h_score"]
        df["actual_1h_total"] = df["home_1h_score"] + df["away_1h_score"]

        # 1H spread covered
        if "1h_spread_line" in df.columns:
            df["1h_spread_covered"] = (df["actual_1h_margin"] > -df["1h_spread_line"]).astype(int)
            df.loc[df["1h_spread_line"].isna(), "1h_spread_covered"] = None
        elif "spread_line" in df.columns:
            # Derive from FG line if 1H line not available
            df["1h_spread_line"] = df["spread_line"] / 2
            df["1h_spread_covered"] = (df["actual_1h_margin"] > -df["1h_spread_line"]).astype(int)

        # 1H total over
        if "1h_total_line" in df.columns:
            df["1h_total_over"] = (df["actual_1h_total"] > df["1h_total_line"]).astype(int)
            df.loc[df["1h_total_line"].isna(), "1h_total_over"] = None
        elif "total_line" in df.columns:
            df["1h_total_line"] = df["total_line"] / 2
            df["1h_total_over"] = (df["actual_1h_total"] > df["1h_total_line"]).astype(int)

        logger.info("Created 1H labels from quarter data")

    # Create Q1 labels
    if "home_q1" in df.columns and "away_q1" in df.columns:
        df["home_q1_win"] = (df["home_q1"].fillna(0) > df["away_q1"].fillna(0)).astype(int)
        df["actual_q1_margin"] = df["home_q1"].fillna(0) - df["away_q1"].fillna(0)
        df["actual_q1_total"] = df["home_q1"].fillna(0) + df["away_q1"].fillna(0)

        # Q1 spread covered
        if "q1_spread_line" in df.columns:
            df["q1_spread_covered"] = (df["actual_q1_margin"] > -df["q1_spread_line"]).astype(int)
            df.loc[df["q1_spread_line"].isna(), "q1_spread_covered"] = None
        elif "spread_line" in df.columns:
            df["q1_spread_line"] = df["spread_line"] / 4
            df["q1_spread_covered"] = (df["actual_q1_margin"] > -df["q1_spread_line"]).astype(int)

        # Q1 total over
        if "q1_total_line" in df.columns:
            df["q1_total_over"] = (df["actual_q1_total"] > df["q1_total_line"]).astype(int)
            df.loc[df["q1_total_line"].isna(), "q1_total_over"] = None
        elif "total_line" in df.columns:
            df["q1_total_line"] = df["total_line"] / 4
            df["q1_total_over"] = (df["actual_q1_total"] > df["q1_total_line"]).astype(int)

        logger.info("Created Q1 labels from quarter data")

    logger.info(f"Loaded {len(df)} games from {df['date'].min().date()} to {df['date'].max().date()}")
    return df


def build_period_features(
    df: pd.DataFrame,
    model_key: str,
    min_history: int = 30,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build features for a specific model using period-specific historical data.

    Args:
        df: Full training DataFrame
        model_key: Model key (e.g., "q1_spread", "1h_total", "fg_moneyline")
        min_history: Minimum games before first prediction

    Returns:
        Tuple of (features_df, labels)
    """
    config = MODEL_CONFIGS[model_key]
    period = config["period"]
    label_col = config["label_col"]
    line_col = config["line_col"]

    # Filter to games with valid labels
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in data")

    valid_df = df[df[label_col].notna()].copy()
    logger.info(f"[{model_key}] {len(valid_df)} games with valid labels")

    if len(valid_df) < min_history + 50:
        raise ValueError(f"Insufficient data: {len(valid_df)} games (need {min_history + 50}+)")

    fe = FeatureEngineer(lookback=10)
    all_features = []
    labels = []

    for i in range(min_history, len(valid_df)):
        game = valid_df.iloc[i]
        historical = valid_df.iloc[:i]

        # Build period-specific features
        try:
            period_features = fe.compute_period_rolling_stats(
                historical, game["home_team"], pd.to_datetime(game["date"]), period
            )
            opp_period_features = fe.compute_period_rolling_stats(
                historical, game["away_team"], pd.to_datetime(game["date"]), period
            )

            if not period_features or not opp_period_features:
                continue

            # Build feature dict
            features = {}
            for key, val in period_features.items():
                features[f"home_{key}"] = val
            for key, val in opp_period_features.items():
                features[f"away_{key}"] = val

            # Add differentials
            suffix = f"_{period}" if period != "fg" else ""
            ppg_key = f"ppg{suffix}"
            margin_key = f"margin{suffix}"

            if ppg_key in period_features and ppg_key in opp_period_features:
                features[f"ppg_diff{suffix}"] = period_features[ppg_key] - opp_period_features[ppg_key]
            if margin_key in period_features and margin_key in opp_period_features:
                features[f"margin_diff{suffix}"] = period_features[margin_key] - opp_period_features[margin_key]

            # Add context features (rest, HCA)
            rest_features = build_context_features(game, historical, period, fe)
            features.update(rest_features)

            # Add line feature if available
            if line_col and line_col in game.index and pd.notna(game[line_col]):
                features[line_col] = game[line_col]

            all_features.append(features)
            labels.append(game[label_col])

        except Exception as e:
            logger.debug(f"Error processing game {i}: {e}")
            continue

    if len(all_features) < 100:
        raise ValueError(f"Only {len(all_features)} valid samples (need 100+)")

    features_df = pd.DataFrame(all_features)
    labels_series = pd.Series(labels, name=label_col).astype(int)

    logger.info(f"[{model_key}] Built {len(features_df)} training samples with {len(features_df.columns)} features")
    return features_df, labels_series


def build_context_features(
    game: pd.Series,
    historical: pd.DataFrame,
    period: str,
    fe: FeatureEngineer,
) -> Dict[str, float]:
    """Build context features (rest, HCA, travel) scaled for period."""
    from src.modeling.team_factors import get_home_court_advantage

    features = {}
    game_date = pd.to_datetime(game["date"])
    scaling = PERIOD_SCALING.get(period, PERIOD_SCALING["fg"])

    # Rest days
    try:
        home_rest = fe.compute_rest_days(historical, game["home_team"], game_date, default_rest=3)
        away_rest = fe.compute_rest_days(historical, game["away_team"], game_date, default_rest=3)
    except Exception:
        home_rest = 3
        away_rest = 3

    features["home_rest_days"] = home_rest
    features["away_rest_days"] = away_rest
    features["rest_diff"] = home_rest - away_rest
    features["home_b2b"] = 1 if home_rest <= 1 else 0
    features["away_b2b"] = 1 if away_rest <= 1 else 0

    # Scaled rest adjustment
    base_rest_adj = (home_rest - away_rest) * 0.5
    suffix = f"_{period}" if period != "fg" else ""
    features[f"rest_adj{suffix}"] = base_rest_adj * scaling["rest_factor"]

    # Dynamic HCA scaled for period
    base_hca = get_home_court_advantage(game["home_team"])
    features[f"dynamic_hca{suffix}"] = base_hca * scaling["hca_factor"]

    # Travel features (scaled)
    try:
        travel = fe.compute_travel_features(
            historical, game["away_team"], game["home_team"],
            game_date, is_home=False
        )
        features["away_travel_distance"] = travel["travel_distance"]
        features[f"travel_fatigue{suffix}"] = travel["travel_fatigue"] * scaling["travel_factor"]
    except Exception:
        features["away_travel_distance"] = 0
        features[f"travel_fatigue{suffix}"] = 0

    return features


def train_model(
    model_key: str,
    features_df: pd.DataFrame,
    labels: pd.Series,
    model_type: str = "logistic",
    use_calibration: bool = True,
) -> Tuple[object, List[str], Dict]:
    """
    Train a single model.

    Returns:
        Tuple of (model, feature_columns, metrics)
    """
    ModelClass = MODEL_CLASSES[model_key]

    # Filter to available features
    available_features = [col for col in features_df.columns if features_df[col].notna().sum() > len(features_df) * 0.5]
    X = features_df[available_features].copy()
    X = X.fillna(X.median())

    # Initialize model
    model = ModelClass(
        name=model_key,
        model_type=model_type,
        feature_columns=available_features,
        use_calibration=use_calibration,
    )

    # Train
    logger.info(f"[{model_key}] Training {model_type} model on {len(X)} samples, {len(available_features)} features")
    model.fit(X, labels)

    # Evaluate on training data (for logging)
    metrics = model.evaluate(X, labels)
    logger.info(
        f"[{model_key}] Training metrics: "
        f"Accuracy={metrics.accuracy:.1%}, ROI={metrics.roi:+.1%}, Brier={metrics.brier:.4f}"
    )

    return model, model.feature_columns, {
        "accuracy": metrics.accuracy,
        "roi": metrics.roi,
        "brier": metrics.brier,
        "n_samples": len(X),
        "n_features": len(model.feature_columns),
    }


def save_model(
    model: object,
    feature_columns: List[str],
    model_key: str,
    metrics: Dict,
) -> Path:
    """Save model and metadata to disk."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    config = MODEL_CONFIGS[model_key]
    model_path = MODELS_DIR / config["model_file"]

    payload = {
        "pipeline": model.pipeline,
        "model": model.model,
        "feature_columns": feature_columns,
        "name": model_key,
        "meta": {
            "model_type": model.model_type,
            "period": config["period"],
            "market": config["market"],
            "label_col": config["label_col"],
            "trained_at": datetime.now().isoformat(),
            "metrics": metrics,
        },
    }

    joblib.dump(payload, model_path)
    logger.info(f"[{model_key}] Saved to {model_path}")
    return model_path


def main():
    parser = argparse.ArgumentParser(description="Train all 9 NBA betting models")
    parser.add_argument(
        "--periods",
        type=str,
        default="all",
        help="Comma-separated periods to train (q1,1h,fg) or 'all'",
    )
    parser.add_argument(
        "--markets",
        type=str,
        default="all",
        help="Comma-separated markets to train (spread,total,moneyline) or 'all'",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="logistic",
        choices=["logistic", "gradient_boosting"],
        help="Model type to use",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Training data file",
    )
    parser.add_argument(
        "--min-history",
        type=int,
        default=50,
        help="Minimum games before first training sample",
    )
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Disable probability calibration",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("NBA v6.0 - TRAIN ALL 9 MODELS")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Determine which models to train
    if args.periods == "all":
        periods = ["q1", "1h", "fg"]
    else:
        periods = [p.strip() for p in args.periods.split(",")]

    if args.markets == "all":
        markets = ["spread", "total", "moneyline"]
    else:
        markets = [m.strip() for m in args.markets.split(",")]

    models_to_train = [
        f"{period}_{market}"
        for period in periods
        for market in markets
        if f"{period}_{market}" in MODEL_CONFIGS
    ]

    print(f"Models to train: {', '.join(models_to_train)}")
    print(f"Model type: {args.model_type}")
    print(f"Calibration: {'disabled' if args.no_calibration else 'enabled'}")

    # Load data
    try:
        df = load_training_data(args.data)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)

    # Train each model
    results = {}
    for model_key in models_to_train:
        print(f"\n{'='*60}")
        print(f"TRAINING: {model_key.upper()}")
        print(f"{'='*60}")

        try:
            # Build features
            features_df, labels = build_period_features(
                df, model_key, min_history=args.min_history
            )

            # Train model
            model, feature_columns, metrics = train_model(
                model_key,
                features_df,
                labels,
                model_type=args.model_type,
                use_calibration=not args.no_calibration,
            )

            # Save model
            model_path = save_model(model, feature_columns, model_key, metrics)

            results[model_key] = {
                "status": "SUCCESS",
                "path": str(model_path),
                "metrics": metrics,
            }

        except Exception as e:
            logger.error(f"[{model_key}] Training failed: {e}")
            import traceback
            traceback.print_exc()
            results[model_key] = {
                "status": "FAILED",
                "error": str(e),
            }

    # Summary
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")

    print("\nResults Summary:")
    print("-" * 40)
    for model_key, result in results.items():
        status = result["status"]
        if status == "SUCCESS":
            metrics = result["metrics"]
            print(
                f"  {model_key:15s} : {status} | "
                f"Acc={metrics['accuracy']:.1%}, ROI={metrics['roi']:+.1%}"
            )
        else:
            print(f"  {model_key:15s} : {status} | {result.get('error', 'Unknown error')}")

    # Count successes
    successes = sum(1 for r in results.values() if r["status"] == "SUCCESS")
    print(f"\nSuccessfully trained: {successes}/{len(models_to_train)} models")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Exit with error if any failures
    if successes < len(models_to_train):
        sys.exit(1)


if __name__ == "__main__":
    main()
