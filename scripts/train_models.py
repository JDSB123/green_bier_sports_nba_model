"""
Train NBA prediction models for 4 active markets (1H + FG spreads/totals).

Usage:
    python scripts/train_models.py                          # Train all 4 markets
    python scripts/train_models.py --market fg              # Train FG only (spread + total)
    python scripts/train_models.py --market 1h              # Train 1H only (spread + total)
    python scripts/train_models.py --model-type gradient_boosting

Markets (All INDEPENDENT models):
    1H (First Half):
        1h_spread    - First Half Spreads
        1h_total     - First Half Totals

    FG (Full Game):
        fg_spread    - Full Game Spreads
        fg_total     - Full Game Totals

ARCHITECTURE:
    Each period (1H, FG) uses INDEPENDENT features computed from
    historical data for that specific period. No cross-period dependencies.

Features include:
    - Team rolling stats (PPG, PAPG, margin, win%)
    - Rest days / back-to-back detection
    - Head-to-head history
    - ELO ratings
    - Injury impact estimation (when injury data available)
    - RLM (Reverse Line Movement) signals (when betting splits available)
    - Period-specific historical performance (1H margin, etc.)
"""
from __future__ import annotations
import argparse
import os
import sys
import random
import logging
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.config import settings
from src.modeling.models import (
    SpreadsModel, TotalsModel, ModelMetrics,
    FirstHalfSpreadsModel, FirstHalfTotalsModel,
)
from src.modeling.model_tracker import ModelTracker, ModelVersion
from src.modeling.feature_config import (
    get_spreads_features,
    get_totals_features,
    filter_available_features,
)
from src.modeling.period_features import MODEL_CONFIGS, get_model_features


# Model version for tracking
MODEL_VERSION = "33.0.8.0"

# Market configurations mapping
MARKET_CONFIG = {
    # Full Game Markets
    "fg_spread": {
        "name": "Full Game Spread",
        "model_class": SpreadsModel,
        "label_col": "spread_covered",
        "line_col": "spread_line",
        "period": "fg",
        "output_file": "fg_spread_model.joblib",
    },
    "fg_total": {
        "name": "Full Game Total",
        "model_class": TotalsModel,
        "label_col": "total_over",
        "line_col": "total_line",
        "period": "fg",
        "output_file": "fg_total_model.joblib",
    },
    # First Half Markets
    "1h_spread": {
        "name": "First Half Spread",
        "model_class": FirstHalfSpreadsModel,
        "label_col": "1h_spread_covered",
        "line_col": "1h_spread_line",
        "period": "1h",
        "output_file": "1h_spread_model.pkl",
    },
    "1h_total": {
        "name": "First Half Total",
        "model_class": FirstHalfTotalsModel,
        "label_col": "1h_total_over",
        "line_col": "1h_total_line",
        "period": "1h",
        "output_file": "1h_total_model.pkl",
    },
}


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

    # Create injury impact columns if not present (spread + total)
    injury_cols = [
        # Spread injury features
        "home_injury_spread_impact", "away_injury_spread_impact",
        "injury_spread_diff", "home_star_out", "away_star_out",
        # Total injury features
        "home_injury_total_impact", "away_injury_total_impact",
        "injury_total_diff",
    ]

    for col in injury_cols:
        if col not in df.columns:
            df[col] = 0.0

    print(f"  [OK] Added injury feature columns (spread + total)")
    return df


def enrich_with_rlm_features(df: pd.DataFrame, splits_df: pd.DataFrame) -> pd.DataFrame:
    """Add RLM and betting splits features to training data."""
    if splits_df is None or splits_df.empty:
        return df

    # RLM feature columns (spread + total)
    rlm_cols = [
        # Spread RLM features
        "is_rlm_spread", "sharp_side_spread",
        "spread_public_home_pct", "spread_ticket_money_diff",
        "spread_movement",
        # Total RLM features
        "is_rlm_total", "sharp_side_total",
    ]

    for col in rlm_cols:
        if col not in df.columns:
            df[col] = 0.0

    print(f"  [OK] Added RLM feature columns (spread + total)")
    return df


def train_single_market(
    market_key: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_type: str,
    output_dir: str,
    tracker: ModelTracker,
) -> Optional[ModelMetrics]:
    """
    Train a single market model (one of the 4 independent models).
    
    Args:
        market_key: One of 'fg_spread', 'fg_total', '1h_spread', '1h_total'.
        train_df: Training data DataFrame
        test_df: Test data DataFrame
        model_type: 'logistic' or 'gradient_boosting'
        output_dir: Directory to save model
        tracker: ModelTracker for version registration
        
    Returns:
        Test ModelMetrics if successful, None otherwise
    """
    import pickle
    import joblib
    
    config = MARKET_CONFIG.get(market_key)
    if not config:
        print(f"  [ERROR] Unknown market: {market_key}")
        return None
    
    print(f"\n{'='*60}")
    print(f"Training {config['name']} Model ({market_key})")
    print(f"{'='*60}")
    
    label_col = config["label_col"]
    line_col = config["line_col"]
    period = config["period"]
    model_class = config["model_class"]
    output_file = config["output_file"]
    
    # Check if label column exists
    if label_col not in train_df.columns:
        print(f"  [WARN] Label column '{label_col}' not found. Skipping.")
        return None
    
    # Get features for this market
    try:
        if period == "fg":
            if "spread" in market_key:
                features = get_spreads_features()
            else:
                features = get_totals_features()
        else:
            market_type = "spread" if "spread" in market_key else "total"
            try:
                features = get_model_features(period, market_type)
            except Exception:
                # Fallback to FG features
                if market_type == "spread":
                    features = get_spreads_features()
                else:
                    features = get_totals_features()
    except Exception as e:
        print(f"  [WARN] Could not get features: {e}")
        features = get_spreads_features() if "spread" in market_key else get_totals_features()

    # For 1H models, use lower threshold and fallback to FG features if needed
    min_pct = 0.3
    if period == "1h":
        min_pct = 0.15  # Lower threshold for period-specific models

    try:
        available_features = filter_available_features(features, train_df.columns.tolist(), min_required_pct=min_pct)
    except ValueError:
        # Fallback: use FG features for 1H models
        print(f"  [INFO] Using FG features for {market_key} (period features unavailable)")
        if "spread" in market_key:
            features = get_spreads_features()
        else:
            features = get_totals_features()
        available_features = filter_available_features(features, train_df.columns.tolist(), min_required_pct=0.3)

    print(f"  Using {len(available_features)} of {len(features)} possible features")
    
    if len(available_features) < 5:
        print(f"  [WARN] Too few features available ({len(available_features)}). Skipping.")
        return None
    
    # Filter data
    if line_col:
        train_market = train_df[train_df[line_col].notna() & train_df[label_col].notna()].copy()
        test_market = test_df[test_df[line_col].notna() & test_df[label_col].notna()].copy()
    else:
        train_market = train_df[train_df[label_col].notna()].copy()
        test_market = test_df[test_df[label_col].notna()].copy()
    
    print(f"  Train size: {len(train_market)}, Test size: {len(test_market)}")
    
    if len(train_market) < 20:
        print(f"  [WARN] Insufficient data ({len(train_market)} games). Skipping.")
        return None
    
    # Initialize and train model
    model = model_class(
        name=f"{market_key}_classifier",
        model_type=model_type,
        feature_columns=available_features,
    )
    
    model.fit(train_market, train_market[label_col])
    
    # Evaluate on training
    train_metrics = model.evaluate(train_market, train_market[label_col])
    print_metrics(f"{config['name']} (Train)", train_metrics)
    
    # Evaluate on test
    test_metrics = None
    if len(test_market) > 0:
        test_metrics = model.evaluate(test_market, test_market[label_col])
        print_metrics(f"{config['name']} (Test)", test_metrics)
        
        # High-confidence bucket analysis
        try:
            proba = model.predict_proba(test_market)[:, 1]
            actual = test_market[label_col].values
            preds = model.predict(test_market)
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
                print(f"  High-conf (>=60%): {acc_high:.1%} acc on {high_n} bets, ROI: {roi_high:+.1%}")
        except Exception as e:
            logger.debug(f"Could not compute high-conf stats: {e}")
    
    # Save model
    model_path = os.path.join(output_dir, output_file)
    
    if output_file.endswith(".pkl"):
        # Save as pkl with separate features file
        with open(model_path, "wb") as f:
            pickle.dump(model.pipeline, f)
        features_path = os.path.join(output_dir, output_file.replace("_model.pkl", "_features.pkl"))
        with open(features_path, "wb") as f:
            pickle.dump(model.feature_columns, f)
        print(f"  Saved: {model_path} + {features_path}")
    else:
        # Save as joblib
        model.save(model_path)
        print(f"  Saved: {model_path}")
    
    # Register version
    try:
        version_id = f"{MODEL_VERSION}-{market_key}-{model_type}"
        metrics_dict = {
            "accuracy": float(test_metrics.accuracy if test_metrics else train_metrics.accuracy),
            "log_loss": float(test_metrics.log_loss if test_metrics else train_metrics.log_loss),
            "brier": float(test_metrics.brier if test_metrics else train_metrics.brier),
            "roi": float(test_metrics.roi if test_metrics else train_metrics.roi),
        }
        tracker.register_version(
            ModelVersion(
                version=version_id,
                model_type=market_key,
                algorithm=model_type,
                trained_at=pd.Timestamp.utcnow().isoformat(),
                train_samples=len(train_market),
                test_samples=len(test_market),
                features_count=len(model.feature_columns),
                feature_names=model.feature_columns,
                metrics=metrics_dict,
                file_path=output_file,
            ),
            set_active=True,
        )
    except Exception as e:
        print(f"  [WARN] Could not register model version: {e}")
    
    return test_metrics if test_metrics else train_metrics


def train_all_markets(
    model_type: str = "logistic",
    test_size: float = 0.2,
    output_dir: Optional[str] = None,
    markets: Optional[List[str]] = None,
) -> Dict[str, ModelMetrics]:
    """
    Train all 4 independent market models.
    
    Args:
        model_type: 'logistic' or 'gradient_boosting'
        test_size: Proportion for test split
        output_dir: Directory to save models
        markets: List of market keys to train (default: all 9)
        
    Returns:
        Dictionary of market_key -> ModelMetrics
    """
    from typing import List, Dict
    
    output_dir = output_dir or os.path.join(settings.data_processed_dir, "models")
    os.makedirs(output_dir, exist_ok=True)
    tracker = ModelTracker()
    
    print(f"\n{'='*70}")
    print(f"NBA v6.0 - Training All 4 Independent Market Models")
    print(f"{'='*70}")
    print(f"Model Type: {model_type}")
    print(f"Output Dir: {output_dir}")
    
    # Determine which markets to train
    if markets is None:
        markets = list(MARKET_CONFIG.keys())
    
    print(f"Markets to train: {', '.join(markets)}")
    
    # Load training data
    print("\nLoading supplementary data...")
    injuries_df, splits_df = load_supplementary_data(settings.data_processed_dir)
    
    training_path = os.path.join(settings.data_processed_dir, "training_data.csv")
    fh_path = os.path.join(settings.data_processed_dir, "training_data_fh.csv")
    
    if os.path.exists(fh_path):
        print(f"Using first-half augmented training file: {fh_path}")
        training_path = fh_path
    
    if not os.path.exists(training_path):
        print("\nNo training data available!")
        print("Run: python scripts/build_fresh_training_data.py")
        return {}
    
    training_df = pd.read_csv(training_path)
    training_df["date"] = pd.to_datetime(training_df["date"], errors="coerce")
    
    if training_df.empty:
        print("Training data file is empty!")
        return {}
    
    print(f"Training dataset size: {len(training_df)} games")
    
    # Enrich with additional features
    print("\nEnriching training data...")
    training_df = enrich_with_injury_features(training_df, injuries_df)
    training_df = enrich_with_rlm_features(training_df, splits_df)
    
    # Compute derived labels if missing
    training_df = ensure_all_labels(training_df)
    
    # Temporal split
    training_df = training_df.sort_values("date")
    split_idx = int(len(training_df) * (1 - test_size))
    train_df = training_df.iloc[:split_idx]
    test_df = training_df.iloc[split_idx:]
    
    print(f"\nTrain size: {len(train_df)}, Test size: {len(test_df)}")
    
    # Train each market
    results = {}
    for market_key in markets:
        metrics = train_single_market(
            market_key=market_key,
            train_df=train_df,
            test_df=test_df,
            model_type=model_type,
            output_dir=output_dir,
            tracker=tracker,
        )
        if metrics:
            results[market_key] = metrics
    
    # Summary
    print(f"\n{'='*70}")
    print(f"Training Complete - {len(results)}/4 markets trained successfully")
    print(f"{'='*70}")
    
    for market_key, metrics in results.items():
        print(f"  {market_key}: {metrics.accuracy:.1%} acc, {metrics.roi:+.1%} ROI")
    
    return results


def ensure_all_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all 6 market labels exist in the DataFrame."""
    df = df.copy()
    
    # Full game labels
    if "spread_covered" not in df.columns and "spread_line" in df.columns:
        df["actual_margin"] = df["home_score"] - df["away_score"]
        df["spread_covered"] = df.apply(
            lambda r: int(r["actual_margin"] > -r["spread_line"]) 
            if pd.notna(r.get("spread_line")) else None,
            axis=1
        )
    
    if "total_over" not in df.columns and "total_line" in df.columns:
        df["actual_total"] = df["home_score"] + df["away_score"]
        df["total_over"] = df.apply(
            lambda r: int(r["actual_total"] > r["total_line"])
            if pd.notna(r.get("total_line")) else None,
            axis=1
        )
    
    # First half labels (if quarter data available)
    if "home_q1" in df.columns and "home_q2" in df.columns:
        for c in ["home_q1", "home_q2", "away_q1", "away_q2"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        
        df["home_1h_score"] = df["home_q1"].fillna(0) + df["home_q2"].fillna(0)
        df["away_1h_score"] = df["away_q1"].fillna(0) + df["away_q2"].fillna(0)
        
        if "home_1h_win" not in df.columns:
            df["home_1h_win"] = (df["home_1h_score"] > df["away_1h_score"]).astype(int)
        
        if "actual_1h_margin" not in df.columns:
            df["actual_1h_margin"] = df["home_1h_score"] - df["away_1h_score"]
            df["actual_1h_total"] = df["home_1h_score"] + df["away_1h_score"]
        
        if "1h_spread_covered" not in df.columns and "1h_spread_line" in df.columns:
            df["1h_spread_covered"] = df.apply(
                lambda r: int(r["actual_1h_margin"] > -r["1h_spread_line"])
                if pd.notna(r.get("1h_spread_line")) else None,
                axis=1
            )
        
        if "1h_total_over" not in df.columns and "1h_total_line" in df.columns:
            df["1h_total_over"] = df.apply(
                lambda r: int(r["actual_1h_total"] > r["1h_total_line"])
                if pd.notna(r.get("1h_total_line")) else None,
                axis=1
            )
    
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
            from src.modeling.models import FirstHalfSpreadsModel, FirstHalfTotalsModel, TeamTotalsModel

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

    parser = argparse.ArgumentParser(description="Train NBA prediction models (v6.0 - 4 independent markets)")
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
        default="all",
        help="Which markets to train: fg (full game), 1h (first half), or all (default)",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Train ensemble models (logistic + gradient boosting)",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use legacy training mode (train_models + train_first_half_models)",
    )
    args = parser.parse_args()

    # Determine which markets to train
    if args.market == "all":
        markets = list(MARKET_CONFIG.keys())
    elif args.market == "fg":
        markets = ["fg_spread", "fg_total"]
    elif args.market == "1h":
        markets = ["1h_spread", "1h_total"]
    else:
        markets = list(MARKET_CONFIG.keys())
    
    # Legacy mode (backward compatibility)
    if args.legacy:
        print("\n[INFO] Legacy mode: Using old training functions")
        model_type = args.model_type
        if args.ensemble:
            if args.market in ["fg", "all"]:
                train_models(model_type="logistic", test_size=args.test_size, output_dir=args.output_dir)
                train_models(model_type="gradient_boosting", test_size=args.test_size, output_dir=args.output_dir)
            if args.market in ["1h", "all"]:
                train_first_half_models(test_size=args.test_size, output_dir=args.output_dir)
        else:
            if args.market in ["fg", "all"]:
                train_models(model_type=model_type, test_size=args.test_size, output_dir=args.output_dir)
            if args.market in ["1h", "all"]:
                train_first_half_models(model_type=model_type, test_size=args.test_size, output_dir=args.output_dir)
        return

    # New unified training mode (v6.0)
    if args.ensemble:
        print("\n[INFO] Ensemble mode: Training both logistic and gradient boosting models")
        print("\n>>> Training with logistic regression...")
        train_all_markets(
            model_type="logistic",
            test_size=args.test_size,
            output_dir=args.output_dir,
            markets=markets,
        )
        print("\n>>> Training with gradient boosting...")
        train_all_markets(
            model_type="gradient_boosting",
            test_size=args.test_size,
            output_dir=args.output_dir,
            markets=markets,
        )
    else:
        train_all_markets(
            model_type=args.model_type,
            test_size=args.test_size,
            output_dir=args.output_dir,
            markets=markets,
        )


if __name__ == "__main__":
    main()
