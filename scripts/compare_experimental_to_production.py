#!/usr/bin/env python3
"""
Compare experimental models against production baseline.

This script evaluates both production and experimental models on the same
held-out test data to determine if the experimental model is an improvement.

Usage:
    python scripts/compare_experimental_to_production.py
    python scripts/compare_experimental_to_production.py --experiment v2_gradient_boosting
    python scripts/compare_experimental_to_production.py --test-data data/processed/current_season.csv

Output:
    Comparison table showing accuracy and ROI for each market
    Statistical significance test results
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Directories
PRODUCTION_DIR = PROJECT_ROOT / "models" / "production"
EXPERIMENTAL_DIR = PROJECT_ROOT / "models" / "experimental"
DATA_DIR = PROJECT_ROOT / "data"


def load_production_model(market: str):
    """Load production model for a market."""
    model_path = PRODUCTION_DIR / f"{market}_model.joblib"
    if not model_path.exists():
        logger.warning(f"Production model not found: {model_path}")
        return None
    return joblib.load(model_path)


def load_experimental_model(experiment_name: str, market: str):
    """Load experimental model for a market."""
    model_path = EXPERIMENTAL_DIR / experiment_name / f"{market}_model.joblib"
    if not model_path.exists():
        logger.warning(f"Experimental model not found: {model_path}")
        return None
    return joblib.load(model_path)


def get_model_features(model) -> list[str]:
    """Extract feature names from a model."""
    if isinstance(model, dict):
        return model.get("feature_names", [])
    if hasattr(model, "feature_names"):
        return model.feature_names
    if hasattr(model, "feature_names_"):
        return model.feature_names_
    return []


def predict_with_model(model, X: pd.DataFrame) -> np.ndarray:
    """Get predictions from a model."""
    if isinstance(model, dict):
        inner_model = model.get("model")
        if inner_model is None:
            raise ValueError("Model dict has no 'model' key")
        return inner_model.predict_proba(X)[:, 1]
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def evaluate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict[str, float]:
    """Evaluate a model on test data."""
    y_proba = predict_with_model(model, X)
    y_pred = (y_proba >= 0.5).astype(int)
    
    accuracy = (y_pred == y).mean()
    
    # ROI at -110 odds
    correct = (y_pred == y).astype(int)
    winnings = correct * 100 - (1 - correct) * 110
    roi = winnings.sum() / (len(y) * 110)
    
    return {
        "accuracy": accuracy,
        "roi": roi,
        "n_samples": len(y),
    }


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Compute bootstrap confidence interval for difference in accuracy.
    
    Returns:
        Dictionary with mean difference, CI bounds, and p-value
    """
    n = len(y_true)
    diffs = []
    
    for _ in range(n_bootstrap):
        # Sample with replacement
        idx = np.random.choice(n, size=n, replace=True)
        
        # Calculate accuracy for each model
        acc_a = (y_pred_a[idx] == y_true[idx]).mean()
        acc_b = (y_pred_b[idx] == y_true[idx]).mean()
        
        diffs.append(acc_b - acc_a)  # Positive = B is better
    
    diffs = np.array(diffs)
    
    # Calculate statistics
    mean_diff = diffs.mean()
    ci_lower = np.percentile(diffs, alpha / 2 * 100)
    ci_upper = np.percentile(diffs, (1 - alpha / 2) * 100)
    
    # P-value: proportion of bootstraps where difference has opposite sign
    if mean_diff > 0:
        p_value = (diffs <= 0).mean()
    else:
        p_value = (diffs >= 0).mean()
    
    return {
        "mean_diff": mean_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "significant": p_value < alpha,
    }


def compare_markets(
    experiment_name: str,
    test_df: pd.DataFrame,
    markets: list[str] = None,
) -> dict[str, Any]:
    """
    Compare production vs experimental models for all markets.
    
    Args:
        experiment_name: Name of experimental model directory
        test_df: Test data DataFrame
        markets: List of markets to compare (default: all 4)
    
    Returns:
        Comparison results dictionary
    """
    markets = markets or ["fg_spread", "fg_total", "1h_spread", "1h_total"]
    
    label_map = {
        "fg_spread": "fg_spread_covered",
        "fg_total": "fg_total_over",
        "1h_spread": "1h_spread_covered",
        "1h_total": "1h_total_over",
    }
    
    results = {}
    
    for market in markets:
        logger.info(f"\n{'='*40}")
        logger.info(f"Market: {market.upper()}")
        logger.info(f"{'='*40}")
        
        label_col = label_map[market]
        
        if label_col not in test_df.columns:
            logger.warning(f"  Label column not found: {label_col}")
            continue
        
        # Load models
        prod_model = load_production_model(market)
        exp_model = load_experimental_model(experiment_name, market)
        
        if prod_model is None or exp_model is None:
            logger.warning(f"  Could not load both models for {market}")
            continue
        
        # Get features
        prod_features = get_model_features(prod_model)
        exp_features = get_model_features(exp_model)
        
        # Prepare test data for production model
        prod_available = [f for f in prod_features if f in test_df.columns]
        if len(prod_available) < len(prod_features) * 0.8:
            logger.warning(f"  Production model missing many features: {len(prod_available)}/{len(prod_features)}")
        
        # Prepare test data for experimental model
        exp_available = [f for f in exp_features if f in test_df.columns]
        if len(exp_available) < len(exp_features):
            logger.warning(f"  Experimental model missing features: {len(exp_available)}/{len(exp_features)}")
        
        # Filter to valid rows
        valid_df = test_df[test_df[label_col].notna()].copy()
        
        if len(valid_df) < 50:
            logger.warning(f"  Insufficient test data: {len(valid_df)} games")
            continue
        
        y = valid_df[label_col].astype(int).values
        
        # Evaluate production model
        try:
            X_prod = valid_df[prod_available].fillna(valid_df[prod_available].median())
            prod_metrics = evaluate_model(prod_model, X_prod, y)
            logger.info(f"  Production:   acc={prod_metrics['accuracy']:.1%}, roi={prod_metrics['roi']:+.1%}")
        except Exception as e:
            logger.error(f"  Production model error: {e}")
            prod_metrics = None
        
        # Evaluate experimental model
        try:
            X_exp = valid_df[exp_available].fillna(valid_df[exp_available].median())
            exp_metrics = evaluate_model(exp_model, X_exp, y)
            logger.info(f"  Experimental: acc={exp_metrics['accuracy']:.1%}, roi={exp_metrics['roi']:+.1%}")
        except Exception as e:
            logger.error(f"  Experimental model error: {e}")
            exp_metrics = None
        
        if prod_metrics and exp_metrics:
            diff_acc = exp_metrics["accuracy"] - prod_metrics["accuracy"]
            diff_roi = exp_metrics["roi"] - prod_metrics["roi"]
            
            logger.info(f"  Difference:   acc={diff_acc:+.1%}, roi={diff_roi:+.1%}")
            
            # Bootstrap test
            y_pred_prod = (predict_with_model(prod_model, X_prod) >= 0.5).astype(int)
            y_pred_exp = (predict_with_model(exp_model, X_exp) >= 0.5).astype(int)
            
            bootstrap_result = bootstrap_confidence_interval(y, y_pred_prod, y_pred_exp)
            
            sig_str = "SIGNIFICANT" if bootstrap_result["significant"] else "not significant"
            logger.info(
                f"  Bootstrap:    diff={bootstrap_result['mean_diff']:+.1%}, "
                f"95% CI=[{bootstrap_result['ci_lower']:+.1%}, {bootstrap_result['ci_upper']:+.1%}], "
                f"p={bootstrap_result['p_value']:.3f} ({sig_str})"
            )
            
            results[market] = {
                "production": prod_metrics,
                "experimental": exp_metrics,
                "difference": {
                    "accuracy": diff_acc,
                    "roi": diff_roi,
                },
                "bootstrap": bootstrap_result,
            }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare experimental models against production baseline"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="v1_expanded_data",
        help="Experiment name to compare (default: v1_expanded_data)"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default=None,
        help="Path to test data CSV (default: data/experimental/training_data_full.csv, last season only)"
    )
    
    args = parser.parse_args()
    
    # Load test data
    if args.test_data:
        test_path = Path(args.test_data)
    else:
        # Use experimental training data, filter to most recent season
        test_path = DATA_DIR / "experimental" / "training_data_full.csv"
    
    if not test_path.exists():
        logger.error(f"Test data not found: {test_path}")
        sys.exit(1)
    
    logger.info(f"Loading test data from {test_path}")
    df = pd.read_csv(test_path, parse_dates=["game_date"])
    
    # Use only the most recent season as held-out test
    most_recent_season = df["season"].max()
    test_df = df[df["season"] == most_recent_season].copy()
    
    logger.info(f"Test set: {len(test_df)} games from season {most_recent_season}")
    
    # Compare
    results = compare_markets(args.experiment, test_df)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    
    better_count = 0
    sig_better_count = 0
    
    for market, r in results.items():
        diff = r["difference"]["accuracy"]
        is_better = diff > 0
        is_sig = r["bootstrap"]["significant"] and diff > 0
        
        if is_better:
            better_count += 1
        if is_sig:
            sig_better_count += 1
        
        status = ""
        if is_sig:
            status = " ✓ SIGNIFICANTLY BETTER"
        elif is_better:
            status = " (better, not significant)"
        
        logger.info(f"  {market}: {diff:+.1%}{status}")
    
    logger.info(f"\nExperimental is better in {better_count}/{len(results)} markets")
    logger.info(f"Statistically significant improvement in {sig_better_count}/{len(results)} markets")
    
    # Recommendation
    if sig_better_count >= 3:
        logger.info("\n🎉 RECOMMENDATION: Promote experimental model to production")
    elif sig_better_count >= 2:
        logger.info("\n⚠️  RECOMMENDATION: Consider promotion, run more tests")
    else:
        logger.info("\n❌ RECOMMENDATION: Keep current production model")
    
    # Save comparison results
    output_path = EXPERIMENTAL_DIR / args.experiment / "comparison_results.json"
    with open(output_path, "w") as f:
        # Convert numpy types to Python types for JSON
        def convert_for_json(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_for_json(v) for v in obj]
            return obj
        
        json.dump(convert_for_json(results), f, indent=2)
    
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
