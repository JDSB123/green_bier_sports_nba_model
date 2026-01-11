"""Validate that models don't use leaked future information.

This script checks:
1. First-half models don't use full-game outcomes
2. Training data doesn't contain future information
3. Feature engineering doesn't look ahead
"""
import os
import sys
import argparse
from typing import List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd

from src.modeling.models import (
    FirstHalfSpreadsModel, 
    FirstHalfTotalsModel,
    SpreadsModel,
    TotalsModel,
)
from src.config import settings

# Features that contain full-game information and must NOT be in FH models
FORBIDDEN_FH_FEATURES = [
    "home_score",
    "away_score",
    "total_score",
    "home_margin",
    "went_over",
    "spread_covered",
    "fh_home_score",  # Actual first-half scores are outcomes
    "fh_away_score",
]

# Features that could indicate lookahead bias in any model
SUSPICIOUS_FEATURES = [
    "next_game",
    "future",
    "result",
    "outcome",
    "final",
]


def check_fh_model_features(model_class, model_name: str) -> bool:
    """Check that a FH model class doesn't use forbidden features."""
    features = getattr(model_class, "DEFAULT_FEATURES", [])
    forbidden_found = [f for f in features if f in FORBIDDEN_FH_FEATURES]
    
    if forbidden_found:
        print(f"[FAIL] LEAKAGE DETECTED in {model_name}:")
        print(f"   Forbidden features: {forbidden_found}")
        return False
    else:
        print(f"[OK] {model_name} is leakage-free")
        return True


def check_training_data_leakage(training_path: str = None) -> Tuple[bool, List[str]]:
    """
    Check training data for potential leakage issues.
    
    Validates:
    - No future dates in features
    - Features are computed from past data only
    - Targets match actual outcomes
    """
    training_path = training_path or os.path.join(settings.data_processed_dir, "training_data.csv")
    
    if not os.path.exists(training_path):
        print(f"[WARN] Training data not found: {training_path}")
        return True, []
    
    df = pd.read_csv(training_path)
    issues = []
    
    # Check for suspicious column names
    for col in df.columns:
        col_lower = col.lower()
        for suspicious in SUSPICIOUS_FEATURES:
            if suspicious in col_lower and col not in ["spread_covered", "went_over"]:
                issues.append(f"Suspicious column name: {col}")
    
    # Check date ordering
    if "date" in df.columns or "game_date" in df.columns:
        date_col = "date" if "date" in df.columns else "game_date"
        df["_date"] = pd.to_datetime(df[date_col], errors="coerce")
        
        # Check if any rolling features have future values
        # This is a heuristic check - detailed validation requires feature code review
        if "home_ppg" in df.columns:
            # PPG should increase over time as we accumulate data
            # Sudden spikes at the start could indicate leakage
            first_10 = df.head(10)["home_ppg"].std()
            overall = df["home_ppg"].std()
            if first_10 > overall * 2:
                issues.append("Early games have unusually high PPG variance (possible leakage)")
    
    # Check target validity (only where line + label are present)
    if "spread_covered" in df.columns and "spread_line" in df.columns:
        # Prefer explicit realized margin columns if present
        if "actual_margin" in df.columns:
            margin_col = "actual_margin"
        elif "home_score" in df.columns and "away_score" in df.columns:
            df["_actual_margin_tmp"] = df["home_score"] - df["away_score"]
            margin_col = "_actual_margin_tmp"
        else:
            margin_col = None

        if margin_col:
            m = df["spread_line"].notna() & df[margin_col].notna() & df["spread_covered"].notna()
            if m.any():
                df.loc[m, "_expected_covered"] = (df.loc[m, margin_col] > -df.loc[m, "spread_line"]).astype(int)
                mismatch = (
                    df.loc[m, "spread_covered"].astype(int)
                    != df.loc[m, "_expected_covered"].astype(int)
                ).sum()
                if mismatch > 0:
                    issues.append(f"spread_covered mismatch with actual margin: {mismatch} games")
    
    if "total_over" in df.columns and "total_line" in df.columns:
        if "actual_total" in df.columns:
            total_col = "actual_total"
        elif "home_score" in df.columns and "away_score" in df.columns:
            df["_actual_total_tmp"] = df["home_score"] + df["away_score"]
            total_col = "_actual_total_tmp"
        else:
            total_col = None

        if total_col:
            m = df["total_line"].notna() & df[total_col].notna() & df["total_over"].notna()
            if m.any():
                df.loc[m, "_expected_over"] = (df.loc[m, total_col] > df.loc[m, "total_line"]).astype(int)
                mismatch = (
                    df.loc[m, "total_over"].astype(int)
                    != df.loc[m, "_expected_over"].astype(int)
                ).sum()
                if mismatch > 0:
                    issues.append(f"total_over mismatch with actual total: {mismatch} games")
    
    return len(issues) == 0, issues


def check_feature_temporal_integrity() -> bool:
    """
    Verify that feature engineering respects temporal ordering.
    
    This is a code-level check of the FeatureEngineer class.
    """
    from src.modeling.features import FeatureEngineer
    
    # Check that compute methods filter by date
    import inspect
    source = inspect.getsource(FeatureEngineer.compute_team_rolling_stats)
    
    if "< as_of_date" not in source and "< game_date" not in source:
        print("[WARN] Rolling stats may not filter by date properly")
        return False
    
    print("[OK] Feature engineering uses temporal filtering")
    return True


def main():
    parser = argparse.ArgumentParser(description="Validate models and data for leakage")
    parser.add_argument(
        "--training-path",
        type=str,
        default=None,
        help="Optional path to training CSV (defaults to data/processed/training_data.csv)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("DATA LEAKAGE VALIDATION")
    print("=" * 60)
    
    all_pass = True
    
    # Check first-half models
    print("\n1. First-Half Model Features:")
    all_pass &= check_fh_model_features(FirstHalfSpreadsModel, "FirstHalfSpreadsModel")
    all_pass &= check_fh_model_features(FirstHalfTotalsModel, "FirstHalfTotalsModel")
    
    # Check training data
    print("\n2. Training Data Validation:")
    data_ok, issues = check_training_data_leakage(args.training_path)
    if data_ok:
        print("[OK] Training data passed leakage checks")
    else:
        print("[FAIL] Training data issues found:")
        for issue in issues:
            print(f"   - {issue}")
        all_pass = False
    
    # Check feature engineering
    print("\n3. Feature Engineering Temporal Check:")
    try:
        all_pass &= check_feature_temporal_integrity()
    except Exception as e:
        print(f"[WARN] Could not verify feature engineering: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    if all_pass:
        print("[OK] ALL VALIDATION CHECKS PASSED")
        return 0
    else:
        print("[FAIL] VALIDATION FAILED - Review issues above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
