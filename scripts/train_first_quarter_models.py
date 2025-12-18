#!/usr/bin/env python3
"""
Train first-quarter (Q1) spread, total, and moneyline models using q1_training_data.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.modeling.models import (
    FirstQuarterSpreadsModel,
    FirstQuarterTotalsModel,
    FirstQuarterMoneylineModel,
    ModelMetrics,
)

MODEL_DIR = Path(settings.data_processed_dir) / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SPREAD_FEATURES = [
    "home_ppg_q1",
    "away_ppg_q1",
    "home_margin_q1",
    "away_margin_q1",
    "ppg_diff_q1",
    "margin_diff_q1",
    "home_win_pct",
    "away_win_pct",
    "win_pct_diff",
    "home_rest_days",
    "away_rest_days",
    "home_b2b",
    "away_b2b",
    "rest_diff",
    "dynamic_hca",
]

TOTAL_FEATURES = SPREAD_FEATURES + [
    "q1_spread_line",
    "q1_total_line",
]

MONEYLINE_FEATURES = SPREAD_FEATURES + [
    "q1_home_ml",
    "q1_away_ml",
    "q1_win_pct_diff",
]


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Q1 dataset not found at {path}. Run scripts/generate_q1_training_data.py first."
        )
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def temporal_split(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * (1 - test_size))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def train_and_evaluate(
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
) -> Tuple[ModelMetrics, ModelMetrics]:
    model.fit(train_df, train_df[target_col])
    train_metrics = model.evaluate(train_df, train_df[target_col])
    test_metrics = ModelMetrics()
    if len(test_df) > 0:
        test_metrics = model.evaluate(test_df, test_df[target_col])
    return train_metrics, test_metrics


def prepare_frame(df: pd.DataFrame, features: List[str], target: str) -> pd.DataFrame:
    available = [f for f in features if f in df.columns]
    missing = set(features) - set(available)
    if missing:
        print(f"[WARN] Missing features ({len(missing)}): {', '.join(sorted(missing))}")
    frame = df[available + [target]].dropna()
    return frame


def save_model(model, filename: str) -> None:
    path = MODEL_DIR / filename
    model.save(str(path))
    print(f"[OK] Saved {filename} ({path})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Q1 spread/total/moneyline models")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(settings.data_processed_dir) / "q1_training_data.parquet",
        help="Path to q1_training_data parquet/csv",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of games reserved for validation (temporal split)",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    print(f"Loaded {len(dataset)} Q1 games from {args.dataset}")

    train_df, test_df = temporal_split(dataset, test_size=args.test_size)
    print(f"Train: {len(train_df)} games | Test: {len(test_df)} games")

    # Spread model
    combined = pd.concat([train_df, test_df], ignore_index=True)
    spread_frame = prepare_frame(combined, SPREAD_FEATURES, "target_spread")
    spread_train, spread_test = temporal_split(spread_frame, args.test_size)
    if len(spread_train) >= 100:
        spread_model = FirstQuarterSpreadsModel(model_type="logistic", feature_columns=SPREAD_FEATURES)
        tr_metrics, te_metrics = train_and_evaluate(spread_model, spread_train, spread_test, "target_spread")
        print(f"\nQ1 Spread - Train Acc {tr_metrics.accuracy:.1%} | Test Acc {te_metrics.accuracy:.1%}")
        save_model(spread_model, "q1_spreads_model.joblib")
    else:
        print("[WARN] Not enough data to train Q1 spread model.")

    # Totals model
    total_frame = prepare_frame(combined, TOTAL_FEATURES, "target_total")
    total_train, total_test = temporal_split(total_frame, args.test_size)
    if len(total_train) >= 100:
        totals_model = FirstQuarterTotalsModel(model_type="logistic", feature_columns=TOTAL_FEATURES)
        tr_metrics, te_metrics = train_and_evaluate(totals_model, total_train, total_test, "target_total")
        print(f"\nQ1 Total - Train Acc {tr_metrics.accuracy:.1%} | Test Acc {te_metrics.accuracy:.1%}")
        save_model(totals_model, "q1_totals_model.joblib")
    else:
        print("[WARN] Not enough data to train Q1 totals model.")

    # Moneyline model
    ml_frame = prepare_frame(combined, MONEYLINE_FEATURES, "target_moneyline")
    ml_train, ml_test = temporal_split(ml_frame, args.test_size)
    if len(ml_train) >= 100:
        moneyline_model = FirstQuarterMoneylineModel(model_type="logistic", feature_columns=MONEYLINE_FEATURES)
        tr_metrics, te_metrics = train_and_evaluate(moneyline_model, ml_train, ml_test, "target_moneyline")
        print(f"\nQ1 Moneyline - Train Acc {tr_metrics.accuracy:.1%} | Test Acc {te_metrics.accuracy:.1%}")
        save_model(moneyline_model, "q1_moneyline_model.joblib")
    else:
        print("[WARN] Not enough data to train Q1 moneyline model.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
