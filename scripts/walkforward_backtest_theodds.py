#!/usr/bin/env python3
"""
Leakage-safe walk-forward backtest using precomputed training features.

This script is designed to be FAST and reproducible:
  - Uses the already-computed feature columns in training_data*.csv
  - Uses strict time ordering (expanding window, step-size test chunks)
  - Requires real 1H lines for 1H markets (no FG/2 approximation)

Typical workflow:
  1) python scripts/cache_theodds_lines.py --seasons 2023-2024 2024-2025 \
         --bookmaker draftkings
  2) python scripts/merge_theodds_lines_into_training_data.py \
         --out data/processed/training_data_theodds.csv
  3) python scripts/validate_leakage.py \
         --training-path data/processed/training_data_theodds.csv
  4) python scripts/walkforward_backtest_theodds.py \
         --data data/processed/training_data_theodds.csv

Outputs timestamped CSV + JSON summaries to data/backtest_results/.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.feature_config import (  # noqa: E402
    filter_available_features,
    get_spreads_features,
    get_totals_features,
)
from src.modeling.period_features import get_model_features  # noqa: E402
from src.modeling.models import (  # noqa: E402
    SpreadsModel,
    TotalsModel,
    FirstHalfSpreadsModel,
    FirstHalfTotalsModel,
)
from src.modeling.unified_features import (  # noqa: E402
    get_all_market_keys,
    get_model_config,
)


@dataclass
class MarketSummary:
    market: str
    model_type: str
    n_bets: int
    accuracy: float
    roi: float
    profit: float
    high_conf_n: int
    high_conf_accuracy: float
    high_conf_roi: float


MARKET_TO_MODEL = {
    "fg_spread": SpreadsModel,
    "fg_total": TotalsModel,
    "1h_spread": FirstHalfSpreadsModel,
    "1h_total": FirstHalfTotalsModel,
}


def profit_flat_110(pred: int, actual: int) -> float:
    return (100.0 / 110.0) if pred == actual else -1.0


def get_features_for_market(market_key: str, df_cols: List[str]) -> List[str]:
    cfg = get_model_config(market_key)

    if cfg.period == "fg":
        base = (
            get_spreads_features()
            if cfg.market_type == "spread"
            else get_totals_features()
        )
        min_pct = 0.30
    else:
        base = get_model_features(cfg.period, cfg.market_type)
        min_pct = 0.15

    feats = filter_available_features(base, df_cols, min_required_pct=min_pct)
    # Never include the label itself as a feature
    feats = [f for f in feats if f != cfg.label_column]
    return feats


def walk_forward_backtest(
    df: pd.DataFrame,
    market_key: str,
    model_type: str,
    min_train: int,
    step: int,
    calibrate: bool,
    high_conf: float,
) -> tuple[pd.DataFrame, MarketSummary]:
    cfg = get_model_config(market_key)
    ModelClass = MARKET_TO_MODEL[market_key]

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Require line + label
    df = df[df[cfg.line_column].notna() & df[cfg.label_column].notna()].copy()
    if len(df) < (min_train + step):
        raise ValueError(
            f"{market_key}: insufficient data after filtering (n={len(df)})"
        )

    features = get_features_for_market(market_key, df.columns.tolist())
    if len(features) < 5:
        raise ValueError(
            f"{market_key}: too few features available (n={len(features)})"
        )

    preds_rows: List[Dict[str, Any]] = []

    i = min_train
    while i < len(df):
        train = df.iloc[:i].copy()
        test = df.iloc[i:i + step].copy()
        if len(test) == 0:
            break

        model = ModelClass(
            name=f"{market_key}_wf",
            model_type=model_type,
            feature_columns=features,
            use_calibration=calibrate,
        )
        model.fit(train, train[cfg.label_column].astype(int))

        proba = model.predict_proba(test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        actual = test[cfg.label_column].astype(int).values
        conf = np.where(pred == 1, proba, 1.0 - proba)

        for j in range(len(test)):
            preds_rows.append({
                "date": test.iloc[j]["date"],
                "home_team": test.iloc[j].get("home_team"),
                "away_team": test.iloc[j].get("away_team"),
                "market": market_key,
                "model_type": model_type,
                "proba": float(proba[j]),
                "confidence": float(conf[j]),
                "pred": int(pred[j]),
                "actual": int(actual[j]),
                "profit": float(
                    profit_flat_110(int(pred[j]), int(actual[j]))
                ),
                "line": float(test.iloc[j][cfg.line_column]),
            })

        i += step

    out = pd.DataFrame(preds_rows)
    acc = float((out["pred"] == out["actual"]).mean()) if len(out) else 0.0
    prof = float(out["profit"].sum()) if len(out) else 0.0
    roi = float(prof / len(out)) if len(out) else 0.0

    hc = out[out["confidence"] >= high_conf].copy()
    hc_acc = float((hc["pred"] == hc["actual"]).mean()) if len(hc) else 0.0
    hc_prof = float(hc["profit"].sum()) if len(hc) else 0.0
    hc_roi = float(hc_prof / len(hc)) if len(hc) else 0.0

    summary = MarketSummary(
        market=market_key,
        model_type=model_type,
        n_bets=int(len(out)),
        accuracy=acc,
        roi=roi,
        profit=prof,
        high_conf_n=int(len(hc)),
        high_conf_accuracy=hc_acc,
        high_conf_roi=hc_roi,
    )
    return out, summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward backtest (precomputed features)")
    p.add_argument(
        "--data",
        type=str,
        default="data/processed/training_data_theodds.csv",
    )
    p.add_argument(
        "--markets",
        type=str,
        default="all",
        help="Comma-separated market keys or 'all'",
    )
    p.add_argument(
        "--model-type",
        type=str,
        default="logistic",
        choices=["logistic", "gradient_boosting"],
    )
    p.add_argument(
        "--min-train",
        type=int,
        default=800,
        help="Min training samples before first test chunk",
    )
    p.add_argument(
        "--step",
        type=int,
        default=200,
        help="Test chunk size (games) per iteration",
    )
    p.add_argument(
        "--calibrate",
        action="store_true",
        help="Enable isotonic calibration (slower)",
    )
    p.add_argument(
        "--high-conf",
        type=float,
        default=0.60,
        help="High confidence threshold",
    )
    p.add_argument("--output-dir", type=str, default="data/backtest_results")
    p.add_argument("--tag", type=str, default="theodds")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    data_path = PROJECT_ROOT / args.data
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data: {data_path}")

    df = pd.read_csv(data_path)

    if args.markets.strip().lower() == "all":
        markets = get_all_market_keys()
    else:
        markets = [m.strip() for m in args.markets.split(",") if m.strip()]

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.tag}_" if args.tag else ""

    all_preds: List[pd.DataFrame] = []
    summaries: List[MarketSummary] = []

    for market_key in markets:
        if market_key not in MARKET_TO_MODEL:
            raise ValueError(
                f"Unknown market: {market_key} "
                f"(valid: {list(MARKET_TO_MODEL.keys())})"
            )

        # Market-specific min_train override for sparse 1H data
        min_train = args.min_train
        if market_key.startswith("1h_") and min_train > 300:
            min_train = 250

        preds, summary = walk_forward_backtest(
            df=df,
            market_key=market_key,
            model_type=args.model_type,
            min_train=min_train,
            step=args.step,
            calibrate=args.calibrate,
            high_conf=args.high_conf,
        )

        all_preds.append(preds)
        summaries.append(summary)

        print(
            f"[OK] {market_key}: n={summary.n_bets}, acc={summary.accuracy:.1%}, "
            f"roi={summary.roi:+.1%}, hc_n={summary.high_conf_n}, "
            f"hc_roi={summary.high_conf_roi:+.1%}"
        )

    preds_all = (
        pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    )
    preds_path = out_dir / f"backtest_preds_{tag}{args.model_type}_{run_ts}.csv"
    preds_all.to_csv(preds_path, index=False)

    summary_payload = {
        "generated_at": run_ts,
        "data": str(data_path),
        "model_type": args.model_type,
        "min_train": args.min_train,
        "step": args.step,
        "calibrate": bool(args.calibrate),
        "high_conf": args.high_conf,
        "summaries": [asdict(s) for s in summaries],
    }
    summary_path = out_dir / f"backtest_summary_{tag}{args.model_type}_{run_ts}.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print(f"[OK] Wrote predictions: {preds_path}")
    print(f"[OK] Wrote summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

