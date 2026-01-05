#!/usr/bin/env python3
"""
Optimize confidence thresholds from walk-forward prediction outputs.

Given one or more `backtest_preds_*.csv` files (from
scripts/walkforward_backtest_theodds.py), scan a confidence threshold grid and
find the ROI-maximizing cutoff per market with a minimum bet count.

This does NOT re-train models; it's pure post-analysis for strategy selection.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class ThresholdResult:
    market: str
    model_type: str
    best_threshold: float
    best_roi: float
    best_n: int


def optimize_for_market(
    df: pd.DataFrame,
    market: str,
    model_type: str,
    thresholds: np.ndarray,
    min_bets: int,
) -> Optional[ThresholdResult]:
    sub = df[
        (df["market"] == market) & (df["model_type"] == model_type)
    ].copy()
    if sub.empty:
        return None

    best_th = None
    best_roi = None
    best_n = None

    for th in thresholds:
        s = sub[sub["confidence"] >= th]
        if len(s) < min_bets:
            continue
        roi = float(s["profit"].sum() / len(s))
        if best_roi is None or roi > best_roi:
            best_roi = roi
            best_th = float(th)
            best_n = int(len(s))

    if best_roi is None or best_th is None or best_n is None:
        return None

    return ThresholdResult(
        market=market,
        model_type=model_type,
        best_threshold=best_th,
        best_roi=float(best_roi),
        best_n=int(best_n),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optimize confidence thresholds by ROI"
    )
    p.add_argument(
        "--preds",
        nargs="+",
        required=True,
        help="One or more backtest prediction CSV files",
    )
    p.add_argument("--min", dest="min_th", type=float, default=0.50)
    p.add_argument("--max", dest="max_th", type=float, default=0.76)
    p.add_argument("--step", dest="step_th", type=float, default=0.02)
    p.add_argument("--min-bets", type=int, default=200)
    p.add_argument(
        "--out-dir",
        type=str,
        default="data/backtest_results",
        help="Output directory for threshold optimization results",
    )
    p.add_argument("--tag", type=str, default="thresholds")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    thresholds = np.arange(args.min_th, args.max_th, args.step_th)

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"{args.tag}_" if args.tag else ""

    all_results: List[ThresholdResult] = []

    for preds_path in args.preds:
        path = Path(preds_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        df = pd.read_csv(path)

        required = {"market", "model_type", "confidence", "profit"}
        if not required.issubset(df.columns):
            missing = sorted(required - set(df.columns))
            raise ValueError(f"{path}: missing required columns: {missing}")

        markets = sorted(df["market"].dropna().unique().tolist())
        model_types = sorted(df["model_type"].dropna().unique().tolist())

        for m in markets:
            for mt in model_types:
                res = optimize_for_market(
                    df=df,
                    market=m,
                    model_type=mt,
                    thresholds=thresholds,
                    min_bets=args.min_bets,
                )
                if res:
                    all_results.append(res)

    payload: Dict[str, object] = {
        "generated_at": ts,
        "min_bets": args.min_bets,
        "grid": {
            "min": args.min_th,
            "max": args.max_th,
            "step": args.step_th,
        },
        "results": [asdict(r) for r in all_results],
    }

    out_json = out_dir / f"optimized_thresholds_{tag}{ts}.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[OK] Wrote: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
