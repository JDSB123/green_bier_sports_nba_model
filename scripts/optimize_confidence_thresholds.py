#!/usr/bin/env python3
"""
Optimize confidence/edge thresholds using canonical training data.

This script runs a single production backtest and then sweeps threshold grids
to surface the best combinations by ROI or accuracy.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import backtest_production


def _parse_markets(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    markets = [m.strip() for m in value.split(",") if m.strip()]
    return markets or None


def _build_grid(min_val: float, max_val: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("step must be > 0")
    if max_val < min_val:
        raise ValueError("max must be >= min")
    values = []
    current = min_val
    while current <= max_val + 1e-9:
        values.append(round(current, 4))
        current += step
    return values


def _edge_from_bet(bet: backtest_production.BetResult) -> Optional[float]:
    if bet.line is None:
        return None
    if bet.predicted_value is None:
        return None
    if "spread" in bet.market:
        return abs(bet.predicted_value + bet.line)
    return abs(bet.predicted_value - bet.line)


def _summarize(
    bets: List[backtest_production.BetResult],
    confidence_threshold: float,
    edge_threshold: float,
    pricing_enabled: bool,
) -> Optional[Dict[str, float]]:
    filtered: List[backtest_production.BetResult] = []
    for bet in bets:
        edge_value = _edge_from_bet(bet)
        if edge_value is None:
            continue
        if bet.confidence < confidence_threshold:
            continue
        if edge_value < edge_threshold:
            continue
        filtered.append(bet)

    n_bets = len(filtered)
    if n_bets == 0:
        return None

    wins = sum(1 for b in filtered if b.won)
    accuracy = wins / n_bets

    if pricing_enabled:
        total_profit = sum(float(b.profit or 0.0) for b in filtered)
        roi = total_profit / n_bets
    else:
        total_profit = None
        roi = None

    return {
        "n_bets": n_bets,
        "accuracy": round(accuracy, 4),
        "total_profit": None if total_profit is None else round(total_profit, 4),
        "roi": None if roi is None else round(roi, 4),
        "confidence_threshold": round(confidence_threshold, 4),
        "edge_threshold": round(edge_threshold, 4),
    }


def _rank_results(
    rows: List[Dict[str, float]],
    objective: str,
) -> List[Dict[str, float]]:
    def _value(row: Dict[str, float]) -> float:
        value = row.get(objective)
        return float(value) if value is not None else float("-inf")

    return sorted(
        rows,
        key=lambda r: (_value(r), r.get("accuracy", 0.0), r.get("n_bets", 0)),
        reverse=True,
    )


def run(args: argparse.Namespace) -> Dict[str, List[Dict[str, float]]]:
    markets = _parse_markets(args.markets)

    pricing_enabled = not args.no_pricing
    if pricing_enabled and (args.spread_juice is None or args.total_juice is None):
        raise ValueError("pricing enabled requires --spread-juice and --total-juice")

    objective = args.objective
    if objective is None:
        objective = "accuracy" if args.no_pricing else "roi"
    if args.no_pricing and objective in {"roi", "total_profit"}:
        raise ValueError("objective requires pricing; remove --no-pricing")

    confidence_grid = _build_grid(
        args.confidence_min, args.confidence_max, args.confidence_step
    )
    edge_grid = _build_grid(args.edge_min, args.edge_max, args.edge_step)

    results = backtest_production.run_backtest(
        data_path=args.data,
        models_dir=args.models_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        spread_juice=args.spread_juice,
        total_juice=args.total_juice,
        pricing_enabled=pricing_enabled,
        min_train_games=args.min_train_games,
        max_games=args.max_games,
        markets_filter=markets,
    )

    summary: Dict[str, List[Dict[str, float]]] = {}
    for market_key, bets in results.items():
        if not bets:
            continue
        rows: List[Dict[str, float]] = []
        for confidence_threshold in confidence_grid:
            for edge_threshold in edge_grid:
                row = _summarize(
                    bets=bets,
                    confidence_threshold=confidence_threshold,
                    edge_threshold=edge_threshold,
                    pricing_enabled=pricing_enabled,
                )
                if not row:
                    continue
                if row["n_bets"] < args.min_bets:
                    continue
                rows.append(row)

        ranked = _rank_results(rows, objective)
        summary[market_key] = ranked

        print("")
        print(f"{market_key.upper()} - top {args.top} (objective={objective})")
        for row in ranked[: args.top]:
            roi_str = "n/a" if row["roi"] is None else f"{row['roi']:+.3f}"
            profit_str = "n/a" if row["total_profit"] is None else f"{row['total_profit']:+.2f}"
            print(
                f"  conf>={row['confidence_threshold']:.2f} "
                f"edge>={row['edge_threshold']:.2f} "
                f"bets={row['n_bets']} "
                f"acc={row['accuracy']:.3f} "
                f"roi={roi_str} "
                f"profit={profit_str}"
            )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize confidence/edge thresholds.")
    parser.add_argument("--data", default="data/processed/training_data.csv")
    parser.add_argument("--models-dir", default="models/production")
    parser.add_argument("--markets", help="Comma-separated markets (fg_spread,fg_total,1h_spread,1h_total)")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--max-games", type=int)
    parser.add_argument("--min-train-games", type=int, default=100)
    parser.add_argument("--spread-juice", type=int)
    parser.add_argument("--total-juice", type=int)
    parser.add_argument("--no-pricing", action="store_true")
    parser.add_argument("--objective", choices=["roi", "accuracy", "total_profit"])
    parser.add_argument("--confidence-min", type=float, default=0.55)
    parser.add_argument("--confidence-max", type=float, default=0.80)
    parser.add_argument("--confidence-step", type=float, default=0.02)
    parser.add_argument("--edge-min", type=float, default=0.0)
    parser.add_argument("--edge-max", type=float, default=5.0)
    parser.add_argument("--edge-step", type=float, default=0.5)
    parser.add_argument("--min-bets", type=int, default=50)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--output-json")

    args = parser.parse_args()
    summary = run(args)

    if args.output_json:
        output_path = Path(args.output_json)
        payload = {
            "data": args.data,
            "models_dir": args.models_dir,
            "markets": _parse_markets(args.markets),
            "pricing_enabled": not args.no_pricing,
            "results": summary,
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
