#!/usr/bin/env python3
"""
Log model performance to model_pack.json for version control.

Usage:
    python scripts/log_model_performance.py --market fg_moneyline --accuracy 0.68 --roi 0.30 --predictions 50
    python scripts/log_model_performance.py --from-tracker  # Auto-calculate from pick tracker
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_PACK_PATH = PROJECT_ROOT / "models" / "production" / "model_pack.json"


def load_model_pack() -> dict:
    """Load the model pack JSON."""
    if not MODEL_PACK_PATH.exists():
        raise FileNotFoundError(f"Model pack not found at {MODEL_PACK_PATH}")
    return json.loads(MODEL_PACK_PATH.read_text())


def save_model_pack(data: dict) -> None:
    """Save the model pack JSON."""
    MODEL_PACK_PATH.write_text(json.dumps(data, indent=2))
    print(f"Updated {MODEL_PACK_PATH}")


def log_performance(
    market: str,
    accuracy: float,
    roi: float,
    predictions: int,
    date: str | None = None,
    notes: str | None = None,
) -> None:
    """Log a performance entry to the model pack."""
    model_pack = load_model_pack()

    if "performance_history" not in model_pack:
        model_pack["performance_history"] = []

    entry = {
        "date": date or datetime.now().strftime("%Y-%m-%d"),
        "market": market,
        "accuracy": round(accuracy, 4),
        "roi": round(roi, 4),
        "predictions": predictions,
    }

    if notes:
        entry["notes"] = notes

    model_pack["performance_history"].append(entry)

    # Keep only last 100 entries per market to prevent unbounded growth
    market_entries = [e for e in model_pack["performance_history"] if e["market"] == market]
    if len(market_entries) > 100:
        # Remove oldest entries for this market
        to_remove = len(market_entries) - 100
        model_pack["performance_history"] = [
            e for i, e in enumerate(model_pack["performance_history"])
            if e["market"] != market or i >= to_remove
        ]

    save_model_pack(model_pack)
    print(f"Logged: {market} - {accuracy:.1%} accuracy, {roi:+.1%} ROI ({predictions} picks)")


def calculate_from_tracker() -> dict:
    """Calculate performance from the pick tracker."""
    try:
        from src.tracking.tracker import PickTracker

        tracker = PickTracker()
        results = {}

        for market in [
            "fg_spread", "fg_total", "fg_moneyline",
            "1h_spread", "1h_total", "1h_moneyline",
            "q1_spread", "q1_total", "q1_moneyline",
        ]:
            picks = tracker.get_picks_by_market(market)
            resolved = [p for p in picks if p.get("result") in ["win", "loss", "push"]]

            if len(resolved) >= 10:
                wins = sum(1 for p in resolved if p["result"] == "win")
                losses = sum(1 for p in resolved if p["result"] == "loss")

                if wins + losses > 0:
                    accuracy = wins / (wins + losses)
                    # Simplified ROI calculation assuming -110 odds
                    roi = (wins * 0.909 - losses) / (wins + losses)
                    results[market] = {
                        "accuracy": accuracy,
                        "roi": roi,
                        "predictions": wins + losses,
                    }

        return results
    except Exception as e:
        print(f"Error calculating from tracker: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Log model performance to model_pack.json")
    parser.add_argument("--market", type=str, help="Market name (e.g., fg_moneyline)")
    parser.add_argument("--accuracy", type=float, help="Accuracy (0-1)")
    parser.add_argument("--roi", type=float, help="ROI (-1 to +inf)")
    parser.add_argument("--predictions", type=int, help="Number of predictions")
    parser.add_argument("--date", type=str, help="Date (YYYY-MM-DD), defaults to today")
    parser.add_argument("--notes", type=str, help="Optional notes")
    parser.add_argument("--from-tracker", action="store_true", help="Auto-calculate from pick tracker")
    parser.add_argument("--show-history", action="store_true", help="Show performance history")

    args = parser.parse_args()

    if args.show_history:
        model_pack = load_model_pack()
        history = model_pack.get("performance_history", [])
        if not history:
            print("No performance history recorded yet.")
        else:
            print(f"\n{'Date':<12} {'Market':<15} {'Accuracy':<10} {'ROI':<10} {'Picks':<8}")
            print("-" * 55)
            for entry in history[-20:]:  # Show last 20
                print(f"{entry['date']:<12} {entry['market']:<15} {entry['accuracy']:.1%}      {entry['roi']:+.1%}      {entry['predictions']}")
        return

    if args.from_tracker:
        results = calculate_from_tracker()
        if not results:
            print("No resolved picks found in tracker.")
            return

        for market, stats in results.items():
            log_performance(
                market=market,
                accuracy=stats["accuracy"],
                roi=stats["roi"],
                predictions=stats["predictions"],
                notes="Auto-calculated from pick tracker",
            )
        return

    if not all([args.market, args.accuracy is not None, args.roi is not None, args.predictions]):
        parser.print_help()
        sys.exit(1)

    log_performance(
        market=args.market,
        accuracy=args.accuracy,
        roi=args.roi,
        predictions=args.predictions,
        date=args.date,
        notes=args.notes,
    )


if __name__ == "__main__":
    main()
