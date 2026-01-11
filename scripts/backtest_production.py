#!/usr/bin/env python3
"""
Backtest Production Models on Historical Data

This script backtests the ACTUAL production models using the SAME feature
engineering pipeline used in live predictions. This ensures accurate comparison.

Usage:
    # Accuracy-only (no ROI/profit; avoids any pricing assumptions)
    python scripts/backtest_production.py --no-pricing

    # Explicit constant odds (still an assumption, but never silent)
    python scripts/backtest_production.py --spread-juice -110 --total-juice -110

    python scripts/backtest_production.py --start-date 2024-01-01 --end-date 2025-01-01 --no-pricing
"""
import argparse
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib

from src.modeling.features import FeatureEngineer
from src.config import settings
from src.modeling.season_utils import get_season_for_date

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# region agent log
_DEBUG_LOG_PATH = Path(__file__).resolve().parents[1] / ".cursor" / "debug.log"


def _agent_log(run_id: str, hypothesis_id: str, location: str, message: str, data: dict) -> None:
    try:
        payload = {
            "sessionId": "debug-session",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        _DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass

# endregion


@dataclass
class BetResult:
    """Result of a single bet."""
    date: str
    home_team: str
    away_team: str
    market: str
    side: str  # home/away or over/under
    line: float
    confidence: float
    predicted_value: float  # predicted margin or total
    actual_value: float  # actual margin or total
    won: bool
    odds: Optional[int]  # American odds (None if pricing disabled / unavailable)
    profit: Optional[float]  # Units profit/loss (None if pricing disabled / unavailable)


def load_production_model(model_path: Path) -> Tuple[object, List[str]]:
    """Load a production model and its feature list."""
    data = joblib.load(model_path)
    
    # Handle different model formats
    if isinstance(data, dict):
        model = data.get("pipeline") or data.get("model")
        features = data.get("feature_columns", [])
    else:
        model = data
        features = model.feature_names_in_.tolist() if hasattr(model, 'feature_names_in_') else []
    
    return model, features


def calculate_profit(won: bool, odds: int) -> float:
    """Calculate profit for a bet (risk 1 unit)."""
    if not won:
        return -1.0
    
    if odds < 0:
        return 100 / abs(odds)
    else:
        return odds / 100


def run_backtest(
    data_path: str,
    models_dir: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    spread_juice: Optional[int] = None,
    total_juice: Optional[int] = None,
    pricing_enabled: bool = True,
    min_train_games: int = 100,
    max_games: Optional[int] = None,
) -> Dict[str, List[BetResult]]:
    """
    Run walk-forward backtest using production models and feature engineering.
    
    Args:
        data_path: Path to training data CSV
        models_dir: Path to production models directory
        start_date: Start date for backtest (YYYY-MM-DD)
        end_date: End date for backtest (YYYY-MM-DD)
        spread_juice: Odds for spread bets (required if pricing_enabled)
        total_juice: Odds for total bets (required if pricing_enabled)
        pricing_enabled: If False, compute accuracy only (skip ROI/profit)
        min_train_games: Minimum games before making predictions
        
    Returns:
        Dictionary of market -> list of bet results
    """
    print("=" * 70)
    print("PRODUCTION MODEL BACKTEST")
    print("=" * 70)
    print(f"Data: {data_path}")
    print(f"Models: {models_dir}")
    if pricing_enabled:
        print(f"Spread juice: {spread_juice}, Total juice: {total_juice}")
    else:
        print("Pricing: DISABLED (accuracy-only; ROI/profit not computed)")
    print(f"Min training games: {min_train_games}")
    print()
    
    # Load data
    df = pd.read_csv(data_path, low_memory=False)
    
    # Ensure date column - prefer game_date as primary
    if "game_date" in df.columns:
        df["date"] = pd.to_datetime(df["game_date"], format="mixed", errors="coerce")
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")
    else:
        raise ValueError("No date column found in data!")
    
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # region agent log
    run_id = "run1"
    s2526 = df["date"] >= pd.to_datetime("2025-10-01")
    _agent_log(
        run_id=run_id,
        hypothesis_id="H1",
        location="scripts/backtest_production.py:run_backtest:after_load",
        message="Backtest input coverage for 1H scores/labels (2025-26 subset)",
        data={
            "rows": int(len(df)),
            "rows_2526": int(s2526.sum()),
            "home_1h_2526_non_null": int(df.loc[s2526, "home_1h"].notna().sum()) if "home_1h" in df.columns else None,
            "home_q1_2526_non_null": int(df.loc[s2526, "home_q1"].notna().sum()) if "home_q1" in df.columns else None,
            "1h_spread_covered_2526_non_null": int(df.loc[s2526, "1h_spread_covered"].notna().sum()) if "1h_spread_covered" in df.columns else None,
            "1h_total_over_2526_non_null": int(df.loc[s2526, "1h_total_over"].notna().sum()) if "1h_total_over" in df.columns else None,
        },
    )
    # endregion
    
    print(f"Loaded {len(df)} games from {df['date'].min()} to {df['date'].max()}")
    
    # IMPORTANT: Keep full dataset for historical reference
    # Only filter the games we PREDICT on, not the historical data
    full_historical_df = df.copy()
    
    # Filter date range for PREDICTION targets only
    predict_df = df.copy()
    if start_date:
        predict_df = predict_df[predict_df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        predict_df = predict_df[predict_df["date"] <= pd.to_datetime(end_date)]
    
    print(f"Games to predict: {len(predict_df)}")
    print(f"Historical games available: {len(full_historical_df)}")
    
    # Limit games if specified
    if max_games is not None and len(predict_df) > max_games:
        predict_df = predict_df.head(max_games)
        print(f"Limited to first {max_games} games for testing")
    
    # Standardize column names
    col_mapping = {
        "home": "home_team",
        "away": "away_team",
        "home_score": "home_score",
        "away_score": "away_score",
    }
    for old, new in col_mapping.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    
    # Ensure required columns
    required = ["date", "home_team", "away_team", "home_score", "away_score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Load production models
    models_dir = Path(models_dir)
    markets = {}
    
    if pricing_enabled:
        model_configs = [
            ("fg_spread", "fg_spread_model.joblib", "fg_spread_covered", "fg_spread_line", spread_juice),
            ("fg_total", "fg_total_model.joblib", "fg_total_over", "fg_total_line", total_juice),
            ("1h_spread", "1h_spread_model.joblib", "1h_spread_covered", "1h_spread_line", spread_juice),
            ("1h_total", "1h_total_model.joblib", "1h_total_over", "1h_total_line", total_juice),
        ]
    else:
        model_configs = [
            ("fg_spread", "fg_spread_model.joblib", "fg_spread_covered", "fg_spread_line", None),
            ("fg_total", "fg_total_model.joblib", "fg_total_over", "fg_total_line", None),
            ("1h_spread", "1h_spread_model.joblib", "1h_spread_covered", "1h_spread_line", None),
            ("1h_total", "1h_total_model.joblib", "1h_total_over", "1h_total_line", None),
        ]
    
    for market_key, model_file, label_col, line_col, odds in model_configs:
        model_path = models_dir / model_file
        if model_path.exists():
            model, features = load_production_model(model_path)
            markets[market_key] = {
                "model": model,
                "features": features,
                "label_col": label_col,
                "line_col": line_col,
                "odds": odds,
            }
            print(f"  Loaded {market_key}: {len(features)} features")
        else:
            print(f"  [WARN] Model not found: {model_path}")
    
    if not markets:
        raise ValueError("No models loaded!")
    
    # Initialize feature engineer
    fe = FeatureEngineer(lookback=10)
    
    # Walk-forward backtest
    results: Dict[str, List[BetResult]] = {k: [] for k in markets}
    
    print(f"\nRunning walk-forward backtest...")
    
    for idx, game in predict_df.iterrows():
        game_date = game["date"]
        
        # Get historical data (games before this one) from FULL dataset
        historical = full_historical_df[full_historical_df["date"] < game_date].copy()
        
        # Need minimum games for feature computation
        if len(historical) < min_train_games:
            continue
        
        # Compute features for this game
        try:
            base_features = fe.build_game_features(game, historical)
        except Exception as e:
            logger.debug(f"Could not compute features for {game_date}: {e}")
            continue
        
        if not base_features:
            continue
        
        # Make predictions for each market
        for market_key, config in markets.items():
            model = config["model"]
            model_features = config["features"]
            label_col = config["label_col"]
            line_col = config["line_col"]
            odds = config["odds"]
            
            # Check if we have the line
            if line_col not in game or pd.isna(game.get(line_col)):
                continue
            
            line = game[line_col]

            # IMPORTANT:
            # Production models expect generic feature names "spread_line" and "total_line"
            # for BOTH FG and 1H markets. We must map the correct period-specific lines
            # into those two feature slots.
            if market_key.startswith("1h_"):
                model_spread_line_col = "1h_spread_line"
                model_total_line_col = "1h_total_line"
            else:
                model_spread_line_col = "fg_spread_line"
                model_total_line_col = "fg_total_line"

            # Require both line features for model input (avoid silent 0-fill on key fields)
            if model_spread_line_col not in game or pd.isna(game.get(model_spread_line_col)):
                continue
            if model_total_line_col not in game or pd.isna(game.get(model_total_line_col)):
                continue
            
            # Check if we have the actual outcome
            if label_col not in game or pd.isna(game.get(label_col)):
                continue
            
            actual_label = int(game[label_col])

            # Build per-market feature frame with correct line mapping
            features = dict(base_features)
            features["spread_line"] = float(game[model_spread_line_col])
            features["total_line"] = float(game[model_total_line_col])
            # Also include raw column names (useful for debugging / feature parity checks)
            features[model_spread_line_col] = float(game[model_spread_line_col])
            features[model_total_line_col] = float(game[model_total_line_col])

            # Derived "line vs model" features expected by production models
            # (normally computed inside FeatureEngineer when game contains spread_line/total_line)
            if "predicted_margin" in features:
                features["spread_vs_predicted"] = features["predicted_margin"] - (-features["spread_line"])
            if "predicted_total" in features:
                features["total_vs_predicted"] = features["predicted_total"] - features["total_line"]

            feature_df = pd.DataFrame([features])
            
            # Prepare features for model
            # Fill missing features with 0 (model was trained this way)
            X = feature_df.reindex(columns=model_features, fill_value=0)
            
            # Get prediction
            try:
                proba = model.predict_proba(X)[0]
                pred_class = 1 if proba[1] > 0.5 else 0
                confidence = max(proba)
            except Exception as e:
                logger.debug(f"Prediction failed for {market_key}: {e}")
                continue
            
            # Determine actual value
            if "spread" in market_key:
                if "1h" in market_key:
                    # 1H margin
                    if "home_1h" in game and "away_1h" in game:
                        actual_value = game["home_1h"] - game["away_1h"]
                    elif "home_q1" in game and "home_q2" in game:
                        home_1h = game.get("home_q1", 0) + game.get("home_q2", 0)
                        away_1h = game.get("away_q1", 0) + game.get("away_q2", 0)
                        actual_value = home_1h - away_1h
                    else:
                        continue
                    predicted_value = features.get("predicted_margin_1h", 0)
                else:
                    actual_value = game["home_score"] - game["away_score"]
                    predicted_value = features.get("predicted_margin", 0)
                
                side = "home" if pred_class == 1 else "away"
            else:
                # Total
                if "1h" in market_key:
                    if "home_1h" in game and "away_1h" in game:
                        actual_value = game["home_1h"] + game["away_1h"]
                    elif "home_q1" in game and "home_q2" in game:
                        home_1h = game.get("home_q1", 0) + game.get("home_q2", 0)
                        away_1h = game.get("away_q1", 0) + game.get("away_q2", 0)
                        actual_value = home_1h + away_1h
                    else:
                        continue
                    predicted_value = features.get("predicted_total_1h", 0)
                else:
                    actual_value = game["home_score"] + game["away_score"]
                    predicted_value = features.get("predicted_total", 0)
                
                side = "over" if pred_class == 1 else "under"
            
            # Did we win?
            won = (pred_class == actual_label)
            profit = calculate_profit(won, odds) if odds is not None else None
            
            results[market_key].append(BetResult(
                date=str(game_date.date()),
                home_team=game["home_team"],
                away_team=game["away_team"],
                market=market_key,
                side=side,
                line=line,
                confidence=confidence,
                predicted_value=predicted_value,
                actual_value=actual_value,
                won=won,
                odds=odds,
                profit=profit,
            ))
        
        # Progress
        if len(results["fg_spread"]) % 50 == 0 and len(results["fg_spread"]) > 0:
            fg_bets = len(results["fg_spread"])
            print(f"  Made {fg_bets} FG spread bets so far...")
    
    return results


def get_season_from_date(date_str: str) -> str:
    """
    Get NBA season string from date.
    
    NBA seasons run Oct-Jun, so:
    - 2024-11-15 -> 2024-2025
    - 2025-02-01 -> 2024-2025
    """
    dt = pd.to_datetime(date_str, format="mixed", errors="coerce")
    if pd.isna(dt):
        return "unknown"
    season = get_season_for_date(dt.date())
    return season or "offseason"


def print_summary(results: Dict[str, List[BetResult]], output_json: Optional[str] = None) -> Dict:
    """Print backtest summary and optionally save to JSON."""
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)
    
    total_bets = 0
    total_wins = 0
    total_profit: Optional[float] = 0.0
    
    summary = {"markets": {}, "by_season": {}}
    
    for market, bets in results.items():
        if not bets:
            print(f"\n{market.upper()}: No bets")
            continue
        
        n_bets = len(bets)
        n_wins = sum(1 for b in bets if b.won)
        accuracy = n_wins / n_bets

        pricing_available = all(b.profit is not None for b in bets)
        profit: Optional[float] = sum(b.profit for b in bets) if pricing_available else None
        roi: Optional[float] = (profit / n_bets * 100) if profit is not None else None
        
        # High confidence subset
        high_conf_bets = [b for b in bets if b.confidence >= 0.60]
        if high_conf_bets:
            hc_wins = sum(1 for b in high_conf_bets if b.won)
            hc_acc = hc_wins / len(high_conf_bets)
            hc_pricing_available = all(b.profit is not None for b in high_conf_bets)
            hc_profit: Optional[float] = (
                sum(b.profit for b in high_conf_bets) if hc_pricing_available else None
            )
            hc_roi: Optional[float] = (
                hc_profit / len(high_conf_bets) * 100 if hc_profit is not None else None
            )
        else:
            hc_acc = 0
            hc_roi = None
            hc_wins = 0
        
        print(f"\n{market.upper()}")
        if roi is None or profit is None:
            print(f"  All bets:     {n_bets:4d} bets, {accuracy:5.1%} acc")
        else:
            print(f"  All bets:     {n_bets:4d} bets, {accuracy:5.1%} acc, {roi:+6.1f}% ROI, {profit:+7.1f} units")
        if high_conf_bets:
            if hc_roi is None or hc_profit is None:
                print(f"  High-conf(60%+): {len(high_conf_bets):4d} bets, {hc_acc:5.1%} acc")
            else:
                print(f"  High-conf(60%+): {len(high_conf_bets):4d} bets, {hc_acc:5.1%} acc, {hc_roi:+6.1f}% ROI, {hc_profit:+7.1f} units")
        
        summary["markets"][market] = {
            "n_bets": n_bets,
            "wins": n_wins,
            "accuracy": accuracy,
            "roi": roi,
            "profit": profit,
            "high_conf": {
                "n_bets": len(high_conf_bets),
                "wins": hc_wins,
                "accuracy": hc_acc if high_conf_bets else 0,
                "roi": hc_roi,
            }
        }
        
        total_bets += n_bets
        total_wins += n_wins
        if total_profit is not None and profit is not None:
            total_profit += profit
        else:
            total_profit = None
    
    print("\n" + "-" * 70)
    if total_bets > 0:
        if total_profit is None:
            print(f"TOTAL: {total_bets} bets, {total_wins/total_bets:.1%} accuracy")
        else:
            print(f"TOTAL: {total_bets} bets, {total_wins/total_bets:.1%} accuracy, {total_profit/total_bets*100:+.1f}% ROI, {total_profit:+.1f} units")
    
    # === BY SEASON BREAKDOWN ===
    print("\n" + "=" * 70)
    print("RESULTS BY SEASON")
    print("=" * 70)
    
    # Collect all seasons
    all_bets = []
    for market, bets in results.items():
        for bet in bets:
            bet_dict = {
                "date": bet.date,
                "market": bet.market,
                "won": bet.won,
                "profit": bet.profit,
                "confidence": bet.confidence,
            }
            all_bets.append(bet_dict)
    
    if all_bets:
        # Group by season
        season_data: Dict[str, Dict[str, List]] = {}
        for bet in all_bets:
            season = get_season_from_date(bet["date"])
            if season not in season_data:
                season_data[season] = {"bets": [], "markets": {}}
            season_data[season]["bets"].append(bet)
        
        # Print header
        print(f"\n{'Season':<12} | {'FG Spread':^15} | {'FG Total':^15} | {'1H Spread':^15} | {'1H Total':^15}")
        print("-" * 80)
        
        for season in sorted(season_data.keys()):
            bets_in_season = season_data[season]["bets"]
            
            # Calculate per-market stats
            market_stats = {}
            for market in ["fg_spread", "fg_total", "1h_spread", "1h_total"]:
                market_bets = [b for b in bets_in_season if b["market"] == market]
                if market_bets:
                    wins = sum(1 for b in market_bets if b["won"])
                    acc = wins / len(market_bets) * 100
                    market_stats[market] = f"{acc:.1f}% ({len(market_bets)})"
                else:
                    market_stats[market] = "--"
            
            print(f"{season:<12} | {market_stats.get('fg_spread', '--'):^15} | {market_stats.get('fg_total', '--'):^15} | {market_stats.get('1h_spread', '--'):^15} | {market_stats.get('1h_total', '--'):^15}")
            
            # Store in summary
            summary["by_season"][season] = {
                "total_bets": len(bets_in_season),
                "wins": sum(1 for b in bets_in_season if b["won"]),
                "profit": (
                    sum(b["profit"] for b in bets_in_season)
                    if all(b.get("profit") is not None for b in bets_in_season)
                    else None
                ),
            }
            for market in ["fg_spread", "fg_total", "1h_spread", "1h_total"]:
                market_bets = [b for b in bets_in_season if b["market"] == market]
                if market_bets:
                    summary["by_season"][season][market] = {
                        "n_bets": len(market_bets),
                        "accuracy": sum(1 for b in market_bets if b["won"]) / len(market_bets),
                    }
    
    # Save to JSON if requested
    if output_json:
        import json
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to: {output_json}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Backtest production models on historical data")
    parser.add_argument(
        "--data",
        default="data/processed/training_data_all_seasons.csv",
        help="Path to training data CSV (default: merged all seasons file)",
    )
    parser.add_argument(
        "--models-dir",
        default="models/production",
        help="Path to production models directory",
    )
    parser.add_argument(
        "--start-date",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--no-pricing",
        action="store_true",
        help="Accuracy-only mode (do not compute ROI/profit; avoids any odds assumptions)",
    )
    parser.add_argument(
        "--spread-juice",
        type=int,
        default=None,
        help="Spread bet odds (American). Required unless --no-pricing is set.",
    )
    parser.add_argument(
        "--total-juice",
        type=int,
        default=None,
        help="Total bet odds (American). Required unless --no-pricing is set.",
    )
    parser.add_argument(
        "--min-train",
        type=int,
        default=100,
        help="Minimum historical games before predicting",
    )
    parser.add_argument(
        "--output-json",
        default="data/backtest_results/production_backtest_results.json",
        help="Path to save JSON results",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Maximum games to process (for quick testing)",
    )
    args = parser.parse_args()

    pricing_enabled = not args.no_pricing
    if pricing_enabled and (args.spread_juice is None or args.total_juice is None):
        raise ValueError(
            "Pricing is enabled but odds were not provided. "
            "Either pass --spread-juice and --total-juice explicitly, "
            "or run with --no-pricing for accuracy-only mode."
        )
    
    results = run_backtest(
        data_path=args.data,
        models_dir=args.models_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        spread_juice=args.spread_juice,
        total_juice=args.total_juice,
        pricing_enabled=pricing_enabled,
        min_train_games=args.min_train,
        max_games=args.max_games,
    )
    
    print_summary(results, output_json=args.output_json)


if __name__ == "__main__":
    main()
