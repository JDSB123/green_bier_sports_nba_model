#!/usr/bin/env python3
"""
Backtest Production Models on Historical Data

This script backtests the ACTUAL production models using the SAME feature
engineering pipeline used in live predictions. This ensures accurate comparison.

Usage:
    python scripts/backtest_production.py
    python scripts/backtest_production.py --start-date 2024-01-01 --end-date 2025-01-01
"""
import argparse
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib

from src.modeling.features import FeatureEngineer
from src.config import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


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
    odds: int  # American odds
    profit: float  # Units profit/loss


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
    spread_juice: int = -110,
    total_juice: int = -110,
    min_train_games: int = 100,
) -> Dict[str, List[BetResult]]:
    """
    Run walk-forward backtest using production models and feature engineering.
    
    Args:
        data_path: Path to training data CSV
        models_dir: Path to production models directory
        start_date: Start date for backtest (YYYY-MM-DD)
        end_date: End date for backtest (YYYY-MM-DD)
        spread_juice: Odds for spread bets (user must specify)
        total_juice: Odds for total bets (user must specify)
        min_train_games: Minimum games before making predictions
        
    Returns:
        Dictionary of market -> list of bet results
    """
    print("=" * 70)
    print("PRODUCTION MODEL BACKTEST")
    print("=" * 70)
    print(f"Data: {data_path}")
    print(f"Models: {models_dir}")
    print(f"Spread juice: {spread_juice}, Total juice: {total_juice}")
    print(f"Min training games: {min_train_games}")
    print()
    
    # Load data
    df = pd.read_csv(data_path, low_memory=False)
    
    # Ensure date column
    date_col = "game_date" if "game_date" in df.columns else "date"
    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    print(f"Loaded {len(df)} games from {df['date'].min()} to {df['date'].max()}")
    
    # Filter date range
    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["date"] <= pd.to_datetime(end_date)]
    
    print(f"After date filter: {len(df)} games")
    
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
    
    model_configs = [
        ("fg_spread", "fg_spread_model.joblib", "spread_covered", "fg_spread_line", spread_juice),
        ("fg_total", "fg_total_model.joblib", "total_over", "fg_total_line", total_juice),
        ("1h_spread", "1h_spread_model.joblib", "1h_spread_covered", "1h_spread_line", spread_juice),
        ("1h_total", "1h_total_model.joblib", "1h_total_over", "1h_total_line", total_juice),
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
    
    for idx, game in df.iterrows():
        game_date = game["date"]
        
        # Get historical data (games before this one)
        historical = df[df["date"] < game_date].copy()
        
        # Need minimum games for feature computation
        if len(historical) < min_train_games:
            continue
        
        # Compute features for this game
        try:
            features = fe.build_game_features(game, historical)
        except Exception as e:
            logger.debug(f"Could not compute features for {game_date}: {e}")
            continue
        
        if not features:
            continue
        
        # Add line features
        line_col_map = {
            "fg_spread_line": "spread_line",
            "fg_total_line": "total_line",
            "1h_spread_line": "1h_spread_line",
            "1h_total_line": "1h_total_line",
        }
        
        for data_col, feat_col in line_col_map.items():
            if data_col in game and pd.notna(game.get(data_col)):
                features[feat_col] = game[data_col]
                features[data_col] = game[data_col]  # Also add with original name
        
        # Create feature DataFrame
        feature_df = pd.DataFrame([features])
        
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
            
            # Check if we have the actual outcome
            if label_col not in game or pd.isna(game.get(label_col)):
                continue
            
            actual_label = int(game[label_col])
            
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
            profit = calculate_profit(won, odds)
            
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
        if idx % 500 == 0 and idx > 0:
            print(f"  Processed {idx} games...")
    
    return results


def print_summary(results: Dict[str, List[BetResult]]) -> None:
    """Print backtest summary."""
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)
    
    total_bets = 0
    total_wins = 0
    total_profit = 0.0
    
    for market, bets in results.items():
        if not bets:
            print(f"\n{market.upper()}: No bets")
            continue
        
        n_bets = len(bets)
        n_wins = sum(1 for b in bets if b.won)
        profit = sum(b.profit for b in bets)
        accuracy = n_wins / n_bets
        roi = profit / n_bets * 100
        
        # High confidence subset
        high_conf_bets = [b for b in bets if b.confidence >= 0.60]
        if high_conf_bets:
            hc_wins = sum(1 for b in high_conf_bets if b.won)
            hc_profit = sum(b.profit for b in high_conf_bets)
            hc_acc = hc_wins / len(high_conf_bets)
            hc_roi = hc_profit / len(high_conf_bets) * 100
        else:
            hc_acc = hc_roi = 0
            hc_wins = 0
        
        print(f"\n{market.upper()}")
        print(f"  All bets:     {n_bets:4d} bets, {accuracy:5.1%} acc, {roi:+6.1f}% ROI, {profit:+7.1f} units")
        if high_conf_bets:
            print(f"  High-conf(60%+): {len(high_conf_bets):4d} bets, {hc_acc:5.1%} acc, {hc_roi:+6.1f}% ROI, {hc_profit:+7.1f} units")
        
        total_bets += n_bets
        total_wins += n_wins
        total_profit += profit
    
    print("\n" + "-" * 70)
    if total_bets > 0:
        print(f"TOTAL: {total_bets} bets, {total_wins/total_bets:.1%} accuracy, {total_profit/total_bets*100:+.1f}% ROI, {total_profit:+.1f} units")


def main():
    parser = argparse.ArgumentParser(description="Backtest production models on historical data")
    parser.add_argument(
        "--data",
        default="data/processed/training_data_complete_2023.csv",
        help="Path to training data CSV",
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
        "--spread-juice",
        type=int,
        default=-110,
        help="Spread bet odds (default: -110)",
    )
    parser.add_argument(
        "--total-juice",
        type=int,
        default=-110,
        help="Total bet odds (default: -110)",
    )
    parser.add_argument(
        "--min-train",
        type=int,
        default=100,
        help="Minimum historical games before predicting",
    )
    args = parser.parse_args()
    
    results = run_backtest(
        data_path=args.data,
        models_dir=args.models_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        spread_juice=args.spread_juice,
        total_juice=args.total_juice,
        min_train_games=args.min_train,
    )
    
    print_summary(results)


if __name__ == "__main__":
    main()
