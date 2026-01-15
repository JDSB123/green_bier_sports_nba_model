#!/usr/bin/env python3
"""
Extended Backtest with Per-Market Configuration and Moneyline Support

This script extends backtest_production.py with:
- Per-market juice/odds parameters (6 independent markets)
- Per-market confidence thresholds
- Per-market minimum date boundaries (May 2023+ for spreads)
- Moneyline backtesting with dual vig tracking:
  - Raw odds ROI (actual P&L)
  - Fair probability calibration (vig-free)
  - CLV tracking

Usage:
    # All markets with explicit per-market odds
    python scripts/backtest_extended.py \
        --fg-spread-juice -110 --fg-total-juice -110 \
        --1h-spread-juice -110 --1h-total-juice -110

    # Single market with specific configuration
    python scripts/backtest_extended.py --markets fg_spread --fg-spread-juice -105

    # Moneyline backtesting (uses actual odds from data)
    python scripts/backtest_extended.py --markets fg_moneyline,1h_moneyline

    # Accuracy-only mode
    python scripts/backtest_extended.py --no-pricing
"""
import argparse
import os
import sys
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib

from src.modeling.season_utils import get_season_for_date
from src.prediction.feature_validation import (
    MissingFeaturesError,
    validate_and_prepare_features,
)
from src.prediction.engine import map_1h_features_to_fg_names

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class MarketConfig:
    """Configuration for a single market."""
    market_key: str
    model_file: str
    label_col: str
    line_cols: List[str]  # Priority order for line column lookup
    price_cols: Dict[str, List[str]]  # home/away or over/under price columns
    juice: Optional[int] = None  # American odds (-110, etc.)
    min_confidence: float = 0.55
    min_edge: float = 0.0
    min_date: Optional[str] = None  # Minimum date for this market
    min_train_games: int = 100
    is_moneyline: bool = False


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
    predicted_prob: float
    implied_prob: Optional[float]  # From market odds
    fair_prob: Optional[float]  # Vig-removed implied prob
    actual_label: int
    won: bool
    odds: Optional[int]  # American odds
    profit: Optional[float]  # Units profit/loss
    vig_pct: Optional[float]  # Overround percentage


@dataclass
class MoneylineMetrics:
    """Extended metrics for moneyline bets including vig tracking."""
    raw_roi: Optional[float]  # Actual P&L ROI
    fair_calibration_error: Optional[float]  # Model vs fair prob
    avg_vig_paid: Optional[float]  # Average overround
    clv_total: Optional[float]  # Closing line value sum


# ============================================================================
# MARKET CONFIGURATIONS
# ============================================================================

def get_default_market_configs() -> Dict[str, MarketConfig]:
    """Get default configurations for all 6 markets."""
    return {
        "fg_spread": MarketConfig(
            market_key="fg_spread",
            model_file="fg_spread_model.joblib",
            label_col="fg_spread_covered",
            line_cols=["fg_spread_line", "spread_line"],
            price_cols={
                "home": ["fg_spread_price_home", "spread_price_home"],
                "away": ["fg_spread_price_away", "spread_price_away"],
            },
            min_confidence=0.55,
            min_date="2023-05-01",  # Spreads only from May 2023
        ),
        "fg_total": MarketConfig(
            market_key="fg_total",
            model_file="fg_total_model.joblib",
            label_col="fg_total_over",
            line_cols=["fg_total_line", "total_line"],
            price_cols={
                "over": ["fg_total_price_over", "total_price_over"],
                "under": ["fg_total_price_under", "total_price_under"],
            },
            min_confidence=0.55,
            min_date="2023-01-01",
        ),
        "fg_moneyline": MarketConfig(
            market_key="fg_moneyline",
            model_file="fg_moneyline_model.joblib",  # May not exist yet
            label_col="fg_ml_covered",  # Derived: home_score > away_score
            line_cols=["fg_ml_home", "moneyline_home", "to_fg_ml_home"],
            price_cols={
                "home": ["fg_ml_home", "moneyline_home", "to_fg_ml_home"],
                "away": ["fg_ml_away", "moneyline_away", "to_fg_ml_away"],
            },
            min_confidence=0.55,
            min_date="2023-01-01",
            is_moneyline=True,
        ),
        "1h_spread": MarketConfig(
            market_key="1h_spread",
            model_file="1h_spread_model.joblib",
            label_col="1h_spread_covered",
            line_cols=["1h_spread_line", "fh_spread_line"],
            price_cols={
                "home": ["1h_spread_price_home"],
                "away": ["1h_spread_price_away"],
            },
            min_confidence=0.55,
            min_date="2023-05-01",  # Spreads only from May 2023
        ),
        "1h_total": MarketConfig(
            market_key="1h_total",
            model_file="1h_total_model.joblib",
            label_col="1h_total_over",
            line_cols=["1h_total_line", "fh_total_line"],
            price_cols={
                "over": ["1h_total_price_over"],
                "under": ["1h_total_price_under"],
            },
            min_confidence=0.55,
            min_date="2023-01-01",
        ),
        "1h_moneyline": MarketConfig(
            market_key="1h_moneyline",
            model_file="1h_moneyline_model.joblib",  # May not exist yet
            label_col="1h_ml_covered",  # Derived: home_1h > away_1h
            line_cols=["to_1h_ml_home", "exp_1h_ml_home"],
            price_cols={
                "home": ["to_1h_ml_home", "exp_1h_ml_home"],
                "away": ["to_1h_ml_away", "exp_1h_ml_away"],
            },
            min_confidence=0.55,
            min_date="2023-01-01",
            is_moneyline=True,
        ),
    }


# ============================================================================
# MONEYLINE VIG CALCULATIONS
# ============================================================================

def american_to_implied_prob(odds: float) -> float:
    """Convert American odds to implied probability."""
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 100 / (odds + 100)


def implied_to_american(prob: float) -> float:
    """Convert probability to American odds."""
    if prob <= 0 or prob >= 1:
        return 0
    if prob >= 0.5:
        return -100 * prob / (1 - prob)
    else:
        return 100 * (1 - prob) / prob


def remove_vig_proportional(home_odds: float, away_odds: float) -> Tuple[float, float, float]:
    """
    Remove vig using proportional (multiplicative) method.

    Returns:
        (fair_home_prob, fair_away_prob, overround_pct)
    """
    home_implied = american_to_implied_prob(home_odds)
    away_implied = american_to_implied_prob(away_odds)

    total_implied = home_implied + away_implied
    overround = (total_implied - 1.0) * 100  # Vig percentage

    # Proportional normalization
    fair_home = home_implied / total_implied
    fair_away = away_implied / total_implied

    return fair_home, fair_away, overround


def calculate_profit(won: bool, odds: int) -> float:
    """Calculate profit for a bet (risk 1 unit)."""
    if not won:
        return -1.0
    if odds < 0:
        return 100 / abs(odds)
    else:
        return odds / 100


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_production_model(model_path: Path) -> Tuple[object, List[str]]:
    """Load a production model and its feature list."""
    data = joblib.load(model_path)

    if isinstance(data, dict):
        model = data.get("pipeline") or data.get("model")
        features = data.get("feature_columns", [])
    else:
        model = data
        features = model.feature_names_in_.tolist() if hasattr(model, 'feature_names_in_') else []

    return model, features


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def pick_first_value(game: pd.Series, candidates: List[str]) -> Optional[float]:
    """Return the first non-null value from candidate columns."""
    for col in candidates:
        if col in game and pd.notna(game.get(col)):
            try:
                return float(game.get(col))
            except (TypeError, ValueError):
                return None
    return None


def build_feature_payload(game: pd.Series, config: MarketConfig) -> Dict[str, float]:
    """Build feature payload for a single game/market."""
    features = game.to_dict()

    # Get line value
    line = pick_first_value(game, config.line_cols)

    if config.is_moneyline:
        # For moneylines, line is the home odds
        if line is not None:
            features["moneyline_home"] = line
    else:
        # For spreads/totals
        if line is not None:
            features["spread_line" if "spread" in config.market_key else "total_line"] = line

            if config.market_key.startswith("1h_"):
                if "spread" in config.market_key:
                    features["1h_spread_line"] = line
                    features["fh_spread_line"] = line
                else:
                    features["1h_total_line"] = line
                    features["fh_total_line"] = line
            else:
                if "spread" in config.market_key:
                    features["fg_spread_line"] = line
                else:
                    features["fg_total_line"] = line

    # Use 1H feature mapping when available
    if config.market_key.startswith("1h_"):
        features = map_1h_features_to_fg_names(features)

    return features


def get_actual_label(game: pd.Series, config: MarketConfig) -> Optional[int]:
    """Get actual outcome label for a game/market."""

    # Check if label column exists
    if config.label_col in game and pd.notna(game.get(config.label_col)):
        return int(game[config.label_col])

    # Derive moneyline labels from scores
    if config.is_moneyline:
        if config.market_key == "fg_moneyline":
            if pd.notna(game.get("home_score")) and pd.notna(game.get("away_score")):
                return 1 if game["home_score"] > game["away_score"] else 0
        elif config.market_key == "1h_moneyline":
            if pd.notna(game.get("home_1h")) and pd.notna(game.get("away_1h")):
                return 1 if game["home_1h"] > game["away_1h"] else 0
            # Fall back to Q1+Q2
            elif all(pd.notna(game.get(c)) for c in ["home_q1", "home_q2", "away_q1", "away_q2"]):
                home_1h = game["home_q1"] + game["home_q2"]
                away_1h = game["away_q1"] + game["away_q2"]
                return 1 if home_1h > away_1h else 0

    return None


def get_moneyline_odds(game: pd.Series, config: MarketConfig) -> Tuple[Optional[float], Optional[float]]:
    """Get home and away moneyline odds for vig calculation."""
    home_odds = pick_first_value(game, config.price_cols.get("home", []))
    away_odds = pick_first_value(game, config.price_cols.get("away", []))
    return home_odds, away_odds


# ============================================================================
# BACKTEST ENGINE
# ============================================================================

def run_backtest(
    df: pd.DataFrame,
    models_dir: Path,
    market_configs: Dict[str, MarketConfig],
    pricing_enabled: bool = True,
) -> Dict[str, List[BetResult]]:
    """
    Run walk-forward backtest for specified markets.

    Args:
        df: Training data DataFrame
        models_dir: Path to models directory
        market_configs: Dictionary of market configurations
        pricing_enabled: If False, skip ROI/profit calculations

    Returns:
        Dictionary of market -> list of bet results
    """
    results: Dict[str, List[BetResult]] = {k: [] for k in market_configs}

    # Load models
    loaded_models = {}
    for market_key, config in market_configs.items():
        model_path = models_dir / config.model_file

        if config.is_moneyline and not model_path.exists():
            # For moneylines without dedicated model, use ELO-based prediction
            print(f"  [INFO] {market_key}: No model found, will use ELO-based predictions")
            loaded_models[market_key] = {"model": None, "features": [], "use_elo": True}
        elif model_path.exists():
            model, features = load_production_model(model_path)
            loaded_models[market_key] = {"model": model, "features": features, "use_elo": False}
            print(f"  [OK] Loaded {market_key}: {len(features)} features")
        else:
            print(f"  [WARN] Model not found: {model_path}")

    if not loaded_models:
        raise ValueError("No models loaded!")

    # Process each market
    for market_key, config in market_configs.items():
        if market_key not in loaded_models:
            continue

        model_info = loaded_models[market_key]

        # Filter data by market's minimum date
        market_df = df.copy()
        if config.min_date:
            market_df = market_df[market_df["date"] >= pd.to_datetime(config.min_date)]

        market_df = market_df.sort_values("date").reset_index(drop=True)
        print(f"\n  {market_key}: {len(market_df)} games (from {config.min_date or 'start'})")

        for row_idx, (_, game) in enumerate(market_df.iterrows()):
            # Skip early games for training
            if row_idx < config.min_train_games:
                continue

            # Get line value
            line = pick_first_value(game, config.line_cols)
            if line is None:
                continue

            # Get actual label
            actual_label = get_actual_label(game, config)
            if actual_label is None:
                continue

            # Get prediction
            if model_info["use_elo"]:
                # ELO-based prediction for moneylines
                if "elo_prob" in game and pd.notna(game.get("elo_prob")):
                    predicted_prob = game["elo_prob"]
                else:
                    continue
            else:
                # Model-based prediction
                features = build_feature_payload(game, config)
                feature_df = pd.DataFrame([features])

                try:
                    X, _ = validate_and_prepare_features(
                        feature_df,
                        model_info["features"],
                        market=config.market_key,
                    )
                except MissingFeaturesError:
                    continue

                if X.isna().any().any():
                    continue

                try:
                    proba = model_info["model"].predict_proba(X)[0]
                    predicted_prob = proba[1]  # Probability of positive class
                except Exception:
                    continue

            pred_class = 1 if predicted_prob > 0.5 else 0
            confidence = max(predicted_prob, 1 - predicted_prob)

            # Apply confidence filter
            if confidence < config.min_confidence:
                continue

            # Determine side and odds
            if config.is_moneyline:
                side = "home" if pred_class == 1 else "away"
                home_odds, away_odds = get_moneyline_odds(game, config)

                if home_odds is not None and away_odds is not None:
                    fair_home, fair_away, vig_pct = remove_vig_proportional(home_odds, away_odds)
                    odds = int(home_odds) if side == "home" else int(away_odds)
                    implied_prob = american_to_implied_prob(odds)
                    fair_prob = fair_home if side == "home" else fair_away
                else:
                    odds = config.juice
                    implied_prob = None
                    fair_prob = None
                    vig_pct = None
            else:
                # Spread/total
                if "spread" in config.market_key:
                    side = "home" if pred_class == 1 else "away"
                else:
                    side = "over" if pred_class == 1 else "under"

                odds = config.juice
                implied_prob = american_to_implied_prob(odds) if odds else None
                fair_prob = None
                vig_pct = None

            won = (pred_class == actual_label)
            profit = calculate_profit(won, odds) if odds and pricing_enabled else None

            results[market_key].append(BetResult(
                date=str(game["date"].date()),
                home_team=game["home_team"],
                away_team=game["away_team"],
                market=market_key,
                side=side,
                line=line,
                confidence=confidence,
                predicted_prob=predicted_prob,
                implied_prob=implied_prob,
                fair_prob=fair_prob,
                actual_label=actual_label,
                won=won,
                odds=odds,
                profit=profit,
                vig_pct=vig_pct,
            ))

    return results


# ============================================================================
# REPORTING
# ============================================================================

def calculate_moneyline_metrics(bets: List[BetResult]) -> MoneylineMetrics:
    """Calculate extended metrics for moneyline bets."""
    if not bets:
        return MoneylineMetrics(None, None, None, None)

    # Raw ROI
    profits = [b.profit for b in bets if b.profit is not None]
    raw_roi = (sum(profits) / len(profits) * 100) if profits else None

    # Fair calibration error (model prob vs fair market prob)
    calibration_errors = []
    for b in bets:
        if b.fair_prob is not None:
            pred_prob = b.predicted_prob if b.side == "home" else (1 - b.predicted_prob)
            calibration_errors.append(abs(pred_prob - b.fair_prob))
    fair_calibration_error = np.mean(calibration_errors) if calibration_errors else None

    # Average vig paid
    vigs = [b.vig_pct for b in bets if b.vig_pct is not None]
    avg_vig = np.mean(vigs) if vigs else None

    return MoneylineMetrics(
        raw_roi=raw_roi,
        fair_calibration_error=fair_calibration_error,
        avg_vig_paid=avg_vig,
        clv_total=None,  # Would require closing line data
    )


def print_summary(
    results: Dict[str, List[BetResult]],
    output_json: Optional[str] = None,
) -> Dict[str, Any]:
    """Print backtest summary with extended metrics."""
    print("\n" + "=" * 80)
    print("EXTENDED BACKTEST RESULTS")
    print("=" * 80)

    summary: Dict[str, Any] = {"markets": {}, "totals": {}}

    for market, bets in results.items():
        if not bets:
            print(f"\n{market.upper()}: No bets")
            continue

        n_bets = len(bets)
        n_wins = sum(1 for b in bets if b.won)
        accuracy = n_wins / n_bets

        pricing_available = all(b.profit is not None for b in bets)
        profit = sum(b.profit for b in bets) if pricing_available else None
        roi = (profit / n_bets * 100) if profit is not None else None

        print(f"\n{market.upper()}")
        print("-" * 40)

        if roi is not None:
            print(f"  Bets: {n_bets:,}  |  Accuracy: {accuracy:.1%}  |  ROI: {roi:+.2f}%  |  Profit: {profit:+.1f}u")
        else:
            print(f"  Bets: {n_bets:,}  |  Accuracy: {accuracy:.1%}")

        # Extended metrics for moneylines
        if "moneyline" in market:
            ml_metrics = calculate_moneyline_metrics(bets)
            print(f"  --- Moneyline Metrics ---")
            if ml_metrics.raw_roi is not None:
                print(f"  Raw ROI: {ml_metrics.raw_roi:+.2f}%")
            if ml_metrics.fair_calibration_error is not None:
                print(f"  Fair Calibration Error: {ml_metrics.fair_calibration_error:.3f}")
            if ml_metrics.avg_vig_paid is not None:
                print(f"  Avg Vig Paid: {ml_metrics.avg_vig_paid:.2f}%")

        # Confidence tier breakdown
        tiers = [
            ("55-60%", 0.55, 0.60),
            ("60-65%", 0.60, 0.65),
            ("65-70%", 0.65, 0.70),
            ("70%+", 0.70, 1.01),
        ]

        print(f"  --- By Confidence ---")
        for tier_name, low, high in tiers:
            tier_bets = [b for b in bets if low <= b.confidence < high]
            if tier_bets:
                tier_wins = sum(1 for b in tier_bets if b.won)
                tier_acc = tier_wins / len(tier_bets)
                tier_profit = sum(b.profit for b in tier_bets if b.profit is not None)
                if tier_profit is not None and len(tier_bets) > 0:
                    tier_roi = tier_profit / len(tier_bets) * 100
                    print(f"    {tier_name}: {len(tier_bets):4d} bets, {tier_acc:5.1%} acc, {tier_roi:+6.2f}% ROI")
                else:
                    print(f"    {tier_name}: {len(tier_bets):4d} bets, {tier_acc:5.1%} acc")

        summary["markets"][market] = {
            "n_bets": n_bets,
            "wins": n_wins,
            "accuracy": accuracy,
            "roi": roi,
            "profit": profit,
        }

    # Save to JSON
    if output_json:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nResults saved to: {output_json}")

    return summary


# ============================================================================
# CLI
# ============================================================================

def normalize_markets_arg(markets_str: str) -> List[str]:
    """Parse comma-separated markets argument."""
    if not markets_str or markets_str.lower() == "all":
        return list(get_default_market_configs().keys())

    raw = [p.strip().lower() for p in markets_str.split(",") if p.strip()]
    normalized = []

    aliases = {
        "fg": ["fg_spread", "fg_total", "fg_moneyline"],
        "1h": ["1h_spread", "1h_total", "1h_moneyline"],
        "spread": ["fg_spread", "1h_spread"],
        "spreads": ["fg_spread", "1h_spread"],
        "total": ["fg_total", "1h_total"],
        "totals": ["fg_total", "1h_total"],
        "moneyline": ["fg_moneyline", "1h_moneyline"],
        "ml": ["fg_moneyline", "1h_moneyline"],
    }

    valid_markets = set(get_default_market_configs().keys())

    for part in raw:
        if part in aliases:
            normalized.extend(aliases[part])
        elif part in valid_markets:
            normalized.append(part)
        else:
            raise ValueError(f"Unknown market: {part}")

    # Dedupe while preserving order
    seen = set()
    return [m for m in normalized if not (m in seen or seen.add(m))]


def main():
    parser = argparse.ArgumentParser(
        description="Extended backtest with per-market configuration and moneyline support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data/model paths
    parser.add_argument("--data", default="data/processed/master_training_data.csv")
    parser.add_argument("--models-dir", default="models/production")
    parser.add_argument("--output-json", default="data/backtest_results/extended_backtest_results.json")

    # Market selection
    parser.add_argument(
        "--markets", default="all",
        help="Comma-separated markets: fg, 1h, spread, total, moneyline, ml, or specific keys"
    )

    # Date filters
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")

    # Pricing mode
    parser.add_argument("--no-pricing", action="store_true", help="Accuracy-only mode")

    # Per-market juice (American odds)
    parser.add_argument("--fg-spread-juice", type=int, default=-110)
    parser.add_argument("--fg-total-juice", type=int, default=-110)
    parser.add_argument("--1h-spread-juice", type=int, default=-110)
    parser.add_argument("--1h-total-juice", type=int, default=-110)
    # Moneylines use actual odds from data, not fixed juice

    # Per-market confidence thresholds
    parser.add_argument("--fg-spread-conf", type=float, default=0.55)
    parser.add_argument("--fg-total-conf", type=float, default=0.55)
    parser.add_argument("--fg-ml-conf", type=float, default=0.55)
    parser.add_argument("--1h-spread-conf", type=float, default=0.55)
    parser.add_argument("--1h-total-conf", type=float, default=0.55)
    parser.add_argument("--1h-ml-conf", type=float, default=0.55)

    # Per-market minimum training games
    parser.add_argument("--min-train", type=int, default=100)

    args = parser.parse_args()

    print("=" * 80)
    print("EXTENDED BACKTEST")
    print("=" * 80)

    # Parse markets
    market_keys = normalize_markets_arg(args.markets)
    print(f"Markets: {', '.join(market_keys)}")

    # Build market configs
    default_configs = get_default_market_configs()
    market_configs = {}

    for key in market_keys:
        config = default_configs[key]

        # Apply per-market overrides
        if key == "fg_spread":
            config.juice = args.fg_spread_juice
            config.min_confidence = args.fg_spread_conf
        elif key == "fg_total":
            config.juice = args.fg_total_juice
            config.min_confidence = args.fg_total_conf
        elif key == "fg_moneyline":
            config.min_confidence = args.fg_ml_conf
        elif key == "1h_spread":
            config.juice = getattr(args, "1h_spread_juice")
            config.min_confidence = getattr(args, "1h_spread_conf")
        elif key == "1h_total":
            config.juice = getattr(args, "1h_total_juice")
            config.min_confidence = getattr(args, "1h_total_conf")
        elif key == "1h_moneyline":
            config.min_confidence = getattr(args, "1h_ml_conf")

        config.min_train_games = args.min_train
        market_configs[key] = config

    # Load data
    data_path = Path(args.data)
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path, low_memory=False)

    # Parse dates
    df["date"] = pd.to_datetime(df.get("game_date", df.get("date")), errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    print(f"Loaded {len(df)} games from {df['date'].min().date()} to {df['date'].max().date()}")

    # Apply date filters
    if args.start_date:
        df = df[df["date"] >= pd.to_datetime(args.start_date)]
    if args.end_date:
        df = df[df["date"] <= pd.to_datetime(args.end_date)]

    print(f"After date filter: {len(df)} games")

    # Run backtest
    models_dir = Path(args.models_dir)
    pricing_enabled = not args.no_pricing

    results = run_backtest(
        df=df,
        models_dir=models_dir,
        market_configs=market_configs,
        pricing_enabled=pricing_enabled,
    )

    # Print summary
    print_summary(results, output_json=args.output_json)


if __name__ == "__main__":
    main()
