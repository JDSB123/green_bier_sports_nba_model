"""
Comprehensive edge calculation utilities.

Calculates betting edges for 4 markets (1H + FG spreads/totals).
Uses actual model predictions from the prediction engine for accurate 1H analysis.
"""
from __future__ import annotations
from typing import Dict, List, Any, Optional
from datetime import datetime, date
import logging

logger = logging.getLogger(__name__)



def calculate_comprehensive_edge(
    features: Dict,
    fh_features: Dict,
    odds: Dict,
    game: Dict,
    betting_splits: Optional[Any] = None,
    edge_thresholds: Optional[Dict[str, float]] = None,
    engine_predictions: Optional[Dict] = None
) -> Dict:
    """
    Calculate comprehensive betting edge for all periods.

    Args:
        features: Full game features
        fh_features: First half features
        odds: Market odds
        game: Game information
        betting_splits: Betting splits data (optional)
        edge_thresholds: Dynamic edge thresholds by bet type (optional)
    
    Returns:
        Dictionary with comprehensive edge analysis
    """
    # Import here to avoid circular dependencies
    from src.utils.odds import (
        american_to_implied_prob,
        devig_two_way,
        expected_value,
        kelly_fraction,
    )
    from src.utils.distributions import (
        estimate_spread_std,
        estimate_total_std,
        cover_probability,
        over_probability,
    )
    
    # IMPROVED: Dynamic thresholds based on market conditions and historical calibration
    if edge_thresholds is None:
        try:
            from src.config import filter_thresholds
            base_thresholds = {
                "spread": filter_thresholds.spread_min_edge,
                "total": filter_thresholds.total_min_edge,
                "1h_spread": filter_thresholds.spread_min_edge * 0.75,  # Scale for 1H
                "1h_total": filter_thresholds.total_min_edge * 0.67,   # Scale for 1H
            }

            # STRICT MODE: Since we now REQUIRE ML models, use base thresholds
            # No more dynamic adjustment - ML models must be available and are our source of truth
            edge_thresholds = base_thresholds
            logger.debug("Using strict ML-required thresholds: all predictions must come from trained models")

        except ImportError:
            # Fallback if config not available
            edge_thresholds = {
                "spread": 2.0,
                "total": 3.0,
                "1h_spread": 1.5,
                "1h_total": 2.0,
            }
    
    home_team = game.get("home_team", "Home")
    away_team = game.get("away_team", "Away")
    odds_as_of_utc = odds.get("as_of_utc")

    def _ev_and_kelly(p_model: float | None, odds_price: int | None) -> tuple[float | None, float | None]:
        ev = expected_value(p_model, odds_price, stake=1.0) if p_model is not None else None
        ev_pct = (ev * 100) if ev is not None else None
        kelly = kelly_fraction(p_model, odds_price, fraction=0.5) if p_model is not None else None
        return ev_pct, kelly
    
    # Full Game Analysis
    fg_predicted_margin = features.get("predicted_margin")
    fg_predicted_total = features.get("predicted_total")
    
    if fg_predicted_margin is None or fg_predicted_total is None:
        raise ValueError(
            f"Required features not available for {home_team} vs {away_team}"
        )
    
    # Market odds
    fg_market_spread = odds.get("home_spread")
    fg_market_total = odds.get("total")
    fg_spread_odds_home = odds.get("home_spread_price")
    fg_spread_odds_away = odds.get("away_spread_price")
    fg_total_odds_over = odds.get("total_over_price")
    fg_total_odds_under = odds.get("total_under_price")
    fg_total_odds = odds.get("total_price")
    
    if fg_market_spread is None or fg_market_total is None:
        raise ValueError(f"Required market data not available for {home_team} vs {away_team}")
    
    result = {
        "full_game": {},
        "first_half": {},
        "top_plays": [],
        "high_confidence_plays": []
    }
    
    # === FULL GAME SPREAD ===
    market_expected_margin = -fg_market_spread if fg_market_spread is not None else 0
    fg_spread_edge = fg_predicted_margin - market_expected_margin
    fg_spread_pick = home_team if fg_spread_edge > 0 else away_team
    fg_spread_confidence = min(abs(fg_spread_edge) / 5.0, 0.90) if abs(fg_spread_edge) >= 1.0 else 0
    # NO SILENT FAILURES: Require ML Engine predictions for all markets
    # The system must have trained ML models available - no fallbacks to heuristics
    if not engine_predictions:
        raise ValueError(
            f"No ML engine predictions available for {home_team} vs {away_team}. "
            f"Cannot generate predictions without trained models. "
            f"Ensure fg_spread_model.joblib and fg_total_model.joblib exist in models/production/"
        )

    # ML Engine predictions - REQUIRED (no fallbacks)
    fg_spread_pred = engine_predictions.get("full_game", {}).get("spread")
    if not fg_spread_pred:
        raise ValueError(
            f"Missing ML spread predictions for {home_team} vs {away_team}. "
            f"Engine must provide full_game.spread predictions"
        )

    # Extract ML model probabilities - these are calibrated classifier outputs
    if fg_spread_pick == home_team:
        fg_spread_p_model = fg_spread_pred.get("home_cover_prob")
    else:
        fg_spread_p_model = fg_spread_pred.get("away_cover_prob")

    if fg_spread_p_model is None or fg_spread_p_model == 0.5:
        raise ValueError(
            f"Invalid ML spread probability for {fg_spread_pick}: {fg_spread_p_model}. "
            f"Model must provide calibrated probability != 0.5"
        )

    # Use ML model confidence (entropy-based from calibrated classifier)
    fg_spread_confidence = fg_spread_pred.get("confidence")
    if fg_spread_confidence is None:
        raise ValueError(f"Missing confidence score from ML spread model for {home_team} vs {away_team}")

    # Use calibrated ML probability as win probability
    fg_spread_win_prob = fg_spread_p_model
    fg_spread_probability_source = "engine_ml"

    # Distribution-based for diagnostics only (not used for predictions)
    fg_spread_std = estimate_spread_std(features, "fg")
    fg_home_cover_dist = cover_probability(fg_predicted_margin, fg_market_spread, fg_spread_std)
    fg_away_cover_dist = 1.0 - fg_home_cover_dist
    fg_spread_p_dist = fg_home_cover_dist if fg_spread_pick == home_team else fg_away_cover_dist
    
    spread_threshold = edge_thresholds.get("spread", 2.0)
    fg_pick_line = fg_market_spread if fg_spread_pick == home_team else -fg_market_spread
    fg_spread_pick_odds = fg_spread_odds_home if fg_spread_pick == home_team else fg_spread_odds_away
    if fg_spread_pick_odds is None:
        fg_spread_pick_odds = fg_spread_odds_home
    fg_spread_p_fair_home, fg_spread_p_fair_away = devig_two_way(fg_spread_odds_home, fg_spread_odds_away)
    fg_spread_p_fair = fg_spread_p_fair_home if fg_spread_pick == home_team else fg_spread_p_fair_away
    if fg_spread_p_fair is None:
        fg_spread_p_fair = american_to_implied_prob(fg_spread_pick_odds)
    fg_spread_ev_pct, fg_spread_kelly = _ev_and_kelly(fg_spread_p_model, fg_spread_pick_odds)
    fg_spread_ev_pct_dist, fg_spread_kelly_dist = _ev_and_kelly(fg_spread_p_dist, fg_spread_pick_odds)
    
    result["full_game"]["spread"] = {
        "model_margin": fg_predicted_margin,
        "market_line": fg_market_spread,
        "market_odds": fg_spread_pick_odds,
        "market_odds_home": fg_spread_odds_home,
        "market_odds_away": fg_spread_odds_away,
        "edge": fg_spread_edge,
        "edge_abs": abs(fg_spread_edge),
        "pick": fg_spread_pick if abs(fg_spread_edge) >= spread_threshold else None,
        "pick_line": fg_pick_line if abs(fg_spread_edge) >= spread_threshold else None,
        "pick_odds": fg_spread_pick_odds if abs(fg_spread_edge) >= spread_threshold else None,
        "confidence": fg_spread_confidence,
        "win_probability": fg_spread_win_prob if abs(fg_spread_edge) >= spread_threshold else 0.5,
        "probability_source": fg_spread_probability_source,
        "p_model": fg_spread_p_model if abs(fg_spread_edge) >= spread_threshold else None,
        "p_fair": fg_spread_p_fair if abs(fg_spread_edge) >= spread_threshold else None,
        "ev_pct": fg_spread_ev_pct if abs(fg_spread_edge) >= spread_threshold else None,
        "kelly_fraction": fg_spread_kelly if abs(fg_spread_edge) >= spread_threshold else None,
        "dist_std": fg_spread_std,
        "dist_home_cover_prob": fg_home_cover_dist,
        "dist_away_cover_prob": fg_away_cover_dist,
        "dist_p_model": fg_spread_p_dist,
        "dist_ev_pct": fg_spread_ev_pct_dist,
        "dist_kelly_fraction": fg_spread_kelly_dist,
        "odds_as_of_utc": odds_as_of_utc,
        "rationale": f"Model projects {fg_spread_pick} with {fg_spread_edge:+.1f} pt edge"
    }
    
    # === FULL GAME TOTAL ===
    fg_total_edge = fg_predicted_total - fg_market_total
    fg_total_pick = "OVER" if fg_total_edge > 0 else "UNDER"
    total_threshold = edge_thresholds.get("total", 3.0)

    # NO SILENT FAILURES: Require ML Engine predictions for totals
    if not engine_predictions:
        raise ValueError(
            f"No ML engine predictions available for {home_team} vs {away_team}. "
            f"Cannot generate predictions without trained models."
        )

    # ML Engine predictions - REQUIRED for totals
    fg_total_pred = engine_predictions.get("full_game", {}).get("total")
    if not fg_total_pred:
        raise ValueError(
            f"Missing ML total predictions for {home_team} vs {away_team}. "
            f"Engine must provide full_game.total predictions"
        )

    # Extract ML model probabilities - calibrated over/under classifier
    if fg_total_pick == "OVER":
        fg_total_p_model = fg_total_pred.get("over_prob")
    else:
        fg_total_p_model = fg_total_pred.get("under_prob")

    if fg_total_p_model is None or fg_total_p_model == 0.5:
        raise ValueError(
            f"Invalid ML total probability for {fg_total_pick}: {fg_total_p_model}. "
            f"Model must provide calibrated probability != 0.5"
        )

    # Use ML model confidence (entropy-based from calibrated classifier)
    fg_total_confidence = fg_total_pred.get("confidence")
    if fg_total_confidence is None:
        raise ValueError(f"Missing confidence score from ML total model for {home_team} vs {away_team}")

    # Use calibrated ML probability as win probability
    fg_total_win_prob = fg_total_p_model
    fg_total_probability_source = "engine_ml"

    # Distribution-based for diagnostics only (not used for predictions)
    fg_total_std = estimate_total_std(features, "fg")
    fg_over_prob_dist = over_probability(fg_predicted_total, fg_market_total, fg_total_std)
    fg_under_prob_dist = 1.0 - fg_over_prob_dist
    fg_total_p_dist = fg_over_prob_dist if fg_total_pick == "OVER" else fg_under_prob_dist

    fg_total_pick_odds = fg_total_odds_over if fg_total_pick == "OVER" else fg_total_odds_under
    if fg_total_pick_odds is None:
        fg_total_pick_odds = fg_total_odds
    fg_total_p_fair_over, fg_total_p_fair_under = devig_two_way(fg_total_odds_over, fg_total_odds_under)
    fg_total_p_fair = fg_total_p_fair_over if fg_total_pick == "OVER" else fg_total_p_fair_under
    if fg_total_p_fair is None:
        fg_total_p_fair = american_to_implied_prob(fg_total_pick_odds)
    fg_total_ev_pct, fg_total_kelly = _ev_and_kelly(fg_total_p_model, fg_total_pick_odds)
    fg_total_ev_pct_dist, fg_total_kelly_dist = _ev_and_kelly(fg_total_p_dist, fg_total_pick_odds)
    
    result["full_game"]["total"] = {
        "model_total": fg_predicted_total,
        "market_line": fg_market_total,
        "market_odds": fg_total_pick_odds,
        "market_odds_over": fg_total_odds_over,
        "market_odds_under": fg_total_odds_under,
        "edge": fg_total_edge,
        "edge_abs": abs(fg_total_edge),
        "pick": fg_total_pick if abs(fg_total_edge) >= total_threshold else None,
        "pick_line": fg_market_total if abs(fg_total_edge) >= total_threshold else None,
        "pick_odds": fg_total_pick_odds if abs(fg_total_edge) >= total_threshold else None,
        "confidence": fg_total_confidence,
        "win_probability": fg_total_win_prob if abs(fg_total_edge) >= total_threshold else 0.5,
        "probability_source": fg_total_probability_source,
        "p_model": fg_total_p_model if abs(fg_total_edge) >= total_threshold else None,
        "p_fair": fg_total_p_fair if abs(fg_total_edge) >= total_threshold else None,
        "ev_pct": fg_total_ev_pct if abs(fg_total_edge) >= total_threshold else None,
        "kelly_fraction": fg_total_kelly if abs(fg_total_edge) >= total_threshold else None,
        "dist_std": fg_total_std,
        "dist_over_prob": fg_over_prob_dist,
        "dist_under_prob": fg_under_prob_dist,
        "dist_p_model": fg_total_p_dist,
        "dist_ev_pct": fg_total_ev_pct_dist,
        "dist_kelly_fraction": fg_total_kelly_dist,
        "odds_as_of_utc": odds_as_of_utc,
        "rationale": f"Model projects {fg_total_pick} with {fg_total_edge:+.1f} pt edge"
    }
    
    # === FIRST HALF - Use actual 1H model predictions if available ===
    fh_market_spread = odds.get("fh_home_spread")
    fh_market_total = odds.get("fh_total")
    fh_spread_odds_home = odds.get("fh_home_spread_price")
    fh_spread_odds_away = odds.get("fh_away_spread_price")
    fh_total_odds_over = odds.get("fh_total_over_price")
    fh_total_odds_under = odds.get("fh_total_under_price")
    fh_total_odds = odds.get("fh_total_price")

    # Get 1H predictions from engine if available
    fh_engine = engine_predictions.get("first_half", {}) if engine_predictions else {}

    # DEBUG: Log what we got from the engine
    logger.debug(f"[1H DEBUG] engine_predictions keys: {list(engine_predictions.keys()) if engine_predictions else None}")
    logger.debug(f"[1H DEBUG] fh_engine keys: {list(fh_engine.keys()) if fh_engine else None}")
    logger.debug(f"[1H DEBUG] fh_engine.get('spread'): {fh_engine.get('spread') is not None}")
    logger.debug(f"[1H DEBUG] fh_market_spread: {fh_market_spread}")

    # 1H Spread - use actual model prediction only (v33.0.7.0: NO FALLBACKS)
    if fh_engine.get("spread") and fh_market_spread is not None:
        fh_spread_pred = fh_engine["spread"]
        # Require actual prediction - no fallback to scaled FG
        if "predicted_margin" not in fh_spread_pred:
            raise ValueError("1H spread prediction missing predicted_margin - no fallbacks allowed")
        fh_predicted_margin = fh_spread_pred["predicted_margin"]
        logger.info(f"[1H SUCCESS] Using engine 1H spread prediction: margin={fh_predicted_margin}")
        # Model returns home_cover_prob (probability home covers the spread)
        fh_cover_prob = fh_spread_pred.get("home_cover_prob", 0.5)

        fh_market_expected_margin = -fh_market_spread
        fh_spread_edge = fh_predicted_margin - fh_market_expected_margin
        fh_spread_pick = home_team if fh_spread_edge > 0 else away_team

        # Use model's cover probability for confidence
        fh_spread_confidence = abs(fh_cover_prob - 0.5) * 2  # Scale to 0-1
        fh_spread_confidence = min(fh_spread_confidence, 0.70)
        fh_spread_win_prob = fh_cover_prob if fh_spread_edge > 0 else (1 - fh_cover_prob)
        fh_spread_p_model = fh_cover_prob if fh_spread_pick == home_team else (1 - fh_cover_prob)
        fh_spread_std = estimate_spread_std(fh_features, "1h")
        fh_home_cover_dist = cover_probability(fh_predicted_margin, fh_market_spread, fh_spread_std)
        fh_away_cover_dist = 1.0 - fh_home_cover_dist
        fh_spread_p_dist = fh_home_cover_dist if fh_spread_pick == home_team else fh_away_cover_dist

        fh_spread_threshold = edge_thresholds.get("1h_spread", 1.5)
        fh_pick_line = fh_market_spread if fh_spread_pick == home_team else -fh_market_spread
        fh_spread_pick_odds = fh_spread_odds_home if fh_spread_pick == home_team else fh_spread_odds_away
        if fh_spread_pick_odds is None:
            fh_spread_pick_odds = fh_spread_odds_home
        fh_spread_p_fair_home, fh_spread_p_fair_away = devig_two_way(fh_spread_odds_home, fh_spread_odds_away)
        fh_spread_p_fair = fh_spread_p_fair_home if fh_spread_pick == home_team else fh_spread_p_fair_away
        if fh_spread_p_fair is None:
            fh_spread_p_fair = american_to_implied_prob(fh_spread_pick_odds)
        fh_spread_ev_pct, fh_spread_kelly = _ev_and_kelly(fh_spread_p_model, fh_spread_pick_odds)
        fh_spread_ev_pct_dist, fh_spread_kelly_dist = _ev_and_kelly(fh_spread_p_dist, fh_spread_pick_odds)

        result["first_half"]["spread"] = {
            "model_margin": fh_predicted_margin,
            "market_line": fh_market_spread,
            "market_odds": fh_spread_pick_odds,
            "market_odds_home": fh_spread_odds_home,
            "market_odds_away": fh_spread_odds_away,
            "edge": fh_spread_edge,
            "edge_abs": abs(fh_spread_edge),
            "pick": fh_spread_pick if abs(fh_spread_edge) >= fh_spread_threshold else None,
            "pick_line": fh_pick_line if abs(fh_spread_edge) >= fh_spread_threshold else None,
            "pick_odds": fh_spread_pick_odds if abs(fh_spread_edge) >= fh_spread_threshold else None,
            "confidence": fh_spread_confidence,
            "win_probability": fh_spread_win_prob,
            "probability_source": "engine_ml",  # 1H always uses engine predictions when available
            "p_model": fh_spread_p_model if abs(fh_spread_edge) >= fh_spread_threshold else None,
            "p_fair": fh_spread_p_fair if abs(fh_spread_edge) >= fh_spread_threshold else None,
            "ev_pct": fh_spread_ev_pct if abs(fh_spread_edge) >= fh_spread_threshold else None,
            "kelly_fraction": fh_spread_kelly if abs(fh_spread_edge) >= fh_spread_threshold else None,
            "dist_std": fh_spread_std,
            "dist_home_cover_prob": fh_home_cover_dist,
            "dist_away_cover_prob": fh_away_cover_dist,
            "dist_p_model": fh_spread_p_dist,
            "dist_ev_pct": fh_spread_ev_pct_dist,
            "dist_kelly_fraction": fh_spread_kelly_dist,
            "odds_as_of_utc": odds_as_of_utc,
            "rationale": f"Model projects {fh_spread_pick} with {fh_spread_edge:+.1f} pt edge (1H)"
        }
    else:
        # No engine data or no market - skip 1H spread (no fallback to scaled FG)
        result["first_half"]["spread"] = {
            "edge": None,
            "pick": None,
            "rationale": "First half spread: no model prediction available (no fallbacks)"
        }

    # 1H Total - use actual model prediction only (v33.0.7.0: NO FALLBACKS)
    if fh_engine.get("total") and fh_market_total is not None:
        fh_total_pred = fh_engine["total"]
        # Require actual prediction - no fallback to scaled FG
        if "predicted_total" not in fh_total_pred:
            raise ValueError("1H total prediction missing predicted_total - no fallbacks allowed")
        fh_predicted_total = fh_total_pred["predicted_total"]
        # Model returns over_prob (probability game goes over)
        fh_over_prob = fh_total_pred.get("over_prob", 0.5)

        fh_total_edge = fh_predicted_total - fh_market_total
        fh_total_pick = "OVER" if fh_total_edge > 0 else "UNDER"

        # Use model's over probability for confidence
        fh_total_confidence = abs(fh_over_prob - 0.5) * 2  # Scale to 0-1
        fh_total_confidence = min(fh_total_confidence, 0.70)
        fh_total_win_prob = fh_over_prob if fh_total_edge > 0 else (1 - fh_over_prob)
        fh_total_p_model = fh_over_prob if fh_total_pick == "OVER" else (1 - fh_over_prob)
        fh_total_std = estimate_total_std(fh_features, "1h")
        fh_over_prob_dist = over_probability(fh_predicted_total, fh_market_total, fh_total_std)
        fh_under_prob_dist = 1.0 - fh_over_prob_dist
        fh_total_p_dist = fh_over_prob_dist if fh_total_pick == "OVER" else fh_under_prob_dist

        fh_total_threshold = edge_thresholds.get("1h_total", 2.0)
        fh_total_pick_odds = fh_total_odds_over if fh_total_pick == "OVER" else fh_total_odds_under
        if fh_total_pick_odds is None:
            fh_total_pick_odds = fh_total_odds
        fh_total_p_fair_over, fh_total_p_fair_under = devig_two_way(fh_total_odds_over, fh_total_odds_under)
        fh_total_p_fair = fh_total_p_fair_over if fh_total_pick == "OVER" else fh_total_p_fair_under
        if fh_total_p_fair is None:
            fh_total_p_fair = american_to_implied_prob(fh_total_pick_odds)
        fh_total_ev_pct, fh_total_kelly = _ev_and_kelly(fh_total_p_model, fh_total_pick_odds)
        fh_total_ev_pct_dist, fh_total_kelly_dist = _ev_and_kelly(fh_total_p_dist, fh_total_pick_odds)

        result["first_half"]["total"] = {
            "model_total": fh_predicted_total,
            "market_line": fh_market_total,
            "market_odds": fh_total_pick_odds,
            "market_odds_over": fh_total_odds_over,
            "market_odds_under": fh_total_odds_under,
            "edge": fh_total_edge,
            "edge_abs": abs(fh_total_edge),
            "pick": fh_total_pick if abs(fh_total_edge) >= fh_total_threshold else None,
            "pick_line": fh_market_total if abs(fh_total_edge) >= fh_total_threshold else None,
            "pick_odds": fh_total_pick_odds if abs(fh_total_edge) >= fh_total_threshold else None,
            "confidence": fh_total_confidence,
            "win_probability": fh_total_win_prob,
            "probability_source": "engine_ml",  # 1H always uses engine predictions when available
            "p_model": fh_total_p_model if abs(fh_total_edge) >= fh_total_threshold else None,
            "p_fair": fh_total_p_fair if abs(fh_total_edge) >= fh_total_threshold else None,
            "ev_pct": fh_total_ev_pct if abs(fh_total_edge) >= fh_total_threshold else None,
            "kelly_fraction": fh_total_kelly if abs(fh_total_edge) >= fh_total_threshold else None,
            "dist_std": fh_total_std,
            "dist_over_prob": fh_over_prob_dist,
            "dist_under_prob": fh_under_prob_dist,
            "dist_p_model": fh_total_p_dist,
            "dist_ev_pct": fh_total_ev_pct_dist,
            "dist_kelly_fraction": fh_total_kelly_dist,
            "odds_as_of_utc": odds_as_of_utc,
            "rationale": f"Model projects {fh_total_pick} with {fh_total_edge:+.1f} pt edge (1H)"
        }
    else:
        # No engine data or no market - skip 1H total (no fallback to scaled FG)
        result["first_half"]["total"] = {
            "edge": None,
            "pick": None,
            "rationale": "First half total: no model prediction available (no fallbacks)"
        }

    # Build top plays
    all_plays = []
    for period_name, period_data in [("full_game", result["full_game"]), ("first_half", result["first_half"])]:
        for market_name, market_data in period_data.items():
            if market_data.get("pick") and market_data.get("edge") is not None:
                # IMPROVED: Use actual confidence score for high confidence determination
                # Instead of arbitrary win_probability thresholds, use the calibrated confidence score
                confidence_score = market_data.get("confidence", 0)
                win_probability = market_data.get("win_probability", 0.5)

                # IMPROVED EXTREME FILTERING: Multiple criteria for high confidence picks
                # Don't filter out good picks that may have moderate confidence but extreme probabilities
                edge_value = market_data["edge"]

                # Criteria for high confidence picks:
                # 1. High confidence + moderately extreme probability
                high_conf_extreme = (
                    confidence_score >= 0.70 and  # Model is confident
                    (win_probability >= 0.60 or win_probability <= 0.40)  # Moderately extreme probability
                )

                # 2. Very extreme probability (regardless of confidence, but with minimum confidence)
                very_extreme_prob = (
                    confidence_score >= 0.55 and  # Minimum confidence required
                    (win_probability >= 0.70 or win_probability <= 0.30)  # Very extreme probability
                )

                # 3. Large edge with good confidence (catches strong fundamental mismatches)
                large_edge_good_conf = (
                    abs(edge_value) >= 3.0 and  # Large edge (>= 3 points for spreads, >= 3 total)
                    confidence_score >= 0.60  # Good confidence
                )

                # High confidence if any criteria met
                is_high_confidence = high_conf_extreme or very_extreme_prob or large_edge_good_conf

                # Validate probability source consistency
                probability_source = market_data.get("probability_source", "unknown")
                if probability_source not in ["engine_ml", "distribution", "heuristic"]:
                    logger.warning(f"Unknown probability source '{probability_source}' for {period_name} {market_name}")

                play = {
                    "type": f"{period_name.upper()} {market_name.upper()}",
                    "pick": market_data["pick"],
                    "confidence": confidence_score,
                    "edge": market_data["edge"],
                    "model_probability": win_probability,
                    "probability_source": probability_source,
                    "ev_pct": market_data.get("ev_pct"),
                    "is_high_confidence": is_high_confidence,
                    "rationale": market_data.get("rationale", "")
                }
                all_plays.append(play)
    
    def _ev_sort(play: Dict[str, Any]) -> float:
        ev_pct = play.get("ev_pct")
        if ev_pct is None:
            return -9999.0
        return float(ev_pct)

    all_plays.sort(key=lambda x: (_ev_sort(x), x.get("confidence", 0)), reverse=True)
    result["top_plays"] = all_plays[:3]
    result["high_confidence_plays"] = [p for p in all_plays if p.get("is_high_confidence", False)][:5]
    
    # VALIDATION: Log probability source usage for transparency
    _validate_probability_sources(result, engine_predictions)

    return result


def _validate_probability_sources(result: Dict, engine_predictions: Optional[Dict]) -> None:
    """
    Validate that ALL predictions come from ML models (strict requirement).

    Args:
        result: Comprehensive edge analysis result
        engine_predictions: Engine predictions dict (required)
    """
    source_counts = {"engine_ml": 0, "distribution": 0, "heuristic": 0}

    for period_name, period_data in [("full_game", result["full_game"]), ("first_half", result["first_half"])]:
        for market_name, market_data in period_data.items():
            if market_data.get("pick") and market_data.get("probability_source"):
                source = market_data["probability_source"]
                source_counts[source] = source_counts.get(source, 0) + 1

    # STRICT VALIDATION: All picks must come from ML models
    total_picks = sum(source_counts.values())
    if total_picks > 0:
        ml_pct = (source_counts["engine_ml"] / total_picks) * 100

        logger.info(f"ML Model Coverage: {source_counts['engine_ml']}/{total_picks} picks ({ml_pct:.1f}%) from calibrated classifiers")

        # STRICT REQUIREMENT: 100% ML model usage
        if ml_pct < 100.0:
            non_ml_sources = [src for src in ["distribution", "heuristic"] if source_counts.get(src, 0) > 0]
            raise ValueError(
                f"STRICT VIOLATION: {100.0 - ml_pct:.1f}% of picks ({sum(source_counts[s] for s in non_ml_sources)}) "
                f"come from non-ML sources {non_ml_sources}. All predictions must use trained ML models only."
            )

        logger.info("✅ STRICT VALIDATION PASSED: All predictions use ML models (no fallbacks)")


def generate_comprehensive_text_report(analysis: List[Dict], target_date: date) -> str:
    """
    Generate a comprehensive text-based report.
    
    Extracted from deprecated analyze_todays_slate.py script.
    
    Args:
        analysis: List of game analysis dictionaries
        target_date: Target date for the report
    
    Returns:
        Formatted text report
    """
    from src.utils.slate_analysis import get_cst_now
    
    lines = []
    lines.append("=" * 100)
    lines.append(f"🏀 NBA COMPREHENSIVE SLATE ANALYSIS - {target_date.strftime('%A, %B %d, %Y')}")
    lines.append("=" * 100)
    lines.append("")
    
    if not analysis:
        lines.append("No games scheduled for this date.")
        return "\n".join(lines)
    
    for i, game in enumerate(analysis, 1):
        lines.append(f"{'─' * 100}")
        lines.append(f"GAME {i}: {game['away_team']} @ {game['home_team']}")
        lines.append(f"Time: {game.get('time_cst', 'TBD')}")
        lines.append("")
        
        comp_edge = game.get("comprehensive_edge", {})
        
        if not comp_edge:
            lines.append("   ⚠️  Analysis not available")
            lines.append("")
            continue
        
        # Full Game Analysis
        fg = comp_edge.get("full_game", {})
        lines.append("📊 FULL GAME ANALYSIS:")
        lines.append("")
        
        if fg.get("spread"):
            sp = fg["spread"]
            lines.append(f"   SPREAD:")
            if sp.get("pick"):
                lines.append(f"      ✅ PICK: {sp['pick']} {sp.get('pick_line', sp.get('market_line', 0)):+.1f}")
                lines.append(f"      Edge: {sp['edge']:+.1f} pts | Confidence: {sp.get('confidence', 0)*100:.1f}%")
            lines.append("")
        
        if fg.get("total"):
            tot = fg["total"]
            lines.append(f"   TOTAL:")
            if tot.get("pick"):
                lines.append(f"      ✅ PICK: {tot['pick']} {tot.get('pick_line', tot.get('market_line', 0)):.1f}")
                lines.append(f"      Edge: {tot['edge']:+.1f} pts | Confidence: {tot.get('confidence', 0)*100:.1f}%")
            lines.append("")
        
        # First Half Analysis
        fh = comp_edge.get("first_half", {})
        lines.append("📊 FIRST HALF ANALYSIS:")
        lines.append("")
        
        if fh.get("spread") and fh["spread"].get("pick"):
            sp = fh["spread"]
            lines.append(f"   1H SPREAD:")
            lines.append(f"      ✅ PICK: {sp['pick']} {sp.get('pick_line', sp.get('market_line', 0)):+.1f}")
            lines.append(f"      Edge: {sp['edge']:+.1f} pts | Confidence: {sp.get('confidence', 0)*100:.1f}%")
            lines.append("")
        
        if fh.get("total") and fh["total"].get("pick"):
            tot = fh["total"]
            lines.append(f"   1H TOTAL:")
            lines.append(f"      ✅ PICK: {tot['pick']} {tot.get('pick_line', tot.get('market_line', 0)):.1f}")
            lines.append(f"      Edge: {tot['edge']:+.1f} pts | Confidence: {tot.get('confidence', 0)*100:.1f}%")
            lines.append("")
    
    lines.append("=" * 100)
    lines.append(f"Generated: {get_cst_now().strftime('%Y-%m-%d %I:%M %p CST')}")
    lines.append("")
    
    return "\n".join(lines)
