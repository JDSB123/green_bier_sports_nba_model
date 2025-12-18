"""
Comprehensive edge calculation utilities.

Extracted from deprecated analyze_todays_slate.py script.
These functions calculate betting edges for full game and first half markets.
"""
from __future__ import annotations
from typing import Dict, List, Any, Optional
from datetime import datetime, date

# Note: These functions have dependencies on models that may need to be refactored
# For now, keeping the core logic but simplifying dependencies


def calculate_comprehensive_edge(
    features: Dict,
    fh_features: Dict,
    odds: Dict,
    game: Dict,
    betting_splits: Optional[Any] = None,
    edge_thresholds: Optional[Dict[str, float]] = None
) -> Dict:
    """
    Calculate comprehensive betting edge for full game and first half.
    
    NOTE: This is a simplified version extracted from the deprecated script.
    The original had dependencies on models from scripts/archive/ which don't exist.
    This version uses simplified calculations that work with the current codebase.
    
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
    from src.utils.slate_analysis import american_to_implied_prob
    
    # Default thresholds if not provided
    if edge_thresholds is None:
        edge_thresholds = {
            "spread": 2.0,
            "total": 3.0,
            "moneyline": 0.03,
            "1h_spread": 1.5,
            "1h_total": 2.0,
        }
    
    home_team = game.get("home_team", "Home")
    away_team = game.get("away_team", "Away")
    
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
    fg_home_ml = odds.get("home_ml")
    fg_away_ml = odds.get("away_ml")
    fg_spread_odds = odds.get("home_spread_price", -110)
    fg_total_odds = odds.get("total_price", -110)
    
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
    fg_spread_win_prob = 0.5 + (abs(fg_spread_edge) * 0.03)
    fg_spread_win_prob = max(0.51, min(0.80, fg_spread_win_prob)) if abs(fg_spread_edge) >= 0.5 else 0.5
    
    spread_threshold = edge_thresholds.get("spread", 2.0)
    fg_pick_line = fg_market_spread if fg_spread_pick == home_team else -fg_market_spread
    
    result["full_game"]["spread"] = {
        "model_margin": fg_predicted_margin,
        "market_line": fg_market_spread,
        "market_odds": fg_spread_odds,
        "edge": fg_spread_edge,
        "pick": fg_spread_pick if abs(fg_spread_edge) >= spread_threshold else None,
        "pick_line": fg_pick_line if abs(fg_spread_edge) >= spread_threshold else None,
        "pick_odds": fg_spread_odds if abs(fg_spread_edge) >= spread_threshold else None,
        "confidence": fg_spread_confidence,
        "win_probability": fg_spread_win_prob if abs(fg_spread_edge) >= spread_threshold else 0.5,
        "rationale": f"Model projects {fg_spread_pick} with {fg_spread_edge:+.1f} pt edge"
    }
    
    # === FULL GAME TOTAL ===
    fg_total_edge = fg_predicted_total - fg_market_total
    fg_total_pick = "OVER" if fg_total_edge > 0 else "UNDER"
    total_threshold = edge_thresholds.get("total", 3.0)
    fg_total_confidence = min(abs(fg_total_edge) / 10.0, 0.90) if abs(fg_total_edge) >= (total_threshold * 0.75) else 0
    fg_total_win_prob = 0.5 + (abs(fg_total_edge) * 0.025)
    fg_total_win_prob = max(0.51, min(0.75, fg_total_win_prob)) if abs(fg_total_edge) >= 1.0 else 0.5
    
    result["full_game"]["total"] = {
        "model_total": fg_predicted_total,
        "market_line": fg_market_total,
        "market_odds": fg_total_odds,
        "edge": fg_total_edge,
        "pick": fg_total_pick if abs(fg_total_edge) >= total_threshold else None,
        "pick_line": fg_market_total if abs(fg_total_edge) >= total_threshold else None,
        "pick_odds": fg_total_odds if abs(fg_total_edge) >= total_threshold else None,
        "confidence": fg_total_confidence,
        "win_probability": fg_total_win_prob if abs(fg_total_edge) >= total_threshold else 0.5,
        "rationale": f"Model projects {fg_total_pick} with {fg_total_edge:+.1f} pt edge"
    }
    
    # === FULL GAME MONEYLINE ===
    fg_ml_pick = None
    fg_ml_confidence = 0
    fg_ml_rationale = "No moneyline value found."
    
    if fg_home_ml and fg_away_ml:
        # Simplified ML calculation
        model_home_prob = 0.5 + (fg_predicted_margin * 0.02)  # Simplified
        model_home_prob = max(0.4, min(0.9, model_home_prob))
        market_home_prob = american_to_implied_prob(fg_home_ml)
        fg_ml_edge_home = model_home_prob - market_home_prob
        
        model_away_prob = 1 - model_home_prob
        market_away_prob = american_to_implied_prob(fg_away_ml)
        fg_ml_edge_away = model_away_prob - market_away_prob
        
        if fg_ml_edge_home > 0.03 or fg_ml_edge_away > 0.03:
            if fg_ml_edge_home > fg_ml_edge_away:
                fg_ml_pick = home_team
                fg_ml_confidence = min(fg_ml_edge_home * 2.5, 0.75)
                fg_ml_rationale = f"Model gives {home_team} {model_home_prob*100:.1f}% vs market {market_home_prob*100:.1f}%"
            else:
                fg_ml_pick = away_team
                fg_ml_confidence = min(fg_ml_edge_away * 2.5, 0.75)
                fg_ml_rationale = f"Model gives {away_team} {model_away_prob*100:.1f}% vs market {market_away_prob*100:.1f}%"
    
    result["full_game"]["moneyline"] = {
        "market_home_odds": fg_home_ml,
        "market_away_odds": fg_away_ml,
        "market_home_prob": american_to_implied_prob(fg_home_ml) if fg_home_ml else 0.5,
        "market_away_prob": american_to_implied_prob(fg_away_ml) if fg_away_ml else 0.5,
        "edge_home": fg_ml_edge_home if fg_home_ml and fg_away_ml else None,
        "edge_away": fg_ml_edge_away if fg_home_ml and fg_away_ml else None,
        "pick": fg_ml_pick,
        "confidence": fg_ml_confidence,
        "rationale": fg_ml_rationale
    }
    
    # === FIRST HALF (Simplified) ===
    # Use scaled full game predictions for first half
    fh_predicted_margin = fg_predicted_margin * 0.5  # Simplified scaling
    fh_predicted_total = fg_predicted_total * 0.5
    
    fh_market_spread = odds.get("fh_home_spread")
    fh_market_total = odds.get("fh_total")
    fh_spread_odds = odds.get("fh_home_spread_price")
    fh_total_odds = odds.get("fh_total_price")
    
    if fh_market_spread is not None and fh_spread_odds is not None:
        fh_market_expected_margin = -fh_market_spread
        fh_spread_edge = fh_predicted_margin - fh_market_expected_margin
        fh_spread_pick = home_team if fh_spread_edge > 0 else away_team
        fh_spread_confidence = min(abs(fh_spread_edge) / 6.0, 0.70) if abs(fh_spread_edge) >= 1.5 else 0
        fh_spread_win_prob = 0.5 + (abs(fh_spread_edge) * 0.02)
        fh_spread_win_prob = max(0.51, min(0.65, fh_spread_win_prob)) if abs(fh_spread_edge) >= 0.5 else 0.5
        
        fh_spread_threshold = edge_thresholds.get("1h_spread", 1.5)
        fh_pick_line = fh_market_spread if fh_spread_pick == home_team else -fh_market_spread
        
        result["first_half"]["spread"] = {
            "model_margin": fh_predicted_margin,
            "market_line": fh_market_spread,
            "market_odds": fh_spread_odds,
            "edge": fh_spread_edge,
            "pick": fh_spread_pick if abs(fh_spread_edge) >= fh_spread_threshold else None,
            "pick_line": fh_pick_line if abs(fh_spread_edge) >= fh_spread_threshold else None,
            "pick_odds": fh_spread_odds if abs(fh_spread_edge) >= fh_spread_threshold else None,
            "confidence": fh_spread_confidence,
            "win_probability": fh_spread_win_prob if abs(fh_spread_edge) >= fh_spread_threshold else 0.5,
            "rationale": f"Model projects {fh_spread_pick} with {fh_spread_edge:+.1f} pt edge (1H)"
        }
    else:
        result["first_half"]["spread"] = {
            "edge": None,
            "pick": None,
            "rationale": "First half spread market not available"
        }
    
    if fh_market_total is not None and fh_total_odds is not None:
        fh_total_edge = fh_predicted_total - fh_market_total
        fh_total_pick = "OVER" if fh_total_edge > 0 else "UNDER"
        fh_total_threshold = edge_thresholds.get("1h_total", 2.0)
        fh_total_confidence = min(abs(fh_total_edge) / 8.0, 0.70) if abs(fh_total_edge) >= fh_total_threshold else 0
        fh_total_win_prob = 0.5 + (abs(fh_total_edge) * 0.035)
        fh_total_win_prob = max(0.51, min(0.70, fh_total_win_prob)) if abs(fh_total_edge) >= 1.0 else 0.5
        
        result["first_half"]["total"] = {
            "model_total": fh_predicted_total,
            "market_line": fh_market_total,
            "market_odds": fh_total_odds,
            "edge": fh_total_edge,
            "pick": fh_total_pick if abs(fh_total_edge) >= fh_total_threshold else None,
            "pick_line": fh_market_total if abs(fh_total_edge) >= fh_total_threshold else None,
            "pick_odds": fh_total_odds if abs(fh_total_edge) >= fh_total_threshold else None,
            "confidence": fh_total_confidence,
            "win_probability": fh_total_win_prob if abs(fh_total_edge) >= fh_total_threshold else 0.5,
            "rationale": f"Model projects {fh_total_pick} with {fh_total_edge:+.1f} pt edge (1H)"
        }
    else:
        result["first_half"]["total"] = {
            "edge": None,
            "pick": None,
            "rationale": "First half total market not available"
        }
    
    result["first_half"]["moneyline"] = {
        "pick": None,
        "rationale": "First half moneyline analysis not implemented"
    }
    
    # Build top plays
    all_plays = []
    for period_name, period_data in [("full_game", result["full_game"]), ("first_half", result["first_half"])]:
        for market_name, market_data in period_data.items():
            if market_data.get("pick") and market_data.get("edge") is not None:
                play = {
                    "type": f"{period_name.upper()} {market_name.upper()}",
                    "pick": market_data["pick"],
                    "confidence": market_data.get("confidence", 0),
                    "edge": market_data["edge"],
                    "model_probability": market_data.get("win_probability", 0.5),
                    "is_high_confidence": market_data.get("win_probability", 0.5) > 0.60 or market_data.get("win_probability", 0.5) < 0.40,
                    "rationale": market_data.get("rationale", "")
                }
                all_plays.append(play)
    
    all_plays.sort(key=lambda x: x["confidence"], reverse=True)
    result["top_plays"] = all_plays[:3]
    result["high_confidence_plays"] = [p for p in all_plays if p.get("is_high_confidence", False)][:5]
    
    return result


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
        
        if fg.get("moneyline"):
            ml = fg["moneyline"]
            if ml.get("pick"):
                lines.append(f"   MONEYLINE:")
                lines.append(f"      ✅ PICK: {ml['pick']}")
                lines.append(f"      Confidence: {ml.get('confidence', 0)*100:.1f}%")
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
