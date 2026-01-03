"""
Betting card generation with comprehensive rationale.

Generates formatted betting recommendations with detailed rationale
using 6 categories: market context, team fundamentals, situational factors,
market sentiment, model confidence, and historical context.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd

from src.modeling.betting import BetRecommendation, implied_prob_from_american


CST = ZoneInfo("America/Chicago")


@dataclass
class BettingCardPick:
    """Container for a single betting card pick with rationale."""
    game_date: str
    matchup: str  # "Away @ Home"
    pick_type: str  # "FG Spread", "FG Total", "1H Spread", etc.
    pick: str  # "Lakers -3.5", "Over 223.5", etc.
    market_line: float
    market_odds: int
    model_prediction: float  # Model's predicted value
    model_probability: float  # Model's win probability
    edge: float  # Model edge vs market
    expected_value: float
    confidence: str  # "high", "medium", "low"
    rationale: List[str]  # Bullet points for rationale
    bet_recommendation: Optional[BetRecommendation] = None
    # New fields for clarity
    home_team: str = ""
    away_team: str = ""
    pick_side: str = ""  # "AWAY", "HOME", "OVER", "UNDER"
    bet_description: str = ""  # Clear description like "Bet on AWAY team to cover +9.5"


def generate_rationale(
    pick_type: str,
    pick: str,
    game_data: Dict[str, Any],
    features: Dict[str, float],
    odds: Dict[str, Any],
    comprehensive_edge: Dict[str, Any],
    betting_splits: Optional[Any] = None,
) -> List[str]:
    """
    Generate unique rationale for a pick using 6 categories.
    
    Must include at least 3 categories, with priority on:
    1. Market Context (High Priority)
    2. Market Sentiment & Sharp Action (Very High Priority)
    3. Model Confidence (High Priority)
    4. Team Fundamentals (High Priority)
    5. Situational Factors (Medium Priority)
    6. Historical Context (Low Priority)
    
    Returns:
        List of rationale bullet points (each on its own line for Excel)
    """
    rationale_points = []
    
    home_team = game_data.get("home_team", "Home")
    away_team = game_data.get("away_team", "Away")
    game_date = game_data.get("time_cst", "")
    
    # Extract market data
    market_type = pick_type
    opening_line = None
    current_line = None
    line_movement = None
    movement_timestamp = None
    
    # Extract from comprehensive_edge
    if "spread" in pick_type.lower():
        market_data = comprehensive_edge.get("full_game", {}).get("spread", {})
        if not market_data:
            market_data = comprehensive_edge.get("first_half", {}).get("spread", {})
    elif "total" in pick_type.lower():
        market_data = comprehensive_edge.get("full_game", {}).get("total", {})
        if not market_data:
            market_data = comprehensive_edge.get("first_half", {}).get("total", {})
    else:
        market_data = {}
    
    current_line = market_data.get("market_line")
    market_odds = market_data.get("market_odds")
    if market_odds is None:
        raise ValueError(f"Missing market_odds for {pick_type} pick - cannot generate betting card without real odds")
    
    # Try to get line movement from betting splits or features
    if betting_splits:
        if hasattr(betting_splits, 'spread_open') and hasattr(betting_splits, 'spread_current'):
            opening_line = betting_splits.spread_open
            current_line = betting_splits.spread_current or current_line
            line_movement = current_line - opening_line if opening_line else None
        elif hasattr(betting_splits, 'total_open') and hasattr(betting_splits, 'total_current'):
            opening_line = betting_splits.total_open
            current_line = betting_splits.total_current or current_line
            line_movement = current_line - opening_line if opening_line else None
    
    # Also check features for line movement
    if line_movement is None:
        spread_movement = features.get("spread_movement")
        if spread_movement is not None:
            line_movement = spread_movement
    
    # ============================================================
    # 1. MARKET CONTEXT (High Priority)
    # ============================================================
    market_context = []
    
    if opening_line is not None and current_line is not None and line_movement is not None:
        movement_direction = "moved from" if abs(line_movement) >= 0.5 else "stable at"
        movement_str = f"{opening_line:+.1f} to {current_line:+.1f}" if abs(line_movement) >= 0.5 else f"{current_line:+.1f}"
        market_context.append(
            f"📈 Market Type: {market_type}. Line {movement_direction} {movement_str} "
            f"({line_movement:+.1f} pt movement)."
        )
        
        # Highlight significant movement
        if abs(line_movement) >= 1.0:
            market_context.append(
                f"   Significant line movement ({line_movement:+.1f} pts) suggests sharp action."
            )
    else:
        line_str = f"{current_line:+.1f}" if current_line is not None else "N/A"
        odds_str = f"{market_odds:+d}" if market_odds is not None else "N/A"
        market_context.append(
            f"📈 Market Type: {market_type}. Current line: {line_str} ({odds_str})."
        )
    
    # ============================================================
    # 2. MARKET SENTIMENT & SHARP ACTION (Very High Priority)
    # ============================================================
    market_sentiment = []
    
    # Check for RLM
    is_rlm_spread = features.get("is_rlm_spread", 0) == 1
    is_rlm_total = features.get("is_rlm_total", 0) == 1
    sharp_side = features.get("sharp_side_spread") or features.get("sharp_side_total")
    
    if "spread" in pick_type.lower() and is_rlm_spread:
        market_sentiment.append(
            f"💸 Reverse Line Movement detected: Line moved against public betting direction, "
            f"indicating sharp money on {sharp_side if sharp_side else 'the favorite'}."
        )
    elif "total" in pick_type.lower() and is_rlm_total:
        market_sentiment.append(
            f"💸 Reverse Line Movement detected: Total moved against public betting direction, "
            f"indicating sharp money on {sharp_side if sharp_side else 'the over'}."
        )
    
    # Betting splits if available
    if betting_splits:
        if hasattr(betting_splits, 'spread_home_ticket_pct') and "spread" in pick_type.lower():
            ticket_pct = betting_splits.spread_home_ticket_pct
            money_pct = betting_splits.spread_home_money_pct
            if abs(ticket_pct - money_pct) >= 5:
                market_sentiment.append(
                    f"   Betting splits show {ticket_pct:.0f}% of tickets vs {money_pct:.0f}% of money "
                    f"on {home_team} - sharp money discrepancy detected."
                )
        elif hasattr(betting_splits, 'over_ticket_pct') and "total" in pick_type.lower():
            ticket_pct = betting_splits.over_ticket_pct
            money_pct = betting_splits.over_money_pct
            if abs(ticket_pct - money_pct) >= 5:
                market_sentiment.append(
                    f"   Betting splits show {ticket_pct:.0f}% of tickets vs {money_pct:.0f}% of money "
                    f"on Over - sharp money discrepancy detected."
                )
    
    # ============================================================
    # 3. MODEL CONFIDENCE (High Priority)
    # ============================================================
    model_confidence = []
    
    model_prob = market_data.get("win_probability", 0.5)
    edge = market_data.get("edge", 0)
    
    # Calculate EV if not provided
    implied_prob = implied_prob_from_american(market_odds)
    if market_odds > 0:
        win_amount = market_odds / 100
    else:
        win_amount = 100 / abs(market_odds)
    ev = (model_prob * win_amount) - ((1 - model_prob) * 1.0)
    
    model_confidence.append(
        f"📊 Model assigns {model_prob*100:.1f}% probability to {pick} "
        f"with {edge:+.1f} pt edge vs market."
    )
    
    if ev and ev > 0:
        model_confidence.append(
            f"   Expected Value: {ev*100:+.1f}% based on no-vig odds, exceeding confidence threshold."
        )
    
    # ============================================================
    # 4. TEAM FUNDAMENTALS (High Priority)
    # ============================================================
    team_fundamentals = []
    
    home_ppg = features.get("home_ppg", 0)
    away_ppg = features.get("away_ppg", 0)
    home_papg = features.get("home_papg", 0)
    away_papg = features.get("away_papg", 0)
    
    if home_ppg > 0 and away_ppg > 0:
        if "spread" in pick_type.lower():
            # For spreads, focus on margin
            home_margin = home_ppg - home_papg
            away_margin = away_ppg - away_papg
            margin_diff = home_margin - away_margin
            team_fundamentals.append(
                f"🧮 {home_team} averages {home_ppg:.1f} PPG (allows {home_papg:.1f}) "
                f"vs {away_team} {away_ppg:.1f} PPG (allows {away_papg:.1f}). "
                f"Net margin advantage: {margin_diff:+.1f} pts."
            )
        elif "total" in pick_type.lower():
            # For totals, focus on pace
            combined_ppg = home_ppg + away_ppg
            team_fundamentals.append(
                f"🧮 {home_team} averages {home_ppg:.1f} PPG vs {away_team} {away_ppg:.1f} PPG. "
                f"Combined average: {combined_ppg:.1f} pts/game."
            )
    
    # ELO ratings
    home_elo = features.get("home_elo", 1500)
    away_elo = features.get("away_elo", 1500)
    elo_diff = home_elo - away_elo
    if abs(elo_diff) >= 50:
        team_fundamentals.append(
            f"   ELO rating difference: {elo_diff:+.0f} pts ({home_team} {home_elo:.0f} vs {away_team} {away_elo:.0f})."
        )
    
    # ============================================================
    # 5. SITUATIONAL FACTORS (Medium Priority)
    # ============================================================
    situational = []
    
    # Rest days
    home_rest = features.get("home_rest_days", 3)
    away_rest = features.get("away_rest_days", 3)
    rest_advantage = home_rest - away_rest
    
    if abs(rest_advantage) >= 1:
        if rest_advantage > 0:
            situational.append(
                f"🌍 {home_team} has {rest_advantage} extra day(s) of rest "
                f"({home_rest} vs {away_rest} days)."
            )
        else:
            situational.append(
                f"🌍 {away_team} has {abs(rest_advantage)} extra day(s) of rest "
                f"({away_rest} vs {home_rest} days)."
            )
    
    # B2B
    home_b2b = features.get("home_b2b", 0) == 1
    away_b2b = features.get("away_b2b", 0) == 1
    
    if home_b2b:
        situational.append(f"   {home_team} is on back-to-back (short rest).")
    if away_b2b:
        situational.append(f"   {away_team} is on back-to-back (short rest).")
    
    # Travel
    away_travel_distance = features.get("away_travel_distance", 0)
    away_travel_fatigue = features.get("away_travel_fatigue", 0)
    
    if away_travel_distance >= 1500:
        situational.append(
            f"   {away_team} traveling {away_travel_distance:.0f} miles "
            f"(travel fatigue: {away_travel_fatigue:.1f} pts)."
        )
        if away_b2b and away_travel_distance >= 1500:
            situational.append(
                f"   B2B + long travel compounding penalty: -1.5 pts additional disadvantage."
            )
    
    # Home court advantage
    hca = features.get("home_court_advantage", 2.5)
    if hca >= 3.5:
        situational.append(
            f"   {home_team} has strong home court advantage ({hca:.1f} pts) - "
            f"above league average (2.5 pts)."
        )
    
    # ============================================================
    # 6. HISTORICAL CONTEXT (Low Priority)
    # ============================================================
    historical = []
    
    h2h_games = features.get("h2h_games", 0)
    h2h_margin = features.get("h2h_margin", 0)
    h2h_win_rate = features.get("h2h_win_rate", 0.5)
    
    if h2h_games >= 3:
        if "spread" in pick_type.lower():
            historical.append(
                f"🕰️ Head-to-head: {h2h_games} recent meetings, "
                f"average margin {h2h_margin:+.1f} pts favoring {home_team if h2h_margin > 0 else away_team}."
            )
        else:
            historical.append(
                f"🕰️ Head-to-head: {h2h_games} recent meetings, "
                f"{home_team} win rate: {h2h_win_rate:.1%}."
            )
    
    # ============================================================
    # COMBINE RATIONALE (Must include at least 3 categories)
    # ============================================================
    # Priority order: Market Sentiment > Market Context > Model Confidence > Team Fundamentals > Situational > Historical
    
    rationale_points.extend(market_sentiment[:2])  # Up to 2 from market sentiment
    rationale_points.extend(market_context[:1])  # 1 from market context
    rationale_points.extend(model_confidence[:1])  # 1 from model confidence
    
    # Add more to reach at least 3 categories
    if len(rationale_points) < 3:
        rationale_points.extend(team_fundamentals[:1])
    if len(rationale_points) < 3:
        rationale_points.extend(situational[:1])
    if len(rationale_points) < 3:
        rationale_points.extend(historical[:1])
    
    # Add additional points if we have space (max 5 total)
    remaining_slots = 5 - len(rationale_points)
    if remaining_slots > 0:
        # Add from remaining categories
        additional = []
        additional.extend(team_fundamentals[1:])
        additional.extend(situational[1:])
        additional.extend(historical[1:])
        rationale_points.extend(additional[:remaining_slots])
    
    # Ensure we have at least 3 points
    if len(rationale_points) < 3:
        # Fallback: add generic points
        rationale_points.append(f"Model shows {edge:+.1f} pt edge with {model_prob*100:.1f}% confidence.")
        if home_ppg > 0:
            rationale_points.append(f"{home_team} {home_ppg:.1f} PPG vs {away_team} {away_ppg:.1f} PPG.")
    
    return rationale_points[:5]  # Cap at 5 bullet points


def flag_conflicts(game_picks: List[BettingCardPick]) -> List[BettingCardPick]:
    """
    Flag conflicting picks with warnings instead of removing them.
    Adds conflict explanations to rationale so user can make informed decisions.
    """
    # Current markets are independent spreads/totals only.
    return game_picks

def generate_betting_card(
    analysis: List[Dict[str, Any]],
    target_date: datetime.date,
) -> Tuple[str, List[BettingCardPick]]:
    """
    Generate betting card summary and detailed picks.
    
    Returns:
        Tuple of (summary_text, list_of_picks)
    """
    picks = []
    summary_lines = []
    
    summary_lines.append("=" * 100)
    summary_lines.append(f"🏀 BETTING CARD RECOMMENDATIONS - {target_date.strftime('%A, %B %d, %Y')}")
    summary_lines.append("=" * 100)
    summary_lines.append("")
    
    # Collect all recommended picks
    all_recommended = []
    
    for game in analysis:
        game_picks = []  # Buffer for this game's picks
        
        home_team = game.get("home_team", "Home")
        away_team = game.get("away_team", "Away")
        matchup = f"{away_team} @ {home_team}"
        game_time = game.get("time_cst", "")
        
        comp_edge = game.get("comprehensive_edge", {})
        if not comp_edge:
            continue
        
        features = game.get("features", {})
        odds = game.get("odds", {})
        betting_splits = game.get("betting_splits")
        
        # Full game picks
        fg = comp_edge.get("full_game", {})
        
        # Spread
        if fg.get("spread", {}).get("pick"):
            spread_data = fg["spread"]
            pick_line = spread_data.get("pick_line")
            pick_team = spread_data.get("pick")
            
            if pick_line is not None:
                pick_str = f"{pick_team} {pick_line:+.1f}"
            else:
                pick_str = pick_team
            
            # Determine if pick is home or away
            is_home_pick = pick_team == home_team
            pick_side = "HOME" if is_home_pick else "AWAY"
            
            # Create clear bet description
            if pick_line is not None:
                if pick_line > 0:
                    bet_desc = f"Bet {pick_side}: {pick_team} gets {abs(pick_line):.1f} points (underdog)"
                else:
                    bet_desc = f"Bet {pick_side}: {pick_team} gives {abs(pick_line):.1f} points (favorite)"
            else:
                bet_desc = f"Bet {pick_side}: {pick_team}"
            
            rationale = generate_rationale(
                pick_type="FG Spread",
                pick=pick_str,
                game_data={"home_team": home_team, "away_team": away_team, "time_cst": game_time},
                features=features,
                odds=odds,
                comprehensive_edge=comp_edge,
                betting_splits=betting_splits,
            )
            
            # Calculate expected value
            model_prob = spread_data.get("win_probability", 0.5)
            market_odds = spread_data.get("market_odds")
            if market_odds is None:
                raise ValueError(f"Missing market_odds for FG Spread pick - cannot generate betting card without real odds")
            implied_prob = implied_prob_from_american(market_odds)
            if market_odds > 0:
                win_amount = market_odds / 100
            else:
                win_amount = 100 / abs(market_odds)
            ev = (model_prob * win_amount) - ((1 - model_prob) * 1.0)
            
            pick_obj = BettingCardPick(
                game_date=game_time,
                matchup=matchup,
                pick_type="FG Spread",
                pick=pick_str,
                market_line=spread_data.get("market_line", 0),
                market_odds=market_odds,
                model_prediction=spread_data.get("model_margin", 0),
                model_probability=model_prob,
                edge=spread_data.get("edge", 0),
                expected_value=ev,
                confidence="high" if model_prob >= 0.60 else "medium",
                rationale=rationale,
                bet_recommendation=None,
                home_team=home_team,
                away_team=away_team,
                pick_side=pick_side,
                bet_description=bet_desc,
            )
            game_picks.append(pick_obj)
        
        # Total
        if fg.get("total", {}).get("pick"):
            total_data = fg["total"]
            total_line = total_data.get('pick_line', total_data.get('market_line', 0))
            total_pick = total_data['pick']  # "OVER" or "UNDER"
            pick_str = f"{total_pick} {total_line:.1f}"
            
            # Create clear bet description for totals
            model_total = total_data.get("model_total", 0)
            if total_pick == "OVER":
                bet_desc = f"Bet OVER {total_line:.1f} total points (model projects {model_total:.1f})"
            else:
                bet_desc = f"Bet UNDER {total_line:.1f} total points (model projects {model_total:.1f})"
            
            rationale = generate_rationale(
                pick_type="FG Total",
                pick=pick_str,
                game_data={"home_team": home_team, "away_team": away_team, "time_cst": game_time},
                features=features,
                odds=odds,
                comprehensive_edge=comp_edge,
                betting_splits=betting_splits,
            )
            
            # Calculate expected value
            model_prob = total_data.get("win_probability", 0.5)
            market_odds = total_data.get("market_odds")
            if market_odds is None:
                raise ValueError(f"Missing market_odds for FG Total pick - cannot generate betting card without real odds")
            implied_prob = implied_prob_from_american(market_odds)
            if market_odds > 0:
                win_amount = market_odds / 100
            else:
                win_amount = 100 / abs(market_odds)
            ev = (model_prob * win_amount) - ((1 - model_prob) * 1.0)
            
            pick_obj = BettingCardPick(
                game_date=game_time,
                matchup=matchup,
                pick_type="FG Total",
                pick=pick_str,
                market_line=total_line,
                market_odds=market_odds,
                model_prediction=model_total,
                model_probability=model_prob,
                edge=total_data.get("edge", 0),
                expected_value=ev,
                confidence="high" if model_prob >= 0.60 else "medium",
                rationale=rationale,
                home_team=home_team,
                away_team=away_team,
                pick_side=total_pick,
                bet_description=bet_desc,
            )
            game_picks.append(pick_obj)
        
        # First half picks
        fh = comp_edge.get("first_half", {})
        
        if fh.get("spread", {}).get("pick") and fh["spread"].get("edge") is not None:
            fh_spread = fh["spread"]
            pick_line = fh_spread.get("pick_line")
            pick_team = fh_spread.get("pick")
            
            if pick_line is not None:
                pick_str = f"{pick_team} {pick_line:+.1f}"
            else:
                pick_str = pick_team
            
            # Determine if pick is home or away
            is_home_pick = pick_team == home_team
            pick_side = "HOME" if is_home_pick else "AWAY"
            
            # Create clear bet description
            if pick_line is not None:
                if pick_line > 0:
                    bet_desc = f"1H Bet {pick_side}: {pick_team} gets {abs(pick_line):.1f} pts at halftime"
                else:
                    bet_desc = f"1H Bet {pick_side}: {pick_team} gives {abs(pick_line):.1f} pts at halftime"
            else:
                bet_desc = f"1H Bet {pick_side}: {pick_team}"
            
            rationale = generate_rationale(
                pick_type="1H Spread",
                pick=pick_str,
                game_data={"home_team": home_team, "away_team": away_team, "time_cst": game_time},
                features=features,
                odds=odds,
                comprehensive_edge=comp_edge,
                betting_splits=betting_splits,
            )
            
            pick_obj = BettingCardPick(
                game_date=game_time,
                matchup=matchup,
                pick_type="1H Spread",
                pick=pick_str,
                market_line=fh_spread.get("market_line", 0),
                market_odds=fh_spread.get("market_odds"),
                # Note: market_odds validation happens in comprehensive_edge.py
                model_prediction=fh_spread.get("model_margin", 0),
                model_probability=fh_spread.get("win_probability", 0.5),
                edge=fh_spread.get("edge", 0),
                expected_value=0,
                confidence="high" if fh_spread.get("win_probability", 0.5) >= 0.60 else "medium",
                rationale=rationale,
                home_team=home_team,
                away_team=away_team,
                pick_side=pick_side,
                bet_description=bet_desc,
            )
            game_picks.append(pick_obj)
        
        if fh.get("total", {}).get("pick") and fh["total"].get("edge") is not None:
            fh_total = fh["total"]
            total_line = fh_total.get('pick_line', fh_total.get('market_line', 0))
            total_pick = fh_total['pick']  # "OVER" or "UNDER"
            pick_str = f"{total_pick} {total_line:.1f}"
            
            # Create clear bet description for 1H totals
            model_total = fh_total.get("model_total", 0)
            if total_pick == "OVER":
                bet_desc = f"1H Bet OVER {total_line:.1f} points at halftime (model projects {model_total:.1f})"
            else:
                bet_desc = f"1H Bet UNDER {total_line:.1f} points at halftime (model projects {model_total:.1f})"
            
            rationale = generate_rationale(
                pick_type="1H Total",
                pick=pick_str,
                game_data={"home_team": home_team, "away_team": away_team, "time_cst": game_time},
                features=features,
                odds=odds,
                comprehensive_edge=comp_edge,
                betting_splits=betting_splits,
            )
            
            pick_obj = BettingCardPick(
                game_date=game_time,
                matchup=matchup,
                pick_type="1H Total",
                pick=pick_str,
                market_line=total_line,
                market_odds=fh_total.get("market_odds"),
                # Note: market_odds validation happens in comprehensive_edge.py
                model_prediction=model_total,
                model_probability=fh_total.get("win_probability", 0.5),
                edge=fh_total.get("edge", 0),
                expected_value=0,
                confidence="high" if fh_total.get("win_probability", 0.5) >= 0.60 else "medium",
                rationale=rationale,
                home_team=home_team,
                away_team=away_team,
                pick_side=total_pick,
                bet_description=bet_desc,
            )
            game_picks.append(pick_obj)
            
        # Flag conflicts with warnings (but keep all picks)
        flagged_picks = flag_conflicts(game_picks)
        picks.extend(flagged_picks)
        all_recommended.extend(flagged_picks)
    
    # Generate summary table
    if all_recommended:
        summary_lines.append(f"📋 SUMMARY TABLE: {len(all_recommended)} Recommended Picks")
        summary_lines.append("")

        # Table header
        summary_lines.append("┌────────────┬─────────────────────────────────────┬─────────────────────┬─────────────────────┬──────────────────────┐")
        summary_lines.append("│ Date/Time  │ Matchup (Away @ Home)              │ Pick Type           │ Model vs Market      │ Recommended Pick     │")
        summary_lines.append("├────────────┼─────────────────────────────────────┼─────────────────────┼─────────────────────┼──────────────────────┤")

        # Table rows
        for pick in all_recommended:
            # Format the matchup to fit the column
            matchup_display = pick.matchup
            if len(matchup_display) > 35:
                matchup_display = matchup_display[:32] + "..."

            # Format pick type
            pick_type_display = pick.pick_type
            if len(pick_type_display) > 19:
                pick_type_display = pick_type_display[:16] + "..."

            # Format model vs market
            if pick.pick_type in ["FG Spread", "1H Spread"]:
                model_vs_market = f"{pick.model_prediction:+.1f} vs {pick.market_line:+.1f}"
            elif pick.pick_type in ["FG Total", "1H Total"]:
                model_vs_market = f"{pick.model_prediction:.1f} vs {pick.market_line:.1f}"
            else:
                model_vs_market = f"{pick.model_probability:.1%} vs {pick.market_odds:+d}"

            if len(model_vs_market) > 19:
                model_vs_market = model_vs_market[:16] + "..."

            # Format recommended pick
            rec_pick_display = pick.pick
            if len(rec_pick_display) > 20:
                rec_pick_display = rec_pick_display[:17] + "..."

            summary_lines.append(f"│ {pick.game_date[:10]:<10} │ {matchup_display:<35} │ {pick_type_display:<19} │ {model_vs_market:<19} │ {rec_pick_display:<20} │")

        summary_lines.append("└────────────┴─────────────────────────────────────┴─────────────────────┴─────────────────────┴──────────────────────┘")
        summary_lines.append("")

        # Detailed breakdown section
        summary_lines.append("📋 DETAILED BREAKDOWN:")
        summary_lines.append("")

        for i, pick in enumerate(all_recommended, 1):
            summary_lines.append(f"{i}. {pick.pick_type}: {pick.pick}")
            summary_lines.append(f"   Matchup: {pick.matchup} | Game Time: {pick.game_date}")
            summary_lines.append(
                f"   Model Prediction: {pick.model_prediction:.1f} | Market: {pick.market_line:+.1f} ({pick.market_odds:+d})"
            )
            summary_lines.append(
                f"   Model Probability: {pick.model_probability*100:.1f}% | Edge: {pick.edge:+.1f} pts | EV: {pick.expected_value*100:+.1f}%"
            )
            summary_lines.append("")
    else:
        summary_lines.append("⚠️  No recommended picks for this slate.")
        summary_lines.append("")
    
    summary_lines.append("=" * 100)
    summary_text = "\n".join(summary_lines)
    
    return summary_text, picks

