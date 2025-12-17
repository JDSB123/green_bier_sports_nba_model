#!/usr/bin/env python3
"""
NBA Today's Slate Analysis
==========================
Fetches today's NBA games, builds features, and generates a clean visualization
for the weekly lineup.

Usage:
    python scripts/analyze_todays_slate.py
    python scripts/analyze_todays_slate.py --date 2025-12-07
    python scripts/analyze_todays_slate.py --output weekly_lineup.png
"""
import asyncio
import argparse
import json
import os
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from zoneinfo import ZoneInfo
import statistics
import math

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.modeling.clv_tracker import CLVTracker
from src.modeling.prediction_logger import PredictionLogger
from src.modeling.edge_thresholds import get_edge_thresholds_for_game

from improved_fh_model import ImprovedFirstHalfModel
from improved_ml_model import ImprovedMoneylineModel

# Central Standard Time
CST = ZoneInfo("America/Chicago")

# Initialize enhanced models
fh_model = ImprovedFirstHalfModel() if ImprovedFirstHalfModel else None
ml_model = ImprovedMoneylineModel() if ImprovedMoneylineModel else None

# NBA Key Numbers for spread betting
NBA_KEY_NUMBERS = [0.5, 1.0, 2.5, 3.0, 4.5, 5.0, 6.5, 7.0, 7.5, 10.0]
KEY_NUMBER_ADJUSTMENTS = {
    0.5: 1.02,  # Half point lines
    1.0: 1.01,
    2.5: 1.02,
    3.0: 1.03,  # Most common margin
    4.5: 1.02,
    5.0: 1.015,
    6.5: 1.02,
    7.0: 1.025,  # Common favorite margin
    7.5: 1.02,
    10.0: 1.02,  # Double digits
}


def get_cst_now() -> datetime:
    """Get current time in CST."""
    return datetime.now(CST)


def parse_utc_time(iso_string: str) -> datetime:
    """Parse ISO UTC time string to datetime."""
    if iso_string.endswith("Z"):
        iso_string = iso_string[:-1] + "+00:00"
    dt = datetime.fromisoformat(iso_string).replace(tzinfo=timezone.utc)
    # Round to nearest 5 minutes if odd
    minutes = dt.minute
    rounded_min = round(minutes / 5.0) * 5
    if rounded_min == 60:
        dt = dt.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    else:
        dt = dt.replace(minute=rounded_min, second=0, microsecond=0)
    return dt


def to_cst(dt: datetime) -> datetime:
    """Convert datetime to CST."""
    return dt.astimezone(CST)


def format_cst_time(dt: datetime) -> str:
    """Format datetime as CST string."""
    cst_dt = to_cst(dt)
    return cst_dt.strftime("%I:%M %p CST")


def get_target_date(date_str: str = None) -> datetime.date:
    """Get target date for analysis."""
    now_cst = get_cst_now()
    
    if date_str is None or date_str.lower() == "today":
        return now_cst.date()
    elif date_str.lower() == "tomorrow":
        return (now_cst + timedelta(days=1)).date()
    else:
        return datetime.strptime(date_str, "%Y-%m-%d").date()


def filter_games_for_date(games: list, target_date: datetime.date) -> list:
    """Filter games to only include those on the target date (in CST)."""
    filtered = []
    for game in games:
        commence_time = game.get("commence_time")
        if not commence_time:
            continue
        try:
            game_dt = parse_utc_time(commence_time)
            game_cst = to_cst(game_dt)
            if game_cst.date() == target_date:
                filtered.append(game)
        except Exception:
            continue
    return filtered


def american_to_implied_prob(american_odds: int) -> float:
    """Convert American odds to implied probability."""
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)


def prob_to_american_odds(prob: float) -> int:
    """Convert probability to American odds."""
    if prob >= 0.5:
        return int(-100 * prob / (1 - prob))
    else:
        return int(100 * (1 - prob) / prob)


def spread_to_moneyline(spread: float) -> Tuple[int, int]:
    """
    Convert spread to approximate moneyline odds.
    Returns (favorite_ml, underdog_ml).
    """
    # Empirical formula: ML ≈ spread^1.5 * 11
    abs_spread = abs(spread)
    if abs_spread == 0:
        return (-110, -110)
    
    # Calculate favorite odds
    fav_prob = 0.5 + (abs_spread * 0.04)  # ~4% per point
    fav_prob = min(0.95, max(0.55, fav_prob))
    
    fav_ml = prob_to_american_odds(fav_prob)
    dog_ml = prob_to_american_odds(1 - fav_prob)
    
    if spread > 0:  # Home is favorite
        return (fav_ml, dog_ml)
    else:  # Away is favorite
        return (dog_ml, fav_ml)


def extract_consensus_odds(game: Dict) -> Dict[str, Any]:
    """Extract consensus odds from all bookmakers."""
    bookmakers = game.get("bookmakers", [])
    
    # Collect all odds
    h2h_home = []
    h2h_away = []
    spreads_home = []
    spreads_away = []
    totals = []
    # First half markets
    fh_spreads_home = []
    fh_spreads_away = []
    fh_totals = []
    
    home_team = game.get("home_team")
    away_team = game.get("away_team")
    
    for bm in bookmakers:
        for market in bm.get("markets", []):
            key = market.get("key")
            outcomes = market.get("outcomes", [])
            
            if key == "h2h":
                for out in outcomes:
                    if out.get("name") == home_team:
                        price = out.get("price")
                        if price is not None:
                            h2h_home.append(price)
                    elif out.get("name") == away_team:
                        price = out.get("price")
                        if price is not None:
                            h2h_away.append(price)
            
            elif key == "spreads":
                for out in outcomes:
                    if out.get("name") == home_team:
                        spreads_home.append({
                            "price": out.get("price"),
                            "point": out.get("point")
                        })
                    elif out.get("name") == away_team:
                        spreads_away.append({
                            "price": out.get("price"),
                            "point": out.get("point")
                        })
            
            elif key == "totals":
                for out in outcomes:
                    if out.get("name") == "Over":
                        totals.append({
                            "point": out.get("point"),
                            "price": out.get("price")
                        })
                    elif out.get("name") == "Under":
                        # Also collect Under prices for reference
                        totals.append({
                            "point": out.get("point"),
                            "price": out.get("price"),
                            "side": "Under"
                        })
            
            # First half markets (check for various naming conventions)
            elif key and ("h1" in key.lower() or "first_half" in key.lower() or "1h" in key.lower()):
                if "spread" in key.lower() or "handicap" in key.lower():
                    # First half spread
                    for out in outcomes:
                        if out.get("name") == home_team:
                            fh_spreads_home.append({
                                "price": out.get("price"),
                                "point": out.get("point")
                            })
                        elif out.get("name") == away_team:
                            fh_spreads_away.append({
                                "price": out.get("price"),
                                "point": out.get("point")
                            })
                elif "total" in key.lower() or "over_under" in key.lower():
                    # First half total
                    for out in outcomes:
                        if out.get("name") == "Over":
                            fh_totals.append({
                                "point": out.get("point"),
                                "price": out.get("price")
                            })
                        elif out.get("name") == "Under":
                            fh_totals.append({
                                "point": out.get("point"),
                                "price": out.get("price"),
                                "side": "Under"
                            })
    
    # Calculate consensus with median for ML to handle outliers
    result = {
        "home_ml": None,
        "away_ml": None,
        "home_spread": None,
        "home_spread_price": None,
        "total": None,
        "total_price": None,  # Over price (Under typically same)
        "home_implied_prob": None,
        "away_implied_prob": None,
        # First half odds
        "fh_home_spread": None,
        "fh_home_spread_price": None,
        "fh_total": None,
        "fh_total_price": None,
    }
    
    if h2h_home:
        median_home = statistics.median(h2h_home)
        # Clip extreme values
        if abs(median_home) > 2000:
            median_home = 1000 * (1 if median_home > 0 else -1)
        result["home_ml"] = int(median_home)
        result["home_implied_prob"] = american_to_implied_prob(result["home_ml"])
    if h2h_away:
        median_away = statistics.median(h2h_away)
        # Clip extreme values
        if abs(median_away) > 2000:
            median_away = 1000 * (1 if median_away > 0 else -1)
        result["away_ml"] = int(median_away)
        result["away_implied_prob"] = american_to_implied_prob(result["away_ml"])
    if spreads_home:
        result["home_spread"] = sum(s["point"] for s in spreads_home) / len(spreads_home)
        # Use median for prices to handle outliers
        spread_prices = [s["price"] for s in spreads_home if s["price"] is not None]
        if spread_prices:
            result["home_spread_price"] = int(statistics.median(spread_prices))
    if totals:
        # Filter to Over outcomes for total line
        over_totals = [t for t in totals if t.get("side") != "Under"]
        if over_totals:
            result["total"] = sum(t["point"] for t in over_totals) / len(over_totals)
            # Extract Over prices
            over_prices = [t["price"] for t in over_totals if t.get("price") is not None]
            if over_prices:
                result["total_price"] = int(statistics.median(over_prices))
    
    # First half spread consensus
    if fh_spreads_home:
        result["fh_home_spread"] = sum(s["point"] for s in fh_spreads_home) / len(fh_spreads_home)
        fh_spread_prices = [s["price"] for s in fh_spreads_home if s["price"] is not None]
        if fh_spread_prices:
            result["fh_home_spread_price"] = int(statistics.median(fh_spread_prices))
    
    # First half total consensus
    if fh_totals:
        fh_over_totals = [t for t in fh_totals if t.get("side") != "Under"]
        if fh_over_totals:
            result["fh_total"] = sum(t["point"] for t in fh_over_totals) / len(fh_over_totals)
            fh_over_prices = [t["price"] for t in fh_over_totals if t.get("price") is not None]
            if fh_over_prices:
                result["fh_total_price"] = int(statistics.median(fh_over_prices))
    
    return result


async def fetch_todays_games(target_date: datetime.date) -> List[Dict]:
    """Fetch games for target date from The Odds API."""
    from src.ingestion import the_odds
    
    print("\n📡 Fetching games from The Odds API...")
    try:
        # Fetch main odds (h2h, spreads, totals) from the main endpoint
        all_games = await the_odds.fetch_odds(markets="h2h,spreads,totals")
        print(f"   Retrieved {len(all_games)} total NBA games")
        
        # Fetch first half markets from the event-specific endpoint
        # (First half markets are only available via /events/{id}/odds, not /odds)
        print("   Fetching first half markets...")
        for game in all_games:
            event_id = game.get("id")
            if event_id:
                try:
                    fh_data = await the_odds.fetch_event_odds(
                        event_id=event_id,
                        markets="spreads_h1,totals_h1"
                    )
                    # Merge first half bookmaker data into the game
                    if fh_data and fh_data.get("bookmakers"):
                        for fh_bm in fh_data["bookmakers"]:
                            # Find matching bookmaker in main game data
                            for bm in game.get("bookmakers", []):
                                if bm.get("key") == fh_bm.get("key"):
                                    # Add first half markets to this bookmaker
                                    bm.setdefault("markets", []).extend(fh_bm.get("markets", []))
                                    break
                            else:
                                # Bookmaker not in main data, add it
                                game.setdefault("bookmakers", []).append(fh_bm)
                except Exception as e:
                    # First half markets may not be available for all games
                    print(f"      [WARN] Could not fetch 1H markets for {game.get('away_team')} @ {game.get('home_team')}: {e}")
        
        # Filter to target date
        games = filter_games_for_date(all_games, target_date)
        print(f"   Filtered to {len(games)} games for {target_date.strftime('%A, %B %d')}")
        
        return games
    except Exception as e:
        print(f"   ❌ Error fetching from API: {e}")
        print("   Attempting to load from cached data...")
        
        # Try loading from cached file
        odds_dir = Path(settings.data_raw_dir) / "the_odds"
        json_files = sorted(odds_dir.glob("odds_*.json"), reverse=True)
        
        if json_files:
            with open(json_files[0], "r") as f:
                all_games = json.load(f)
            print(f"   Loaded {len(all_games)} games from {json_files[0].name}")
            games = filter_games_for_date(all_games, target_date)
            print(f"   Filtered to {len(games)} games for {target_date.strftime('%A, %B %d')}")
            return games
        
        return []


async def build_team_features(home_team: str, away_team: str) -> Optional[Dict]:
    """Build features using API-Basketball data."""
    try:
        from scripts.build_rich_features import RichFeatureBuilder
        
        builder = RichFeatureBuilder(league_id=12, season=settings.current_season)
        features = await builder.build_game_features(home_team, away_team)
        return features
    except Exception as e:
        print(f"      ⚠️  Feature building failed: {e}")
        return None


def create_simple_features(game: Dict, odds: Dict) -> Dict:
    """Create simple features when API data is unavailable."""
    home_prob = odds.get("home_implied_prob", 0.5)
    away_prob = odds.get("away_implied_prob", 0.5)
    spread = odds.get("home_spread", 0)
    
    if fh_model is None:
        raise RuntimeError("ImprovedFirstHalfModel is required; import failed.")

    # Calculate dynamic FH shares (requires live model)
    home_team = game.get("home_team", "Home")
    away_team = game.get("away_team", "Away")
    home_share, away_share = fh_model.calculate_dynamic_fh_shares(
        home_team, away_team
    )
    
    return {
        "home_ppg": 110,  # League average
        "away_ppg": 110,
        "predicted_margin": -spread if spread else 0,  # Invert spread for margin
        "predicted_total": odds.get("total", 220) or 220,
        "home_win_pct": home_prob,
        "away_win_pct": away_prob,
        "home_elo": 1500 + (home_prob - 0.5) * 400,
        "away_elo": 1500 + (away_prob - 0.5) * 400,
        "home_blowout_rate": 0.0,
        "away_blowout_rate": 0.0,
        "blowout_rate_diff": 0.0,
        "home_half_share": home_share,
        "away_half_share": away_share,
        "half_share_diff": home_share - away_share,
        "home_dominance_index": 0.0,
        "away_dominance_index": 0.0,
    }


def generate_rationale(
    play_type: str,
    pick: str,
    line: float,
    odds: int,
    edge: float,
    model_prob: float,
    features: Dict,
    betting_splits: Optional[Any],
    home_team: str,
    away_team: str,
    model_prediction: float,
    market_line: float,
    opening_line: Optional[float] = None,
    game_time: Optional[datetime] = None
) -> str:
    """
    Generate detailed rationale for a pick following the required format.
    
    Must include at least 3 of 6 categories:
    1. Market Context (High Priority) - line movement, timestamps
    2. Team Fundamentals (High Priority) - NBA metrics
    3. Situational Factors (Medium Priority) - rest, travel, home/away
    4. Market Sentiment & Sharp Action (Very High Priority) - betting splits, RLM
    5. Model Confidence (High Priority) - probability, EV, confidence
    6. Historical Context (Low Priority) - H2H, trends
    
    Returns bullet-point formatted string (newline-separated for Excel).
    """
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo
    
    rationale_bullets = []
    is_first_half = "1H" in play_type
    now_cst = datetime.now(ZoneInfo("America/Chicago"))
    
    # Calculate expected value
    if odds > 0:
        profit = odds / 100
    else:
        profit = 100 / abs(odds)
    ev = (model_prob * profit) - (1 - model_prob)
    ev_pct = ev * 100
    
    # ============================================================
    # 1. MARKET CONTEXT (High Priority)
    # ============================================================
    market_context = []
    
    if opening_line is not None and opening_line != market_line:
        line_movement = market_line - opening_line
        movement_direction = "up" if line_movement > 0 else "down"
        movement_magnitude = abs(line_movement)
        
        # Determine if movement is significant
        # Different thresholds: ≥1.0 for spreads, ≥0.5 for totals (per requirements)
        movement_threshold = 1.0 if "SPREAD" in play_type else 0.5
        if movement_magnitude >= movement_threshold:
            if "SPREAD" in play_type:
                # For spreads, determine which team benefited
                if line_movement < 0:  # Line moved more negative (toward home)
                    benefited_team = home_team
                else:  # Line moved more positive (toward away)
                    benefited_team = away_team
                market_context.append(
                    f"📈 Line moved from {opening_line:+.1f} to {market_line:+.1f} "
                    f"({movement_direction} {movement_magnitude:.1f} pts), suggesting sharp action on {benefited_team}."
                )
            elif "TOTAL" in play_type:
                market_context.append(
                    f"📈 Total moved from {opening_line:.1f} to {market_line:.1f} "
                    f"({movement_direction} {movement_magnitude:.1f} pts)."
                )
        
        # Check if movement occurred within 24 hours
        if game_time:
            try:
                if isinstance(game_time, str):
                    game_dt = parse_utc_time(game_time)
                else:
                    game_dt = game_time
                game_cst = to_cst(game_dt)
                hours_until_game = (game_cst - now_cst).total_seconds() / 3600
                
                if 0 < hours_until_game <= 24 and movement_magnitude >= 0.5:
                    market_context.append(
                        f"⏰ Movement occurred within 24 hours of tipoff, indicating late sharp action."
                    )
            except Exception:
                pass
    
    # ============================================================
    # 4. MARKET SENTIMENT & SHARP ACTION (Very High Priority)
    # ============================================================
    sharp_action = []
    
    if betting_splits:
        try:
            if "SPREAD" in play_type:
                # Fix: Compare full pick to team names, not just first word
                # Pick is the full team name (e.g., "Los Angeles Lakers"), not just first word
                is_home_pick = pick == home_team
                
                home_tickets = getattr(betting_splits, "spread_home_ticket_pct", 50)
                home_money = getattr(betting_splits, "spread_home_money_pct", 50)
                away_tickets = getattr(betting_splits, "spread_away_ticket_pct", 50)
                away_money = getattr(betting_splits, "spread_away_money_pct", 50)
                
                pick_tickets = home_tickets if is_home_pick else away_tickets
                pick_money = home_money if is_home_pick else away_money
                opp_tickets = away_tickets if is_home_pick else home_tickets
                
                # Reverse Line Movement detection
                rlm = getattr(betting_splits, "spread_rlm", False)
                sharp_side = getattr(betting_splits, "sharp_spread_side", None)
                
                if rlm and sharp_side:
                    if (is_home_pick and sharp_side == "home") or (not is_home_pick and sharp_side == "away"):
                        sharp_action.append(
                            f"💸 Reverse line movement detected: despite {opp_tickets:.0f}% of bets on opponent, "
                            f"line moved in favor of {pick} — classic sharp money signal."
                        )
                
                # Ticket vs Money divergence
                ticket_money_diff = abs(pick_tickets - pick_money)
                if ticket_money_diff >= 10:
                    if pick_money > pick_tickets:
                        sharp_action.append(
                            f"💸 Sharp money indicator: only {pick_tickets:.0f}% tickets but {pick_money:.0f}% money "
                            f"on {pick}, suggesting professional action."
                        )
                    else:
                        sharp_action.append(
                            f"💸 Public heavy: {pick_tickets:.0f}% tickets vs {pick_money:.0f}% money on {pick}."
                        )
                elif pick_tickets < 35:
                    sharp_action.append(
                        f"💸 Contrarian play: fading the public ({opp_tickets:.0f}% on opponent)."
                    )
                elif pick_tickets > 65:
                    sharp_action.append(
                        f"💸 Public consensus: {pick_tickets:.0f}% of bets on {pick}."
                    )

            elif "TOTAL" in play_type:
                is_over = "OVER" in pick.upper()
                
                over_tickets = getattr(betting_splits, "over_ticket_pct", 50)
                over_money = getattr(betting_splits, "over_money_pct", 50)
                under_tickets = getattr(betting_splits, "under_ticket_pct", 50)
                under_money = getattr(betting_splits, "under_money_pct", 50)
                
                pick_tickets = over_tickets if is_over else under_tickets
                pick_money = over_money if is_over else under_money
                opp_tickets = under_tickets if is_over else over_tickets
                
                # RLM for totals
                total_rlm = getattr(betting_splits, "total_rlm", False)
                sharp_total_side = getattr(betting_splits, "sharp_total_side", None)
                
                if total_rlm and sharp_total_side:
                    if (is_over and sharp_total_side == "over") or (not is_over and sharp_total_side == "under"):
                        sharp_action.append(
                            f"💸 Reverse line movement: {opp_tickets:.0f}% bets on {('UNDER' if is_over else 'OVER')}, "
                            f"but line moved toward {pick} — sharp action detected."
                        )
                
                ticket_money_diff = abs(pick_tickets - pick_money)
                if ticket_money_diff >= 10 and pick_money > pick_tickets:
                    sharp_action.append(
                        f"💸 Sharp money: {pick_money:.0f}% money vs {pick_tickets:.0f}% tickets on {pick}."
                    )
        except Exception:
            pass
    
    # ============================================================
    # 2. TEAM FUNDAMENTALS (High Priority) - NBA Metrics
    # ============================================================
    team_fundamentals = []
    
    if not is_first_half:
        home_ppg = features.get("home_ppg", 0)
        away_ppg = features.get("away_ppg", 0)
        home_papg = features.get("home_papg", 0)
        away_papg = features.get("away_papg", 0)
        
        if "SPREAD" in play_type:
            # Fix: Compare full pick to team names, not just first word
            # Pick is the full team name (e.g., "Los Angeles Lakers"), not just first word
            is_home_pick = pick == home_team
            pick_team_ppg = home_ppg if is_home_pick else away_ppg
            opp_team_papg = away_papg if is_home_pick else home_papg
            
            if pick_team_ppg > 0 and opp_team_papg > 0:
                team_fundamentals.append(
                    f"🧮 {pick} averages {pick_team_ppg:.1f} PPG and faces a defense allowing {opp_team_papg:.1f} PPG."
                )
        
        elif "TOTAL" in play_type:
            combined_ppg = home_ppg + away_ppg
            if combined_ppg > 0:
                team_fundamentals.append(
                    f"🧮 Combined offensive output: {home_team} {home_ppg:.1f} PPG + {away_team} {away_ppg:.1f} PPG = {combined_ppg:.1f} PPG."
                )
        
        # Offensive/Defensive efficiency
        home_elo = features.get("home_elo", 1500)
        away_elo = features.get("away_elo", 1500)
        if home_elo > 0 and away_elo > 0:
            elo_diff = home_elo - away_elo
            if abs(elo_diff) >= 50:
                stronger_team = home_team if elo_diff > 0 else away_team
                team_fundamentals.append(
                    f"🧮 {stronger_team} holds significant ELO advantage ({abs(elo_diff):.0f} points)."
                )
    
    # ============================================================
    # 3. SITUATIONAL FACTORS (Medium Priority)
    # ============================================================
    situational = []
    
    if not is_first_half:
        # Rest advantage
        rest_adj = features.get("rest_margin_adj", 0)
        if abs(rest_adj) >= 1.5:
            benefit_team = home_team if rest_adj > 0 else away_team
            home_rest = features.get("home_days_rest", 2)
            away_rest = features.get("away_days_rest", 2)
            if home_rest > 0 and away_rest > 0:
                if rest_adj > 0:
                    situational.append(
                        f"🌍 {home_team} has rest advantage: {home_rest} days rest vs {away_team}'s {away_rest} days ({abs(rest_adj):.1f} pt edge)."
                    )
                else:
                    situational.append(
                        f"🌍 {away_team} has rest advantage: {away_rest} days rest vs {home_team}'s {home_rest} days ({abs(rest_adj):.1f} pt edge)."
                    )
        
        # Pace factor for totals
        if "TOTAL" in play_type:
            pace = features.get("expected_pace_factor", 1.0)
            if pace > 1.02:
                situational.append(
                    f"🌍 High pace expected (factor: {pace:.2f}), favoring higher scoring."
                )
            elif pace < 0.98:
                situational.append(
                    f"🌍 Slow pace expected (factor: {pace:.2f}), favoring lower scoring."
                )
    
    # ============================================================
    # 5. MODEL CONFIDENCE (High Priority)
    # ============================================================
    model_confidence = []
    
    model_confidence.append(
        f"📊 Model assigns {model_prob*100:.1f}% probability to {pick} with {edge:+.1f} pt edge."
    )
    
    if abs(ev_pct) >= 5:
        model_confidence.append(
            f"📊 Expected value: {ev_pct:+.1f}% based on current odds ({odds:+d})."
        )
    
    # Confidence strength
    if model_prob >= 0.65 or model_prob <= 0.35:
        model_confidence.append(
            f"📊 High-confidence play: probability {'exceeds' if model_prob >= 0.65 else 'below'} 65% threshold."
        )
    
    # ============================================================
    # 6. HISTORICAL CONTEXT (Low Priority)
    # ============================================================
    historical = []
    
    if not is_first_half:
        h2h_win_rate = features.get("h2h_win_rate", None)
        if h2h_win_rate is not None and h2h_win_rate != 0.5:
            if h2h_win_rate > 0.6:
                historical.append(
                    f"🕰️ {home_team} has won {h2h_win_rate*100:.0f}% of recent head-to-head meetings."
                )
            elif h2h_win_rate < 0.4:
                historical.append(
                    f"🕰️ {away_team} has won {(1-h2h_win_rate)*100:.0f}% of recent head-to-head meetings."
                )
    
    # ============================================================
    # ASSEMBLE RATIONALE (Must include at least 3 categories)
    # Priority: Market Context, Sharp Action, Model Confidence
    # ============================================================
    
    # Always include Model Confidence (required)
    rationale_bullets.extend(model_confidence[:1])  # Top model confidence item
    
    # High priority: Market Context and Sharp Action
    if market_context:
        rationale_bullets.extend(market_context[:1])  # Top market context item
    if sharp_action:
        rationale_bullets.extend(sharp_action[:1])  # Top sharp action item
    
    # Count categories used so far
    categories_used = sum([
        1 if model_confidence else 0,
        1 if market_context and market_context[:1] else 0,
        1 if sharp_action and sharp_action[:1] else 0
    ])
    
    # Fill remaining slots to reach minimum 3 categories
    if categories_used < 3 and team_fundamentals:
        rationale_bullets.extend(team_fundamentals[:1])
        categories_used += 1
    
    if categories_used < 3 and situational:
        rationale_bullets.extend(situational[:1])
        categories_used += 1
    
    if categories_used < 3 and historical:
        rationale_bullets.extend(historical[:1])
        categories_used += 1
    
    # Ensure we have at least 3 items (can repeat categories if needed)
    while len(rationale_bullets) < 3:
        if market_context and len(market_context) > 1:
            rationale_bullets.append(market_context[1])
        elif team_fundamentals and len(team_fundamentals) > 1:
            rationale_bullets.append(team_fundamentals[1])
        elif situational and len(situational) > 1:
            rationale_bullets.append(situational[1])
        elif model_confidence and len(model_confidence) > 1:
            rationale_bullets.append(model_confidence[1])
        else:
            break
    
    # Return as newline-separated bullet points (for Excel cell formatting)
    return "\n".join(rationale_bullets[:3])  # Limit to 3 sentences as per requirements


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
    
    Args:
        features: Full game features
        fh_features: First half features
        odds: Market odds
        game: Game information
        betting_splits: Betting splits data (optional)
        edge_thresholds: Dynamic edge thresholds by bet type (optional)
    """
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
    
    if fg_predicted_margin is None:
        raise ValueError(
            f"Required feature 'predicted_margin' not available for {home_team} vs {away_team}. "
            f"Cannot calculate edge without model predictions."
        )
    
    if fg_predicted_total is None:
        raise ValueError(
            f"Required feature 'predicted_total' not available for {home_team} vs {away_team}. "
            f"Cannot calculate edge without model predictions."
        )
    fg_home_score = features.get("home_ppg", 110)
    fg_away_score = features.get("away_ppg", 110)
    
    # Calculate predicted scores for clarity
    fg_pred_home_score = (fg_predicted_total + fg_predicted_margin) / 2
    fg_pred_away_score = (fg_predicted_total - fg_predicted_margin) / 2
    
    # Market odds (extract before first half calculations)
    fg_market_spread = odds.get("home_spread")
    fg_market_total = odds.get("total")
    
    if fg_market_spread is None:
        raise ValueError(
            f"Required market data 'home_spread' not available for {home_team} vs {away_team}. "
            f"Cannot calculate edge without market spread."
        )
    
    if fg_market_total is None:
        raise ValueError(
            f"Required market data 'total' not available for {home_team} vs {away_team}. "
            f"Cannot calculate edge without market total."
        )
    fg_home_ml = odds.get("home_ml")
    fg_away_ml = odds.get("away_ml")
    
    # Extract actual market odds (required for accurate edge calculation)
    fg_spread_odds = odds.get("home_spread_price")
    fg_total_odds = odds.get("total_price")
    
    if fg_spread_odds is None:
        raise ValueError(
            f"Required market data 'home_spread_price' not available for {home_team} vs {away_team}. "
            f"Cannot calculate accurate edge without spread odds."
        )
    
    if fg_total_odds is None:
        raise ValueError(
            f"Required market data 'total_price' not available for {home_team} vs {away_team}. "
            f"Cannot calculate accurate edge without total odds."
        )
    
    # First Half Analysis (use dynamic team-specific share estimates)
    if fh_model is None:
        raise RuntimeError("ImprovedFirstHalfModel is required; import failed.")

    # Get dynamic shares based on team tendencies
    home_share, away_share = fh_model.calculate_dynamic_fh_shares(
        home_team,
        away_team,
        is_back_to_back=features.get("is_back_to_back"),
        recent_form=features.get("recent_form")
    )
    
    # Calculate FH scores with dynamic shares
    fh_home_score = fg_pred_home_score * home_share
    fh_away_score = fg_pred_away_score * away_share
    
    # Use calibrated margin adjustment from enhanced model
    pace_factor = features.get("expected_pace_factor", 1.0)
    fh_predicted_margin, margin_std = fh_model.calibrate_fh_margin(
        fg_predicted_margin, home_share, away_share, pace_factor
    )
    
    # Calculate FH total with variance
    fh_predicted_total, total_std = fh_model.calculate_fh_total(
        fg_predicted_total, home_share, away_share, pace_factor
    )
    
    fh_pred_home_score = (fh_predicted_total + fh_predicted_margin) / 2
    fh_pred_away_score = (fh_predicted_total - fh_predicted_margin) / 2
    
    # First half market odds - required for first half analysis
    fh_market_spread = odds.get("fh_home_spread")
    fh_spread_odds = odds.get("fh_home_spread_price")
    fh_market_total = odds.get("fh_total")
    fh_total_odds = odds.get("fh_total_price")
    
    # First half markets are optional - if not available, skip first half analysis
    # Do not estimate as estimates are unreliable
    if fh_market_spread is None or fh_spread_odds is None:
        fh_market_spread = None
        fh_spread_odds = None
        print(f"      [WARN] First half spread market not available - skipping 1H spread analysis")
    
    if fh_market_total is None or fh_total_odds is None:
        fh_market_total = None
        fh_total_odds = None
        print(f"      [WARN] First half total market not available - skipping 1H total analysis")
    
    result = {
        "full_game": {},
        "first_half": {},
        "top_plays": []
    }
    
    # === FULL GAME ===
    # Spread
    # Edge calculation: predicted_margin vs market expectation
    # home_spread convention: -3.5 means home is favored by 3.5 (home gives 3.5 points)
    # Market expects: home_score - away_score = 3.5 (home wins by 3.5)
    # So market_expected_margin = -home_spread
    # Edge = predicted_margin - market_expected_margin = predicted_margin - (-home_spread) = predicted_margin + home_spread
    # But if home_spread = -3.5, then: edge = predicted_margin + (-3.5) = predicted_margin - 3.5
    # This means: if we predict margin > 3.5, edge > 0, we like home
    # The original formula was: edge = predicted_margin - (-home_spread)
    # Which expands to: edge = predicted_margin + home_spread (same as above)
    # So the original was correct! But let's make it clearer:
    market_expected_margin = -fg_market_spread if fg_market_spread is not None else 0
    fg_spread_edge = fg_predicted_margin - market_expected_margin
    # Positive edge means we like home more than market, negative means we like away more
    fg_spread_pick = home_team if fg_spread_edge > 0 else away_team
    # More realistic confidence: cap at 90%
    # Adjusted divisor to 5.0 to better reflect betting value (5pt edge = 100% confidence base)
    fg_spread_confidence = min(abs(fg_spread_edge) / 5.0, 0.90) if abs(fg_spread_edge) >= 1.0 else 0
    blowout_diff = features.get("blowout_rate_diff", 0.0)
    dominance_bonus = min(abs(blowout_diff) * 2, 0.1)
    fg_spread_confidence = min(fg_spread_confidence + dominance_bonus, 0.95)
    
    # Calculate win probability for spread (relative to PICK)
    # Assuming 1 point edge ≈ 3% probability shift
    # Use abs(edge) because we always pick the side with the edge
    fg_spread_win_prob = 0.5 + (abs(fg_spread_edge) * 0.03)
    fg_spread_win_prob = max(0.51, min(0.80, fg_spread_win_prob)) if abs(fg_spread_edge) >= 0.5 else 0.5
    
    # Adjust for key numbers if available
    if fh_model and fg_market_spread is not None:
        key_adjustment = fh_model.adjust_for_key_numbers(fg_predicted_margin, fg_market_spread)
        fg_spread_confidence *= key_adjustment

    # Calculate correct pick_line based on which team we're picking
    fg_pick_line = None
    spread_threshold = edge_thresholds.get("spread", 2.0)
    if abs(fg_spread_edge) >= spread_threshold:
        if fg_spread_pick == home_team:
            fg_pick_line = fg_market_spread
        else:  # picking away team
            fg_pick_line = -fg_market_spread if fg_market_spread is not None else None
    
    # Extract opening line and game time for rationale
    opening_spread = None
    if betting_splits:
        opening_spread = getattr(betting_splits, "spread_open", None)
    game_time = game.get("commence_time")
    
    # Build detailed spread rationale
    fg_spread_rationale = generate_rationale(
        play_type="FG SPREAD",
        pick=fg_spread_pick,
        line=fg_pick_line if fg_pick_line is not None else fg_market_spread,
        odds=fg_spread_odds,
        edge=fg_spread_edge,
        model_prob=fg_spread_win_prob,
        features=features,
        betting_splits=betting_splits,
        home_team=home_team,
        away_team=away_team,
        model_prediction=fg_predicted_margin,
        market_line=fg_market_spread,
        opening_line=opening_spread,
        game_time=game_time
    )
    
    # Total
    fg_total_edge = fg_predicted_total - fg_market_total
    fg_total_pick = "OVER" if fg_total_edge > 0 else "UNDER"
    total_threshold = edge_thresholds.get("total", 3.0)
    # Cap total confidence at 90%
    # Adjusted divisor to 10.0 (10pt edge = 100% confidence base)
    fg_total_confidence = min(abs(fg_total_edge) / 10.0, 0.90) if abs(fg_total_edge) >= (total_threshold * 0.75) else 0
    
    # FIX: Pace volatility should REDUCE confidence, not increase it
    pace_volatility = min(
        (features.get("home_blowout_rate", 0.0) + features.get("away_blowout_rate", 0.0)), 0.25
    )
    # Higher volatility = lower confidence
    volatility_reduction = 1.0 - (pace_volatility * 0.5)  # Up to 12.5% reduction
    fg_total_confidence = fg_total_confidence * volatility_reduction
    
    # Calculate over/under probability
    fg_total_win_prob = 0.5 + (abs(fg_total_edge) * 0.025) + min(pace_volatility * 0.05, 0.05)
    fg_total_win_prob = max(0.51, min(0.75, fg_total_win_prob)) if abs(fg_total_edge) >= 1.0 else 0.5
    
    # Extract opening total for rationale
    opening_total = None
    if betting_splits:
        opening_total = getattr(betting_splits, "total_open", None)
    
    # Enhanced total rationale
    fg_total_rationale = generate_rationale(
        play_type="FG TOTAL",
        pick=fg_total_pick,
        line=fg_market_total,
        odds=fg_total_odds,
        edge=fg_total_edge,
        model_prob=fg_total_win_prob,
        features=features,
        betting_splits=betting_splits,
        home_team=home_team,
        away_team=away_team,
        model_prediction=fg_predicted_total,
        market_line=fg_market_total,
        opening_line=opening_total,
        game_time=game_time
    )
    
    # Moneyline - Require enhanced model (no silent fallback)
    if ml_model is None:
        raise RuntimeError("ImprovedMoneylineModel is required; import failed.")

    home_elo = features.get("home_elo", 1500)
    away_elo = features.get("away_elo", 1500)
    pace_factor = features.get("expected_pace_factor", 1.0)
    fg_model_home_ml, fg_model_away_ml = ml_model.spread_to_moneyline_enhanced(
        fg_predicted_margin, home_elo, away_elo, pace_factor
    )
    fg_ml_edge_home = None
    fg_ml_edge_away = None
    fg_ml_pick = None
    fg_ml_confidence = 0
    fg_ml_rationale = "No moneyline value found."
    
    def _ml_allowed(model_prob: float, edge: Optional[float]) -> bool:
        if edge is None:
            return False
        # Moneyline requires either strong model confidence OR significant edge
        # Increased from 32% to 40% to avoid too many underdog picks
        if model_prob >= 0.40:
            return True
        # Or require at least 8% edge (increased from 3% to be more selective)
        return edge >= 0.08

    dominance_trigger = abs(blowout_diff) >= 0.04

    if fg_home_ml and fg_away_ml:
        # Calculate value: (model prob * payout) - 1
        model_home_prob = american_to_implied_prob(fg_model_home_ml)
        market_home_prob = american_to_implied_prob(fg_home_ml)
        fg_ml_edge_home = model_home_prob - market_home_prob
        
        model_away_prob = american_to_implied_prob(fg_model_away_ml)
        market_away_prob = american_to_implied_prob(fg_away_ml)
        fg_ml_edge_away = model_away_prob - market_away_prob
        
        home_ev = ml_model.calculate_expected_value(model_home_prob, fg_home_ml)
        away_ev = ml_model.calculate_expected_value(model_away_prob, fg_away_ml)
        
        home_allowed = _ml_allowed(model_home_prob, fg_ml_edge_home)
        away_allowed = _ml_allowed(model_away_prob, fg_ml_edge_away)

        if home_ev > away_ev and home_allowed:
            fg_ml_pick = home_team
            # Cap ML confidence at 75%
            fg_ml_confidence = min(fg_ml_edge_home * 2.5 + dominance_bonus, 0.75)
            fg_ml_rationale = f"Model gives {home_team} {model_home_prob*100:.1f}% win probability vs market's {market_home_prob*100:.1f}%. "
            fg_ml_rationale += f"Expected value: {home_ev*100:+.1f}% on {home_team} ({fg_home_ml:+d})."
        elif away_allowed and (away_ev > home_ev or not home_allowed):
            fg_ml_pick = away_team
            fg_ml_confidence = min(fg_ml_edge_away * 2.5 + dominance_bonus, 0.75)
            fg_ml_rationale = f"Model gives {away_team} {model_away_prob*100:.1f}% win probability vs market's {market_away_prob*100:.1f}%. "
            fg_ml_rationale += f"Expected value: {away_ev*100:+.1f}% on {away_team} ({fg_away_ml:+d})."
        else:
            fg_ml_rationale = f"No significant edge. Model: {home_team} {model_home_prob*100:.1f}% vs Market: {market_home_prob*100:.1f}%."
    
    # fg_pick_line already calculated above for rationale generation
    
    result["full_game"] = {
        "spread": {
            "model_margin": fg_predicted_margin,
            "model_home_score": fg_pred_home_score,
            "model_away_score": fg_pred_away_score,
            "market_line": fg_market_spread,
            "market_odds": fg_spread_odds,
            "edge": fg_spread_edge,
            "pick": fg_spread_pick if abs(fg_spread_edge) >= spread_threshold else None,
            "pick_line": fg_pick_line,
            "pick_odds": fg_spread_odds if abs(fg_spread_edge) >= spread_threshold else None,
            "confidence": fg_spread_confidence,
            "win_probability": fg_spread_win_prob if abs(fg_spread_edge) >= spread_threshold else 0.5,
            "rationale": fg_spread_rationale
        }
    }
    
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
        "rationale": fg_total_rationale
    }
    result["full_game"]["moneyline"] = {
        "model_home_odds": fg_model_home_ml,
        "model_away_odds": fg_model_away_ml,
        "market_home_odds": fg_home_ml,
        "market_away_odds": fg_away_ml,
        "model_home_prob": american_to_implied_prob(fg_model_home_ml) if fg_model_home_ml else 0.5,
        "model_away_prob": american_to_implied_prob(fg_model_away_ml) if fg_model_away_ml else 0.5,
        "market_home_prob": american_to_implied_prob(fg_home_ml) if fg_home_ml else 0.5,
        "market_away_prob": american_to_implied_prob(fg_away_ml) if fg_away_ml else 0.5,
        "edge_home": fg_ml_edge_home,
        "edge_away": fg_ml_edge_away,
        "pick": fg_ml_pick,
        "confidence": fg_ml_confidence,
        "rationale": fg_ml_rationale
    }
    
    # === FIRST HALF ===
    # CALIBRATION NOTE: FH predictions have R²=44% from FG, so inherently more uncertain
    # Historical data shows FH spread edge → win prob slope is ~2% per point
    # Only calculate if first half markets are available (do not estimate)
    if fh_market_spread is not None and fh_spread_odds is not None:
        fh_market_expected_margin = -fh_market_spread
        fh_spread_edge = fh_predicted_margin - fh_market_expected_margin
        fh_spread_pick = home_team if fh_spread_edge > 0 else away_team
        # Cap FH confidence at 70% (lower than FG due to higher variance, R²=44%)
        # Divisor 6.0 (6pt edge = 100% base, capped at 70%)
        fh_spread_confidence = min(abs(fh_spread_edge) / 6.0, 0.70) if abs(fh_spread_edge) >= 1.5 else 0
        
        # Adjust for key numbers in FH
        if fh_model and fh_market_spread is not None:
            fh_key_adjustment = fh_model.adjust_for_key_numbers(fh_predicted_margin, fh_market_spread)
            fh_spread_confidence *= fh_key_adjustment
        
        # Win prob: calibrated to ~2% per point (historical shows 2-3%)
        fh_spread_win_prob = 0.5 + (abs(fh_spread_edge) * 0.02)
        fh_spread_win_prob = max(0.51, min(0.65, fh_spread_win_prob)) if abs(fh_spread_edge) >= 0.5 else 0.5
        
        # Calculate correct first half pick_line based on which team we're picking
        fh_pick_line = None
        fh_spread_threshold = edge_thresholds.get("1h_spread", 1.5)
        if abs(fh_spread_edge) >= fh_spread_threshold:
            if fh_spread_pick == home_team:
                fh_pick_line = fh_market_spread
            else:  # picking away team
                fh_pick_line = -fh_market_spread
        
        # First half opening line (use full game opening as proxy if 1H not available)
        fh_opening_spread = opening_spread  # Use FG opening as proxy
        
        fh_spread_rationale = generate_rationale(
            play_type="1H SPREAD",
            pick=fh_spread_pick,
            line=fh_pick_line if fh_pick_line is not None else fh_market_spread,
            odds=fh_spread_odds,
            edge=fh_spread_edge,
            model_prob=fh_spread_win_prob,
            features=features,
            betting_splits=betting_splits,
            home_team=home_team,
            away_team=away_team,
            model_prediction=fh_predicted_margin,
            market_line=fh_market_spread,
            opening_line=fh_opening_spread,
            game_time=game_time
        )
    else:
        # First half spread not available - skip analysis
        fh_spread_edge = None
        fh_spread_pick = None
        fh_spread_confidence = 0
        fh_spread_win_prob = 0.5
        fh_pick_line = None
        fh_spread_rationale = "First half spread market not available - analysis skipped."
    
    if fh_market_total is not None and fh_total_odds is not None:
        fh_total_edge = fh_predicted_total - fh_market_total
        fh_total_pick = "OVER" if fh_total_edge > 0 else "UNDER"
        fh_total_threshold = edge_thresholds.get("1h_total", 2.0)
        # Cap FH total confidence at 70% (lower than FG due to higher variance)
        fh_total_confidence = min(abs(fh_total_edge) / 8.0, 0.70) if abs(fh_total_edge) >= fh_total_threshold else 0
        
        # Win prob: calibrated to ~3.5% per point (matches historical data well)
        fh_total_win_prob = 0.5 + (abs(fh_total_edge) * 0.035)
        fh_total_win_prob = max(0.51, min(0.70, fh_total_win_prob)) if abs(fh_total_edge) >= 1.0 else 0.5
        
        average_half_share = max(0.35, min(0.55, (home_share + away_share) / 2))
        # First half opening total (use full game opening as proxy if 1H not available)
        fh_opening_total = opening_total  # Use FG opening as proxy
        
        fh_total_rationale = generate_rationale(
            play_type="1H TOTAL",
            pick=fh_total_pick,
            line=fh_market_total,
            odds=fh_total_odds,
            edge=fh_total_edge,
            model_prob=fh_total_win_prob,
            features=features,
            betting_splits=betting_splits,
            home_team=home_team,
            away_team=away_team,
            model_prediction=fh_predicted_total,
            market_line=fh_market_total,
            opening_line=fh_opening_total,
            game_time=game_time
        )
    else:
        # First half total not available - skip analysis
        fh_total_edge = None
        fh_total_pick = None
        fh_total_confidence = 0
        fh_total_win_prob = 0.5
        fh_total_rationale = "First half total market not available - analysis skipped."
    
    fh_model_home_ml, fh_model_away_ml = spread_to_moneyline(fh_predicted_margin)
    fh_ml_pick = None
    fh_ml_rationale = f"Model projects "
    
    if abs(fh_predicted_margin) >= 1.5:
        fh_ml_pick = home_team if fh_predicted_margin > 0 else away_team
        leader = home_team if fh_predicted_margin > 0 else away_team
        fh_ml_prob = american_to_implied_prob(fh_model_home_ml if fh_predicted_margin > 0 else fh_model_away_ml)
        fh_ml_rationale += f"{leader} to lead at halftime ({fh_ml_prob*100:.1f}% probability based on {abs(fh_predicted_margin):.1f} pt edge)."
    else:
        fh_ml_rationale += "tight first half, no clear leader projected."
    
    result["first_half"] = {
        "spread": {
            "model_margin": fh_predicted_margin,
            "model_home_score": fh_pred_home_score,
            "model_away_score": fh_pred_away_score,
            "market_line": fh_market_spread,
            "market_odds": fh_spread_odds,
            "edge": fh_spread_edge,
            "pick": fh_spread_pick if fh_spread_edge is not None and abs(fh_spread_edge) >= fh_spread_threshold else None,
            "pick_line": fh_pick_line,
            "pick_odds": fh_spread_odds if fh_spread_edge is not None and abs(fh_spread_edge) >= fh_spread_threshold else None,
            "confidence": fh_spread_confidence,
            "win_probability": fh_spread_win_prob if fh_spread_edge is not None and abs(fh_spread_edge) >= fh_spread_threshold else 0.5,
            "rationale": fh_spread_rationale
        },
        "total": {
            "model_total": fh_predicted_total,
            "market_line": fh_market_total,
            "market_odds": fh_total_odds,
            "edge": fh_total_edge,
            "pick": fh_total_pick if fh_total_edge is not None and abs(fh_total_edge) >= fh_total_threshold else None,
            "pick_line": fh_market_total if fh_total_edge is not None and abs(fh_total_edge) >= fh_total_threshold else None,
            "pick_odds": fh_total_odds if fh_total_edge is not None and abs(fh_total_edge) >= fh_total_threshold else None,
            "confidence": fh_total_confidence,
            "win_probability": fh_total_win_prob if fh_total_edge is not None and abs(fh_total_edge) >= fh_total_threshold else 0.5,
            "rationale": fh_total_rationale
        },
        "moneyline": {
            "model_home_odds": fh_model_home_ml,
            "model_away_odds": fh_model_away_ml,
            "model_home_prob": american_to_implied_prob(fh_model_home_ml),
            "model_away_prob": american_to_implied_prob(fh_model_away_ml),
            "pick": fh_ml_pick,
            "rationale": fh_ml_rationale
        }
    }
    
    # Import betting utilities
    from src.modeling.betting import (
        BetRecommendation,
        annotate_value_bets,
    )
    
    # Compile all plays with bet recommendations
    all_plays = []
    bet_recommendations = []
    
    # Full game spread
    if result["full_game"]["spread"]["pick"]:
        spread_play = result["full_game"]["spread"]
        model_prob = spread_play.get("win_probability", 0.5)
        pick_line = spread_play.get("pick_line")
        
        # Bet sizing is handled externally; this model provides probabilities and EVs.
        bet_rec = {
            "pick": f"{result['full_game']['spread']['pick']} {pick_line:+.1f}" if pick_line is not None else result['full_game']['spread']['pick'],
            "line": pick_line if pick_line is not None else fg_market_spread,
            "odds": fg_spread_odds,
            "model_probability": model_prob,
            "edge": spread_play["edge"],
            "expected_value": (model_prob * (100/110) - (1-model_prob)) if fg_spread_odds == -110 else 0,
            "rationale": spread_play["rationale"],
        }
        
        play = {
            "type": "FG Spread",
            "pick": f"{result['full_game']['spread']['pick']} {pick_line:+.1f}" if pick_line is not None else result['full_game']['spread']['pick'],
            "confidence": spread_play["confidence"],
            "edge": spread_play["edge"],
            "model_probability": model_prob,
            "is_high_confidence": model_prob > 0.60 or model_prob < 0.40,
            "rationale": spread_play["rationale"],
            "bet_recommendation": bet_rec,
            "correlation_type": "spread",
        }
        all_plays.append(play)
        if bet_rec:
            bet_recommendations.append(bet_rec)
    
    # Full game total
    if result["full_game"]["total"]["pick"]:
        total_play = result["full_game"]["total"]
        model_prob = total_play.get("win_probability", 0.5)
        
        # Bet sizing is handled externally; this model provides probabilities and EVs.
        bet_rec = {
            "pick": f"{total_play['pick']} {fg_market_total:.1f}",
            "line": fg_market_total,
            "odds": fg_total_odds,
            "model_probability": model_prob,
            "edge": total_play["edge"],
            "expected_value": (model_prob * (100/110) - (1-model_prob)) if fg_total_odds == -110 else 0,
            "rationale": total_play["rationale"],
        }
        
        play = {
            "type": "FG Total",
            "pick": f"{total_play['pick']} {fg_market_total:.1f}",
            "confidence": total_play["confidence"],
            "edge": total_play["edge"],
            "model_probability": model_prob,
            "is_high_confidence": model_prob > 0.60 or model_prob < 0.40,
            "rationale": total_play["rationale"],
            "bet_recommendation": bet_rec,
            "correlation_type": "total",
        }
        all_plays.append(play)
        if bet_rec:
            bet_recommendations.append(bet_rec)
    
    # Moneyline
    if result["full_game"]["moneyline"]["pick"]:
        ml_play = result["full_game"]["moneyline"]
        ml_odds = fg_home_ml if ml_play["pick"] == home_team else fg_away_ml
        model_prob = ml_play.get("model_home_prob", 0.5) if ml_play["pick"] == home_team else ml_play.get("model_away_prob", 0.5)
        
        # Calculate EV for ML
        if ml_odds > 0:
            profit = ml_odds / 100
        else:
            profit = 100 / abs(ml_odds)
        ml_ev = (model_prob * profit) - (1 - model_prob)
        
        # Bet sizing is handled externally; this model provides probabilities and EVs.
        bet_rec = {
            "pick": f"{ml_play['pick']} ({ml_odds:+d})",
            "line": 0,  # Moneyline has no line
            "odds": ml_odds,
            "model_probability": model_prob,
            "edge": ml_play["edge_home"] if ml_play["pick"] == home_team else ml_play["edge_away"],
            "expected_value": ml_ev,
            "rationale": ml_play["rationale"],
        }
        
        play = {
            "type": "FG Moneyline",
            "pick": f"{ml_play['pick']} ({ml_odds:+d})",
            "confidence": ml_play["confidence"],
            "edge": ml_play["edge_home"] if ml_play["pick"] == home_team else ml_play["edge_away"],
            "model_probability": model_prob,
            "is_high_confidence": model_prob > 0.60 or model_prob < 0.40,
            "rationale": ml_play["rationale"],
            "bet_recommendation": bet_rec,
            "correlation_type": "ml",
        }
        all_plays.append(play)
        if bet_rec:
            bet_recommendations.append(bet_rec)
    
    # First half spread (only if market available)
    if result["first_half"]["spread"]["pick"] and result["first_half"]["spread"]["edge"] is not None:
        fh_spread = result["first_half"]["spread"]
        model_prob = fh_spread.get("win_probability", 0.5)
        fh_pick_line = fh_spread.get("pick_line")
        
        play = {
            "type": "1H Spread",
            "pick": f"{fh_spread['pick']} {fh_pick_line:+.1f}" if fh_pick_line is not None else fh_spread['pick'],
            "confidence": fh_spread["confidence"],
            "edge": fh_spread["edge"],
            "model_probability": model_prob,
            "is_high_confidence": model_prob > 0.60 or model_prob < 0.40,
            "rationale": fh_spread["rationale"],
        }
        all_plays.append(play)
    
    # First half total (only if market available)
    if result["first_half"]["total"]["pick"] and result["first_half"]["total"]["edge"] is not None:
        fh_total = result["first_half"]["total"]
        model_prob = fh_total.get("win_probability", 0.5)
        
        play = {
            "type": "1H Total",
            "pick": f"{fh_total['pick']} {fh_total['market_line']:.1f}" if fh_total.get('market_line') is not None else fh_total['pick'],
            "confidence": fh_total["confidence"],
            "edge": fh_total["edge"],
            "model_probability": model_prob,
            "is_high_confidence": model_prob > 0.60 or model_prob < 0.40,
            "rationale": fh_total["rationale"],
        }
        all_plays.append(play)
    
    # Build correlated_bets dictionary from plays for correlation analysis
    correlated_bets = {
        "spread": None,
        "total": None,
        "ml": None,
    }
    for play in all_plays:
        corr_type = play.get("correlation_type")
        if corr_type == "spread":
            correlated_bets["spread"] = play
        elif corr_type == "total":
            correlated_bets["total"] = play
        elif corr_type == "ml":
            correlated_bets["ml"] = play
    
    # Apply correlation adjustments if multiple bets on same game
    if sum(1 for b in correlated_bets.values() if b) > 1:
        # Check for highly correlated bets (spread + ML on same team)
        if correlated_bets["spread"] and correlated_bets["ml"]:
            spread_pick = correlated_bets["spread"]["pick"].split()[0]  # Extract team name
            ml_pick = correlated_bets["ml"]["pick"].split()[0]  # Extract team name
            
            if spread_pick == ml_pick:
                # Highly correlated - reduce confidence on the weaker play
                for play in all_plays:
                    if play.get("correlation_type") == "ml":
                        play["confidence"] *= 0.7  # Reduce ML confidence by 30%
                        play["correlation_warning"] = "Reduced confidence due to correlation with spread bet"
        
        # Warn about multiple bets on same game
        total_bets = sum(1 for b in correlated_bets.values() if b)
        if total_bets >= 2:
            for play in all_plays:
                if play.get("correlation_type"):
                    play["correlated_game"] = True
                    play["total_game_bets"] = total_bets
    
    # Sort by adjusted confidence
    all_plays.sort(key=lambda x: x["confidence"], reverse=True)
    
    # Separate high-confidence plays
    high_conf_plays = [p for p in all_plays if p.get("is_high_confidence", False)]
    
    result["top_plays"] = all_plays[:3]  # Top 3 overall
    result["high_confidence_plays"] = high_conf_plays[:5]  # Top 5 high-confidence
    result["bet_recommendations"] = sorted(bet_recommendations, key=lambda x: x["expected_value"], reverse=True)
    
    return result


def calculate_edge(features: Dict, odds: Dict) -> Dict:
    """Calculate betting edge based on model vs market (legacy compatibility)."""
    result = {
        "spread_edge": None,
        "total_edge": None,
        "ml_edge": None,
        "recommendation": None,
        "confidence": None,
    }
    
    predicted_margin = features.get("predicted_margin", 0)
    market_spread = odds.get("home_spread")
    
    if market_spread is not None:
        # Positive edge = model likes home more than market
        # market_spread is from home perspective: -3.5 means home favored by 3.5
        # So market expects margin = -market_spread
        market_expected_margin = -market_spread
        result["spread_edge"] = predicted_margin - market_expected_margin
        
        # Recommendation based on edge
        if abs(result["spread_edge"]) >= 2:
            if result["spread_edge"] > 0:
                result["recommendation"] = "HOME"
                result["confidence"] = min(abs(result["spread_edge"]) / 5, 1.0)
            else:
                result["recommendation"] = "AWAY"
                result["confidence"] = min(abs(result["spread_edge"]) / 5, 1.0)
    
    predicted_total = features.get("predicted_total", 220)
    market_total = odds.get("total")
    
    if market_total:
        result["total_edge"] = predicted_total - market_total
    
    return result


def generate_comprehensive_text_report(analysis: List[Dict], target_date: datetime.date) -> str:
    """Generate a comprehensive text-based report with first half and full game analysis."""
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
        lines.append(f"Time: {game['time_cst']}")
        lines.append("")
        
        comp_edge = game.get("comprehensive_edge", {})
        
        if not comp_edge:
            lines.append("   ⚠️  Analysis not available")
            lines.append("")
            continue
        
        # === HIGH-CONFIDENCE PLAYS (7.3% ROI strategy) ===
        high_conf = comp_edge.get("high_confidence_plays", [])
        if high_conf:
            lines.append("[HIGH-CONFIDENCE PLAYS] (Model probability >60% or <40%):")
            lines.append("   Expected ROI: +7.3% (vs +4.4% for all bets)")
            lines.append("")
            for idx, play in enumerate(high_conf, 1):
                conf_pct = play["confidence"] * 100
                prob_pct = play.get("model_probability", 0.5) * 100
                bet_rec = play.get("bet_recommendation")
                
                lines.append(f"   {idx}. {play['type']}: {play['pick']}")
                lines.append(f"      Model Probability: {prob_pct:.1f}% | Edge: {play['edge']:+.1f} pts")
                if bet_rec:
                    lines.append(f"      Expected Value: {bet_rec['expected_value']:+.2%}")
                lines.append(f"      {play['rationale']}")
            lines.append("")
        
        # === TOP PLAYS (All) ===
        top_plays = comp_edge.get("top_plays", [])
        if top_plays:
            lines.append("[TOP PLAYS] (All confidence levels):")
            for idx, play in enumerate(top_plays, 1):
                conf_pct = play["confidence"] * 100
                hc_marker = "[HIGH-CONF]" if play.get("is_high_confidence", False) else ""
                lines.append(f"   {idx}. {play['type']}: {play['pick']} {hc_marker}")
                lines.append(f"      Edge: {play['edge']:+.1f} pts | Confidence: {conf_pct:.1f}%")
                lines.append(f"      {play['rationale']}")
            lines.append("")
        
        # === FULL GAME ANALYSIS ===
        fg = comp_edge.get("full_game", {})
        lines.append("📊 FULL GAME ANALYSIS:")
        lines.append("")
        
        # Spread
        if fg.get("spread"):
            sp = fg["spread"]
            lines.append(f"   SPREAD:")
            lines.append(f"      Market Line: {game['home_team']} {sp['market_line']:+.1f} ({sp['market_odds']:+d})")
            lines.append(f"      Model Projects: {game['home_team']} {sp['model_home_score']:.1f} - {game['away_team']} {sp['model_away_score']:.1f}")
            lines.append(f"      Model Margin: {game['home_team']} {sp['model_margin']:+.1f}")
            lines.append(f"      Edge: {sp['edge']:+.1f} pts")
            if sp.get("pick"):
                conf_pct = sp['confidence'] * 100
                win_prob = sp.get('win_probability', 0.5) * 100
                lines.append(f"      ✅ PICK: {sp['pick']} {sp['pick_line']:+.1f} ({sp['pick_odds']:+d})")
                lines.append(f"      Win Probability: {win_prob:.1f}% | Confidence: {conf_pct:.1f}%")
            lines.append(f"      📝 {sp['rationale']}")
            lines.append("")
        
        # Total
        if fg.get("total"):
            tot = fg["total"]
            lines.append(f"   TOTAL:")
            lines.append(f"      Market Line: O/U {tot['market_line']:.1f} ({tot['market_odds']:+d})")
            lines.append(f"      Model Projects: {tot['model_total']:.1f} total points")
            lines.append(f"      Edge: {tot['edge']:+.1f} pts")
            if tot.get("pick"):
                conf_pct = tot['confidence'] * 100
                win_prob = tot.get('win_probability', 0.5) * 100
                lines.append(f"      ✅ PICK: {tot['pick']} {tot['pick_line']:.1f} ({tot['pick_odds']:+d})")
                lines.append(f"      Win Probability: {win_prob:.1f}% | Confidence: {conf_pct:.1f}%")
            lines.append(f"      📝 {tot['rationale']}")
            lines.append("")
        
        # Moneyline
        if fg.get("moneyline"):
            ml = fg["moneyline"]
            if ml.get("market_home_odds") and ml.get("market_away_odds"):
                lines.append(f"   MONEYLINE:")
                lines.append(f"      Market Odds: {game['home_team']} ({ml['market_home_odds']:+d}) | {game['away_team']} ({ml['market_away_odds']:+d})")
                lines.append(f"      Market Implied: {game['home_team']} {ml['market_home_prob']*100:.1f}% | {game['away_team']} {ml['market_away_prob']*100:.1f}%")
                lines.append(f"      Model Odds:  {game['home_team']} ({ml['model_home_odds']:+d}) | {game['away_team']} ({ml['model_away_odds']:+d})")
                lines.append(f"      Model Implied: {game['home_team']} {ml['model_home_prob']*100:.1f}% | {game['away_team']} {ml['model_away_prob']*100:.1f}%")
                if ml.get("edge_home") is not None:
                    lines.append(f"      Probability Edge: Home {ml['edge_home']*100:+.1f}% | Away {ml['edge_away']*100:+.1f}%")
                if ml.get("pick"):
                    conf_pct = ml['confidence'] * 100
                    pick_odds = ml['market_home_odds'] if ml['pick'] == game['home_team'] else ml['market_away_odds']
                    lines.append(f"      ✅ PICK: {ml['pick']} ({pick_odds:+d})")
                    lines.append(f"      Confidence: {conf_pct:.1f}%")
                lines.append(f"      📝 {ml['rationale']}")
                lines.append("")
        
        # === FIRST HALF ANALYSIS ===
        fh = comp_edge.get("first_half", {})
        lines.append("📊 FIRST HALF ANALYSIS:")
        lines.append("")
        
        # Spread
        if fh.get("spread"):
            sp = fh["spread"]
            lines.append(f"   1H SPREAD:")
            if sp.get('market_line') is not None and sp.get('market_odds') is not None:
                lines.append(f"      Market Line: {game['home_team']} {sp['market_line']:+.1f} ({sp['market_odds']:+d})")
            else:
                lines.append(f"      Market Line: Not available")
            lines.append(f"      Model Projects: {game['home_team']} {sp['model_home_score']:.1f} - {game['away_team']} {sp['model_away_score']:.1f}")
            lines.append(f"      Model Margin: {game['home_team']} {sp['model_margin']:+.1f}")
            if sp.get('edge') is not None:
                lines.append(f"      Edge: {sp['edge']:+.1f} pts")
            else:
                lines.append(f"      Edge: Analysis skipped (market data unavailable)")
            if sp.get("pick"):
                conf_pct = sp['confidence'] * 100
                win_prob = sp.get('win_probability', 0.5) * 100
                pick_line_str = f" {sp['pick_line']:+.1f}" if sp.get('pick_line') is not None else ""
                pick_odds_str = f" ({sp['pick_odds']:+d})" if sp.get('pick_odds') is not None else ""
                lines.append(f"      ✅ PICK: {sp['pick']}{pick_line_str}{pick_odds_str}")
                lines.append(f"      Win Probability: {win_prob:.1f}% | Confidence: {conf_pct:.1f}%")
            lines.append(f"      📝 {sp['rationale']}")
            lines.append("")
        
        # Total
        if fh.get("total"):
            tot = fh["total"]
            lines.append(f"   1H TOTAL:")
            if tot.get('market_line') is not None and tot.get('market_odds') is not None:
                lines.append(f"      Market Line: O/U {tot['market_line']:.1f} ({tot['market_odds']:+d})")
            else:
                lines.append(f"      Market Line: Not available")
            lines.append(f"      Model Projects: {tot['model_total']:.1f} total points at half")
            if tot.get('edge') is not None:
                lines.append(f"      Edge: {tot['edge']:+.1f} pts")
            else:
                lines.append(f"      Edge: Analysis skipped (market data unavailable)")
            if tot.get("pick"):
                conf_pct = tot['confidence'] * 100
                win_prob = tot.get('win_probability', 0.5) * 100
                pick_line_str = f" {tot['pick_line']:.1f}" if tot.get('pick_line') is not None else ""
                pick_odds_str = f" ({tot['pick_odds']:+d})" if tot.get('pick_odds') is not None else ""
                lines.append(f"      ✅ PICK: {tot['pick']}{pick_line_str}{pick_odds_str}")
                lines.append(f"      Win Probability: {win_prob:.1f}% | Confidence: {conf_pct:.1f}%")
            lines.append(f"      📝 {tot['rationale']}")
            lines.append("")
        
        # Moneyline
        if fh.get("moneyline"):
            ml = fh["moneyline"]
            if ml.get("model_home_odds") and ml.get("model_away_odds"):
                lines.append(f"   1H MONEYLINE:")
                lines.append(f"      Model Odds: {game['home_team']} ({ml['model_home_odds']:+d}) | {game['away_team']} ({ml['model_away_odds']:+d})")
                lines.append(f"      Model Implied: {game['home_team']} {ml['model_home_prob']*100:.1f}% | {game['away_team']} {ml['model_away_prob']*100:.1f}%")
                if ml.get("pick"):
                    lines.append(f"      💡 LEAN: {ml['pick']} to lead at halftime")
                lines.append(f"      📝 {ml['rationale']}")
                lines.append("")
    
    lines.append("=" * 100)
    lines.append(f"Generated: {get_cst_now().strftime('%Y-%m-%d %I:%M %p CST')}")
    lines.append("")
    lines.append("Legend:")
    lines.append("  [HIGH-CONF] = Model probability >60% or <40% (recommended for 7.3% ROI)")
    lines.append("  Edge = Model projection vs market line (positive = betting opportunity)")
    lines.append("  Expected Value = Expected profit per $1 bet")
    lines.append("  FG = Full Game | 1H = First Half")
    lines.append("")
    lines.append("Betting Strategy:")
    lines.append("  - High-Confidence Plays: Focus here for best ROI (+7.3% vs +4.4% overall)")
    lines.append("  - Minimum Edge: 2% for spreads/totals, 3% for moneylines")
    
    return "\n".join(lines)


def generate_text_report(analysis: List[Dict], target_date: datetime.date) -> str:
    """Generate a text-based report (legacy version)."""
    # Check if we have comprehensive edge analysis
    if analysis and "comprehensive_edge" in analysis[0]:
        return generate_comprehensive_text_report(analysis, target_date)
    
    # Fall back to old format
    lines = []
    lines.append("=" * 80)
    lines.append(f"🏀 NBA SLATE ANALYSIS - {target_date.strftime('%A, %B %d, %Y')}")
    lines.append("=" * 80)
    lines.append("")
    
    if not analysis:
        lines.append("No games scheduled for this date.")
        return "\n".join(lines)
    
    for i, game in enumerate(analysis, 1):
        lines.append(f"{'─' * 80}")
        lines.append(f"GAME {i}: {game['away_team']} @ {game['home_team']}")
        lines.append(f"Time: {game['time_cst']}")
        lines.append("")
        
        odds = game.get("odds", {})
        features = game.get("features", {})
        edge = game.get("edge", {})
        
        # Odds section
        lines.append("📊 MARKET ODDS:")
        if odds.get("home_ml"):
            lines.append(f"   Moneyline: {game['home_team']} ({odds['home_ml']:+d}) vs {game['away_team']} ({odds['away_ml']:+d})")
        if odds.get("home_spread") is not None:
            lines.append(f"   Spread: {game['home_team']} {odds['home_spread']:+.1f}")
        if odds.get("total"):
            lines.append(f"   Total: {odds['total']:.1f}")
        lines.append("")
        
        # Model predictions
        lines.append("🎯 MODEL PREDICTIONS:")
        if features:
            lines.append(f"   Predicted Margin: {features.get('predicted_margin', 0):+.1f} (home perspective)")
            lines.append(f"   Predicted Total: {features.get('predicted_total', 220):.1f}")
            if features.get("home_ppg"):
                lines.append(f"   Home PPG: {features['home_ppg']:.1f} | Away PPG: {features['away_ppg']:.1f}")
            if features.get("home_elo"):
                lines.append(f"   ELO: {game['home_team']} {features['home_elo']:.0f} vs {game['away_team']} {features['away_elo']:.0f}")
            # Key drivers for transparency/debugging
            if features.get("expected_pace_factor") is not None:
                lines.append(f"   Pace Factor: {features['expected_pace_factor']:.2f}")
            if features.get("rest_margin_adj") is not None:
                lines.append(f"   Rest Adj (margin): {features['rest_margin_adj']:+.2f}")
            if features.get("form_margin_adj") is not None:
                lines.append(f"   Form Adj (margin): {features['form_margin_adj']:+.2f}")
        lines.append("")
        
        # Edge analysis
        if edge.get("spread_edge") is not None:
            lines.append("💰 EDGE ANALYSIS:")
            lines.append(f"   Spread Edge: {edge['spread_edge']:+.1f} pts")
            if edge.get("total_edge") is not None:
                lines.append(f"   Total Edge: {edge['total_edge']:+.1f} pts")
            if edge.get("recommendation"):
                conf_stars = "⭐" * int(edge.get("confidence", 0) * 5)
                lines.append(f"   Recommendation: {edge['recommendation']} {conf_stars}")
        lines.append("")
    
    lines.append("=" * 80)
    lines.append(f"Generated: {get_cst_now().strftime('%Y-%m-%d %I:%M %p CST')}")
    
    return "\n".join(lines)


def create_visualization(analysis: List[Dict], target_date: datetime.date, output_path: str):
    """Create a clean visualization figure for the weekly lineup."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.gridspec import GridSpec
        import numpy as np
    except ImportError:
        print("⚠️  matplotlib not installed. Skipping visualization.")
        return
    
    # Set up the figure with a dark theme
    plt.style.use('dark_background')
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    
    n_games = len(analysis)
    if n_games == 0:
        print("No games to visualize.")
        return
    
    # Figure size based on number of games
    fig_height = max(10, 2.8 * n_games + 4)
    fig = plt.figure(figsize=(16, fig_height), facecolor='#0a0e14')
    
    # Custom colors - refined palette
    GOLD = '#f0b429'
    CYAN = '#22d3ee'
    GREEN = '#10b981'
    RED = '#ef4444'
    PURPLE = '#8b5cf6'
    ORANGE = '#f97316'
    GRAY = '#94a3b8'
    LIGHT_GRAY = '#cbd5e1'
    WHITE = '#f8fafc'
    BG_DARK = '#0a0e14'
    BG_CARD = '#1e293b'
    BORDER = '#334155'
    
    # Title area
    fig.text(
        0.5, 0.97,
        "NBA SLATE ANALYSIS",
        ha='center',
        fontsize=32,
        fontweight='bold',
        color=GOLD,
        family='DejaVu Sans'
    )
    
    # Subtitle with date
    fig.text(
        0.5, 0.935,
        target_date.strftime('%A, %B %d, %Y'),
        ha='center',
        fontsize=16,
        color=GRAY,
        style='italic'
    )
    
    # Create grid for games
    gs = GridSpec(n_games + 1, 3, figure=fig, height_ratios=[0.25] + [1] * n_games,
                  hspace=0.35, wspace=0.25, left=0.04, right=0.96, top=0.90, bottom=0.06)
    
    # Header row
    header_ax = fig.add_subplot(gs[0, :])
    header_ax.set_facecolor(BG_DARK)
    header_ax.axis('off')
    
    headers = ['MATCHUP', 'MARKET & MODEL', 'EDGE ANALYSIS']
    header_colors = [PURPLE, CYAN, GREEN]
    for i, (header, hcolor) in enumerate(zip(headers, header_colors)):
        header_ax.text(
            0.17 + i * 0.33, 0.5, header,
            ha='center', va='center',
            fontsize=13, fontweight='bold',
            color=hcolor,
            transform=header_ax.transAxes
        )
        # Underline
        header_ax.plot([0.05 + i * 0.33, 0.29 + i * 0.33], [0.1, 0.1],
                      color=hcolor, linewidth=2, transform=header_ax.transAxes, alpha=0.5)
    
    # Game rows
    for idx, game in enumerate(analysis):
        row = idx + 1
        
        # Matchup column
        ax1 = fig.add_subplot(gs[row, 0])
        ax1.set_facecolor(BG_CARD)
        ax1.axis('off')
        
        # Draw rounded rectangle background
        rect = mpatches.FancyBboxPatch(
            (0.02, 0.02), 0.96, 0.96,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=BG_CARD,
            edgecolor=PURPLE,
            linewidth=2.5,
            transform=ax1.transAxes
        )
        ax1.add_patch(rect)
        
        # Game number badge
        ax1.text(0.08, 0.88, f"#{idx+1}", ha='left', va='top',
                fontsize=10, fontweight='bold', color=PURPLE, 
                transform=ax1.transAxes, alpha=0.7)
        
        # Team names with better spacing
        ax1.text(0.5, 0.72, game['away_team'], ha='center', va='center',
                fontsize=12, fontweight='bold', color=LIGHT_GRAY, transform=ax1.transAxes)
        ax1.text(0.5, 0.50, '@', ha='center', va='center',
                fontsize=14, color=GRAY, transform=ax1.transAxes, alpha=0.6)
        ax1.text(0.5, 0.28, game['home_team'], ha='center', va='center',
                fontsize=12, fontweight='bold', color=GOLD, transform=ax1.transAxes)
        
        # Game time with icon
        ax1.text(0.5, 0.08, game['time_cst'], ha='center', va='bottom',
                fontsize=10, color=CYAN, fontweight='bold', transform=ax1.transAxes)
        
        # Odds & Predictions column
        ax2 = fig.add_subplot(gs[row, 1])
        ax2.set_facecolor(BG_CARD)
        ax2.axis('off')
        
        rect2 = mpatches.FancyBboxPatch(
            (0.02, 0.02), 0.96, 0.96,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=BG_CARD,
            edgecolor=BORDER,
            linewidth=1.5,
            transform=ax2.transAxes
        )
        ax2.add_patch(rect2)
        
        odds = game.get('odds', {})
        features = game.get('features', {})
        
        # Left side: Market odds
        ax2.text(0.08, 0.88, "MARKET", ha='left', va='top',
                fontsize=9, fontweight='bold', color=GRAY, transform=ax2.transAxes)
        
        y_left = 0.72
        if odds.get('home_spread') is not None:
            ax2.text(0.08, y_left, f"Spread: {odds['home_spread']:+.1f}", ha='left', va='center',
                    fontsize=10, color=WHITE, transform=ax2.transAxes)
            y_left -= 0.16
        
        if odds.get('total'):
            ax2.text(0.08, y_left, f"Total: {odds['total']:.1f}", ha='left', va='center',
                    fontsize=10, color=WHITE, transform=ax2.transAxes)
            y_left -= 0.16
        
        if odds.get('home_ml'):
            ml_str = f"ML: {odds['home_ml']:+d}"
            ax2.text(0.08, y_left, ml_str, ha='left', va='center',
                    fontsize=10, color=WHITE, transform=ax2.transAxes)
        
        # Vertical divider
        ax2.plot([0.5, 0.5], [0.15, 0.85], color=BORDER, linewidth=1, 
                transform=ax2.transAxes, alpha=0.5)
        
        # Right side: Model predictions
        ax2.text(0.55, 0.88, "MODEL", ha='left', va='top',
                fontsize=9, fontweight='bold', color=CYAN, transform=ax2.transAxes)
        
        y_right = 0.72
        if features.get('predicted_margin') is not None:
            pred_margin = features['predicted_margin']
            margin_color = GREEN if pred_margin > 0 else RED if pred_margin < 0 else WHITE
            ax2.text(0.55, y_right, f"Margin: {pred_margin:+.1f}", ha='left', va='center',
                    fontsize=10, color=margin_color, fontweight='bold', transform=ax2.transAxes)
            y_right -= 0.16
        
        if features.get('predicted_total'):
            ax2.text(0.55, y_right, f"Total: {features['predicted_total']:.1f}", ha='left', va='center',
                    fontsize=10, color=WHITE, transform=ax2.transAxes)
            y_right -= 0.16
        
        if features.get('home_elo') and features.get('away_elo'):
            elo_diff = features['home_elo'] - features['away_elo']
            elo_color = GREEN if elo_diff > 0 else RED if elo_diff < 0 else WHITE
            ax2.text(0.55, y_right, f"ELO Diff: {elo_diff:+.0f}", ha='left', va='center',
                    fontsize=10, color=elo_color, transform=ax2.transAxes)
        
        # Edge Analysis column
        ax3 = fig.add_subplot(gs[row, 2])
        ax3.set_facecolor(BG_CARD)
        ax3.axis('off')
        
        edge = game.get('edge', {})
        spread_edge = edge.get('spread_edge')
        recommendation = edge.get('recommendation')
        confidence = edge.get('confidence', 0)
        
        if recommendation:
            # Draw recommendation box with highlighted border
            rec_color = GREEN if recommendation == 'HOME' else ORANGE
            
            rect3 = mpatches.FancyBboxPatch(
                (0.02, 0.02), 0.96, 0.96,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor=BG_CARD,
                edgecolor=rec_color,
                linewidth=3,
                transform=ax3.transAxes
            )
            ax3.add_patch(rect3)
            
            # Recommendation label
            rec_team = game['home_team'] if recommendation == 'HOME' else game['away_team']
            ax3.text(0.5, 0.82, "PICK", ha='center', va='center',
                    fontsize=10, fontweight='bold', color=rec_color, 
                    transform=ax3.transAxes, alpha=0.8)
            ax3.text(0.5, 0.62, rec_team, ha='center', va='center',
                    fontsize=11, fontweight='bold', color=WHITE, transform=ax3.transAxes)
            
            # Edge value
            if spread_edge is not None:
                edge_color = GREEN if abs(spread_edge) >= 3 else GOLD if abs(spread_edge) >= 2 else WHITE
                ax3.text(0.5, 0.42, f"Edge: {abs(spread_edge):.1f} pts", ha='center', va='center',
                        fontsize=11, fontweight='bold', color=edge_color, transform=ax3.transAxes)
            
            # Confidence indicator (bars instead of stars)
            conf_bars = int(confidence * 5)
            bar_width = 0.08
            bar_spacing = 0.10
            start_x = 0.5 - (2.5 * bar_spacing)
            
            for i in range(5):
                bar_color = rec_color if i < conf_bars else BORDER
                bar_alpha = 1.0 if i < conf_bars else 0.3
                rect_bar = mpatches.FancyBboxPatch(
                    (start_x + i * bar_spacing, 0.12), bar_width, 0.12,
                    boxstyle="round,pad=0.01,rounding_size=0.02",
                    facecolor=bar_color,
                    edgecolor='none',
                    alpha=bar_alpha,
                    transform=ax3.transAxes
                )
                ax3.add_patch(rect_bar)
            
            # Confidence label
            ax3.text(0.5, 0.06, f"{confidence*100:.0f}% CONF", ha='center', va='bottom',
                    fontsize=8, color=GRAY, transform=ax3.transAxes)
        else:
            rect3 = mpatches.FancyBboxPatch(
                (0.02, 0.02), 0.96, 0.96,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor=BG_CARD,
                edgecolor=BORDER,
                linewidth=1.5,
                transform=ax3.transAxes
            )
            ax3.add_patch(rect3)
            
            ax3.text(0.5, 0.55, "NO EDGE", ha='center', va='center',
                    fontsize=12, fontweight='bold', color=GRAY, transform=ax3.transAxes)
            
            if spread_edge is not None:
                ax3.text(0.5, 0.35, f"({spread_edge:+.1f} pts)", ha='center', va='center',
                        fontsize=10, color=GRAY, transform=ax3.transAxes, alpha=0.7)
    
    # Footer
    fig.text(
        0.5, 0.015,
        f"Generated: {get_cst_now().strftime('%Y-%m-%d %I:%M %p CST')} | Source: The Odds API + API-Basketball",
        ha='center',
        fontsize=10,
        color=GRAY
    )
    
    # Save figure
    plt.savefig(output_path, dpi=150, facecolor=BG_DARK, edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Visualization saved to: {output_path}")


async def analyze_slate(
    date_str: str = None,
    output_path: str = None,
    use_api: bool = True,
    home_filter: str = None,
    away_filter: str = None,
    upcoming_only: bool = False,
):
    """Main analysis function."""
    target_date = get_target_date(date_str)
    now_cst = get_cst_now()
    
    # Initialize tracking systems
    clv_tracker = CLVTracker()
    prediction_logger = PredictionLogger()
    
    # Get dynamic edge thresholds for this date
    edge_thresholds = get_edge_thresholds_for_game(
        game_date=target_date,
        bet_types=["spread", "total", "moneyline", "1h_spread", "1h_total"]
    )
    
    print("=" * 80)
    print("🏀 NBA SLATE ANALYSIS")
    print("=" * 80)
    print(f"Current time: {now_cst.strftime('%A, %B %d, %Y at %I:%M %p CST')}")
    print(f"Target date: {target_date.strftime('%A, %B %d, %Y')}")
    print(f"\n📊 Dynamic Edge Thresholds:")
    for bet_type, threshold in edge_thresholds.items():
        print(f"   {bet_type}: {threshold:.2f}")
    
    # Fetch games
    games = await fetch_todays_games(target_date)
    
    if not games:
        print("\n❌ No games found for this date.")
        return []
    
    # Filter for upcoming games if requested
    if upcoming_only:
        print("\n⏱️  Filtering for upcoming games only...")
        upcoming_games = []
        for game in games:
            commence_time = game.get("commence_time")
            if commence_time:
                try:
                    game_dt = parse_utc_time(commence_time)
                    game_cst = to_cst(game_dt)
                    if game_cst > now_cst:
                        upcoming_games.append(game)
                except Exception:
                    continue
        
        print(f"   Removed {len(games) - len(upcoming_games)} games that have already started.")
        games = upcoming_games
        
        if not games:
            print("\n❌ No upcoming games found for this date.")
            return []
    
    # Fetch betting splits if available
    betting_splits_dict = {}
    try:
        from src.ingestion.betting_splits import fetch_public_betting_splits
        print("\n📊 Fetching betting splits...")
        betting_splits_dict = await fetch_public_betting_splits(games, source="auto")
        print(f"   [OK] Loaded betting splits for {len(betting_splits_dict)} games")
    except Exception as e:
        print(f"   [WARN] Could not fetch betting splits: {e}")
        print(f"   [INFO] Continuing without betting splits data")
    
    # Sort by commence time
    games.sort(key=lambda x: x.get("commence_time", ""))
    
    print(f"\n📋 Found {len(games)} games:\n")
    
    analysis = []
    
    filtered_games = games
    if home_filter and away_filter:
        hf = home_filter.lower()
        af = away_filter.lower()
        filtered_games = [
            g
            for g in games
            if g.get("home_team", "").lower() == hf and g.get("away_team", "").lower() == af
        ]
        if not filtered_games:
            print(f"\n❌ No game found for {away_filter} @ {home_filter} on {target_date}")
            return []
        print(f"\n🎯 Filtering to single matchup: {away_filter} @ {home_filter}")

    for i, game in enumerate(filtered_games, 1):
        home_team = game.get("home_team")
        away_team = game.get("away_team")
        commence_time = game.get("commence_time")
        
        if not home_team or not away_team:
            continue
        
        # Parse time
        if commence_time:
            game_dt = parse_utc_time(commence_time)
            time_cst = format_cst_time(game_dt)
            game_cst = to_cst(game_dt)
            is_live = game_cst <= now_cst
        else:
            time_cst = "TBD"
            is_live = False
        
        status = "🔴 LIVE" if is_live else "⏰"
        print(f"   [{i}] {away_team} @ {home_team} - {time_cst} CST {status}")
        
        # Extract odds
        odds = extract_consensus_odds(game)
        
        # Get betting splits for this game if available
        game_key = f"{away_team}@{home_team}"
        betting_splits = betting_splits_dict.get(game_key)
        
        # Build features
        features = None
        if use_api and settings.api_basketball_key:
            print(f"      Building features...")
            features = await build_team_features(home_team, away_team)
            # Add betting splits to features if available
            if betting_splits and features:
                from src.ingestion.betting_splits import splits_to_features
                splits_features = splits_to_features(betting_splits)
                features.update(splits_features)
        
        if not features:
            # Fall back to simple features derived from odds when API is unavailable
            features = create_simple_features(game, odds)
            if use_api and settings.api_basketball_key:
                # Only warn if we expected to get rich features but couldn't
                print(f"      [WARN] Using simple features (API features unavailable)")
        
        # Calculate comprehensive edge (includes first half and full game)
        fh_features = {}  # First half features derived from full game
        comprehensive_edge = calculate_comprehensive_edge(
            features, fh_features, odds, game, betting_splits, edge_thresholds
        )
        
        # Legacy edge for backward compatibility
        edge = calculate_edge(features, odds)
        
        # Log prediction for retrospective analysis
        prediction_data = {
            "full_game": {
                "spread": {
                    "model_margin": comprehensive_edge.get("full_game", {}).get("spread", {}).get("model_margin"),
                    "market_line": comprehensive_edge.get("full_game", {}).get("spread", {}).get("market_line"),
                    "edge": comprehensive_edge.get("full_game", {}).get("spread", {}).get("edge"),
                },
                "total": {
                    "model_total": comprehensive_edge.get("full_game", {}).get("total", {}).get("model_total"),
                    "market_line": comprehensive_edge.get("full_game", {}).get("total", {}).get("market_line"),
                    "edge": comprehensive_edge.get("full_game", {}).get("total", {}).get("edge"),
                },
                "moneyline": comprehensive_edge.get("full_game", {}).get("moneyline", {}),
            },
            "first_half": {
                "spread": {
                    "model_margin": comprehensive_edge.get("first_half", {}).get("spread", {}).get("model_margin"),
                    "market_line": comprehensive_edge.get("first_half", {}).get("spread", {}).get("market_line"),
                    "edge": comprehensive_edge.get("first_half", {}).get("spread", {}).get("edge"),
                },
                "total": {
                    "model_total": comprehensive_edge.get("first_half", {}).get("total", {}).get("model_total"),
                    "market_line": comprehensive_edge.get("first_half", {}).get("total", {}).get("market_line"),
                    "edge": comprehensive_edge.get("first_half", {}).get("total", {}).get("edge"),
                },
            },
        }
        
        prediction_logger.log_prediction(
            game_date=target_date,
            home_team=home_team,
            away_team=away_team,
            predictions=prediction_data,
            features=features,
            odds=odds,
            metadata={
                "edge_thresholds": edge_thresholds,
                "model_version": "v4.0",
                "is_live": is_live,
            }
        )
        
        # Record predictions for CLV tracking
        fg_spread = comprehensive_edge.get("full_game", {}).get("spread", {})
        if fg_spread.get("model_margin") is not None:
            clv_tracker.record_prediction(
                game_date=target_date,
                home_team=home_team,
                away_team=away_team,
                bet_type="spread",
                model_line=fg_spread.get("model_margin"),
                opening_line=fg_spread.get("market_line"),
                metadata={"edge": fg_spread.get("edge")}
            )
        
        fg_total = comprehensive_edge.get("full_game", {}).get("total", {})
        if fg_total.get("model_total") is not None:
            clv_tracker.record_prediction(
                game_date=target_date,
                home_team=home_team,
                away_team=away_team,
                bet_type="total",
                model_line=fg_total.get("model_total"),
                opening_line=fg_total.get("market_line"),
                metadata={"edge": fg_total.get("edge")}
            )
        
        analysis.append({
            "home_team": home_team,
            "away_team": away_team,
            "time_cst": f"{time_cst} CST",
            "commence_time": commence_time,
            "is_live": is_live,
            "odds": odds,
            "features": features,
            "edge": edge,
            "comprehensive_edge": comprehensive_edge,
            "betting_splits": betting_splits,  # Include for rationale generation
        })
    
    # Generate text report
    print("\n")
    report = generate_text_report(analysis, target_date)
    print(report)
    
    # Save report
    report_dir = PROJECT_ROOT / "data" / "processed"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = report_dir / f"slate_analysis_{target_date.strftime('%Y%m%d')}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n📄 Report saved to: {report_path}")
    
    # Create visualization
    if output_path is None:
        output_path = str(report_dir / f"weekly_lineup_{target_date.strftime('%Y%m%d')}.png")
    
    create_visualization(analysis, target_date, output_path)
    
    # Save JSON analysis
    json_path = report_dir / f"slate_analysis_{target_date.strftime('%Y%m%d')}.json"
    with open(json_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"📊 JSON data saved to: {json_path}")
    
    # Generate betting card with rationale
    print("\n" + "=" * 80)
    print("📋 GENERATING BETTING CARD WITH RATIONALE")
    print("=" * 80)
    
    try:
        from src.modeling.betting_card import generate_betting_card
        from src.modeling.betting_card_export import export_betting_card_to_excel, export_betting_card_to_html, update_pick_tracker
        
        # Generate betting card summary and picks
        summary_text, picks = generate_betting_card(analysis, target_date)
        
        # Print summary
        print("\n" + summary_text)
        
        # Export to Excel
        excel_path = export_betting_card_to_excel(picks, target_date, summary_text)

        # Export to HTML
        html_path = export_betting_card_to_html(picks, target_date, summary_text)

        # Update pick tracker
        tracker_path = update_pick_tracker(picks, target_date)
        
        print(f"\n✅ Betting card generation complete!")
        print(f"   Excel file: {excel_path}")
        print(f"   HTML file: {html_path}")
        print(f"   Tracker file: {tracker_path}")
        
    except Exception as e:
        print(f"\n⚠️  Error generating betting card: {e}")
        import traceback
        traceback.print_exc()
    
    # Print CLV statistics summary
    print("\n" + "=" * 80)
    print("📊 CLOSING LINE VALUE (CLV) STATISTICS")
    print("=" * 80)
    try:
        clv_stats = clv_tracker.get_clv_stats()
        if "error" not in clv_stats:
            print(f"   Total Predictions: {clv_stats['n_predictions']}")
            print(f"   Completed (with closing lines): {clv_stats['n_completed']}")
            print(f"   Completion Rate: {clv_stats['completion_rate']:.1%}")
            if clv_stats['n_completed'] > 0:
                print(f"   Average CLV: {clv_stats['avg_clv']:+.2f} pts")
                print(f"   Beat Closing Line Rate: {clv_stats['beat_closing_rate']:.1%}")
                print(f"   CLV Range: [{clv_stats['min_clv']:+.2f}, {clv_stats['max_clv']:+.2f}]")
                print(f"\n   💡 Tip: Update closing lines after games to track CLV performance")
        else:
            print(f"   {clv_stats['error']}")
    except Exception as e:
        print(f"   ⚠️  Could not calculate CLV stats: {e}")
    
    return analysis


def main():
    # Fix Windows console encoding for emoji support
    import sys
    import io
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    
    parser = argparse.ArgumentParser(description="Analyze today's NBA slate")
    parser.add_argument("--date", help="Date for analysis (YYYY-MM-DD, 'today', or 'tomorrow')")
    parser.add_argument("--output", "-o", help="Output path for visualization")
    parser.add_argument("--no-api", action="store_true", help="Skip API-Basketball feature building")
    parser.add_argument("--home", help="Home team for single-game analysis (requires --away)")
    parser.add_argument("--away", help="Away team for single-game analysis (requires --home)")
    parser.add_argument("--upcoming", action="store_true", help="Only analyze games that haven't started yet")
    args = parser.parse_args()
    
    asyncio.run(
        analyze_slate(
            args.date,
            args.output,
            use_api=not args.no_api,
            home_filter=args.home,
            away_filter=args.away,
            upcoming_only=args.upcoming,
        )
    )


if __name__ == "__main__":
    main()

