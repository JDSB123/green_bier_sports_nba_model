"""
Generate predictions for ALL markets using unified prediction engine.

Production-ready predictor with smart filtering for all markets:
- Full Game: Spreads, Totals, Moneyline
- First Half: Spreads, Totals, Moneyline
- First Quarter: Spreads, Totals, Moneyline
"""
import asyncio
import argparse
import random
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "processed" / "models"

from src.config import settings
from src.ingestion import the_odds
from src.ingestion.betting_splits import fetch_public_betting_splits
from scripts.build_rich_features import RichFeatureBuilder
from src.prediction import UnifiedPredictionEngine

# Central Standard Time
CST = ZoneInfo("America/Chicago")


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
    
    Returns bullet-point formatted string.
    """
    rationale_bullets = []
    is_partial_market = ("1H" in play_type) or ("Q1" in play_type)
    now_cst = datetime.now(CST)
    
    # Calculate expected value
    if odds > 0:
        profit = odds / 100
    else:
        profit = 100 / abs(odds or 110)
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
        
        movement_threshold = 1.0 if "SPREAD" in play_type else 0.5
        if movement_magnitude >= movement_threshold:
            if "SPREAD" in play_type:
                benefited_team = home_team if line_movement < 0 else away_team
                market_context.append(
                    f"[LINE] Line moved from {opening_line:+.1f} to {market_line:+.1f} "
                    f"({movement_direction} {movement_magnitude:.1f} pts), suggesting sharp action on {benefited_team}."
                )
            elif "TOTAL" in play_type:
                market_context.append(
                    f"[LINE] Total moved from {opening_line:.1f} to {market_line:.1f} "
                    f"({movement_direction} {movement_magnitude:.1f} pts)."
                )
    
    # ============================================================
    # 4. MARKET SENTIMENT & SHARP ACTION (Very High Priority)
    # ============================================================
    sharp_action = []
    
    if betting_splits:
        try:
            # We use features built from splits in BuildRichFeatures
            if "SPREAD" in play_type:
                is_home_pick = "home" in pick.lower()
                
                # Use features directly since betting_splits might be a dict or object
                pick_tickets = features.get("spread_public_home_pct", 50) if is_home_pick else features.get("spread_public_away_pct", 50)
                pick_money = features.get("spread_money_home_pct", 50) if is_home_pick else features.get("spread_money_away_pct", 50)
                opp_tickets = 100 - pick_tickets
                
                rlm = features.get("is_rlm_spread", 0) > 0
                sharp_side = features.get("sharp_side_spread", 0)
                
                if rlm:
                    is_sharp_on_pick = (is_home_pick and sharp_side > 0) or (not is_home_pick and sharp_side < 0)
                    if is_sharp_on_pick:
                        sharp_action.append(
                            f"[SHARP] Reverse line movement detected: despite {opp_tickets:.0f}% of bets on opponent, "
                            f"line moved in favor of {pick} — classic sharp money signal."
                        )
                
                ticket_money_diff = features.get("spread_ticket_money_diff", 0)
                if abs(ticket_money_diff) >= 10:
                    sharp_action.append(
                        f"[SHARP] Sharp money indicator: money/ticket divergence of {abs(ticket_money_diff):.0f}% detected."
                    )
            elif "TOTAL" in play_type:
                is_over = "OVER" in pick.upper()
                pick_tickets = features.get("over_public_pct", 50) if is_over else features.get("under_public_pct", 50)
                rlm = features.get("is_rlm_total", 0) > 0
                if rlm:
                    sharp_action.append(f"[SHARP] Reverse line movement detected on the {pick}.")
        except Exception:
            pass
    
    # ============================================================
    # 2. TEAM FUNDAMENTALS (High Priority)
    # ============================================================
    team_fundamentals = []
    
    if not is_partial_market:
        home_ppg = features.get("home_ppg", 0)
        away_ppg = features.get("away_ppg", 0)
        
        if "SPREAD" in play_type:
            is_home_pick = "home" in pick.lower()
            pick_team_ppg = home_ppg if is_home_pick else away_ppg
            team_fundamentals.append(f"[STATS] {pick.title()} offensive efficiency: {pick_team_ppg:.1f} season PPG.")
        
        elif "TOTAL" in play_type:
            combined_ppg = home_ppg + away_ppg
            team_fundamentals.append(f"[STATS] Combined offensive output: {combined_ppg:.1f} PPG season average.")
        
        home_elo = features.get("home_elo", 1500)
        away_elo = features.get("away_elo", 1500)
        elo_diff = abs(home_elo - away_elo)
        if elo_diff >= 50:
            stronger = home_team if home_elo > away_elo else away_team
            team_fundamentals.append(f"[ELO] {stronger} holds significant ELO advantage ({elo_diff:.0f} pts).")
    
    # ============================================================
    # 3. SITUATIONAL FACTORS (Medium Priority)
    # ============================================================
    situational = []
    
    if not is_partial_market:
        rest_adj = features.get("rest_margin_adj", 0)
        if abs(rest_adj) >= 1.5:
            benefit_team = home_team if rest_adj > 0 else away_team
            situational.append(f"[REST] {benefit_team} has a situational rest advantage (+{abs(rest_adj):.1f} pts).")
        
        travel_fatigue = features.get("away_travel_fatigue", 0)
        if travel_fatigue >= 2.0:
            situational.append(f"[FATIGUE] Away team fatigue factor: high travel impact ({travel_fatigue:.1f} pts penalty).")
    
    # ============================================================
    # 5. MODEL CONFIDENCE (High Priority)
    # ============================================================
    model_confidence = [
        f"[MODEL] Model assigns {model_prob:.1%} probability to {pick} with {edge:+.1f} pt edge."
    ]
    
    if abs(ev_pct) >= 5:
        model_confidence.append(f"[MODEL] Expected value: {ev_pct:+.1f}% based on market odds.")
    
    # ============================================================
    # ASSEMBLE RATIONALE
    # ============================================================
    rationale_bullets.extend(model_confidence[:1])
    if market_context: rationale_bullets.extend(market_context[:1])
    if sharp_action: rationale_bullets.extend(sharp_action[:1])
    
    # Fill to 3 bullets
    backups = team_fundamentals + situational
    for b in backups:
        if len(rationale_bullets) >= 3: break
        rationale_bullets.append(b)
        
    return " | ".join(rationale_bullets[:3])


def get_cst_now() -> datetime:
    """Get current time in CST."""
    return datetime.now(CST)


def parse_utc_time(iso_string: str) -> datetime:
    """Parse ISO UTC time string to datetime."""
    if iso_string.endswith("Z"):
        iso_string = iso_string[:-1] + "+00:00"
    return datetime.fromisoformat(iso_string).replace(tzinfo=timezone.utc)


def to_cst(dt: datetime) -> datetime:
    """Convert datetime to CST."""
    return dt.astimezone(CST)


def format_cst_time(dt: datetime) -> str:
    """Format datetime as CST string."""
    cst_dt = to_cst(dt)
    return cst_dt.strftime("%a %b %d, %I:%M %p CST")


def get_target_date(date_str: str = None) -> datetime.date:
    """Get target date for predictions."""
    now_cst = get_cst_now()

    if date_str is None or date_str.lower() == "tomorrow":
        return (now_cst + timedelta(days=1)).date()
    elif date_str.lower() == "today":
        return now_cst.date()
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


async def fetch_upcoming_games(target_date: datetime.date):
    """Fetch upcoming games from APIs, filtered to target date."""
    print("\nFetching upcoming games...")

    # Fetch main odds data (FG)
    odds_data = await the_odds.fetch_odds()
    print(f"  [OK] The Odds: {len(odds_data)} total games from API")

    odds_data = filter_games_for_date(odds_data, target_date)
    date_str = target_date.strftime("%A, %B %d, %Y")
    print(f"  [OK] Filtered to {len(odds_data)} games for {date_str}")

    # Fetch event-specific odds for 1H markets
    print(f"  [INFO] Fetching 1H markets for {len(odds_data)} games...")
    enriched_games = []
    for game in odds_data:
        event_id = game.get("id")
        if event_id:
            try:
                # Fetch 1H odds specifically
                event_odds = await the_odds.fetch_event_odds(
                    event_id,
                    markets="spreads_h1,totals_h1,h2h_h1,spreads_q1,totals_q1,h2h_q1"
                )
                
                # MERGE instead of overwrite
                # Keep existing bookmakers (FG) and add new ones (1H)
                existing_bms = {bm["key"]: bm for bm in game.get("bookmakers", [])}
                new_bms = event_odds.get("bookmakers", [])
                
                for nbm in new_bms:
                    if nbm["key"] in existing_bms:
                        # Add markets to existing bookmaker
                        existing_markets = {m["key"]: m for m in existing_bms[nbm["key"]].get("markets", [])}
                        for nm in nbm.get("markets", []):
                            existing_markets[nm["key"]] = nm
                        existing_bms[nbm["key"]]["markets"] = list(existing_markets.values())
                    else:
                        existing_bms[nbm["key"]] = nbm
                
                game["bookmakers"] = list(existing_bms.values())
            except Exception as e:
                print(f"  [WARN] Could not fetch 1H odds for event {event_id}: {e}")
        enriched_games.append(game)

    return enriched_games


def extract_lines(game: dict, home_team: str):
    """
    Extract all available betting lines from game data.

    Returns:
        Dict with FG and 1H lines for spreads, totals, and moneyline
    """
    lines = {
        # Full game
        "fg_spread": None,
        "fg_total": None,
        "fg_home_ml": None,
        "fg_away_ml": None,
        # First half
        "fh_spread": None,
        "fh_total": None,
        "fh_home_ml": None,
        "fh_away_ml": None,
        # First quarter
        "q1_spread": None,
        "q1_total": None,
        "q1_home_ml": None,
        "q1_away_ml": None,
    }

    for bm in game.get("bookmakers", []):
        for market in bm.get("markets", []):
            market_key = market.get("key")

            # Full game spreads
            if market_key == "spreads":
                for outcome in market.get("outcomes", []):
                    if outcome.get("name") == home_team:
                        lines["fg_spread"] = outcome.get("point")
                        break

            # Full game totals
            elif market_key == "totals":
                for outcome in market.get("outcomes", []):
                    lines["fg_total"] = outcome.get("point")
                    break

            # Full game moneyline
            elif market_key == "h2h":
                for outcome in market.get("outcomes", []):
                    if outcome.get("name") == home_team:
                        lines["fg_home_ml"] = outcome.get("price")
                    else:
                        lines["fg_away_ml"] = outcome.get("price")

            # First half spreads
            elif market_key == "spreads_h1":
                for outcome in market.get("outcomes", []):
                    if outcome.get("name") == home_team:
                        lines["fh_spread"] = outcome.get("point")
                        break

            # First half totals
            elif market_key == "totals_h1":
                for outcome in market.get("outcomes", []):
                    lines["fh_total"] = outcome.get("point")
                    break

            # First half moneyline
            elif market_key == "h2h_h1":
                for outcome in market.get("outcomes", []):
                    if outcome.get("name") == home_team:
                        lines["fh_home_ml"] = outcome.get("price")
                    else:
                        lines["fh_away_ml"] = outcome.get("price")

            elif market_key == "spreads_q1":
                for outcome in market.get("outcomes", []):
                    if outcome.get("name") == home_team:
                        lines["q1_spread"] = outcome.get("point")
                        break

            elif market_key == "totals_q1":
                for outcome in market.get("outcomes", []):
                    lines["q1_total"] = outcome.get("point")
                    break

            elif market_key == "h2h_q1":
                for outcome in market.get("outcomes", []):
                    if outcome.get("name") == home_team:
                        lines["q1_home_ml"] = outcome.get("price")
                    else:
                        lines["q1_away_ml"] = outcome.get("price")

    return lines


async def predict_games_async(date: str = None, use_betting_splits: bool = True):
    """Generate predictions for all markets using unified prediction engine."""
    target_date = get_target_date(date)
    now_cst = get_cst_now()

    print("=" * 80)
    print("NBA PREDICTIONS - UNIFIED ENGINE (ALL MARKETS)")
    print("=" * 80)
    print(f"Current time: {now_cst.strftime('%A, %B %d, %Y at %I:%M %p CST')}")
    print(f"Target date: {target_date.strftime('%A, %B %d, %Y')}")
    print(f"Markets: FG + 1H (Spreads, Totals, Moneyline)")

    # Fetch games
    games = await fetch_upcoming_games(target_date)
    if not games:
        print("\n[WARN] No upcoming games found")
        return

    print(f"\nProcessing {len(games)} games...")

    # Fetch betting splits if enabled
    betting_splits_dict = {}
    if use_betting_splits:
        print("\nFetching public betting percentages...")
        try:
            betting_splits_dict = await fetch_public_betting_splits(games, source="auto")
            print(f"  [OK] Loaded betting splits for {len(betting_splits_dict)} games")
        except Exception as e:
            print(f"  [WARN] Failed to fetch betting splits: {e}")
            print(f"  [INFO] Continuing without betting splits data")

    # Initialize feature builder and unified prediction engine
    feature_builder = RichFeatureBuilder(league_id=12, season=settings.current_season)

    print("\nInitializing unified prediction engine...")
    engine = UnifiedPredictionEngine(models_dir=MODELS_DIR)
    print(f"  [OK] Loaded spread predictor")
    print(f"  [OK] Loaded total predictor")
    print(f"  [OK] Loaded moneyline predictor")

    # Generate predictions
    predictions = []

    for i, game in enumerate(games, 1):
        home_team = game.get("home_team")
        away_team = game.get("away_team")
        commence_time = game.get("commence_time")

        if not home_team or not away_team:
            print(f"\n[{i}/{len(games)}] Skipping game (missing team names)")
            continue

        # Format game time
        if commence_time:
            game_dt = parse_utc_time(commence_time)
            time_str = format_cst_time(game_dt)
        else:
            time_str = "TBD"

        print(f"\n{'='*80}")
        print(f"[{i}/{len(games)}] {away_team} @ {home_team}")
        print(f"Game time: {time_str}")
        print(f"{'='*80}")

        try:
            # Get betting splits if available
            game_key = f"{away_team}@{home_team}"
            splits = betting_splits_dict.get(game_key)

            # Build features
            features = await feature_builder.build_game_features(home_team, away_team, betting_splits=splits)

            # Extract all lines
            lines = extract_lines(game, home_team)

            # Generate predictions for ALL markets
            all_preds = engine.predict_all_markets(
                features,
                fg_spread_line=lines["fg_spread"],
                fg_total_line=lines["fg_total"],
                fg_home_ml_odds=lines["fg_home_ml"],
                fg_away_ml_odds=lines["fg_away_ml"],
                fh_spread_line=lines["fh_spread"],
                fh_total_line=lines["fh_total"],
                fh_home_ml_odds=lines["fh_home_ml"],
                fh_away_ml_odds=lines["fh_away_ml"],
                q1_spread_line=lines["q1_spread"],
                q1_total_line=lines["q1_total"],
                q1_home_ml_odds=lines["q1_home_ml"],
                q1_away_ml_odds=lines["q1_away_ml"],
            )

            fg_preds = all_preds["full_game"]
            fh_preds = all_preds["first_half"]
            q1_preds = all_preds["first_quarter"]

            # Display FG predictions
            print("\nFULL GAME:")
            display_market_predictions(fg_preds, lines, market_type="fg", features=features, home_team=home_team, away_team=away_team, splits=splits)

            # Display 1H predictions
            print("\nFIRST HALF:")
            display_market_predictions(fh_preds, lines, market_type="fh", features=features, home_team=home_team, away_team=away_team, splits=splits)

            print("\nFIRST QUARTER:")
            display_market_predictions(q1_preds, lines, market_type="q1", features=features, home_team=home_team, away_team=away_team, splits=splits)

            # Store prediction
            if commence_time:
                game_cst = to_cst(parse_utc_time(commence_time))
                date_cst = game_cst.strftime("%Y-%m-%d %I:%M %p CST")
            else:
                date_cst = datetime.now(CST).strftime("%Y-%m-%d %I:%M %p CST")

            prediction_dict = {
                "date": date_cst,
                "home_team": home_team,
                "away_team": away_team,
            }

            # Add FG predictions
            prediction_dict.update(format_predictions_for_csv(fg_preds, lines, prefix="fg"))

            # Add 1H predictions
            prediction_dict.update(format_predictions_for_csv(fh_preds, lines, prefix="fh"))

            # Add Q1 predictions
            prediction_dict.update(format_predictions_for_csv(q1_preds, lines, prefix="q1"))

            predictions.append(prediction_dict)

        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save and display results
    if predictions:
        save_predictions(predictions, target_date=target_date)
    else:
        print("\n[ERROR] No predictions generated")


def display_market_predictions(preds: dict, lines: dict, market_type: str, features: dict, home_team: str, away_team: str, splits: Any = None):
    """Display predictions for a market type (FG, 1H, Q1)."""
    if market_type == "fg":
        prefix = "fg"
    elif market_type == "fh":
        prefix = "fh"
    else:
        prefix = "q1"

    # Spread
    spread_pred = preds["spread"]
    print(f"  [SPREAD] Predicted margin: {spread_pred['predicted_margin']:+.1f} (home)")
    print(f"           Vegas line: {lines[f'{prefix}_spread'] or 'N/A'}")
    print(f"           Bet: {spread_pred['bet_side']} ({spread_pred['confidence']:.1%})")
    if spread_pred['edge'] is not None:
        print(f"           Edge: {spread_pred['edge']:+.1f} pts")
    
    # Generate rationale
    if spread_pred['edge'] is not None:
        rat = generate_rationale(
            play_type=f"{market_type.upper()}_SPREAD",
            pick=spread_pred['bet_side'],
            line=lines[f'{prefix}_spread'] or 0,
            odds=-110, # default
            edge=spread_pred['edge'],
            model_prob=spread_pred['confidence'],
            features=features,
            betting_splits=splits,
            home_team=home_team,
            away_team=away_team,
            model_prediction=spread_pred['predicted_margin'],
            market_line=lines[f'{prefix}_spread'] or 0
        )
        spread_pred['rationale'] = rat
        print(f"           Rationale: {rat}")

    print(f"           {'[PLAY]' if spread_pred['passes_filter'] else '[SKIP]'}"
          f"{'' if spread_pred['passes_filter'] else ': ' + spread_pred['filter_reason']}")

    # Total
    total_pred = preds["total"]
    print(f"  [TOTAL] Predicted total: {total_pred['predicted_total']:.1f}")
    print(f"          Vegas line: {lines[f'{prefix}_total'] or 'N/A'}")
    print(f"          Bet: {total_pred['bet_side']} ({total_pred['confidence']:.1%})")
    if total_pred['edge'] is not None:
        print(f"          Edge: {total_pred['edge']:+.1f} pts")
    
    if total_pred['edge'] is not None:
        rat = generate_rationale(
            play_type=f"{market_type.upper()}_TOTAL",
            pick=total_pred['bet_side'],
            line=lines[f'{prefix}_total'] or 0,
            odds=-110,
            edge=total_pred['edge'],
            model_prob=total_pred['confidence'],
            features=features,
            betting_splits=splits,
            home_team=home_team,
            away_team=away_team,
            model_prediction=total_pred['predicted_total'],
            market_line=lines[f'{prefix}_total'] or 0
        )
        total_pred['rationale'] = rat
        print(f"          Rationale: {rat}")

    print(f"          {'[PLAY]' if total_pred['passes_filter'] else '[SKIP]'}"
          f"{'' if total_pred['passes_filter'] else ': ' + total_pred['filter_reason']}")

    # Moneyline
    ml_pred = preds["moneyline"]
    print(f"  [ML] Predicted winner: {ml_pred['predicted_winner']} ({ml_pred['confidence']:.1%})")
    print(f"       Home odds: {lines[f'{prefix}_home_ml'] or 'N/A'}, Away odds: {lines[f'{prefix}_away_ml'] or 'N/A'}")
    if ml_pred['recommended_bet']:
        print(f"       Bet: {ml_pred['recommended_bet']}")
        edge = ml_pred['home_edge'] if ml_pred['recommended_bet'] == 'home' else ml_pred['away_edge']
        if edge is not None:
            print(f"       Value edge: {edge:+.1%}")
            rat = f"[MODEL] Model assigns {ml_pred['confidence']:.1%} to {ml_pred['predicted_winner']}."
            if edge >= 0.05:
                rat += f" High value edge of {edge:+.1%} found at market odds."
            ml_pred['rationale'] = rat
            print(f"       Rationale: {rat}")

    print(f"       {'[PLAY]' if ml_pred['passes_filter'] else '[SKIP]'}"
          f"{'' if ml_pred['passes_filter'] else ': ' + ml_pred['filter_reason']}")


def format_predictions_for_csv(preds: dict, lines: dict, prefix: str) -> dict:
    """Format predictions for CSV output."""
    spread_pred = preds["spread"]
    total_pred = preds["total"]
    ml_pred = preds["moneyline"]

    return {
        # Spread
        f"{prefix}_spread_line": lines[f"{prefix}_spread"],
        f"{prefix}_spread_pred_margin": round(spread_pred['predicted_margin'], 1),
        f"{prefix}_spread_edge": round(spread_pred['edge'], 1) if spread_pred['edge'] else None,
        f"{prefix}_spread_bet_side": spread_pred['bet_side'],
        f"{prefix}_spread_confidence": round(spread_pred['confidence'], 3),
        f"{prefix}_spread_passes_filter": spread_pred['passes_filter'],
        f"{prefix}_spread_filter_reason": spread_pred['filter_reason'] or "",
        f"{prefix}_spread_rationale": spread_pred.get('rationale', ""),
        # Total
        f"{prefix}_total_line": lines[f"{prefix}_total"],
        f"{prefix}_total_pred": round(total_pred['predicted_total'], 1),
        f"{prefix}_total_edge": round(total_pred['edge'], 1) if total_pred['edge'] else None,
        f"{prefix}_total_bet_side": total_pred['bet_side'],
        f"{prefix}_total_confidence": round(total_pred['confidence'], 3),
        f"{prefix}_total_passes_filter": total_pred['passes_filter'],
        f"{prefix}_total_filter_reason": total_pred['filter_reason'] or "",
        f"{prefix}_total_rationale": total_pred.get('rationale', ""),
        # Moneyline
        f"{prefix}_ml_home_odds": lines[f"{prefix}_home_ml"],
        f"{prefix}_ml_away_odds": lines[f"{prefix}_away_ml"],
        f"{prefix}_ml_predicted_winner": ml_pred['predicted_winner'],
        f"{prefix}_ml_recommended_bet": ml_pred['recommended_bet'],
        f"{prefix}_ml_confidence": round(ml_pred['confidence'], 3),
        f"{prefix}_ml_home_edge": round(ml_pred['home_edge'], 3) if ml_pred['home_edge'] else None,
        f"{prefix}_ml_away_edge": round(ml_pred['away_edge'], 3) if ml_pred['away_edge'] else None,
        f"{prefix}_ml_passes_filter": ml_pred['passes_filter'],
        f"{prefix}_ml_filter_reason": ml_pred['filter_reason'] or "",
        f"{prefix}_ml_rationale": ml_pred.get('rationale', ""),
    }


def generate_formatted_text_report(df: pd.DataFrame, target_date: datetime.date) -> str:
    """Generate formatted text report similar to slate_analysis format."""
    lines = []
    
    # Header
    date_str = target_date.strftime("%A, %B %d, %Y")
    lines.append("=" * 100)
    lines.append(f"NBA COMPREHENSIVE SLATE ANALYSIS - {date_str}")
    lines.append("=" * 100)
    lines.append("")
    
    def calculate_fire_rating(confidence: float, edge: float, edge_type: str) -> int:
        """Calculate fire rating (1-5) based on confidence and edge."""
        # Normalize edge to 0-1 scale for comparison
        if pd.isna(edge):
            edge_norm = 0.0
        elif edge_type == "pct":
            edge_norm = min(abs(edge) / 0.20, 1.0)  # 20% edge = max
        else:  # pts
            edge_norm = min(abs(edge) / 10.0, 1.0)  # 10 pts = max
        
        # Combine confidence and edge (weighted average)
        combined_score = (confidence * 0.6) + (edge_norm * 0.4)
        
        # Map to 1-5 fires
        if combined_score >= 0.85:
            return 5
        elif combined_score >= 0.70:
            return 4
        elif combined_score >= 0.60:
            return 3
        elif combined_score >= 0.52:
            return 2
        else:
            return 1
    
    def american_odds_to_implied_prob(odds: int) -> float:
        """Convert American odds to implied probability."""
        if odds < 0:
            return abs(odds) / (abs(odds) + 100)
        else:
            return 100 / (odds + 100)
    
    # Collect all plays first for executive summary
    all_plays_summary = []
    for _, row in df.iterrows():
        matchup = f"{row['away_team']} @ {row['home_team']}"
        
        # FG Spread
        if row.get("fg_spread_passes_filter") and pd.notna(row.get('fg_spread_line')):
            edge = row.get('fg_spread_edge')
            conf = row['fg_spread_confidence']
            fire = calculate_fire_rating(conf, edge if pd.notna(edge) else 0, "pts")
            model_prob = conf
            market_prob = 0.524  # -110 odds
            model_pred = row.get('fg_spread_pred_margin')
            market_line = row.get('fg_spread_line')
            all_plays_summary.append({
                "matchup": matchup,
                "market": "FG Spread",
                "pick": f"{row['fg_spread_bet_side']} {row['fg_spread_line']:+.1f}",
                "edge": edge,
                "confidence": conf,
                "edge_type": "pts",
                "fire_rating": fire,
                "model_prob": model_prob,
                "market_prob": market_prob,
                "model_pred": model_pred,  # Predicted margin
                "market_line": market_line,  # Market spread line
                "pred_type": "margin",
                "odds": -110,
            })
        
        # FG Total
        if row.get("fg_total_passes_filter") and pd.notna(row.get('fg_total_line')):
            edge = row.get('fg_total_edge')
            conf = row['fg_total_confidence']
            fire = calculate_fire_rating(conf, edge if pd.notna(edge) else 0, "pts")
            model_prob = conf
            market_prob = 0.524  # -110 odds
            model_pred = row.get('fg_total_pred')  # Predicted total points
            market_line = row.get('fg_total_line')  # Market total line
            all_plays_summary.append({
                "matchup": matchup,
                "market": "FG Total",
                "pick": f"{row['fg_total_bet_side']} {row['fg_total_line']:.1f}",
                "edge": edge,
                "confidence": conf,
                "edge_type": "pts",
                "fire_rating": fire,
                "model_prob": model_prob,
                "market_prob": market_prob,
                "model_pred": model_pred,
                "market_line": market_line,
                "pred_type": "total",
                "odds": -110,
            })
        
        # FG Moneyline
        if row.get("fg_ml_passes_filter") and row.get("fg_ml_recommended_bet"):
            bet = row['fg_ml_recommended_bet']
            # Only use odds that match the bet side - don't fallback to opposite side
            if bet == 'home':
                odds = int(row['fg_ml_home_odds']) if pd.notna(row.get('fg_ml_home_odds')) else None
            elif bet == 'away':
                odds = int(row['fg_ml_away_odds']) if pd.notna(row.get('fg_ml_away_odds')) else None
            else:
                odds = None
            if pd.notna(odds) and odds is not None:
                edge = row['fg_ml_home_edge'] if bet == 'home' else row['fg_ml_away_edge']
                conf = row['fg_ml_confidence']
                fire = calculate_fire_rating(conf, edge if pd.notna(edge) else 0, "pct")
                model_prob = conf  # Win probability
                market_prob = american_odds_to_implied_prob(odds)
                all_plays_summary.append({
                    "matchup": matchup,
                    "market": "FG Moneyline",
                    "pick": f"{bet.capitalize()}",
                    "edge": edge,
                    "confidence": conf,
                    "edge_type": "pct",
                    "fire_rating": fire,
                    "model_prob": model_prob,
                    "market_prob": market_prob,
                    "odds": odds,
                })
        
        # 1H Spread
        if row.get("fh_spread_passes_filter") and pd.notna(row.get('fh_spread_line')):
            edge = row.get('fh_spread_edge')
            conf = row['fh_spread_confidence']
            fire = calculate_fire_rating(conf, edge if pd.notna(edge) else 0, "pts")
            model_prob = conf
            market_prob = 0.524  # -110 odds
            model_pred = row.get('fh_spread_pred_margin')
            market_line = row.get('fh_spread_line')
            all_plays_summary.append({
                "matchup": matchup,
                "market": "1H Spread",
                "pick": f"{row['fh_spread_bet_side']} {row['fh_spread_line']:+.1f}",
                "edge": edge,
                "confidence": conf,
                "edge_type": "pts",
                "fire_rating": fire,
                "model_prob": model_prob,
                "market_prob": market_prob,
                "model_pred": model_pred,
                "market_line": market_line,
                "pred_type": "margin",
                "odds": -110,
            })
        
        # 1H Total
        if row.get("fh_total_passes_filter") and pd.notna(row.get('fh_total_line')):
            edge = row.get('fh_total_edge')
            conf = row['fh_total_confidence']
            fire = calculate_fire_rating(conf, edge if pd.notna(edge) else 0, "pts")
            model_prob = conf
            market_prob = 0.524  # -110 odds
            model_pred = row.get('fh_total_pred')
            market_line = row.get('fh_total_line')
            all_plays_summary.append({
                "matchup": matchup,
                "market": "1H Total",
                "pick": f"{row['fh_total_bet_side']} {row['fh_total_line']:.1f}",
                "edge": edge,
                "confidence": conf,
                "edge_type": "pts",
                "fire_rating": fire,
                "model_prob": model_prob,
                "market_prob": market_prob,
                "model_pred": model_pred,
                "market_line": market_line,
                "pred_type": "total",
                "odds": -110,
            })
        
        # 1H Moneyline
        if row.get("fh_ml_passes_filter") and row.get("fh_ml_recommended_bet"):
            bet = row['fh_ml_recommended_bet']
            # Only use odds that match the bet side - don't fallback to opposite side
            if bet == 'home':
                odds = int(row['fh_ml_home_odds']) if pd.notna(row.get('fh_ml_home_odds')) else None
            elif bet == 'away':
                odds = int(row['fh_ml_away_odds']) if pd.notna(row.get('fh_ml_away_odds')) else None
            else:
                odds = None
            if pd.notna(odds) and odds is not None:
                edge = row['fh_ml_home_edge'] if bet == 'home' else row['fh_ml_away_edge']
                conf = row['fh_ml_confidence']
                fire = calculate_fire_rating(conf, edge if pd.notna(edge) else 0, "pct")
                model_prob = conf
                market_prob = american_odds_to_implied_prob(odds)
                all_plays_summary.append({
                    "matchup": matchup,
                    "market": "1H Moneyline",
                    "pick": f"{bet.capitalize()}",
                    "edge": edge,
                    "confidence": conf,
                    "edge_type": "pct",
                    "fire_rating": fire,
                    "model_prob": model_prob,
                    "market_prob": market_prob,
                    "odds": odds,
                })

        # Q1 Spread
        if row.get("q1_spread_passes_filter") and pd.notna(row.get('q1_spread_line')):
            edge = row.get('q1_spread_edge')
            conf = row['q1_spread_confidence']
            fire = calculate_fire_rating(conf, edge if pd.notna(edge) else 0, "pts")
            model_prob = conf
            market_prob = 0.524
            model_pred = row.get('q1_spread_pred_margin')
            market_line = row.get('q1_spread_line')
            all_plays_summary.append({
                "matchup": matchup,
                "market": "Q1 Spread",
                "pick": f"{row['q1_spread_bet_side']} {row['q1_spread_line']:+.1f}",
                "edge": edge,
                "confidence": conf,
                "edge_type": "pts",
                "fire_rating": fire,
                "model_prob": model_prob,
                "market_prob": market_prob,
                "model_pred": model_pred,
                "market_line": market_line,
                "pred_type": "margin",
                "odds": -110,
            })

        # Q1 Total
        if row.get("q1_total_passes_filter") and pd.notna(row.get('q1_total_line')):
            edge = row.get('q1_total_edge')
            conf = row['q1_total_confidence']
            fire = calculate_fire_rating(conf, edge if pd.notna(edge) else 0, "pts")
            model_prob = conf
            market_prob = 0.524
            model_pred = row.get('q1_total_pred')
            market_line = row.get('q1_total_line')
            all_plays_summary.append({
                "matchup": matchup,
                "market": "Q1 Total",
                "pick": f"{row['q1_total_bet_side']} {row['q1_total_line']:.1f}",
                "edge": edge,
                "confidence": conf,
                "edge_type": "pts",
                "fire_rating": fire,
                "model_prob": model_prob,
                "market_prob": market_prob,
                "model_pred": model_pred,
                "market_line": market_line,
                "pred_type": "total",
                "odds": -110,
            })

        # Q1 Moneyline
        if row.get("q1_ml_passes_filter") and row.get("q1_ml_recommended_bet"):
            bet = row['q1_ml_recommended_bet']
            if bet == 'home':
                odds = int(row['q1_ml_home_odds']) if pd.notna(row.get('q1_ml_home_odds')) else None
            elif bet == 'away':
                odds = int(row['q1_ml_away_odds']) if pd.notna(row.get('q1_ml_away_odds')) else None
            else:
                odds = None
            if pd.notna(odds) and odds is not None:
                edge = row['q1_ml_home_edge'] if bet == 'home' else row['q1_ml_away_edge']
                conf = row['q1_ml_confidence']
                fire = calculate_fire_rating(conf, edge if pd.notna(edge) else 0, "pct")
                model_prob = conf
                market_prob = american_odds_to_implied_prob(odds)
                all_plays_summary.append({
                    "matchup": matchup,
                    "market": "Q1 Moneyline",
                    "pick": f"{bet.capitalize()}",
                    "edge": edge,
                    "confidence": conf,
                    "edge_type": "pct",
                    "fire_rating": fire,
                    "model_prob": model_prob,
                    "market_prob": market_prob,
                    "odds": odds,
                })
    
    # EXECUTIVE SUMMARY (Bottom Line Up Front)
    lines.append("=" * 100)
    lines.append("EXECUTIVE SUMMARY - BOTTOM LINE UP FRONT")
    lines.append("=" * 100)
    lines.append("")
    
    lines.append(f"TOTAL GAMES: {len(df)}")
    lines.append(f"TOTAL PLAYS: {len(all_plays_summary)}")
    lines.append("")
    
    if all_plays_summary:
        # Sort by confidence (descending)
        sorted_plays = sorted(all_plays_summary, key=lambda x: x['confidence'], reverse=True)
        
        # Count by market
        fg_count = sum(1 for p in all_plays_summary if p['market'].startswith('FG'))
        fh_count = sum(1 for p in all_plays_summary if p['market'].startswith('1H'))
        q1_count = sum(1 for p in all_plays_summary if p['market'].startswith('Q1'))
        spread_count = sum(1 for p in all_plays_summary if 'Spread' in p['market'])
        total_count = sum(1 for p in all_plays_summary if 'Total' in p['market'])
        ml_count = sum(1 for p in all_plays_summary if 'Moneyline' in p['market'])
        
        lines.append(f"BREAKDOWN:")
        lines.append(f"  Full Game: {fg_count} | First Half: {fh_count} | First Quarter: {q1_count}")
        lines.append(f"  Spreads: {spread_count} | Totals: {total_count} | Moneyline: {ml_count}")
        lines.append("")
        lines.append("TOP RECOMMENDED PLAYS (sorted by fire rating, then confidence):")
        lines.append("")
        
        # Sort by fire rating (desc) then confidence (desc)
        sorted_plays = sorted(all_plays_summary, key=lambda x: (x['fire_rating'], x['confidence']), reverse=True)
        
        # Show top 10 plays
        for idx, play in enumerate(sorted_plays[:10], 1):
            # Fire rating (1-5 fires)
            fire_display = "🔥" * play['fire_rating'] + " " * (5 - play['fire_rating'])
            
            # Edge
            edge_str = ""
            if pd.notna(play['edge']):
                if play['edge_type'] == 'pct':
                    edge_str = f"{play['edge']:+.1%}"
                else:
                    edge_str = f"{play['edge']:+.1f} pts"
            else:
                edge_str = "N/A"
            
            # Odds display
            if play.get('odds'):
                if play['odds'] == -110:
                    odds_str = "-110"
                else:
                    odds_str = f"{play['odds']:+d}"
            else:
                odds_str = "N/A"
            
            # Model prediction vs Market line
            model_pred_str = ""
            if play.get('pred_type') == 'margin' and pd.notna(play.get('model_pred')) and pd.notna(play.get('market_line')):
                model_pred_str = f"Model predicts {play['model_pred']:+.1f} margin | Market line: {play['market_line']:+.1f}"
            elif play.get('pred_type') == 'total' and pd.notna(play.get('model_pred')) and pd.notna(play.get('market_line')):
                # For totals, show both regression prediction and classifier recommendation
                model_pred_val = play['model_pred']
                market_val = play['market_line']
                pick_bet = play['pick'].split()[0].upper()  # Extract "OVER" or "UNDER" from pick
                
                # Check if regression prediction aligns with classifier recommendation
                if (pick_bet == 'OVER' and model_pred_val > market_val) or \
                   (pick_bet == 'UNDER' and model_pred_val < market_val):
                    # Regression and classifier agree
                    model_pred_str = f"Model regression: {model_pred_val:.1f} pts | Market: {market_val:.1f} | Classifier: {pick_bet} {play['confidence']:.0%}"
                else:
                    # Regression and classifier disagree - show both
                    reg_recommendation = "OVER" if model_pred_val > market_val else "UNDER"
                    model_pred_str = f"Model regression: {model_pred_val:.1f} pts → suggests {reg_recommendation} | Market: {market_val:.1f} | Model classifier: {pick_bet} {play['confidence']:.0%}"
            else:
                # Moneyline or missing data - use probability comparison
                model_vs_market = play['model_prob'] - play['market_prob']
                prob_diff_str = f"{model_vs_market:+.1%}"
                model_pred_str = f"Model: {play['model_prob']:.1%} win prob | Market: {play['market_prob']:.1%} | Diff: {prob_diff_str}"
            
            lines.append(f"  {idx}. [{fire_display}] {play['matchup']}")
            lines.append(f"     {play['market']}: {play['pick']} | Odds: {odds_str}")
            lines.append(f"     {model_pred_str}")
            lines.append(f"     Confidence: {play['confidence']:.1%} | Edge: {edge_str}")
            lines.append("")
    else:
        lines.append("NO PLAYS TODAY - All games filtered out")
        lines.append("")
    
    lines.append("=" * 100)
    lines.append("DETAILED GAME-BY-GAME ANALYSIS")
    lines.append("=" * 100)
    lines.append("")
    
    # Process each game
    for game_idx, (_, row) in enumerate(df.iterrows(), 1):
        matchup = f"{row['away_team']} @ {row['home_team']}"
        game_time = row['date']
        
        lines.append("-" * 100)
        lines.append(f"GAME {game_idx}: {matchup}")
        lines.append(f"Time: {game_time}")
        lines.append("")
        
        # Collect top plays for this game
        top_plays = []
        
        # FG Spread
        if row.get("fg_spread_passes_filter") and pd.notna(row.get('fg_spread_line')):
            top_plays.append({
                "market": "FG Spread",
                "pick": f"{row['fg_spread_bet_side']} {row['fg_spread_line']}",
                "edge": row.get('fg_spread_edge'),
                "confidence": row['fg_spread_confidence'],
                "rationale": row.get('fg_spread_rationale', ''),
            })
        
        # FG Total
        if row.get("fg_total_passes_filter") and pd.notna(row.get('fg_total_line')):
            top_plays.append({
                "market": "FG Total",
                "pick": f"{row['fg_total_bet_side']} {row['fg_total_line']:.1f}",
                "edge": row.get('fg_total_edge'),
                "confidence": row['fg_total_confidence'],
                "rationale": row.get('fg_total_rationale', ''),
            })
        
        # FG Moneyline
        if row.get("fg_ml_passes_filter") and row.get("fg_ml_recommended_bet"):
            bet = row['fg_ml_recommended_bet']
            # Only use odds that match the bet side
            if bet == 'home':
                odds = row['fg_ml_home_odds'] if pd.notna(row.get('fg_ml_home_odds')) else None
            elif bet == 'away':
                odds = row['fg_ml_away_odds'] if pd.notna(row.get('fg_ml_away_odds')) else None
            else:
                odds = None
            if pd.notna(odds):
                edge = row['fg_ml_home_edge'] if bet == 'home' else row['fg_ml_away_edge']
                top_plays.append({
                    "market": "FG Moneyline",
                    "pick": f"{bet} ({int(odds):+d})",
                    "edge": edge,
                    "confidence": row['fg_ml_confidence'],
                    "rationale": row.get('fg_ml_rationale', ''),
                })
        
        # Display top plays
        if top_plays:
            lines.append("[TOP PLAYS] (All confidence levels):")
            for idx, play in enumerate(top_plays[:3], 1):  # Top 3 plays
                edge_str = f"{play['edge']:+.1f} pts" if pd.notna(play['edge']) else "N/A"
                lines.append(f"   {idx}. {play['market']}: {play['pick']}")
                lines.append(f"      Edge: {edge_str} | Confidence: {play['confidence']:.1%}")
                if play['rationale']:
                    lines.append(f"      {play['rationale']}")
            lines.append("")
        
        # FULL GAME ANALYSIS
        lines.append("FULL GAME ANALYSIS:")
        lines.append("")
        
        # Spread
        lines.append("   SPREAD:")
        if pd.notna(row.get('fg_spread_line')):
            lines.append(f"      Market Line: {row['home_team']} {row['fg_spread_line']:+.1f} (-110)")
            if pd.notna(row.get('fg_spread_pred_margin')):
                lines.append(f"      Model Projects: {row['fg_spread_pred_margin']:+.1f} margin (home)")
            if pd.notna(row.get('fg_spread_edge')):
                lines.append(f"      Edge: {row['fg_spread_edge']:+.1f} pts")
            if row.get("fg_spread_passes_filter"):
                lines.append(f"      [PLAY] {row['fg_spread_bet_side']} {row['fg_spread_line']:+.1f} (-110)")
            else:
                lines.append(f"      [SKIP] {row.get('fg_spread_filter_reason', 'Filtered')}")
            lines.append(f"      Win Probability: {row['fg_spread_confidence']:.1%} | Confidence: {row['fg_spread_confidence']:.1%}")
            if row.get('fg_spread_rationale'):
                lines.append(f"      Rationale: {row['fg_spread_rationale']}")
        else:
            lines.append("      Market Line: N/A")
        lines.append("")
        
        # Total
        lines.append("   TOTAL:")
        if pd.notna(row.get('fg_total_line')):
            lines.append(f"      Market Line: O/U {row['fg_total_line']:.1f} (-110)")
            if pd.notna(row.get('fg_total_pred')):
                lines.append(f"      Model Projects: {row['fg_total_pred']:.1f} total points")
            if pd.notna(row.get('fg_total_edge')):
                lines.append(f"      Edge: {row['fg_total_edge']:+.1f} pts")
            if row.get("fg_total_passes_filter"):
                lines.append(f"      [PLAY] {row['fg_total_bet_side']} {row['fg_total_line']:.1f} (-110)")
            else:
                lines.append(f"      [SKIP] {row.get('fg_total_filter_reason', 'Filtered')}")
            lines.append(f"      Probability: {row['fg_total_confidence']:.1%}")
            if row.get('fg_total_rationale'):
                lines.append(f"      Rationale: {row['fg_total_rationale']}")
        else:
            lines.append("      Market Line: N/A")
        lines.append("")
        
        # Moneyline
        lines.append("   MONEYLINE:")
        if pd.notna(row.get('fg_ml_home_odds')):
            home_odds = int(row['fg_ml_home_odds'])
            away_odds = int(row['fg_ml_away_odds']) if pd.notna(row.get('fg_ml_away_odds')) else None
            away_odds_str = f"{away_odds:+d}" if away_odds is not None else "N/A"
            lines.append(f"      Market Odds: {row['home_team']} ({home_odds:+d}) | {row['away_team']} ({away_odds_str})")
            if row.get("fg_ml_recommended_bet") and row.get("fg_ml_passes_filter"):
                bet = row['fg_ml_recommended_bet']
                edge = row['fg_ml_home_edge'] if bet == 'home' else row['fg_ml_away_edge']
                lines.append(f"      [PLAY] {bet.capitalize()} ({home_odds if bet == 'home' else away_odds:+d})")
                if pd.notna(edge):
                    lines.append(f"      Value Edge: {edge:+.1%}")
            lines.append(f"      Confidence: {row['fg_ml_confidence']:.1%}")
            if row.get('fg_ml_rationale'):
                lines.append(f"      Rationale: {row['fg_ml_rationale']}")
        else:
            lines.append("      Market Odds: N/A")
        lines.append("")
        
        # FIRST HALF ANALYSIS
        lines.append("FIRST HALF ANALYSIS:")
        lines.append("")
        
        # 1H Spread
        lines.append("   1H SPREAD:")
        if pd.notna(row.get('fh_spread_line')):
            lines.append(f"      Market Line: {row['home_team']} {row['fh_spread_line']:+.1f} (-110)")
            if pd.notna(row.get('fh_spread_pred_margin')):
                lines.append(f"      Model Projects: {row['fh_spread_pred_margin']:+.1f} margin (home)")
            if pd.notna(row.get('fh_spread_edge')):
                lines.append(f"      Edge: {row['fh_spread_edge']:+.1f} pts")
            if row.get("fh_spread_passes_filter"):
                lines.append(f"      [PLAY] {row['fh_spread_bet_side']} {row['fh_spread_line']:+.1f} (-110)")
            else:
                lines.append(f"      [SKIP] {row.get('fh_spread_filter_reason', 'Filtered')}")
            if row.get('fh_spread_rationale'):
                lines.append(f"      Rationale: {row['fh_spread_rationale']}")
        else:
            lines.append("      Market Line: N/A")
        lines.append("")
        
        # 1H Total
        lines.append("   1H TOTAL:")
        if pd.notna(row.get('fh_total_line')):
            lines.append(f"      Market Line: O/U {row['fh_total_line']:.1f} (-110)")
            if pd.notna(row.get('fh_total_pred')):
                lines.append(f"      Model Projects: {row['fh_total_pred']:.1f} total points at half")
            if pd.notna(row.get('fh_total_edge')):
                lines.append(f"      Edge: {row['fh_total_edge']:+.1f} pts")
            if row.get("fh_total_passes_filter"):
                lines.append(f"      [PLAY] {row['fh_total_bet_side']} {row['fh_total_line']:.1f} (-110)")
            else:
                lines.append(f"      [SKIP] {row.get('fh_total_filter_reason', 'Filtered')}")
            if row.get('fh_total_rationale'):
                lines.append(f"      Rationale: {row['fh_total_rationale']}")
        else:
            lines.append("      Market Line: N/A")
        lines.append("")
        
        # 1H Moneyline
        lines.append("   1H MONEYLINE:")
        if pd.notna(row.get('fh_ml_home_odds')):
            home_odds = int(row['fh_ml_home_odds'])
            away_odds = int(row['fh_ml_away_odds']) if pd.notna(row.get('fh_ml_away_odds')) else None
            away_odds_str = f"{away_odds:+d}" if away_odds is not None else "N/A"
            lines.append(f"      Market Odds: {row['home_team']} ({home_odds:+d}) | {row['away_team']} ({away_odds_str})")
            if row.get("fh_ml_recommended_bet") and row.get("fh_ml_passes_filter"):
                bet = row['fh_ml_recommended_bet']
                edge = row['fh_ml_home_edge'] if bet == 'home' else row['fh_ml_away_edge']
                lines.append(f"      [PLAY] {bet.capitalize()} ({home_odds if bet == 'home' else away_odds:+d})")
                if pd.notna(edge):
                    lines.append(f"      Value Edge: {edge:+.1%}")
            if row.get('fh_ml_rationale'):
                lines.append(f"      Rationale: {row['fh_ml_rationale']}")
        else:
            lines.append("      Market Odds: N/A")
        lines.append("")
    
    # Footer
    lines.append("=" * 100)
    now_str = datetime.now(CST).strftime("%Y-%m-%d %I:%M %p CST")
    lines.append(f"Generated: {now_str}")
    lines.append("")
    lines.append("Legend:")
    lines.append("  [PLAY] = Passes model filters (recommended bet)")
    lines.append("  [SKIP] = Filtered out (low confidence or insufficient edge)")
    lines.append("  Edge = Model projection vs market line (positive = betting opportunity)")
    lines.append("  Confidence = Model confidence level (accounts for uncertainty)")
    lines.append("  FG = Full Game | 1H = First Half")
    lines.append("")
    
    return "\n".join(lines)


def save_predictions(predictions: list, target_date: Optional[datetime.date] = None):
    """Save predictions and generate betting card."""
    output_path = DATA_DIR / "processed" / "predictions_v3.csv"
    df = pd.DataFrame(predictions)
    df.to_csv(output_path, index=False)
    print(f"\n{'='*80}")
    print(f"[OK] Saved {len(predictions)} predictions to {output_path}")
    print(f"{'='*80}")

    # Generate betting card with ALL markets
    all_plays = []

    for _, row in df.iterrows():
        matchup = f"{row['away_team']} @ {row['home_team']}"
        date = row['date']

        # FG Spreads
        if row.get("fg_spread_passes_filter") and pd.notna(row.get('fg_spread_line')):
            all_plays.append({
                "matchup": matchup,
                "date": date,
                "period": "FG",
                "market": "SPREAD",
                "pick": f"{row['fg_spread_bet_side']} {row['fg_spread_line']}",
                "confidence": row['fg_spread_confidence'],
                "edge": row.get('fg_spread_edge'),
                "rationale": row.get('fg_spread_rationale', ""),
            })

        # FG Totals
        if row.get("fg_total_passes_filter") and pd.notna(row.get('fg_total_line')):
            all_plays.append({
                "matchup": matchup,
                "date": date,
                "period": "FG",
                "market": "TOTAL",
                "pick": f"{row['fg_total_bet_side']} {row['fg_total_line']:.1f}",
                "confidence": row['fg_total_confidence'],
                "edge": row.get('fg_total_edge'),
                "rationale": row.get('fg_total_rationale', ""),
            })

        # FG Moneyline
        if row.get("fg_ml_passes_filter") and row.get("fg_ml_recommended_bet"):
            bet = row['fg_ml_recommended_bet']
            # Only use odds that match the bet side
            if bet == 'home':
                odds = row['fg_ml_home_odds'] if pd.notna(row.get('fg_ml_home_odds')) else None
            elif bet == 'away':
                odds = row['fg_ml_away_odds'] if pd.notna(row.get('fg_ml_away_odds')) else None
            else:
                odds = None
            if pd.notna(odds):
                edge = row['fg_ml_home_edge'] if bet == 'home' else row['fg_ml_away_edge']
                all_plays.append({
                    "matchup": matchup,
                    "date": date,
                    "period": "FG",
                    "market": "ML",
                    "pick": f"{bet} ({int(odds):+d})",
                    "confidence": row['fg_ml_confidence'],
                    "edge": edge,
                    "rationale": row.get('fg_ml_rationale', ""),
                })

        # 1H Spreads
        if row.get("fh_spread_passes_filter") and pd.notna(row.get('fh_spread_line')):
            all_plays.append({
                "matchup": matchup,
                "date": date,
                "period": "1H",
                "market": "SPREAD",
                "pick": f"{row['fh_spread_bet_side']} {row['fh_spread_line']}",
                "confidence": row['fh_spread_confidence'],
                "edge": row.get('fh_spread_edge'),
                "rationale": row.get('fh_spread_rationale', ""),
            })

        # 1H Totals
        if row.get("fh_total_passes_filter") and pd.notna(row.get('fh_total_line')):
            all_plays.append({
                "matchup": matchup,
                "date": date,
                "period": "1H",
                "market": "TOTAL",
                "pick": f"{row['fh_total_bet_side']} {row['fh_total_line']:.1f}",
                "confidence": row['fh_total_confidence'],
                "edge": row.get('fh_total_edge'),
                "rationale": row.get('fh_total_rationale', ""),
            })

        # 1H Moneyline
        if row.get("fh_ml_passes_filter") and row.get("fh_ml_recommended_bet"):
            bet = row['fh_ml_recommended_bet']
            # Only use odds that match the bet side
            if bet == 'home':
                odds = row['fh_ml_home_odds'] if pd.notna(row.get('fh_ml_home_odds')) else None
            elif bet == 'away':
                odds = row['fh_ml_away_odds'] if pd.notna(row.get('fh_ml_away_odds')) else None
            else:
                odds = None
            if pd.notna(odds):
                edge = row['fh_ml_home_edge'] if bet == 'home' else row['fh_ml_away_edge']
                all_plays.append({
                    "matchup": matchup,
                    "date": date,
                    "period": "1H",
                    "market": "ML",
                    "pick": f"{bet} ({int(odds):+d})",
                    "confidence": row['fh_ml_confidence'],
                    "edge": edge,
                    "rationale": row.get('fh_ml_rationale', ""),
                })

    print("\n" + "=" * 80)
    print("BETTING CARD - ALL MARKETS (FG + 1H)")
    print("=" * 80)

    if all_plays:
        # Sort by confidence descending
        all_plays = sorted(all_plays, key=lambda x: x['confidence'], reverse=True)

        for play in all_plays:
            print(f"\n{play['matchup']}")
            print(f"  Game Time: {play['date']}")
            print(f"  Period: {play['period']} | Market: {play['market']}")
            print(f"  Pick: {play['pick']} ({play['confidence']:.1%})")
            if play['edge'] is not None:
                if play['market'] == 'ML':
                    print(f"  Value Edge: {play['edge']:+.1%}")
                else:
                    print(f"  Edge: {play['edge']:+.1f} pts")
            if play['rationale']:
                print(f"  Rationale: {play['rationale']}")

        print("\n" + "=" * 80)
        print(f"TOTAL PLAYS: {len(all_plays)}")

        # Count by market
        fg_count = sum(1 for p in all_plays if p['period'] == 'FG')
        fh_count = sum(1 for p in all_plays if p['period'] == '1H')
        spread_count = sum(1 for p in all_plays if p['market'] == 'SPREAD')
        total_count = sum(1 for p in all_plays if p['market'] == 'TOTAL')
        ml_count = sum(1 for p in all_plays if p['market'] == 'ML')

        print(f"  FG: {fg_count}, 1H: {fh_count}")
        print(f"  Spreads: {spread_count}, Totals: {total_count}, ML: {ml_count}")
        print("=" * 80)

        # Save betting card
        betting_card_df = pd.DataFrame(all_plays)
        betting_card_path = DATA_DIR / "processed" / "betting_card_v3.csv"
        betting_card_df.to_csv(betting_card_path, index=False)
        print(f"[OK] Saved betting card to {betting_card_path}")
        
        # Generate and save formatted text report
        if target_date:
            text_report = generate_formatted_text_report(df, target_date)
            text_path = DATA_DIR / "processed" / f"slate_analysis_{target_date.strftime('%Y%m%d')}.txt"
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(text_report)
            print(f"[OK] Saved formatted text report to {text_path}")

    else:
        print("\nNO PLAYS TODAY")
        print("All games filtered out - no bets meet criteria")
        print("=" * 80)
        
        # Still generate text report even if no plays
        if target_date and len(df) > 0:
            text_report = generate_formatted_text_report(df, target_date)
            text_path = DATA_DIR / "processed" / f"slate_analysis_{target_date.strftime('%Y%m%d')}.txt"
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(text_report)
            print(f"[OK] Saved formatted text report to {text_path}")


def main():
    # Set random seeds
    np.random.seed(42)
    random.seed(42)
    logger.info("Random seeds set to 42 for reproducibility")
    
    print("\n⚠️  NOTE: Running in LOCAL mode. For Docker execution, use 'python scripts/run_slate.py' or './run.ps1'")

    parser = argparse.ArgumentParser(description="Generate NBA predictions for all markets")
    parser.add_argument("--date", help="Date for predictions (YYYY-MM-DD, 'today', or 'tomorrow')")
    parser.add_argument("--no-betting-splits", action="store_true",
                       help="Disable betting splits integration")
    args = parser.parse_args()

    asyncio.run(predict_games_async(args.date, use_betting_splits=not args.no_betting_splits))


if __name__ == "__main__":
    main()
