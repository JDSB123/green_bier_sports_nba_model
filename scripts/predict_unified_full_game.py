"""
Generate predictions for ALL markets using unified prediction engine.

Production-ready predictor with smart filtering for all markets:
- Full Game: Spreads, Totals
- First Half: Spreads, Totals

WEIGHTED COMBINATION SYSTEM (v34.3.0):
Sharp signals now affect THE PICK (not just confidence):
- combined_edge = (model_edge × 0.55) + (sharp_edge × 0.45)
- If signals agree → strong play with high confidence
- If signals conflict → combined edge determines final side (or NO PLAY)
- NO more "bet against sharps with lower confidence"

Key change from v34.1.0:
- OLD: Model says HOME, sharps say AWAY → Still bet HOME with penalty
- NEW: Model says HOME, sharps say AWAY → Math decides (or NO PLAY)
"""
from src.prediction import UnifiedPredictionEngine
from src.features.rich_features import RichFeatureBuilder
from src.ingestion.betting_splits import (
    fetch_public_betting_splits,
    fetch_sharp_square_lines,
    sharp_square_to_features,
    SharpDataUnavailableError,
)
# v34.3.0: Weighted Combination System (sharp signals affect THE PICK, not just confidence)
from src.prediction.sharp_weighted import (
    apply_weighted_combination_spread,
    apply_weighted_combination_total,
)
from src.ingestion import the_odds
from src.ingestion.standardize import (
    to_cst,
    CST,
    UTC,
    normalize_outcome_name,
    standardize_game_data,
)
from src.config import settings, filter_thresholds
import sys
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime, timedelta, timezone
import shutil
import logging
import random
import argparse
import asyncio
import os
from dotenv import load_dotenv
load_dotenv()


# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models" / "production"
ARCHIVE_DIR = PROJECT_ROOT / "archive"

# NOTE: CST and to_cst imported from src.ingestion.standardize (single source of truth)


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
    is_partial_market = "1H" in play_type
    now_cst = datetime.now(CST)

    # Calculate expected value (skip if no real odds available)
    if odds is not None:
        if odds > 0:
            profit = odds / 100
        else:
            profit = 100 / abs(odds)
        ev = (model_prob * profit) - (1 - model_prob)
        ev_pct = ev * 100
    else:
        ev_pct = None  # No fake EV calculation without real odds

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

    # PINNACLE VS SQUARE BOOK LINE COMPARISON (Most reliable sharp indicator)
    has_pinnacle = features.get("has_pinnacle_data", 0) > 0
    if has_pinnacle:
        if "SPREAD" in play_type:
            spread_diff = features.get("spread_sharp_square_diff", 0)
            is_home_pick = "home" in pick.lower()
            # Positive spread_diff = Pinnacle favors home more than square books
            if abs(spread_diff) >= 0.3:  # Meaningful difference
                pinnacle_favors = "home" if spread_diff > 0 else "away"
                if (is_home_pick and spread_diff > 0) or (not is_home_pick and spread_diff < 0):
                    sharp_action.append(
                        f"[PINNACLE] Sharp book (Pinnacle) shows {abs(spread_diff):.1f} pts difference vs square books, "
                        f"favoring {pick} — professional money positioned here."
                    )
        elif "TOTAL" in play_type:
            total_diff = features.get("total_sharp_square_diff", 0)
            is_over = "OVER" in pick.upper()
            # Positive total_diff = Pinnacle total higher than square books (sharps on over)
            if abs(total_diff) >= 0.5:  # Meaningful difference for totals
                if (is_over and total_diff > 0) or (not is_over and total_diff < 0):
                    sharp_action.append(
                        f"[PINNACLE] Sharp book (Pinnacle) total differs by {abs(total_diff):.1f} pts from square books, "
                        f"favoring {pick}."
                    )

    # ACTION NETWORK SPLITS (Ticket % vs Money % divergence)
    if betting_splits:
        try:
            # We use features built from splits in BuildRichFeatures
            if "SPREAD" in play_type:
                is_home_pick = "home" in pick.lower()

                # Use features directly since betting_splits might be a dict or object
                pick_tickets = features.get("spread_public_home_pct", 50) if is_home_pick else features.get(
                    "spread_public_away_pct", 50)
                pick_money = features.get("spread_money_home_pct", 50) if is_home_pick else features.get(
                    "spread_money_away_pct", 50)
                opp_tickets = 100 - pick_tickets

                rlm = features.get("is_rlm_spread", 0) > 0
                sharp_side = features.get("sharp_side_spread", 0)

                if rlm:
                    is_sharp_on_pick = (is_home_pick and sharp_side > 0) or (
                        not is_home_pick and sharp_side < 0)
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
                pick_tickets = features.get(
                    "over_public_pct", 50) if is_over else features.get("under_public_pct", 50)
                rlm = features.get("is_rlm_total", 0) > 0
                if rlm:
                    sharp_action.append(
                        f"[SHARP] Reverse line movement detected on the {pick}.")
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
            team_fundamentals.append(
                f"[STATS] {pick.title()} offensive efficiency: {pick_team_ppg:.1f} season PPG.")

        elif "TOTAL" in play_type:
            combined_ppg = home_ppg + away_ppg
            team_fundamentals.append(
                f"[STATS] Combined offensive output: {combined_ppg:.1f} PPG season average.")

        # ELO removed: we rely on live features only; avoid referencing legacy ratings

    # ============================================================
    # 3. SITUATIONAL FACTORS (Medium Priority)
    # ============================================================
    situational = []

    if not is_partial_market:
        rest_adj = features.get("rest_margin_adj", 0)
        if abs(rest_adj) >= 1.5:
            benefit_team = home_team if rest_adj > 0 else away_team
            situational.append(
                f"[REST] {benefit_team} has a situational rest advantage (+{abs(rest_adj):.1f} pts).")

        travel_fatigue = features.get("away_travel_fatigue", 0)
        if travel_fatigue >= 2.0:
            situational.append(
                f"[FATIGUE] Away team fatigue factor: high travel impact ({travel_fatigue:.1f} pts penalty).")

    # ============================================================
    # 5. MODEL CONFIDENCE (High Priority)
    # ============================================================
    model_confidence = [
        f"[MODEL] Model assigns {model_prob:.1%} probability to {pick} with {edge:+.1f} pt edge."
    ]

    if ev_pct is not None and abs(ev_pct) >= 5:
        model_confidence.append(
            f"[MODEL] Expected value: {ev_pct:+.1f}% based on market odds.")

    # ============================================================
    # ASSEMBLE RATIONALE
    # ============================================================
    rationale_bullets.extend(model_confidence[:1])
    if market_context:
        rationale_bullets.extend(market_context[:1])
    if sharp_action:
        rationale_bullets.extend(sharp_action[:1])

    # Fill to 3 bullets
    backups = team_fundamentals + situational
    for b in backups:
        if len(rationale_bullets) >= 3:
            break
        rationale_bullets.append(b)

    return " | ".join(rationale_bullets[:3])


def get_cst_now() -> datetime:
    """Get current time in CST."""
    return datetime.now(CST)


def format_cst_time(dt: datetime) -> str:
    """Format datetime as CST string."""
    cst_dt = to_cst(dt)
    return cst_dt.strftime("%a %b %d, %I:%M %p CST")


def get_target_date(date_str: str = None) -> datetime.date:
    """Get target date for predictions.
    
    IMPORTANT: Defaults to TODAY (CST) to avoid timezone confusion.
    The Odds API returns UTC times, so a game at 7pm CST on Jan 24
    appears as Jan 25 00:00 UTC. We always convert to CST first.
    """
    now_cst = get_cst_now()

    if date_str is None or date_str.lower() == "today":
        return now_cst.date()
    elif date_str.lower() == "tomorrow":
        return (now_cst + timedelta(days=1)).date()
    else:
        return datetime.strptime(date_str, "%Y-%m-%d").date()


def parse_utc_time(iso_string: str) -> datetime:
    """Parse ISO UTC time string to datetime."""
    if iso_string.endswith("Z"):
        iso_string = iso_string.replace("Z", "+00:00")
    return datetime.fromisoformat(iso_string)


def filter_games_for_date(games: list, target_date: datetime.date) -> list:
    """Filter games to only include those on the target date (in CST)."""
    filtered = []
    for game in games:
        # Prefer standardized CST date if present
        date_str = game.get("date")
        if date_str:
            try:
                # date_str may be a date-only string from standardize_game_data
                if isinstance(date_str, str) and len(date_str) == 10 and date_str[4] == "-" and date_str[7] == "-":
                    if datetime.strptime(date_str, "%Y-%m-%d").date() == target_date:
                        filtered.append(game)
                    continue
                # Otherwise treat as datetime-like and convert to CST
                cst_dt = to_cst(date_str)
                if cst_dt and cst_dt.date() == target_date:
                    filtered.append(game)
                continue
            except Exception:
                pass

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
                    markets="spreads_h1,totals_h1"
                )

                # MERGE instead of overwrite
                # Keep existing bookmakers (FG) and add new ones (1H)
                existing_bms = {
                    bm["key"]: bm for bm in game.get("bookmakers", [])}
                new_bms = event_odds.get("bookmakers", [])

                for nbm in new_bms:
                    if nbm["key"] in existing_bms:
                        # Add markets to existing bookmaker
                        existing_markets = {
                            m["key"]: m for m in existing_bms[nbm["key"]].get("markets", [])}
                        for nm in nbm.get("markets", []):
                            existing_markets[nm["key"]] = nm
                        existing_bms[nbm["key"]]["markets"] = list(
                            existing_markets.values())
                    else:
                        existing_bms[nbm["key"]] = nbm

                game["bookmakers"] = list(existing_bms.values())
            except Exception as e:
                print(
                    f"  [WARN] Could not fetch 1H odds for event {event_id}: {e}")
        enriched_games.append(game)

    # Enforce standardization + CST date normalization after enrichment
    standardized_games = []
    invalid = 0
    for game in enriched_games:
        standardized = standardize_game_data(game, source="the_odds")
        if standardized.get("_data_valid", False):
            standardized_games.append(standardized)
        else:
            invalid += 1

    if invalid:
        print(f"  [WARN] Skipped {invalid} invalid game(s) after standardization")

    return standardized_games


def extract_lines(game: dict, home_team: str):
    """
    Extract all available betting lines from game data.

    Returns:
        Dict with FG and 1H lines for spreads and totals
    """
    lines = {
        # Full game
        "fg_spread": None,
        "fg_total": None,
        # First half
        "fh_spread": None,
        "fh_total": None,
    }

    home_team_norm = normalize_outcome_name(home_team, source="the_odds")

    for bm in game.get("bookmakers", []):
        for market in bm.get("markets", []):
            market_key = market.get("key")

            # Full game spreads
            if market_key == "spreads":
                for outcome in market.get("outcomes", []):
                    outcome_name = normalize_outcome_name(
                        outcome.get("name"), source="the_odds"
                    )
                    if outcome_name == home_team_norm:
                        lines["fg_spread"] = outcome.get("point")
                        break

            # Full game totals
            elif market_key == "totals":
                for outcome in market.get("outcomes", []):
                    lines["fg_total"] = outcome.get("point")
                    break

            # First half spreads
            elif market_key == "spreads_h1":
                for outcome in market.get("outcomes", []):
                    outcome_name = normalize_outcome_name(
                        outcome.get("name"), source="the_odds"
                    )
                    if outcome_name == home_team_norm:
                        lines["fh_spread"] = outcome.get("point")
                        break

            # First half totals
            elif market_key == "totals_h1":
                for outcome in market.get("outcomes", []):
                    lines["fh_total"] = outcome.get("point")
                    break

    # Coerce to float where possible
    for key in list(lines.keys()):
        value = lines[key]
        if value is None:
            continue
        try:
            lines[key] = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"[extract_lines] Invalid line value for {key}: {value}"
            )

    # Enforce presence of all market lines (no silent fallbacks)
    missing = [k for k, v in lines.items() if v is None]
    if missing:
        market_keys = sorted({
            m.get("key")
            for bm in game.get("bookmakers", [])
            for m in bm.get("markets", [])
            if m.get("key") is not None
        })
        raise ValueError(
            f"[extract_lines] Missing required market lines {missing} for "
            f"{game.get('away_team')} @ {game.get('home_team')}. "
            f"Available markets: {market_keys}"
        )

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
    print(f"Markets: FG + 1H (Spreads, Totals)")

    # Fetch games
    games = await fetch_upcoming_games(target_date)
    if not games:
        print("\n[WARN] No upcoming games found")
        return

    print(f"\nProcessing {len(games)} games...")

    # Fetch betting splits if enabled
    betting_splits_dict = {}
    sharp_square_dict = {}  # Pinnacle vs Square book comparison

    if use_betting_splits:
        print("\nFetching public betting percentages (Action Network)...")
        try:
            betting_splits_dict = await fetch_public_betting_splits(games, source="auto")
            print(
                f"  [OK] Loaded betting splits for {len(betting_splits_dict)} games")
        except Exception as e:
            print(f"  [WARN] Failed to fetch betting splits: {e}")
            print(f"  [INFO] Continuing without betting splits data")

        # CRITICAL: Fetch sharp vs square book line comparison (Pinnacle vs DraftKings/FanDuel)
        print("\nFetching sharp vs square book lines (Pinnacle vs Square)...")
        try:
            sharp_square_data = await fetch_sharp_square_lines()
            for comp in sharp_square_data:
                game_key = f"{comp.away_team}@{comp.home_team}"
                sharp_square_dict[game_key] = comp
            print(
                f"  [OK] Loaded Pinnacle data for {len(sharp_square_dict)} games")
            for comp in sharp_square_data:
                spread_diff_fmt = f"{comp.spread_diff:+.2f}" if comp.spread_diff is not None else "NA"
                total_diff_fmt = f"{comp.total_diff:+.2f}" if comp.total_diff is not None else "NA"
                print(
                    f"    • {comp.away_team}@{comp.home_team}: spread_diff={spread_diff_fmt}, total_diff={total_diff_fmt}")
        except SharpDataUnavailableError as e:
            # HARD FAILURE - Pinnacle data is REQUIRED
            print(f"  [CRITICAL] Sharp book data unavailable: {e}")
            print(
                f"  [CRITICAL] Cannot generate predictions without Pinnacle data!")
            raise  # Propagate error - no degraded mode allowed

    # Initialize feature builder and unified prediction engine
    feature_builder = RichFeatureBuilder(
        league_id=12, season=settings.current_season)

    print("\nInitializing unified prediction engine...")
    engine = UnifiedPredictionEngine(models_dir=MODELS_DIR)
    print(f"  [OK] Loaded spread predictor")
    print(f"  [OK] Loaded total predictor")


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
            features = await feature_builder.build_game_features(
                home_team,
                away_team,
                betting_splits=splits,
            )

            # CRITICAL: Merge sharp/square book features (Pinnacle vs DraftKings/FanDuel)
            sharp_square_comp = sharp_square_dict.get(game_key)
            if sharp_square_comp:
                sharp_features = sharp_square_to_features(sharp_square_comp)
                features.update(sharp_features)
                spread_diff_fmt = (
                    f"{sharp_square_comp.spread_diff:+.2f}"
                    if sharp_square_comp.spread_diff is not None
                    else "NA"
                )
                total_diff_fmt = (
                    f"{sharp_square_comp.total_diff:+.2f}"
                    if sharp_square_comp.total_diff is not None
                    else "NA"
                )
                print(
                    f"  [SHARP] Pinnacle spread={sharp_square_comp.sharp_spread} vs Square avg={sharp_square_comp.square_spread} (diff={spread_diff_fmt})")
                print(
                    f"  [SHARP] Pinnacle total={sharp_square_comp.sharp_total} vs Square avg={sharp_square_comp.square_total} (diff={total_diff_fmt})")
            else:
                # This should not happen with HARD FAILURE mode, but log it
                print(
                    f"  [ERROR] No Pinnacle data for {game_key} - this should not happen!")

            # Extract all lines
            lines = extract_lines(game, home_team)

            # Generate predictions for ALL markets
            all_preds = engine.predict_all_markets(
                features,
                fg_spread_line=lines["fg_spread"],
                fg_total_line=lines["fg_total"],
                fh_spread_line=lines["fh_spread"],
                fh_total_line=lines["fh_total"],
            )

            fg_preds = all_preds["full_game"]
            fh_preds = all_preds["first_half"]

            # Display FG predictions
            print("\nFULL GAME:")
            display_market_predictions(fg_preds, lines, market_type="fg", features=features,
                                       home_team=home_team, away_team=away_team, splits=splits)

            # Display 1H predictions
            print("\nFIRST HALF:")
            display_market_predictions(fh_preds, lines, market_type="fh", features=features,
                                       home_team=home_team, away_team=away_team, splits=splits)

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
                "home_record": f"{features.get('home_wins', 0)}-{features.get('home_losses', 0)}",
                "away_record": f"{features.get('away_wins', 0)}-{features.get('away_losses', 0)}",
            }

            # Add FG predictions
            prediction_dict.update(format_predictions_for_csv(
                fg_preds, lines, prefix="fg"))

            # Add 1H predictions
            prediction_dict.update(format_predictions_for_csv(
                fh_preds, lines, prefix="fh"))

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


def _is_already_weighted(pred: dict) -> bool:
    """Return True if sharp-weighted fields are already present on prediction."""
    return pred.get("combined_edge") is not None and pred.get("sharp_edge") is not None


def _weighted_view(pred: dict, min_edge: float) -> dict:
    """Normalize weighted fields for display without reapplying weights."""
    combined_edge = pred.get("combined_edge")
    model_side = pred.get("model_side")
    model_edge = pred.get("model_edge")
    sharp_side = pred.get("sharp_side") or "neutral"
    sharp_edge = pred.get("sharp_edge")
    signals_agree = bool(pred.get("signals_agree", False))

    side_flipped = pred.get("side_flipped")
    if side_flipped is None:
        bet_side = pred.get("bet_side")
        side_flipped = bool(model_side and bet_side and bet_side != "NO_PLAY" and model_side != bet_side)

    is_play = False
    if combined_edge is not None:
        is_play = abs(combined_edge) >= min_edge

    rationale = pred.get("sharp_rationale", [])

    return {
        "combined_edge": combined_edge,
        "model_side": model_side,
        "model_edge": model_edge,
        "sharp_side": sharp_side,
        "sharp_edge": sharp_edge,
        "signals_agree": signals_agree,
        "side_flipped": side_flipped,
        "is_play": is_play,
        "rationale": rationale,
    }


def display_market_predictions(preds: dict, lines: dict, market_type: str, features: dict, home_team: str, away_team: str, splits: Any = None):
    """
    Display predictions for a market type (FG or 1H).

    WEIGHTED COMBINATION SYSTEM (v34.3.0):
    Sharp signals now affect THE PICK (not just confidence):
    - combined_edge = (model_edge × 0.55) + (sharp_edge × 0.45)
    - If signals agree → strong play
    - If signals conflict → combined math determines side (or NO PLAY)
    - NO more "bet against sharps with lower confidence"
    """
    if market_type == "fg":
        prefix = "fg"
    elif market_type == "fh":
        prefix = "fh"
    else:
        raise ValueError(f"Unsupported market type: {market_type}")

    # =========================================================================
    # SPREAD PREDICTION WITH WEIGHTED COMBINATION
    # =========================================================================
    if "spread" in preds:
        spread_pred = preds["spread"]
        original_side = spread_pred.get('bet_side', '')
        original_conf = spread_pred.get('confidence', 0.5)

        # APPLY WEIGHTED COMBINATION only if not already applied in engine
        if not _is_already_weighted(spread_pred):
            min_edge_override = (
                filter_thresholds.fh_spread_min_edge
                if market_type == "fh"
                else filter_thresholds.spread_min_edge
            )
            spread_pred, _ = apply_weighted_combination_spread(
                spread_pred,
                features,
                market_spread=lines.get(f'{prefix}_spread'),
                min_edge_override=min_edge_override,
            )
            preds["spread"] = spread_pred  # Update in place

        min_edge_display = (
            filter_thresholds.fh_spread_min_edge
            if market_type == "fh"
            else filter_thresholds.spread_min_edge
        )
        weighted_view = _weighted_view(spread_pred, min_edge_display)

        print(
            f"  [SPREAD] Predicted margin: {spread_pred['predicted_margin']:+.1f} (home)")
        print(f"           Vegas line: {lines[f'{prefix}_spread'] or 'N/A'}")

        # Show weighted combination result
        if weighted_view["is_play"]:
            side_indicator = ""
            if weighted_view["side_flipped"]:
                side_indicator = f" [FLIPPED from {weighted_view['model_side'].upper()}]"
            elif weighted_view["signals_agree"]:
                side_indicator = " [SIGNALS AGREE]"
            print(
                f"           Bet: {spread_pred['bet_side'].upper()} ({spread_pred['confidence']:.1%}){side_indicator}")
        else:
            print(f"           Bet: NO PLAY (signals cancel)")

        # Show combination breakdown
        print(f"           Model: {weighted_view['model_side'].upper()} {weighted_view['model_edge']:+.1f} pts | Sharp: {weighted_view['sharp_side'].upper()} {weighted_view['sharp_edge']:+.1f} pts")
        print(
            f"           Combined edge: {weighted_view['combined_edge']:+.2f} pts")

        # Show sharp rationale
        # Last 2 lines (most important)
        for r in weighted_view["rationale"][-2:]:
            print(f"           {r}")

        # Generate rationale
        if spread_pred.get('edge') is not None and weighted_view["is_play"]:
            rat = generate_rationale(
                play_type=f"{market_type.upper()}_SPREAD",
                pick=spread_pred['bet_side'],
                line=lines[f'{prefix}_spread'] or 0,
                odds=None,
                edge=spread_pred.get('edge', weighted_view["combined_edge"]),
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
    else:
        print(f"  [SPREAD] Model not loaded")

    # =========================================================================
    # TOTAL PREDICTION WITH WEIGHTED COMBINATION
    # =========================================================================
    if "total" in preds:
        total_pred = preds["total"]
        original_side = total_pred.get('bet_side', '')
        original_conf = total_pred.get('confidence', 0.5)

        # APPLY WEIGHTED COMBINATION only if not already applied in engine
        if not _is_already_weighted(total_pred):
            min_edge_override = (
                filter_thresholds.fh_total_min_edge
                if market_type == "fh"
                else filter_thresholds.total_min_edge
            )
            total_pred, _ = apply_weighted_combination_total(
                total_pred,
                features,
                market_total=lines.get(f'{prefix}_total'),
                min_edge_override=min_edge_override,
            )
            preds["total"] = total_pred  # Update in place

        min_edge_display = (
            filter_thresholds.fh_total_min_edge
            if market_type == "fh"
            else filter_thresholds.total_min_edge
        )
        weighted_view = _weighted_view(total_pred, min_edge_display)

        print(
            f"  [TOTAL] Predicted total: {total_pred['predicted_total']:.1f}")
        print(f"          Vegas line: {lines[f'{prefix}_total'] or 'N/A'}")

        # Show weighted combination result
        if weighted_view["is_play"]:
            side_indicator = ""
            if weighted_view["side_flipped"]:
                side_indicator = f" [FLIPPED from {weighted_view['model_side'].upper()}]"
            elif weighted_view["signals_agree"]:
                side_indicator = " [SIGNALS AGREE]"
            print(
                f"          Bet: {total_pred['bet_side'].upper()} ({total_pred['confidence']:.1%}){side_indicator}")
        else:
            print(f"          Bet: NO PLAY (signals cancel)")

        # Show combination breakdown
        print(f"          Model: {weighted_view['model_side'].upper()} {weighted_view['model_edge']:+.1f} pts | Sharp: {weighted_view['sharp_side'].upper()} {weighted_view['sharp_edge']:+.1f} pts")
        print(
            f"          Combined edge: {weighted_view['combined_edge']:+.2f} pts")

        # Show sharp rationale
        # Last 2 lines (most important)
        for r in weighted_view["rationale"][-2:]:
            print(f"          {r}")

        if total_pred.get('edge') is not None and weighted_view["is_play"]:
            rat = generate_rationale(
                play_type=f"{market_type.upper()}_TOTAL",
                pick=total_pred['bet_side'],
                line=lines[f'{prefix}_total'] or 0,
                odds=None,
                edge=total_pred.get('edge', weighted_view["combined_edge"]),
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
    else:
        print(f"  [TOTAL] Model not loaded")


def format_predictions_for_csv(preds: dict, lines: dict, prefix: str) -> dict:
    """Format predictions for CSV output (includes weighted combination info)."""
    spread_pred = preds.get("spread", {})
    total_pred = preds.get("total", {})

    return {
        # Spread
        f"{prefix}_spread_line": lines.get(f"{prefix}_spread"),
        f"{prefix}_spread_pred_margin": round(spread_pred.get('predicted_margin', 0), 1) if spread_pred else None,
        f"{prefix}_spread_model_edge": round(spread_pred.get('model_edge', 0), 2) if spread_pred else None,
        f"{prefix}_spread_sharp_edge": round(spread_pred.get('sharp_edge', 0), 2) if spread_pred else None,
        f"{prefix}_spread_combined_edge": round(spread_pred.get('combined_edge', 0), 2) if spread_pred else None,
        f"{prefix}_spread_bet_side": spread_pred.get('bet_side'),
        f"{prefix}_spread_model_side": spread_pred.get('model_side'),
        f"{prefix}_spread_sharp_side": spread_pred.get('sharp_side'),
        f"{prefix}_spread_confidence": round(spread_pred.get('confidence', 0), 3) if spread_pred else None,
        f"{prefix}_spread_signals_agree": spread_pred.get('signals_agree', False),
        f"{prefix}_spread_side_flipped": spread_pred.get('side_flipped', False),
        f"{prefix}_spread_passes_filter": spread_pred.get('passes_filter', False),
        f"{prefix}_spread_filter_reason": spread_pred.get('filter_reason', "") or "",
        f"{prefix}_spread_rationale": spread_pred.get('rationale', ""),
        # Total
        f"{prefix}_total_line": lines.get(f"{prefix}_total"),
        f"{prefix}_total_pred": round(total_pred.get('predicted_total', 0), 1) if total_pred else None,
        f"{prefix}_total_model_edge": round(total_pred.get('model_edge', 0), 2) if total_pred else None,
        f"{prefix}_total_sharp_edge": round(total_pred.get('sharp_edge', 0), 2) if total_pred else None,
        f"{prefix}_total_combined_edge": round(total_pred.get('combined_edge', 0), 2) if total_pred else None,
        f"{prefix}_total_bet_side": total_pred.get('bet_side'),
        f"{prefix}_total_model_side": total_pred.get('model_side'),
        f"{prefix}_total_sharp_side": total_pred.get('sharp_side'),
        f"{prefix}_total_confidence": round(total_pred.get('confidence', 0), 3) if total_pred else None,
        f"{prefix}_total_signals_agree": total_pred.get('signals_agree', False),
        f"{prefix}_total_side_flipped": total_pred.get('side_flipped', False),
        f"{prefix}_total_passes_filter": total_pred.get('passes_filter', False),
        f"{prefix}_total_filter_reason": total_pred.get('filter_reason', "") or "",
        f"{prefix}_total_rationale": total_pred.get('rationale', ""),
    }


def generate_executive_html(all_plays: list, target_date: datetime.date) -> str:
    """
    Generate EXECUTIVE SUMMARY HTML - clean, scannable betting card.
    
    This is the "BLUF" (Bottom Line Up Front) view for quick decision making.
    Shows: Time | Matchup | Pick | Edge | Fire Rating
    """
    date_str = target_date.strftime("%A, %B %d, %Y")
    generated_at = datetime.now(CST).strftime("%Y-%m-%d %I:%M %p CST")
    
    # Sort by fire rating (desc), then by edge (desc)
    def sort_key(p):
        fire = p.get('fire_rating', 0)
        edge = abs(p.get('edge', 0)) if p.get('edge') else 0
        return (-fire, -edge)
    
    sorted_plays = sorted(all_plays, key=sort_key)
    
    # Build rows
    rows_html = ""
    for p in sorted_plays:
        fire = p.get('fire_rating', 1)
        fire_class = "elite" if fire >= 4 else "strong" if fire == 3 else "good" if fire == 2 else ""
        fire_label = "🔥" * fire
        
        edge = p.get('edge', 0)
        edge_str = f"{edge:+.1f}" if edge else "—"
        edge_class = "positive" if edge and edge > 0 else "negative" if edge and edge < 0 else ""
        
        conf = p.get('confidence', 0)
        conf_str = f"{conf:.0%}" if conf else "—"
        
        rows_html += f"""
        <tr class="{fire_class}">
            <td>{p.get('time_cst', p.get('date', ''))}</td>
            <td class="matchup">{p.get('matchup', '')}</td>
            <td class="period">{p.get('period', '')}</td>
            <td class="pick"><strong>{p.get('pick', '')}</strong></td>
            <td class="{edge_class}">{edge_str}</td>
            <td>{conf_str}</td>
            <td class="fire">{fire_label}</td>
        </tr>"""
    
    # Count by tier
    elite_count = sum(1 for p in sorted_plays if p.get('fire_rating', 0) >= 4)
    strong_count = sum(1 for p in sorted_plays if p.get('fire_rating', 0) == 3)
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NBA Executive Summary - {date_str}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ 
            font-family: 'Segoe UI', -apple-system, sans-serif; 
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e8e8e8; 
            padding: 20px;
            min-height: 100vh;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        
        .header {{ 
            background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
            border-radius: 16px; 
            padding: 24px 32px; 
            margin-bottom: 24px;
            border: 1px solid #0f3460;
        }}
        .header h1 {{ 
            font-size: 28px; 
            font-weight: 700; 
            margin-bottom: 8px;
            background: linear-gradient(90deg, #00d4ff, #7c3aed);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .header .subtitle {{ color: #94a3b8; font-size: 14px; }}
        
        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .summary-card {{
            background: #1e293b;
            border-radius: 12px;
            padding: 16px;
            text-align: center;
            border: 1px solid #334155;
        }}
        .summary-card .number {{ font-size: 32px; font-weight: 700; color: #00d4ff; }}
        .summary-card .label {{ font-size: 12px; color: #94a3b8; text-transform: uppercase; }}
        .summary-card.elite .number {{ color: #f59e0b; }}
        .summary-card.strong .number {{ color: #10b981; }}
        
        table {{ 
            width: 100%; 
            border-collapse: collapse; 
            background: #1e293b;
            border-radius: 12px;
            overflow: hidden;
        }}
        th {{ 
            background: #0f172a; 
            padding: 14px 12px; 
            text-align: left; 
            font-weight: 600;
            color: #94a3b8;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        td {{ 
            padding: 14px 12px; 
            border-bottom: 1px solid #334155;
            font-size: 14px;
        }}
        tr:hover {{ background: #334155; }}
        tr.elite {{ background: rgba(245, 158, 11, 0.1); }}
        tr.strong {{ background: rgba(16, 185, 129, 0.1); }}
        
        .matchup {{ font-weight: 500; }}
        .period {{ 
            color: #94a3b8; 
            font-size: 12px;
            background: #334155;
            padding: 4px 8px;
            border-radius: 4px;
            display: inline-block;
        }}
        .pick {{ color: #00d4ff; }}
        .positive {{ color: #10b981; }}
        .negative {{ color: #ef4444; }}
        .fire {{ font-size: 16px; }}
        
        .footer {{ 
            text-align: center; 
            padding: 24px; 
            color: #64748b; 
            font-size: 12px; 
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏀 NBA EXECUTIVE SUMMARY</h1>
            <div class="subtitle">{date_str} • Generated {generated_at}</div>
        </div>
        
        <div class="summary-cards">
            <div class="summary-card">
                <div class="number">{len(sorted_plays)}</div>
                <div class="label">Total Plays</div>
            </div>
            <div class="summary-card elite">
                <div class="number">{elite_count}</div>
                <div class="label">Elite (4-5🔥)</div>
            </div>
            <div class="summary-card strong">
                <div class="number">{strong_count}</div>
                <div class="label">Strong (3🔥)</div>
            </div>
        </div>
        
        <table>
            <thead>
                <tr>
                    <th>Time</th>
                    <th>Matchup</th>
                    <th>Period</th>
                    <th>Pick</th>
                    <th>Edge</th>
                    <th>Conf</th>
                    <th>Rating</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
        
        <div class="footer">
            Green Bier Sports Ventures • NBA Model v34.3.0
        </div>
    </div>
</body>
</html>"""
    return html


def generate_detailed_html(all_plays: list, target_date: datetime.date) -> str:
    """
    Generate DETAILED RATIONALE HTML - full analysis for each pick.
    
    Shows complete rationale, sharp signals, model breakdown for each pick.
    """
    date_str = target_date.strftime("%A, %B %d, %Y")
    generated_at = datetime.now(CST).strftime("%Y-%m-%d %I:%M %p CST")
    
    # Sort by fire rating (desc), then by edge (desc)
    def sort_key(p):
        fire = p.get('fire_rating', 0)
        edge = abs(p.get('edge', 0)) if p.get('edge') else 0
        return (-fire, -edge)
    
    sorted_plays = sorted(all_plays, key=sort_key)
    
    # Build cards
    cards_html = ""
    for p in sorted_plays:
        fire = p.get('fire_rating', 1)
        fire_label = "🔥" * fire
        tier = "ELITE" if fire >= 4 else "STRONG" if fire == 3 else "GOOD" if fire == 2 else "STANDARD"
        tier_class = tier.lower()
        
        edge = p.get('edge', 0)
        edge_str = f"{edge:+.1f} pts" if edge else "—"
        
        conf = p.get('confidence', 0)
        conf_str = f"{conf:.0%}" if conf else "—"
        
        rationale = p.get('rationale', 'No rationale available')
        # Parse rationale into bullets
        rationale_parts = rationale.split(' | ') if rationale else []
        rationale_html = ""
        for part in rationale_parts:
            if part.startswith('[MODEL]'):
                icon = "🤖"
                part = part.replace('[MODEL]', '').strip()
            elif part.startswith('[SHARP]'):
                icon = "💰"
                part = part.replace('[SHARP]', '').strip()
            elif part.startswith('[PINNACLE]'):
                icon = "📊"
                part = part.replace('[PINNACLE]', '').strip()
            elif part.startswith('[STATS]'):
                icon = "📈"
                part = part.replace('[STATS]', '').strip()
            elif part.startswith('[REST]'):
                icon = "😴"
                part = part.replace('[REST]', '').strip()
            elif part.startswith('[LINE]'):
                icon = "📉"
                part = part.replace('[LINE]', '').strip()
            elif part.startswith('[ELO]'):
                icon = "🏆"
                part = part.replace('[ELO]', '').strip()
            else:
                icon = "•"
            rationale_html += f'<li><span class="icon">{icon}</span> {part}</li>'
        
        cards_html += f"""
        <div class="pick-card {tier_class}">
            <div class="card-header">
                <div class="tier-badge {tier_class}">{tier} {fire_label}</div>
                <div class="time">{p.get('time_cst', p.get('date', ''))}</div>
            </div>
            <div class="matchup">{p.get('matchup', '')}</div>
            <div class="pick-line">
                <span class="period-badge">{p.get('period', '')}</span>
                <span class="market-badge">{p.get('market', '')}</span>
                <span class="pick-value">{p.get('pick', '')}</span>
            </div>
            <div class="metrics">
                <div class="metric">
                    <div class="metric-value">{edge_str}</div>
                    <div class="metric-label">Edge</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{conf_str}</div>
                    <div class="metric-label">Confidence</div>
                </div>
            </div>
            <div class="rationale">
                <div class="rationale-title">Analysis</div>
                <ul>{rationale_html}</ul>
            </div>
        </div>"""
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NBA Detailed Analysis - {date_str}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ 
            font-family: 'Segoe UI', -apple-system, sans-serif; 
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e8e8e8; 
            padding: 20px;
            min-height: 100vh;
        }}
        .container {{ max-width: 900px; margin: 0 auto; }}
        
        .header {{ 
            background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
            border-radius: 16px; 
            padding: 24px 32px; 
            margin-bottom: 24px;
            border: 1px solid #0f3460;
            text-align: center;
        }}
        .header h1 {{ 
            font-size: 28px; 
            font-weight: 700; 
            margin-bottom: 8px;
            background: linear-gradient(90deg, #00d4ff, #7c3aed);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .header .subtitle {{ color: #94a3b8; font-size: 14px; }}
        
        .pick-card {{
            background: #1e293b;
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
            border-left: 4px solid #334155;
        }}
        .pick-card.elite {{ border-left-color: #f59e0b; background: linear-gradient(90deg, rgba(245,158,11,0.1) 0%, #1e293b 100%); }}
        .pick-card.strong {{ border-left-color: #10b981; background: linear-gradient(90deg, rgba(16,185,129,0.1) 0%, #1e293b 100%); }}
        .pick-card.good {{ border-left-color: #3b82f6; }}
        
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }}
        .tier-badge {{
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            background: #334155;
        }}
        .tier-badge.elite {{ background: #f59e0b; color: #000; }}
        .tier-badge.strong {{ background: #10b981; color: #000; }}
        .time {{ color: #64748b; font-size: 13px; }}
        
        .matchup {{ 
            font-size: 20px; 
            font-weight: 600; 
            margin-bottom: 12px;
        }}
        
        .pick-line {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 16px;
        }}
        .period-badge, .market-badge {{
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 600;
            background: #334155;
            color: #94a3b8;
        }}
        .pick-value {{
            font-size: 18px;
            font-weight: 700;
            color: #00d4ff;
        }}
        
        .metrics {{
            display: flex;
            gap: 24px;
            margin-bottom: 16px;
            padding: 12px;
            background: #0f172a;
            border-radius: 8px;
        }}
        .metric {{ text-align: center; }}
        .metric-value {{ font-size: 20px; font-weight: 700; color: #10b981; }}
        .metric-label {{ font-size: 11px; color: #64748b; text-transform: uppercase; }}
        
        .rationale {{
            background: #0f172a;
            border-radius: 8px;
            padding: 16px;
        }}
        .rationale-title {{
            font-size: 12px;
            font-weight: 600;
            color: #64748b;
            text-transform: uppercase;
            margin-bottom: 12px;
        }}
        .rationale ul {{ list-style: none; }}
        .rationale li {{
            padding: 8px 0;
            border-bottom: 1px solid #1e293b;
            display: flex;
            align-items: flex-start;
            gap: 10px;
            font-size: 14px;
            line-height: 1.5;
        }}
        .rationale li:last-child {{ border-bottom: none; }}
        .rationale .icon {{ font-size: 16px; }}
        
        .footer {{ 
            text-align: center; 
            padding: 24px; 
            color: #64748b; 
            font-size: 12px; 
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏀 NBA DETAILED ANALYSIS</h1>
            <div class="subtitle">{date_str} • Generated {generated_at} • {len(sorted_plays)} Picks</div>
        </div>
        
        {cards_html}
        
        <div class="footer">
            Green Bier Sports Ventures • NBA Model v34.3.0 • Full Rationale Report
        </div>
    </div>
</body>
</html>"""
    return html


def generate_formatted_text_report(df: pd.DataFrame, target_date: datetime.date) -> str:
    """Generate formatted text report in table format as requested."""
    lines = []

    # Header
    date_str = target_date.strftime("%A, %B %d, %Y")
    lines.append("=" * 120)
    lines.append(f"NBA BETTING CARD - {date_str}")
    lines.append("=" * 120)
    lines.append("")

    # Table Header
    # DATE | MATCHUP | TYPE | PICK | MODEL | VEGAS | EDGE | FIRE
    header = f"{'DATE/TIME (CST)':<20} | {'MATCHUP':<30} | {'TYPE':<4} | {'PICK':<25} | {'MODEL':<8} | {'VEGAS':<8} | {'EDGE':<6} | {'FIRE':<5}"
    lines.append(header)
    lines.append("-" * len(header))

    def calculate_fire_rating(confidence: float, edge: float) -> int:
        """Calculate fire rating (1-5) based on confidence and edge."""
        # Normalize edge (pts) to 0-1 scale (10 pts = max)
        edge_norm = min(abs(edge) / 10.0, 1.0)
        # Combine confidence and edge
        combined_score = (confidence * 0.6) + (edge_norm * 0.4)

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

    # Process each play
    for _, row in df.iterrows():
        # Construct Matchup with Records
        home_rec = row.get('home_record', '')
        away_rec = row.get('away_record', '')
        # Shorten matchup string to fit
        matchup = f"{row['away_team']} @ {row['home_team']}"

        date_time = row['date'].replace(" CST", "")  # Shorten slightly

        # Helper to add row
        def add_row(market_type, pick, model_val, market_val, edge, conf, passes):
            if not passes:
                return

            fire = calculate_fire_rating(conf, edge)
            fire_str = "🔥" * fire

            # Determine Type (FG or 1H)
            if "FG" in market_type:
                type_str = "FG"
            elif "1H" in market_type:
                type_str = "1H"
            else:
                type_str = "??"

            # Format Pick
            # market_val is the line (usually home line for spread)
            if "Spread" in market_type:
                if pick.lower() == "home":
                    team_name = row['home_team']
                    line_val = market_val
                else:  # away
                    team_name = row['away_team']
                    line_val = -market_val

                # Format line with + sign if positive
                line_str = f"{line_val:+.1f}"
                pick_str = f"{team_name} {line_str}"

                # Handle near-zero margin display (avoid "-0.0")
                if abs(model_val) < 0.05:
                    model_str = "0.0"
                else:
                    model_str = f"{model_val:+.1f}"
                market_str = f"{market_val:+.1f}"

            else:  # Total
                pick_str = f"{pick.upper()} {market_val:.1f}"
                model_str = f"{model_val:.1f}"
                market_str = f"{market_val:.1f}"

            # Add odds (assuming -110 flat for now as we don't predict exact vig)
            odds = -110
            pick_str += f" ({odds})"

            edge_str = f"{edge:+.1f}"

            line = f"{date_time:<20} | {matchup:<30} | {type_str:<4} | {pick_str:<25} | {model_str:<8} | {market_str:<8} | {edge_str:<6} | {fire_str:<5}"
            lines.append(line)

        # FG Spread
        if pd.notna(row.get('fg_spread_line')):
            add_row(
                "FG Spread",
                row.get('fg_spread_bet_side'),
                row.get('fg_spread_pred_margin'),
                row.get('fg_spread_line'),
                row.get('fg_spread_combined_edge', np.nan),
                row.get('fg_spread_confidence', np.nan),
                row.get('fg_spread_passes_filter', False),
            )

        # FG Total
        if pd.notna(row.get('fg_total_line')):
            add_row(
                "FG Total",
                row.get('fg_total_bet_side'),
                row.get('fg_total_pred'),
                row.get('fg_total_line'),
                row.get('fg_total_combined_edge', np.nan),
                row.get('fg_total_confidence', np.nan),
                row.get('fg_total_passes_filter', False),
            )

        # 1H Spread
        if pd.notna(row.get('fh_spread_line')):
            add_row(
                "1H Spread",
                row.get('fh_spread_bet_side'),
                row.get('fh_spread_pred_margin'),
                row.get('fh_spread_line'),
                row.get('fh_spread_combined_edge', np.nan),
                row.get('fh_spread_confidence', np.nan),
                row.get('fh_spread_passes_filter', False),
            )

        # 1H Total
        if pd.notna(row.get('fh_total_line')):
            add_row(
                "1H Total",
                row.get('fh_total_bet_side'),
                row.get('fh_total_pred'),
                row.get('fh_total_line'),
                row.get('fh_total_combined_edge', np.nan),
                row.get('fh_total_confidence', np.nan),
                row.get('fh_total_passes_filter', False),
            )

    lines.append("")
    lines.append("=" * 120)
    lines.append("END OF CARD")
    lines.append("=" * 120)

    return "\n".join(lines)


def save_predictions(predictions: list, target_date: Optional[datetime.date] = None):
    """Save predictions and generate betting card."""
    output_path = DATA_DIR / "processed" / "predictions_v3.csv"
    df = pd.DataFrame(predictions)
    df.to_csv(output_path, index=False)
    print(f"\n{'='*80}")
    print(f"[OK] Saved {len(predictions)} predictions to {output_path}")
    print(f"{'='*80}")

    def archive_copy(src: Path, dest_dir: Path, filename: str) -> None:
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest_dir / filename)

    archive_ts = datetime.now(CST).strftime("%Y%m%d_%H%M%S")
    date_tag = target_date.strftime(
        "%Y%m%d") if target_date else datetime.now(CST).strftime("%Y%m%d")

    try:
        archive_copy(
            output_path,
            ARCHIVE_DIR / "predictions",
            f"predictions_{date_tag}_{archive_ts}.csv",
        )
    except Exception as e:
        print(f"[WARN] Failed to archive predictions: {e}")

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
                "edge": row.get('fg_spread_combined_edge'),
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
                "edge": row.get('fg_total_combined_edge'),
                "rationale": row.get('fg_total_rationale', ""),
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
                "edge": row.get('fh_spread_combined_edge'),
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
                "edge": row.get('fh_total_combined_edge'),
                "rationale": row.get('fh_total_rationale', ""),
            })

    # Generate formatted text report
    text_report = generate_formatted_text_report(df, target_date)

    # Print to console
    print(text_report)

    # Save report to file
    report_path = DATA_DIR / "processed" / \
        f"slate_analysis_{target_date.strftime('%Y%m%d') if target_date else 'latest'}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(text_report)
    print(f"[OK] Saved formatted text report to {report_path}")
    try:
        archive_copy(
            report_path,
            ARCHIVE_DIR / "analysis",
            f"slate_analysis_{date_tag}_{archive_ts}.txt",
        )
    except Exception as e:
        print(f"[WARN] Failed to archive slate analysis: {e}")

    if all_plays:
        # Calculate fire rating for each play
        def calculate_fire_rating(confidence: float, edge: float) -> int:
            edge_norm = min(abs(edge) / 10.0, 1.0) if edge else 0
            combined_score = (confidence * 0.6) + (edge_norm * 0.4) if confidence else 0
            if combined_score >= 0.85:
                return 5
            elif combined_score >= 0.70:
                return 4
            elif combined_score >= 0.60:
                return 3
            elif combined_score >= 0.52:
                return 2
            return 1
        
        for play in all_plays:
            play['fire_rating'] = calculate_fire_rating(
                play.get('confidence', 0), 
                play.get('edge', 0)
            )
            play['time_cst'] = play.get('date', '')  # For HTML compatibility
        
        # Save betting card CSV
        betting_card_df = pd.DataFrame(all_plays)
        betting_card_path = DATA_DIR / "processed" / "betting_card_v3.csv"
        betting_card_df.to_csv(betting_card_path, index=False)
        print(f"[OK] Saved betting card to {betting_card_path}")
        
        # Generate and save EXECUTIVE SUMMARY HTML
        executive_html = generate_executive_html(all_plays, target_date)
        executive_path = DATA_DIR / "processed" / "executive_summary.html"
        with open(executive_path, "w", encoding="utf-8") as f:
            f.write(executive_html)
        print(f"[OK] Saved executive summary HTML to {executive_path}")
        
        # Generate and save DETAILED RATIONALE HTML
        detailed_html = generate_detailed_html(all_plays, target_date)
        detailed_path = DATA_DIR / "processed" / "detailed_rationale.html"
        with open(detailed_path, "w", encoding="utf-8") as f:
            f.write(detailed_html)
        print(f"[OK] Saved detailed rationale HTML to {detailed_path}")
        
        try:
            archive_copy(
                betting_card_path,
                ARCHIVE_DIR / "picks",
                f"betting_card_{date_tag}_{archive_ts}.csv",
            )
            archive_copy(
                executive_path,
                ARCHIVE_DIR / "picks",
                f"executive_summary_{date_tag}_{archive_ts}.html",
            )
            archive_copy(
                detailed_path,
                ARCHIVE_DIR / "picks",
                f"detailed_rationale_{date_tag}_{archive_ts}.html",
            )
        except Exception as e:
            print(f"[WARN] Failed to archive betting card/HTML: {e}")
    else:
        print("\nNO PLAYS TODAY")
        print("All games filtered out - no bets meet criteria")
        print("=" * 80)

        # Still generate text report even if no plays
        if target_date and len(df) > 0:
            text_report = generate_formatted_text_report(df, target_date)
            text_path = DATA_DIR / "processed" / \
                f"slate_analysis_{target_date.strftime('%Y%m%d')}.txt"
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(text_report)
            print(f"[OK] Saved formatted text report to {text_path}")
            try:
                archive_copy(
                    text_path,
                    ARCHIVE_DIR / "analysis",
                    f"slate_analysis_{date_tag}_{archive_ts}.txt",
                )
            except Exception as e:
                print(f"[WARN] Failed to archive slate analysis: {e}")


def main():
    # Set random seeds
    np.random.seed(42)
    random.seed(42)
    logger.info("Random seeds set to 42 for reproducibility")

    print("\nWARNING: Running in LOCAL mode. For Docker execution, use 'python scripts/predict_unified_slate.py' or './run.ps1'")

    parser = argparse.ArgumentParser(
        description="Generate NBA predictions for all markets")
    parser.add_argument(
        "--date", default="today",
        help="Date for predictions: 'today' (default), 'tomorrow', or YYYY-MM-DD")
    parser.add_argument("--no-betting-splits", action="store_true",
                        help="Disable betting splits integration")
    args = parser.parse_args()

    asyncio.run(predict_games_async(
        args.date, use_betting_splits=not args.no_betting_splits))


if __name__ == "__main__":
    main()
