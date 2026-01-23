"""
Generate predictions for ALL markets using unified prediction engine.

Production-ready predictor with smart filtering for all markets:
- Full Game: Spreads, Totals
- First Half: Spreads, Totals
"""
from src.prediction import UnifiedPredictionEngine
from src.features.rich_features import RichFeatureBuilder
from src.ingestion.betting_splits import fetch_public_betting_splits
from src.ingestion import the_odds
from src.ingestion.standardize import to_cst, CST, UTC
from src.config import settings
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

        home_elo = features.get("home_elo", 1500)
        away_elo = features.get("away_elo", 1500)
        elo_diff = abs(home_elo - away_elo)
        if elo_diff >= 50:
            stronger = home_team if home_elo > away_elo else away_team
            team_fundamentals.append(
                f"[ELO] {stronger} holds significant ELO advantage ({elo_diff:.0f} pts).")

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

    return enriched_games


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
    if use_betting_splits:
        print("\nFetching public betting percentages...")
        try:
            betting_splits_dict = await fetch_public_betting_splits(games, source="auto")
            print(
                f"  [OK] Loaded betting splits for {len(betting_splits_dict)} games")
        except Exception as e:
            print(f"  [WARN] Failed to fetch betting splits: {e}")
            print(f"  [INFO] Continuing without betting splits data")

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
            features = await feature_builder.build_game_features(home_team, away_team, betting_splits=splits)

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


def display_market_predictions(preds: dict, lines: dict, market_type: str, features: dict, home_team: str, away_team: str, splits: Any = None):
    """Display predictions for a market type (FG or 1H)."""
    if market_type == "fg":
        prefix = "fg"
    elif market_type == "fh":
        prefix = "fh"
    else:
        raise ValueError(f"Unsupported market type: {market_type}")

    # Spread
    if "spread" in preds:
        spread_pred = preds["spread"]
        print(
            f"  [SPREAD] Predicted margin: {spread_pred['predicted_margin']:+.1f} (home)")
        print(f"           Vegas line: {lines[f'{prefix}_spread'] or 'N/A'}")
        print(
            f"           Bet: {spread_pred['bet_side']} ({spread_pred['confidence']:.1%})")
        if spread_pred['edge'] is not None:
            print(f"           Edge: {spread_pred['edge']:+.1f} pts")

        # Generate rationale
        if spread_pred['edge'] is not None:
            rat = generate_rationale(
                play_type=f"{market_type.upper()}_SPREAD",
                pick=spread_pred['bet_side'],
                line=lines[f'{prefix}_spread'] or 0,
                odds=None,  # No fake odds - rationale will handle missing odds
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
    else:
        print(f"  [SPREAD] Model not loaded")

    # Total
    if "total" in preds:
        total_pred = preds["total"]
        print(
            f"  [TOTAL] Predicted total: {total_pred['predicted_total']:.1f}")
        print(f"          Vegas line: {lines[f'{prefix}_total'] or 'N/A'}")
        print(
            f"          Bet: {total_pred['bet_side']} ({total_pred['confidence']:.1%})")
        if total_pred['edge'] is not None:
            print(f"          Edge: {total_pred['edge']:+.1f} pts")

        if total_pred['edge'] is not None:
            rat = generate_rationale(
                play_type=f"{market_type.upper()}_TOTAL",
                pick=total_pred['bet_side'],
                line=lines[f'{prefix}_total'] or 0,
                odds=None,  # No fake odds - rationale will handle missing odds
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
    else:
        print(f"  [TOTAL] Model not loaded")


def format_predictions_for_csv(preds: dict, lines: dict, prefix: str) -> dict:
    """Format predictions for CSV output."""
    spread_pred = preds.get("spread", {})
    total_pred = preds.get("total", {})

    return {
        # Spread
        f"{prefix}_spread_line": lines.get(f"{prefix}_spread"),
        f"{prefix}_spread_pred_margin": round(spread_pred.get('predicted_margin', 0), 1) if spread_pred else None,
        f"{prefix}_spread_edge": round(spread_pred.get('edge', 0), 1) if spread_pred and spread_pred.get('edge') else None,
        f"{prefix}_spread_bet_side": spread_pred.get('bet_side'),
        f"{prefix}_spread_confidence": round(spread_pred.get('confidence', 0), 3) if spread_pred else None,
        f"{prefix}_spread_passes_filter": spread_pred.get('passes_filter', False),
        f"{prefix}_spread_filter_reason": spread_pred.get('filter_reason', "") or "",
        f"{prefix}_spread_rationale": spread_pred.get('rationale', ""),
        # Total
        f"{prefix}_total_line": lines.get(f"{prefix}_total"),
        f"{prefix}_total_pred": round(total_pred.get('predicted_total', 0), 1) if total_pred else None,
        f"{prefix}_total_edge": round(total_pred.get('edge', 0), 1) if total_pred and total_pred.get('edge') else None,
        f"{prefix}_total_bet_side": total_pred.get('bet_side'),
        f"{prefix}_total_confidence": round(total_pred.get('confidence', 0), 3) if total_pred else None,
        f"{prefix}_total_passes_filter": total_pred.get('passes_filter', False),
        f"{prefix}_total_filter_reason": total_pred.get('filter_reason', "") or "",
        f"{prefix}_total_rationale": total_pred.get('rationale', ""),
    }


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

            # Add odds
            pick_str += f" ({odds})"

            edge_str = f"{edge:+.1f}"

            line = f"{date_time:<20} | {matchup:<30} | {type_str:<4} | {pick_str:<25} | {model_str:<8} | {market_str:<8} | {edge_str:<6} | {fire_str:<5}"
            lines.append(line)

        # FG Spread
        if pd.notna(row.get('fg_spread_line')):
            add_row("FG Spread", row['fg_spread_bet_side'], row['fg_spread_pred_margin'], row['fg_spread_line'],
                    row['fg_spread_edge'], row['fg_spread_confidence'], row['fg_spread_passes_filter'])

        # FG Total
        if pd.notna(row.get('fg_total_line')):
            add_row("FG Total", row['fg_total_bet_side'], row['fg_total_pred'], row['fg_total_line'],
                    row['fg_total_edge'], row['fg_total_confidence'], row['fg_total_passes_filter'])

        # 1H Spread
        if pd.notna(row.get('fh_spread_line')):
            add_row("1H Spread", row['fh_spread_bet_side'], row['fh_spread_pred_margin'], row['fh_spread_line'],
                    row['fh_spread_edge'], row['fh_spread_confidence'], row['fh_spread_passes_filter'])

        # 1H Total
        if pd.notna(row.get('fh_total_line')):
            add_row("1H Total", row['fh_total_bet_side'], row['fh_total_pred'], row['fh_total_line'],
                    row['fh_total_edge'], row['fh_total_confidence'], row['fh_total_passes_filter'])

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
        # Save betting card CSV
        betting_card_df = pd.DataFrame(all_plays)
        betting_card_path = DATA_DIR / "processed" / "betting_card_v3.csv"
        betting_card_df.to_csv(betting_card_path, index=False)
        print(f"[OK] Saved betting card to {betting_card_path}")
        try:
            archive_copy(
                betting_card_path,
                ARCHIVE_DIR / "picks",
                f"betting_card_{date_tag}_{archive_ts}.csv",
            )
        except Exception as e:
            print(f"[WARN] Failed to archive betting card: {e}")
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
        "--date", help="Date for predictions (YYYY-MM-DD, 'today', or 'tomorrow')")
    parser.add_argument("--no-betting-splits", action="store_true",
                        help="Disable betting splits integration")
    args = parser.parse_args()

    asyncio.run(predict_games_async(
        args.date, use_betting_splits=not args.no_betting_splits))


if __name__ == "__main__":
    main()
