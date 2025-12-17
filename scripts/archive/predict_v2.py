"""
Generate predictions using modular prediction engine.

Production-ready predictor with smart filtering for spreads and totals.
"""
import asyncio
import argparse
import random
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
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
from src.prediction import PredictionEngine, SpreadFilter, TotalFilter

# Central Standard Time
CST = ZoneInfo("America/Chicago")


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

    odds_data = await the_odds.fetch_odds()
    print(f"  [OK] The Odds: {len(odds_data)} total games from API")

    odds_data = filter_games_for_date(odds_data, target_date)
    date_str = target_date.strftime("%A, %B %d, %Y")
    print(f"  [OK] Filtered to {len(odds_data)} games for {date_str}")

    return odds_data


async def predict_games_async(date: str = None, use_betting_splits: bool = True):
    """Generate predictions for upcoming games using modular prediction engine."""
    target_date = get_target_date(date)
    now_cst = get_cst_now()

    print("=" * 80)
    print("NBA PREDICTIONS - MODULAR ENGINE (SPREADS + TOTALS)")
    print("=" * 80)
    print(f"Current time: {now_cst.strftime('%A, %B %d, %Y at %I:%M %p CST')}")
    print(f"Target date: {target_date.strftime('%A, %B %d, %Y')}")

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

    # Initialize feature builder and prediction engine
    feature_builder = RichFeatureBuilder(league_id=12, season=settings.current_season)

    print("\nInitializing prediction engine...")
    engine = PredictionEngine(
        models_dir=MODELS_DIR,
        spread_filter=SpreadFilter(),  # Smart filtering for spreads
        total_filter=TotalFilter(use_filter=False),  # No filter for totals (baseline is best)
    )
    print(f"  [OK] Loaded spreads model: {len(engine.spread_features)} features")
    print(f"  [OK] Loaded totals model: {len(engine.total_features)} features")

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

        print(f"\n[{i}/{len(games)}] {away_team} @ {home_team}")
        print(f"  Game time: {time_str}")

        try:
            # Get betting splits if available
            game_key = f"{away_team}@{home_team}"
            splits = betting_splits_dict.get(game_key)

            # Build features
            features = await feature_builder.build_game_features(home_team, away_team, betting_splits=splits)

            # Extract lines from odds
            spread_line = None
            total_line = None
            for bm in game.get("bookmakers", []):
                for market in bm.get("markets", []):
                    if market.get("key") == "spreads":
                        for outcome in market.get("outcomes", []):
                            if outcome.get("name") == home_team:
                                spread_line = outcome.get("point")
                                break
                    elif market.get("key") == "totals":
                        for outcome in market.get("outcomes", []):
                            total_line = outcome.get("point")
                            break

            # Generate predictions using modular engine
            preds = engine.predict_game(features, spread_line, total_line)
            spread_pred = preds["spread"]
            total_pred = preds["total"]

            # Display spread prediction
            print(f"  [SPREAD] Predicted margin: {spread_pred['predicted_margin']:+.1f} (home)")
            print(f"           Vegas line: {spread_line if spread_line else 'N/A'}")
            print(f"           Confidence: {spread_pred['confidence']:.1%} on {spread_pred['bet_side']}")
            if spread_pred['edge'] is not None:
                print(f"           Edge: {spread_pred['edge']:+.1f} pts")
            if spread_pred['passes_filter']:
                print(f"           [PLAY] Spread bet recommended")
            else:
                print(f"           [SKIP] {spread_pred['filter_reason']}")

            # Display total prediction
            print(f"  [TOTAL] Predicted total: {total_pred['predicted_total']:.1f}")
            print(f"          Vegas line: {total_line if total_line else 'N/A'}")
            print(f"          Confidence: {total_pred['confidence']:.1%} on {total_pred['bet_side']}")
            if total_pred['edge'] is not None:
                print(f"          Edge: {total_pred['edge']:+.1f} pts")
            if total_pred['passes_filter']:
                print(f"          [PLAY] Total bet recommended")
            else:
                print(f"          [SKIP] {total_pred['filter_reason']}")

            # Store prediction
            if commence_time:
                game_cst = to_cst(parse_utc_time(commence_time))
                date_cst = game_cst.strftime("%Y-%m-%d %I:%M %p CST")
            else:
                date_cst = datetime.now(CST).strftime("%Y-%m-%d %I:%M %p CST")

            predictions.append({
                "date": date_cst,
                "home_team": home_team,
                "away_team": away_team,
                # Spread
                "spread_line": spread_line,
                "spread_pred_margin": round(spread_pred['predicted_margin'], 1),
                "spread_edge": round(spread_pred['edge'], 1) if spread_pred['edge'] else None,
                "spread_bet_side": spread_pred['bet_side'],
                "spread_confidence": round(spread_pred['confidence'], 3),
                "spread_passes_filter": spread_pred['passes_filter'],
                "spread_filter_reason": spread_pred['filter_reason'] or "",
                # Total
                "total_line": total_line,
                "total_pred": round(total_pred['predicted_total'], 1),
                "total_edge": round(total_pred['edge'], 1) if total_pred['edge'] else None,
                "total_bet_side": total_pred['bet_side'],
                "total_confidence": round(total_pred['confidence'], 3),
                "total_passes_filter": total_pred['passes_filter'],
                "total_filter_reason": total_pred['filter_reason'] or "",
            })

        except Exception as e:
            print(f"  [ERROR] {e}")
            continue

    # Save and display results
    if predictions:
        save_predictions(predictions)
    else:
        print("\n[ERROR] No predictions generated")


def save_predictions(predictions: list):
    """Save predictions and generate betting card."""
    output_path = DATA_DIR / "processed" / "predictions.csv"
    df = pd.DataFrame(predictions)
    df.to_csv(output_path, index=False)
    print(f"\n{'='*80}")
    print(f"[OK] Saved {len(predictions)} predictions to {output_path}")
    print(f"{'='*80}")

    # Generate betting card
    spread_plays = df[df["spread_passes_filter"] == True].copy()
    total_plays = df[df["total_passes_filter"] == True].copy()

    print("\n" + "=" * 80)
    print("BETTING CARD - FILTERED PLAYS")
    print("=" * 80)
    print(f"Strategy: Smart filtering (spreads + totals)")
    print(f"Spreads: Avoid 3-6 pts, require 5% edge (expect 60.6% / +15.7% ROI)")
    print(f"Totals: Baseline model (expect 59.2% / +13.1% ROI)")
    print("=" * 80)

    total_plays_count = len(spread_plays) + len(total_plays)

    if total_plays_count > 0:
        # Combine and sort by market type and edge
        all_plays = []

        for _, row in spread_plays.iterrows():
            all_plays.append({
                "matchup": f"{row['away_team']} @ {row['home_team']}",
                "date": row['date'],
                "market": "SPREAD",
                "pick": f"{row['spread_bet_side']} ({row['spread_confidence']:.1%})",
                "line": row['spread_line'],
                "edge": row['spread_edge'],
                "confidence": row['spread_confidence'],
            })

        for _, row in total_plays.iterrows():
            all_plays.append({
                "matchup": f"{row['away_team']} @ {row['home_team']}",
                "date": row['date'],
                "market": "TOTAL",
                "pick": f"{row['total_bet_side']} {row['total_line']:.1f} ({row['total_confidence']:.1%})",
                "line": row['total_line'],
                "edge": row['total_edge'],
                "confidence": row['total_confidence'],
            })

        # Sort by confidence descending
        all_plays = sorted(all_plays, key=lambda x: x['confidence'], reverse=True)

        for play in all_plays:
            print(f"\n{play['matchup']}")
            print(f"  Game Time: {play['date']}")
            print(f"  Market: {play['market']}")
            print(f"  Pick: {play['pick']}")
            if play['edge'] is not None:
                print(f"  Edge: {play['edge']:+.1f} pts")

        print("\n" + "=" * 80)
        print(f"TOTAL PLAYS: {total_plays_count} ({len(spread_plays)} spreads, {len(total_plays)} totals)")
        print("=" * 80)

        # Save betting card
        betting_card_df = pd.DataFrame(all_plays)
        betting_card_path = DATA_DIR / "processed" / "betting_card.csv"
        betting_card_df.to_csv(betting_card_path, index=False)
        print(f"[OK] Saved betting card to {betting_card_path}")

    else:
        print("\nNO PLAYS TODAY")
        print("All games filtered out - no bets meet criteria")
        print("=" * 80)

    # Show filter summary
    total_games = len(df)
    spread_filtered = len(df[df["spread_passes_filter"] == False])
    total_filtered = len(df[df["total_passes_filter"] == False])

    print(f"\nFilter Summary:")
    print(f"  Total games analyzed: {total_games}")
    print(f"  Spread plays: {len(spread_plays)} (filtered out: {spread_filtered})")
    print(f"  Total plays: {len(total_plays)} (filtered out: {total_filtered})")

    if spread_filtered > 0:
        print(f"\n  Spread filter reasons:")
        filter_counts = df[df["spread_passes_filter"] == False]["spread_filter_reason"].value_counts()
        for reason, count in filter_counts.items():
            print(f"    - {reason}: {count}")


def main():
    # Set random seeds
    np.random.seed(42)
    random.seed(42)
    logger.info("Random seeds set to 42 for reproducibility")

    parser = argparse.ArgumentParser(description="Generate NBA predictions")
    parser.add_argument("--date", help="Date for predictions (YYYY-MM-DD, 'today', or 'tomorrow')")
    parser.add_argument("--no-betting-splits", action="store_true",
                       help="Disable betting splits integration")
    args = parser.parse_args()

    asyncio.run(predict_games_async(args.date, use_betting_splits=not args.no_betting_splits))


if __name__ == "__main__":
    main()
