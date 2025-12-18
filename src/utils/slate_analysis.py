"""
Utility functions for slate analysis.

Extracted from deprecated analyze_todays_slate.py script.
These functions are used by the API and Docker analysis scripts.
"""
from __future__ import annotations
from datetime import datetime, timedelta, timezone, date
from typing import Dict, List, Any, Optional
from zoneinfo import ZoneInfo
import statistics

from src.ingestion import the_odds

# Central Standard Time
CST = ZoneInfo("America/Chicago")


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


def get_target_date(date_str: str | None = None) -> date:
    """Get target date for analysis."""
    now_cst = get_cst_now()
    
    if date_str is None or date_str.lower() == "today":
        return now_cst.date()
    elif date_str.lower() == "tomorrow":
        return (now_cst + timedelta(days=1)).date()
    else:
        return datetime.strptime(date_str, "%Y-%m-%d").date()


async def fetch_todays_games(target_date: date) -> List[Dict]:
    """
    Fetch games for a specific date, including first half markets.
    
    First half markets (spreads_h1, totals_h1, h2h_h1) are only available
    through the event-specific endpoint, so we fetch them separately for each game.
    """
    # Fetch main odds data (full game markets)
    raw_games = await the_odds.fetch_odds()
    filtered_games = filter_games_for_date(raw_games, target_date)
    
    # Enrich with first half markets for each game
    enriched_games = []
    for game in filtered_games:
        event_id = game.get("id")
        if event_id:
            try:
                # Fetch 1H odds specifically (spreads_h1, totals_h1, h2h_h1)
                event_odds = await the_odds.fetch_event_odds(
                    event_id,
                    markets="spreads_h1,totals_h1,h2h_h1"
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
                # Log warning but continue - first half markets are optional
                import logging
                logging.getLogger(__name__).warning(
                    f"Could not fetch 1H odds for event {event_id}: {e}"
                )
        
        enriched_games.append(game)
    
    return enriched_games


def filter_games_for_date(games: list, target_date: date) -> list:
    """Filter games to only include those on the target date (in CST)."""
    if games is None:
        return []
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
                        totals.append({
                            "point": out.get("point"),
                            "price": out.get("price"),
                            "side": "Under"
                        })
            
            # First half markets
            # Market keys from API: "spreads_h1", "totals_h1", "h2h_h1"
            elif key and ("h1" in key.lower() or "first_half" in key.lower() or "1h" in key.lower() or key.lower() == "spreads_h1" or key.lower() == "totals_h1"):
                if "spread" in key.lower() or "handicap" in key.lower() or key.lower() == "spreads_h1":
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
                elif "total" in key.lower() or "over_under" in key.lower() or key.lower() == "totals_h1":
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
    
    # Calculate consensus
    result = {
        "home_ml": None,
        "away_ml": None,
        "home_spread": None,
        "home_spread_price": None,
        "total": None,
        "total_price": None,
        "home_implied_prob": None,
        "away_implied_prob": None,
        "fh_home_spread": None,
        "fh_home_spread_price": None,
        "fh_total": None,
        "fh_total_price": None,
    }
    
    if h2h_home:
        median_home = statistics.median(h2h_home)
        if abs(median_home) > 2000:
            median_home = 1000 * (1 if median_home > 0 else -1)
        result["home_ml"] = int(median_home)
        result["home_implied_prob"] = american_to_implied_prob(result["home_ml"])
    if h2h_away:
        median_away = statistics.median(h2h_away)
        if abs(median_away) > 2000:
            median_away = 1000 * (1 if median_away > 0 else -1)
        result["away_ml"] = int(median_away)
        result["away_implied_prob"] = american_to_implied_prob(result["away_ml"])
    
    if spreads_home:
        median_spread = statistics.median([s.get("point", 0) for s in spreads_home if s.get("point") is not None])
        median_price = statistics.median([s.get("price", -110) for s in spreads_home if s.get("price") is not None])
        result["home_spread"] = float(median_spread)
        result["home_spread_price"] = int(median_price)
    
    if totals:
        median_total = statistics.median([t.get("point", 220) for t in totals if t.get("point") is not None])
        median_price = statistics.median([t.get("price", -110) for t in totals if t.get("price") is not None])
        result["total"] = float(median_total)
        result["total_price"] = int(median_price)
    
    if fh_spreads_home:
        median_fh_spread = statistics.median([s.get("point", 0) for s in fh_spreads_home if s.get("point") is not None])
        median_fh_price = statistics.median([s.get("price", -110) for s in fh_spreads_home if s.get("price") is not None])
        result["fh_home_spread"] = float(median_fh_spread)
        result["fh_home_spread_price"] = int(median_fh_price)
    
    if fh_totals:
        median_fh_total = statistics.median([t.get("point", 110) for t in fh_totals if t.get("point") is not None])
        median_fh_price = statistics.median([t.get("price", -110) for t in fh_totals if t.get("price") is not None])
        result["fh_total"] = float(median_fh_total)
        result["fh_total_price"] = int(median_fh_price)
    
    return result
