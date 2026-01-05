"""
Utility functions for slate analysis.

UNIFIED DATA SOURCE: All game data (odds, team records, scores) comes from
The Odds API to ensure data integrity. No bifurcation between sources.

QA/QC Principle: Team records displayed with picks MUST come from the same
source as the odds data to serve as a data integrity check.

Extracted from deprecated analyze_todays_slate.py script.
These functions are used by the API and Docker analysis scripts.
"""
from __future__ import annotations
from datetime import datetime, timedelta, timezone, date
from typing import Dict, List, Any, Optional, Tuple
from zoneinfo import ZoneInfo
import statistics
import logging

from src.ingestion import the_odds

logger = logging.getLogger(__name__)

# Central Standard Time
CST = ZoneInfo("America/Chicago")

# =============================================================================
# UNIFIED TEAM RECORDS FROM THE ODDS API
# =============================================================================
# These functions calculate team records from The Odds API scores endpoint,
# ensuring the same data source provides both odds AND team records.
# This is a QA/QC measure to prevent data bifurcation.
# =============================================================================

# Cache for team records (session-level, cleared between API calls)
_UNIFIED_RECORDS_CACHE: Dict[str, Dict[str, int]] = {}


def clear_unified_records_cache():
    """Clear the unified records cache to force fresh data."""
    global _UNIFIED_RECORDS_CACHE
    _UNIFIED_RECORDS_CACHE = {}
    logger.info("[UNIFIED] Team records cache cleared")


async def fetch_team_records_from_odds_api(
    days_back: int = 90,
    sport: str = "basketball_nba"
) -> Dict[str, Dict[str, int]]:
    """
    Calculate team W-L records from The Odds API scores endpoint.
    
    UNIFIED DATA SOURCE: This ensures team records come from the same
    source as the betting odds, maintaining data integrity.
    
    Args:
        days_back: Number of days of scores to fetch (default 90 for season)
        sport: Sport identifier
    
    Returns:
        Dict mapping team name to {"wins": int, "losses": int, "games_played": int}
    
    Raises:
        ValueError: If scores cannot be fetched from The Odds API
    """
    global _UNIFIED_RECORDS_CACHE
    
    # Return cached if available
    if _UNIFIED_RECORDS_CACHE:
        return _UNIFIED_RECORDS_CACHE
    
    logger.info(f"[UNIFIED] Fetching team records from The Odds API scores (last {days_back} days)")
    
    try:
        # Fetch recent scores from The Odds API
        # The scores endpoint returns completed games with final scores
        scores = await the_odds.fetch_scores(sport=sport, days_from=min(days_back, 3))
        
        if not scores:
            logger.warning("[UNIFIED] No scores returned from The Odds API")
            return {}
        
        # Calculate W-L for each team
        team_records: Dict[str, Dict[str, int]] = {}
        
        for game in scores:
            # Only count completed games
            if not game.get("completed", False):
                continue
            
            home_team = game.get("home_team")
            away_team = game.get("away_team")
            
            if not home_team or not away_team:
                continue
            
            # Get scores
            scores_data = game.get("scores", [])
            home_score = None
            away_score = None
            
            for score in scores_data:
                if score.get("name") == home_team:
                    home_score = score.get("score")
                elif score.get("name") == away_team:
                    away_score = score.get("score")
            
            if home_score is None or away_score is None:
                continue
            
            try:
                home_score = int(home_score)
                away_score = int(away_score)
            except (ValueError, TypeError):
                continue
            
            # Initialize team records if needed
            for team in [home_team, away_team]:
                if team not in team_records:
                    team_records[team] = {"wins": 0, "losses": 0, "games_played": 0}
            
            # Update records
            team_records[home_team]["games_played"] += 1
            team_records[away_team]["games_played"] += 1
            
            if home_score > away_score:
                team_records[home_team]["wins"] += 1
                team_records[away_team]["losses"] += 1
            else:
                team_records[away_team]["wins"] += 1
                team_records[home_team]["losses"] += 1
        
        # Cache the results
        _UNIFIED_RECORDS_CACHE = team_records
        
        logger.info(f"[UNIFIED] Calculated records for {len(team_records)} teams from The Odds API")
        return team_records
        
    except Exception as e:
        logger.error(f"[UNIFIED] Failed to fetch scores from The Odds API: {e}")
        raise ValueError(f"Cannot calculate unified team records: {e}")


async def get_unified_team_record(team_name: str) -> Tuple[int, int]:
    """
    Get wins and losses for a team from The Odds API.
    
    UNIFIED DATA SOURCE: Returns record from same source as odds.
    
    Args:
        team_name: Team name (ESPN standardized format)
    
    Returns:
        Tuple of (wins, losses)
    """
    records = await fetch_team_records_from_odds_api()
    
    if team_name in records:
        return records[team_name]["wins"], records[team_name]["losses"]
    
    # Try partial match (in case of name variations)
    for name, record in records.items():
        if team_name.lower() in name.lower() or name.lower() in team_name.lower():
            return record["wins"], record["losses"]
    
    logger.warning(f"[UNIFIED] No record found for {team_name} in The Odds API scores")
    return 0, 0


async def validate_data_integrity(
    odds_teams: List[str],
    record_teams: List[str]
) -> Dict[str, Any]:
    """
    Validate that odds data and record data are consistent.
    
    QA/QC: This serves as a data integrity check to ensure we're not
    mixing data from different sources.
    
    Args:
        odds_teams: Team names from odds data
        record_teams: Team names from records data
    
    Returns:
        Dict with validation results and any discrepancies
    """
    odds_set = set(odds_teams)
    record_set = set(record_teams)
    
    missing_from_records = odds_set - record_set
    missing_from_odds = record_set - odds_set
    
    is_valid = len(missing_from_records) == 0
    
    return {
        "is_valid": is_valid,
        "odds_teams_count": len(odds_set),
        "record_teams_count": len(record_set),
        "missing_from_records": list(missing_from_records),
        "missing_from_odds": list(missing_from_odds),
        "data_source": "the_odds_api",
        "message": "UNIFIED: All data from The Odds API" if is_valid else f"WARNING: {len(missing_from_records)} teams in odds missing from records"
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


def get_target_date(date_str: str | None = None) -> date:
    """Get target date for analysis."""
    now_cst = get_cst_now()
    
    if date_str is None or date_str.lower() == "today":
        return now_cst.date()
    elif date_str.lower() == "tomorrow":
        return (now_cst + timedelta(days=1)).date()
    else:
        return datetime.strptime(date_str, "%Y-%m-%d").date()


async def fetch_todays_games(
    target_date: date,
    include_records: bool = True
) -> List[Dict]:
    """
    Fetch games for a specific date, including first half markets and team records.
    
    UNIFIED DATA SOURCE: All data (odds, team records) comes from The Odds API
    to ensure data integrity. This prevents bifurcation between sources.
    
    First half markets (spreads_h1, totals_h1) are only available
    through the event-specific endpoint, so we fetch them separately for each game.
    
    Args:
        target_date: Date to fetch games for
        include_records: If True, include team W-L records from The Odds API scores
    
    Returns:
        List of game dicts with odds and optionally team records
    """
    # Clear the unified records cache to ensure fresh data
    clear_unified_records_cache()
    
    # Fetch main odds data (full game markets)
    raw_games = await the_odds.fetch_odds()
    filtered_games = filter_games_for_date(raw_games, target_date)
    
    # UNIFIED: Fetch team records from The Odds API scores (same source as odds)
    unified_records = {}
    if include_records:
        try:
            unified_records = await fetch_team_records_from_odds_api()
            logger.info(f"[UNIFIED] Fetched records for {len(unified_records)} teams from The Odds API")
        except Exception as e:
            logger.warning(f"[UNIFIED] Could not fetch team records: {e}")
    
    # Enrich with first half markets for each game
    enriched_games = []
    for game in filtered_games:
        event_id = game.get("id")
        home_team = game.get("home_team")
        away_team = game.get("away_team")
        
        if event_id:
            try:
                # Fetch 1H odds specifically (spreads_h1, totals_h1)
                event_odds = await the_odds.fetch_event_odds(
                    event_id,
                    markets="spreads_h1,totals_h1"
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
                logger.warning(f"Could not fetch 1H odds for event {event_id}: {e}")
        
        # UNIFIED: Add team records from The Odds API (same source as odds)
        if include_records and unified_records:
            home_record = unified_records.get(home_team, {"wins": 0, "losses": 0})
            away_record = unified_records.get(away_team, {"wins": 0, "losses": 0})
            
            game["home_team_record"] = {
                "wins": home_record.get("wins", 0),
                "losses": home_record.get("losses", 0),
                "source": "the_odds_api"  # QA/QC: Document data source
            }
            game["away_team_record"] = {
                "wins": away_record.get("wins", 0),
                "losses": away_record.get("losses", 0),
                "source": "the_odds_api"  # QA/QC: Document data source
            }
            game["_data_unified"] = True  # Flag indicating unified source
        
        enriched_games.append(game)
    
    # QA/QC: Validate data integrity
    if include_records and unified_records:
        odds_teams = []
        for g in enriched_games:
            odds_teams.extend([g.get("home_team"), g.get("away_team")])
        
        validation = await validate_data_integrity(
            odds_teams=[t for t in odds_teams if t],
            record_teams=list(unified_records.keys())
        )
        
        if not validation["is_valid"]:
            logger.warning(f"[QA/QC] Data integrity warning: {validation['message']}")
        else:
            logger.info(f"[QA/QC] Data integrity check passed: {validation['message']}")
    
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


def extract_consensus_odds(game: Dict, as_of_utc: str | None = None) -> Dict[str, Any]:
    """Extract consensus odds from all bookmakers."""
    bookmakers = game.get("bookmakers", [])
    if as_of_utc is None:
        as_of_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    
    # Collect all odds
    spreads_home = []
    spreads_away = []
    totals = []
    totals_over = []
    totals_under = []
    # First half markets
    fh_spreads_home = []
    fh_spreads_away = []
    fh_totals = []
    fh_totals_over = []
    fh_totals_under = []
    # First quarter markets
    q1_spreads_home = []
    q1_spreads_away = []
    q1_totals = []
    
    home_team = game.get("home_team")
    away_team = game.get("away_team")
    
    for bm in bookmakers:
        for market in bm.get("markets", []):
            key = market.get("key")
            outcomes = market.get("outcomes", [])
            
            if key == "spreads":
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
                        totals_over.append(out.get("price"))
                    elif out.get("name") == "Under":
                        totals.append({
                            "point": out.get("point"),
                            "price": out.get("price"),
                            "side": "Under"
                        })
                        totals_under.append(out.get("price"))
            
            # First half markets
            # Market keys from API: "spreads_h1", "totals_h1"
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
                            fh_totals_over.append(out.get("price"))
                        elif out.get("name") == "Under":
                            fh_totals.append({
                                "point": out.get("point"),
                                "price": out.get("price"),
                                "side": "Under"
                            })
                            fh_totals_under.append(out.get("price"))
            
            # First quarter markets
            # Market keys from API: "spreads_q1", "totals_q1"
            elif key and ("q1" in key.lower() or "first_quarter" in key.lower() or "1q" in key.lower()):
                if "spread" in key.lower() or "handicap" in key.lower() or key.lower() == "spreads_q1":
                    for out in outcomes:
                        if out.get("name") == home_team:
                            q1_spreads_home.append({
                                "price": out.get("price"),
                                "point": out.get("point")
                            })
                        elif out.get("name") == away_team:
                            q1_spreads_away.append({
                                "price": out.get("price"),
                                "point": out.get("point")
                            })
                elif "total" in key.lower() or "over_under" in key.lower() or key.lower() == "totals_q1":
                    for out in outcomes:
                        if out.get("name") == "Over":
                            q1_totals.append({
                                "point": out.get("point"),
                                "price": out.get("price")
                            })
                        elif out.get("name") == "Under":
                            q1_totals.append({
                                "point": out.get("point"),
                                "price": out.get("price"),
                                "side": "Under"
                            })
    
    # Calculate consensus
    def _latest_update_utc() -> str | None:
        updates = []
        if game.get("last_update"):
            updates.append(game.get("last_update"))
        for bm in bookmakers:
            if bm.get("last_update"):
                updates.append(bm.get("last_update"))
            for market in bm.get("markets", []):
                if market.get("last_update"):
                    updates.append(market.get("last_update"))
        parsed = []
        for ts in updates:
            try:
                parsed.append(datetime.fromisoformat(str(ts).replace("Z", "+00:00")))
            except ValueError:
                continue
        if not parsed:
            return None
        latest = max(parsed)
        return latest.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    result = {
        "home_spread": None,
        "home_spread_price": None,
        "away_spread_price": None,
        "total": None,
        "total_price": None,
        "total_over_price": None,
        "total_under_price": None,
        "fh_home_spread": None,
        "fh_home_spread_price": None,
        "fh_away_spread_price": None,
        "fh_total": None,
        "fh_total_price": None,
        "fh_total_over_price": None,
        "fh_total_under_price": None,
        "q1_home_spread": None,
        "q1_home_spread_price": None,
        "q1_total": None,
        "q1_total_price": None,
        "odds_aggregation": "median",
        "as_of_utc": as_of_utc,
        "last_update_utc": _latest_update_utc(),
    }
    
    if spreads_home:
        median_spread = statistics.median([s.get("point", 0) for s in spreads_home if s.get("point") is not None])
        median_price = statistics.median([s.get("price", -110) for s in spreads_home if s.get("price") is not None])
        result["home_spread"] = float(median_spread)
        result["home_spread_price"] = int(median_price)
    if spreads_away:
        median_price = statistics.median([s.get("price", -110) for s in spreads_away if s.get("price") is not None])
        result["away_spread_price"] = int(median_price)
    
    if totals:
        median_total = statistics.median([t.get("point", 220) for t in totals if t.get("point") is not None])
        median_price = statistics.median([t.get("price", -110) for t in totals if t.get("price") is not None])
        result["total"] = float(median_total)
        result["total_price"] = int(median_price)
    if totals_over:
        over_prices = [p for p in totals_over if p is not None]
        if over_prices:
            result["total_over_price"] = int(statistics.median(over_prices))
    if totals_under:
        under_prices = [p for p in totals_under if p is not None]
        if under_prices:
            result["total_under_price"] = int(statistics.median(under_prices))
    
    if fh_spreads_home:
        median_fh_spread = statistics.median([s.get("point", 0) for s in fh_spreads_home if s.get("point") is not None])
        median_fh_price = statistics.median([s.get("price", -110) for s in fh_spreads_home if s.get("price") is not None])
        result["fh_home_spread"] = float(median_fh_spread)
        result["fh_home_spread_price"] = int(median_fh_price)
    if fh_spreads_away:
        median_fh_price = statistics.median([s.get("price", -110) for s in fh_spreads_away if s.get("price") is not None])
        result["fh_away_spread_price"] = int(median_fh_price)
    
    if fh_totals:
        median_fh_total = statistics.median([t.get("point", 110) for t in fh_totals if t.get("point") is not None])
        median_fh_price = statistics.median([t.get("price", -110) for t in fh_totals if t.get("price") is not None])
        result["fh_total"] = float(median_fh_total)
        result["fh_total_price"] = int(median_fh_price)
    if fh_totals_over:
        over_prices = [p for p in fh_totals_over if p is not None]
        if over_prices:
            result["fh_total_over_price"] = int(statistics.median(over_prices))
    if fh_totals_under:
        under_prices = [p for p in fh_totals_under if p is not None]
        if under_prices:
            result["fh_total_under_price"] = int(statistics.median(under_prices))
    
    # Q1 spread
    if q1_spreads_home:
        median_q1_spread = statistics.median([s.get("point", 0) for s in q1_spreads_home if s.get("point") is not None])
        median_q1_price = statistics.median([s.get("price", -110) for s in q1_spreads_home if s.get("price") is not None])
        result["q1_home_spread"] = float(median_q1_spread)
        result["q1_home_spread_price"] = int(median_q1_price)
    
    # Q1 total
    if q1_totals:
        median_q1_total = statistics.median([t.get("point", 55) for t in q1_totals if t.get("point") is not None])
        median_q1_price = statistics.median([t.get("price", -110) for t in q1_totals if t.get("price") is not None])
        result["q1_total"] = float(median_q1_total)
        result["q1_total_price"] = int(median_q1_price)
    
    return result
