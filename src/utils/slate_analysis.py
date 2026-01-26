"""
Utility functions for slate analysis.

ODDS SOURCE: The Odds API provides all betting lines for the slate.
RECORDS SOURCE: ESPN standings are used for season W-L records, with a
fallback to The Odds API scores (recent games only) if ESPN is unavailable.

QA/QC Principle: All sources are normalized to ESPN team names to prevent
team mismatch across odds, records, and feature generation.

DATE ALIGNMENT PRINCIPLE: All data sources MUST query the SAME target date.
The DateContext class ensures consistent date formatting for each API.

Extracted from deprecated analyze_todays_slate.py script.
These functions are used by the API and Docker analysis scripts.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from src.ingestion import the_odds
from src.ingestion.standardize import CST, normalize_outcome_name
from src.ingestion.standardize import to_cst as _canonical_to_cst

logger = logging.getLogger(__name__)

# Eastern Time - NBA's official scheduling timezone
# ESPN and all NBA schedules use ET for game dates
ET = ZoneInfo("America/New_York")


# =============================================================================
# DATE CONTEXT - SINGLE SOURCE OF TRUTH FOR DATE ALIGNMENT
# =============================================================================
# All data sources must query the SAME target date. This class provides
# date formatting for each API to ensure consistency.
#
# IMPORTANT: NBA uses Eastern Time (ET) for scheduling, NOT UTC or CST.
# All date calculations should use ET to match ESPN and NBA official times.
# =============================================================================


@dataclass
class DateContext:
    """
    Single source of truth for date alignment across all data sources.

    When requesting a slate for a specific date, ALL data sources must
    query the same date. This class provides the proper date format
    for each API endpoint.

    Date Format Reference:
    - The Odds API: UTC ISO8601 (YYYY-MM-DDTHH:MM:SSZ), filters by commence_time
    - ESPN: YYYYMMDD string (interpreted as Eastern Time)
    - API-Basketball: YYYY-MM-DD string
    - Action Network: YYYY-MM-DD string (Eastern Time)

    CRITICAL: NBA uses EASTERN TIME (ET) for scheduling. All date windows
    are calculated from midnight ET to 11:59:59 PM ET, then converted to UTC
    for The Odds API filtering. This matches ESPN's behavior.
    """

    target_date: date
    # When this context was created (for staleness checks)
    created_at: datetime

    @classmethod
    def from_request(cls, date_str: str | None = None) -> "DateContext":
        """
        Create DateContext from a request date string.

        Args:
            date_str: Date string (YYYY-MM-DD, 'today', 'tomorrow', or None)

        Returns:
            DateContext with validated target date
        """
        # Use ET (Eastern Time) as the reference - NBA's official timezone
        now_et = datetime.now(ET)

        if date_str is None or date_str.lower() == "today":
            target = now_et.date()
        elif date_str.lower() == "tomorrow":
            target = (now_et + timedelta(days=1)).date()
        else:
            try:
                target = datetime.strptime(date_str, "%Y-%m-%d").date()
            except ValueError as e:
                raise ValueError(f"Invalid date format '{date_str}'. Use YYYY-MM-DD.") from e

        return cls(target_date=target, created_at=now_et)

    @property
    def as_iso(self) -> str:
        """YYYY-MM-DD format (API-Basketball, general use)."""
        return self.target_date.isoformat()

    @property
    def as_espn(self) -> str:
        """YYYYMMDD format (ESPN - interprets this as Eastern Time)."""
        return self.target_date.strftime("%Y%m%d")

    @property
    def as_utc_start(self) -> str:
        """
        UTC start of day ISO8601 (The Odds API filtering).

        Converts midnight ET to UTC to match ESPN's date interpretation.
        Example: 2026-01-26 midnight ET = 2026-01-26T05:00:00Z
        """
        # Start of day in ET (midnight), converted to UTC
        start_et = datetime.combine(self.target_date, datetime.min.time(), tzinfo=ET)
        start_utc = start_et.astimezone(timezone.utc)
        return start_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    @property
    def as_utc_end(self) -> str:
        """
        UTC end of day ISO8601 (The Odds API filtering).

        Converts 23:59:59 ET to UTC to match ESPN's date interpretation.
        Example: 2026-01-26 23:59:59 ET = 2026-01-27T04:59:59Z
        """
        # End of day in ET (23:59:59), converted to UTC
        end_et = datetime.combine(
            self.target_date, datetime.max.time().replace(microsecond=0), tzinfo=ET
        )
        end_utc = end_et.astimezone(timezone.utc)
        return end_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    @property
    def as_action_network(self) -> str:
        """YYYY-MM-DD format (Action Network uses ET, same as ISO)."""
        return self.as_iso

    @property
    def as_api_basketball_tz(self) -> str:
        """
        IANA timezone string for API-Basketball queries.

        API-Basketball requires explicit timezone parameter to interpret
        the date correctly. Without it, dates are interpreted as UTC,
        causing late-night ET games to appear on the wrong date.
        """
        return "America/New_York"

    def is_stale(self, max_age_seconds: int = 300) -> bool:
        """Check if this context is stale (>5 minutes old by default)."""
        now = datetime.now(ET)
        age = (now - self.created_at).total_seconds()
        return age > max_age_seconds

    def validate_game_date(self, commence_time_utc: str) -> bool:
        """
        Validate that a game's commence time falls on the target date (ET).

        Args:
            commence_time_utc: ISO8601 UTC timestamp from The Odds API

        Returns:
            True if game is on target date in Eastern Time, False otherwise
        """
        try:
            if commence_time_utc.endswith("Z"):
                commence_time_utc = f"{commence_time_utc[:-1]}+00:00"
            dt_utc = datetime.fromisoformat(commence_time_utc)
            dt_et = dt_utc.astimezone(ET)  # Convert to ET (NBA's timezone)
            return dt_et.date() == self.target_date
        except (ValueError, TypeError):
            return False

    def __str__(self) -> str:
        return f"DateContext({self.as_iso}, created={self.created_at.isoformat()})"


def validate_date_alignment(
    date_ctx: DateContext,
    games: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Validate that fetched games align with the requested date.

    Args:
        date_ctx: The DateContext for this request
        games: List of game dicts with commence_time

    Returns:
        Dict with validation results:
        - is_valid: True if all games are on target date
        - target_date: The requested date
        - games_on_date: Count of games matching target date
        - games_off_date: Count of games NOT on target date
        - off_date_games: List of mismatched games (for debugging)

    Raises:
        ValueError: If more than 20% of games are off-date (data source issue)
    """
    on_date = []
    off_date = []

    for game in games:
        commence_time = game.get("commence_time")
        if not commence_time:
            off_date.append(game)
            continue

        if date_ctx.validate_game_date(commence_time):
            on_date.append(game)
        else:
            off_date.append(game)

    total = len(games)
    off_date_pct = len(off_date) / total if total > 0 else 0

    result = {
        "is_valid": len(off_date) == 0,
        "target_date": date_ctx.as_iso,
        "games_on_date": len(on_date),
        "games_off_date": len(off_date),
        "off_date_games": [
            {
                "home": g.get("home_team"),
                "away": g.get("away_team"),
                "commence": g.get("commence_time"),
            }
            for g in off_date[:5]  # Limit for payload size
        ],
    }

    # Warn if significant date mismatch
    if off_date_pct > 0.2 and total > 0:
        logger.warning(
            f"DATE ALIGNMENT WARNING: {len(off_date)}/{total} games ({off_date_pct:.0%}) "
            f"not on target date {date_ctx.as_iso}. This may indicate a timezone issue."
        )

    return result


# NOTE: CST imported from src.ingestion.standardize (single source of truth)

# =============================================================================
# TEAM RECORDS (ESPN PRIMARY, THE ODDS FALLBACK)
# =============================================================================
# ESPN standings provide season W-L records. If ESPN fails, fall back to The
# Odds API scores endpoint (limited to recent days) to avoid missing records.
# =============================================================================


# Cache for team records (session-level, cleared between API calls)
_UNIFIED_RECORDS_CACHE: Dict[str, Dict[str, int]] = {}
_UNIFIED_RECORDS_SOURCE: Optional[str] = None


def clear_unified_records_cache():
    """Clear the unified records cache to force fresh data."""
    global _UNIFIED_RECORDS_CACHE, _UNIFIED_RECORDS_SOURCE
    _UNIFIED_RECORDS_CACHE = {}
    _UNIFIED_RECORDS_SOURCE = None
    logger.info("[UNIFIED] Team records cache cleared")


async def fetch_team_records(
    days_back: int = 90,
    sport: str = "basketball_nba",
) -> Tuple[Dict[str, Dict[str, int]], Optional[str]]:
    """
    Fetch team records with ESPN as primary, The Odds API as fallback.

    Args:
        days_back: Number of days for The Odds fallback window (capped by API)
        sport: Sport identifier for The Odds API

    Returns:
        Tuple of (records dict, source label)
    """
    global _UNIFIED_RECORDS_CACHE, _UNIFIED_RECORDS_SOURCE

    if _UNIFIED_RECORDS_CACHE:
        return _UNIFIED_RECORDS_CACHE, _UNIFIED_RECORDS_SOURCE

    try:
        from src.ingestion.espn import fetch_espn_standings

        standings = await fetch_espn_standings()
        records: Dict[str, Dict[str, int]] = {}
        for team_name, standing in standings.items():
            games_played = standing.wins + standing.losses
            records[team_name] = {
                "wins": standing.wins,
                "losses": standing.losses,
                "games_played": games_played,
            }
        _UNIFIED_RECORDS_CACHE = records
        _UNIFIED_RECORDS_SOURCE = "espn"
        logger.info(f"[RECORDS] Loaded {len(records)} team records from ESPN standings")
        return records, _UNIFIED_RECORDS_SOURCE
    except Exception as e:
        logger.warning(
            f"[RECORDS] ESPN standings unavailable, falling back to The Odds scores: {e}"
        )

    try:
        records = await fetch_team_records_from_odds_api(days_back=days_back, sport=sport)
        days_from = min(days_back, 3)
        _UNIFIED_RECORDS_CACHE = records
        _UNIFIED_RECORDS_SOURCE = f"the_odds_api_last_{days_from}_days"
        logger.info(
            f"[RECORDS] Loaded {len(records)} team records from The Odds scores (last {days_from} days)"
        )
        return records, _UNIFIED_RECORDS_SOURCE
    except Exception as e:
        logger.warning(f"[RECORDS] Could not fetch records from The Odds scores: {e}")
        _UNIFIED_RECORDS_SOURCE = None
        return {}, None


async def fetch_team_records_from_odds_api(
    days_back: int = 90, sport: str = "basketball_nba"
) -> Dict[str, Dict[str, int]]:
    """
    Calculate team W-L records from The Odds API scores endpoint.

    NOTE: The Odds API scores endpoint only supports a short lookback.
    This function is intended as a fallback when ESPN standings are unavailable.

    Args:
        days_back: Number of days of scores to fetch (capped by API to 3)
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

    days_from = min(days_back, 3)
    logger.info(f"[UNIFIED] Fetching team records from The Odds API scores (last {days_from} days)")

    try:
        # Fetch recent scores from The Odds API
        # The scores endpoint returns completed games with final scores
        scores = await the_odds.fetch_scores(sport=sport, days_from=days_from)

        if not scores:
            logger.warning("[UNIFIED] No scores returned from The Odds API")
            return {}

        # Calculate W-L for each team
        # NOTE: Scores from the_odds.fetch_scores() are standardized to ESPN format
        # via standardize_game_data(), so team names are already normalized.
        # Records dictionary uses ESPN team names as keys (e.g., "Los Angeles Lakers").
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
                score_name = normalize_outcome_name(score.get("name"), source="the_odds")
                if score_name == home_team:
                    home_score = score.get("score")
                elif score_name == away_team:
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


def _lookup_team_record_with_synonyms(
    team_name: str, records: Dict[str, Dict[str, int]]
) -> Dict[str, int]:
    """
    Look up team record with synonym-aware matching.

    NO SILENT FALLBACKS: Raises ValueError if record cannot be found.

    Args:
        team_name: Team name to look up
        records: Dictionary of team records (team name -> {"wins": int, "losses": int})

    Returns:
        Dictionary with "wins" and "losses" keys

    Raises:
        ValueError: If team record cannot be found (even after synonym matching)
    """
    from src.ingestion.standardize import normalize_team_to_espn
    from src.utils.team_names import get_canonical_name, normalize_team_name

    # Try exact match first
    if team_name in records:
        return records[team_name]

    # Normalize team name to standard format
    normalized_name, is_valid = normalize_team_to_espn(team_name, source="unified_records")

    if not is_valid:
        available_teams = list(records.keys())[:10]
        raise ValueError(
            f"INVALID TEAM NAME: '{team_name}' cannot be normalized. "
            f"Available teams: {available_teams}"
        )

    # Try exact match with normalized name
    if normalized_name in records:
        return records[normalized_name]

    # Handle team name synonyms (e.g., "LA Clippers" <-> "Los Angeles Clippers")
    # Some teams have multiple valid standard format names. Both normalize to the same
    # canonical ID, so we check all records keys that map to the same canonical ID.
    canonical_id = normalize_team_name(normalized_name)
    if canonical_id.startswith("nba_"):
        canonical_form = get_canonical_name(canonical_id)
        # Try canonical form if different from normalized (MASTER db canonical)
        if canonical_form != normalized_name and canonical_form in records:
            return records[canonical_form]

        # Check all record keys to find ones that normalize to the same canonical ID
        # This handles synonyms like "LA Clippers" <-> "Los Angeles Clippers"
        for record_key in records:
            record_canonical_id = normalize_team_name(record_key)
            if record_canonical_id == canonical_id:
                # Found a synonym - return its record
                return records[record_key]

    # NO SILENT FALLBACK: Fail loudly if record cannot be found
    available_teams = list(records.keys())[:10]
    raise ValueError(
        f"MISSING TEAM RECORD: No record found for '{team_name}' "
        f"(normalized: '{normalized_name}', canonical_id: '{canonical_id}'). "
        f"Available teams in records: {available_teams}. "
        f"This indicates a data integrity issue - team name mismatch between odds and records."
    )


async def get_unified_team_record(team_name: str) -> Tuple[int, int]:
    """
    Get wins and losses for a team from the records source.

    Records source uses ESPN standings when available, with The Odds scores
    as a fallback. Team names are standardized to ESPN format for matching.

    Args:
        team_name: Team name (standardized format)

    Returns:
        Tuple of (wins, losses)

    Raises:
        ValueError: If team record cannot be found (no silent fallback)
    """
    records, _ = await fetch_team_records()
    record_dict = _lookup_team_record_with_synonyms(team_name, records)
    return record_dict["wins"], record_dict["losses"]


async def validate_data_integrity(
    odds_teams: List[str],
    record_teams: List[str],
    records_source: Optional[str] = None,
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

    source_label = records_source or "unknown"
    return {
        "is_valid": is_valid,
        "odds_teams_count": len(odds_set),
        "record_teams_count": len(record_set),
        "missing_from_records": list(missing_from_records),
        "missing_from_odds": list(missing_from_odds),
        "data_source": {
            "odds": "the_odds_api",
            "records": source_label,
        },
        "message": (
            f"Records source: {source_label}"
            if is_valid
            else f"WARNING: {len(missing_from_records)} teams in odds missing from records (records source: {source_label})"
        ),
    }


def get_cst_now() -> datetime:
    """Get current time in CST."""
    return datetime.now(CST)


def parse_utc_time(iso_string: str) -> datetime:
    """Parse ISO UTC time string to datetime."""
    if iso_string.endswith("Z"):
        iso_string = f"{iso_string[:-1]}+00:00"
    dt = datetime.fromisoformat(iso_string)
    # If no timezone info, assume UTC (API default)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def to_cst(dt: datetime) -> datetime:
    """Convert datetime to CST (uses canonical implementation)."""
    return _canonical_to_cst(dt)


def get_target_date(date_str: str | None = None) -> date:
    """
    Get target date for analysis.

    DEPRECATED: Use DateContext.from_request() for new code.
    This function is kept for backward compatibility.
    """
    return DateContext.from_request(date_str).target_date


async def fetch_todays_games(
    target_date: date,
    include_records: bool = True,
    date_context: DateContext | None = None,
) -> List[Dict]:
    """
    Fetch games for a specific date, including first half markets and team records.

    ODDS SOURCE: The Odds API provides betting lines.
    RECORDS SOURCE: ESPN standings provide season W-L, with The Odds scores
    as a fallback when ESPN is unavailable.

    First half markets (spreads_h1, totals_h1) are only available
    through the event-specific endpoint, so we fetch them separately for each game.

    DATE ALIGNMENT: When date_context is provided, uses UTC window filtering
    at the API level to ensure all data sources query the same date.

    Args:
        target_date: Date to fetch games for
        include_records: If True, include team W-L records from the records source
        date_context: Optional DateContext for precise UTC window filtering

    Returns:
        List of game dicts with odds and optionally team records
    """
    # Clear the unified records cache to ensure fresh data
    clear_unified_records_cache()

    # Build date context if not provided (for backwards compatibility)
    if date_context is None:
        date_context = DateContext(target_date)
        logger.debug(f"[DATE ALIGNMENT] Created DateContext for {target_date}")

    # Fetch main odds data with DATE FILTERING at API level
    # This prevents UTC/CST drift issues by filtering server-side
    logger.info(
        f"[DATE ALIGNMENT] Fetching odds for {target_date} "
        f"(UTC window: {date_context.as_utc_start} to {date_context.as_utc_end})"
    )
    raw_games = await the_odds.fetch_odds(
        commence_time_from=date_context.as_utc_start,
        commence_time_to=date_context.as_utc_end,
    )

    # Additional client-side filter as safety net (handles edge cases)
    filtered_games = filter_games_for_date(raw_games, target_date)

    if len(filtered_games) != len(raw_games):
        logger.warning(
            f"[DATE ALIGNMENT] API returned {len(raw_games)} games, "
            f"filtered to {len(filtered_games)} for {target_date} CST"
        )

    # Fetch team records from the records source (ESPN primary, The Odds fallback)
    unified_records = {}
    records_source = None
    if include_records:
        try:
            unified_records, records_source = await fetch_team_records()
            if records_source:
                logger.info(
                    f"[RECORDS] Fetched records for {len(unified_records)} teams from {records_source}"
                )
            else:
                logger.warning("[RECORDS] Records source unavailable")
        except Exception as e:
            logger.warning(f"[RECORDS] Could not fetch team records: {e}")

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
                    event_id, markets="spreads_h1,totals_h1"
                )

                # MERGE instead of overwrite
                # Keep existing bookmakers (FG) and add new ones (1H)
                existing_bms = {bm["key"]: bm for bm in game.get("bookmakers", [])}
                new_bms = event_odds.get("bookmakers", [])

                for nbm in new_bms:
                    if nbm["key"] in existing_bms:
                        # Add markets to existing bookmaker
                        existing_markets = {
                            m["key"]: m for m in existing_bms[nbm["key"]].get("markets", [])
                        }
                        for nm in nbm.get("markets", []):
                            existing_markets[nm["key"]] = nm
                        existing_bms[nbm["key"]]["markets"] = list(existing_markets.values())
                    else:
                        existing_bms[nbm["key"]] = nbm

                game["bookmakers"] = list(existing_bms.values())
            except Exception as e:
                logger.warning(f"Could not fetch 1H odds for event {event_id}: {e}")

        # Add team records from the configured records source
        # NO SILENT FALLBACKS: Use synonym-aware lookup, fail loudly if records missing
        if include_records and unified_records:
            try:
                home_record = _lookup_team_record_with_synonyms(home_team, unified_records)
                away_record = _lookup_team_record_with_synonyms(away_team, unified_records)

                game["home_team_record"] = {
                    "wins": home_record["wins"],
                    "losses": home_record["losses"],
                    "source": records_source or "unknown",
                }
                game["away_team_record"] = {
                    "wins": away_record["wins"],
                    "losses": away_record["losses"],
                    "source": records_source or "unknown",
                }
                # Flag indicating odds+records normalized to ESPN names
                game["_data_unified"] = True
            except ValueError as e:
                # NO SILENT FALLBACK: Log ERROR and skip game if records cannot be found
                logger.error(
                    f"[UNIFIED RECORDS] FAILED to lookup team records for game "
                    f"{away_team} @ {home_team}: {e}. "
                    f"Skipping game - NO PLACEHOLDER DATA."
                )
                # Skip this game - don't add it to enriched_games
                continue

        enriched_games.append(game)

    # QA/QC: Validate data integrity
    if include_records and unified_records:
        odds_teams = []
        for g in enriched_games:
            odds_teams.extend([g.get("home_team"), g.get("away_team")])

        validation = await validate_data_integrity(
            odds_teams=[t for t in odds_teams if t],
            record_teams=list(unified_records.keys()),
            records_source=records_source,
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
        # Prefer standardized CST date if available
        game_date = game.get("date")
        if game_date:
            try:
                if date.fromisoformat(str(game_date)) == target_date:
                    filtered.append(game)
                continue
            except Exception:
                pass

        commence_time_cst = game.get("commence_time_cst")
        if commence_time_cst:
            try:
                game_cst = datetime.fromisoformat(commence_time_cst)
                if game_cst.tzinfo is None:
                    game_cst = game_cst.replace(tzinfo=CST)
                if game_cst.astimezone(CST).date() == target_date:
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
                    out_name = normalize_outcome_name(out.get("name"), source="the_odds")
                    if out_name == home_team:
                        spreads_home.append({"price": out.get("price"), "point": out.get("point")})
                    elif out_name == away_team:
                        spreads_away.append({"price": out.get("price"), "point": out.get("point")})

            elif key == "totals":
                for out in outcomes:
                    out_name = normalize_outcome_name(out.get("name"), source="the_odds")
                    if out_name == "Over":
                        totals.append({"point": out.get("point"), "price": out.get("price")})
                        totals_over.append(out.get("price"))
                    elif out_name == "Under":
                        totals.append(
                            {"point": out.get("point"), "price": out.get("price"), "side": "Under"}
                        )
                        totals_under.append(out.get("price"))

            # First half markets (strict whitelist + sanity bounds)
            elif key:
                k = str(key).lower()
                if k == "spreads_h1":
                    for out in outcomes:
                        out_name = normalize_outcome_name(out.get("name"), source="the_odds")
                        if out_name == home_team:
                            fh_spreads_home.append(
                                {"price": out.get("price"), "point": out.get("point")}
                            )
                        elif out_name == away_team:
                            fh_spreads_away.append(
                                {"price": out.get("price"), "point": out.get("point")}
                            )
                elif k == "totals_h1":
                    for out in outcomes:
                        name = normalize_outcome_name(out.get("name"), source="the_odds")
                        point = out.get("point")
                        price = out.get("price")
                        # Sanity bounds for 1H totals to avoid misparsed FG/alt lines
                        try:
                            if point is None:
                                continue
                            pval = float(point)
                        except (TypeError, ValueError):
                            continue
                        if not (70.0 <= pval <= 130.0):
                            # Skip clearly invalid 1H totals
                            continue
                        if name == "Over":
                            fh_totals.append({"point": pval, "price": price})
                            fh_totals_over.append(price)
                        elif name == "Under":
                            fh_totals.append({"point": pval, "price": price, "side": "Under"})
                            fh_totals_under.append(price)

            # First quarter markets
            # Market keys from API: "spreads_q1", "totals_q1"
            elif key and (
                "q1" in key.lower() or "first_quarter" in key.lower() or "1q" in key.lower()
            ):
                if (
                    "spread" in key.lower()
                    or "handicap" in key.lower()
                    or key.lower() == "spreads_q1"
                ):
                    for out in outcomes:
                        out_name = normalize_outcome_name(out.get("name"), source="the_odds")
                        if out_name == home_team:
                            q1_spreads_home.append(
                                {"price": out.get("price"), "point": out.get("point")}
                            )
                        elif out_name == away_team:
                            q1_spreads_away.append(
                                {"price": out.get("price"), "point": out.get("point")}
                            )
                elif (
                    "total" in key.lower()
                    or "over_under" in key.lower()
                    or key.lower() == "totals_q1"
                ):
                    for out in outcomes:
                        out_name = normalize_outcome_name(out.get("name"), source="the_odds")
                        if out_name == "Over":
                            q1_totals.append({"point": out.get("point"), "price": out.get("price")})
                        elif out_name == "Under":
                            q1_totals.append(
                                {
                                    "point": out.get("point"),
                                    "price": out.get("price"),
                                    "side": "Under",
                                }
                            )

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
        median_spread = statistics.median(
            [s.get("point", 0) for s in spreads_home if s.get("point") is not None]
        )
        median_price = statistics.median(
            [s.get("price", -110) for s in spreads_home if s.get("price") is not None]
        )
        result["home_spread"] = float(median_spread)
        result["home_spread_price"] = int(median_price)
    if spreads_away:
        median_price = statistics.median(
            [s.get("price", -110) for s in spreads_away if s.get("price") is not None]
        )
        result["away_spread_price"] = int(median_price)

    if totals:
        median_total = statistics.median(
            [t.get("point", 220) for t in totals if t.get("point") is not None]
        )
        median_price = statistics.median(
            [t.get("price", -110) for t in totals if t.get("price") is not None]
        )
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
        median_fh_spread = statistics.median(
            [s.get("point", 0) for s in fh_spreads_home if s.get("point") is not None]
        )
        median_fh_price = statistics.median(
            [s.get("price", -110) for s in fh_spreads_home if s.get("price") is not None]
        )
        result["fh_home_spread"] = float(median_fh_spread)
        result["fh_home_spread_price"] = int(median_fh_price)
    if fh_spreads_away:
        median_fh_price = statistics.median(
            [s.get("price", -110) for s in fh_spreads_away if s.get("price") is not None]
        )
        result["fh_away_spread_price"] = int(median_fh_price)

    if fh_totals:
        median_fh_total = statistics.median(
            [t.get("point", 110) for t in fh_totals if t.get("point") is not None]
        )
        median_fh_price = statistics.median(
            [t.get("price", -110) for t in fh_totals if t.get("price") is not None]
        )
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
        median_q1_spread = statistics.median(
            [s.get("point", 0) for s in q1_spreads_home if s.get("point") is not None]
        )
        median_q1_price = statistics.median(
            [s.get("price", -110) for s in q1_spreads_home if s.get("price") is not None]
        )
        result["q1_home_spread"] = float(median_q1_spread)
        result["q1_home_spread_price"] = int(median_q1_price)

    # Q1 total
    if q1_totals:
        median_q1_total = statistics.median(
            [t.get("point", 55) for t in q1_totals if t.get("point") is not None]
        )
        median_q1_price = statistics.median(
            [t.get("price", -110) for t in q1_totals if t.get("price") is not None]
        )
        result["q1_total"] = float(median_q1_total)
        result["q1_total_price"] = int(median_q1_price)

    return result
