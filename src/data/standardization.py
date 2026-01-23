"""
SINGLE SOURCE OF TRUTH: Data Standardization Module

ALL data ingestion MUST go through this module for:
1. Team name standardization (canonical full names)
2. Timezone conversion to CST (Central Standard Time)
3. Match key generation
4. Date/time formatting

TIMEZONE STANDARD: All dates/times stored as CST (America/Chicago)
- NBA games are played in US timezones (EST/CST/MST/PST)
- CST is central to all US timezones, minimizing offset issues
- All API data (UTC) is converted to CST before storage

TEAM NAME STANDARD: Full canonical names (e.g., "Los Angeles Lakers")
- All abbreviations mapped to full names
- Case-insensitive matching
- Uses src/ingestion/team_mapping.json as source of truth
"""
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Union
from zoneinfo import ZoneInfo

# CST timezone - SINGLE SOURCE OF TRUTH for all dates
CST = ZoneInfo("America/Chicago")
UTC = ZoneInfo("UTC")

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
TEAM_MAPPING_FILE = PROJECT_ROOT / "src" / "ingestion" / "team_mapping.json"

# Load team mapping at module level
_TEAM_MAPPING: Dict[str, list] = {}
_VARIANT_TO_CANONICAL: Dict[str, str] = {}
_CANONICAL_TO_FULL: Dict[str, str] = {}


def _load_team_mapping():
    """Load team mapping from JSON file (called once at import)."""
    global _TEAM_MAPPING, _VARIANT_TO_CANONICAL, _CANONICAL_TO_FULL

    if TEAM_MAPPING_FILE.exists():
        with open(TEAM_MAPPING_FILE, "r") as f:
            _TEAM_MAPPING = json.load(f)

        # Build reverse lookup
        for canonical_id, variants in _TEAM_MAPPING.items():
            for variant in variants:
                _VARIANT_TO_CANONICAL[variant.lower().strip()] = canonical_id

    # Canonical ID to full name mapping
    # CRITICAL: These must match ESPN's standings API format exactly
    _CANONICAL_TO_FULL = {
        "nba_atl": "Atlanta Hawks",
        "nba_bos": "Boston Celtics",
        "nba_bkn": "Brooklyn Nets",
        "nba_cha": "Charlotte Hornets",
        "nba_chi": "Chicago Bulls",
        "nba_cle": "Cleveland Cavaliers",
        "nba_dal": "Dallas Mavericks",
        "nba_den": "Denver Nuggets",
        "nba_det": "Detroit Pistons",
        "nba_gsw": "Golden State Warriors",
        "nba_hou": "Houston Rockets",
        "nba_ind": "Indiana Pacers",
        "nba_lac": "LA Clippers",  # ESPN uses "LA Clippers" NOT "Los Angeles Clippers"
        "nba_lal": "Los Angeles Lakers",
        "nba_mem": "Memphis Grizzlies",
        "nba_mia": "Miami Heat",
        "nba_mil": "Milwaukee Bucks",
        "nba_min": "Minnesota Timberwolves",
        "nba_nop": "New Orleans Pelicans",
        "nba_nyk": "New York Knicks",
        "nba_okc": "Oklahoma City Thunder",
        "nba_orl": "Orlando Magic",
        "nba_phi": "Philadelphia 76ers",
        "nba_phx": "Phoenix Suns",
        "nba_por": "Portland Trail Blazers",
        "nba_sac": "Sacramento Kings",
        "nba_sas": "San Antonio Spurs",
        "nba_tor": "Toronto Raptors",
        "nba_utah": "Utah Jazz",
        "nba_was": "Washington Wizards",
    }


# Load at module import
_load_team_mapping()


# =============================================================================
# TEAM NAME STANDARDIZATION
# =============================================================================

# Additional abbreviation mappings (supplements team_mapping.json)
# CRITICAL: Use ESPN's exact team names ("LA Clippers" NOT "Los Angeles Clippers")
ABBREV_TO_FULL = {
    # Standard 2-3 letter abbreviations
    "atl": "Atlanta Hawks", "bos": "Boston Celtics", "bkn": "Brooklyn Nets",
    "cha": "Charlotte Hornets", "chi": "Chicago Bulls", "cle": "Cleveland Cavaliers",
    "dal": "Dallas Mavericks", "den": "Denver Nuggets", "det": "Detroit Pistons",
    "gs": "Golden State Warriors", "gsw": "Golden State Warriors",
    "hou": "Houston Rockets", "ind": "Indiana Pacers",
    "lac": "LA Clippers", "lal": "Los Angeles Lakers",
    "mem": "Memphis Grizzlies", "mia": "Miami Heat", "mil": "Milwaukee Bucks",
    "min": "Minnesota Timberwolves", "no": "New Orleans Pelicans",
    "nop": "New Orleans Pelicans", "nyk": "New York Knicks", "ny": "New York Knicks",
    "okc": "Oklahoma City Thunder", "orl": "Orlando Magic",
    "phi": "Philadelphia 76ers", "phx": "Phoenix Suns", "pho": "Phoenix Suns",
    "por": "Portland Trail Blazers", "sac": "Sacramento Kings",
    "sa": "San Antonio Spurs", "sas": "San Antonio Spurs",
    "tor": "Toronto Raptors", "utah": "Utah Jazz", "uta": "Utah Jazz",
    "was": "Washington Wizards", "wsh": "Washington Wizards",
    # Historical
    "nj": "Brooklyn Nets", "njn": "Brooklyn Nets",
    "sea": "Oklahoma City Thunder",
    "noh": "New Orleans Pelicans", "nok": "New Orleans Pelicans",
    # Nicknames only
    "hawks": "Atlanta Hawks", "celtics": "Boston Celtics", "nets": "Brooklyn Nets",
    "hornets": "Charlotte Hornets", "bulls": "Chicago Bulls", "cavaliers": "Cleveland Cavaliers",
    "cavs": "Cleveland Cavaliers", "mavericks": "Dallas Mavericks", "mavs": "Dallas Mavericks",
    "nuggets": "Denver Nuggets", "pistons": "Detroit Pistons", "warriors": "Golden State Warriors",
    "rockets": "Houston Rockets", "pacers": "Indiana Pacers", "clippers": "LA Clippers",
    "lakers": "Los Angeles Lakers", "grizzlies": "Memphis Grizzlies", "heat": "Miami Heat",
    "bucks": "Milwaukee Bucks", "timberwolves": "Minnesota Timberwolves", "wolves": "Minnesota Timberwolves",
    "pelicans": "New Orleans Pelicans", "knicks": "New York Knicks", "thunder": "Oklahoma City Thunder",
    "magic": "Orlando Magic", "76ers": "Philadelphia 76ers", "sixers": "Philadelphia 76ers",
    "suns": "Phoenix Suns", "blazers": "Portland Trail Blazers", "trail blazers": "Portland Trail Blazers",
    "kings": "Sacramento Kings", "spurs": "San Antonio Spurs", "raptors": "Toronto Raptors",
    "jazz": "Utah Jazz", "wizards": "Washington Wizards",
}


def standardize_team_name(team: str) -> str:
    """
    Convert any team name variant to canonical full name.

    SINGLE SOURCE OF TRUTH for team name standardization.

    Args:
        team: Any team name variant (abbreviation, nickname, full name)

    Returns:
        Canonical full name (e.g., "Los Angeles Lakers")

    Examples:
        >>> standardize_team_name("lal")
        'Los Angeles Lakers'
        >>> standardize_team_name("Lakers")
        'Los Angeles Lakers'
        >>> standardize_team_name("Los Angeles Lakers")
        'Los Angeles Lakers'
    """
    if not team or not isinstance(team, str):
        return team

    team_lower = team.lower().strip()

    # Check direct abbreviation match
    if team_lower in ABBREV_TO_FULL:
        return ABBREV_TO_FULL[team_lower]

    # Check team_mapping.json variants
    if team_lower in _VARIANT_TO_CANONICAL:
        canonical_id = _VARIANT_TO_CANONICAL[team_lower]
        if canonical_id in _CANONICAL_TO_FULL:
            return _CANONICAL_TO_FULL[canonical_id]

    # Already a full name? Return as-is with title case
    for full_name in ABBREV_TO_FULL.values():
        if team_lower == full_name.lower():
            return full_name

    # Return original with title case as fallback
    return team.strip()


# =============================================================================
# TIMEZONE CONVERSION
# =============================================================================

def to_cst(dt: Union[datetime, str, None], source_is_utc: bool = True) -> Optional[datetime]:
    """
    Convert any datetime to CST (Central Standard Time).

    SINGLE SOURCE OF TRUTH for timezone conversion.
    All data should be stored in CST.

    Args:
        dt: datetime object or ISO string
        source_is_utc: If True (default), naive datetimes are treated as UTC.
                       If False, naive datetimes are treated as already CST.

    Returns:
        datetime in CST timezone (timezone-aware)
    """
    if dt is None:
        return None

    # Parse string to datetime
    if isinstance(dt, str):
        try:
            # Handle ISO format with Z suffix (UTC)
            if dt.endswith("Z"):
                dt = dt[:-1] + "+00:00"

            # Parse ISO format
            parsed = datetime.fromisoformat(dt)

            # If no timezone, use source_is_utc to determine
            if parsed.tzinfo is None:
                if source_is_utc:
                    parsed = parsed.replace(tzinfo=UTC)
                else:
                    # Already in CST/local time, just localize
                    parsed = parsed.replace(tzinfo=CST)

            dt = parsed
        except ValueError:
            # Try other common formats
            for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y"]:
                try:
                    parsed = datetime.strptime(dt, fmt)
                    if source_is_utc:
                        parsed = parsed.replace(tzinfo=UTC)
                    else:
                        parsed = parsed.replace(tzinfo=CST)
                    dt = parsed
                    break
                except ValueError:
                    continue
            else:
                return None

    # Convert to CST
    if dt.tzinfo is None:
        if source_is_utc:
            dt = dt.replace(tzinfo=UTC)
        else:
            dt = dt.replace(tzinfo=CST)

    return dt.astimezone(CST)


def to_cst_local(dt: Union[datetime, str, None]) -> Optional[datetime]:
    """
    Convert a local US datetime (already in US timezone) to CST.

    Use this for data sources like Kaggle that use local US dates.

    Args:
        dt: datetime in local US time (not UTC)

    Returns:
        datetime in CST timezone
    """
    return to_cst(dt, source_is_utc=False)


def to_cst_date(dt: Union[datetime, str, None], source_is_utc: bool = True) -> Optional[str]:
    """
    Convert datetime to CST date string (YYYY-MM-DD).

    Args:
        dt: datetime or ISO string
        source_is_utc: If True (default), naive datetimes are treated as UTC.

    Returns:
        Date string in YYYY-MM-DD format (CST)
    """
    cst_dt = to_cst(dt, source_is_utc=source_is_utc)
    if cst_dt is None:
        return None
    return cst_dt.strftime("%Y-%m-%d")


def to_cst_date_local(dt: Union[datetime, str, None]) -> Optional[str]:
    """
    Convert local US datetime to CST date string.

    Use this for data sources like Kaggle that use local US dates.
    """
    return to_cst_date(dt, source_is_utc=False)


# =============================================================================
# MATCH KEY GENERATION
# =============================================================================

def generate_match_key(
    game_date: Union[datetime, str],
    home_team: str,
    away_team: str,
    source_is_utc: bool = True,
) -> str:
    """
    Generate canonical match key for game identification.

    SINGLE SOURCE OF TRUTH for match key generation.
    All data sources MUST use this function for game matching.

    Format: "YYYY-MM-DD_home team name_away team name"
    - Date in CST
    - Team names lowercase, standardized
    - Home team before away team (consistent ordering)

    Args:
        game_date: Game datetime (converted to CST date)
        home_team: Home team name (any variant)
        away_team: Away team name (any variant)
        source_is_utc: If True (default), datetime is UTC. If False, already local.

    Returns:
        Canonical match key string
    """
    # Convert to CST date
    date_str = to_cst_date(game_date, source_is_utc=source_is_utc)
    if not date_str:
        date_str = str(game_date)[:10]

    # Standardize team names
    home = standardize_team_name(home_team).lower()
    away = standardize_team_name(away_team).lower()

    return f"{date_str}_{home}_{away}"


# =============================================================================
# DATA STANDARDIZATION
# =============================================================================

def standardize_game_record(
    record: Dict[str, Any],
    source: str = "unknown",
) -> Dict[str, Any]:
    """
    Standardize a game record from any data source.

    SINGLE ENTRY POINT for game data standardization.

    Args:
        record: Raw game record from any source
        source: Source identifier (kaggle, theodds, api_basketball, etc.)

    Returns:
        Standardized game record with:
        - game_date_cst: CST datetime
        - date_str: YYYY-MM-DD in CST
        - home_team: Canonical full name
        - away_team: Canonical full name
        - match_key: Canonical match key
    """
    result = record.copy()

    # Extract datetime field (varies by source)
    dt_field = None
    for field in ["commence_time", "game_date", "date", "date_game", "game_date_est"]:
        if field in record and record[field]:
            dt_field = record[field]
            break

    # Convert to CST
    if dt_field:
        result["game_date_cst"] = to_cst(dt_field)
        result["date_str"] = to_cst_date(dt_field)

    # Standardize team names
    home_field = None
    away_field = None

    for hf in ["home_team", "home", "team_city_name_home"]:
        if hf in record and record[hf]:
            home_field = record[hf]
            if "team_nickname_home" in record:
                home_field = f"{record['team_city_name_home']} {record['team_nickname_home']}"
            break

    for af in ["away_team", "away", "team_city_name_away"]:
        if af in record and record[af]:
            away_field = record[af]
            if "team_nickname_away" in record:
                away_field = f"{record['team_city_name_away']} {record['team_nickname_away']}"
            break

    if home_field:
        result["home_team"] = standardize_team_name(home_field)
    if away_field:
        result["away_team"] = standardize_team_name(away_field)

    # Generate match key
    if result.get("game_date_cst") and result.get("home_team") and result.get("away_team"):
        result["match_key"] = generate_match_key(
            result["game_date_cst"],
            result["home_team"],
            result["away_team"],
        )

    # Add source for traceability
    result["_source"] = source

    return result


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_all_team_names() -> list[str]:
    """Get list of all canonical team names."""
    return list(set(ABBREV_TO_FULL.values()))


def is_valid_team(team: str) -> bool:
    """Check if a team name is valid/known."""
    std = standardize_team_name(team)
    return std in get_all_team_names()
