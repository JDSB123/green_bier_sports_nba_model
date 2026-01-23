"""
Data Standardization Module - Unified Team Name Matching.

This module provides functions to standardize team names, dates, and game data
from different sources (The Odds API, API-Basketball, etc.) to a canonical format.

IMPORTANT CLARIFICATION:
- "Standard format" team names (e.g., "Los Angeles Lakers") are a NAMING CONVENTION
- This naming style happens to match ESPN's format, but data comes from MULTIPLE sources
- ESPN is used as a data source ONLY for schedules (see src/ingestion/espn.py)
- Most data (odds, stats) comes from The Odds API and API-Basketball
- Team records are sourced from ESPN standings in the serving layer

Uses the MASTER team_mapping.json database via src.utils.team_names for:
- Canonical team IDs (nba_lal, nba_bos, etc.)
- Comprehensive variant matching (95+ variants)
- Fuzzy matching for edge cases

Standard team names are the OUTPUT format. Internal processing uses canonical IDs.

STANDARD FORMAT: "AWAY TEAM vs. HOME TEAM"
- All game data uses 'away_team' and 'home_team' fields
- away_team always comes before home_team
- Display format: "AWAY TEAM vs. HOME TEAM"
"""
from __future__ import annotations
from src.config import settings
from src.utils.team_names import (
    normalize_team_name as master_normalize,
    get_canonical_name,
    VARIANT_TO_CANONICAL,
    CANONICAL_NAMES,
)

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import json

from src.config import get_nba_season
from src.utils.logging import get_logger

# Central Standard Time - SINGLE SOURCE OF TRUTH for all game times
CST = ZoneInfo("America/Chicago")
UTC = ZoneInfo("UTC")

# Import MASTER team name system

logger = get_logger(__name__)

# Best-effort read-through cache with TTL (does NOT replace team_mapping.json as source of truth)
_VARIANT_OVERRIDE_CACHE_PATH = Path(
    settings.data_processed_dir) / "cache" / "team_variant_overrides.json"
_VARIANT_OVERRIDE_CACHE: dict[str, str] | None = None
_VARIANT_OVERRIDE_CACHE_LOADED_AT: float | None = None
_VARIANT_OVERRIDE_CACHE_TTL_HOURS = 24  # Team names rarely change, 24h TTL


def _load_variant_override_cache() -> dict[str, str]:
    """Load variant override cache with TTL check.

    Added TTL to prevent stale cache issues.
    """
    global _VARIANT_OVERRIDE_CACHE, _VARIANT_OVERRIDE_CACHE_LOADED_AT
    import time

    current_time = time.time()

    # Check if cache is still fresh
    if _VARIANT_OVERRIDE_CACHE is not None and _VARIANT_OVERRIDE_CACHE_LOADED_AT is not None:
        age_hours = (current_time - _VARIANT_OVERRIDE_CACHE_LOADED_AT) / 3600
        if age_hours < _VARIANT_OVERRIDE_CACHE_TTL_HOURS:
            return _VARIANT_OVERRIDE_CACHE
        else:
            logger.info(
                f"Team variant cache expired (age: {age_hours:.1f}h), reloading")
            _VARIANT_OVERRIDE_CACHE = None

    try:
        if _VARIANT_OVERRIDE_CACHE_PATH.exists():
            with open(_VARIANT_OVERRIDE_CACHE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    _VARIANT_OVERRIDE_CACHE = {
                        str(k).lower(): str(v) for k, v in data.items()}
                    _VARIANT_OVERRIDE_CACHE_LOADED_AT = current_time
                    return _VARIANT_OVERRIDE_CACHE
    except Exception as e:
        logger.warning(f"Failed to load variant override cache: {e}")

    _VARIANT_OVERRIDE_CACHE = {}
    _VARIANT_OVERRIDE_CACHE_LOADED_AT = current_time
    return _VARIANT_OVERRIDE_CACHE


def _persist_variant_override_cache(cache: dict[str, str]) -> None:
    """Persist variant override cache to disk."""
    try:
        _VARIANT_OVERRIDE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_VARIANT_OVERRIDE_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2, sort_keys=True)
    except Exception as e:
        logger.warning(f"Failed to persist variant override cache: {e}")


def _cache_variant_override(*, raw: str, normalized: str) -> None:
    """Cache a resolved variant for faster future normalization."""
    try:
        if not raw or not normalized:
            return
        cache = _load_variant_override_cache()
        key = str(raw).strip().lower()
        if not key:
            return
        if cache.get(key) == normalized:
            return
        cache[key] = normalized
        _persist_variant_override_cache(cache)
    except Exception as e:
        logger.warning(f"Failed to cache variant override: {e}")


# ESPN team names (canonical format)
# CRITICAL: This must match EXACTLY what ESPN's standings API returns
# ESPN uses "LA Clippers" (NOT "Los Angeles Clippers") in their standings API
ESPN_TEAM_NAMES = {
    "Atlanta Hawks",
    "Boston Celtics",
    "Brooklyn Nets",
    "Charlotte Hornets",
    "Chicago Bulls",
    "Cleveland Cavaliers",
    "Dallas Mavericks",
    "Denver Nuggets",
    "Detroit Pistons",
    "Golden State Warriors",
    "Houston Rockets",
    "Indiana Pacers",
    "LA Clippers",  # ESPN uses "LA Clippers", NOT "Los Angeles Clippers"
    "Los Angeles Lakers",
    "Memphis Grizzlies",
    "Miami Heat",
    "Milwaukee Bucks",
    "Minnesota Timberwolves",
    "New Orleans Pelicans",
    "New York Knicks",
    "Oklahoma City Thunder",
    "Orlando Magic",
    "Philadelphia 76ers",
    "Phoenix Suns",
    "Portland Trail Blazers",
    "Sacramento Kings",
    "San Antonio Spurs",
    "Toronto Raptors",
    "Utah Jazz",
    "Washington Wizards",
}

# Mapping from other sources to ESPN team names
# COMPREHENSIVE: Includes all known variants from The Odds API, API-Basketball, Action Network, etc.
TEAM_NAME_MAPPING = {
    # === ATLANTA HAWKS ===
    "Atlanta Hawks": "Atlanta Hawks",
    "ATL Hawks": "Atlanta Hawks",
    "Hawks": "Atlanta Hawks",
    "ATL": "Atlanta Hawks",
    "Atlanta": "Atlanta Hawks",

    # === BOSTON CELTICS ===
    "Boston Celtics": "Boston Celtics",
    "BOS Celtics": "Boston Celtics",
    "Celtics": "Boston Celtics",
    "BOS": "Boston Celtics",
    "Boston": "Boston Celtics",

    # === BROOKLYN NETS ===
    "Brooklyn Nets": "Brooklyn Nets",
    "BKN Nets": "Brooklyn Nets",
    "BK Nets": "Brooklyn Nets",
    "Nets": "Brooklyn Nets",
    "BKN": "Brooklyn Nets",
    "Brooklyn": "Brooklyn Nets",

    # === CHARLOTTE HORNETS ===
    "Charlotte Hornets": "Charlotte Hornets",
    "CHA Hornets": "Charlotte Hornets",
    "Hornets": "Charlotte Hornets",
    "CHA": "Charlotte Hornets",
    "Charlotte": "Charlotte Hornets",

    # === CHICAGO BULLS ===
    "Chicago Bulls": "Chicago Bulls",
    "CHI Bulls": "Chicago Bulls",
    "Bulls": "Chicago Bulls",
    "CHI": "Chicago Bulls",
    "Chicago": "Chicago Bulls",

    # === CLEVELAND CAVALIERS ===
    "Cleveland Cavaliers": "Cleveland Cavaliers",
    "CLE Cavaliers": "Cleveland Cavaliers",
    "Cavaliers": "Cleveland Cavaliers",
    "Cavs": "Cleveland Cavaliers",
    "CLE": "Cleveland Cavaliers",
    "Cleveland": "Cleveland Cavaliers",

    # === DALLAS MAVERICKS ===
    "Dallas Mavericks": "Dallas Mavericks",
    "DAL Mavericks": "Dallas Mavericks",
    "Mavericks": "Dallas Mavericks",
    "Mavs": "Dallas Mavericks",
    "DAL": "Dallas Mavericks",
    "Dallas": "Dallas Mavericks",

    # === DENVER NUGGETS ===
    "Denver Nuggets": "Denver Nuggets",
    "DEN Nuggets": "Denver Nuggets",
    "Nuggets": "Denver Nuggets",
    "DEN": "Denver Nuggets",
    "Denver": "Denver Nuggets",

    # === DETROIT PISTONS ===
    "Detroit Pistons": "Detroit Pistons",
    "DET Pistons": "Detroit Pistons",
    "Pistons": "Detroit Pistons",
    "DET": "Detroit Pistons",
    "Detroit": "Detroit Pistons",

    # === GOLDEN STATE WARRIORS ===
    "Golden State Warriors": "Golden State Warriors",
    "GS Warriors": "Golden State Warriors",
    "GSW Warriors": "Golden State Warriors",
    "Warriors": "Golden State Warriors",
    "GSW": "Golden State Warriors",
    "GS": "Golden State Warriors",
    "Golden State": "Golden State Warriors",

    # === HOUSTON ROCKETS ===
    "Houston Rockets": "Houston Rockets",
    "HOU Rockets": "Houston Rockets",
    "Rockets": "Houston Rockets",
    "HOU": "Houston Rockets",
    "Houston": "Houston Rockets",

    # === INDIANA PACERS ===
    "Indiana Pacers": "Indiana Pacers",
    "IND Pacers": "Indiana Pacers",
    "Pacers": "Indiana Pacers",
    "IND": "Indiana Pacers",
    "Indiana": "Indiana Pacers",

    # === LA CLIPPERS ===
    "LA Clippers": "LA Clippers",
    "Los Angeles Clippers": "LA Clippers",
    "LAC Clippers": "LA Clippers",
    "L.A. Clippers": "LA Clippers",
    "Clippers": "LA Clippers",
    "LAC": "LA Clippers",

    # === LOS ANGELES LAKERS ===
    "Los Angeles Lakers": "Los Angeles Lakers",
    "LA Lakers": "Los Angeles Lakers",
    "LAL Lakers": "Los Angeles Lakers",
    "L.A. Lakers": "Los Angeles Lakers",
    "Lakers": "Los Angeles Lakers",
    "LAL": "Los Angeles Lakers",

    # === MEMPHIS GRIZZLIES ===
    "Memphis Grizzlies": "Memphis Grizzlies",
    "MEM Grizzlies": "Memphis Grizzlies",
    "Grizzlies": "Memphis Grizzlies",
    "Grizz": "Memphis Grizzlies",
    "MEM": "Memphis Grizzlies",
    "Memphis": "Memphis Grizzlies",

    # === MIAMI HEAT ===
    "Miami Heat": "Miami Heat",
    "MIA Heat": "Miami Heat",
    "Heat": "Miami Heat",
    "MIA": "Miami Heat",
    "Miami": "Miami Heat",

    # === MILWAUKEE BUCKS ===
    "Milwaukee Bucks": "Milwaukee Bucks",
    "MIL Bucks": "Milwaukee Bucks",
    "Bucks": "Milwaukee Bucks",
    "MIL": "Milwaukee Bucks",
    "Milwaukee": "Milwaukee Bucks",

    # === MINNESOTA TIMBERWOLVES ===
    "Minnesota Timberwolves": "Minnesota Timberwolves",
    "MIN Timberwolves": "Minnesota Timberwolves",
    "Timberwolves": "Minnesota Timberwolves",
    "Twolves": "Minnesota Timberwolves",
    "T-Wolves": "Minnesota Timberwolves",
    "Wolves": "Minnesota Timberwolves",
    "MIN": "Minnesota Timberwolves",
    "Minnesota": "Minnesota Timberwolves",

    # === NEW ORLEANS PELICANS ===
    "New Orleans Pelicans": "New Orleans Pelicans",
    "NO Pelicans": "New Orleans Pelicans",
    "NOP Pelicans": "New Orleans Pelicans",
    "Pelicans": "New Orleans Pelicans",
    "Pels": "New Orleans Pelicans",
    "NOP": "New Orleans Pelicans",
    "NO": "New Orleans Pelicans",
    "New Orleans": "New Orleans Pelicans",

    # === NEW YORK KNICKS ===
    "New York Knicks": "New York Knicks",
    "NY Knicks": "New York Knicks",
    "NYK Knicks": "New York Knicks",
    "Knicks": "New York Knicks",
    "NYK": "New York Knicks",
    "NY": "New York Knicks",
    "New York": "New York Knicks",

    # === OKLAHOMA CITY THUNDER ===
    "Oklahoma City Thunder": "Oklahoma City Thunder",
    "OKC Thunder": "Oklahoma City Thunder",
    "Thunder": "Oklahoma City Thunder",
    "OKC": "Oklahoma City Thunder",
    "Oklahoma City": "Oklahoma City Thunder",
    "Oklahoma": "Oklahoma City Thunder",

    # === ORLANDO MAGIC ===
    "Orlando Magic": "Orlando Magic",
    "ORL Magic": "Orlando Magic",
    "Magic": "Orlando Magic",
    "ORL": "Orlando Magic",
    "Orlando": "Orlando Magic",

    # === PHILADELPHIA 76ERS ===
    "Philadelphia 76ers": "Philadelphia 76ers",
    "PHI 76ers": "Philadelphia 76ers",
    "Phi 76ers": "Philadelphia 76ers",
    "76ers": "Philadelphia 76ers",
    "Sixers": "Philadelphia 76ers",
    "PHI": "Philadelphia 76ers",
    "Philadelphia": "Philadelphia 76ers",
    "Philly": "Philadelphia 76ers",

    # === PHOENIX SUNS ===
    "Phoenix Suns": "Phoenix Suns",
    "PHX Suns": "Phoenix Suns",
    "PHO Suns": "Phoenix Suns",
    "Suns": "Phoenix Suns",
    "PHX": "Phoenix Suns",
    "PHO": "Phoenix Suns",
    "Phoenix": "Phoenix Suns",

    # === PORTLAND TRAIL BLAZERS ===
    "Portland Trail Blazers": "Portland Trail Blazers",
    "POR Trail Blazers": "Portland Trail Blazers",
    "Trail Blazers": "Portland Trail Blazers",
    "Blazers": "Portland Trail Blazers",
    "POR": "Portland Trail Blazers",
    "Portland": "Portland Trail Blazers",

    # === SACRAMENTO KINGS ===
    "Sacramento Kings": "Sacramento Kings",
    "SAC Kings": "Sacramento Kings",
    "Kings": "Sacramento Kings",
    "SAC": "Sacramento Kings",
    "Sacramento": "Sacramento Kings",

    # === SAN ANTONIO SPURS ===
    "San Antonio Spurs": "San Antonio Spurs",
    "SA Spurs": "San Antonio Spurs",
    "SAS Spurs": "San Antonio Spurs",
    "Spurs": "San Antonio Spurs",
    "SAS": "San Antonio Spurs",
    "SA": "San Antonio Spurs",
    "San Antonio": "San Antonio Spurs",

    # === TORONTO RAPTORS ===
    "Toronto Raptors": "Toronto Raptors",
    "TOR Raptors": "Toronto Raptors",
    "Raptors": "Toronto Raptors",
    "Raps": "Toronto Raptors",
    "TOR": "Toronto Raptors",
    "Toronto": "Toronto Raptors",

    # === UTAH JAZZ ===
    "Utah Jazz": "Utah Jazz",
    "UTA Jazz": "Utah Jazz",
    "Jazz": "Utah Jazz",
    "UTA": "Utah Jazz",
    "Utah": "Utah Jazz",

    # === WASHINGTON WIZARDS ===
    "Washington Wizards": "Washington Wizards",
    "WAS Wizards": "Washington Wizards",
    "WSH Wizards": "Washington Wizards",
    "Wizards": "Washington Wizards",
    "Wiz": "Washington Wizards",
    "WAS": "Washington Wizards",
    "WSH": "Washington Wizards",
    "Washington": "Washington Wizards",
}

# Reverse mapping: ESPN -> common abbreviations (for fuzzy matching)
ESPN_TO_ABBREV = {
    "Los Angeles Lakers": ["LAL", "Lakers", "LA Lakers"],
    "Boston Celtics": ["BOS", "Celtics"],
    "New York Knicks": ["NYK", "Knicks", "NY Knicks"],
    "Golden State Warriors": ["GSW", "Warriors", "GS Warriors"],
    "Brooklyn Nets": ["BKN", "Nets"],
    "Miami Heat": ["MIA", "Heat"],
    "Philadelphia 76ers": ["PHI", "76ers", "Sixers"],
    "LA Clippers": ["LAC", "Clippers", "Los Angeles Clippers"],
    "Denver Nuggets": ["DEN", "Nuggets"],
    "Oklahoma City Thunder": ["OKC", "Thunder"],
    "New Orleans Pelicans": ["NOP", "Pelicans"],
    "San Antonio Spurs": ["SAS", "Spurs"],
    "Portland Trail Blazers": ["POR", "Trail Blazers", "Blazers"],
    "Phoenix Suns": ["PHX", "Suns"],
    "Atlanta Hawks": ["ATL", "Hawks"],
    "Detroit Pistons": ["DET", "Pistons"],
    "Cleveland Cavaliers": ["CLE", "Cavaliers", "Cavs"],
    "Chicago Bulls": ["CHI", "Bulls"],
    "Indiana Pacers": ["IND", "Pacers"],
    "Dallas Mavericks": ["DAL", "Mavericks", "Mavs"],
    "Minnesota Timberwolves": ["MIN", "Timberwolves", "Twolves"],
    "Utah Jazz": ["UTA", "Jazz"],
    "Sacramento Kings": ["SAC", "Kings"],
    "Orlando Magic": ["ORL", "Magic"],
    "Toronto Raptors": ["TOR", "Raptors"],
    "Houston Rockets": ["HOU", "Rockets"],
    "Washington Wizards": ["WAS", "Wizards"],
    "Memphis Grizzlies": ["MEM", "Grizzlies"],
    "Milwaukee Bucks": ["MIL", "Bucks"],
    "Charlotte Hornets": ["CHA", "Hornets"],
}


def is_valid_espn_team_name(team_name: str) -> bool:
    """
    Check if a team name is a valid ESPN team name.

    Args:
        team_name: Team name to validate

    Returns:
        True if team name is in ESPN_TEAM_NAMES set
    """
    return team_name in ESPN_TEAM_NAMES


def to_cst(dt_value) -> Optional[datetime]:
    """
    Convert any datetime to CST (Central Standard Time).

    CRITICAL: All game times must be converted to CST for consistent date matching.
    The Odds API and API-Basketball return times in UTC, which can cause
    games to appear on the wrong date without conversion.

    Args:
        dt_value: datetime object, ISO string, or None

    Returns:
        datetime in CST timezone (timezone-aware), or None if conversion fails
    """
    if dt_value is None:
        return None

    # Parse string to datetime
    if isinstance(dt_value, str):
        try:
            # Handle ISO format with Z suffix (UTC)
            if dt_value.endswith("Z"):
                dt_value = dt_value[:-1] + "+00:00"

            # Parse ISO format
            parsed = datetime.fromisoformat(dt_value)

            # If no timezone, assume UTC (API default)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)

            dt_value = parsed
        except (ValueError, TypeError):
            return None

    # Convert to CST
    if isinstance(dt_value, datetime):
        if dt_value.tzinfo is None:
            dt_value = dt_value.replace(tzinfo=UTC)
        return dt_value.astimezone(CST)

    return None


def normalize_team_to_espn(team_name: str, source: str = "unknown", raise_on_failure: bool = False) -> Tuple[str, bool]:
    """
    Normalize team name from any source to standard format.

    NOTE: Function name uses "espn" for historical reasons, but this is a
    NAMING CONVENTION only (standard format like "Los Angeles Lakers").
    Data comes from multiple sources (The Odds API, API-Basketball, etc.).

    USES MASTER DATABASE: team_mapping.json via src.utils.team_names
    - Canonical IDs (nba_lal, nba_bos, etc.) for internal matching
    - Standard full names as OUTPUT format (e.g., "Los Angeles Lakers")
    - Fuzzy matching for edge cases (95+ variants supported)

    Args:
        team_name: Team name from any source
        source: Source identifier ("the_odds", "api_basketball", etc.)
        raise_on_failure: If True, raise ValueError when normalization fails (default: False)

    Returns:
        Tuple of (normalized_name: str, is_valid: bool)
        - normalized_name: ESPN team name if successful, empty string if failed
        - is_valid: True if name is a valid ESPN team name, False otherwise

    Raises:
        ValueError: If raise_on_failure=True and normalization fails
    """
    if not team_name:
        logger.error(
            f"Empty team name from source '{source}' - CANNOT STANDARDIZE")
        _record_team_variant(source=source, raw=team_name,
                             normalized="", is_valid=False, reason="empty")
        if raise_on_failure:
            raise ValueError(f"Empty team name from source '{source}'")
        return "", False

    original_name = team_name
    team_name = team_name.strip()

    if not team_name:
        logger.error(
            f"Team name is whitespace only from source '{source}' - CANNOT STANDARDIZE")
        _record_team_variant(source=source, raw=original_name,
                             normalized="", is_valid=False, reason="whitespace")
        if raise_on_failure:
            raise ValueError(
                f"Whitespace-only team name from source '{source}'")
        return "", False

    # Fast path: cached resolved variant
    try:
        cached = _load_variant_override_cache().get(original_name.strip().lower())
        if cached and cached in ESPN_TEAM_NAMES:
            _record_team_variant(source=source, raw=original_name,
                                 normalized=cached, is_valid=True, reason="variant_cache")
            return cached, True
    except Exception:
        pass

    # Check if already in ESPN format (exact match)
    if team_name in ESPN_TEAM_NAMES:
        _record_team_variant(source=source, raw=original_name,
                             normalized=team_name, is_valid=True, reason="already_espn")
        _cache_variant_override(raw=original_name, normalized=team_name)
        return team_name, True

    # ==========================================
    # MASTER DATABASE LOOKUP (team_mapping.json)
    # Uses canonical IDs for matching, ESPN names for output
    # ==========================================
    canonical_id = master_normalize(team_name)

    if canonical_id.startswith("nba_"):
        # Found in MASTER database - get ESPN full name
        espn_name = get_canonical_name(canonical_id)
        logger.debug(
            f"MASTER matched '{original_name}' -> {canonical_id} -> '{espn_name}' (source: {source})")
        _record_team_variant(source=source, raw=original_name,
                             normalized=espn_name, is_valid=True, reason="master_db")
        _cache_variant_override(raw=original_name, normalized=espn_name)
        return espn_name, True

    # ==========================================
    # FALLBACK: Legacy hardcoded mappings
    # (for any variants not yet in MASTER database)
    # ==========================================

    # Check direct mapping (exact match)
    if team_name in TEAM_NAME_MAPPING:
        result = TEAM_NAME_MAPPING[team_name]
        logger.debug(
            f"Legacy mapped '{original_name}' -> '{result}' (source: {source})")
        _record_team_variant(source=source, raw=original_name,
                             normalized=result, is_valid=True, reason="legacy_map_exact")
        _cache_variant_override(raw=original_name, normalized=result)
        return result, True

    # Try case-insensitive match in mapping
    team_lower = team_name.lower().strip()
    for key, espn_name in TEAM_NAME_MAPPING.items():
        if key.lower().strip() == team_lower:
            logger.debug(
                f"Case-insensitive mapped '{original_name}' -> '{espn_name}' (source: {source})")
            _record_team_variant(source=source, raw=original_name, normalized=espn_name,
                                 is_valid=True, reason="legacy_map_case_insensitive")
            _cache_variant_override(raw=original_name, normalized=espn_name)
            return espn_name, True

    # Try fuzzy matching (check if team name contains ESPN team name or vice versa)
    for espn_name, abbrevs in ESPN_TO_ABBREV.items():
        espn_lower = espn_name.lower()
        # Check if team name contains ESPN name or vice versa
        if team_lower in espn_lower or espn_lower in team_lower:
            logger.info(
                f"Fuzzy matched '{original_name}' -> '{espn_name}' (source: {source})")
            _record_team_variant(source=source, raw=original_name,
                                 normalized=espn_name, is_valid=True, reason="fuzzy_contains")
            _cache_variant_override(raw=original_name, normalized=espn_name)
            return espn_name, True
        # Check abbreviations
        for abbrev in abbrevs:
            abbrev_lower = abbrev.lower().strip()
            if abbrev_lower == team_lower or team_lower in abbrev_lower or abbrev_lower in team_lower:
                logger.info(
                    f"Abbrev matched '{original_name}' -> '{espn_name}' (source: {source})")
                _record_team_variant(source=source, raw=original_name,
                                     normalized=espn_name, is_valid=True, reason="fuzzy_abbrev")
                _cache_variant_override(
                    raw=original_name, normalized=espn_name)
                return espn_name, True

    # Try matching against ESPN team names directly (case-insensitive substring)
    for espn_name in ESPN_TEAM_NAMES:
        if team_lower in espn_name.lower() or espn_name.lower() in team_lower:
            logger.info(
                f"Substring matched '{original_name}' -> '{espn_name}' (source: {source})")
            _record_team_variant(source=source, raw=original_name,
                                 normalized=espn_name, is_valid=True, reason="fuzzy_substring")
            _cache_variant_override(raw=original_name, normalized=espn_name)
            return espn_name, True

    # If no match found, log ERROR (not warning) - this is a data quality issue
    logger.error(
        f"❌ FAILED TO NORMALIZE team name '{original_name}' from source '{source}' to ESPN format. "
        f"Not found in MASTER database (team_mapping.json) or legacy mappings. "
        f"Add variant to team_mapping.json to fix."
    )
    _record_team_variant(source=source, raw=original_name,
                         normalized="", is_valid=False, reason="no_match")

    if raise_on_failure:
        raise ValueError(
            f"Could not normalize team name '{original_name}' from source '{source}' to ESPN format. "
            f"Add variant to src/ingestion/team_mapping.json."
        )

    # Return empty string to indicate failure - DO NOT return original name (prevents fake data)
    return "", False


def _record_team_variant(*, source: str, raw: str, normalized: str, is_valid: bool, reason: str) -> None:
    """Append observed team-name variants to a JSONL cache for later reconciliation.

    This is intentionally best-effort and must never crash ingestion/prediction.
    """
    try:
        # Keep this under processed/cache so both prod + backtest can read it.
        cache_dir = Path(settings.data_processed_dir) / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        out_path = cache_dir / "team_name_variants.jsonl"
        payload = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "source": source,
            "raw": raw,
            "normalized": normalized,
            "is_valid": bool(is_valid),
            "reason": reason,
        }
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        # Never break core flow for diagnostics
        return


def standardize_game_data(
    game_data: Dict[str, Any],
    source: str,
    espn_schedule: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Standardize game data from any source to ESPN format.

    This function ALWAYS standardizes team names. It handles various input formats
    and ensures team names are normalized to ESPN format.

    Standard format: "AWAY TEAM vs. HOME TEAM"
    All game data will have 'away_team' and 'home_team' fields in that order.

    Args:
        game_data: Game data dictionary from source
        source: Source identifier ("the_odds", "api_basketball", etc.)
        espn_schedule: Optional ESPN schedule to match against

    Returns:
        Standardized game data with ESPN team names in "AWAY vs. HOME" format

    Note:
        This function never fails - it always returns standardized data even if
        team name matching is incomplete.
    """
    standardized = game_data.copy()

    # Extract team names (handle different source formats)
    # Try multiple possible keys for team names
    home_team_raw = (
        standardized.get("home_team")
        or standardized.get("homeTeam")
        or standardized.get("home")
        or (standardized.get("teams", {}).get("home", {}) if isinstance(standardized.get("teams"), dict) else {}).get("name", "")
        or ""
    )
    away_team_raw = (
        standardized.get("away_team")
        or standardized.get("awayTeam")
        or standardized.get("away")
        or (standardized.get("teams", {}).get("away", {}) if isinstance(standardized.get("teams"), dict) else {}).get("name", "")
        or ""
    )

    # ALWAYS normalize team names to ESPN format (mandatory)
    # Track validation status to prevent using fake/unstandardized data
    home_team_valid = False
    away_team_valid = False

    if home_team_raw:
        standardized["home_team"], home_team_valid = normalize_team_to_espn(
            str(home_team_raw), source)
        if not home_team_valid:
            logger.error(
                f"❌ INVALID DATA: Failed to standardize home_team '{home_team_raw}' from source '{source}'")
            standardized["home_team"] = ""  # Set to empty to indicate invalid
    else:
        logger.error(
            f"❌ MISSING DATA: Missing home_team in game data from source '{source}'")
        standardized["home_team"] = ""

    if away_team_raw:
        standardized["away_team"], away_team_valid = normalize_team_to_espn(
            str(away_team_raw), source)
        if not away_team_valid:
            logger.error(
                f"❌ INVALID DATA: Failed to standardize away_team '{away_team_raw}' from source '{source}'")
            standardized["away_team"] = ""  # Set to empty to indicate invalid
    else:
        logger.error(
            f"❌ MISSING DATA: Missing away_team in game data from source '{source}'")
        standardized["away_team"] = ""

    # Add validation flags to the standardized data
    standardized["_home_team_valid"] = home_team_valid
    standardized["_away_team_valid"] = away_team_valid
    standardized["_data_valid"] = home_team_valid and away_team_valid

    # Log error if data is invalid
    if not standardized["_data_valid"]:
        logger.error(
            f"❌ INVALID GAME DATA from source '{source}': "
            f"home_team_valid={home_team_valid}, away_team_valid={away_team_valid}. "
            f"This game data should NOT be used for merging/predictions."
        )

    # Ensure standard format: "AWAY TEAM vs. HOME TEAM"
    # Remove any non-standard keys
    for key in ["homeTeam", "awayTeam", "home", "away"]:
        if key in standardized and key not in ["home_team", "away_team"]:
            del standardized[key]

    # Ensure away_team comes before home_team in the dict (for consistency)
    # Create new dict with correct order
    result = {}
    if "away_team" in standardized:
        result["away_team"] = standardized["away_team"]
    if "home_team" in standardized:
        result["home_team"] = standardized["home_team"]

    # Copy all other fields
    for key, value in standardized.items():
        if key not in ["away_team", "home_team"]:
            result[key] = value

    # Normalize date if present - CRITICAL: Convert to CST before extracting date
    # The Odds API returns UTC times, which can cause date mismatches
    # Example: A 10pm CST game is 4am UTC the NEXT day
    if "date" in result or "commence_time" in result or "start_time" in result:
        date_key = "date" if "date" in result else (
            "commence_time" if "commence_time" in result else "start_time")
        date_value = result.get(date_key)
        if date_value:
            try:
                # Check if this is a date-only string (no time component)
                # Date-only strings should be treated as local dates, not UTC midnight
                if isinstance(date_value, str) and len(date_value) == 10 and date_value[4] == "-" and date_value[7] == "-":
                    # Pure date string like "2025-01-15" - treat as local CST date
                    result["date"] = date_value
                    result["season"] = get_nba_season(
                        datetime.strptime(date_value, "%Y-%m-%d").date())
                else:
                    # Full datetime - convert to CST first, then extract date
                    cst_dt = to_cst(date_value)
                    if cst_dt:
                        result["date"] = cst_dt.date().isoformat()
                        result["commence_time_cst"] = cst_dt.isoformat()
                        result["season"] = get_nba_season(cst_dt.date())
                    else:
                        logger.warning(
                            f"Could not convert date to CST: {date_value}")
            except Exception as e:
                logger.warning(f"Error normalizing date '{date_value}': {e}")

    # Add source metadata
    result["_source"] = source
    result["_standardized"] = True

    return result


def format_game_string(away_team: str, home_team: str) -> str:
    """
    Format game as standard string: "AWAY TEAM vs. HOME TEAM"

    Args:
        away_team: Away team name
        home_team: Home team name

    Returns:
        Formatted string: "AWAY TEAM vs. HOME TEAM"
    """
    return f"{away_team} vs. {home_team}"


def match_game_to_espn_schedule(
    game_data: Dict[str, Any],
    espn_schedule: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Match game data to ESPN schedule entry.

    Format: AWAY TEAM vs. HOME TEAM

    Args:
        game_data: Standardized game data (with away_team and home_team)
        espn_schedule: List of ESPN schedule entries

    Returns:
        Matching ESPN schedule entry or None
    """
    away_team = game_data.get("away_team")
    home_team = game_data.get("home_team")
    game_date = game_data.get("date")

    if not away_team or not home_team:
        return None

    for espn_game in espn_schedule:
        espn_away = espn_game.get("away_team")
        espn_home = espn_game.get("home_team")
        espn_date = espn_game.get("date")

        # Match: AWAY vs. HOME format
        if (espn_away == away_team and espn_home == home_team):
            # Check date if available
            if game_date and espn_date:
                if game_date == espn_date:
                    return espn_game
            else:
                return espn_game

    return None


# =============================================================================
# MATCH KEY GENERATION
# =============================================================================

def generate_match_key(
    game_date,
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
    cst_dt = to_cst(game_date)
    if cst_dt:
        date_str = cst_dt.strftime("%Y-%m-%d")
    else:
        date_str = str(game_date)[:10] if game_date else "unknown"

    # Standardize team names
    home_normalized, _ = normalize_team_to_espn(home_team, source="match_key")
    away_normalized, _ = normalize_team_to_espn(away_team, source="match_key")

    return f"{date_str}_{home_normalized.lower()}_{away_normalized.lower()}"
