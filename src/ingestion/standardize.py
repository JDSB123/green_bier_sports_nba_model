"""
Data Standardization Module - ESPN as Single Source of Truth.

This module provides functions to standardize team names, dates, and game data
from different sources (The Odds API, API-Basketball) to match ESPN's format.

ESPN team names are the canonical format. All other sources must be mapped
to ESPN team names before processing.

STANDARD FORMAT: "AWAY TEAM vs. HOME TEAM"
- All game data uses 'away_team' and 'home_team' fields
- away_team always comes before home_team
- Display format: "AWAY TEAM vs. HOME TEAM"
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json

from src.config import get_nba_season
from src.utils.logging import get_logger

logger = get_logger(__name__)

# ESPN team names (canonical format)
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
    "LA Clippers",
    "Los Angeles Clippers",  # ESPN uses both
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
TEAM_NAME_MAPPING = {
    # The Odds API -> ESPN
    "Philadelphia 76ers": "Philadelphia 76ers",
    "Phi 76ers": "Philadelphia 76ers",
    "Phoenix Suns": "Phoenix Suns",
    "PHX Suns": "Phoenix Suns",
    "Los Angeles Lakers": "Los Angeles Lakers",
    "LA Lakers": "Los Angeles Lakers",
    "Los Angeles Clippers": "LA Clippers",
    "LA Clippers": "LA Clippers",
    "Golden State Warriors": "Golden State Warriors",
    "GS Warriors": "Golden State Warriors",
    "New York Knicks": "New York Knicks",
    "NY Knicks": "New York Knicks",
    "Brooklyn Nets": "Brooklyn Nets",
    "BKN Nets": "Brooklyn Nets",
    "San Antonio Spurs": "San Antonio Spurs",
    "SA Spurs": "San Antonio Spurs",
    "New Orleans Pelicans": "New Orleans Pelicans",
    "NO Pelicans": "New Orleans Pelicans",
    "Oklahoma City Thunder": "Oklahoma City Thunder",
    "OKC Thunder": "Oklahoma City Thunder",
    "Portland Trail Blazers": "Portland Trail Blazers",
    "POR Trail Blazers": "Portland Trail Blazers",
    "Toronto Raptors": "Toronto Raptors",
    "TOR Raptors": "Toronto Raptors",
    "Washington Wizards": "Washington Wizards",
    "WAS Wizards": "Washington Wizards",
    # API-Basketball -> ESPN (common variations)
    "Lakers": "Los Angeles Lakers",
    "Celtics": "Boston Celtics",
    "Knicks": "New York Knicks",
    "Warriors": "Golden State Warriors",
    "Nets": "Brooklyn Nets",
    "Heat": "Miami Heat",
    "76ers": "Philadelphia 76ers",
    "Sixers": "Philadelphia 76ers",
    "Clippers": "LA Clippers",
    "Nuggets": "Denver Nuggets",
    "Thunder": "Oklahoma City Thunder",
    "Pelicans": "New Orleans Pelicans",
    "Spurs": "San Antonio Spurs",
    "Trail Blazers": "Portland Trail Blazers",
    "Blazers": "Portland Trail Blazers",
    "Suns": "Phoenix Suns",
    "Hawks": "Atlanta Hawks",
    "Pistons": "Detroit Pistons",
    "Cavaliers": "Cleveland Cavaliers",
    "Cavs": "Cleveland Cavaliers",
    "Bulls": "Chicago Bulls",
    "Pacers": "Indiana Pacers",
    "Mavericks": "Dallas Mavericks",
    "Mavs": "Dallas Mavericks",
    "Timberwolves": "Minnesota Timberwolves",
    "Twolves": "Minnesota Timberwolves",
    "Jazz": "Utah Jazz",
    "Kings": "Sacramento Kings",
    "Magic": "Orlando Magic",
    "Raptors": "Toronto Raptors",
    "Rockets": "Houston Rockets",
    "Wizards": "Washington Wizards",
    "Grizzlies": "Memphis Grizzlies",
    "Bucks": "Milwaukee Bucks",
    "Hornets": "Charlotte Hornets",
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


def normalize_team_to_espn(team_name: str, source: str = "unknown") -> str:
    """
    Normalize team name from any source to ESPN format.
    
    This function ALWAYS attempts to standardize team names. It uses multiple
    strategies to match team names and logs warnings for unmatched names.
    
    Args:
        team_name: Team name from any source
        source: Source identifier ("the_odds", "api_basketball", etc.)
    
    Returns:
        ESPN team name (canonical format), or original if no match found
    
    Note:
        Never raises exceptions - always returns a string to ensure ingestion continues
    """
    if not team_name:
        logger.warning(f"Empty team name from source '{source}'")
        return ""
    
    original_name = team_name
    team_name = team_name.strip()
    
    if not team_name:
        logger.warning(f"Team name is whitespace only from source '{source}'")
        return original_name
    
    # Check if already in ESPN format (exact match)
    if team_name in ESPN_TEAM_NAMES:
        return team_name
    
    # Check direct mapping (exact match)
    if team_name in TEAM_NAME_MAPPING:
        result = TEAM_NAME_MAPPING[team_name]
        logger.debug(f"Direct mapped '{original_name}' -> '{result}' (source: {source})")
        return result
    
    # Try case-insensitive match in mapping
    team_lower = team_name.lower().strip()
    for key, espn_name in TEAM_NAME_MAPPING.items():
        if key.lower().strip() == team_lower:
            logger.debug(f"Case-insensitive mapped '{original_name}' -> '{espn_name}' (source: {source})")
            return espn_name
    
    # Try fuzzy matching (check if team name contains ESPN team name or vice versa)
    for espn_name, abbrevs in ESPN_TO_ABBREV.items():
        espn_lower = espn_name.lower()
        # Check if team name contains ESPN name or vice versa
        if team_lower in espn_lower or espn_lower in team_lower:
            logger.info(f"Fuzzy matched '{original_name}' -> '{espn_name}' (source: {source})")
            return espn_name
        # Check abbreviations
        for abbrev in abbrevs:
            abbrev_lower = abbrev.lower().strip()
            if abbrev_lower == team_lower or team_lower in abbrev_lower or abbrev_lower in team_lower:
                logger.info(f"Abbrev matched '{original_name}' -> '{espn_name}' (source: {source})")
                return espn_name
    
    # Try matching against ESPN team names directly (case-insensitive substring)
    for espn_name in ESPN_TEAM_NAMES:
        if team_lower in espn_name.lower() or espn_name.lower() in team_lower:
            logger.info(f"Substring matched '{original_name}' -> '{espn_name}' (source: {source})")
            return espn_name
    
    # If no match found, log warning but return original (fail-safe)
    logger.warning(
        f"⚠️  COULD NOT NORMALIZE team name '{original_name}' from source '{source}' to ESPN format. "
        f"This may cause data merging issues. Returning original name."
    )
    return team_name


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
    if home_team_raw:
        standardized["home_team"] = normalize_team_to_espn(str(home_team_raw), source)
    else:
        logger.warning(f"Missing home_team in game data from source '{source}'")
        standardized["home_team"] = ""
    
    if away_team_raw:
        standardized["away_team"] = normalize_team_to_espn(str(away_team_raw), source)
    else:
        logger.warning(f"Missing away_team in game data from source '{source}'")
        standardized["away_team"] = ""
    
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
    
    # Normalize date if present
    if "date" in result or "commence_time" in result or "start_time" in result:
        date_key = "date" if "date" in result else ("commence_time" if "commence_time" in result else "start_time")
        date_value = result.get(date_key)
        if date_value:
            try:
                if isinstance(date_value, str):
                    # Try parsing ISO format
                    if "T" in date_value or "Z" in date_value:
                        dt = datetime.fromisoformat(date_value.replace("Z", "+00:00"))
                    else:
                        dt = datetime.strptime(date_value, "%Y-%m-%d")
                    result["date"] = dt.date().isoformat()
                    result["season"] = get_nba_season(dt.date())
                elif isinstance(date_value, datetime):
                    result["date"] = date_value.date().isoformat()
                    result["season"] = get_nba_season(date_value.date())
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

