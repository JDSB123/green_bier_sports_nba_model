"""
Automated team name reconciliation across different APIs.

Handles inconsistent team naming between:
- The Odds API (e.g., "Los Angeles Lakers")
- API-Basketball (e.g., "Lakers")
- BetsAPI (various formats)
- ESPN (e.g., "LAL")
"""
import json
from pathlib import Path
from typing import Optional, List
import difflib


# Load team mapping
MAPPING_FILE = Path(__file__).parent.parent / 'ingestion' / 'team_mapping.json'

with open(MAPPING_FILE, 'r') as f:
    TEAM_MAPPING = json.load(f)

# Create reverse lookup: any variant -> canonical ID
VARIANT_TO_CANONICAL = {}
for canonical_id, variants in TEAM_MAPPING.items():
    for variant in variants:
        VARIANT_TO_CANONICAL[variant.lower()] = canonical_id

# Canonical team names (full names)
CANONICAL_NAMES = {
    "nba_lal": "Los Angeles Lakers",
    "nba_bos": "Boston Celtics",
    "nba_nyk": "New York Knicks",
    "nba_gsw": "Golden State Warriors",
    "nba_bkn": "Brooklyn Nets",
    "nba_mia": "Miami Heat",
    "nba_phi": "Philadelphia 76ers",
    "nba_lac": "Los Angeles Clippers",
    "nba_den": "Denver Nuggets",
    "nba_okc": "Oklahoma City Thunder",
    "nba_nop": "New Orleans Pelicans",
    "nba_sas": "San Antonio Spurs",
    "nba_por": "Portland Trail Blazers",
    "nba_phx": "Phoenix Suns",
    "nba_atl": "Atlanta Hawks",
    "nba_det": "Detroit Pistons",
    "nba_cle": "Cleveland Cavaliers",
    "nba_chi": "Chicago Bulls",
    "nba_ind": "Indiana Pacers",
    "nba_dal": "Dallas Mavericks",
    "nba_min": "Minnesota Timberwolves",
    "nba_utah": "Utah Jazz",
    "nba_sac": "Sacramento Kings",
    "nba_orl": "Orlando Magic",
    "nba_tor": "Toronto Raptors",
    "nba_hou": "Houston Rockets",
    "nba_was": "Washington Wizards",
    "nba_mem": "Memphis Grizzlies",
    "nba_mil": "Milwaukee Bucks",
    "nba_cha": "Charlotte Hornets",
}


def normalize_team_name(team_name: str) -> str:
    """
    Normalize team name to canonical ID.

    Args:
        team_name: Any variant of team name

    Returns:
        Canonical ID (e.g., "nba_lal") or original if no match

    Examples:
        >>> normalize_team_name("Los Angeles Lakers")
        'nba_lal'
        >>> normalize_team_name("Lakers")
        'nba_lal'
        >>> normalize_team_name("LAL")
        'nba_lal'
    """
    if not team_name:
        return ""

    # Try exact match (case-insensitive)
    team_lower = team_name.lower().strip()
    if team_lower in VARIANT_TO_CANONICAL:
        return VARIANT_TO_CANONICAL[team_lower]

    # Try fuzzy match
    best_match = find_best_match(team_lower, VARIANT_TO_CANONICAL.keys())
    if best_match:
        return VARIANT_TO_CANONICAL[best_match]

    # Return original if no match
    return team_name


def get_canonical_name(team_id: str) -> str:
    """
    Get full canonical name from team ID.

    Args:
        team_id: Canonical team ID (e.g., "nba_lal")

    Returns:
        Full team name (e.g., "Los Angeles Lakers")
    """
    return CANONICAL_NAMES.get(team_id, team_id)


def find_best_match(query: str, candidates: List[str], threshold: float = 0.8) -> Optional[str]:
    """
    Find best fuzzy match for query string.

    Args:
        query: String to match
        candidates: List of candidate strings
        threshold: Minimum similarity score (0-1)

    Returns:
        Best matching candidate or None
    """
    matches = difflib.get_close_matches(query, candidates, n=1, cutoff=threshold)
    return matches[0] if matches else None


def reconcile_team_names(team1: str, team2: str) -> tuple[str, str]:
    """
    Reconcile two team names to canonical IDs.

    Args:
        team1: First team name (any variant)
        team2: Second team name (any variant)

    Returns:
        Tuple of (canonical_id1, canonical_id2)

    Examples:
        >>> reconcile_team_names("Lakers", "Boston Celtics")
        ('nba_lal', 'nba_bos')
    """
    return (normalize_team_name(team1), normalize_team_name(team2))


def get_team_abbreviation(team_name: str) -> str:
    """
    Get 3-letter abbreviation for team.

    Args:
        team_name: Any team name variant

    Returns:
        3-letter abbreviation (e.g., "LAL")
    """
    canonical_id = normalize_team_name(team_name)
    if canonical_id.startswith('nba_'):
        abbrev = canonical_id[4:].upper()
        # Handle special cases
        if abbrev == "UTAH":
            return "UTA"
        elif abbrev == "PHI":
            return "PHI"
        return abbrev
    return team_name[:3].upper()


def are_same_team(team1: str, team2: str) -> bool:
    """
    Check if two team names refer to the same team.

    Args:
        team1: First team name
        team2: Second team name

    Returns:
        True if same team, False otherwise

    Examples:
        >>> are_same_team("Lakers", "Los Angeles Lakers")
        True
        >>> are_same_team("Lakers", "Celtics")
        False
    """
    id1 = normalize_team_name(team1)
    id2 = normalize_team_name(team2)
    return id1 == id2 and id1.startswith('nba_')


# Export main functions
__all__ = [
    'normalize_team_name',
    'get_canonical_name',
    'reconcile_team_names',
    'get_team_abbreviation',
    'are_same_team',
]
