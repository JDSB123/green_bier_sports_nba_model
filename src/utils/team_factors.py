"""src.utils.team_factors

Runtime-safe team context utilities.

This module provides:
- Team-specific home court advantage constants
- Arena locations and travel distance estimation
- Timezone offsets and timezone difference
- Lightweight travel fatigue heuristic

Important: This module is intended for PRODUCTION runtime.
It does not read historical datasets from disk.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

from src.utils.team_names import get_canonical_name, normalize_team_name

# =============================================================================
# TEAM-SPECIFIC HOME COURT ADVANTAGE (HCA)
# =============================================================================

TEAM_HOME_COURT_ADVANTAGE: Dict[str, float] = {
    # High altitude advantage
    "Denver Nuggets": 4.2,
    "Utah Jazz": 3.8,
    # Strong traditional home courts
    "Boston Celtics": 3.5,
    "Golden State Warriors": 3.2,
    "Phoenix Suns": 3.0,
    "Miami Heat": 3.2,
    "Oklahoma City Thunder": 3.0,
    "Cleveland Cavaliers": 2.8,
    "Memphis Grizzlies": 2.8,
    # Above average
    "Philadelphia 76ers": 2.7,
    "Milwaukee Bucks": 2.7,
    "New York Knicks": 2.6,
    "Dallas Mavericks": 2.6,
    "Minnesota Timberwolves": 2.6,
    "Toronto Raptors": 2.8,
    "Portland Trail Blazers": 2.6,
    # League average-ish
    "Los Angeles Lakers": 2.5,
    "Los Angeles Clippers": 2.3,
    "Chicago Bulls": 2.5,
    "Indiana Pacers": 2.5,
    "Houston Rockets": 2.5,
    "New Orleans Pelicans": 2.5,
    "San Antonio Spurs": 2.5,
    "Sacramento Kings": 2.5,
    "Atlanta Hawks": 2.5,
    "Detroit Pistons": 2.3,
    "Charlotte Hornets": 2.4,
    "Orlando Magic": 2.4,
    "Washington Wizards": 2.3,
    "Brooklyn Nets": 2.2,
}

DEFAULT_HCA = 2.5


def get_home_court_advantage(team_name: str) -> float:
    """Get team-specific home court advantage in points."""
    team_name_normalized = normalize_team_name(team_name)
    team_canonical = get_canonical_name(team_name_normalized)

    if team_canonical in TEAM_HOME_COURT_ADVANTAGE:
        return TEAM_HOME_COURT_ADVANTAGE[team_canonical]

    team_lower = team_name.lower()
    for full_name, hca in TEAM_HOME_COURT_ADVANTAGE.items():
        if full_name.lower() in team_lower or team_lower in full_name.lower():
            return hca

    for full_name, hca in TEAM_HOME_COURT_ADVANTAGE.items():
        city = full_name.rsplit(" ", 1)[0].lower()
        if city in team_lower:
            return hca

    return DEFAULT_HCA


# =============================================================================
# ARENA LOCATIONS (latitude, longitude)
# =============================================================================

ARENA_LOCATIONS: Dict[str, Tuple[float, float]] = {
    "Atlanta Hawks": (33.7573, -84.3963),
    "Boston Celtics": (42.3662, -71.0621),
    "Brooklyn Nets": (40.6826, -73.9754),
    "Charlotte Hornets": (35.2251, -80.8392),
    "Chicago Bulls": (41.8807, -87.6742),
    "Cleveland Cavaliers": (41.4965, -81.6882),
    "Dallas Mavericks": (32.7905, -96.8103),
    "Denver Nuggets": (39.7487, -105.0077),
    "Detroit Pistons": (42.3410, -83.0553),
    "Golden State Warriors": (37.7680, -122.3877),
    "Houston Rockets": (29.7508, -95.3621),
    "Indiana Pacers": (39.7640, -86.1555),
    "Los Angeles Clippers": (34.0430, -118.2673),
    "Los Angeles Lakers": (34.0430, -118.2673),
    "Memphis Grizzlies": (35.1382, -90.0505),
    "Miami Heat": (25.7814, -80.1870),
    "Milwaukee Bucks": (43.0451, -87.9172),
    "Minnesota Timberwolves": (44.9795, -93.2761),
    "New Orleans Pelicans": (29.9490, -90.0821),
    "New York Knicks": (40.7505, -73.9934),
    "Oklahoma City Thunder": (35.4634, -97.5151),
    "Orlando Magic": (28.5392, -81.3839),
    "Philadelphia 76ers": (39.9012, -75.1720),
    "Phoenix Suns": (33.4457, -112.0712),
    "Portland Trail Blazers": (45.5316, -122.6668),
    "Sacramento Kings": (38.5802, -121.4997),
    "San Antonio Spurs": (29.4270, -98.4375),
    "Toronto Raptors": (43.6435, -79.3791),
    "Utah Jazz": (40.7683, -111.9011),
    "Washington Wizards": (38.8981, -77.0209),
}


# =============================================================================
# TIME ZONES (hours offset from EST)
# =============================================================================

TEAM_TIMEZONE_OFFSET: Dict[str, int] = {
    # Eastern
    "Atlanta Hawks": 0,
    "Boston Celtics": 0,
    "Brooklyn Nets": 0,
    "Charlotte Hornets": 0,
    "Cleveland Cavaliers": 0,
    "Detroit Pistons": 0,
    "Indiana Pacers": 0,
    "Miami Heat": 0,
    "New York Knicks": 0,
    "Orlando Magic": 0,
    "Philadelphia 76ers": 0,
    "Toronto Raptors": 0,
    "Washington Wizards": 0,
    # Central
    "Chicago Bulls": -1,
    "Dallas Mavericks": -1,
    "Houston Rockets": -1,
    "Memphis Grizzlies": -1,
    "Milwaukee Bucks": -1,
    "Minnesota Timberwolves": -1,
    "New Orleans Pelicans": -1,
    "Oklahoma City Thunder": -1,
    "San Antonio Spurs": -1,
    # Mountain
    "Denver Nuggets": -2,
    "Phoenix Suns": -2,
    "Utah Jazz": -2,
    # Pacific
    "Golden State Warriors": -3,
    "Los Angeles Clippers": -3,
    "Los Angeles Lakers": -3,
    "Portland Trail Blazers": -3,
    "Sacramento Kings": -3,
}


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two points on Earth (miles)."""
    r = 3959
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return r * c


def get_travel_distance(from_team: str, to_team: str) -> Optional[float]:
    """Estimate travel distance between two team arenas in miles."""
    from_normalized = normalize_team_name(from_team)
    to_normalized = normalize_team_name(to_team)

    from_canonical = get_canonical_name(from_normalized)
    to_canonical = get_canonical_name(to_normalized)

    from_loc = ARENA_LOCATIONS.get(from_canonical)
    to_loc = ARENA_LOCATIONS.get(to_canonical)

    if from_loc is None or to_loc is None:
        # Fallback: partial match heuristics
        for team_name, coords in ARENA_LOCATIONS.items():
            if team_name.lower() in from_team.lower() or from_team.lower() in team_name.lower():
                from_loc = coords
                break
        for team_name, coords in ARENA_LOCATIONS.items():
            if team_name.lower() in to_team.lower() or to_team.lower() in team_name.lower():
                to_loc = coords
                break

    if from_loc is None or to_loc is None:
        return None

    return haversine_distance(from_loc[0], from_loc[1], to_loc[0], to_loc[1])


def get_timezone_difference(team1: str, team2: str) -> int:
    """Return timezone difference in hours (positive means team1 is further west)."""
    team1_canonical = get_canonical_name(normalize_team_name(team1))
    team2_canonical = get_canonical_name(normalize_team_name(team2))

    offset1 = TEAM_TIMEZONE_OFFSET.get(team1_canonical, 0)
    offset2 = TEAM_TIMEZONE_OFFSET.get(team2_canonical, 0)
    return offset1 - offset2


def calculate_travel_fatigue(
    distance_miles: Optional[float],
    timezone_change: int,
    rest_days: int,
    is_back_to_back: bool = False,
) -> float:
    """Lightweight travel fatigue heuristic returning a 0..1-ish penalty.

    Args:
        distance_miles: Travel distance in miles.
        timezone_change: Hours of timezone change.
        rest_days: Days of rest since last game.
        is_back_to_back: Whether the team is on a back-to-back.
    """
    if distance_miles is None:
        return 0.0

    # Normalize distance to a 0..1-ish scale (cross-country ~2500 miles)
    distance_factor = min(max(distance_miles / 2500.0, 0.0), 1.0)
    timezone_factor = min(abs(timezone_change) / 3.0, 1.0)

    # Less rest -> more fatigue
    rest_factor = 1.0
    if rest_days >= 2:
        rest_factor = 0.5
    if rest_days >= 3:
        rest_factor = 0.3

    # Back-to-back amplifies fatigue.
    if is_back_to_back:
        rest_factor *= 1.25

    return (0.6 * distance_factor + 0.4 * timezone_factor) * rest_factor


def get_team_context_features(team_name: str) -> Dict[str, float]:
    """Return lightweight context features that are not dependent on historical files."""
    hca = get_home_court_advantage(team_name)
    return {
        "home_court_advantage": hca,
        "dynamic_hca": hca,
    }
