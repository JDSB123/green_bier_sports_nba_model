"""
Team-specific factors for NBA prediction models.

Contains:
- Home court advantage by team (HCA)
- Arena locations for travel distance calculations
- Time zone data
- Historical performance factors

Data sourced from:
- NBA research on home court advantage
- Arena coordinates from official NBA data
- Historical performance analysis
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple
from src.utils.team_names import normalize_team_name

# Team name normalization is now provided by single source:
# src.utils.team_names.normalize_team_name (line 63)
#
# This ensures consistent handling of team shorthand names across
# all modules in the pipeline.

# =============================================================================
# TEAM-SPECIFIC HOME COURT ADVANTAGE (HCA)
# =============================================================================
# 
# League average HCA is ~2.5 points. Some teams have significantly
# stronger/weaker home court effects due to:
# - Altitude (Denver is 5,280 ft - visiting teams fatigue faster)
# - Arena acoustics and crowd intensity
# - Historical performance patterns
# - Travel burden on opponents (western teams)
#
# Values are in POINTS added to home team's expected margin
# =============================================================================

TEAM_HOME_COURT_ADVANTAGE: Dict[str, float] = {
    # High altitude advantage
    "Denver Nuggets": 4.2,       # Altitude + strong home crowd, historically elite HCA
    "Utah Jazz": 3.8,            # Altitude (~4,300 ft) + hostile crowd
    
    # Strong traditional home courts
    "Boston Celtics": 3.5,       # Historic arena, loud crowd
    "Golden State Warriors": 3.2, # Chase Center loud, travel burden for opponents
    "Phoenix Suns": 3.0,         # Desert heat factor + travel burden
    "Miami Heat": 3.2,           # Culture, humidity acclimation
    "Oklahoma City Thunder": 3.0, # Loudest arena, travel burden
    "Cleveland Cavaliers": 2.8,  # Loud crowd
    "Memphis Grizzlies": 2.8,    # FedEx Forum known as one of loudest
    
    # Above average
    "Philadelphia 76ers": 2.7,
    "Milwaukee Bucks": 2.7,
    "New York Knicks": 2.6,      # MSG atmosphere
    "Dallas Mavericks": 2.6,
    "Minnesota Timberwolves": 2.6, # Northern location = travel factor
    "Toronto Raptors": 2.8,      # International travel + crowd
    "Portland Trail Blazers": 2.6, # Remote location + loyal fans
    
    # League average
    "Los Angeles Lakers": 2.5,   # Great team but lots of away fans at games
    "Los Angeles Clippers": 2.3, # Share arena, less intense home crowd
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
    "Brooklyn Nets": 2.2,        # Transplant city, many away fans
}

# Default HCA for unknown teams
DEFAULT_HCA = 2.5


def get_home_court_advantage(team_name: str) -> float:
    """
    Get team-specific home court advantage in points.
    
    Args:
        team_name: Full team name (e.g., "Denver Nuggets")
        
    Returns:
        Points of home court advantage for this team
    """
    team_name_normalized = normalize_team_name(team_name)
    from src.utils.team_names import get_canonical_name
    team_canonical = get_canonical_name(team_name_normalized)

    # Try exact match first
    if team_canonical in TEAM_HOME_COURT_ADVANTAGE:
        return TEAM_HOME_COURT_ADVANTAGE[team_canonical]
    
    # Try partial match (for variations like "LA Lakers")
    team_lower = team_name.lower()
    for full_name, hca in TEAM_HOME_COURT_ADVANTAGE.items():
        if full_name.lower() in team_lower or team_lower in full_name.lower():
            return hca
    
    # Check for city name matches
    for full_name, hca in TEAM_HOME_COURT_ADVANTAGE.items():
        city = full_name.rsplit(" ", 1)[0].lower()  # Get city part
        if city in team_lower:
            return hca
    
    return DEFAULT_HCA


# =============================================================================
# ARENA LOCATIONS (latitude, longitude)
# =============================================================================
# Used to calculate travel distance between games

ARENA_LOCATIONS: Dict[str, Tuple[float, float]] = {
    "Atlanta Hawks": (33.7573, -84.3963),           # State Farm Arena
    "Boston Celtics": (42.3662, -71.0621),          # TD Garden
    "Brooklyn Nets": (40.6826, -73.9754),           # Barclays Center
    "Charlotte Hornets": (35.2251, -80.8392),       # Spectrum Center
    "Chicago Bulls": (41.8807, -87.6742),           # United Center
    "Cleveland Cavaliers": (41.4965, -81.6882),     # Rocket Mortgage FieldHouse
    "Dallas Mavericks": (32.7905, -96.8103),        # American Airlines Center
    "Denver Nuggets": (39.7487, -105.0077),         # Ball Arena
    "Detroit Pistons": (42.3410, -83.0553),         # Little Caesars Arena
    "Golden State Warriors": (37.7680, -122.3877),  # Chase Center
    "Houston Rockets": (29.7508, -95.3621),         # Toyota Center
    "Indiana Pacers": (39.7640, -86.1555),          # Gainbridge Fieldhouse
    "Los Angeles Clippers": (34.0430, -118.2673),   # Crypto.com Arena (shared)
    "Los Angeles Lakers": (34.0430, -118.2673),     # Crypto.com Arena
    "Memphis Grizzlies": (35.1382, -90.0505),       # FedExForum
    "Miami Heat": (25.7814, -80.1870),              # Kaseya Center
    "Milwaukee Bucks": (43.0451, -87.9172),         # Fiserv Forum
    "Minnesota Timberwolves": (44.9795, -93.2761),  # Target Center
    "New Orleans Pelicans": (29.9490, -90.0821),    # Smoothie King Center
    "New York Knicks": (40.7505, -73.9934),         # Madison Square Garden
    "Oklahoma City Thunder": (35.4634, -97.5151),   # Paycom Center
    "Orlando Magic": (28.5392, -81.3839),           # Amway Center
    "Philadelphia 76ers": (39.9012, -75.1720),      # Wells Fargo Center
    "Phoenix Suns": (33.4457, -112.0712),           # Footprint Center
    "Portland Trail Blazers": (45.5316, -122.6668), # Moda Center
    "Sacramento Kings": (38.5802, -121.4997),       # Golden 1 Center
    "San Antonio Spurs": (29.4270, -98.4375),       # Frost Bank Center
    "Toronto Raptors": (43.6435, -79.3791),         # Scotiabank Arena
    "Utah Jazz": (40.7683, -111.9011),              # Delta Center
    "Washington Wizards": (38.8981, -77.0209),      # Capital One Arena
}


# =============================================================================
# TIME ZONES (hours offset from EST)
# =============================================================================

TEAM_TIMEZONE_OFFSET: Dict[str, int] = {
    # Eastern (0 offset from EST)
    "Atlanta Hawks": 0, "Boston Celtics": 0, "Brooklyn Nets": 0,
    "Charlotte Hornets": 0, "Cleveland Cavaliers": 0, "Detroit Pistons": 0,
    "Indiana Pacers": 0, "Miami Heat": 0, "New York Knicks": 0,
    "Orlando Magic": 0, "Philadelphia 76ers": 0, "Toronto Raptors": 0,
    "Washington Wizards": 0,
    
    # Central (-1 from EST)
    "Chicago Bulls": -1, "Dallas Mavericks": -1, "Houston Rockets": -1,
    "Memphis Grizzlies": -1, "Milwaukee Bucks": -1, "Minnesota Timberwolves": -1,
    "New Orleans Pelicans": -1, "Oklahoma City Thunder": -1, "San Antonio Spurs": -1,
    
    # Mountain (-2 from EST)
    "Denver Nuggets": -2, "Phoenix Suns": -2, "Utah Jazz": -2,
    
    # Pacific (-3 from EST)
    "Golden State Warriors": -3, "Los Angeles Clippers": -3, 
    "Los Angeles Lakers": -3, "Portland Trail Blazers": -3, 
    "Sacramento Kings": -3,
}


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great-circle distance between two points on Earth.
    
    Args:
        lat1, lon1: Coordinates of first point (degrees)
        lat2, lon2: Coordinates of second point (degrees)
        
    Returns:
        Distance in miles
    """
    R = 3959  # Earth's radius in miles
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c


def get_travel_distance(from_team: str, to_team: str) -> Optional[float]:
    """
    Calculate travel distance between two team arenas.
    
    Args:
        from_team: Team traveling from
        to_team: Team traveling to
        
    Returns:
        Distance in miles, or None if coordinates not found
    """
    from_normalized = normalize_team_name(from_team)
    to_normalized = normalize_team_name(to_team)
    
    # Convert team IDs to full canonical names for lookup
    from src.utils.team_names import get_canonical_name
    from_canonical = get_canonical_name(from_normalized)
    to_canonical = get_canonical_name(to_normalized)

    from_loc = ARENA_LOCATIONS.get(from_canonical)
    to_loc = ARENA_LOCATIONS.get(to_canonical)
    
    # Find from location
    for team_name, coords in ARENA_LOCATIONS.items():
        if team_name.lower() in from_team.lower() or from_team.lower() in team_name.lower():
            from_loc = coords
            break
    
    # Find to location
    for team_name, coords in ARENA_LOCATIONS.items():
        if team_name.lower() in to_team.lower() or to_team.lower() in team_name.lower():
            to_loc = coords
            break
    
    if from_loc is None or to_loc is None:
        return None
    
    return haversine_distance(from_loc[0], from_loc[1], to_loc[0], to_loc[1])


def get_timezone_difference(team1: str, team2: str) -> int:
    """
    Get timezone difference between two teams.
    
    Returns:
        Difference in hours (positive = team1 is further west)
    """
    team1_normalized = normalize_team_name(team1)
    team2_normalized = normalize_team_name(team2)

    from src.utils.team_names import get_canonical_name
    team1_canonical = get_canonical_name(team1_normalized)
    team2_canonical = get_canonical_name(team2_normalized)
    
    tz1 = TEAM_TIMEZONE_OFFSET.get(team1_canonical, 0)
    tz2 = TEAM_TIMEZONE_OFFSET.get(team2_canonical, 0)
    
    if tz1 == 0:
        for team_name, offset in TEAM_TIMEZONE_OFFSET.items():
            if team_name.lower() in team1_normalized.lower() or team1_normalized.lower() in team_name.lower():
                tz1 = offset
                break
    
    if tz2 == 0:
        for team_name, offset in TEAM_TIMEZONE_OFFSET.items():
            if team_name.lower() in team2_normalized.lower() or team2_normalized.lower() in team_name.lower():
                tz2 = offset
                break
    
    return tz1 - tz2


# =============================================================================
# TRAVEL/REST INTERACTION FACTORS
# =============================================================================

def calculate_travel_fatigue(
    distance_miles: float,
    rest_days: int,
    timezone_change: int = 0,
    is_back_to_back: bool = False,
) -> float:
    """
    Calculate travel fatigue adjustment in points.
    
    Research shows:
    - Long travel (>1500 miles) = -1 to -2 pts
    - B2B + long travel = compounding effect (-3 to -4 pts)
    - Timezone changes (especially east-to-west late games) = additional fatigue
    
    Args:
        distance_miles: Travel distance
        rest_days: Days since last game
        timezone_change: Hours of timezone change (negative = traveling west)
        is_back_to_back: True if playing on consecutive days
        
    Returns:
        Point adjustment (negative = fatigue penalty)
    """
    fatigue = 0.0
    
    # Base travel fatigue
    if distance_miles >= 2500:
        fatigue -= 1.5  # Cross-country flight
    elif distance_miles >= 1500:
        fatigue -= 1.0  # Long flight
    elif distance_miles >= 800:
        fatigue -= 0.5  # Medium flight
    # Short trips (<800 miles) have minimal impact
    
    # Timezone adjustment (jet lag)
    # East-to-west travel is harder on NBA schedule (late body clock)
    if timezone_change < 0:  # Traveling west
        fatigue -= abs(timezone_change) * 0.3
    elif timezone_change > 0:  # Traveling east
        fatigue -= abs(timezone_change) * 0.2
    
    # NOTE: B2B penalty is already handled in rest_adjustment() (-2.5 pts)
    # We do NOT multiply travel fatigue by B2B factor to avoid double-counting
    # The rest_adjustment and travel_fatigue are separate, additive factors:
    # - rest_adjustment: handles game frequency (B2B = -2.5 pts)
    # - travel_fatigue: handles distance/timezone only

    # Rest can mitigate travel fatigue
    if rest_days >= 2:
        fatigue *= 0.5  # Well-rested team handles travel better

    return fatigue


def get_travel_features(
    team: str,
    opponent: str,
    previous_opponent: Optional[str],
    rest_days: int,
) -> Dict[str, float]:
    """
    Calculate travel-related features for a team.
    
    Args:
        team: Team playing
        opponent: Current opponent (for game location)
        previous_opponent: Last game opponent (for travel calculation)
        rest_days: Days since last game
        
    Returns:
        Dict of travel features
    """
    features = {
        "travel_distance": 0.0,
        "timezone_change": 0,
        "travel_fatigue_adj": 0.0,
        "is_long_trip": 0,
        "is_cross_country": 0,
    }
    
    if previous_opponent is None:
        return features
    
    # Calculate travel from previous game location to current game
    # If playing at home, travel is from previous opponent's arena to home
    # If playing away, travel is from previous location to opponent's arena
    
    # Simplified: assume travel from previous opponent's city to current opponent's city
    distance = get_travel_distance(previous_opponent, opponent)
    
    if distance is not None:
        features["travel_distance"] = distance
        features["is_long_trip"] = 1 if distance >= 1500 else 0
        features["is_cross_country"] = 1 if distance >= 2500 else 0
        
        tz_change = get_timezone_difference(previous_opponent, opponent)
        features["timezone_change"] = tz_change
        
        features["travel_fatigue_adj"] = calculate_travel_fatigue(
            distance_miles=distance,
            rest_days=rest_days,
            timezone_change=tz_change,
            is_back_to_back=(rest_days == 0 or rest_days == 1),
        )
    
    return features


# =============================================================================
# COMBINED FACTORS FOR PREDICTION
# =============================================================================

def get_team_context_features(
    home_team: str,
    away_team: str,
    home_rest_days: int = 3,
    away_rest_days: int = 3,
    home_previous_opponent: Optional[str] = None,
    away_previous_opponent: Optional[str] = None,
) -> Dict[str, float]:
    """
    Get comprehensive team context features including HCA and travel.
    
    Args:
        home_team: Home team name
        away_team: Away team name
        home_rest_days: Days rest for home team
        away_rest_days: Days rest for away team
        home_previous_opponent: Home team's last opponent
        away_previous_opponent: Away team's last opponent
        
    Returns:
        Dict of context features
    """
    features = {}
    
    # Home court advantage
    features["home_court_advantage"] = get_home_court_advantage(home_team)
    
    # Away team travel (they're traveling to home team's arena)
    away_travel = get_travel_features(
        team=away_team,
        opponent=home_team,
        previous_opponent=away_previous_opponent,
        rest_days=away_rest_days,
    )
    
    features["away_travel_distance"] = away_travel["travel_distance"]
    features["away_travel_fatigue"] = away_travel["travel_fatigue_adj"]
    features["away_timezone_change"] = away_travel["timezone_change"]
    features["away_is_long_trip"] = away_travel["is_long_trip"]
    
    # Home team might also have travel fatigue if coming off road trip
    home_travel = get_travel_features(
        team=home_team,
        opponent=home_team,  # Home game
        previous_opponent=home_previous_opponent,
        rest_days=home_rest_days,
    )
    
    features["home_travel_fatigue"] = home_travel["travel_fatigue_adj"]
    
    # Combined travel differential (positive = home advantage from away fatigue)
    features["travel_advantage"] = (
        -away_travel["travel_fatigue_adj"]  # Away team penalty helps home
        - home_travel["travel_fatigue_adj"]  # Home team fatigue hurts them
    )
    
    # B2B + travel interaction
    features["away_b2b_travel_penalty"] = 0.0
    if away_rest_days <= 1 and away_travel["is_long_trip"]:
        features["away_b2b_travel_penalty"] = -1.5  # Additional penalty
    
    return features


