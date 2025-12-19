"""
Dynamic edge threshold calculation based on season sample size.

Early in the season, we should be more conservative with edge thresholds
as we have less data to validate model performance.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Dict


def calculate_dynamic_edge_threshold(
    game_date: date,
    season_start_date: date,
    bet_type: str,
    base_threshold: float = 2.0,
) -> float:
    """
    Calculate dynamic edge threshold based on how far into the season we are.
    
    Args:
        game_date: Date of the game
        season_start_date: Start date of the NBA season (typically mid-October)
        bet_type: Type of bet ("spread", "total", "moneyline", "1h_spread", "1h_total")
        base_threshold: Base threshold in points (default 2.0 for spreads)
        
    Returns:
        Adjusted edge threshold
    """
    # Calculate days into season
    days_into_season = (game_date - season_start_date).days
    
    # Base thresholds by bet type
    base_thresholds = {
        "spread": 2.0,
        "total": 3.0,
        "moneyline": 0.03,  # 3% probability edge
        "1h_spread": 1.5,
        "1h_total": 2.0,
    }
    
    if bet_type in base_thresholds:
        base = base_thresholds[bet_type]
    else:
        base = base_threshold
    
    # Early season (first 30 days): More conservative (+50% threshold)
    if days_into_season < 30:
        multiplier = 1.5
    # Early-mid season (30-60 days): Moderately conservative (+25% threshold)
    elif days_into_season < 60:
        multiplier = 1.25
    # Mid season (60-120 days): Standard threshold
    elif days_into_season < 120:
        multiplier = 1.0
    # Late season (120+ days): Slightly more aggressive (-10% threshold)
    else:
        multiplier = 0.9
    
    # For probability-based thresholds (moneyline), adjust differently
    if bet_type == "moneyline":
        # Probability thresholds: early season = +0.01 (1%), late season = -0.005 (0.5%)
        if days_into_season < 30:
            adjustment = 0.01
        elif days_into_season < 60:
            adjustment = 0.005
        elif days_into_season < 120:
            adjustment = 0.0
        else:
            adjustment = -0.005
        
        return base + adjustment
    
    # For point-based thresholds (spreads, totals)
    return base * multiplier


def get_season_start_date(season_year: int) -> date:
    """
    Get the start date of an NBA season.
    
    Args:
        season_year: Year the season starts (e.g., 2024 for 2024-25 season)
        
    Returns:
        Season start date (typically mid-October)
    """
    # NBA season typically starts around October 15-20
    return date(season_year, 10, 15)


def get_current_season_start() -> date:
    """Get the start date of the current NBA season."""
    today = date.today()
    # If we're before October, season started last year
    if today.month < 10:
        return date(today.year - 1, 10, 15)
    else:
        return date(today.year, 10, 15)


def get_edge_thresholds_for_game(
    game_date: date,
    bet_types: list[str],
) -> Dict[str, float]:
    """
    Get edge thresholds for multiple bet types for a game.
    
    Args:
        game_date: Date of the game
        bet_types: List of bet types to get thresholds for
        
    Returns:
        Dict mapping bet_type -> threshold
    """
    season_start = get_current_season_start()
    
    thresholds = {}
    for bet_type in bet_types:
        thresholds[bet_type] = calculate_dynamic_edge_threshold(
            game_date=game_date,
            season_start_date=season_start,
            bet_type=bet_type,
        )
    
    return thresholds
