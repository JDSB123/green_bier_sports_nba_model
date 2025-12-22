from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd

from src.modeling.team_factors import (
    calculate_travel_fatigue,
    get_home_court_advantage,
    get_timezone_difference,
    get_travel_distance,
)
from src.utils.team_names import normalize_team_name
TRAVEL_FEATURE_COLUMNS: Iterable[str] = (
    "away_travel_distance",
    "away_timezone_change",
    "away_travel_fatigue",
    "is_away_long_trip",
    "is_away_cross_country",
    "away_b2b_travel_penalty",
    "travel_advantage",
    "home_court_advantage",
)


def _safe_int(value: float | int | None, default: int = 3) -> int:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    return int(value)


def augment_travel_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure travel-related modeling features exist on the provided DataFrame.

    The function keeps the original row order, but internally walks games in
    chronological order so that previous locations are accurate.
    """
    if df.empty:
        for col in TRAVEL_FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = []
        return df

    required_cols = {"date", "home_team", "away_team"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"augment_travel_features requires columns {missing}")

    working = df.copy()
    working["_orig_order"] = np.arange(len(working))
    working["date"] = pd.to_datetime(working["date"], errors="coerce")

    for col in TRAVEL_FEATURE_COLUMNS:
        if col not in working.columns:
            working[col] = 0.0

    previous_locations: Dict[str, str] = {}

    for idx in working.sort_values("date").index:
        row = working.loc[idx]
        home_team = row["home_team"]
        away_team = row["away_team"]

        normalized_home = normalize_team_name(home_team)
        normalized_away = normalize_team_name(away_team)

        last_location = previous_locations.get(normalized_away)
        rest_days = _safe_int(row.get("away_rest_days"), default=3)
        is_b2b = bool(row.get("away_b2b", 0)) or rest_days <= 1

        if last_location:
            distance = get_travel_distance(last_location, normalized_home) or 0.0
            tz_change = get_timezone_difference(last_location, normalized_home)
        else:
            distance = get_travel_distance(normalized_away, normalized_home) or 0.0
            tz_change = get_timezone_difference(normalized_away, normalized_home)

        is_long_trip = int(distance >= 1500)
        is_cross_country = int(distance >= 2500)
        travel_fatigue = calculate_travel_fatigue(
            distance_miles=distance,
            rest_days=rest_days,
            timezone_change=tz_change,
            is_back_to_back=is_b2b,
        )

        penalty = -1.5 if is_b2b and distance >= 1500 else 0.0
        home_hca = get_home_court_advantage(normalized_home)

        working.at[idx, "away_travel_distance"] = float(distance)
        working.at[idx, "away_timezone_change"] = int(tz_change)
        working.at[idx, "away_travel_fatigue"] = float(travel_fatigue)
        working.at[idx, "is_away_long_trip"] = is_long_trip
        working.at[idx, "is_away_cross_country"] = is_cross_country
        working.at[idx, "away_b2b_travel_penalty"] = penalty
        working.at[idx, "travel_advantage"] = -float(travel_fatigue)
        working.at[idx, "home_court_advantage"] = float(home_hca)

        # Update previous locations for future games
        previous_locations[normalized_home] = normalized_home
        previous_locations[normalized_away] = normalize_team_name(home_team)

    working = working.sort_values("_orig_order").drop(columns="_orig_order")
    return working







