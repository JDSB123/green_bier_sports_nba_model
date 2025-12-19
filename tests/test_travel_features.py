import pandas as pd

from src.modeling.travel import augment_travel_features
from src.modeling.team_factors import (
    get_home_court_advantage,
    get_travel_distance,
)


def test_augment_travel_features_adds_expected_columns():
    df = pd.DataFrame({
        "date": [pd.Timestamp("2023-10-25"), pd.Timestamp("2023-10-26")],
        "home_team": ["Bucks", "Bulls"],
        "away_team": ["Celtics", "Celtics"],
        "home_rest_days": [3, 2],
        "away_rest_days": [3, 0],
        "home_b2b": [0, 0],
        "away_b2b": [0, 1],
    })

    augmented = augment_travel_features(df)

    assert "away_travel_distance" in augmented.columns
    assert "home_court_advantage" in augmented.columns
    # Second game should have non-zero travel since Celtics just flew from Milwaukee to Chicago
    assert augmented.loc[1, "away_travel_distance"] > 0
    assert augmented.loc[1, "away_b2b_travel_penalty"] <= 0
    assert augmented.loc[0, "home_court_advantage"] == get_home_court_advantage("Bucks")


def test_team_aliases_map_to_canonical_names():
    hca_alias = get_home_court_advantage("Trailblazers")
    hca_canonical = get_home_court_advantage("Portland Trail Blazers")
    assert hca_alias == hca_canonical
    assert get_travel_distance("Trailblazers", "Lakers") > 0







