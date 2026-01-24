from __future__ import annotations

from datetime import date

import pytest


def test_parse_utc_time_rounding():
    from src.utils.slate_analysis import parse_utc_time

    dt = parse_utc_time("2026-01-01T10:02:11Z")
    assert dt.tzinfo is not None
    assert dt.minute == 0
    assert dt.second == 0

    dt2 = parse_utc_time("2026-01-01T10:58:00Z")
    assert dt2.minute == 0
    assert dt2.hour == 11


def test_extract_consensus_odds_medians_and_last_update():
    from src.utils.slate_analysis import extract_consensus_odds

    game = {
        "home_team": "Los Angeles Lakers",
        "away_team": "Boston Celtics",
        "last_update": "2026-01-01T00:00:00Z",
        "bookmakers": [
            {
                "key": "a",
                "last_update": "2026-01-01T00:05:00Z",
                "markets": [
                    {
                        "key": "spreads",
                        "outcomes": [
                            {"name": "Los Angeles Lakers", "price": -110, "point": -5.0},
                            {"name": "Boston Celtics", "price": -110, "point": 5.0},
                        ],
                    },
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "price": -105, "point": 230.5},
                            {"name": "Under", "price": -115, "point": 230.5},
                        ],
                    },
                ],
            },
            {
                "key": "b",
                "last_update": "2026-01-01T00:10:00Z",
                "markets": [
                    {
                        "key": "spreads_h1",
                        "outcomes": [
                            {"name": "Los Angeles Lakers", "price": -110, "point": -2.5},
                            {"name": "Boston Celtics", "price": -110, "point": 2.5},
                        ],
                    },
                    {
                        "key": "totals_h1",
                        "outcomes": [
                            {"name": "Over", "price": -110, "point": 115.0},
                            {"name": "Under", "price": -110, "point": 115.0},
                            # Invalid 1H total should be dropped by sanity bounds
                            {"name": "Over", "price": -110, "point": 200.0},
                        ],
                    },
                ],
            },
        ],
    }

    odds = extract_consensus_odds(game, as_of_utc="2026-01-01T00:00:00Z")
    assert odds["home_spread"] == -5.0
    assert odds["total"] == 230.5
    assert odds["fh_home_spread"] == -2.5
    assert odds["fh_total"] == 115.0
    assert odds["last_update_utc"] == "2026-01-01T00:10:00Z"


def test_lookup_team_record_with_synonyms_and_invalid_name():
    from src.utils.slate_analysis import _lookup_team_record_with_synonyms

    records = {
        "Los Angeles Clippers": {"wins": 10, "losses": 5, "games_played": 15},
        "Los Angeles Lakers": {"wins": 8, "losses": 7, "games_played": 15},
    }

    rec = _lookup_team_record_with_synonyms("LA Clippers", records)
    assert rec["wins"] == 10

    with pytest.raises(ValueError):
        _lookup_team_record_with_synonyms("Not A Real Team", records)


def test_validate_data_integrity_detects_missing():
    import asyncio

    from src.utils.slate_analysis import validate_data_integrity

    result = asyncio.run(
        validate_data_integrity(
            odds_teams=["A", "B"],
            record_teams=["A"],
            records_source="espn",
        )
    )
    assert result["is_valid"] is False
    assert "B" in result["missing_from_records"]
