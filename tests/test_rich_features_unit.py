from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest


def _make_game(game_id: int, home_id: int, away_id: int, home_name: str, away_name: str, day: int, home_score: int, away_score: int):
    return {
        "id": game_id,
        "date": f"2026-01-{day:02d}T00:00:00Z",
        "status": {"short": "FT"},
        "teams": {
            "home": {"id": home_id, "name": home_name},
            "away": {"id": away_id, "name": away_name},
        },
        "scores": {
            "home": {"total": home_score, "quarter_1": home_score // 4, "quarter_2": home_score // 4},
            "away": {"total": away_score, "quarter_1": away_score // 4, "quarter_2": away_score // 4},
        },
    }


@pytest.mark.asyncio
async def test_rich_feature_builder_build_game_features_smoke(monkeypatch):
    from src.config import settings
    from src.features.rich_features import RichFeatureBuilder

    RichFeatureBuilder.clear_persistent_cache()

    # Keep strict split flags off for this smoke run.
    monkeypatch.setattr(settings, "require_action_network_splits", False, raising=False)
    monkeypatch.setattr(settings, "require_real_splits", True, raising=False)

    async def fetch_teams(search: str, **_kwargs):
        if "Lakers" in search:
            return {"response": [{"id": 1}]}
        if "Celtics" in search:
            return {"response": [{"id": 2}]}
        return {"response": []}

    async def fetch_statistics(team: int, **_kwargs):
        # Minimal structure used by build_game_features.
        base_for = 115.0 if team == 1 else 112.0
        base_against = 110.0 if team == 1 else 113.0
        return {
            "response": {
                "points": {
                    "for": {"average": {"all": str(base_for)}},
                    "against": {"average": {"all": str(base_against)}},
                }
            }
        }

    async def fetch_h2h(h2h: str, **_kwargs):
        # Provide one H2H game.
        return {
            "response": [
                _make_game(900, 1, 2, "Los Angeles Lakers", "Boston Celtics", 1, 120, 110)
            ]
        }

    async def fetch_standings(**_kwargs):
        return {
            "response": [
                {
                    "team": {"id": 1, "name": "Los Angeles Lakers"},
                    "games": {"win": {"total": 10}, "lose": {"total": 5}, "played": 15},
                    "position": 3,
                },
                {
                    "team": {"id": 2, "name": "Boston Celtics"},
                    "games": {"win": {"total": 9}, "lose": {"total": 6}, "played": 15},
                    "position": 4,
                },
            ]
        }

    async def fetch_games(**_kwargs):
        # Mix games so both teams get a recent history.
        games = []
        games.extend(
            [
                _make_game(100 + i, 1, 99, "Los Angeles Lakers", "Other", 8 - i, 110 + i, 100 + i)
                for i in range(6)
            ]
        )
        games.extend(
            [
                _make_game(200 + i, 98, 2, "Other", "Boston Celtics", 8 - i, 105 + i, 111 + i)
                for i in range(6)
            ]
        )
        return {"response": games}

    async def fetch_game_stats_teams(id: int, **_kwargs):
        # Simple per-game box score payload.
        return {
            "response": [
                {
                    "team": {"id": 1},
                    "field_goals": {"total": 40, "attempts": 80},
                    "threepoint_goals": {"total": 12, "attempts": 34},
                    "freethrows_goals": {"total": 18, "attempts": 22},
                    "rebounds": {"total": 44, "offence": 10, "defense": 34},
                    "assists": 25,
                    "steals": 7,
                    "blocks": 5,
                    "turnovers": 12,
                    "personal_fouls": 18,
                },
                {
                    "team": {"id": 2},
                    "field_goals": {"total": 39, "attempts": 82},
                    "threepoint_goals": {"total": 11, "attempts": 33},
                    "freethrows_goals": {"total": 17, "attempts": 23},
                    "rebounds": {"total": 43, "offence": 9, "defense": 34},
                    "assists": 24,
                    "steals": 6,
                    "blocks": 4,
                    "turnovers": 13,
                    "personal_fouls": 19,
                },
            ]
        }

    # Injuries: fetch succeeds but no injuries.
    async def fetch_all_injuries():
        return []

    async def enrich_injuries_with_stats(injuries):
        return injuries

    monkeypatch.setattr("src.features.rich_features.api_basketball.fetch_teams", fetch_teams)
    monkeypatch.setattr("src.features.rich_features.api_basketball.fetch_statistics", fetch_statistics)
    monkeypatch.setattr("src.features.rich_features.api_basketball.fetch_h2h", fetch_h2h)
    monkeypatch.setattr("src.features.rich_features.api_basketball.fetch_standings", fetch_standings)
    monkeypatch.setattr("src.features.rich_features.api_basketball.fetch_games", fetch_games)
    monkeypatch.setattr("src.features.rich_features.api_basketball.fetch_game_stats_teams", fetch_game_stats_teams)

    monkeypatch.setattr("src.ingestion.injuries.fetch_all_injuries", fetch_all_injuries)
    monkeypatch.setattr("src.ingestion.injuries.enrich_injuries_with_stats", enrich_injuries_with_stats)

    b = RichFeatureBuilder()

    features = await b.build_game_features(
        home_team="Los Angeles Lakers",
        away_team="Boston Celtics",
        game_date=dt.datetime(2026, 1, 10, tzinfo=dt.timezone.utc),
        betting_splits=None,
    )

    assert isinstance(features, dict)
    assert features["predicted_margin"]
    assert features["predicted_total"]
    assert features["has_injury_data"] in (0, 1)
    assert "home_fg_pct" in features


@pytest.mark.asyncio
async def test_rich_features_injuries_df_empty_success(monkeypatch):
    from src.features.rich_features import RichFeatureBuilder

    async def fetch_all_injuries():
        return []

    async def enrich_injuries_with_stats(injuries):
        return injuries

    monkeypatch.setattr("src.ingestion.injuries.fetch_all_injuries", fetch_all_injuries)
    monkeypatch.setattr("src.ingestion.injuries.enrich_injuries_with_stats", enrich_injuries_with_stats)

    b = RichFeatureBuilder()
    df = await b.get_injuries_df()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns)
    assert len(df) == 0


@pytest.mark.asyncio
async def test_rich_features_injuries_df_strict_failure(monkeypatch):
    from src.config import settings
    from src.features.rich_features import RichFeatureBuilder

    monkeypatch.setattr(settings, "require_injury_fetch_success", True, raising=False)

    async def boom():
        raise RuntimeError("injuries down")

    monkeypatch.setattr("src.ingestion.injuries.fetch_all_injuries", boom)

    b = RichFeatureBuilder()
    with pytest.raises(ValueError):
        await b.get_injuries_df()


@pytest.mark.asyncio
async def test_rich_features_strict_splits_enforced(monkeypatch):
    import datetime as dt

    from src.config import settings
    from src.features.rich_features import RichFeatureBuilder
    from src.ingestion.betting_splits import GameSplits

    # Minimal stubs so we can reach the splits enforcement branch without pulling real APIs.
    async def fetch_teams(search: str, **_kwargs):
        return {"response": [{"id": 1 if "Lakers" in search else 2}]}

    async def fetch_statistics(team: int, **_kwargs):
        return {
            "response": {
                "points": {
                    "for": {"average": {"all": "115.0"}},
                    "against": {"average": {"all": "110.0"}},
                }
            }
        }

    async def fetch_h2h(**_kwargs):
        return {"response": []}

    async def fetch_standings(**_kwargs):
        return {
            "response": [
                {
                    "team": {"id": 1, "name": "Los Angeles Lakers"},
                    "games": {"win": {"total": 10}, "lose": {"total": 5}, "played": 15},
                    "position": 3,
                },
                {
                    "team": {"id": 2, "name": "Boston Celtics"},
                    "games": {"win": {"total": 9}, "lose": {"total": 6}, "played": 15},
                    "position": 4,
                },
            ]
        }

    async def fetch_games(**_kwargs):
        return {
            "response": [
                _make_game(101, 1, 99, "Los Angeles Lakers", "Other", 8, 110, 105),
                _make_game(102, 99, 2, "Other", "Boston Celtics", 8, 108, 112),
            ]
        }

    async def fetch_game_stats_teams(**_kwargs):
        return {"response": []}

    async def fetch_all_injuries():
        return []

    async def enrich_injuries_with_stats(injuries):
        return injuries

    monkeypatch.setattr("src.features.rich_features.api_basketball.fetch_teams", fetch_teams)
    monkeypatch.setattr("src.features.rich_features.api_basketball.fetch_statistics", fetch_statistics)
    monkeypatch.setattr("src.features.rich_features.api_basketball.fetch_h2h", fetch_h2h)
    monkeypatch.setattr("src.features.rich_features.api_basketball.fetch_standings", fetch_standings)
    monkeypatch.setattr("src.features.rich_features.api_basketball.fetch_games", fetch_games)
    monkeypatch.setattr("src.features.rich_features.api_basketball.fetch_game_stats_teams", fetch_game_stats_teams)

    monkeypatch.setattr("src.ingestion.injuries.fetch_all_injuries", fetch_all_injuries)
    monkeypatch.setattr("src.ingestion.injuries.enrich_injuries_with_stats", enrich_injuries_with_stats)

    b = RichFeatureBuilder()

    monkeypatch.setattr(settings, "require_action_network_splits", True, raising=False)
    monkeypatch.setattr(settings, "require_real_splits", False, raising=False)

    bad_splits = GameSplits(
        event_id="1",
        home_team="Los Angeles Lakers",
        away_team="Boston Celtics",
        game_time=dt.datetime.now(dt.timezone.utc),
        source="the_odds",
        spread_home_ticket_pct=60.0,
        spread_home_money_pct=40.0,
        over_ticket_pct=55.0,
    )

    with pytest.raises(ValueError):
        await b.build_game_features("Los Angeles Lakers", "Boston Celtics", betting_splits=bad_splits)
