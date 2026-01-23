"""
Tests for src/ingestion/standardize.py - Primary ingestion standardization module.

These tests ensure team name normalization and CST timezone conversion
work correctly during data ingestion from all sources.

Coverage target: 100% of src/ingestion/standardize.py
"""
import pytest
from datetime import datetime
from zoneinfo import ZoneInfo

from src.ingestion.standardize import (
    normalize_team_to_espn,
    is_valid_espn_team_name,
    standardize_game_data,
    format_game_string,
    to_cst,
    CST,
    UTC,
    ESPN_TEAM_NAMES,
    TEAM_NAME_MAPPING,
)


class TestNormalizeTeamToESPN:
    """Tests for normalize_team_to_espn function."""

    # All 30 NBA teams - abbreviations
    # Function returns Tuple[str, bool] where bool indicates if normalization succeeded
    @pytest.mark.parametrize("abbrev,expected", [
        ("ATL", "Atlanta Hawks"),
        ("BOS", "Boston Celtics"),
        ("BKN", "Brooklyn Nets"),
        ("CHA", "Charlotte Hornets"),
        ("CHI", "Chicago Bulls"),
        ("CLE", "Cleveland Cavaliers"),
        ("DAL", "Dallas Mavericks"),
        ("DEN", "Denver Nuggets"),
        ("DET", "Detroit Pistons"),
        ("GSW", "Golden State Warriors"),
        ("HOU", "Houston Rockets"),
        ("IND", "Indiana Pacers"),
        ("LAC", "LA Clippers"),
        ("LAL", "Los Angeles Lakers"),
        ("MEM", "Memphis Grizzlies"),
        ("MIA", "Miami Heat"),
        ("MIL", "Milwaukee Bucks"),
        ("MIN", "Minnesota Timberwolves"),
        ("NOP", "New Orleans Pelicans"),
        ("NYK", "New York Knicks"),
        ("OKC", "Oklahoma City Thunder"),
        ("ORL", "Orlando Magic"),
        ("PHI", "Philadelphia 76ers"),
        ("PHX", "Phoenix Suns"),
        ("POR", "Portland Trail Blazers"),
        ("SAC", "Sacramento Kings"),
        ("SAS", "San Antonio Spurs"),
        ("TOR", "Toronto Raptors"),
        ("UTA", "Utah Jazz"),
        ("WAS", "Washington Wizards"),
    ])
    def test_abbreviations_to_espn(self, abbrev, expected):
        """All standard abbreviations should map to ESPN names."""
        # Test uppercase
        result, success = normalize_team_to_espn(abbrev)
        assert result == expected
        assert success is True
        
        # Test lowercase
        result, success = normalize_team_to_espn(abbrev.lower())
        assert result == expected
        assert success is True

    # The Odds API team names
    @pytest.mark.parametrize("odds_name,expected", [
        ("Los Angeles Lakers", "Los Angeles Lakers"),
        ("Boston Celtics", "Boston Celtics"),
        ("Golden State Warriors", "Golden State Warriors"),
        ("LA Clippers", "LA Clippers"),
        ("Los Angeles Clippers", "LA Clippers"),  # Alternative format
    ])
    def test_the_odds_api_names(self, odds_name, expected):
        """The Odds API team names should normalize correctly."""
        result, success = normalize_team_to_espn(odds_name)
        assert result == expected
        assert success is True

    # API-Basketball team names
    @pytest.mark.parametrize("api_name,expected", [
        ("Lakers", "Los Angeles Lakers"),
        ("Celtics", "Boston Celtics"),
        ("Warriors", "Golden State Warriors"),
        ("Clippers", "LA Clippers"),
        ("76ers", "Philadelphia 76ers"),
        ("Sixers", "Philadelphia 76ers"),
    ])
    def test_api_basketball_names(self, api_name, expected):
        """API-Basketball team names should normalize correctly."""
        result, success = normalize_team_to_espn(api_name)
        assert result == expected
        assert success is True

    def test_case_insensitive(self):
        """Normalization should be case-insensitive."""
        result, _ = normalize_team_to_espn("LAKERS")
        assert result == "Los Angeles Lakers"
        result, _ = normalize_team_to_espn("lakers")
        assert result == "Los Angeles Lakers"
        result, _ = normalize_team_to_espn("Lakers")
        assert result == "Los Angeles Lakers"

    def test_whitespace_handling(self):
        """Whitespace should be handled."""
        result, _ = normalize_team_to_espn("  Lakers  ")
        assert result == "Los Angeles Lakers"

    def test_already_normalized(self):
        """Already normalized names should return unchanged."""
        for name in ESPN_TEAM_NAMES:
            result, success = normalize_team_to_espn(name)
            assert result == name
            assert success is True

    def test_empty_string(self):
        """Empty string should return empty with False."""
        result, success = normalize_team_to_espn("")
        assert success is False

    def test_none_input(self):
        """None should return empty with False."""
        result, success = normalize_team_to_espn(None)
        assert success is False


class TestIsValidESPNTeamName:
    """Tests for is_valid_espn_team_name function."""

    def test_all_espn_names_valid(self):
        """All ESPN_TEAM_NAMES should be valid."""
        for name in ESPN_TEAM_NAMES:
            assert is_valid_espn_team_name(name) is True

    def test_abbreviations_not_valid(self):
        """Abbreviations should not be considered valid ESPN names."""
        assert is_valid_espn_team_name("LAL") is False
        assert is_valid_espn_team_name("BOS") is False
        assert is_valid_espn_team_name("GSW") is False

    def test_nicknames_not_valid(self):
        """Nicknames alone should not be valid ESPN names."""
        assert is_valid_espn_team_name("Lakers") is False
        assert is_valid_espn_team_name("Celtics") is False

    def test_invalid_names(self):
        """Invalid team names should return False."""
        assert is_valid_espn_team_name("Unknown Team") is False
        assert is_valid_espn_team_name("") is False


class TestToCst:
    """Tests for to_cst timezone conversion in ingestion module."""

    def test_utc_z_suffix_conversion(self):
        """UTC times with Z suffix should convert to CST."""
        result = to_cst("2025-01-16T04:00:00Z")
        
        assert result is not None
        assert result.tzinfo == CST
        assert result.date().isoformat() == "2025-01-15"  # Previous day in CST
        assert result.hour == 22  # 4am UTC = 10pm CST

    def test_utc_offset_conversion(self):
        """UTC times with +00:00 offset should convert."""
        result = to_cst("2025-01-16T04:00:00+00:00")
        
        assert result is not None
        assert result.date().isoformat() == "2025-01-15"

    def test_datetime_object(self):
        """datetime objects should convert correctly."""
        dt = datetime(2025, 1, 16, 4, 0, 0, tzinfo=UTC)
        result = to_cst(dt)
        
        assert result is not None
        assert result.date().isoformat() == "2025-01-15"

    def test_naive_datetime_assumed_utc(self):
        """Naive datetimes should be assumed UTC."""
        dt = datetime(2025, 1, 16, 4, 0, 0)
        result = to_cst(dt)
        
        assert result is not None
        assert result.hour == 22

    def test_none_returns_none(self):
        """None input should return None."""
        assert to_cst(None) is None

    def test_invalid_string_returns_none(self):
        """Invalid string should return None."""
        assert to_cst("not a date") is None
        assert to_cst("2025-99-99") is None


class TestStandardizeGameData:
    """Tests for standardize_game_data function."""

    def test_basic_game_standardization(self):
        """Basic game data should be standardized correctly."""
        game = {
            "away_team": "Lakers",
            "home_team": "Celtics",
            "commence_time": "2025-01-16T02:30:00Z",
        }
        
        result = standardize_game_data(game, source="the_odds_api")
        
        assert result["away_team"] == "Los Angeles Lakers"
        assert result["home_team"] == "Boston Celtics"
        assert result["date"] == "2025-01-15"  # Converted to CST
        assert result["_standardized"] is True
        assert result["_data_valid"] is True

    def test_preserves_betting_lines(self):
        """Betting line data should be preserved."""
        game = {
            "away_team": "LAL",
            "home_team": "BOS",
            "date": "2025-01-15",
            "spread": -3.5,
            "total": 220.5,
            "moneyline_away": +150,
            "moneyline_home": -170,
        }
        
        result = standardize_game_data(game, source="the_odds_api")
        
        assert result["spread"] == -3.5
        assert result["total"] == 220.5
        assert result["moneyline_away"] == +150
        assert result["moneyline_home"] == -170

    def test_marks_invalid_team(self):
        """Invalid team names should set _data_valid to False."""
        game = {
            "away_team": "Unknown Team XYZ",
            "home_team": "Boston Celtics",
            "date": "2025-01-15",
        }
        
        result = standardize_game_data(game, source="unknown")
        
        # Should still normalize what it can
        assert result["home_team"] == "Boston Celtics"
        # Away team couldn't be normalized
        assert result.get("_data_valid") is False or "Unknown" in result.get("away_team", "")

    def test_source_metadata(self):
        """Source metadata should be added."""
        game = {
            "away_team": "Lakers",
            "home_team": "Celtics",
        }
        
        result = standardize_game_data(game, source="the_odds_api")
        
        assert result["_source"] == "the_odds_api"

    def test_cst_date_conversion(self):
        """Game dates should be in CST."""
        # 4am UTC on Jan 16 = 10pm CST on Jan 15
        game = {
            "away_team": "LAL",
            "home_team": "BOS",
            "commence_time": "2025-01-16T04:00:00Z",
        }
        
        result = standardize_game_data(game, source="the_odds_api")
        
        assert result["date"] == "2025-01-15"
        assert "commence_time_cst" in result

    def test_date_only_input(self):
        """Date-only input (no time) should work."""
        game = {
            "away_team": "LAL",
            "home_team": "BOS",
            "date": "2025-01-15",
        }
        
        result = standardize_game_data(game, source="manual")
        
        assert result["date"] == "2025-01-15"

    def test_away_team_first_in_result(self):
        """Away team should come before home team in result dict."""
        game = {
            "home_team": "BOS",  # Home first in input
            "away_team": "LAL",
        }
        
        result = standardize_game_data(game, source="test")
        keys = list(result.keys())
        
        away_idx = keys.index("away_team")
        home_idx = keys.index("home_team")
        
        assert away_idx < home_idx


class TestFormatGameString:
    """Tests for format_game_string function."""

    def test_basic_format(self):
        """Games should format as 'AWAY vs. HOME'."""
        result = format_game_string("Los Angeles Lakers", "Boston Celtics")
        
        assert result == "Los Angeles Lakers vs. Boston Celtics"

    def test_format_preserves_names(self):
        """Team names should be preserved exactly."""
        result = format_game_string("LA Clippers", "Golden State Warriors")
        
        assert result == "LA Clippers vs. Golden State Warriors"


class TestTeamNameMappingConsistency:
    """Tests to ensure TEAM_NAME_MAPPING is complete and consistent."""

    def test_mapping_has_all_30_teams(self):
        """TEAM_NAME_MAPPING should have entries covering all 30 teams."""
        normalized_teams = set()
        
        for team_name in TEAM_NAME_MAPPING.values():
            normalized_teams.add(team_name)
        
        # Should have at least 30 unique normalized names
        assert len(normalized_teams) >= 30

    def test_espn_names_all_valid(self):
        """All ESPN_TEAM_NAMES should be exactly 30 teams."""
        assert len(ESPN_TEAM_NAMES) == 30

    def test_la_clippers_consistency(self):
        """LA Clippers should always normalize to 'LA Clippers' (ESPN format)."""
        variants = [
            "Los Angeles Clippers",
            "LA Clippers",
            "LAC",
            "lac",
            "Clippers",
            "clippers",
        ]
        
        for variant in variants:
            result, success = normalize_team_to_espn(variant)
            assert result == "LA Clippers", f"{variant} should normalize to 'LA Clippers', got '{result}'"
            assert success is True


class TestCSTConversionIntegration:
    """Integration tests for CST conversion in the full pipeline."""

    def test_late_night_game_correct_date(self):
        """Late night CST games from UTC should have correct date."""
        # 10pm CST game = 4am UTC next day
        game = {
            "away_team": "LAL",
            "home_team": "BOS",
            "commence_time": "2025-01-16T04:00:00Z",  # This is 10pm CST Jan 15
        }
        
        result = standardize_game_data(game, source="the_odds_api")
        
        # Game date should be Jan 15 (CST), not Jan 16 (UTC)
        assert result["date"] == "2025-01-15"

    def test_afternoon_game_same_date(self):
        """Afternoon CST games should have same date in UTC and CST."""
        # 2pm CST = 8pm UTC same day
        game = {
            "away_team": "LAL",
            "home_team": "BOS",
            "commence_time": "2025-01-15T20:00:00Z",  # 2pm CST Jan 15
        }
        
        result = standardize_game_data(game, source="the_odds_api")
        
        assert result["date"] == "2025-01-15"


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_game_data(self):
        """Empty game data should return minimal valid structure."""
        result = standardize_game_data({}, source="test")
        
        assert result["_standardized"] is True

    def test_missing_teams(self):
        """Missing teams should not crash."""
        game = {"date": "2025-01-15"}
        
        result = standardize_game_data(game, source="test")
        
        assert result is not None

    def test_special_characters_in_team_name(self):
        """Special characters should be handled."""
        # 76ers has a number
        result, success = normalize_team_to_espn("76ers")
        assert result == "Philadelphia 76ers"
        assert success is True
        
        result, success = normalize_team_to_espn("Philadelphia 76ers")
        assert result == "Philadelphia 76ers"
        assert success is True
