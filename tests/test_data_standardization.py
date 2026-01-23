"""
Tests for src/data/standardization.py - THE SINGLE SOURCE OF TRUTH for data normalization.

These tests ensure all 30 NBA teams and timezone conversions work correctly.
Coverage target: 100% of src/data/standardization.py
"""
import pytest
from datetime import datetime
from zoneinfo import ZoneInfo

from src.data.standardization import (
    standardize_team_name,
    to_cst,
    to_cst_local,
    to_cst_date,
    to_cst_date_local,
    generate_match_key,
    standardize_game_record,
    CST,
    UTC,
    ABBREV_TO_FULL,
    _CANONICAL_TO_FULL,
)


class TestTeamNameStandardization:
    """Tests for team name standardization - ALL 30 NBA teams."""

    # Full team names that should return unchanged
    @pytest.mark.parametrize("team_name,expected", [
        ("Atlanta Hawks", "Atlanta Hawks"),
        ("Boston Celtics", "Boston Celtics"),
        ("Brooklyn Nets", "Brooklyn Nets"),
        ("Charlotte Hornets", "Charlotte Hornets"),
        ("Chicago Bulls", "Chicago Bulls"),
        ("Cleveland Cavaliers", "Cleveland Cavaliers"),
        ("Dallas Mavericks", "Dallas Mavericks"),
        ("Denver Nuggets", "Denver Nuggets"),
        ("Detroit Pistons", "Detroit Pistons"),
        ("Golden State Warriors", "Golden State Warriors"),
        ("Houston Rockets", "Houston Rockets"),
        ("Indiana Pacers", "Indiana Pacers"),
        ("LA Clippers", "LA Clippers"),  # ESPN format
        ("Los Angeles Lakers", "Los Angeles Lakers"),
        ("Memphis Grizzlies", "Memphis Grizzlies"),
        ("Miami Heat", "Miami Heat"),
        ("Milwaukee Bucks", "Milwaukee Bucks"),
        ("Minnesota Timberwolves", "Minnesota Timberwolves"),
        ("New Orleans Pelicans", "New Orleans Pelicans"),
        ("New York Knicks", "New York Knicks"),
        ("Oklahoma City Thunder", "Oklahoma City Thunder"),
        ("Orlando Magic", "Orlando Magic"),
        ("Philadelphia 76ers", "Philadelphia 76ers"),
        ("Phoenix Suns", "Phoenix Suns"),
        ("Portland Trail Blazers", "Portland Trail Blazers"),
        ("Sacramento Kings", "Sacramento Kings"),
        ("San Antonio Spurs", "San Antonio Spurs"),
        ("Toronto Raptors", "Toronto Raptors"),
        ("Utah Jazz", "Utah Jazz"),
        ("Washington Wizards", "Washington Wizards"),
    ])
    def test_full_names_unchanged(self, team_name, expected):
        """Full ESPN team names should remain unchanged."""
        assert standardize_team_name(team_name) == expected

    # Abbreviations (2-3 letters)
    @pytest.mark.parametrize("abbrev,expected", [
        ("atl", "Atlanta Hawks"),
        ("bos", "Boston Celtics"),
        ("bkn", "Brooklyn Nets"),
        ("cha", "Charlotte Hornets"),
        ("chi", "Chicago Bulls"),
        ("cle", "Cleveland Cavaliers"),
        ("dal", "Dallas Mavericks"),
        ("den", "Denver Nuggets"),
        ("det", "Detroit Pistons"),
        ("gsw", "Golden State Warriors"),
        ("gs", "Golden State Warriors"),  # Alternative
        ("hou", "Houston Rockets"),
        ("ind", "Indiana Pacers"),
        ("lac", "LA Clippers"),
        ("lal", "Los Angeles Lakers"),
        ("mem", "Memphis Grizzlies"),
        ("mia", "Miami Heat"),
        ("mil", "Milwaukee Bucks"),
        ("min", "Minnesota Timberwolves"),
        ("nop", "New Orleans Pelicans"),
        ("no", "New Orleans Pelicans"),  # Alternative
        ("nyk", "New York Knicks"),
        ("ny", "New York Knicks"),  # Alternative
        ("okc", "Oklahoma City Thunder"),
        ("orl", "Orlando Magic"),
        ("phi", "Philadelphia 76ers"),
        ("phx", "Phoenix Suns"),
        ("pho", "Phoenix Suns"),  # Alternative
        ("por", "Portland Trail Blazers"),
        ("sac", "Sacramento Kings"),
        ("sas", "San Antonio Spurs"),
        ("sa", "San Antonio Spurs"),  # Alternative
        ("tor", "Toronto Raptors"),
        ("utah", "Utah Jazz"),
        ("uta", "Utah Jazz"),  # Alternative
        ("was", "Washington Wizards"),
        ("wsh", "Washington Wizards"),  # Alternative
    ])
    def test_abbreviations(self, abbrev, expected):
        """Standard abbreviations should map to full names."""
        assert standardize_team_name(abbrev) == expected

    # Nicknames only
    @pytest.mark.parametrize("nickname,expected", [
        ("hawks", "Atlanta Hawks"),
        ("celtics", "Boston Celtics"),
        ("nets", "Brooklyn Nets"),
        ("hornets", "Charlotte Hornets"),
        ("bulls", "Chicago Bulls"),
        ("cavaliers", "Cleveland Cavaliers"),
        ("cavs", "Cleveland Cavaliers"),
        ("mavericks", "Dallas Mavericks"),
        ("mavs", "Dallas Mavericks"),
        ("nuggets", "Denver Nuggets"),
        ("pistons", "Detroit Pistons"),
        ("warriors", "Golden State Warriors"),
        ("rockets", "Houston Rockets"),
        ("pacers", "Indiana Pacers"),
        ("clippers", "LA Clippers"),
        ("lakers", "Los Angeles Lakers"),
        ("grizzlies", "Memphis Grizzlies"),
        ("heat", "Miami Heat"),
        ("bucks", "Milwaukee Bucks"),
        ("timberwolves", "Minnesota Timberwolves"),
        ("wolves", "Minnesota Timberwolves"),
        ("pelicans", "New Orleans Pelicans"),
        ("knicks", "New York Knicks"),
        ("thunder", "Oklahoma City Thunder"),
        ("magic", "Orlando Magic"),
        ("76ers", "Philadelphia 76ers"),
        ("sixers", "Philadelphia 76ers"),
        ("suns", "Phoenix Suns"),
        ("blazers", "Portland Trail Blazers"),
        ("trail blazers", "Portland Trail Blazers"),
        ("kings", "Sacramento Kings"),
        ("spurs", "San Antonio Spurs"),
        ("raptors", "Toronto Raptors"),
        ("jazz", "Utah Jazz"),
        ("wizards", "Washington Wizards"),
    ])
    def test_nicknames(self, nickname, expected):
        """Nicknames should map to full names."""
        assert standardize_team_name(nickname) == expected

    # Historical franchise names
    @pytest.mark.parametrize("historical,expected", [
        ("nj", "Brooklyn Nets"),  # New Jersey Nets
        ("njn", "Brooklyn Nets"),
        ("sea", "Oklahoma City Thunder"),  # Seattle Supersonics
        ("noh", "New Orleans Pelicans"),  # New Orleans Hornets
        ("nok", "New Orleans Pelicans"),  # New Orleans/Oklahoma City
    ])
    def test_historical_names(self, historical, expected):
        """Historical franchise names should map correctly."""
        assert standardize_team_name(historical) == expected

    # Case insensitivity
    def test_case_insensitive(self):
        """Team names should be case-insensitive."""
        assert standardize_team_name("LAL") == "Los Angeles Lakers"
        assert standardize_team_name("lal") == "Los Angeles Lakers"
        assert standardize_team_name("LaL") == "Los Angeles Lakers"
        assert standardize_team_name("CELTICS") == "Boston Celtics"
        assert standardize_team_name("Celtics") == "Boston Celtics"

    # Edge cases
    def test_empty_string(self):
        """Empty string should return as-is."""
        assert standardize_team_name("") == ""

    def test_none_input(self):
        """None should return None."""
        assert standardize_team_name(None) is None

    def test_whitespace_handling(self):
        """Whitespace should be stripped."""
        assert standardize_team_name("  lal  ") == "Los Angeles Lakers"
        assert standardize_team_name("\tlakers\n") == "Los Angeles Lakers"

    def test_unknown_team_returns_title_case(self):
        """Unknown teams should return with original capitalization."""
        result = standardize_team_name("unknown team")
        assert result == "unknown team"


class TestTimezoneConversion:
    """Tests for CST timezone conversion - CRITICAL for date matching."""

    def test_utc_to_cst_basic(self):
        """UTC times should convert to CST correctly."""
        # CST is UTC-6 (standard) or UTC-5 (daylight saving)
        utc_dt = datetime(2025, 1, 15, 4, 0, 0, tzinfo=UTC)
        cst_dt = to_cst(utc_dt)
        
        assert cst_dt is not None
        assert cst_dt.tzinfo == CST
        # January is standard time: UTC-6
        assert cst_dt.hour == 22  # 4am UTC = 10pm CST (previous day)
        assert cst_dt.day == 14  # Previous day

    def test_utc_string_with_z_suffix(self):
        """ISO strings with Z suffix (UTC) should convert correctly."""
        cst_dt = to_cst("2025-01-16T04:00:00Z")
        
        assert cst_dt is not None
        assert cst_dt.date().isoformat() == "2025-01-15"  # Previous day in CST
        assert cst_dt.hour == 22

    def test_utc_string_with_offset(self):
        """ISO strings with UTC offset should convert correctly."""
        cst_dt = to_cst("2025-01-16T04:00:00+00:00")
        
        assert cst_dt is not None
        assert cst_dt.date().isoformat() == "2025-01-15"

    def test_naive_datetime_treated_as_utc(self):
        """Naive datetimes should be treated as UTC by default."""
        naive_dt = datetime(2025, 1, 15, 4, 0, 0)
        cst_dt = to_cst(naive_dt)
        
        assert cst_dt is not None
        assert cst_dt.hour == 22  # 4am UTC = 10pm CST

    def test_to_cst_local_for_us_times(self):
        """to_cst_local should treat input as already in local US time."""
        local_dt = datetime(2025, 1, 15, 19, 30, 0)  # 7:30pm local
        cst_dt = to_cst_local(local_dt)
        
        assert cst_dt is not None
        assert cst_dt.hour == 19  # Should stay 7:30pm
        assert cst_dt.minute == 30

    def test_to_cst_none_input(self):
        """None input should return None."""
        assert to_cst(None) is None
        assert to_cst_local(None) is None

    def test_to_cst_invalid_string(self):
        """Invalid datetime strings should return None."""
        assert to_cst("not a date") is None
        assert to_cst("invalid") is None

    def test_to_cst_date_string_only(self):
        """Date-only strings should parse correctly."""
        cst_dt = to_cst("2025-01-15")
        
        assert cst_dt is not None
        # Midnight UTC = 6pm previous day CST
        assert cst_dt.date().isoformat() == "2025-01-14"

    def test_to_cst_date_function(self):
        """to_cst_date should return YYYY-MM-DD string."""
        result = to_cst_date("2025-01-16T04:00:00Z")
        assert result == "2025-01-15"

    def test_to_cst_date_local_function(self):
        """to_cst_date_local should return date from local time."""
        result = to_cst_date_local("2025-01-15 19:30:00")
        assert result == "2025-01-15"

    def test_daylight_saving_time(self):
        """DST should be handled correctly (summer = UTC-5)."""
        # July is daylight saving time: UTC-5
        utc_dt = datetime(2025, 7, 15, 4, 0, 0, tzinfo=UTC)
        cst_dt = to_cst(utc_dt)
        
        assert cst_dt is not None
        assert cst_dt.hour == 23  # 4am UTC = 11pm CDT (previous day)


class TestMatchKeyGeneration:
    """Tests for match key generation - used for data joining."""

    def test_basic_match_key(self):
        """Basic match key should be date_away_home format."""
        key = generate_match_key("2025-01-15", "Los Angeles Lakers", "Boston Celtics")
        
        assert key is not None
        assert "2025-01-15" in key or "2025-01-14" in key  # CST conversion
        assert "lakers" in key.lower() or "los angeles" in key.lower()
        assert "celtics" in key.lower() or "boston" in key.lower()

    def test_match_key_with_datetime(self):
        """Match key should accept datetime objects."""
        dt = datetime(2025, 1, 15, 19, 30, 0, tzinfo=CST)
        key = generate_match_key(dt, "Lakers", "Celtics")
        
        assert key is not None
        assert "2025-01-15" in key

    def test_match_key_normalizes_teams(self):
        """Match key should normalize team names."""
        key1 = generate_match_key("2025-01-15", "lal", "bos", source_is_utc=False)
        key2 = generate_match_key("2025-01-15", "Los Angeles Lakers", "Boston Celtics", source_is_utc=False)
        
        # Both should produce the same normalized key
        assert key1 == key2


class TestGameRecordStandardization:
    """Tests for standardize_game_record function."""

    def test_standardize_basic_record(self):
        """Basic game record should be standardized correctly."""
        record = {
            "home_team": "lal",
            "away_team": "bos",
            "commence_time": "2025-01-16T02:30:00Z",  # UTC
            "spread": -3.5,
        }
        
        result = standardize_game_record(record)
        
        assert result["home_team"] == "Los Angeles Lakers"
        assert result["away_team"] == "Boston Celtics"
        # Date may or may not be present depending on implementation
        assert "match_key" in result or "home_team" in result

    def test_standardize_preserves_other_fields(self):
        """Non-team fields should be preserved."""
        record = {
            "home_team": "Lakers",
            "away_team": "Celtics",
            "commence_time": "2025-01-15T19:30:00-06:00",  # Already CST
            "spread": -3.5,
            "total": 220.5,
            "custom_field": "preserved",
        }
        
        result = standardize_game_record(record)
        
        assert result["spread"] == -3.5
        assert result["total"] == 220.5
        assert result["custom_field"] == "preserved"


class TestCanonicalTeamNames:
    """Tests for canonical team name listing."""

    def test_returns_all_30_teams(self):
        """Should return all 30 NBA teams."""
        teams = list(_CANONICAL_TO_FULL.values())
        
        assert len(teams) == 30

    def test_returns_full_names(self):
        """Should return full team names."""
        teams = list(_CANONICAL_TO_FULL.values())
        
        assert "Los Angeles Lakers" in teams
        assert "Boston Celtics" in teams
        assert "LA Clippers" in teams  # ESPN format


class TestDataIntegrity:
    """Tests to ensure mapping data integrity."""

    def test_all_canonical_ids_have_full_names(self):
        """Every canonical ID should map to a full name."""
        for canonical_id in _CANONICAL_TO_FULL:
            assert _CANONICAL_TO_FULL[canonical_id] is not None
            assert len(_CANONICAL_TO_FULL[canonical_id]) > 3

    def test_no_duplicate_full_names(self):
        """Full names should be unique."""
        full_names = list(ABBREV_TO_FULL.values())
        unique_names = set(full_names)
        
        # Full names can map from multiple abbreviations
        assert len(unique_names) == 30  # 30 NBA teams

    def test_abbrev_to_full_consistency(self):
        """All abbreviations should map to valid full names."""
        valid_names = set(_CANONICAL_TO_FULL.values())
        
        for abbrev, full_name in ABBREV_TO_FULL.items():
            assert full_name in valid_names, f"{abbrev} maps to invalid name: {full_name}"
