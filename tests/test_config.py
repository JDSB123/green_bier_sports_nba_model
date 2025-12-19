"""Tests for configuration module."""
import os
from datetime import date
import pytest

from src.config import get_nba_season, get_current_nba_season, Settings


class TestNBASeason:
    """Tests for NBA season calculation."""

    def test_get_nba_season_october(self):
        """Test that October belongs to current year's season."""
        test_date = date(2025, 10, 15)
        assert get_nba_season(test_date) == "2025-2026"

    def test_get_nba_season_december(self):
        """Test that December belongs to current year's season."""
        test_date = date(2025, 12, 15)
        assert get_nba_season(test_date) == "2025-2026"

    def test_get_nba_season_january(self):
        """Test that January belongs to previous year's season."""
        test_date = date(2026, 1, 15)
        assert get_nba_season(test_date) == "2025-2026"

    def test_get_nba_season_april(self):
        """Test that April belongs to previous year's season."""
        test_date = date(2026, 4, 15)
        assert get_nba_season(test_date) == "2025-2026"

    def test_get_nba_season_september(self):
        """Test that September belongs to previous year's season."""
        test_date = date(2025, 9, 15)
        assert get_nba_season(test_date) == "2024-2025"

    def test_get_current_nba_season(self):
        """Test that current season returns a valid season string."""
        current = get_current_nba_season()
        assert isinstance(current, str)
        assert "-" in current
        years = current.split("-")
        assert len(years) == 2
        assert int(years[1]) == int(years[0]) + 1


class TestSettings:
    """Tests for Settings dataclass."""

    def test_settings_initialization(self):
        """Test that settings can be initialized."""
        settings = Settings()
        assert settings.current_season is not None
        assert settings.the_odds_base_url.startswith("http")
        assert settings.api_basketball_base_url.startswith("http")

    def test_settings_from_env(self, monkeypatch):
        """Test that settings can be overridden from environment."""
        monkeypatch.setenv("CURRENT_SEASON", "2024-2025")
        monkeypatch.setenv("THE_ODDS_API_KEY", "test_key_123")
        
        settings = Settings()
        assert settings.current_season == "2024-2025"
        assert settings.the_odds_api_key == "test_key_123"

    def test_seasons_to_process_parsing(self, monkeypatch):
        """Test that seasons list is parsed correctly from env."""
        monkeypatch.setenv("SEASONS_TO_PROCESS", "2022-2023,2023-2024,2024-2025")
        
        settings = Settings()
        assert len(settings.seasons_to_process) == 3
        assert "2022-2023" in settings.seasons_to_process
        assert "2024-2025" in settings.seasons_to_process

