"""Tests for data ingestion modules."""
import pytest
from unittest.mock import Mock, AsyncMock, patch
import httpx


class TestTheOddsAPI:
    """Tests for The Odds API client."""

    @pytest.mark.asyncio
    async def test_fetch_odds_success(self, monkeypatch):
        """Test successful odds fetching."""
        from src.ingestion.the_odds import fetch_odds
        
        # Mock response data
        mock_data = [
            {
                "id": "test123",
                "sport_key": "basketball_nba",
                "home_team": "Lakers",
                "away_team": "Warriors",
                "bookmakers": [],
            }
        ]
        
        # Mock httpx.AsyncClient
        mock_response = Mock()
        mock_response.json.return_value = mock_data
        mock_response.raise_for_status = Mock()
        
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
        
        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await fetch_odds()
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["id"] == "test123"

    @pytest.mark.asyncio
    async def test_save_odds(self, tmp_path, monkeypatch):
        """Test saving odds data to file."""
        from src.ingestion.the_odds import save_odds
        
        # Use temporary directory
        monkeypatch.setattr("src.ingestion.the_odds.settings.data_raw_dir", str(tmp_path))
        
        mock_data = [{"game_id": 1, "odds": 100}]
        
        path = await save_odds(mock_data)
        
        assert path is not None
        assert "odds_" in path
        assert path.endswith(".json")
        
        # Verify file was created
        import os
        assert os.path.exists(path)


class TestAPIBasketball:
    """Tests for API-Basketball client."""

    def test_client_initialization(self):
        """Test that APIBasketballClient initializes correctly."""
        from src.ingestion.api_basketball import APIBasketballClient
        
        client = APIBasketballClient(season="2024-2025", league_id=12)
        
        assert client.season == "2024-2025"
        assert client.league_id == 12
        assert client.headers is not None
        assert "x-apisports-key" in client.headers

    @pytest.mark.asyncio
    async def test_fetch_teams_success(self, monkeypatch):
        """Test successful team fetching."""
        from src.ingestion.api_basketball import APIBasketballClient
        
        # Mock response
        mock_response = {
            "response": [
                {"id": 1, "name": "Lakers", "code": "LAL"},
                {"id": 2, "name": "Warriors", "code": "GSW"},
            ]
        }
        
        # Create client
        client = APIBasketballClient()
        
        # Mock the _fetch method
        client._fetch = AsyncMock(return_value=mock_response)
        client._save = Mock(return_value="/fake/path/teams.json")
        
        result = await client.fetch_teams()
        
        assert result.name == "teams"
        assert result.count == 2
        assert len(client._teams) == 2

    @pytest.mark.asyncio
    async def test_fetch_games_success(self, monkeypatch):
        """Test successful game fetching."""
        from src.ingestion.api_basketball import APIBasketballClient
        
        # Mock response
        mock_response = {
            "response": [
                {"id": 100, "status": {"long": "Finished"}},
                {"id": 101, "status": {"long": "Finished"}},
                {"id": 102, "status": {"long": "Scheduled"}},
            ]
        }
        
        client = APIBasketballClient()
        client._fetch = AsyncMock(return_value=mock_response)
        client._save = Mock(return_value="/fake/path/games.json")
        
        result = await client.fetch_games()
        
        assert result.name == "games"
        assert result.count == 3
        assert len(client._games) == 3


class TestBettingSplits:
    """Tests for betting splits integration."""

    def test_betting_splits_module_exists(self):
        """Test that betting splits module can be imported."""
        try:
            from src.ingestion import betting_splits
            assert betting_splits is not None
        except ImportError:
            pytest.skip("Betting splits module not yet implemented")


class TestInjuries:
    """Tests for injury data ingestion."""

    def test_injuries_module_exists(self):
        """Test that injuries module can be imported."""
        try:
            from src.ingestion import injuries
            assert injuries is not None
        except ImportError:
            pytest.skip("Injuries module not yet implemented")

