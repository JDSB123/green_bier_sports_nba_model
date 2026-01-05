"""Tests for the FastAPI serving application - v33.0.8.0 STRICT MODE.

Active markets: 1H + FG (spreads, totals). Q1 is disabled.
"""
import pytest
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient

@pytest.fixture
def mock_engine():
    """Create a mock UnifiedPredictionEngine for 4 markets (1H + FG)."""
    engine = MagicMock()
    engine.predict_all_markets.return_value = {
        "first_half": {
            "spread": {
                "home_cover_prob": 0.60,
                "away_cover_prob": 0.40,
                "predicted_margin": 2.5,
                "confidence": 0.60,
                "bet_side": "home",
                "edge": 1.5,
                "model_edge_pct": 0.10,
                "passes_filter": True,
                "filter_reason": None,
            },
            "total": {
                "over_prob": 0.52,
                "under_prob": 0.48,
                "predicted_total": 112.5,
                "confidence": 0.52,
                "bet_side": "over",
                "edge": 1.5,
                "model_edge_pct": 0.02,
                "passes_filter": False,
                "filter_reason": "Insufficient edge",
            },
        },
        "full_game": {
            "spread": {
                "home_cover_prob": 0.65,
                "away_cover_prob": 0.35,
                "predicted_margin": 5.5,
                "confidence": 0.65,
                "bet_side": "home",
                "edge": 2.5,
                "model_edge_pct": 0.15,
                "passes_filter": True,
                "filter_reason": None,
            },
            "total": {
                "over_prob": 0.55,
                "under_prob": 0.45,
                "predicted_total": 225.5,
                "confidence": 0.55,
                "bet_side": "over",
                "edge": 3.5,
                "model_edge_pct": 0.05,
                "passes_filter": True,
                "filter_reason": None,
            },
        },
    }
    # Mock get_model_info for health endpoint
    engine.get_model_info.return_value = {
                        "version": "NBA_v33.0.10.0",
        "architecture": "1H + FG spreads/totals only",
        "markets": 4,
        "markets_list": [
            "1h_spread", "1h_total",
            "fg_spread", "fg_total",
        ],
        "periods": ["first_half", "full_game"],
    }
    return engine

@pytest.fixture
def mock_feature_builder():
    """Create a mock RichFeatureBuilder."""
    builder = MagicMock()
    builder.build_game_features = AsyncMock(return_value={
        "home_ppg": 110.5,
        "away_ppg": 105.3,
        "predicted_margin": 5.5,
        "predicted_total": 225.5,
        "predicted_margin_1h": 2.5,
        "predicted_total_1h": 112.5,
    })
    return builder

@pytest.fixture
def app_with_engine(mock_engine, mock_feature_builder):
    """Create app instance with mocked engine."""
    from src.serving.app import app

    # Set up app state with mocks
    app.state.engine = mock_engine
    app.state.feature_builder = mock_feature_builder

    yield app

    # Cleanup
    app.state.engine = None
    app.state.feature_builder = None

def test_health_endpoint_with_engine(app_with_engine):
    """Test health endpoint when engine is loaded."""
    client = TestClient(app_with_engine)
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    from src.serving.app import app as fastapi_app
    assert data["version"] == fastapi_app.version
    assert data["architecture"] == "1H + FG spreads/totals only"
    assert data["markets"] == 4
    assert data["engine_loaded"] is True
    # Verify only active 4 markets are listed
    assert set(data["markets_list"]) == {
        "1h_spread", "1h_total",
        "fg_spread", "fg_total",
    }

def test_health_endpoint_without_engine():
    """Test health endpoint when engine is not loaded."""
    from src.serving.app import app

    # Ensure no engine
    if hasattr(app.state, 'engine'):
        app.state.engine = None

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    from src.serving.app import app as fastapi_app
    assert data["version"] == fastapi_app.version
    assert data["architecture"] == "1H + FG spreads/totals only"
    assert data["engine_loaded"] is False

def test_predict_game_success(app_with_engine):
    """Test successful single game prediction - v33.0.8.0 (1H + FG)."""
    client = TestClient(app_with_engine)

    # v33.0.8.0: Full game lines required, 1H optional
    request_data = {
        "home_team": "Cleveland Cavaliers",
        "away_team": "Chicago Bulls",
        "fg_spread_line": -5.5,
        "fg_total_line": 222.0,
        "fh_spread_line": -2.5,
        "fh_total_line": 111.0,
    }

    response = client.post("/predict/game", json=request_data)

    assert response.status_code == 200
    data = response.json()
    # Check active periods are present
    assert "first_half" in data
    assert "full_game" in data
    # Check all markets in full_game
    assert "spread" in data["full_game"]
    assert "total" in data["full_game"]

def test_predict_game_missing_lines():
    """Test prediction fails when required lines are missing - STRICT MODE."""
    from src.serving.app import app

    # Engine loaded but request missing required fields
    app.state.engine = MagicMock()

    client = TestClient(app)

    # Missing required fields - should fail validation
    request_data = {
        "home_team": "Cleveland Cavaliers",
        "away_team": "Chicago Bulls",
        # Missing all line fields
    }

    response = client.post("/predict/game", json=request_data)

    # Pydantic validation should reject this
    assert response.status_code == 422  # Validation error

    # Cleanup
    app.state.engine = None

def test_predict_game_no_engine():
    """Test prediction when engine is not loaded."""
    from src.serving.app import app

    # Ensure no engine
    if hasattr(app.state, 'engine'):
        app.state.engine = None

    client = TestClient(app)

    request_data = {
        "home_team": "Cleveland Cavaliers",
        "away_team": "Chicago Bulls",
        "fg_spread_line": -5.5,
        "fg_total_line": 222.0,
    }

    response = client.post("/predict/game", json=request_data)

    assert response.status_code == 503
    assert "Engine not loaded" in response.json()["detail"]

def test_predict_game_with_only_fg_lines(app_with_engine):
    """Test prediction with only full game lines (1H is optional)."""
    client = TestClient(app_with_engine)

    # Minimal request - only FG lines required
    request_data = {
        "home_team": "Cleveland Cavaliers",
        "away_team": "Chicago Bulls",
        "fg_spread_line": -5.5,
        "fg_total_line": 222.0,
    }

    response = client.post("/predict/game", json=request_data)

    assert response.status_code == 200
    data = response.json()
    assert "full_game" in data
