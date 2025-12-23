"""Tests for the FastAPI serving application - v6.0 STRICT MODE.

Tests all 9 INDEPENDENT markets:
- First Quarter: Spread, Total, Moneyline
- First Half: Spread, Total, Moneyline
- Full Game: Spread, Total, Moneyline
"""
import pytest
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def mock_engine():
    """Create a mock UnifiedPredictionEngine for v6.0 (9 markets)."""
    engine = MagicMock()
    engine.predict_all_markets.return_value = {
        "first_quarter": {
            "spread": {
                "home_cover_prob": 0.58,
                "away_cover_prob": 0.42,
                "predicted_margin": 1.2,
                "confidence": 0.58,
                "bet_side": "home",
                "edge": 0.8,
                "model_edge_pct": 0.08,
                "passes_filter": False,
                "filter_reason": "Low edge",
            },
            "total": {
                "over_prob": 0.54,
                "under_prob": 0.46,
                "predicted_total": 56.5,
                "confidence": 0.54,
                "bet_side": "over",
                "edge": 1.0,
                "model_edge_pct": 0.04,
                "passes_filter": False,
                "filter_reason": "Low confidence",
            },
            "moneyline": {
                "home_win_prob": 0.60,
                "away_win_prob": 0.40,
                "predicted_winner": "home",
                "confidence": 0.60,
                "home_implied_prob": 0.55,
                "away_implied_prob": 0.45,
                "home_edge": 0.05,
                "away_edge": -0.05,
                "recommended_bet": "home",
                "passes_filter": True,
                "filter_reason": None,
            },
        },
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
            "moneyline": {
                "home_win_prob": 0.65,
                "away_win_prob": 0.35,
                "predicted_winner": "home",
                "confidence": 0.65,
                "home_implied_prob": 0.55,
                "away_implied_prob": 0.45,
                "home_edge": 0.10,
                "away_edge": -0.10,
                "recommended_bet": "home",
                "passes_filter": True,
                "filter_reason": None,
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
            "moneyline": {
                "home_win_prob": 0.70,
                "away_win_prob": 0.30,
                "predicted_winner": "home",
                "confidence": 0.70,
                "home_implied_prob": 0.60,
                "away_implied_prob": 0.40,
                "home_edge": 0.10,
                "away_edge": -0.10,
                "recommended_bet": "home",
                "passes_filter": True,
                "filter_reason": None,
            },
        },
    }
    # Mock get_model_info for health endpoint
    engine.get_model_info.return_value = {
        "version": "6.6",
        "architecture": "6-model independent (Q1 disabled)",
        "markets": 6,
        "markets_list": [
            "1h_spread", "1h_total", "1h_moneyline",
            "fg_spread", "fg_total", "fg_moneyline",
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
        "predicted_margin_q1": 1.2,
        "predicted_total_q1": 56.5,
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
    """Test health endpoint when engine is loaded - v6.0."""
    client = TestClient(app_with_engine)
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    from src.serving.app import app as fastapi_app
    assert data["version"] == fastapi_app.version
    assert data["architecture"] == "6-model independent (Q1 disabled)"
    assert data["markets"] == 6
    assert data["engine_loaded"] is True
    # Verify only active 6 markets are listed (Q1 disabled)
    assert "1h_spread" in data["markets_list"]
    assert "fg_spread" in data["markets_list"]
    assert "q1_spread" not in data["markets_list"]


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
    assert data["architecture"] == "6-model independent (Q1 disabled)"
    assert data["engine_loaded"] is False


def test_predict_game_success(app_with_engine):
    """Test successful single game prediction - v6.0 all 9 markets."""
    client = TestClient(app_with_engine)

    # v6.0: Full game lines required, period lines optional
    request_data = {
        "home_team": "Cleveland Cavaliers",
        "away_team": "Chicago Bulls",
        "fg_spread_line": -5.5,
        "fg_total_line": 222.0,
        "home_ml_odds": -200,
        "away_ml_odds": 170,
        "fh_spread_line": -2.5,
        "fh_total_line": 111.0,
        "fh_home_ml_odds": -150,
        "fh_away_ml_odds": 130,
        "q1_spread_line": -1.0,
        "q1_total_line": 55.5,
        "q1_home_ml_odds": -120,
        "q1_away_ml_odds": 100,
    }

    response = client.post("/predict/game", json=request_data)

    assert response.status_code == 200
    data = response.json()
    # Check all 3 periods are present
    assert "first_quarter" in data
    assert "first_half" in data
    assert "full_game" in data
    # Check all markets in full_game
    assert "spread" in data["full_game"]
    assert "total" in data["full_game"]
    assert "moneyline" in data["full_game"]
    # Check first_quarter markets
    assert "spread" in data["first_quarter"]
    assert "total" in data["first_quarter"]
    assert "moneyline" in data["first_quarter"]


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
    """Test prediction when engine is not loaded - v6.0 STRICT MODE."""
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
        "home_ml_odds": -200,
        "away_ml_odds": 170,
    }

    response = client.post("/predict/game", json=request_data)

    assert response.status_code == 503
    assert "Engine not loaded" in response.json()["detail"]


def test_predict_game_with_only_fg_lines(app_with_engine):
    """Test prediction with only full game lines (1H and Q1 are optional)."""
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
