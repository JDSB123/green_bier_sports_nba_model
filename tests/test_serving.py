"""Tests for the FastAPI serving application."""
import pytest
from unittest.mock import Mock, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def mock_pipeline():
    """Create a mock ML pipeline."""
    pipeline = MagicMock()
    pipeline.predict_proba.return_value = [[0.3, 0.7]]
    return pipeline


@pytest.fixture
def app_with_model(monkeypatch, mock_pipeline):
    """Create app instance with a mocked model loaded."""
    # Mock the models directory and file operations
    import tempfile
    import os
    import json
    
    with tempfile.TemporaryDirectory() as tmpdir:
        models_dir = os.path.join(tmpdir, "models")
        os.makedirs(models_dir, exist_ok=True)
        
        # Create a mock manifest
        manifest = {
            "models": [
                {"path": "test_model.joblib", "created": "2025-01-01T00:00:00"}
            ]
        }
        with open(os.path.join(models_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f)
        
        # Mock environment variable
        monkeypatch.setenv("DATA_PROCESSED_DIR", tmpdir)
        
        # Mock the load_model function
        def mock_load_model(path):
            return {
                "pipeline": mock_pipeline,
                "name": "test_model",
                "meta": {"accuracy": 0.85}
            }
        
        from src.serving import app as app_module
        monkeypatch.setattr(app_module, "load_pipeline", mock_load_model)
        
        # Import and initialize app
        from src.serving.app import app
        
        # Manually trigger startup
        app.state.pipeline = mock_pipeline
        app.state.model_name = "test_model"
        app.state.meta = {"accuracy": 0.85}
        
        yield app


def test_health_endpoint_with_model(app_with_model):
    """Test health endpoint when model is loaded."""
    client = TestClient(app_with_model)
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_health_endpoint_without_model():
    """Test health endpoint when model is not loaded."""
    from src.serving.app import app
    
    # Reset app state
    app.state.pipeline = None
    app.state.model_name = None
    app.state.meta = {}
    
    client = TestClient(app)
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is False


def test_predict_endpoint_success(app_with_model):
    """Test successful prediction."""
    client = TestClient(app_with_model)
    
    features = {
        "team_a_ppg": 110.5,
        "team_b_ppg": 105.3,
        "team_a_home": 1,
        "team_b_home": 0,
    }
    
    response = client.post("/predict", json={"features": features})
    
    assert response.status_code == 200
    data = response.json()
    assert "model" in data
    assert "probabilities" in data
    assert data["model"] == "test_model"
    assert "class_0" in data["probabilities"]
    assert "class_1" in data["probabilities"]


def test_predict_endpoint_no_model():
    """Test prediction when model is not loaded."""
    from src.serving.app import app
    
    # Reset app state
    app.state.pipeline = None
    
    client = TestClient(app)
    
    features = {"test": 1}
    response = client.post("/predict", json={"features": features})
    
    assert response.status_code == 503
    assert "Model not loaded" in response.json()["detail"]


def test_predict_endpoint_invalid_features(app_with_model, monkeypatch):
    """Test prediction with features that cause an error."""
    # Make the pipeline raise an exception
    def mock_predict_proba(df):
        raise ValueError("Invalid features")
    
    app_with_model.state.pipeline.predict_proba = mock_predict_proba
    
    client = TestClient(app_with_model)
    
    features = {"invalid": "data"}
    response = client.post("/predict", json={"features": features})
    
    assert response.status_code == 500
    assert "Invalid features" in response.json()["detail"]

