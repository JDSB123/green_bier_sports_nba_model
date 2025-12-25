from fastapi.testclient import TestClient
from src.serving.app import app
import os

client = TestClient(app)

def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "engine_loaded" in data

def test_meta_endpoint():
    """Test the meta endpoint."""
    response = client.get("/meta")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert "server_time" in data
    assert "python_version" in data

def test_metrics_endpoint():
    """Test the prometheus metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "nba_api_requests_total" in response.text
