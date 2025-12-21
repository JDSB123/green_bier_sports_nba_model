"""Shared pytest fixtures and configuration hooks."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure the project root (which contains the `src` package) is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT_STR = str(PROJECT_ROOT)
if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)


# =============================================================================
# Environment Variables Setup - MUST run before any src imports
# =============================================================================
# Set all required environment variables for testing BEFORE importing src modules
# This prevents ValueError from src/config.py's _env_required() calls

TEST_ENV_VARS = {
    # API Keys (test values)
    "THE_ODDS_API_KEY": "test_the_odds_api_key_12345",
    "API_BASKETBALL_KEY": "test_api_basketball_key_12345",
    
    # API Base URLs
    "THE_ODDS_BASE_URL": "https://api.the-odds-api.com/v4",
    "API_BASKETBALL_BASE_URL": "https://v1.basketball.api-sports.io",
    
    # Season Configuration
    "CURRENT_SEASON": "2025-2026",
    "SEASONS_TO_PROCESS": "2024-2025,2025-2026",
    
    # Data Directories
    "DATA_RAW_DIR": str(PROJECT_ROOT / "data" / "raw"),
    "DATA_PROCESSED_DIR": str(PROJECT_ROOT / "data" / "processed"),
    
    # Filter Thresholds (defaults for testing)
    "FILTER_SPREAD_MIN_CONFIDENCE": "0.55",
    "FILTER_SPREAD_MIN_EDGE": "1.0",
    "FILTER_TOTAL_MIN_CONFIDENCE": "0.55",
    "FILTER_TOTAL_MIN_EDGE": "1.5",
    "FILTER_MONEYLINE_MIN_CONFIDENCE": "0.55",
    "FILTER_MONEYLINE_MIN_EDGE_PCT": "0.03",
    "FILTER_Q1_MIN_CONFIDENCE": "0.60",
    "FILTER_Q1_MIN_EDGE_PCT": "0.05",
    
    # Optional (empty for tests)
    "BETSAPI_KEY": "",
    "ACTION_NETWORK_USERNAME": "",
    "ACTION_NETWORK_PASSWORD": "",
    "KAGGLE_API_TOKEN": "",
    
    # Logging
    "LOG_LEVEL": "WARNING",
}

# Apply test environment variables
for key, value in TEST_ENV_VARS.items():
    if key not in os.environ:
        os.environ[key] = value


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Ensure test environment variables are set for each test."""
    for key, value in TEST_ENV_VARS.items():
        monkeypatch.setenv(key, value)
