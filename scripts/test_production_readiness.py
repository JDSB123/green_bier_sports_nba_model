#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Production Readiness Test Suite

Tests all endpoints, logic, and integrations to verify production readiness.

ENVIRONMENT CONFIGURATION:
    This script uses the EXACT SAME environment variable defaults as Dockerfile
    (see Dockerfile lines 113-140). This ensures local testing matches production.

    The ONLY variables NOT set here are API keys (THE_ODDS_API_KEY, API_BASKETBALL_KEY),
    which must be provided via:
    - Environment variables
    - Docker secrets (secrets/THE_ODDS_API_KEY, secrets/API_BASKETBALL_KEY)
    - Azure Container App environment variables (in production)

    All other configuration matches Docker exactly - no manual setup needed!
"""
import sys
import os
import asyncio
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import traceback

# Read VERSION file for local defaults (keeps in sync with production tagging)
def _resolve_local_version() -> str:
    version_path = Path(__file__).parent.parent / "VERSION"
    if version_path.exists():
        return version_path.read_text(encoding="utf-8").strip()
    return ""

# ============================================================================
# ENVIRONMENT VARIABLES - MATCHES DOCKERFILE EXACTLY (lines 113-140)
# ============================================================================
# These are set BEFORE importing config to match production Docker environment
# Only API keys need to be provided separately (via secrets/env vars)

# Data Directories (set after PROJECT_ROOT is defined below)
# Will be set to match Docker paths: DATA_RAW_DIR=data/raw, DATA_PROCESSED_DIR=/app/data/processed

# API Base URLs (REQUIRED by src/config.py - matches Dockerfile lines 117-118)
os.environ.setdefault('THE_ODDS_BASE_URL', 'https://api.the-odds-api.com/v4')
os.environ.setdefault('API_BASKETBALL_BASE_URL', 'https://v1.basketball.api-sports.io')

# Season Configuration (REQUIRED by src/config.py - matches Dockerfile lines 121-122)
os.environ.setdefault('CURRENT_SEASON', '2025-2026')
os.environ.setdefault('SEASONS_TO_PROCESS', '2024-2025,2025-2026')

# Filter Thresholds (REQUIRED by src/config.py - matches Dockerfile lines 126-130)
os.environ.setdefault('FILTER_SPREAD_MIN_CONFIDENCE', '0.55')
os.environ.setdefault('FILTER_SPREAD_MIN_EDGE', '1.0')
os.environ.setdefault('FILTER_TOTAL_MIN_CONFIDENCE', '0.55')
os.environ.setdefault('FILTER_TOTAL_MIN_EDGE', '1.5')

# Relax feature validation for test harness to avoid hard fails on synthetic samples
os.environ.setdefault('PREDICTION_FEATURE_MODE', 'warn')

# Optional Configuration (matches Dockerfile)
os.environ.setdefault('ALLOWED_ORIGINS', '*')
os.environ.setdefault('NBA_MODEL_VERSION', _resolve_local_version() or 'unknown')
os.environ.setdefault('NBA_MARKETS', '1h_spread,1h_total,fg_spread,fg_total')
os.environ.setdefault('NBA_PERIODS', 'first_half,full_game')
os.environ.setdefault('NBA_STRICT_MODE', 'true')
os.environ.setdefault('NBA_CACHE_DISABLED', 'true')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set data directory defaults (matches Dockerfile lines 113-114)
# Docker uses: DATA_RAW_DIR=data/raw, DATA_PROCESSED_DIR=/app/data/processed
# For local testing, use project-relative paths
os.environ.setdefault('DATA_RAW_DIR', str(PROJECT_ROOT / 'data' / 'raw'))
os.environ.setdefault('DATA_PROCESSED_DIR', str(PROJECT_ROOT / 'data' / 'processed'))

# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Test results tracking
test_results = {
    "passed": [],
    "failed": [],
    "warnings": [],
    "skipped": []
}


def log_test(name: str, status: str, message: str = "", details: Any = None):
    """Log test result."""
    result = {
        "name": name,
        "status": status,
        "message": message,
        "details": details,
        "timestamp": datetime.now().isoformat()
    }
    
    # Use ASCII-safe symbols for Windows compatibility
    symbols = {
        "pass": "[PASS]",
        "fail": "[FAIL]",
        "warn": "[WARN]",
        "skip": "[SKIP]"
    }
    
    symbol = symbols.get(status, "[?]")
    
    if status == "pass":
        test_results["passed"].append(result)
        print(f"  {symbol} {name}: {message}")
    elif status == "fail":
        test_results["failed"].append(result)
        print(f"  {symbol} {name}: {message}")
        if details:
            print(f"    Details: {details}")
    elif status == "warn":
        test_results["warnings"].append(result)
        print(f"  {symbol} {name}: {message}")
    elif status == "skip":
        test_results["skipped"].append(result)
        print(f"  {symbol} {name}: {message}")


# ============================================================================
# 1. CONFIGURATION & ENVIRONMENT TESTS
# ============================================================================

def test_configuration():
    """Test configuration and environment setup."""
    print("\n" + "=" * 70)
    print("1. CONFIGURATION & ENVIRONMENT")
    print("=" * 70)
    
    try:
        from src.config import settings, PROJECT_ROOT
        
        # Test PROJECT_ROOT
        if PROJECT_ROOT.exists():
            log_test("PROJECT_ROOT", "pass", f"Valid: {PROJECT_ROOT}")
        else:
            log_test("PROJECT_ROOT", "fail", f"Does not exist: {PROJECT_ROOT}")
            return False
        
        # Test required API keys (only these need to be provided - everything else matches Docker)
        required_keys = [
            ("THE_ODDS_API_KEY", settings.the_odds_api_key),
            ("API_BASKETBALL_KEY", settings.api_basketball_key),
        ]
        
        all_keys_present = True
        for key_name, key_value in required_keys:
            if key_value:
                log_test(f"API Key: {key_name}", "pass", "Set (from env/secrets)")
            else:
                log_test(f"API Key: {key_name}", "warn", 
                    "Not set - provide via env var or secrets/ directory (matches Docker)")
                # Don't fail - API keys are expected to come from secrets/env
                # This is normal for local testing without secrets
        
        # Test required environment variables
        required_env = [
            "CURRENT_SEASON",
            "SEASONS_TO_PROCESS",
            "DATA_RAW_DIR",
            "DATA_PROCESSED_DIR",
            "THE_ODDS_BASE_URL",
            "API_BASKETBALL_BASE_URL",
        ]
        
        for env_var in required_env:
            value = getattr(settings, env_var.lower(), None)
            if value:
                log_test(f"Env Var: {env_var}", "pass", f"Set: {value[:50] if isinstance(value, str) else value}")
            else:
                log_test(f"Env Var: {env_var}", "fail", "Missing")
                all_keys_present = False
        
        # Test filter thresholds
        try:
            from src.config import filter_thresholds
            log_test("Filter Thresholds", "pass", "Loaded successfully")
        except Exception as e:
            log_test("Filter Thresholds", "fail", f"Failed to load: {e}")
            all_keys_present = False
        
        return all_keys_present
        
    except Exception as e:
        log_test("Configuration", "fail", f"Exception: {e}", traceback.format_exc())
        return False


# ============================================================================
# 2. MODEL LOADING TESTS
# ============================================================================

def test_model_loading():
    """Test that models can be loaded."""
    print("\n" + "=" * 70)
    print("2. MODEL LOADING")
    print("=" * 70)
    
    try:
        from src.config import settings
        from src.prediction import UnifiedPredictionEngine, ModelNotFoundError
        
        models_dir = Path(settings.data_processed_dir) / "models"
        
        if not models_dir.exists():
            log_test("Models Directory", "fail", f"Does not exist: {models_dir}")
            return False
        
        log_test("Models Directory", "pass", f"Exists: {models_dir}")
        
        # List model files
        model_files = list(models_dir.glob("*.pkl")) + list(models_dir.glob("*.joblib"))
        if model_files:
            log_test("Model Files", "pass", f"Found {len(model_files)} model files")
            for f in model_files[:5]:  # Show first 5
                log_test(f"  - {f.name}", "pass", f"{f.stat().st_size:,} bytes")
        else:
            log_test("Model Files", "fail", "No model files found")
            return False
        
        # Try to load engine
        try:
            engine = UnifiedPredictionEngine(models_dir=models_dir, require_all=True)
            model_info = engine.get_model_info()
            log_test("UnifiedPredictionEngine", "pass", f"Loaded successfully")
            log_test("  Markets", "pass", f"{model_info.get('markets', 0)} markets")
            log_test("  Markets List", "pass", f"{model_info.get('markets_list', [])}")
            
            # Test that we have the expected 4 markets
            expected_markets = ["1h_spread", "1h_total", "fg_spread", "fg_total"]
            actual_markets = model_info.get('markets_list', [])
            missing = [m for m in expected_markets if m not in actual_markets]
            if missing:
                log_test("Market Coverage", "fail", f"Missing markets: {missing}")
                return False
            else:
                log_test("Market Coverage", "pass", "All 4 markets present")
            
            return True
            
        except ModelNotFoundError as e:
            log_test("UnifiedPredictionEngine", "fail", f"Model not found: {e}")
            return False
        except Exception as e:
            log_test("UnifiedPredictionEngine", "fail", f"Exception: {e}", traceback.format_exc())
            return False
            
    except Exception as e:
        log_test("Model Loading", "fail", f"Exception: {e}", traceback.format_exc())
        return False


# ============================================================================
# 3. API ENDPOINT TESTS
# ============================================================================

async def test_api_endpoints():
    """Test all API endpoints."""
    print("\n" + "=" * 70)
    print("3. API ENDPOINT TESTS")
    print("=" * 70)
    
    # Note: These tests assume the API is running
    # In a real scenario, we'd start the API server or connect to it
    
    endpoints = [
        ("GET", "/health", "Health check"),
        ("GET", "/metrics", "Prometheus metrics"),
        ("GET", "/verify", "Model integrity verification"),
        ("GET", "/markets", "Market catalog"),
        ("GET", "/meta", "Metadata"),
        ("GET", "/admin/monitoring", "Monitoring stats"),
        ("GET", "/admin/cache/stats", "Cache statistics"),
    ]
    
    # Test that endpoints are defined in app.py
    try:
        import inspect
        from src.serving import app
        
        # Get all route handlers
        routes = []
        for route in app.routes:
            if hasattr(route, 'path') and hasattr(route, 'methods'):
                for method in route.methods:
                    routes.append((method, route.path))
        
        log_test("FastAPI App", "pass", f"Found {len(routes)} routes")
        
        # Check each endpoint exists
        for method, path, description in endpoints:
            if (method, path) in routes:
                log_test(f"{method} {path}", "pass", description)
            else:
                log_test(f"{method} {path}", "fail", f"Not found: {description}")
        
        return True
        
    except Exception as e:
        log_test("API Endpoints", "fail", f"Exception: {e}", traceback.format_exc())
        return False


# ============================================================================
# 4. PREDICTION ENGINE TESTS
# ============================================================================

def test_prediction_engine():
    """Test prediction engine logic."""
    print("\n" + "=" * 70)
    print("4. PREDICTION ENGINE LOGIC")
    print("=" * 70)
    
    try:
        from src.config import settings
        from src.prediction import UnifiedPredictionEngine
        
        models_dir = Path(settings.data_processed_dir) / "models"
        
        try:
            engine = UnifiedPredictionEngine(models_dir=models_dir, require_all=True)
        except Exception as e:
            log_test("Engine Initialization", "fail", f"Cannot initialize: {e}")
            return False
        
        log_test("Engine Initialization", "pass", "Engine loaded")
        
        # Test with sample features
        test_features = {
            "home_ppg": 115.0,
            "away_ppg": 112.0,
            "home_papg": 110.0,
            "away_papg": 115.0,
            "predicted_margin": 3.0,
            "predicted_total": 227.0,
            "predicted_margin_1h": 1.5,
            "predicted_total_1h": 113.5,
            "home_win_pct": 0.6,
            "away_win_pct": 0.4,
            "home_avg_margin": 2.0,
            "away_avg_margin": -1.0,
            "home_rest_days": 2,
            "away_rest_days": 1,
            "home_b2b": 0,
            "away_b2b": 0,
            "dynamic_hca": 3.0,
            "h2h_win_pct": 0.5,
            "h2h_avg_margin": 0.0,
            "home_1h_ppg": 56.2,
            "away_1h_ppg": 55.0,
            "home_1h_papg": 54.8,
            "away_1h_papg": 55.5,
            "home_1h_avg_margin": 1.2,
            "away_1h_avg_margin": -0.5,
        }
        
        # Test FG prediction
        try:
            fg_pred = engine.predict_full_game(
                features=test_features,
                spread_line=-3.5,
                total_line=225.0,
            )
            
            if "spread" in fg_pred and "total" in fg_pred:
                log_test("FG Prediction", "pass", "Returns spread and total")
                
                # Validate structure
                spread_pred = fg_pred.get("spread", {})
                total_pred = fg_pred.get("total", {})
                
                required_fields = ["side", "confidence", "edge", "passes_filter"]
                for field in required_fields:
                    if field in spread_pred:
                        log_test(f"  FG Spread.{field}", "pass", f"Present")
                    else:
                        log_test(f"  FG Spread.{field}", "fail", "Missing")
                
                for field in required_fields:
                    if field in total_pred:
                        log_test(f"  FG Total.{field}", "pass", f"Present")
                    else:
                        log_test(f"  FG Total.{field}", "fail", "Missing")
            else:
                log_test("FG Prediction", "fail", "Missing spread or total")
                return False
                
        except Exception as e:
            log_test("FG Prediction", "fail", f"Exception: {e}", traceback.format_exc())
            return False
        
        # Test 1H prediction
        try:
            h1_pred = engine.predict_first_half(
                features=test_features,
                spread_line=-1.5,
                total_line=112.5,
            )
            
            if "spread" in h1_pred and "total" in h1_pred:
                log_test("1H Prediction", "pass", "Returns spread and total")
            else:
                log_test("1H Prediction", "fail", "Missing spread or total")
                return False
                
        except Exception as e:
            log_test("1H Prediction", "fail", f"Exception: {e}", traceback.format_exc())
            return False
        
        # Test predict_all_markets
        try:
            all_preds = engine.predict_all_markets(
                features=test_features,
                fg_spread_line=-3.5,
                fg_total_line=225.0,
                fh_spread_line=-1.5,
                fh_total_line=112.5,
            )
            
            if "full_game" in all_preds and "first_half" in all_preds:
                log_test("predict_all_markets", "pass", "Returns both periods")
            else:
                log_test("predict_all_markets", "fail", "Missing periods")
                return False
                
        except Exception as e:
            log_test("predict_all_markets", "fail", f"Exception: {e}", traceback.format_exc())
            return False
        
        return True
        
    except Exception as e:
        log_test("Prediction Engine", "fail", f"Exception: {e}", traceback.format_exc())
        return False


# ============================================================================
# 5. DATA INGESTION TESTS
# ============================================================================

async def test_data_ingestion():
    """Test data ingestion pipelines."""
    print("\n" + "=" * 70)
    print("5. DATA INGESTION")
    print("=" * 70)
    
    # Track failures within this test suite
    failures_before = len(test_results["failed"])
    
    try:
        from src.config import settings
        
        # Test The Odds API client
        try:
            from src.ingestion import the_odds
            
            # Test participants endpoint
            try:
                participants = await the_odds.fetch_participants()
                if participants:
                    log_test("The Odds API - Participants", "pass", f"Fetched {len(participants)} participants")
                else:
                    log_test("The Odds API - Participants", "warn", "No participants returned")
            except Exception as e:
                error_msg = str(e).lower()
                if "401" in error_msg or "403" in error_msg:
                    log_test("The Odds API - Participants", "fail", "Authentication failed")
                else:
                    log_test("The Odds API - Participants", "warn", f"Error: {e}")
            
            # Test odds endpoint (may fail if no games scheduled)
            try:
                games = await the_odds.fetch_odds()
                if games:
                    log_test("The Odds API - Odds", "pass", f"Fetched {len(games)} games")
                else:
                    log_test("The Odds API - Odds", "warn", "No games returned (may be off-season)")
            except Exception as e:
                error_msg = str(e).lower()
                if "401" in error_msg or "403" in error_msg:
                    log_test("The Odds API - Odds", "fail", "Authentication failed")
                else:
                    log_test("The Odds API - Odds", "warn", f"Error: {e}")
                    
        except Exception as e:
            log_test("The Odds API", "fail", f"Import failed: {e}")
        
        # Test API-Basketball client
        try:
            from src.ingestion.api_basketball import APIBasketballClient
            
            client = APIBasketballClient()
            
            # Test teams endpoint
            try:
                result = await client.fetch_teams()
                if result and result.count > 0:
                    log_test("API-Basketball - Teams", "pass", f"Fetched {result.count} teams")
                else:
                    log_test("API-Basketball - Teams", "warn", "No teams returned")
            except Exception as e:
                error_msg = str(e).lower()
                if "401" in error_msg or "403" in error_msg:
                    log_test("API-Basketball - Teams", "fail", "Authentication failed")
                elif "429" in error_msg:
                    log_test("API-Basketball - Teams", "warn", "Rate limited")
                else:
                    log_test("API-Basketball - Teams", "warn", f"Error: {e}")
                    
        except Exception as e:
            log_test("API-Basketball", "fail", f"Import failed: {e}")
        
        # Test team standardization
        try:
            from src.ingestion.standardize import normalize_team_to_espn, ESPN_TEAM_NAMES
            
            test_cases = [
                ("Los Angeles Lakers", True),
                ("Lakers", True),
                ("LAL", True),
                ("Boston Celtics", True),
            ]
            
            for team_name, should_pass in test_cases:
                normalized, is_valid = normalize_team_to_espn(team_name, source="test")
                if should_pass and is_valid:
                    log_test(f"Standardization: {team_name}", "pass", f"→ {normalized}")
                elif not should_pass and not is_valid:
                    log_test(f"Standardization: {team_name}", "pass", "Correctly rejected")
                else:
                    log_test(f"Standardization: {team_name}", "fail", f"Unexpected result: {normalized}, valid={is_valid}")
            
            log_test("Team Standardization", "pass", f"{len(ESPN_TEAM_NAMES)} ESPN teams configured")
            
        except Exception as e:
            log_test("Team Standardization", "fail", f"Exception: {e}", traceback.format_exc())
        
        # Check if any failures were logged during this test suite
        failures_after = len(test_results["failed"])
        has_failures = failures_after > failures_before
        
        return not has_failures
        
    except Exception as e:
        log_test("Data Ingestion", "fail", f"Exception: {e}", traceback.format_exc())
        return False


# ============================================================================
# 6. FEATURE ENGINEERING TESTS
# ============================================================================

async def test_feature_engineering():
    """Test feature engineering."""
    print("\n" + "=" * 70)
    print("6. FEATURE ENGINEERING")
    print("=" * 70)
    
    try:
        from src.features import RichFeatureBuilder
        from src.config import settings
        
        builder = RichFeatureBuilder(season=settings.current_season)
        log_test("RichFeatureBuilder", "pass", "Initialized")
        
        # Test building features for a game
        # Use real team names
        test_home = "Cleveland Cavaliers"
        test_away = "Chicago Bulls"
        
        try:
            features = await builder.build_game_features(test_home, test_away)
            
            if features:
                log_test("build_game_features", "pass", f"Generated {len(features)} features")
                
                # Check for key features
                key_features = [
                    "home_ppg", "away_ppg",
                    "predicted_margin", "predicted_total",
                    "predicted_margin_1h", "predicted_total_1h",
                ]
                
                missing = []
                for feat in key_features:
                    if feat in features:
                        log_test(f"  Feature: {feat}", "pass", f"Present: {features[feat]}")
                    else:
                        missing.append(feat)
                        log_test(f"  Feature: {feat}", "fail", "Missing")
                
                if missing:
                    log_test("Feature Completeness", "warn", f"Missing {len(missing)} key features")
                else:
                    log_test("Feature Completeness", "pass", "All key features present")
            else:
                log_test("build_game_features", "fail", "Returned empty features")
                return False
                
        except Exception as e:
            log_test("build_game_features", "fail", f"Exception: {e}", traceback.format_exc())
            return False
        
        return True
        
    except Exception as e:
        log_test("Feature Engineering", "fail", f"Exception: {e}", traceback.format_exc())
        return False


# ============================================================================
# 7. ERROR HANDLING TESTS
# ============================================================================

def test_error_handling():
    """Test error handling and edge cases."""
    print("\n" + "=" * 70)
    print("7. ERROR HANDLING & EDGE CASES")
    print("=" * 70)
    
    # Track failures within this test suite
    failures_before = len(test_results["failed"])
    
    try:
        from src.ingestion.standardize import normalize_team_to_espn
        
        # Test invalid inputs
        invalid_inputs = [
            None,
            "",
            "   ",
            "Invalid Team Name XYZ",
            "12345",
        ]
        
        for invalid_input in invalid_inputs:
            try:
                # Actually pass None if it's None - don't convert to empty string
                # This tests the function's actual None handling behavior
                result, is_valid = normalize_team_to_espn(invalid_input, source="test")
                
                if not is_valid:
                    log_test(f"Invalid Input: {repr(invalid_input)}", "pass", "Correctly rejected")
                else:
                    log_test(f"Invalid Input: {repr(invalid_input)}", "fail", f"Incorrectly accepted: {result}")
            except TypeError as e:
                # If function doesn't handle None and raises TypeError, that's a failure
                log_test(f"Invalid Input: {repr(invalid_input)}", "fail", f"TypeError (doesn't handle None): {e}")
            except Exception as e:
                log_test(f"Invalid Input: {repr(invalid_input)}", "fail", f"Crashed: {e}")
        
        # Test missing model files
        try:
            from src.prediction import UnifiedPredictionEngine, ModelNotFoundError
            from src.config import settings
            
            fake_dir = Path("/tmp/nonexistent_models")
            try:
                engine = UnifiedPredictionEngine(models_dir=fake_dir, require_all=True)
                log_test("Missing Models", "fail", "Should have raised ModelNotFoundError")
            except ModelNotFoundError:
                log_test("Missing Models", "pass", "Correctly raises ModelNotFoundError")
            except Exception as e:
                log_test("Missing Models", "warn", f"Raised different exception: {e}")
        except Exception as e:
            log_test("Missing Models Test", "fail", f"Exception: {e}")
        
        # Check if any failures were logged during this test suite
        failures_after = len(test_results["failed"])
        has_failures = failures_after > failures_before
        
        return not has_failures
        
    except Exception as e:
        log_test("Error Handling", "fail", f"Exception: {e}", traceback.format_exc())
        return False


# ============================================================================
# 8. AZURE FUNCTION TESTS
# ============================================================================

def test_azure_function():
    """Test Azure Function endpoints."""
    print("\n" + "=" * 70)
    print("8. AZURE FUNCTION ENDPOINTS")
    print("=" * 70)
    
    # Track failures within this test suite
    failures_before = len(test_results["failed"])
    
    try:
        # Check that function_app.py exists and has required endpoints
        func_file = PROJECT_ROOT / "azure" / "function_app" / "function_app.py"
        
        if not func_file.exists():
            log_test("Function App File", "fail", f"Does not exist: {func_file}")
            return False
        
        log_test("Function App File", "pass", f"Exists: {func_file}")
        
        # Read file and check for endpoints
        content = func_file.read_text()
        
        expected_endpoints = [
            ("nba-picks", "Main trigger endpoint"),
            ("menu", "Menu endpoint"),
            ("health", "Health check"),
            ("weekly-lineup/nba", "Website integration"),
            ("bot", "Teams bot endpoint"),
        ]
        
        for endpoint, description in expected_endpoints:
            if f'route="{endpoint}"' in content or f"route='{endpoint}'" in content:
                log_test(f"Function: {endpoint}", "pass", description)
            else:
                log_test(f"Function: {endpoint}", "fail", f"Not found: {description}")
        
        # Check for required functions
        required_functions = [
            "fetch_predictions",
            "clear_api_cache",
            "format_teams_card",
            "post_to_teams",
        ]
        
        for func_name in required_functions:
            if f"def {func_name}" in content:
                log_test(f"Function: {func_name}", "pass", "Defined")
            else:
                log_test(f"Function: {func_name}", "fail", "Not defined")
        
        # Check if any failures were logged during this test suite
        failures_after = len(test_results["failed"])
        has_failures = failures_after > failures_before
        
        return not has_failures
        
    except Exception as e:
        log_test("Azure Function", "fail", f"Exception: {e}", traceback.format_exc())
        return False


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

async def main():
    """Run all production readiness tests."""
    print("=" * 70)
    print("PRODUCTION READINESS TEST SUITE")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project: {PROJECT_ROOT}")
    print()
    print("NOTE: Environment variables match Dockerfile defaults exactly.")
    print("      Only API keys need to be provided (via env/secrets).")
    print()
    
    # Run all test suites
    test_suites = [
        ("Configuration", test_configuration, False),
        ("Model Loading", test_model_loading, False),
        ("API Endpoints", test_api_endpoints, True),
        ("Prediction Engine", test_prediction_engine, False),
        ("Data Ingestion", test_data_ingestion, True),
        ("Feature Engineering", test_feature_engineering, True),
        ("Error Handling", test_error_handling, False),
        ("Azure Function", test_azure_function, False),
    ]
    
    results = {}
    for name, test_func, is_async in test_suites:
        try:
            if is_async:
                result = await test_func()
            else:
                result = test_func()
            results[name] = result
        except Exception as e:
            print(f"\n[ERROR] {name} suite crashed: {e}")
            traceback.print_exc()
            results[name] = False
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    total_tests = len(test_results["passed"]) + len(test_results["failed"]) + len(test_results["warnings"])
    
    print(f"\nTest Suites:")
    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nIndividual Tests:")
    print(f"  Passed: {len(test_results['passed'])}")
    print(f"  Failed: {len(test_results['failed'])}")
    print(f"  Warnings: {len(test_results['warnings'])}")
    print(f"  Skipped: {len(test_results['skipped'])}")
    
    if test_results["failed"]:
        print(f"\nFailed Tests:")
        for result in test_results["failed"]:
            print(f"  - {result['name']}: {result['message']}")
    
    # Save results to file
    results_file = PROJECT_ROOT / "production_readiness_test_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "test_suites": results,
            "individual_tests": test_results,
            "summary": {
                "total_tests": total_tests,
                "passed": len(test_results["passed"]),
                "failed": len(test_results["failed"]),
                "warnings": len(test_results["warnings"]),
                "skipped": len(test_results["skipped"]),
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Determine overall status
    all_suites_passed = all(results.values())
    no_critical_failures = len(test_results["failed"]) == 0
    
    print("\n" + "=" * 70)
    if all_suites_passed and no_critical_failures:
        print("[PASS] PRODUCTION READY: All tests passed!")
        return 0
    elif no_critical_failures:
        print("[WARN] MOSTLY READY: Some warnings, but no critical failures")
        return 0
    else:
        print("[FAIL] NOT READY: Critical failures detected")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))


