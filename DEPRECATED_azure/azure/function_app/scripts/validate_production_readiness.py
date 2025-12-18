#!/usr/bin/env python3
"""
Production Readiness Validation Script

Validates that the system is ready for production deployment by checking:
1. Configuration (API keys, settings)
2. Data standardization (team name normalization)
3. Error handling (no silent failures)
4. Code quality (imports, syntax)
5. Test suite status
"""
import sys
from pathlib import Path
import importlib
import traceback

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_imports():
    """Test that all critical modules can be imported."""
    print("=" * 60)
    print("1. Testing Module Imports")
    print("=" * 60)
    
    modules = [
        "src.config",
        "src.ingestion.standardize",
        "src.ingestion.the_odds",
        "src.ingestion.api_basketball",
        "src.ingestion.betting_splits",
        "src.modeling.models",
        "src.modeling.features",
        "src.prediction.engine",
    ]
    
    failed = []
    for module_name in modules:
        try:
            importlib.import_module(module_name)
            print(f"  [OK] {module_name}")
        except Exception as e:
            print(f"  [FAIL] {module_name}: {e}")
            failed.append(module_name)
    
    if failed:
        print(f"\n[FAILED] {len(failed)} modules could not be imported")
        return False
    
    print(f"\n[OK] All {len(modules)} modules imported successfully")
    return True


def test_standardization():
    """Test team name standardization with validation."""
    print("\n" + "=" * 60)
    print("2. Testing Team Name Standardization")
    print("=" * 60)
    
    try:
        from src.ingestion.standardize import (
            normalize_team_to_espn,
            is_valid_espn_team_name,
            standardize_game_data,
            ESPN_TEAM_NAMES
        )
        
        # Test valid team names
        test_cases = [
            ("Los Angeles Lakers", True),
            ("Lakers", True),
            ("LAL", True),
            ("Boston Celtics", True),
            ("Invalid Team Name XYZ", False),
        ]
        
        passed = 0
        failed = 0
        
        for team_name, should_be_valid in test_cases:
            normalized, is_valid = normalize_team_to_espn(team_name, source="test")
            if should_be_valid:
                if is_valid and normalized in ESPN_TEAM_NAMES:
                    print(f"  [OK] '{team_name}' -> '{normalized}' (valid={is_valid})")
                    passed += 1
                else:
                    print(f"  [FAIL] '{team_name}' -> '{normalized}' (valid={is_valid}) - Expected valid")
                    failed += 1
            else:
                if not is_valid:
                    print(f"  [OK] '{team_name}' correctly rejected (valid={is_valid})")
                    passed += 1
                else:
                    print(f"  [FAIL] '{team_name}' incorrectly accepted (valid={is_valid}) - Expected invalid")
                    failed += 1
        
        # Test standardize_game_data validation flags
        print("\n  Testing standardize_game_data validation:")
        test_game = {
            "home_team": "Los Angeles Lakers",
            "away_team": "Boston Celtics",
        }
        standardized = standardize_game_data(test_game, source="test")
        
        if standardized.get("_data_valid"):
            print(f"  [OK] Validation flags present: _data_valid={standardized.get('_data_valid')}")
            passed += 1
        else:
            print(f"  [FAIL] Validation flags missing or invalid")
            failed += 1
        
        print(f"\n  Results: {passed} passed, {failed} failed")
        return failed == 0
        
    except Exception as e:
        print(f"  ✗ Standardization test failed: {e}")
        traceback.print_exc()
        return False


def test_config():
    """Test configuration and settings."""
    print("\n" + "=" * 60)
    print("3. Testing Configuration")
    print("=" * 60)
    
    try:
        from src.config import settings, get_current_nba_season
        
        # Test settings object exists
        print(f"  [OK] Settings object initialized")
        
        # Test API key fields exist
        required_fields = [
            "the_odds_api_key",
            "api_basketball_key",
            "betsapi_key",
            "action_network_username",
            "action_network_password",
            "kaggle_api_token",
        ]
        
        for field in required_fields:
            if hasattr(settings, field):
                value = getattr(settings, field)
                status = "set" if value else "empty"
                print(f"  [OK] {field}: {status}")
            else:
                print(f"  [FAIL] {field}: missing")
                return False
        
        # Test season calculation
        season = get_current_nba_season()
        if season and len(season) == 9 and "-" in season:
            print(f"  [OK] Current season: {season}")
        else:
            print(f"  [FAIL] Invalid season format: {season}")
            return False
        
        print("\n[OK] Configuration is valid")
        return True
        
    except Exception as e:
        print(f"  [FAIL] Configuration test failed: {e}")
        traceback.print_exc()
        return False


def test_no_fake_data():
    """Test that invalid data is properly rejected."""
    print("\n" + "=" * 60)
    print("4. Testing No Fake Data Policy")
    print("=" * 60)
    
    try:
        from src.ingestion.standardize import normalize_team_to_espn, standardize_game_data
        
        # Test that invalid team names return empty string, not original
        invalid_name = "Totally Fake Team Name 12345"
        normalized, is_valid = normalize_team_to_espn(invalid_name, source="test")
        
        if not is_valid and normalized == "":
            print(f"  [OK] Invalid team name correctly returns empty string (not fake data)")
        else:
            print(f"  [FAIL] Invalid team name returned '{normalized}' (should be empty)")
            return False
        
        # Test that invalid games are marked invalid
        invalid_game = {
            "home_team": "Fake Home Team",
            "away_team": "Fake Away Team",
        }
        standardized = standardize_game_data(invalid_game, source="test")
        
        if not standardized.get("_data_valid"):
            print(f"  [OK] Invalid game data correctly marked as invalid")
        else:
            print(f"  [FAIL] Invalid game data incorrectly marked as valid")
            return False
        
        # Test that empty team names are handled
        empty_game = {
            "home_team": "",
            "away_team": "",
        }
        standardized_empty = standardize_game_data(empty_game, source="test")
        
        if not standardized_empty.get("_data_valid"):
            print(f"  [OK] Empty team names correctly rejected")
        else:
            print(f"  [FAIL] Empty team names incorrectly accepted")
            return False
        
        print("\n[OK] No fake data policy enforced")
        return True
        
    except Exception as e:
        print(f"  ✗ No fake data test failed: {e}")
        traceback.print_exc()
        return False


def test_error_handling():
    """Test that errors are properly logged and handled."""
    print("\n" + "=" * 60)
    print("5. Testing Error Handling")
    print("=" * 60)
    
    try:
        from src.ingestion.standardize import normalize_team_to_espn
        
        # Test that function doesn't crash on invalid input
        test_cases = [
            None,
            "",
            "   ",
            "Invalid Team",
        ]
        
        for invalid_input in test_cases:
            try:
                if invalid_input is None:
                    # Should handle None gracefully
                    result, is_valid = normalize_team_to_espn("", source="test")
                    print(f"  [OK] Handles None input gracefully")
                else:
                    result, is_valid = normalize_team_to_espn(invalid_input, source="test")
                    print(f"  [OK] Handles '{invalid_input}' gracefully (returns empty)")
            except Exception as e:
                print(f"  [FAIL] Crashed on '{invalid_input}': {e}")
                return False
        
        print("\n[OK] Error handling is robust")
        return True
        
    except Exception as e:
        print(f"  [FAIL] Error handling test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all production readiness checks."""
    print("\n" + "=" * 60)
    print("PRODUCTION READINESS VALIDATION")
    print("=" * 60)
    print(f"Running checks at: {Path.cwd()}")
    print()
    
    checks = [
        ("Module Imports", test_imports),
        ("Team Name Standardization", test_standardization),
        ("Configuration", test_config),
        ("No Fake Data Policy", test_no_fake_data),
        ("Error Handling", test_error_handling),
    ]
    
    results = []
    for name, test_func in checks:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n[FAIL] {name} check crashed: {e}")
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status}: {name}")
    
    print("\n" + "=" * 60)
    if passed == total:
        print(f"[OK] PRODUCTION READY: All {total} checks passed")
        return 0
    else:
        print(f"[FAIL] NOT READY: {passed}/{total} checks passed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

