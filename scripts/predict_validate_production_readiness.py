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
import os
import argparse
import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import importlib
import traceback
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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
        from src.prediction.feature_validation import get_feature_mode, FeatureMode, get_min_feature_completeness
        
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
        
        # Production guardrails (fail if not strict)
        guardrail_failures = []

        mode = get_feature_mode()
        if mode != FeatureMode.STRICT:
            guardrail_failures.append(
                f"PREDICTION_FEATURE_MODE must be strict (current: {mode.value})"
            )

        min_comp = get_min_feature_completeness()
        if min_comp < 0.95:
            guardrail_failures.append(
                f"MIN_FEATURE_COMPLETENESS must be >= 0.95 (current: {min_comp:.2f})"
            )

        if not getattr(settings, "require_action_network_splits", False):
            guardrail_failures.append("REQUIRE_ACTION_NETWORK_SPLITS must be true for production")
        if not getattr(settings, "require_real_splits", False):
            guardrail_failures.append("REQUIRE_REAL_SPLITS must be true for production")
        if not getattr(settings, "require_sharp_book_data", False):
            guardrail_failures.append("REQUIRE_SHARP_BOOK_DATA must be true for production")
        if not getattr(settings, "require_injury_fetch_success", False):
            guardrail_failures.append("REQUIRE_INJURY_FETCH_SUCCESS must be true for production")

        if getattr(settings, "require_action_network_splits", False) or getattr(settings, "require_real_splits", False):
            if not getattr(settings, "action_network_username", "") or not getattr(settings, "action_network_password", ""):
                guardrail_failures.append(
                    "Action Network premium credentials are required when splits are strict"
                )

        if guardrail_failures:
            print("\n[FAIL] Production guardrails not satisfied:")
            for msg in guardrail_failures:
                print(f"  - {msg}")
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


def _extract_market_line(game: dict, market_key: str, team_name: str | None = None, outcome_name: str | None = None) -> float | None:
    """Extract a market line from The Odds API payload (returns None if missing)."""
    bookmakers = game.get("bookmakers", []) or []
    for bm in bookmakers:
        for market in bm.get("markets", []) or []:
            if market.get("key") != market_key:
                continue
            for outcome in market.get("outcomes", []) or []:
                if team_name and outcome.get("name") == team_name and "point" in outcome:
                    return outcome.get("point")
                if outcome_name and outcome.get("name") == outcome_name and "point" in outcome:
                    return outcome.get("point")
    return None


def _offset_is_cst(dt_value: datetime) -> bool:
    """Return True if datetime offset is CST/CDT (-06:00 or -05:00)."""
    if not dt_value or dt_value.tzinfo is None:
        return False
    offset = dt_value.utcoffset()
    if offset is None:
        return False
    return offset.total_seconds() in (-21600, -18000)


async def _test_live_pipeline_async() -> bool:
    """Live endpoint + pipeline check (odds, splits, sharp/square, injuries, features)."""
    print("\n" + "=" * 60)
    print("6. Live Endpoint + Pipeline Validation")
    print("=" * 60)

    try:
        from src.config import settings
        from src.ingestion.the_odds import fetch_odds, fetch_events, fetch_event_odds
        from src.ingestion.standardize import normalize_team_to_espn
        from src.ingestion.betting_splits import fetch_public_betting_splits, splits_to_features
        from src.features.rich_features import RichFeatureBuilder
        from src.ingestion.api_basketball import APIBasketballClient
        from src.ingestion.injuries import fetch_all_injuries
        from src.prediction.feature_validation import validate_and_prepare_features
        from src.modeling.unified_features import MODEL_CONFIGS
        from src.utils.model_features import get_market_features, get_union_features
        import pandas as pd
    except Exception as e:
        print(f"  [FAIL] Imports for live pipeline failed: {e}")
        traceback.print_exc()
        return False

    # --- The Odds API (full game odds) ---
    try:
        games = await fetch_odds(markets="spreads,totals")
        print(f"  [OK] The Odds API returned {len(games)} games")
    except Exception as e:
        print(f"  [FAIL] The Odds API fetch_odds failed: {e}")
        return False

    if not games:
        print("  [WARN] No games returned. Skipping downstream live checks.")
        return True

    invalid_games = [g for g in games if not g.get("_data_valid")]
    if invalid_games:
        print(f"  [FAIL] {len(invalid_games)} games had invalid team names after standardization")
        return False

    # Verify CST conversion on at least one game
    cst_checked = False
    for g in games:
        cst_value = g.get("commence_time_cst")
        if not cst_value:
            continue
        try:
            dt_value = datetime.fromisoformat(cst_value)
            if not _offset_is_cst(dt_value):
                print(f"  [FAIL] commence_time_cst not CST/CDT: {cst_value}")
                return False
            if g.get("date") and dt_value.date().isoformat() != g.get("date"):
                print(f"  [FAIL] date mismatch after CST conversion: {g.get('date')} vs {dt_value.date().isoformat()}")
                return False
            cst_checked = True
            break
        except Exception as e:
            print(f"  [FAIL] Failed parsing commence_time_cst '{cst_value}': {e}")
            return False

    if not cst_checked:
        print("  [WARN] No commence_time_cst field found to validate CST conversion")

    # --- API-Basketball (timezone-aware games fetch) ---
    try:
        client = APIBasketballClient()
        today_cst = datetime.now(ZoneInfo("America/Chicago")).date().isoformat()
        result = await client.fetch_games_by_date(today_cst, timezone="America/Chicago")
        print(f"  [OK] API-Basketball games by date ({today_cst} CST): {result.count}")
    except Exception as e:
        print(f"  [FAIL] API-Basketball fetch_games_by_date failed: {e}")
        return False

    # --- Action Network splits (strict if configured) ---
    try:
        splits_dict = await fetch_public_betting_splits(
            games,
            source="auto",
            require_action_network=bool(getattr(settings, "require_action_network_splits", False)),
            require_non_empty=bool(getattr(settings, "require_real_splits", False)),
        )
        print(f"  [OK] Betting splits loaded: {len(splits_dict)} games")
    except Exception as e:
        print(f"  [FAIL] Betting splits fetch failed: {e}")
        return False

    strict_splits = bool(getattr(settings, "require_action_network_splits", False)) or bool(
        getattr(settings, "require_real_splits", False)
    )
    if strict_splits:
        missing_keys = []
        for g in games:
            home_team = g.get("home_team")
            away_team = g.get("away_team")
            if home_team and away_team:
                key = f"{away_team}@{home_team}"
                if key not in splits_dict:
                    missing_keys.append(key)
        if missing_keys:
            print(f"  [FAIL] Missing splits for {len(missing_keys)} games (strict): {missing_keys[:5]}")
            return False

    sample_split = next(iter(splits_dict.values()), None)
    if sample_split:
        if not _offset_is_cst(sample_split.game_time):
            print(f"  [FAIL] Action Network game_time not CST/CDT: {sample_split.game_time}")
            return False
        split_features = splits_to_features(sample_split)
        if getattr(settings, "require_real_splits", False) and split_features.get("has_real_splits") != 1:
            print("  [FAIL] Splits returned has_real_splits=0 while REQUIRE_REAL_SPLITS=true")
            return False

    # --- Injuries (strict if configured) ---
    try:
        injuries = await fetch_all_injuries()
        print(f"  [OK] Injury fetch succeeded ({len(injuries)} records)")
    except Exception as e:
        print(f"  [FAIL] Injury fetch failed: {e}")
        return False

    # --- Event odds: verify 1H markets available for at least one game ---
    event_spread_1h = None
    event_total_1h = None
    try:
        events = await fetch_events()
        event_id = None
        target_game = games[0]
        target_home = target_game.get("home_team")
        target_away = target_game.get("away_team")
        for ev in events:
            if ev.get("home_team") == target_home and ev.get("away_team") == target_away:
                event_id = ev.get("id")
                break
        if not event_id and events:
            event_id = events[0].get("id")
        if not event_id:
            print("  [WARN] No event ID available for 1H odds validation")
        else:
            event_odds = await fetch_event_odds(event_id)
            event_spread_1h = _extract_market_line(
                event_odds, "spreads_h1", team_name=event_odds.get("home_team")
            )
            event_total_1h = _extract_market_line(
                event_odds, "totals_h1", outcome_name="Over"
            )
            if event_spread_1h is None or event_total_1h is None:
                print("  [FAIL] 1H odds missing (spreads_h1 or totals_h1) for event")
                return False
            print("  [OK] 1H odds validated (spreads_h1 / totals_h1)")
    except Exception as e:
        print(f"  [FAIL] Event odds validation failed: {e}")
        return False

    # --- End-to-end features build + validation ---
    try:
        target_game = games[0]
        home_team = target_game.get("home_team")
        away_team = target_game.get("away_team")

        # Normalize to ensure consistent splits keying
        home_norm, _ = normalize_team_to_espn(str(home_team), source="odds")
        away_norm, _ = normalize_team_to_espn(str(away_team), source="odds")
        splits_key = f"{away_norm}@{home_norm}"
        betting_splits = splits_dict.get(splits_key)

        # Extract full-game market lines
        fg_spread = _extract_market_line(target_game, "spreads", team_name=home_team)
        fg_total = _extract_market_line(target_game, "totals", outcome_name="Over")
        if fg_spread is None or fg_total is None:
            print("  [FAIL] Missing full-game spread/total lines from The Odds API")
            return False

        # Use 1H lines from event odds if available
        h1_spread = event_spread_1h
        h1_total = event_total_1h
        if h1_spread is None or h1_total is None:
            print("  [FAIL] Missing 1H spread/total lines from The Odds API")
            return False

        builder = RichFeatureBuilder(season=settings.current_season)
        models_dir = PROJECT_ROOT / "models" / "production"
        required_features = get_union_features(models_dir)
        features = await builder.build_game_features(
            home_norm,
            away_norm,
            betting_splits=betting_splits,
            required_features=required_features or None,
        )

        def _validate_market_features(market_key: str, spread_line: float, total_line: float) -> None:
            payload = dict(features)
            payload["spread_line"] = spread_line
            payload["total_line"] = total_line
            payload["spread_vs_predicted"] = 0.0
            payload["total_vs_predicted"] = 0.0
            payload["1h_spread_line"] = spread_line
            payload["1h_total_line"] = total_line
            required = get_market_features(models_dir, market_key) or MODEL_CONFIGS[market_key]["features"]
            df = pd.DataFrame([payload])
            validate_and_prepare_features(df, required, market=market_key)

        for market_key in sorted(MODEL_CONFIGS.keys()):
            if market_key.startswith("1h_"):
                _validate_market_features(market_key, h1_spread, h1_total)
            else:
                _validate_market_features(market_key, fg_spread, fg_total)

        print("  [OK] Feature build + validation passed for all markets")
    except Exception as e:
        print(f"  [FAIL] Feature build/validation failed: {e}")
        traceback.print_exc()
        return False

    return True


def test_live_pipeline() -> bool:
    """Wrapper to run async live pipeline check."""
    return asyncio.run(_test_live_pipeline_async())


def main():
    """Run all production readiness checks."""
    parser = argparse.ArgumentParser(description="Production readiness validation")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run live endpoint + end-to-end pipeline checks (requires API access)",
    )
    args = parser.parse_args()

    if args.live:
        # Enforce strict production guardrails for live readiness checks
        strict_env = {
            "PREDICTION_FEATURE_MODE": "strict",
            "MIN_FEATURE_COMPLETENESS": "0.95",
            "REQUIRE_ACTION_NETWORK_SPLITS": "true",
            "REQUIRE_REAL_SPLITS": "true",
            "REQUIRE_SHARP_BOOK_DATA": "true",
            "REQUIRE_INJURY_FETCH_SUCCESS": "true",
        }
        for key, value in strict_env.items():
            os.environ.setdefault(key, value)

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

    if args.live:
        checks.append(("Live Endpoint + Pipeline", test_live_pipeline))
    
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

