#!/usr/bin/env python3
"""
Test All API Endpoints

Tests all API endpoints to verify they're working correctly.
Tests both The Odds API and API-Basketball endpoints.
"""
import sys
import asyncio
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.ingestion import the_odds
from src.ingestion.api_basketball import APIBasketballClient
from src.utils.logging import get_logger

logger = get_logger(__name__)


async def test_the_odds_api():
    """Test The Odds API endpoints."""
    print("\n" + "=" * 60)
    print("TESTING THE ODDS API")
    print("=" * 60)
    
    if not settings.the_odds_api_key:
        print("  [SKIP] THE_ODDS_API_KEY not set")
        return {"status": "skipped", "tests": []}
    
    results = []
    
    # Test 1: Main odds endpoint
    print("\n1. Testing /sports/basketball_nba/odds...")
    try:
        games = await the_odds.fetch_odds()
        if games and len(games) > 0:
            print(f"  [OK] Successfully fetched {len(games)} games")
            print(f"  [OK] Sample game: {games[0].get('home_team', 'N/A')} vs {games[0].get('away_team', 'N/A')}")
            results.append({"endpoint": "/sports/basketball_nba/odds", "status": "pass", "count": len(games)})
        else:
            print(f"  [WARN] No games returned (may be off-season)")
            results.append({"endpoint": "/sports/basketball_nba/odds", "status": "warn", "count": 0})
    except Exception as e:
        error_msg = str(e).lower()
        if "401" in error_msg or "403" in error_msg or "unauthorized" in error_msg:
            print(f"  [FAIL] Authentication failed - check API key")
            results.append({"endpoint": "/sports/basketball_nba/odds", "status": "fail", "error": "auth"})
        else:
            print(f"  [FAIL] Error: {e}")
            results.append({"endpoint": "/sports/basketball_nba/odds", "status": "fail", "error": str(e)})
    
    # Test 2: Events endpoint
    print("\n2. Testing /sports/basketball_nba/events...")
    try:
        events = await the_odds.fetch_events()
        if events and len(events) > 0:
            print(f"  [OK] Successfully fetched {len(events)} events")
            results.append({"endpoint": "/sports/basketball_nba/events", "status": "pass", "count": len(events)})
        else:
            print(f"  [WARN] No events returned")
            results.append({"endpoint": "/sports/basketball_nba/events", "status": "warn", "count": 0})
    except Exception as e:
        error_msg = str(e).lower()
        if "401" in error_msg or "403" in error_msg:
            print(f"  [FAIL] Authentication failed")
            results.append({"endpoint": "/sports/basketball_nba/events", "status": "fail", "error": "auth"})
        else:
            print(f"  [FAIL] Error: {e}")
            results.append({"endpoint": "/sports/basketball_nba/events", "status": "fail", "error": str(e)})
    
    # Test 3: Scores endpoint
    print("\n3. Testing /sports/basketball_nba/scores...")
    try:
        scores = await the_odds.fetch_scores()
        if scores and len(scores) > 0:
            print(f"  [OK] Successfully fetched {len(scores)} recent scores")
            results.append({"endpoint": "/sports/basketball_nba/scores", "status": "pass", "count": len(scores)})
        else:
            print(f"  [WARN] No scores returned")
            results.append({"endpoint": "/sports/basketball_nba/scores", "status": "warn", "count": 0})
    except Exception as e:
        error_msg = str(e).lower()
        if "401" in error_msg or "403" in error_msg:
            print(f"  [FAIL] Authentication failed")
            results.append({"endpoint": "/sports/basketball_nba/scores", "status": "fail", "error": "auth"})
        else:
            print(f"  [FAIL] Error: {e}")
            results.append({"endpoint": "/sports/basketball_nba/scores", "status": "fail", "error": str(e)})
    
    return {"status": "completed", "tests": results}


async def test_api_basketball():
    """Test API-Basketball endpoints."""
    print("\n" + "=" * 60)
    print("TESTING API-BASKETBALL")
    print("=" * 60)
    
    if not settings.api_basketball_key:
        print("  [SKIP] API_BASKETBALL_KEY not set")
        return {"status": "skipped", "tests": []}
    
    client = APIBasketballClient()
    results = []
    
    # Test Tier 1 endpoints (essential)
    tier1_endpoints = [
        ("teams", client.fetch_teams),
        ("games", client.fetch_games),
        ("statistics", client.fetch_statistics),
    ]
    
    print("\nTesting TIER 1 - Essential Endpoints:")
    for name, fetch_fn in tier1_endpoints:
        print(f"\n  Testing /{name}...")
        try:
            result = await fetch_fn()
            if result and result.count > 0:
                print(f"    [OK] Successfully fetched {result.count} records")
                results.append({"endpoint": f"/{name}", "status": "pass", "count": result.count, "tier": 1})
            else:
                print(f"    [WARN] No records returned")
                results.append({"endpoint": f"/{name}", "status": "warn", "count": 0, "tier": 1})
        except Exception as e:
            error_msg = str(e).lower()
            if "401" in error_msg or "403" in error_msg or "unauthorized" in error_msg:
                print(f"    [FAIL] Authentication failed - check API key")
                results.append({"endpoint": f"/{name}", "status": "fail", "error": "auth", "tier": 1})
            elif "429" in error_msg or "rate limit" in error_msg:
                print(f"    [WARN] Rate limited - may need to wait")
                results.append({"endpoint": f"/{name}", "status": "warn", "error": "rate_limit", "tier": 1})
            else:
                print(f"    [FAIL] Error: {e}")
                results.append({"endpoint": f"/{name}", "status": "fail", "error": str(e), "tier": 1})
    
    return {"status": "completed", "tests": results}


async def main():
    """Run all API endpoint tests."""
    print("=" * 60)
    print("API ENDPOINT TESTING")
    print("=" * 60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project: {PROJECT_ROOT}")
    print()
    
    # Check API keys
    print("API Key Status:")
    print(f"  THE_ODDS_API_KEY: {'[SET]' if settings.the_odds_api_key else '[NOT SET]'}")
    print(f"  API_BASKETBALL_KEY: {'[SET]' if settings.api_basketball_key else '[NOT SET]'}")
    print()
    
    all_results = {}
    
    # Test The Odds API
    all_results["the_odds"] = await test_the_odds_api()
    
    # Test API-Basketball
    all_results["api_basketball"] = await test_api_basketball()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    total_tests = 0
    passed = 0
    failed = 0
    warned = 0
    skipped = 0
    
    for api_name, api_results in all_results.items():
        if api_results["status"] == "skipped":
            print(f"\n{api_name.upper()}: [SKIPPED] - API key not set")
            skipped += 1
            continue
        
        print(f"\n{api_name.upper()}:")
        for test in api_results["tests"]:
            total_tests += 1
            status = test["status"]
            endpoint = test["endpoint"]
            
            if status == "pass":
                passed += 1
                count = test.get("count", 0)
                print(f"  [PASS] {endpoint} - {count} records")
            elif status == "warn":
                warned += 1
                error = test.get("error", "")
                if error:
                    print(f"  [WARN] {endpoint} - {error}")
                else:
                    print(f"  [WARN] {endpoint} - No data returned")
            elif status == "fail":
                failed += 1
                error = test.get("error", "Unknown error")
                print(f"  [FAIL] {endpoint} - {error}")
    
    print("\n" + "=" * 60)
    print(f"Total: {total_tests} tests")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Warnings: {warned}")
    print(f"  Skipped: {skipped}")
    print("=" * 60)
    
    if failed == 0 and passed > 0:
        print("\n[OK] All tested endpoints are working!")
        return 0
    elif failed > 0:
        print("\n[FAIL] Some endpoints are not working. Check API keys and network.")
        return 1
    else:
        print("\n[WARN] No endpoints tested (keys not set or all skipped)")
        return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
