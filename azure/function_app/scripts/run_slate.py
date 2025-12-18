#!/usr/bin/env python3
"""
NBA Slate Runner - THE SINGLE COMMAND FOR PREDICTIONS
======================================================
This is the ONE command you run for predictions. No confusion.

Usage:
    python scripts/run_slate.py                      # Today's full slate
    python scripts/run_slate.py --date tomorrow     # Tomorrow's slate
    python scripts/run_slate.py --date 2025-12-19   # Specific date
    python scripts/run_slate.py --matchup "Lakers vs Celtics"  # Specific game

Requirements:
    - Docker must be running
    - .env file with API keys

This script:
    1. Ensures Docker stack is running
    2. Waits for API to be healthy
    3. Fetches predictions from strict-api container
    4. Displays formatted results with fire ratings
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

# Fix Windows console encoding
import io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).parent.parent
API_URL = "http://localhost:8090"


def http_get_json(url: str, params: dict | None = None, timeout: int = 30) -> dict:
    """HTTP GET returning parsed JSON (stdlib only)."""
    full_url = f"{url}?{urlencode(params)}" if params else url
    try:
        with urlopen(full_url, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return json.loads(body)
    except HTTPError as e:
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        raise RuntimeError(f"HTTP {e.code} from {full_url}: {body}".strip()) from e
    except URLError as e:
        raise RuntimeError(f"Failed to reach {full_url}: {e}") from e
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON from {full_url}: {e}") from e


def check_docker_running() -> bool:
    """Check if Docker is running."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=10
        )
        return result.returncode == 0
    except Exception:
        return False


def check_stack_running() -> bool:
    """Check if the NBA stack is running."""
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=nba-api", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return "nba-api" in result.stdout
    except Exception:
        return False


def start_stack():
    """Start the Docker stack."""
    print("🐳 Starting Docker stack...")
    result = subprocess.run(
        ["docker", "compose", "up", "-d"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"❌ Failed to start stack: {result.stderr}")
        sys.exit(1)
    print("✅ Stack started")


def wait_for_api(max_wait: int = 60) -> bool:
    """Wait for API to be healthy."""
    print("⏳ Waiting for API to be ready...")
    start = time.time()
    
    while time.time() - start < max_wait:
        try:
            data = http_get_json(f"{API_URL}/health", timeout=5)
            if data.get("engine_loaded"):
                print("✅ API ready (engine loaded)")
                return True
            print("⚠️  API up but engine not loaded - models may be missing")
            return True
        except Exception:
            pass
        time.sleep(2)
    
    print("❌ API did not become ready in time")
    return False


def calculate_fire_rating(confidence: float, edge: float, edge_type: str = "pts") -> str:
    """Calculate fire rating (1-5 fires)."""
    if edge_type == "pct":
        edge_norm = min(abs(edge) / 0.20, 1.0)
    else:
        edge_norm = min(abs(edge) / 10.0, 1.0)
    
    combined = (confidence * 0.6) + (edge_norm * 0.4)
    
    if combined >= 0.85:
        return "🔥🔥🔥🔥🔥"
    elif combined >= 0.70:
        return "🔥🔥🔥🔥"
    elif combined >= 0.60:
        return "🔥🔥🔥"
    elif combined >= 0.52:
        return "🔥🔥"
    else:
        return "🔥"


def format_odds(odds: int) -> str:
    """Format American odds."""
    return f"+{odds}" if odds > 0 else str(odds)


def _match_game(game: dict, matchup_filter: str) -> bool:
    """Match on single team name or 'teamA vs teamB' style strings."""
    home = (game.get("home_team") or "").lower()
    away = (game.get("away_team") or "").lower()
    raw = (matchup_filter or "").strip().lower()
    if not raw:
        return True

    # "Lakers vs Celtics" / "Celtics @ Lakers" / "Celtics at Lakers"
    parts = [p.strip() for p in re.split(r"\s*(?:vs\.?|@|at)\s*", raw) if p.strip()]
    if len(parts) >= 2:
        a, b = parts[0], parts[1]
        return (a in home or a in away) and (b in home or b in away)

    # Single-team filter
    return raw in home or raw in away


def fetch_and_display_slate(date_str: str, matchup_filter: str = None):
    """Fetch slate and display results."""
    print(f"\n{'='*80}")
    print(f"🏀 NBA PREDICTIONS - {date_str.upper()}")
    print(f"{'='*80}\n")
    
    try:
        data = http_get_json(
            f"{API_URL}/slate/{date_str}/comprehensive",
            params={"use_splits": "true"},
            timeout=120,
        )
        analysis = data.get("analysis", [])
        
        if not analysis:
            print("📭 No games found for this date")
            return
        
        # Filter by matchup if specified
        if matchup_filter:
            analysis = [g for g in analysis if _match_game(g, matchup_filter)]
            if not analysis:
                print(f"❌ No games matching '{matchup_filter}'")
                print("\nAvailable games for this date:")
                for g in data.get("analysis", []):
                    home = g.get("home_team", "")
                    away = g.get("away_team", "")
                    time_cst = g.get("time_cst", "TBD")
                    print(f"  - {away} @ {home} ({time_cst})")
                return
        
        print(f"📊 Found {len(analysis)} game(s)\n")
        
        # Display each game
        for game in analysis:
            home = game.get("home_team", "")
            away = game.get("away_team", "")
            time_cst = game.get("time_cst", "TBD")
            odds = game.get("odds", {})
            edge_data = game.get("comprehensive_edge", {})
            
            print(f"{'─'*80}")
            print(f"🎯 {away} @ {home}")
            print(f"⏰ {time_cst}")
            print()
            
            # Full Game picks
            fg = edge_data.get("full_game", {})
            if fg:
                print("  FULL GAME:")
                
                # Spread
                spread = fg.get("spread", {})
                if spread.get("pick"):
                    pick_team = home if spread["pick"] == "HOME" else away
                    line = spread.get("market_line", 0)
                    edge = spread.get("edge", 0)
                    conf = spread.get("confidence", 0)
                    fire = calculate_fire_rating(conf, abs(edge), "pts")
                    print(f"    📌 SPREAD: {pick_team} {line:+.1f}  |  Edge: {edge:+.1f} pts  |  {fire}")
                
                # Total
                total = fg.get("total", {})
                if total.get("pick"):
                    pick_side = total["pick"]
                    line = total.get("market_line", 0)
                    edge = total.get("edge", 0)
                    conf = total.get("confidence", 0)
                    fire = calculate_fire_rating(conf, abs(edge), "pts")
                    print(f"    📌 TOTAL: {pick_side} {line:.1f}  |  Edge: {edge:+.1f} pts  |  {fire}")
                
                # Moneyline
                ml = fg.get("moneyline", {})
                if ml.get("pick"):
                    pick_team = home if ml["pick"] == "HOME" else away
                    ml_odds = odds.get("home_ml" if ml["pick"] == "HOME" else "away_ml", -110)
                    edge = ml.get("edge", 0)
                    conf = ml.get("confidence", 0)
                    fire = calculate_fire_rating(conf, abs(edge), "pct")
                    print(f"    📌 ML: {pick_team} ({format_odds(ml_odds)})  |  Edge: {edge:+.1%}  |  {fire}")
            
            # First Half picks
            fh = edge_data.get("first_half", {})
            if fh:
                has_fh_picks = any(fh.get(m, {}).get("pick") for m in ["spread", "total", "moneyline"])
                if has_fh_picks:
                    print("\n  FIRST HALF:")
                    
                    spread = fh.get("spread", {})
                    if spread.get("pick"):
                        pick_team = home if spread["pick"] == "HOME" else away
                        line = spread.get("market_line", 0)
                        edge = spread.get("edge", 0)
                        conf = spread.get("confidence", 0)
                        fire = calculate_fire_rating(conf, abs(edge), "pts")
                        print(f"    📌 1H SPREAD: {pick_team} {line:+.1f}  |  Edge: {edge:+.1f} pts  |  {fire}")
                    
                    total = fh.get("total", {})
                    if total.get("pick"):
                        pick_side = total["pick"]
                        line = total.get("market_line", 0)
                        edge = total.get("edge", 0)
                        conf = total.get("confidence", 0)
                        fire = calculate_fire_rating(conf, abs(edge), "pts")
                        print(f"    📌 1H TOTAL: {pick_side} {line:.1f}  |  Edge: {edge:+.1f} pts  |  {fire}")
            
            print()
        
        print(f"{'='*80}")
        print("✅ Analysis complete")
        print(f"   More fires = stronger pick (🔥🔥🔥🔥🔥 = best)")
        
    except TimeoutError:
        print("❌ Request timed out - API may be processing")
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="NBA Slate Runner - THE SINGLE COMMAND FOR PREDICTIONS",
        epilog="Examples:\n"
               "  python scripts/run_slate.py\n"
               "  python scripts/run_slate.py --date tomorrow\n"
               "  python scripts/run_slate.py --matchup Lakers\n",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--date", default="today", help="Date: today, tomorrow, or YYYY-MM-DD")
    parser.add_argument("--matchup", help="Filter to specific team (e.g., 'Lakers')")
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🏀 NBA PREDICTION SYSTEM v5.0")
    print("="*80)
    
    # Step 1: Check Docker
    if not check_docker_running():
        print("❌ Docker is not running. Please start Docker Desktop.")
        sys.exit(1)
    print("✅ Docker is running")
    
    # Step 2: Ensure stack is running
    if not check_stack_running():
        start_stack()
    else:
        print("✅ Stack already running")
    
    # Step 3: Wait for API
    if not wait_for_api():
        print("\n💡 Try: docker compose logs strict-api")
        sys.exit(1)
    
    # Step 4: Fetch and display
    fetch_and_display_slate(args.date, args.matchup)


if __name__ == "__main__":
    main()
