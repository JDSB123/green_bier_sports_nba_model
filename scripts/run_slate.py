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

import os

PROJECT_ROOT = Path(__file__).parent.parent

# API URL from environment - no hardcoded ports
API_PORT = os.getenv("NBA_API_PORT", "8090")
API_URL = os.getenv("NBA_API_URL", f"http://localhost:{API_PORT}")


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
            ["docker", "ps", "--filter", "name=nba-v60", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=10
        )
        # Container is named 'nba-v60' per compose; service name is 'nba-v60-api'
        names = result.stdout.splitlines()
        return any(n in ("nba-v60", "nba-v60-api") for n in names)
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


def format_odds(odds: int | None) -> str:
    """Format American odds."""
    if odds is None:
        return "N/A"
    try:
        odds = int(odds)
        return f"+{odds}" if odds > 0 else str(odds)
    except (ValueError, TypeError):
        return str(odds)


def _match_game(game: dict, matchup_filter: str) -> bool:
    """Match on one or more matchup filters.

    Supports:
    - Single team: "Lakers"
    - Specific matchup: "Lakers vs Celtics" / "Celtics @ Lakers"
    - Multiple filters: "Lakers, Celtics @ Knicks, Heat"
    """
    home = (game.get("home_team") or "").lower()
    away = (game.get("away_team") or "").lower()
    raw = (matchup_filter or "").strip()
    if not raw:
        return True

    # Multiple filters separated by commas: match ANY
    filters = [f.strip().lower() for f in raw.split(",") if f.strip()]
    if not filters:
        return True
    for f in filters:
        if _match_single_filter(home=home, away=away, raw=f):
            return True
    return False


def _match_single_filter(*, home: str, away: str, raw: str) -> bool:
    """Match one filter against one game (home/away already lowercased)."""
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

        # --- BLUF TABLE ---
        print(f"{'='*120}")
        print(f"{'BOTTOM LINE UP FRONT (BLUF)':^120}")
        print(f"{'='*120}")
        
        # Header
        # Date/Time | Matchup (Records) | Pick | Odds | Model | Market | Edge | Fire
        header = f"{'Time (CST)':<12} | {'Matchup':<35} | {'Pick':<20} | {'Odds':<6} | {'Model':<8} | {'Market':<8} | {'Edge':<10} | {'Fire'}"
        print(header)
        print("-" * 120)

        for game in analysis:
            home = game.get("home_team", "")
            away = game.get("away_team", "")
            time_cst = game.get("time_cst", "TBD")
            odds = game.get("odds", {})
            edge_data = game.get("comprehensive_edge", {})
            
            # Try to get records from features if available
            features = game.get("features", {})
            # Assuming features might have raw stats, but if not, we skip records for now
            # Or we can try to parse them if they exist in a known format
            # For now, just Matchup
            matchup_str = f"{away} @ {home}"

            # Collect all picks for this game
            picks = []
            
            # Helper to add pick
            def add_pick(market_name, p_data, p_type="spread"):
                if not p_data or not p_data.get("pick"):
                    return
                
                pick_team = p_data.get("pick")
                market_line = p_data.get("market_line", 0)
                market_odds = p_data.get("market_odds")
                edge = p_data.get("edge", 0)
                conf = p_data.get("confidence", 0)
                
                # Fire rating
                fire = calculate_fire_rating(conf, abs(edge), "pct" if p_type == "ml" else "pts")
                
                # Pick display
                if p_type == "ml":
                    pick_str = f"{pick_team}"
                    model_val = "WIN" # Simplified for table
                elif p_type == "total":
                    pick_str = f"{pick_team} {market_line}"
                    model_val = f"{p_data.get('model_total', 0):.1f}"
                else: # spread
                    pick_str = f"{pick_team} {market_line:+.1f}"
                    # Model projection
                    model_margin = p_data.get("model_margin", 0)
                    proj = model_margin if pick_team == home else -model_margin
                    model_val = f"{proj:+.1f}"

                picks.append({
                    "market": market_name,
                    "pick": pick_str,
                    "odds": format_odds(market_odds),
                    "model": model_val,
                    "market_line": f"{market_line:+.1f}" if p_type == "spread" else f"{market_line}",
                    "edge": f"{edge:+.1f}" if p_type != "ml" else f"{edge:+.1%}",
                    "fire": fire
                })

            # Full Game
            fg = edge_data.get("full_game", {})
            add_pick("FG Spread", fg.get("spread"), "spread")
            add_pick("FG Total", fg.get("total"), "total")
            add_pick("FG ML", fg.get("moneyline"), "ml")
            
            # 1H
            fh = edge_data.get("first_half", {})
            add_pick("1H Spread", fh.get("spread"), "spread")
            add_pick("1H Total", fh.get("total"), "total")
            add_pick("1H ML", fh.get("moneyline"), "ml")

            # Print rows
            if picks:
                first = True
                for p in picks:
                    t_str = time_cst if first else ""
                    m_str = matchup_str if first else ""
                    print(f"{t_str:<12} | {m_str:<35} | {p['pick']:<20} | {p['odds']:<6} | {p['model']:<8} | {p['market_line']:<8} | {p['edge']:<10} | {p['fire']}")
                    first = False
                print("-" * 120)
            else:
                # No picks for this game
                print(f"{time_cst:<12} | {matchup_str:<35} | {'No Action':<20} | {'-':<6} | {'-':<8} | {'-':<8} | {'-':<10} | {'-'}")
                print("-" * 120)

        print("\n" + "="*80)
        print("DETAILED RATIONALE")
        print("="*80 + "\n")
        
        # Display each game (Detailed)
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
                    pick_team = spread["pick"]
                    line = spread.get("market_line", 0)
                    market_odds = spread.get("market_odds")
                    edge = spread.get("edge", 0)
                    conf = spread.get("confidence", 0)
                    model_margin = spread.get("model_margin", 0)
                    fire = calculate_fire_rating(conf, abs(edge), "pts")
                    
                    # Calculate model projection for the picked team
                    # model_margin is usually Home Margin
                    proj_margin = model_margin if pick_team == home else -model_margin
                    
                    print(f"    📌 SPREAD: {pick_team} {line:+.1f} ({format_odds(market_odds)})")
                    print(f"       Model: {pick_team} {proj_margin:+.1f}")
                    print(f"       Market: {line:+.1f} ({format_odds(market_odds)})")
                    print(f"       Edge: {edge:+.1f} pts  |  {fire}")
                
                # Total
                total = fg.get("total", {})
                if total.get("pick"):
                    pick_side = total["pick"]
                    line = total.get("market_line", 0)
                    market_odds = total.get("market_odds")
                    edge = total.get("edge", 0)
                    conf = total.get("confidence", 0)
                    model_total = total.get("model_total", 0)
                    fire = calculate_fire_rating(conf, abs(edge), "pts")
                    
                    print(f"    📌 TOTAL: {pick_side} {line:.1f} ({format_odds(market_odds)})")
                    print(f"       Model: {model_total:.1f}")
                    print(f"       Market: {line:.1f} ({format_odds(market_odds)})")
                    print(f"       Edge: {edge:+.1f} pts  |  {fire}")
                
                # Moneyline
                ml = fg.get("moneyline", {})
                if ml.get("pick"):
                    pick_team = ml["pick"]
                    is_home = (pick_team == home)
                    ml_odds = odds.get("home_ml" if is_home else "away_ml")
                    
                    # Edge is split in result
                    edge = ml.get("edge_home" if is_home else "edge_away", 0)
                    if edge is None: edge = 0
                    
                    conf = ml.get("confidence", 0)
                    fire = calculate_fire_rating(conf, abs(edge), "pct")
                    rationale = ml.get("rationale", "")
                    
                    print(f"    📌 ML: {pick_team} ({format_odds(ml_odds)})")
                    print(f"       Model: {rationale}")
                    print(f"       Market: {format_odds(ml_odds)}")
                    print(f"       Edge: {edge:+.1%}  |  {fire}")
            
            # First Half picks
            fh = edge_data.get("first_half", {})
            if fh:
                has_fh_picks = any(fh.get(m, {}).get("pick") for m in ["spread", "total", "moneyline"])
                if has_fh_picks:
                    print("\n  FIRST HALF:")
                    
                    spread = fh.get("spread", {})
                    if spread.get("pick"):
                        pick_team = spread["pick"]
                        line = spread.get("market_line", 0)
                        market_odds = spread.get("market_odds")
                        edge = spread.get("edge", 0)
                        conf = spread.get("confidence", 0)
                        model_margin = spread.get("model_margin", 0)
                        fire = calculate_fire_rating(conf, abs(edge), "pts")
                        
                        proj_margin = model_margin if pick_team == home else -model_margin
                        
                        print(f"    📌 1H SPREAD: {pick_team} {line:+.1f} ({format_odds(market_odds)})")
                        print(f"       Model: {pick_team} {proj_margin:+.1f}")
                        print(f"       Market: {line:+.1f} ({format_odds(market_odds)})")
                        print(f"       Edge: {edge:+.1f} pts  |  {fire}")
                    
                    total = fh.get("total", {})
                    if total.get("pick"):
                        pick_side = total["pick"]
                        line = total.get("market_line", 0)
                        market_odds = total.get("market_odds")
                        edge = total.get("edge", 0)
                        conf = total.get("confidence", 0)
                        model_total = total.get("model_total", 0)
                        fire = calculate_fire_rating(conf, abs(edge), "pts")
                        
                        print(f"    📌 1H TOTAL: {pick_side} {line:.1f} ({format_odds(market_odds)})")
                        print(f"       Model: {model_total:.1f}")
                        print(f"       Market: {line:.1f} ({format_odds(market_odds)})")
                        print(f"       Edge: {edge:+.1f} pts  |  {fire}")
            
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
    print("🏀 NBA PREDICTION SYSTEM v6.0")
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
        print("\n💡 Try: docker compose logs nba-v60")
        sys.exit(1)
    
    # Step 4: Fetch and display
    fetch_and_display_slate(args.date, args.matchup)


if __name__ == "__main__":
    main()
