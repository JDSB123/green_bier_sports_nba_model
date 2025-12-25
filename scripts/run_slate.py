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
    5. Saves output to data/processed/slate_output_YYYYMMDD_HHMMSS.txt
"""
import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen
from zoneinfo import ZoneInfo

# Fix Windows console encoding
import io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
CST = ZoneInfo("America/Chicago")

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


def calculate_fire_rating(confidence: float, edge_pts: float) -> str:
    """Calculate fire rating (1-5 fires). All edges in pts."""
    edge_norm = min(abs(edge_pts) / 10.0, 1.0)

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


def prob_edge_to_pts(prob_edge: float) -> float:
    """Convert probability edge to equivalent point edge.

    Rule of thumb: ~3% probability edge ≈ 1 point edge
    """
    if prob_edge is None:
        return 0.0
    return prob_edge * 33.33  # 3% = 1 pt, so multiply by 33.33


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
    """Fetch slate, display results, and save to file."""
    now_cst = datetime.now(CST)
    output_lines = []

    def log(line: str = ""):
        """Print and store line for file output."""
        print(line)
        output_lines.append(line)

    log(f"\n{'='*100}")
    log(f"NBA PREDICTIONS - {date_str.upper()}")
    log(f"Generated: {now_cst.strftime('%Y-%m-%d %I:%M %p CST')}")
    log(f"{'='*100}\n")

    try:
        data = http_get_json(
            f"{API_URL}/slate/{date_str}/comprehensive",
            params={"use_splits": "true"},
            timeout=120,
        )
        analysis = data.get("analysis", [])

        if not analysis:
            log("No games found for this date")
            return

        # Filter by matchup if specified
        if matchup_filter:
            analysis = [g for g in analysis if _match_game(g, matchup_filter)]
            if not analysis:
                log(f"No games matching '{matchup_filter}'")
                log("\nAvailable games for this date:")
                for g in data.get("analysis", []):
                    home = g.get("home_team", "")
                    away = g.get("away_team", "")
                    time_cst = g.get("time_cst", "TBD")
                    log(f"  - {away} @ {home} ({time_cst})")
                return

        log(f"Found {len(analysis)} game(s)\n")

        # --- BLUF TABLE ---
        log("=" * 140)
        log(f"{'RECOMMENDED PICKS':^140}")
        log("=" * 140)

        # Header
        header = f"{'Time (CST)':<12} | {'Matchup':<42} | {'Pick':<22} | {'Odds':<7} | {'Model':<10} | {'Market':<10} | {'Edge':<8} | {'Fire'}"
        log(header)
        log("-" * 140)

        for game in analysis:
            home = game.get("home_team", "")
            away = game.get("away_team", "")
            time_cst = game.get("time_cst", "TBD")
            odds = game.get("odds", {})
            edge_data = game.get("comprehensive_edge", {})
            features = game.get("features", {})

            # Get team records from features
            home_wins = features.get("home_wins", 0)
            home_losses = features.get("home_losses", 0)
            away_wins = features.get("away_wins", 0)
            away_losses = features.get("away_losses", 0)

            home_record = f"({home_wins}-{home_losses})" if home_wins or home_losses else ""
            away_record = f"({away_wins}-{away_losses})" if away_wins or away_losses else ""

            matchup_str = f"{away} {away_record} @ {home} {home_record}"

            # Collect all picks for this game
            picks = []

            # Helper to add pick - ALL edges in pts
            def add_pick(market_name, p_data, p_type="spread"):
                if not p_data or not p_data.get("pick"):
                    return

                pick_team = p_data.get("pick")
                market_line = p_data.get("market_line", 0)
                # Use pick_line (the spread for the picked team) if available
                pick_line = p_data.get("pick_line") if p_data.get("pick_line") is not None else market_line
                market_odds_val = p_data.get("market_odds")
                conf = p_data.get("confidence", 0)

                # Get edge - convert ML probability edge to pts
                if p_type == "ml":
                    is_home_pick = (pick_team == home)
                    prob_edge = p_data.get("edge_home" if is_home_pick else "edge_away", 0) or 0
                    edge_pts = prob_edge_to_pts(prob_edge)
                    # For ML, get the actual odds for the picked team
                    market_odds_val = odds.get("home_ml" if is_home_pick else "away_ml")
                else:
                    edge_pts = p_data.get("edge", 0) or 0

                # Fire rating - all in pts now
                fire = calculate_fire_rating(conf, abs(edge_pts))

                # Pick display
                if p_type == "ml":
                    pick_str = f"{pick_team} ML"
                    model_val = f"{conf*100:.0f}% WIN"
                    market_val = format_odds(market_odds_val)
                elif p_type == "total":
                    pick_str = f"{pick_team} {market_line}"
                    model_val = f"{p_data.get('model_total', 0):.1f}"
                    market_val = f"{market_line:.1f}"
                else:  # spread
                    pick_str = f"{pick_team} {pick_line:+.1f}"
                    model_margin = p_data.get("model_margin", 0)
                    proj = model_margin if pick_team == home else -model_margin
                    model_val = f"{proj:+.1f}"
                    market_val = f"{pick_line:+.1f}"

                picks.append({
                    "market": market_name,
                    "pick": pick_str,
                    "odds": format_odds(market_odds_val),
                    "model": model_val,
                    "market_line": market_val,
                    "edge": f"{edge_pts:+.1f} pts",
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
                    log(f"{t_str:<12} | {m_str:<42} | {p['pick']:<22} | {p['odds']:<7} | {p['model']:<10} | {p['market_line']:<10} | {p['edge']:<8} | {p['fire']}")
                    first = False
                log("-" * 140)
            else:
                # No picks for this game
                log(f"{time_cst:<12} | {matchup_str:<42} | {'No Action':<22} | {'-':<7} | {'-':<10} | {'-':<10} | {'-':<8} | -")
                log("-" * 140)

        log("\n" + "=" * 100)
        log("DETAILED RATIONALE")
        log("=" * 100 + "\n")

        # Display each game (Detailed)
        for game in analysis:
            home = game.get("home_team", "")
            away = game.get("away_team", "")
            time_cst = game.get("time_cst", "TBD")
            odds = game.get("odds", {})
            edge_data = game.get("comprehensive_edge", {})
            features = game.get("features", {})

            # Get team records
            home_wins = features.get("home_wins", 0)
            home_losses = features.get("home_losses", 0)
            away_wins = features.get("away_wins", 0)
            away_losses = features.get("away_losses", 0)

            home_record = f"({home_wins}-{home_losses})" if home_wins or home_losses else ""
            away_record = f"({away_wins}-{away_losses})" if away_wins or away_losses else ""

            log("-" * 100)
            log(f"GAME: {away} {away_record} @ {home} {home_record}")
            log(f"TIME: {time_cst}")
            log()

            # Full Game picks
            fg = edge_data.get("full_game", {})
            if fg:
                log("  FULL GAME:")

                # Spread
                spread = fg.get("spread", {})
                if spread.get("pick"):
                    pick_team = spread["pick"]
                    market_line = spread.get("market_line", 0)
                    # Use pick_line for display (correct sign for picked team)
                    pick_line = spread.get("pick_line") if spread.get("pick_line") is not None else market_line
                    market_odds_val = spread.get("market_odds")
                    edge = spread.get("edge", 0) or 0
                    conf = spread.get("confidence", 0)
                    model_margin = spread.get("model_margin", 0)
                    fire = calculate_fire_rating(conf, abs(edge))

                    proj_margin = model_margin if pick_team == home else -model_margin

                    log(f"    SPREAD: {pick_team} {pick_line:+.1f} ({format_odds(market_odds_val)})")
                    log(f"       Model: {pick_team} {proj_margin:+.1f}")
                    log(f"       Market: {pick_line:+.1f} ({format_odds(market_odds_val)})")
                    log(f"       Edge: {edge:+.1f} pts  |  {fire}")

                # Total
                total = fg.get("total", {})
                if total.get("pick"):
                    pick_side = total["pick"]
                    line = total.get("market_line", 0)
                    market_odds_val = total.get("market_odds")
                    edge = total.get("edge", 0) or 0
                    conf = total.get("confidence", 0)
                    model_total = total.get("model_total", 0)
                    fire = calculate_fire_rating(conf, abs(edge))

                    log(f"    TOTAL: {pick_side} {line:.1f} ({format_odds(market_odds_val)})")
                    log(f"       Model: {model_total:.1f}")
                    log(f"       Market: {line:.1f} ({format_odds(market_odds_val)})")
                    log(f"       Edge: {edge:+.1f} pts  |  {fire}")

                # Moneyline
                ml = fg.get("moneyline", {})
                if ml.get("pick"):
                    pick_team = ml["pick"]
                    is_home = (pick_team == home)
                    ml_odds = odds.get("home_ml" if is_home else "away_ml")

                    prob_edge = ml.get("edge_home" if is_home else "edge_away", 0) or 0
                    edge_pts = prob_edge_to_pts(prob_edge)

                    conf = ml.get("confidence", 0)
                    fire = calculate_fire_rating(conf, abs(edge_pts))
                    rationale = ml.get("rationale", "")

                    log(f"    ML: {pick_team} ({format_odds(ml_odds)})")
                    log(f"       Model: {rationale}")
                    log(f"       Market: {format_odds(ml_odds)}")
                    log(f"       Edge: {edge_pts:+.1f} pts  |  {fire}")

            # First Half picks
            fh = edge_data.get("first_half", {})
            if fh:
                has_fh_picks = any(fh.get(m, {}).get("pick") for m in ["spread", "total", "moneyline"])
                if has_fh_picks:
                    log("\n  FIRST HALF:")

                    spread = fh.get("spread", {})
                    if spread.get("pick"):
                        pick_team = spread["pick"]
                        market_line = spread.get("market_line", 0)
                        # Use pick_line for display (correct sign for picked team)
                        pick_line = spread.get("pick_line") if spread.get("pick_line") is not None else market_line
                        market_odds_val = spread.get("market_odds")
                        edge = spread.get("edge", 0) or 0
                        conf = spread.get("confidence", 0)
                        model_margin = spread.get("model_margin", 0)
                        fire = calculate_fire_rating(conf, abs(edge))

                        proj_margin = model_margin if pick_team == home else -model_margin

                        log(f"    1H SPREAD: {pick_team} {pick_line:+.1f} ({format_odds(market_odds_val)})")
                        log(f"       Model: {pick_team} {proj_margin:+.1f}")
                        log(f"       Market: {pick_line:+.1f} ({format_odds(market_odds_val)})")
                        log(f"       Edge: {edge:+.1f} pts  |  {fire}")

                    total = fh.get("total", {})
                    if total.get("pick"):
                        pick_side = total["pick"]
                        line = total.get("market_line", 0)
                        market_odds_val = total.get("market_odds")
                        edge = total.get("edge", 0) or 0
                        conf = total.get("confidence", 0)
                        model_total = total.get("model_total", 0)
                        fire = calculate_fire_rating(conf, abs(edge))

                        log(f"    1H TOTAL: {pick_side} {line:.1f} ({format_odds(market_odds_val)})")
                        log(f"       Model: {model_total:.1f}")
                        log(f"       Market: {line:.1f} ({format_odds(market_odds_val)})")
                        log(f"       Edge: {edge:+.1f} pts  |  {fire}")

            log()

        log("=" * 100)
        log("Analysis complete")
        log("   More fires = stronger pick (5 fires = best)")
        log("=" * 100)

        # Save output to file
        timestamp = now_cst.strftime("%Y%m%d_%H%M%S")
        output_file = OUTPUT_DIR / f"slate_output_{timestamp}.txt"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines))
        print(f"\n[SAVED] Output saved to: {output_file}")

    except TimeoutError:
        print("Request timed out - API may be processing")
    except Exception as e:
        print(f"Error: {e}")


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
    print("🏀 NBA PREDICTION SYSTEM v33.0.2.0")
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
