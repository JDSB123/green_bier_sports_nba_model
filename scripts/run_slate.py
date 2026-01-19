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
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen
from zoneinfo import ZoneInfo
from src.utils.version import resolve_version

# Fix Windows console encoding
import io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
ARCHIVE_DIR = PROJECT_ROOT / "archive" / "slate_outputs"
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
            ["docker", "ps", "--filter", "name=nba-v33", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=10
        )
        # Container is named 'nba-v33' per compose; service name is 'nba-v33-api'
        names = result.stdout.splitlines()
        return any(n in ("nba-v33", "nba-v33-api") for n in names)
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


def generate_html_output(analysis: list, date_str: str, now_cst: datetime, odds_data: dict = None) -> str:
    """Generate HTML output for the slate analysis."""

    def fire_to_html(fire_str: str) -> str:
        """Convert fire emojis to HTML spans for styling."""
        count = fire_str.count("🔥")
        return f'<span class="fire fire-{count}">{"🔥" * count}</span>'

    def format_odds_html(odds_val) -> str:
        if odds_val is None:
            return "N/A"
        try:
            odds_val = int(odds_val)
            return f"+{odds_val}" if odds_val > 0 else str(odds_val)
        except (ValueError, TypeError):
            return str(odds_val)

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NBA Picks - {date_str}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        header {{
            text-align: center;
            padding: 30px 0;
            border-bottom: 2px solid #e94560;
            margin-bottom: 30px;
        }}
        h1 {{ font-size: 2.5em; color: #e94560; margin-bottom: 10px; }}
        .subtitle {{ color: #888; font-size: 1.1em; }}
        .summary-box {{
            background: rgba(233, 69, 96, 0.1);
            border: 1px solid #e94560;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 30px;
        }}
        .summary-box h2 {{ color: #e94560; margin-bottom: 15px; }}
        .line-note {{
            color: #f1f5f9;
            font-size: 0.95em;
            margin-bottom: 20px;
            text-align: center;
            letter-spacing: 0.5px;
        }}
        .picks-table {{
            width: 100%;
            border-collapse: collapse;
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
            overflow: hidden;
            margin-bottom: 30px;
        }}
        .picks-table th {{
            background: #e94560;
            color: white;
            padding: 15px 10px;
            text-align: left;
            font-weight: 600;
        }}
        .picks-table td {{
            padding: 12px 10px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        .picks-table tr:hover {{ background: rgba(233, 69, 96, 0.1); }}
        .game-header {{
            background: rgba(233, 69, 96, 0.2) !important;
            font-weight: bold;
        }}
        .pick-cell {{ font-weight: 600; color: #4ade80; }}
        .edge-positive {{ color: #4ade80; }}
        .edge-negative {{ color: #f87171; }}
        .fire {{ font-size: 1.1em; }}
        .fire-5 {{ text-shadow: 0 0 10px #ff6b35; }}
        .fire-4 {{ text-shadow: 0 0 8px #ff8c42; }}
        .tier-elite {{ background: linear-gradient(90deg, rgba(255,215,0,0.2), transparent) !important; }}
        .tier-strong {{ background: linear-gradient(90deg, rgba(192,192,192,0.2), transparent) !important; }}
        .model-val {{ color: #60a5fa; }}
        .market-val {{ color: #a78bfa; }}
        footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            border-top: 1px solid #333;
            margin-top: 30px;
        }}
        @media (max-width: 768px) {{
            .picks-table {{ font-size: 0.85em; }}
            .picks-table th, .picks-table td {{ padding: 8px 5px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🏀 NBA PREDICTIONS</h1>
            <p class="subtitle">{date_str.upper()} | Generated: {now_cst.strftime('%Y-%m-%d %I:%M %p CST')} | {resolve_version().replace("NBA_v", "v")}</p>
        </header>

        <div class="summary-box">
            <h2>📊 {len(analysis)} Games Analyzed</h2>
        </div>
        <p class="line-note">Spreads and lines always reflect the home team’s perspective (positive = home underdog).</p>

        <table class="picks-table">
            <thead>
                <tr>
                    <th>Time</th>
                    <th>Matchup</th>
                    <th>Pick</th>
                    <th>Odds</th>
                    <th>Prediction</th>
                    <th>Line</th>
                    <th>Edge</th>
                    <th>Rating</th>
                </tr>
            </thead>
            <tbody>
'''

    for game in analysis:
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        time_cst = game.get("time_cst", "TBD")
        odds = game.get("odds", {})
        edge_data = game.get("comprehensive_edge", {})
        features = game.get("features", {})

        home_wins = features.get("home_wins", 0)
        home_losses = features.get("home_losses", 0)
        away_wins = features.get("away_wins", 0)
        away_losses = features.get("away_losses", 0)

        home_record = f"({home_wins}-{home_losses})" if home_wins or home_losses else ""
        away_record = f"({away_wins}-{away_losses})" if away_wins or away_losses else ""

        matchup_str = f"{away} {away_record} @ {home} {home_record}"

        # Game header row
        html += f'''            <tr class="game-header">
                <td>{time_cst}</td>
                <td colspan="7">{matchup_str}</td>
            </tr>
'''

        # Collect picks for this game
        picks = []

        def add_pick_html(market_name, p_data, p_type="spread"):
            if not p_data or not p_data.get("pick"):
                return

            pick_team = p_data.get("pick")
            market_line = p_data.get("market_line")
            pick_line = p_data.get("pick_line") if p_data.get("pick_line") is not None else market_line
            market_odds_val = p_data.get("market_odds")
            conf = p_data.get("confidence", 0)

            edge_pts = p_data.get("edge", 0) or 0

            # Fire rating
            edge_norm = min(abs(edge_pts) / 10.0, 1.0)
            combined = (conf * 0.6) + (edge_norm * 0.4)
            if combined >= 0.85:
                fire_count = 5
            elif combined >= 0.70:
                fire_count = 4
            elif combined >= 0.60:
                fire_count = 3
            elif combined >= 0.52:
                fire_count = 2
            else:
                fire_count = 1

            # Pick display - make model prediction explicit
            if p_type == "total":
                pick_str = f"{pick_team} {market_line if market_line is not None else 0:.1f}"
                model_total = p_data.get('model_total', 0)
                model_val = f"Total: {model_total:.1f}"
                market_val = f"{market_line if market_line is not None else 0:.1f}"
                line_display = f"{market_line:.1f}" if market_line is not None else "N/A"
            else:
                pick_str = f"{pick_team} {pick_line:+.1f}"
                model_margin = p_data.get("model_margin", 0)
                # Model margin is home perspective, convert to predicted winner
                if model_margin > 0:
                    model_val = f"{home} by {abs(model_margin):.1f}"
                elif model_margin < 0:
                    model_val = f"{away} by {abs(model_margin):.1f}"
                else:
                    model_val = "Pick'em"
                market_val = f"{pick_line:+.1f}"
                if market_line is not None:
                    home_line = market_line
                    away_line = -market_line
                    line_display = f"{home_line:+.1f}/{away_line:+.1f}"
                else:
                    line_display = "N/A"

            tier_class = "tier-elite" if fire_count >= 5 else ("tier-strong" if fire_count >= 4 else "")
            edge_class = "edge-positive" if edge_pts > 0 else "edge-negative"

            picks.append({
                "market": market_name,
                "pick": pick_str,
                "odds": format_odds_html(market_odds_val),
                "model": model_val,
                "market_line": market_val,
                "line_display": line_display,
                "edge": f"{edge_pts:+.1f}",
                "fire_count": fire_count,
                "tier_class": tier_class,
                "edge_class": edge_class
            })

        # Full Game
        fg = edge_data.get("full_game", {})
        add_pick_html("FG Spread", fg.get("spread"), "spread")
        add_pick_html("FG Total", fg.get("total"), "total")

        # 1H
        fh = edge_data.get("first_half", {})
        add_pick_html("1H Spread", fh.get("spread"), "spread")
        add_pick_html("1H Total", fh.get("total"), "total")

        for p in picks:
            html += f'''            <tr class="{p['tier_class']}">
                <td></td>
                <td>{p['market']}</td>
                <td class="pick-cell">{p['pick']}</td>
                <td>{p['odds']}</td>
                <td class="model-val">{p['model']}</td>
                <td class="market-val">{p['line_display']}</td>
                <td class="{p['edge_class']}">{p['edge']} pts</td>
                <td class="fire fire-{p['fire_count']}">{"🔥" * p['fire_count']}</td>
            </tr>
'''

        if not picks:
            html += f'''            <tr>
                <td></td>
                <td colspan="7" style="color: #666;">No qualifying picks</td>
            </tr>
'''

    html += '''            </tbody>
        </table>

        <footer>
            <p>🔥 = Pick strength (5 fires = strongest) | Model {resolve_version().replace("NBA_v", "v")}</p>
        </footer>
    </div>
</body>
</html>
'''
    return html


def fetch_and_display_slate(date_str: str, matchup_filter: str = None):
    """Fetch slate, display results, and save to file."""
    now_cst = datetime.now(CST)
    output_lines = []
    analysis_data = []  # Store for HTML generation

    def log(line: str = ""):
        """Print and store line for file output."""
        print(line)
        output_lines.append(line)

    def format_ev_line(p_data: dict) -> str | None:
        """Format EV/Kelly line if available."""
        ev_pct = p_data.get("ev_pct")
        kelly = p_data.get("kelly_fraction")
        if ev_pct is None and kelly is None:
            return None
        ev_str = f"{ev_pct:+.1f}%" if isinstance(ev_pct, (int, float)) else "N/A"
        kelly_str = f"{kelly:.2f}" if isinstance(kelly, (int, float)) else "N/A"
        return f"       EV: {ev_str}  |  Kelly: {kelly_str}"

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
        log("=" * 155)
        log(f"{'RECOMMENDED PICKS':^155}")
        log("=" * 155)

        # Header
        header = f"{'Time (CST)':<12} | {'Matchup':<42} | {'Pick':<22} | {'Odds':<7} | {'Prediction':<12} | {'Line':<12} | {'Edge':<8} | {'EV%':<7} | {'Fire'}"
        log(header)
        log("-" * 155)
        log("NOTE: 'Line' column shows home/away view (home_line/away_line; home positive = home underdog); 'Pick' still shows the team we\'re backing.")
        log("-" * 155)

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
                market_line = p_data.get("market_line")
                # Use pick_line (the spread for the picked team) if available
                pick_line = p_data.get("pick_line")
                if pick_line is None:
                    pick_line = market_line
                market_odds_val = p_data.get("market_odds")
                conf = p_data.get("confidence", 0)

                edge_pts = p_data.get("edge", 0) or 0

                # Fire rating - all in pts now
                fire = calculate_fire_rating(conf, abs(edge_pts))

                # Pick display - make model prediction explicit
                if p_type == "total":
                    pick_str = f"{pick_team} {market_line if market_line is not None else 0:.1f}"
                    model_total = p_data.get('model_total', 0)
                    model_val = f"{model_total:.1f}"
                    market_val = f"{market_line if market_line is not None else 0:.1f}"
                    line_display = f"{market_line:.1f}" if market_line is not None else "N/A"
                else:  # spread
                    pick_str = f"{pick_team} {pick_line:+.1f}"
                    model_margin = p_data.get("model_margin", 0)
                    # Show predicted winner with margin
                    if model_margin > 0:
                        model_val = f"{home[:3].upper()} by {abs(model_margin):.1f}"
                    elif model_margin < 0:
                        model_val = f"{away[:3].upper()} by {abs(model_margin):.1f}"
                    else:
                        model_val = "Pick'em"
                    market_val = f"{pick_line:+.1f}"
                    if market_line is not None:
                        home_line = market_line
                        away_line = -market_line
                        line_display = f"{home_line:+.1f}/{away_line:+.1f}"
                    else:
                        line_display = "N/A"

                ev_pct = p_data.get("ev_pct")
                ev_str = f"{ev_pct:+.1f}%" if isinstance(ev_pct, (int, float)) else "N/A"

                picks.append({
                    "market": market_name,
                    "pick": pick_str,
                    "odds": format_odds(market_odds_val),
                    "model": model_val,
                    "market_line": market_val,
                    "line_display": line_display,
                    "edge": f"{edge_pts:+.1f} pts",
                    "ev": ev_str,
                    "fire": fire
                })

            # Full Game
            fg = edge_data.get("full_game", {})
            add_pick("FG Spread", fg.get("spread"), "spread")
            add_pick("FG Total", fg.get("total"), "total")

            # 1H
            fh = edge_data.get("first_half", {})
            add_pick("1H Spread", fh.get("spread"), "spread")
            add_pick("1H Total", fh.get("total"), "total")

            # Print rows
            if picks:
                first = True
                for p in picks:
                    t_str = time_cst if first else ""
                    m_str = matchup_str if first else ""
                    log(f"{t_str:<12} | {m_str:<42} | {p['pick']:<22} | {p['odds']:<7} | {p['model']:<12} | {p['line_display']:<12} | {p['edge']:<8} | {p['ev']:<7} | {p['fire']}")
                    first = False
                log("-" * 155)
            else:
                # No picks for this game
                log(f"{time_cst:<12} | {matchup_str:<42} | {'No Action':<22} | {'-':<7} | {'-':<12} | {'-':<8} | {'-':<8} | {'-':<7} | -")
                log("-" * 155)

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
            home_line = odds.get("home_spread")
            if home_line is not None:
                away_line = -home_line
                log(f"  Market line (home view): {home} {home_line:+.1f} / {away} {away_line:+.1f}")
            fh_home_line = odds.get("fh_home_spread")
            if fh_home_line is not None:
                fh_away_line = -fh_home_line
                log(f"  1H market line (home view): {home} {fh_home_line:+.1f} / {away} {fh_away_line:+.1f}")
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

                    # Determine predicted winner and margin for rationale
                    if model_margin > 0:
                        pred_winner, pred_margin = home, abs(model_margin)
                    elif model_margin < 0:
                        pred_winner, pred_margin = away, abs(model_margin)
                    else:
                        pred_winner, pred_margin = "Toss-up", 0

                    log(f"    SPREAD: {pick_team} {pick_line:+.1f} ({format_odds(market_odds_val)})")
                    log(f"       Model predicts: {pred_winner} wins by {pred_margin:.1f} pts")
                    log(f"       Market line: {pick_team} {pick_line:+.1f}")
                    log(f"       Edge: {edge:+.1f} pts of value")
                    # Explain the edge calculation
                    if pick_team == home:
                        log(f"       Rationale: Model says {home} by {abs(model_margin):.1f}, getting {pick_line:+.1f} → {edge:.1f} pts edge")
                    else:
                        log(f"       Rationale: Model says {pred_winner} by {pred_margin:.1f}, but getting {away} {pick_line:+.1f} → {edge:.1f} pts edge")
                    log(f"       Confidence: {conf:.0%}  |  {fire}")
                    ev_line = format_ev_line(spread)
                    if ev_line:
                        log(ev_line)

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
                    diff = model_total - line

                    log(f"    TOTAL: {pick_side} {line:.1f} ({format_odds(market_odds_val)})")
                    log(f"       Model predicts: {model_total:.1f} total points")
                    log(f"       Market line: {line:.1f}")
                    log(f"       Edge: {abs(edge):.1f} pts of value")
                    if pick_side == "OVER":
                        log(f"       Rationale: Model ({model_total:.1f}) > Line ({line:.1f}) by {abs(diff):.1f} pts → OVER")
                    else:
                        log(f"       Rationale: Model ({model_total:.1f}) < Line ({line:.1f}) by {abs(diff):.1f} pts → UNDER")
                    log(f"       Confidence: {conf:.0%}  |  {fire}")
                    ev_line = format_ev_line(total)
                    if ev_line:
                        log(ev_line)

            # First Half picks
            fh = edge_data.get("first_half", {})
            if fh:
                has_fh_picks = any(fh.get(m, {}).get("pick") for m in ["spread", "total"])
                if has_fh_picks:
                    log("\n  FIRST HALF:")

                    spread = fh.get("spread", {})
                    if spread.get("pick"):
                        pick_team = spread["pick"]
                        market_line = spread.get("market_line", 0)
                        pick_line = spread.get("pick_line") if spread.get("pick_line") is not None else market_line
                        market_odds_val = spread.get("market_odds")
                        edge = spread.get("edge", 0) or 0
                        conf = spread.get("confidence", 0)
                        model_margin = spread.get("model_margin", 0)
                        fire = calculate_fire_rating(conf, abs(edge))

                        if model_margin > 0:
                            pred_winner, pred_margin = home, abs(model_margin)
                        elif model_margin < 0:
                            pred_winner, pred_margin = away, abs(model_margin)
                        else:
                            pred_winner, pred_margin = "Toss-up", 0

                        log(f"    1H SPREAD: {pick_team} {pick_line:+.1f} ({format_odds(market_odds_val)})")
                        log(f"       Model predicts: {pred_winner} leads by {pred_margin:.1f} at half")
                        log(f"       Market line: {pick_team} {pick_line:+.1f}")
                        log(f"       Edge: {edge:+.1f} pts of value")
                        log(f"       Confidence: {conf:.0%}  |  {fire}")
                        ev_line = format_ev_line(spread)
                        if ev_line:
                            log(ev_line)

                    total = fh.get("total", {})
                    if total.get("pick"):
                        pick_side = total["pick"]
                        line = total.get("market_line", 0)
                        market_odds_val = total.get("market_odds")
                        edge = total.get("edge", 0) or 0
                        conf = total.get("confidence", 0)
                        model_total = total.get("model_total", 0)
                        fire = calculate_fire_rating(conf, abs(edge))
                        diff = model_total - line

                        log(f"    1H TOTAL: {pick_side} {line:.1f} ({format_odds(market_odds_val)})")
                        log(f"       Model predicts: {model_total:.1f} 1H points")
                        log(f"       Market line: {line:.1f}")
                        log(f"       Edge: {abs(edge):.1f} pts of value")
                        if pick_side == "OVER":
                            log(f"       Rationale: Model ({model_total:.1f}) > Line ({line:.1f}) → OVER")
                        else:
                            log(f"       Rationale: Model ({model_total:.1f}) < Line ({line:.1f}) → UNDER")
                        log(f"       Confidence: {conf:.0%}  |  {fire}")
                        ev_line = format_ev_line(total)
                        if ev_line:
                            log(ev_line)

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

        # Generate and save HTML output
        html_file = OUTPUT_DIR / f"slate_output_{timestamp}.html"
        html_content = generate_html_output(analysis, date_str, now_cst)
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"[SAVED] HTML output saved to: {html_file}")

        try:
            ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
            archive_file = ARCHIVE_DIR / output_file.name
            shutil.copy2(output_file, archive_file)
            # Also archive HTML
            archive_html = ARCHIVE_DIR / html_file.name
            shutil.copy2(html_file, archive_html)
            print(f"[ARCHIVE] Output archived to: {archive_file}")
        except Exception as e:
            print(f"[WARN] Failed to archive output: {e}")

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
    print(f"🏀 NBA PREDICTION SYSTEM {resolve_version().replace('NBA_v', 'v')} (4 markets: 1H + FG)")
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
        print("\n💡 Try: docker compose logs nba-v33")
        sys.exit(1)
    
    # Step 4: Fetch and display
    fetch_and_display_slate(args.date, args.matchup)


if __name__ == "__main__":
    main()
