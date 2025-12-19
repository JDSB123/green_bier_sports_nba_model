#!/usr/bin/env python3
"""
NBA Slate Analysis via Docker API - DOCKER ONLY
===============================================
Runs analysis EXCLUSIVELY through the Docker containerized prediction service.
This is the ONLY way to run analysis - legacy scripts are disabled.

Usage:
    python scripts/analyze_slate_docker.py --date 2025-12-18
    python scripts/analyze_slate_docker.py --date today
    python scripts/analyze_slate_docker.py --date today --matchup "Lakers"
"""
import argparse
import asyncio
import sys
import re
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import httpx
import json

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CST = ZoneInfo("America/Chicago")
API_BASE_URL = "http://localhost:8090"  # NBA API container port


def _match_game(game: dict, matchup_filter: str) -> bool:
    """Match on one or more matchup filters.

    Supports:
    - Single team: "Lakers"
    - Specific matchup: "Lakers vs Celtics" / "Celtics @ Lakers"
    - Multiple filters: "Lakers, Celtics @ Knicks, Heat"
    """
    if not matchup_filter:
        return True

    home = (game.get("home_team") or "").lower()
    away = (game.get("away_team") or "").lower()
    raw = matchup_filter.strip()
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


def calculate_fire_rating(confidence: float, edge: float, edge_type: str = "pts") -> int:
    """Calculate fire rating (1-5) based on confidence and edge."""
    # Normalize edge to 0-1 scale
    if edge_type == "pct":
        edge_norm = min(abs(edge) / 0.20, 1.0)  # 20% edge = max
    else:  # pts
        edge_norm = min(abs(edge) / 10.0, 1.0)  # 10 pts = max
    
    # Combine confidence and edge (weighted average)
    combined_score = (confidence * 0.6) + (edge_norm * 0.4)
    
    # Map to 1-5 fires
    if combined_score >= 0.85:
        return 5
    elif combined_score >= 0.70:
        return 4
    elif combined_score >= 0.60:
        return 3
    elif combined_score >= 0.52:
        return 2
    else:
        return 1


def american_odds_to_implied_prob(odds: int) -> float:
    """Convert American odds to implied probability."""
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 100 / (odds + 100)


def format_odds(odds: int) -> str:
    """Format odds with + sign for positive."""
    return f"+{odds}" if odds > 0 else str(odds)


def generate_summary_table(analysis_data: list, target_date: str) -> str:
    """Generate summary table with all requested fields."""
    lines = []
    
    # Table header
    lines.append("\n" + "=" * 150)
    lines.append("📊 SUMMARY TABLE - RECOMMENDED PICKS")
    lines.append("=" * 150)
    lines.append("")
    
    # Column headers
    header = (
        f"{'Date/Time (CST)':<20} "
        f"{'Matchup':<35} "
        f"{'Recommended Pick':<30} "
        f"{'Model vs Market':<30} "
        f"{'Edge':<12} "
        f"{'Fire Rating':<12}"
    )
    lines.append(header)
    lines.append("-" * 150)
    
    # Collect all picks
    all_picks = []
    
    for game in analysis_data:
        home_team = game.get("home_team", "")
        away_team = game.get("away_team", "")
        matchup = f"{away_team} @ {home_team}"
        time_cst = game.get("time_cst", "")
        
        comprehensive_edge = game.get("comprehensive_edge", {})
        odds = game.get("odds", {})
        
        # Full Game Spread
        # NOTE: API returns spread.pick as TEAM NAME (not "HOME"/"AWAY"),
        # and spread.pick_line is the line for that picked team (signed correctly).
        fg_spread = comprehensive_edge.get("full_game", {}).get("spread", {})
        if fg_spread.get("pick") and fg_spread.get("pick_line") is not None:
            pick_team = fg_spread.get("pick", "")
            pick_line_val = float(fg_spread.get("pick_line"))
            pick_odds = int(fg_spread.get("pick_odds", fg_spread.get("market_odds", -110)))
            confidence = float(fg_spread.get("confidence", 0) or 0)

            model_margin_home = float(fg_spread.get("model_margin", 0) or 0)  # home - away
            # Convert model margin into picked-team perspective
            if pick_team == away_team:
                model_margin_pick = -model_margin_home
            else:
                model_margin_pick = model_margin_home

            # "Cover edge": how many points the model has over/under the spread (positive = supports the pick)
            cover_edge = model_margin_pick + pick_line_val

            fire_rating = calculate_fire_rating(confidence, abs(cover_edge), "pts")
            fire_emoji = "🔥" * fire_rating

            all_picks.append({
                "time": time_cst,
                "matchup": matchup,
                "pick": f"{pick_team} {pick_line_val:+.1f} ({format_odds(pick_odds)})",
                "model_vs_market": f"Model: {model_margin_pick:+.1f} | Line: {pick_line_val:+.1f}",
                "edge": f"{cover_edge:+.2f} pts",
                "fire": fire_emoji,
                "sort_key": time_cst
            })
        
        # Full Game Total
        fg_total = comprehensive_edge.get("full_game", {}).get("total", {})
        if fg_total.get("pick") and fg_total.get("pick_line") is not None:
            pick_side = str(fg_total.get("pick", "")).upper()  # "OVER" / "UNDER"
            pick_line_val = float(fg_total.get("pick_line"))
            pick_odds = int(fg_total.get("pick_odds", fg_total.get("market_odds", -110)))
            confidence = float(fg_total.get("confidence", 0) or 0)
            model_total = float(fg_total.get("model_total", 0) or 0)

            # Positive = supports the pick by that many points
            if pick_side == "OVER":
                pick_edge = model_total - pick_line_val
            else:  # UNDER
                pick_edge = pick_line_val - model_total

            fire_rating = calculate_fire_rating(confidence, abs(pick_edge), "pts")
            fire_emoji = "🔥" * fire_rating

            all_picks.append({
                "time": time_cst,
                "matchup": matchup,
                "pick": f"{pick_side} {pick_line_val:.1f} ({format_odds(pick_odds)})",
                "model_vs_market": f"Model: {model_total:.1f} | Line: {pick_line_val:.1f}",
                "edge": f"{pick_edge:+.2f} pts",
                "fire": fire_emoji,
                "sort_key": time_cst
            })
        
        # Full Game Moneyline
        # NOTE: moneyline.pick is a TEAM NAME; moneyline edge is stored as edge_home/edge_away.
        fg_ml = comprehensive_edge.get("full_game", {}).get("moneyline", {})
        if fg_ml.get("pick") and fg_ml.get("market_home_odds") is not None and fg_ml.get("market_away_odds") is not None:
            pick_team = fg_ml.get("pick", "")

            market_home_odds = int(fg_ml.get("market_home_odds"))
            market_away_odds = int(fg_ml.get("market_away_odds"))
            market_home_prob = float(fg_ml.get("market_home_prob", 0) or 0)
            market_away_prob = float(fg_ml.get("market_away_prob", 0) or 0)
            edge_home = float(fg_ml.get("edge_home", 0) or 0)
            edge_away = float(fg_ml.get("edge_away", 0) or 0)

            if pick_team == away_team:
                market_odds = market_away_odds
                market_prob = market_away_prob
                edge_prob = edge_away
            else:
                market_odds = market_home_odds
                market_prob = market_home_prob
                edge_prob = edge_home

            model_prob = max(0.0, min(1.0, market_prob + edge_prob))

            # Use model_prob as "confidence" for display scoring
            fire_rating = calculate_fire_rating(model_prob, abs(edge_prob), "pct")
            fire_emoji = "🔥" * fire_rating

            all_picks.append({
                "time": time_cst,
                "matchup": matchup,
                "pick": f"{pick_team} ML ({format_odds(market_odds)})",
                "model_vs_market": f"Model: {model_prob:.1%} | Market: {market_prob:.1%}",
                "edge": f"{edge_prob:+.2%}",
                "fire": fire_emoji,
                "sort_key": time_cst
            })
        
        # First Half Spread
        fh_spread = comprehensive_edge.get("first_half", {}).get("spread", {})
        if fh_spread.get("pick") and fh_spread.get("edge") is not None:
            pick_side = fh_spread.get("pick", "")
            line = fh_spread.get("market_line", 0)
            market_odds = fh_spread.get("market_odds", -110)
            edge = fh_spread.get("edge", 0)
            confidence = fh_spread.get("confidence", 0)
            model_margin = fh_spread.get("model_margin", 0)
            
            if pick_side == "HOME":
                pick_team = home_team
                pick_line = f"{home_team} {line:+.1f}"
            else:
                pick_team = away_team
                pick_line = f"{away_team} {line:+.1f}"
            
            fire_rating = calculate_fire_rating(confidence, abs(edge), "pts")
            fire_emoji = "🔥" * fire_rating
            
            all_picks.append({
                "time": time_cst,
                "matchup": matchup,
                "pick": f"1H {pick_line} ({format_odds(market_odds)})",
                "model_vs_market": f"Model: {model_margin:+.1f} | Market: {line:+.1f}",
                "edge": f"{edge:+.2f} pts",
                "fire": fire_emoji,
                "sort_key": time_cst
            })
        
        # First Half Total
        fh_total = comprehensive_edge.get("first_half", {}).get("total", {})
        if fh_total.get("pick") and fh_total.get("edge") is not None:
            pick_side = fh_total.get("pick", "")
            line = fh_total.get("market_line", 0)
            market_odds = fh_total.get("market_odds", -110)
            edge = fh_total.get("edge", 0)
            confidence = fh_total.get("confidence", 0)
            model_total = fh_total.get("model_total", 0)
            
            pick_line = f"1H {pick_side} {line:.1f}"
            
            fire_rating = calculate_fire_rating(confidence, abs(edge), "pts")
            fire_emoji = "🔥" * fire_rating
            
            all_picks.append({
                "time": time_cst,
                "matchup": matchup,
                "pick": f"{pick_line} ({format_odds(market_odds)})",
                "model_vs_market": f"Model: {model_total:.1f} | Market: {line:.1f}",
                "edge": f"{edge:+.2f} pts",
                "fire": fire_emoji,
                "sort_key": time_cst
            })
    
    # Sort by time
    all_picks.sort(key=lambda x: x["sort_key"])
    
    # Print table rows
    for pick in all_picks:
        row = (
            f"{pick['time']:<20} "
            f"{pick['matchup']:<35} "
            f"{pick['pick']:<30} "
            f"{pick['model_vs_market']:<30} "
            f"{pick['edge']:<12} "
            f"{pick['fire']:<12}"
        )
        lines.append(row)
    
    lines.append("")
    lines.append("=" * 150)
    
    return "\n".join(lines)


async def run_analysis_via_docker(date_str: str, matchup_filter: str | None = None):
    """Run analysis EXCLUSIVELY through Docker API."""
    print("=" * 80)
    print("🏀 NBA SLATE ANALYSIS - DOCKER STACK ONLY")
    print("=" * 80)
    print(f"Connecting to Docker API at {API_BASE_URL}")
    print("⚠️  This is the ONLY way to run analysis - legacy scripts disabled")
    print("")
    
    # Check health
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            health_response = await client.get(f"{API_BASE_URL}/health")
            if health_response.status_code != 200:
                print(f"❌ API health check failed: {health_response.status_code}")
                print("   Make sure the NBA Docker container is running:")
                print("   docker ps --filter 'name=nba'")
                return
            health_data = health_response.json()
            print(f"✅ API Status: {health_data.get('status', 'unknown')}")
            print(f"   Mode: {health_data.get('mode', 'unknown')}")
            print(f"   Markets: {health_data.get('markets', 0)}")
            print(f"   Engine Loaded: {health_data.get('engine_loaded', False)}")
            
            if not health_data.get('engine_loaded', False):
                print("   ⚠️  WARNING: Engine not loaded - models may be missing")
    except Exception as e:
        print(f"❌ Failed to connect to Docker API: {e}")
        print("   Make sure the NBA Docker container is running:")
        print("   docker ps --filter 'name=nba'")
        print("   docker logs nba-api")
        return
    
    # Call comprehensive analysis endpoint
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:  # 5 min timeout for full analysis
            print(f"\n📡 Fetching comprehensive analysis for {date_str}...")
            response = await client.get(
                f"{API_BASE_URL}/slate/{date_str}/comprehensive",
                params={"use_splits": True}
            )
            
            if response.status_code != 200:
                print(f"❌ API request failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return
            
            data = response.json()
            analysis = data.get("analysis", [])
            print(f"✅ Retrieved comprehensive analysis for {len(analysis)} games")
            
            if not analysis:
                print("   No games found for this date")
                return

            if matchup_filter:
                filtered = [g for g in analysis if _match_game(g, matchup_filter)]
                if not filtered:
                    print(f"❌ No games matching '{matchup_filter}'")
                    print("\nAvailable games for this date:")
                    for g in analysis:
                        print(f"  - {g.get('away_team', '')} @ {g.get('home_team', '')} ({g.get('time_cst', 'TBD')})")
                    return
                analysis = filtered
                print(f"🔎 Matchup filter applied: {len(analysis)} game(s) match '{matchup_filter}'")

            # Keep printed/saved data consistent with filtering
            data = dict(data)
            data["analysis"] = analysis
            
            # Generate summary table
            summary_table = generate_summary_table(analysis, date_str)
            print(summary_table)
            
            # Generate text report
            from src.utils.comprehensive_edge import generate_comprehensive_text_report
            from datetime import datetime as dt
            target_date = dt.strptime(data.get("date", date_str), "%Y-%m-%d").date()
            text_report = generate_comprehensive_text_report(analysis, target_date)
            print("\n" + text_report)
            
            # Save reports
            report_dir = PROJECT_ROOT / "data" / "processed"
            report_dir.mkdir(parents=True, exist_ok=True)
            
            date_formatted = data.get("date", date_str).replace("-", "")
            report_path = report_dir / f"slate_analysis_{date_formatted}.txt"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(text_report)
                f.write("\n\n")
                f.write(summary_table)
            print(f"\n📄 Full report saved to: {report_path}")
            
            # Save JSON
            json_path = report_dir / f"slate_analysis_{date_formatted}.json"
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
            print(f"📊 JSON data saved to: {json_path}")
            
    except Exception as e:
        print(f"❌ Error running analysis: {e}")
        import traceback
        traceback.print_exc()


def main():
    # Fix Windows console encoding for emoji support
    import io
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    
    parser = argparse.ArgumentParser(
        description="Analyze NBA slate via Docker API (DOCKER ONLY - legacy scripts disabled)",
        epilog="This script ONLY works through Docker. Make sure the NBA container is running."
    )
    parser.add_argument("--date", help="Date for analysis (YYYY-MM-DD, 'today', or 'tomorrow')")
    parser.add_argument("--matchup", help="Filter to a team or matchup (e.g., 'Lakers' or 'Lakers vs Celtics')")
    args = parser.parse_args()
    
    date_str = args.date or "today"
    asyncio.run(run_analysis_via_docker(date_str, matchup_filter=args.matchup))


if __name__ == "__main__":
    main()
