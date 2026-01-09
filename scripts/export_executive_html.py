#!/usr/bin/env python3
"""
Export NBA Executive Summary to styled HTML file.

This is THE OUTPUT FORMAT:
- Date/Time CST
- Away @ Home matchup with records  
- Model prediction with team and fresh live odds
- Market pricing with team and odds
- Edge (consistent in pts)
- Fire rating

Usage:
    python scripts/export_executive_html.py
    python scripts/export_executive_html.py --date today --api https://nba-gbsv-api.xxx.azurecontainerapps.io
"""
import json
import sys
import os
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
from urllib.request import Request, urlopen
from collections import defaultdict

from src.utils.version import resolve_version
API_PORT = os.getenv("NBA_API_PORT", "8090")
API_BASE = os.getenv("NBA_API_URL", f"http://localhost:{API_PORT}")
CST = ZoneInfo("America/Chicago")


def fetch_executive_data(date: str = "today", api_base: str = None) -> dict:
    """Fetch executive summary from the API."""
    if api_base is None:
        api_base = API_BASE
    url = f"{api_base}/slate/{date}/executive"
    try:
        req = Request(url, headers={"Accept": "application/json"})
        with urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"[ERROR] Failed to fetch from API: {e}")
        sys.exit(1)


def get_fire_tier(fire_rating: str) -> int:
    """Get numeric tier for sorting (3=ELITE, 2=STRONG, 1=GOOD)."""
    if not fire_rating:
        return 1
    fire_count = fire_rating.count('🔥')
    return fire_count if fire_count >= 1 else 1


def generate_html(data: dict, output_path: str):
    """Generate styled HTML from executive summary."""
    plays = data.get("plays", [])
    date_str = data.get("date", "Unknown")
    generated_at = data.get("generated_at", datetime.now(CST).strftime("%Y-%m-%d %I:%M %p CST"))
    version = data.get("version") or resolve_version()
    
    # Categorize plays
    elite_plays = [p for p in plays if get_fire_tier(p.get('fire_rating', '')) >= 3]
    strong_plays = [p for p in plays if get_fire_tier(p.get('fire_rating', '')) == 2]
    good_plays = [p for p in plays if get_fire_tier(p.get('fire_rating', '')) == 1]
    
    # Group by game
    games = defaultdict(list)
    for p in plays:
        games[p.get('matchup', 'Unknown')].append(p)
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NBA Executive Summary - {date_str}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0d0d1a 100%);
            color: #e4e4e4;
            padding: 20px;
            line-height: 1.5;
            min-height: 100vh;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        
        .header {{
            text-align: center;
            padding: 30px 20px;
            background: linear-gradient(135deg, #1e3a5f 0%, #0f2847 100%);
            border-radius: 16px;
            margin-bottom: 25px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .header h1 {{
            font-size: 2.8em;
            margin-bottom: 8px;
            background: linear-gradient(90deg, #00d4ff, #00ff88, #ffd700);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-weight: 800;
            letter-spacing: -1px;
        }}
        .header .meta {{
            color: #88a4c4;
            font-size: 1.1em;
        }}
        .header .meta .version {{
            color: #00d4ff;
            font-weight: 600;
        }}
        
        .summary-box {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .summary-stat {{
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .summary-stat .number {{
            font-size: 2.5em;
            font-weight: 800;
            background: linear-gradient(90deg, #00ff88, #00d4ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .summary-stat.elite .number {{
            background: linear-gradient(90deg, #ff3838, #ff9500);
            -webkit-background-clip: text;
        }}
        .summary-stat.strong .number {{
            background: linear-gradient(90deg, #ff9500, #ffd60a);
            -webkit-background-clip: text;
        }}
        .summary-stat .label {{ color: #888; font-size: 0.9em; margin-top: 5px; }}
        
        .section-title {{
            font-size: 1.5em;
            font-weight: 700;
            margin: 30px 0 15px;
            padding: 12px 20px;
            background: linear-gradient(90deg, rgba(255,56,56,0.2), rgba(255,149,0,0.1));
            border-left: 4px solid #ff3838;
            border-radius: 0 8px 8px 0;
            color: #fff;
        }}
        .section-title.strong {{
            background: linear-gradient(90deg, rgba(255,149,0,0.2), rgba(255,214,10,0.1));
            border-left-color: #ff9500;
        }}
        .section-title.good {{
            background: linear-gradient(90deg, rgba(0,212,255,0.2), rgba(0,255,136,0.1));
            border-left-color: #00d4ff;
        }}
        
        .picks-table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 30px;
            background: rgba(255,255,255,0.02);
            border-radius: 12px;
            overflow: hidden;
        }}
        .picks-table th {{
            background: rgba(0,0,0,0.4);
            color: #00d4ff;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 0.5px;
            padding: 15px 12px;
            text-align: left;
            border-bottom: 2px solid rgba(0,212,255,0.3);
        }}
        .picks-table td {{
            padding: 14px 12px;
            border-bottom: 1px solid rgba(255,255,255,0.05);
            vertical-align: middle;
        }}
        .picks-table tr:hover td {{
            background: rgba(0,212,255,0.05);
        }}
        
        .time {{ color: #00d4ff; font-weight: 600; white-space: nowrap; }}
        .matchup {{ color: #fff; font-weight: 500; }}
        .matchup .records {{ color: #666; font-size: 0.85em; }}
        .period {{ 
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: 700;
            background: rgba(0,212,255,0.2);
            color: #00d4ff;
        }}
        .period.fg {{ background: rgba(0,255,136,0.2); color: #00ff88; }}
        .period.q1 {{ background: rgba(255,214,10,0.2); color: #ffd60a; }}
        .market {{ color: #888; font-size: 0.9em; text-transform: uppercase; }}
        .pick {{
            color: #00ff88;
            font-weight: 700;
            font-size: 1.05em;
        }}
        .odds {{ color: #ffd60a; font-weight: 600; }}
        .model {{ color: #00d4ff; }}
        .line {{ color: #888; }}
        .edge {{
            color: #00ff88;
            font-weight: 700;
            font-size: 1.1em;
        }}
        .edge.negative {{ color: #ff6b6b; }}
        
        .fire {{
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .fire.elite {{
            color: #ff3838;
            text-shadow: 0 0 10px rgba(255,56,56,0.5);
        }}
        .fire.strong {{
            color: #ff9500;
            text-shadow: 0 0 10px rgba(255,149,0,0.5);
        }}
        .fire.good {{
            color: #00d4ff;
        }}
        
        .games-summary {{
            margin: 30px 0;
        }}
        .game-row {{
            display: flex;
            align-items: center;
            padding: 15px 20px;
            background: rgba(255,255,255,0.03);
            border-radius: 8px;
            margin-bottom: 10px;
            border-left: 3px solid #333;
        }}
        .game-row.has-elite {{ border-left-color: #ff3838; background: rgba(255,56,56,0.05); }}
        .game-row.has-strong {{ border-left-color: #ff9500; background: rgba(255,149,0,0.05); }}
        .game-row .game-time {{ color: #00d4ff; width: 140px; font-weight: 600; }}
        .game-row .game-matchup {{ flex: 1; color: #fff; font-weight: 500; }}
        .game-row .game-plays {{ color: #888; text-align: right; }}
        .game-row .game-plays .elite {{ color: #ff3838; font-weight: 700; }}
        .game-row .game-plays .strong {{ color: #ff9500; font-weight: 700; }}
        
        .legend {{
            background: rgba(255,255,255,0.03);
            border-radius: 12px;
            padding: 20px;
            margin-top: 30px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .legend h3 {{ color: #00d4ff; margin-bottom: 15px; }}
        .legend-row {{ display: flex; gap: 30px; margin-bottom: 8px; flex-wrap: wrap; }}
        .legend-item {{ color: #888; font-size: 0.9em; }}
        .legend-item span {{ font-weight: 700; }}
        .legend-item .elite {{ color: #ff3838; }}
        .legend-item .strong {{ color: #ff9500; }}
        .legend-item .good {{ color: #00d4ff; }}
        
        .footer {{
            text-align: center;
            padding: 30px;
            color: #555;
            font-size: 0.9em;
            margin-top: 40px;
        }}
        
        @media (max-width: 1200px) {{
            .picks-table {{ font-size: 0.9em; }}
            .picks-table th, .picks-table td {{ padding: 10px 8px; }}
        }}
        @media (max-width: 768px) {{
            .header h1 {{ font-size: 2em; }}
            .picks-table {{ display: block; overflow-x: auto; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏀 NBA EXECUTIVE SUMMARY</h1>
            <div class="meta">
                {date_str} | Generated: {generated_at} | <span class="version">{version}</span>
            </div>
        </div>
        
        <div class="summary-box">
            <div class="summary-stat">
                <div class="number">{len(plays)}</div>
                <div class="label">TOTAL PLAYS</div>
            </div>
            <div class="summary-stat elite">
                <div class="number">{len(elite_plays)}</div>
                <div class="label">🔥🔥🔥 ELITE</div>
            </div>
            <div class="summary-stat strong">
                <div class="number">{len(strong_plays)}</div>
                <div class="label">🔥🔥 STRONG</div>
            </div>
            <div class="summary-stat">
                <div class="number">{len(good_plays)}</div>
                <div class="label">🔥 GOOD</div>
            </div>
        </div>
"""
    
    # Elite plays section
    if elite_plays:
        html += """
        <div class="section-title">🔥🔥🔥 ELITE PLAYS (Best Bets)</div>
        <table class="picks-table">
            <thead>
                <tr>
                    <th>Time (CST)</th>
                    <th>Matchup</th>
                    <th>Per</th>
                    <th>Market</th>
                    <th>Pick</th>
                    <th>Odds</th>
                    <th>Model</th>
                    <th>Market Line</th>
                    <th>Edge</th>
                    <th>Fire</th>
                </tr>
            </thead>
            <tbody>
"""
        for p in elite_plays:
            html += generate_pick_row(p, "elite")
        html += """
            </tbody>
        </table>
"""
    
    # Strong plays section
    if strong_plays:
        html += """
        <div class="section-title strong">🔥🔥 STRONG PLAYS</div>
        <table class="picks-table">
            <thead>
                <tr>
                    <th>Time (CST)</th>
                    <th>Matchup</th>
                    <th>Per</th>
                    <th>Market</th>
                    <th>Pick</th>
                    <th>Odds</th>
                    <th>Model</th>
                    <th>Market Line</th>
                    <th>Edge</th>
                    <th>Fire</th>
                </tr>
            </thead>
            <tbody>
"""
        for p in strong_plays:
            html += generate_pick_row(p, "strong")
        html += """
            </tbody>
        </table>
"""
    
    # Good plays section
    if good_plays:
        html += """
        <div class="section-title good">🔥 GOOD PLAYS</div>
        <table class="picks-table">
            <thead>
                <tr>
                    <th>Time (CST)</th>
                    <th>Matchup</th>
                    <th>Per</th>
                    <th>Market</th>
                    <th>Pick</th>
                    <th>Odds</th>
                    <th>Model</th>
                    <th>Market Line</th>
                    <th>Edge</th>
                    <th>Fire</th>
                </tr>
            </thead>
            <tbody>
"""
        for p in good_plays:
            html += generate_pick_row(p, "good")
        html += """
            </tbody>
        </table>
"""
    
    # Games summary
    html += """
        <div class="section-title good">📊 PLAYS BY GAME</div>
        <div class="games-summary">
"""
    for matchup, game_plays in games.items():
        elite_count = len([p for p in game_plays if get_fire_tier(p.get('fire_rating', '')) >= 3])
        strong_count = len([p for p in game_plays if get_fire_tier(p.get('fire_rating', '')) == 2])
        time_cst = game_plays[0].get('time_cst', 'TBD')
        
        row_class = ""
        if elite_count > 0:
            row_class = "has-elite"
        elif strong_count > 0:
            row_class = "has-strong"
        
        html += f"""
            <div class="game-row {row_class}">
                <div class="game-time">{time_cst}</div>
                <div class="game-matchup">{matchup}</div>
                <div class="game-plays">
                    {len(game_plays)} plays
                    {f'(<span class="elite">{elite_count} ELITE</span>)' if elite_count > 0 else ''}
                    {f'(<span class="strong">{strong_count} STRONG</span>)' if strong_count > 0 else ''}
                </div>
            </div>
"""
    
    html += """
        </div>
        
        <div class="legend">
            <h3>📖 LEGEND</h3>
            <div class="legend-row">
                <div class="legend-item"><span class="elite">🔥🔥🔥 ELITE</span> = 70%+ confidence AND 5+ pt edge</div>
                <div class="legend-item"><span class="strong">🔥🔥 STRONG</span> = 60%+ confidence AND 3+ pt edge</div>
                <div class="legend-item"><span class="good">🔥 GOOD</span> = Passes all filters</div>
            </div>
            <div class="legend-row">
                <div class="legend-item"><b>PER:</b> FG = Full Game, 1H = First Half, Q1 = First Quarter</div>
                <div class="legend-item"><b>MKT:</b> SPREAD = Point Spread, TOTAL = Over/Under</div>
            </div>
            <div class="legend-row">
                <div class="legend-item"><b>Edge:</b> All edges shown in points (pts) for consistency</div>
            </div>
        </div>
        
        <div class="footer">
            Generated by NBA Prediction System """ + version + """<br>
            Data refreshed: """ + generated_at + """
        </div>
    </div>
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"[SUCCESS] HTML exported to: {output_path}")
    print(f"[INFO] {len(plays)} plays | {len(elite_plays)} ELITE | {len(strong_plays)} STRONG | {len(good_plays)} GOOD")


def generate_pick_row(p: dict, tier: str) -> str:
    """Generate a single pick row for the table."""
    time_cst = p.get('time_cst', 'TBD')
    matchup = p.get('matchup', 'Unknown')
    period = p.get('period', '??')
    market = p.get('market', '??')
    pick = p.get('pick', 'N/A')
    odds = p.get('pick_odds', 'N/A')
    model = p.get('model_prediction', 'N/A')
    line = p.get('market_line', 'N/A')
    edge = p.get('edge', 'N/A')
    fire_rating = p.get('fire_rating', '')
    
    # Period class
    period_class = "1h" if period == "1H" else "fg" if period == "FG" else "q1"
    
    # Fire display
    fire_text = "🔥🔥🔥 ELITE" if tier == "elite" else "🔥🔥 STRONG" if tier == "strong" else "🔥 GOOD"
    
    return f"""
                <tr>
                    <td class="time">{time_cst}</td>
                    <td class="matchup">{matchup}</td>
                    <td><span class="period {period_class}">{period}</span></td>
                    <td class="market">{market}</td>
                    <td class="pick">{pick}</td>
                    <td class="odds">{odds}</td>
                    <td class="model">{model}</td>
                    <td class="line">{line}</td>
                    <td class="edge">{edge}</td>
                    <td class="fire {tier}">{fire_text}</td>
                </tr>
"""


def main():
    parser = argparse.ArgumentParser(description="Export NBA Executive Summary to HTML")
    parser.add_argument("--date", default="today", help="Date for picks (default: today)")
    parser.add_argument("--output", default="nba_picks_today.html", help="Output HTML file path")
    parser.add_argument("--api", default=None, help="API base URL (default: from NBA_API_URL env or localhost:8090)")
    
    args = parser.parse_args()
    
    api_base = args.api or API_BASE
    
    print(f"[FETCH] Fetching executive summary for {args.date}...")
    print(f"[CONFIG] Using API: {api_base}")
    
    data = fetch_executive_data(args.date, api_base)
    
    print(f"[GENERATE] Creating HTML...")
    generate_html(data, args.output)


if __name__ == "__main__":
    main()

