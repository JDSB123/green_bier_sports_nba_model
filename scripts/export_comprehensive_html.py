#!/usr/bin/env python3
"""
Export NBA comprehensive predictions to styled HTML file.

Usage:
    python scripts/export_comprehensive_html.py
    python scripts/export_comprehensive_html.py --date 2025-12-25
    python scripts/export_comprehensive_html.py --output picks.html
"""
import json
import sys
import os
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
from urllib.request import Request, urlopen

API_PORT = os.getenv("NBA_API_PORT", "8090")
API_BASE = os.getenv("NBA_API_URL", f"http://localhost:{API_PORT}")


def fetch_comprehensive_data(date: str = "today", api_base: str = None) -> dict:
    """Fetch comprehensive predictions from the API."""
    if api_base is None:
        api_base = API_BASE
    url = f"{api_base}/slate/{date}/comprehensive"
    try:
        req = Request(url, headers={"Accept": "application/json"})
        with urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"[ERROR] Failed to fetch from API: {e}")
        sys.exit(1)


def get_fire_rating(edge: float, win_prob: float, market_type: str = "spread") -> str:
    """Calculate fire rating based on edge and win probability."""
    if market_type == "total":
        if edge >= 10 and win_prob >= 0.70:
            return "🔥🔥🔥"
        elif edge >= 7 and win_prob >= 0.60:
            return "🔥🔥"
        elif edge >= 5 and win_prob >= 0.55:
            return "🔥"
    else:  # spread
        if edge >= 7 and win_prob >= 0.70:
            return "🔥🔥🔥"
        elif edge >= 5 and win_prob >= 0.60:
            return "🔥🔥"
        elif edge >= 3 and win_prob >= 0.55:
            return "🔥"
    return ""


def generate_html(data: dict, output_path: str):
    """Generate styled HTML from comprehensive predictions."""
    games = data.get("analysis", [])
    date_str = data.get("date", "")
    version = data.get("version", "")
    
    now_cst = datetime.now(ZoneInfo("America/Chicago"))
    timestamp = now_cst.strftime('%B %d, %Y at %I:%M %p CST')
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NBA Predictions - {date_str}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e4e4e4;
            padding: 20px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header {{
            text-align: center;
            padding: 30px 20px;
            background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }}
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            background: linear-gradient(90deg, #00d4ff, #7b2ff7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        .header .meta {{
            color: #888;
            font-size: 0.9em;
        }}
        .game-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 25px;
            border: 1px solid rgba(255,255,255,0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .game-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        }}
        .game-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid rgba(255,255,255,0.1);
        }}
        .matchup {{
            font-size: 1.4em;
            font-weight: bold;
            color: #fff;
        }}
        .records {{
            color: #888;
            font-size: 0.9em;
        }}
        .game-time {{
            color: #00d4ff;
            font-weight: 600;
            font-size: 0.95em;
        }}
        .markets {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
        }}
        .market {{
            background: rgba(255,255,255,0.03);
            padding: 15px;
            border-radius: 8px;
            border-left: 3px solid #7b2ff7;
        }}
        .market.elite {{
            border-left-color: #ff3838;
            background: rgba(255,56,56,0.08);
        }}
        .market.strong {{
            border-left-color: #ff9500;
            background: rgba(255,149,0,0.08);
        }}
        .market.good {{
            border-left-color: #ffd60a;
            background: rgba(255,214,10,0.08);
        }}
        .market-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }}
        .market-type {{
            font-weight: bold;
            font-size: 1.1em;
            color: #fff;
        }}
        .fire-rating {{
            font-size: 1.2em;
        }}
        .market-row {{
            display: flex;
            justify-content: space-between;
            padding: 6px 0;
            font-size: 0.95em;
        }}
        .label {{
            color: #999;
        }}
        .value {{
            color: #fff;
            font-weight: 500;
        }}
        .pick {{
            color: #00d4ff;
            font-weight: bold;
        }}
        .edge {{
            color: #00ff88;
            font-weight: bold;
        }}
        .edge.negative {{
            color: #ff6b6b;
        }}
        .footer {{
            text-align: center;
            padding: 30px;
            color: #666;
            font-size: 0.9em;
            margin-top: 40px;
        }}
        @media (max-width: 768px) {{
            .header h1 {{
                font-size: 1.8em;
            }}
            .matchup {{
                font-size: 1.1em;
            }}
            .markets {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏀 NBA PREDICTIONS</h1>
            <div class="meta">
                {timestamp} | Version {version} | {len(games)} Games
            </div>
        </div>
"""
    
    for game in games:
        away = game.get("away_team", "")
        home = game.get("home_team", "")
        time_cst = game.get("time_cst", "")
        features = game.get("features", {})
        odds = game.get("odds", {})
        edge_data = game.get("comprehensive_edge", {}).get("full_game", {})
        
        away_record = f"{features.get('away_wins', 0)}-{features.get('away_losses', 0)}"
        home_record = f"{features.get('home_wins', 0)}-{features.get('home_losses', 0)}"
        
        spread_data = edge_data.get("spread", {})
        total_data = edge_data.get("total", {})
        
        # Format pick_line with +/- sign for spreads
        spread_pick_line = spread_data.get("pick_line")
        spread_pick_line_str = f"{spread_pick_line:+.1f}" if spread_pick_line is not None else ""
        spread_edge = abs(spread_data.get("edge") or 0)
        total_edge = abs(total_data.get("edge") or 0)
        
        spread_fire = get_fire_rating(spread_edge, spread_data.get("win_probability", 0), "spread")
        total_fire = get_fire_rating(total_edge, total_data.get("win_probability", 0), "total")
        
        spread_class = "elite" if spread_fire.count("🔥") >= 3 else "strong" if spread_fire.count("🔥") == 2 else "good" if spread_fire else ""
        total_class = "elite" if total_fire.count("🔥") >= 3 else "strong" if total_fire.count("🔥") == 2 else "good" if total_fire else ""
        
        html += f"""
        <div class="game-card">
            <div class="game-header">
                <div>
                    <div class="matchup">{away} <span class="records">({away_record})</span> @ {home} <span class="records">({home_record})</span></div>
                </div>
                <div class="game-time">📅 {time_cst}</div>
            </div>
            
            <div class="markets">
                <div class="market {spread_class}">
                    <div class="market-header">
                        <span class="market-type">SPREAD</span>
                        <span class="fire-rating">{spread_fire}</span>
                    </div>
                    <div class="market-row">
                        <span class="label">Model Pick:</span>
                        <span class="value pick">{spread_data.get('pick', '')} {spread_pick_line_str}</span>
                    </div>
                    <div class="market-row">
                        <span class="label">Win Prob:</span>
                        <span class="value">{round(spread_data.get('win_probability', 0) * 100, 1)}%</span>
                    </div>
                    <div class="market-row">
                        <span class="label">Market Line:</span>
                        <span class="value">{odds.get('home_spread', '')} @ {spread_data.get('market_odds', '')}</span>
                    </div>
                    <div class="market-row">
                        <span class="label">Edge:</span>
                        <span class="value edge">{round(spread_edge, 1)} pts</span>
                    </div>
                </div>
                
                <div class="market {total_class}">
                    <div class="market-header">
                        <span class="market-type">TOTAL</span>
                        <span class="fire-rating">{total_fire}</span>
                    </div>
                    <div class="market-row">
                        <span class="label">Model Pick:</span>
                        <span class="value pick">{total_data.get('pick', '')} {total_data.get('pick_line', '')}</span>
                    </div>
                    <div class="market-row">
                        <span class="label">Win Prob:</span>
                        <span class="value">{round(total_data.get('win_probability', 0) * 100, 1)}%</span>
                    </div>
                    <div class="market-row">
                        <span class="label">Market Line:</span>
                        <span class="value">{odds.get('total', '')} @ {total_data.get('market_odds', '')}</span>
                    </div>
                    <div class="market-row">
                        <span class="label">Edge:</span>
                        <span class="value edge">{round(total_edge, 1)} pts</span>
                    </div>
                </div>
                
            </div>
        </div>
"""
    
    html += f"""
        <div class="footer">
            Generated by NBA Prediction System {version}<br>
            Data refreshed: {timestamp}<br>
            🔥🔥🔥 = Elite (70%+ confidence + 7+ pt edge) | 🔥🔥 = Strong (60%+ + 5+ pt) | 🔥 = Good (55%+ + 3+ pt)
        </div>
    </div>
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"[SUCCESS] HTML exported to: {output_path}")
    print(f"[INFO] {len(games)} games included")


def main():
    parser = argparse.ArgumentParser(description="Export NBA predictions to HTML")
    parser.add_argument("--date", default="today", help="Date to fetch (today, tomorrow, or YYYY-MM-DD)")
    parser.add_argument("--output", default="nba_picks_comprehensive.html", help="Output HTML file path")
    parser.add_argument("--api", default=None, help="API base URL")
    
    args = parser.parse_args()
    
    print(f"[FETCH] Fetching comprehensive predictions for {args.date}...")
    data = fetch_comprehensive_data(args.date, args.api)
    
    print(f"[GENERATE] Creating HTML...")
    generate_html(data, args.output)


if __name__ == "__main__":
    main()
