#!/usr/bin/env python3
"""
Export NBA predictions as clean consolidated table HTML.

Usage:
    python scripts/export_table_html.py
    python scripts/export_table_html.py --date 2025-12-25
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
    """Calculate fire rating."""
    if market_type == "total":
        if edge >= 10 and win_prob >= 0.70:
            return "🔥🔥🔥"
        elif edge >= 7 and win_prob >= 0.60:
            return "🔥🔥"
        elif edge >= 5:
            return "🔥"
    elif market_type == "moneyline":
        if edge >= 20:
            return "🔥🔥🔥"
        elif edge >= 15:
            return "🔥🔥"
        elif edge >= 10:
            return "🔥"
    else:  # spread
        if edge >= 7 and win_prob >= 0.70:
            return "🔥🔥🔥"
        elif edge >= 5 and win_prob >= 0.60:
            return "🔥🔥"
        elif edge >= 3:
            return "🔥"
    return ""


def generate_table_html(data: dict, output_path: str):
    """Generate clean table HTML."""
    games = data.get("analysis", [])
    date_str = data.get("date", "")
    version = data.get("version", "")
    
    now_cst = datetime.now(ZoneInfo("America/Chicago"))
    timestamp = now_cst.strftime('%B %d, %Y @ %I:%M %p CST')
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NBA Picks - {date_str}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #0a0e27;
            color: #e0e0e0;
            padding: 20px;
        }}
        .container {{
            max-width: 1600px;
            margin: 0 auto;
        }}
        .header {{
            text-align: center;
            padding: 25px;
            background: linear-gradient(135deg, #1a1f3a 0%, #0f1419 100%);
            border-radius: 10px;
            margin-bottom: 25px;
            border: 1px solid #2a2f4a;
        }}
        .header h1 {{
            font-size: 2em;
            margin-bottom: 8px;
            color: #00d4ff;
        }}
        .header .meta {{
            color: #888;
            font-size: 0.9em;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: #151a2e;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        }}
        thead {{
            background: #1a2035;
        }}
        th {{
            padding: 15px 10px;
            text-align: left;
            font-weight: 600;
            color: #00d4ff;
            border-bottom: 2px solid #2a2f4a;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        td {{
            padding: 12px 10px;
            border-bottom: 1px solid #1f2437;
            font-size: 0.9em;
        }}
        tr:hover {{
            background: rgba(0,212,255,0.05);
        }}
        .time {{
            color: #00d4ff;
            white-space: nowrap;
        }}
        .matchup {{
            font-weight: 600;
            color: #fff;
        }}
        .records {{
            color: #888;
            font-size: 0.85em;
        }}
        .pick {{
            font-weight: 600;
            color: #00ff88;
        }}
        .odds {{
            color: #ccc;
            font-size: 0.85em;
        }}
        .edge {{
            font-weight: bold;
            color: #00ff88;
        }}
        .fire {{
            font-size: 1.1em;
        }}
        .elite {{
            background: rgba(255,56,56,0.1);
        }}
        .strong {{
            background: rgba(255,149,0,0.1);
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.85em;
            margin-top: 30px;
        }}
        @media (max-width: 1200px) {{
            table {{
                font-size: 0.85em;
            }}
            th, td {{
                padding: 10px 6px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏀 NBA PREDICTIONS TABLE</h1>
            <div class="meta">{timestamp} | {version} | {len(games)} Games</div>
        </div>
        
        <table>
            <thead>
                <tr>
                    <th>Time (CST)</th>
                    <th>Matchup</th>
                    <th>Spread Pick</th>
                    <th>Market</th>
                    <th>Edge</th>
                    <th>🔥</th>
                    <th>Total Pick</th>
                    <th>Market</th>
                    <th>Edge</th>
                    <th>🔥</th>
                    <th>ML Pick</th>
                    <th>Market</th>
                    <th>Edge</th>
                    <th>🔥</th>
                </tr>
            </thead>
            <tbody>
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
        ml_data = edge_data.get("moneyline", {})
        
        spread_edge = abs(spread_data.get("edge", 0))
        total_edge = abs(total_data.get("edge", 0))
        ml_edge = abs(ml_data.get("edge_away", 0) * 100)
        
        spread_fire = get_fire_rating(spread_edge, spread_data.get("win_probability", 0), "spread")
        total_fire = get_fire_rating(total_edge, total_data.get("win_probability", 0), "total")
        ml_fire = get_fire_rating(ml_edge, ml_data.get("model_away_prob", 0), "moneyline")
        
        fire_count = max(spread_fire.count("🔥"), total_fire.count("🔥"), ml_fire.count("🔥"))
        row_class = "elite" if fire_count >= 3 else "strong" if fire_count >= 2 else ""
        
        spread_pick = f"{spread_data.get('pick', '')} {spread_data.get('pick_line', '')}" if spread_data.get('pick') else "—"
        total_pick = f"{total_data.get('pick', '')} {total_data.get('pick_line', '')}" if total_data.get('pick') else "—"
        ml_pick = ml_data.get('pick', '—')
        
        html += f"""
                <tr class="{row_class}">
                    <td class="time">{time_cst}</td>
                    <td>
                        <div class="matchup">{away} @ {home}</div>
                        <div class="records">({away_record}) @ ({home_record})</div>
                    </td>
                    <td class="pick">{spread_pick}</td>
                    <td class="odds">{odds.get('home_spread', '')} @ {spread_data.get('market_odds', '')}</td>
                    <td class="edge">{round(spread_edge, 1)} pts</td>
                    <td class="fire">{spread_fire}</td>
                    <td class="pick">{total_pick}</td>
                    <td class="odds">{odds.get('total', '')} @ {total_data.get('market_odds', '')}</td>
                    <td class="edge">{round(total_edge, 1)} pts</td>
                    <td class="fire">{total_fire}</td>
                    <td class="pick">{ml_pick}</td>
                    <td class="odds">Home {odds.get('home_ml', '')} / Away {odds.get('away_ml', '')}</td>
                    <td class="edge">{round(ml_edge, 1)}%</td>
                    <td class="fire">{ml_fire}</td>
                </tr>
"""
    
    html += f"""
            </tbody>
        </table>
        
        <div class="footer">
            NBA Prediction System {version} | {timestamp}<br>
            🔥🔥🔥 Elite | 🔥🔥 Strong | 🔥 Good
        </div>
    </div>
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"[SUCCESS] Table HTML exported to: {output_path}")
    print(f"[INFO] {len(games)} games in table")


def main():
    parser = argparse.ArgumentParser(description="Export NBA picks as table")
    parser.add_argument("--date", default="today", help="Date (today, tomorrow, or YYYY-MM-DD)")
    parser.add_argument("--output", default="nba_picks_table.html", help="Output file")
    parser.add_argument("--api", default=None, help="API base URL")
    
    args = parser.parse_args()
    
    print(f"[FETCH] Fetching predictions for {args.date}...")
    data = fetch_comprehensive_data(args.date, args.api)
    
    print(f"[GENERATE] Creating table HTML...")
    generate_table_html(data, args.output)


if __name__ == "__main__":
    main()
