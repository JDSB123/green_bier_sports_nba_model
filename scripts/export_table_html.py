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
    """Calculate fire rating on 1-5 scale."""
    # Base fire rating on edge and probability
    fires = 0
    
    if market_type == "total":
        if edge >= 12 and win_prob >= 0.75: fires = 5
        elif edge >= 10 and win_prob >= 0.70: fires = 4
        elif edge >= 8 and win_prob >= 0.65: fires = 3
        elif edge >= 6 and win_prob >= 0.60: fires = 2
        elif edge >= 4: fires = 1
    elif market_type == "total_1h":
        # 1H totals are roughly half of full game, so scale thresholds down (~50%)
        if edge >= 6 and win_prob >= 0.75: fires = 5
        elif edge >= 5 and win_prob >= 0.70: fires = 4
        elif edge >= 4 and win_prob >= 0.65: fires = 3
        elif edge >= 3 and win_prob >= 0.60: fires = 2
        elif edge >= 2: fires = 1
    elif market_type == "spread_1h":
        # 1H spreads are roughly half of full game, so scale thresholds down (~50%)
        if edge >= 5 and win_prob >= 0.75: fires = 5
        elif edge >= 4 and win_prob >= 0.70: fires = 4
        elif edge >= 3 and win_prob >= 0.65: fires = 3
        elif edge >= 2 and win_prob >= 0.60: fires = 2
        elif edge >= 1: fires = 1
    else:  # spread (full game)
        if edge >= 10 and win_prob >= 0.75: fires = 5
        elif edge >= 8 and win_prob >= 0.70: fires = 4
        elif edge >= 6 and win_prob >= 0.65: fires = 3
        elif edge >= 4 and win_prob >= 0.60: fires = 2
        elif edge >= 2: fires = 1
        
    return "🔥" * fires


def generate_table_html(data: dict, output_path: str):
    """Generate clean table HTML with requested columns."""
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
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #0a0e27;
            color: #e0e0e0;
            padding: 20px;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .header {{
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #1a1f3a 0%, #0f1419 100%);
            border-radius: 10px;
            margin-bottom: 20px;
            border: 1px solid #2a2f4a;
        }}
        .header h1 {{ font-size: 1.8em; margin-bottom: 5px; color: #00d4ff; }}
        .header .meta {{ color: #888; font-size: 0.9em; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: #151a2e;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        }}
        thead {{ background: #1a2035; }}
        th {{
            padding: 12px;
            text-align: left;
            font-weight: 600;
            color: #00d4ff;
            border-bottom: 2px solid #2a2f4a;
            font-size: 0.85em;
            text-transform: uppercase;
        }}
        td {{
            padding: 12px;
            border-bottom: 1px solid #1f2437;
            font-size: 0.9em;
            vertical-align: middle;
        }}
        tr.game-separator {{ border-top: 2px solid #2a2f4a; }}
        .time {{ color: #888; font-size: 0.85em; }}
        .matchup {{ font-weight: 600; color: #fff; }}
        .pick {{ font-weight: 700; color: #00ff88; }}
        .model {{ color: #ccc; font-size: 0.9em; }}
        .market {{ color: #888; font-size: 0.9em; }}
        .edge {{ font-weight: bold; color: #00ff88; }}
        .fire {{ font-size: 1.2em; letter-spacing: 2px; }}
        .type-badge {{
            display: inline-block;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.75em;
            font-weight: 600;
            margin-right: 8px;
            min-width: 80px;
            text-align: center;
        }}
        .type-spread {{ background: rgba(0, 212, 255, 0.15); color: #00d4ff; }}
        .type-total {{ background: rgba(255, 149, 0, 0.15); color: #ff9500; }}
        .type-1h {{ border: 1px solid rgba(255, 255, 255, 0.2); }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.8em;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏀 NBA PREDICTIONS</h1>
            <div class="meta">{timestamp} | {version} | {len(games)} Games</div>
        </div>
        
        <table>
            <thead>
                <tr>
                    <th>Date & Time (CST)</th>
                    <th>Matchup</th>
                    <th>Recommended Pick</th>
                    <th>Model Prediction</th>
                    <th>Market Pricing</th>
                    <th>Edge</th>
                    <th>Fire Rating</th>
                </tr>
            </thead>
            <tbody>
"""
    
    for i, game in enumerate(games):
        away = game.get("away_team", "")
        home = game.get("home_team", "")
        time_cst = game.get("time_cst", "")
        features = game.get("features", {})
        odds = game.get("odds", {})
        
        edge_fg = game.get("comprehensive_edge", {}).get("full_game", {})
        edge_1h = game.get("comprehensive_edge", {}).get("first_half", {})
        
        # --- FULL GAME SPREAD ---
        spread_data = edge_fg.get("spread", {})
        spread_edge = abs(spread_data.get("edge", 0))
        spread_fire = get_fire_rating(spread_edge, spread_data.get("win_probability", 0), "spread")
        
        pick_team = spread_data.get('pick')
        pick_line = spread_data.get('pick_line')
        pick_odds = spread_data.get('pick_odds', -110)
        
        if pick_team and pick_team != '—':
            line_str = f"+{pick_line}" if pick_line is not None and pick_line > 0 else str(pick_line) if pick_line is not None else ""
            spread_pick_display = f"{pick_team} {line_str} ({pick_odds})"
        else:
            spread_pick_display = "—"
            
        home_score = round(features.get("home_expected_pts", 0), 1)
        away_score = round(features.get("away_expected_pts", 0), 1)
        model_pred_display = f"{away} {away_score} - {home} {home_score}"
        
        home_spread = odds.get('home_spread', '')
        market_display = f"{home} {home_spread} ({spread_data.get('market_odds', '')})"
        
        html += f"""
                <tr class="game-separator">
                    <td class="time">{time_cst}</td>
                    <td class="matchup">{away} @ {home}</td>
                    <td class="pick"><span class="type-badge type-spread">FG SPREAD</span> {spread_pick_display}</td>
                    <td class="model">{model_pred_display}</td>
                    <td class="market">{market_display}</td>
                    <td class="edge">{round(spread_edge, 1)} pts</td>
                    <td class="fire">{spread_fire}</td>
                </tr>
        """
        
        # --- FULL GAME TOTAL ---
        total_data = edge_fg.get("total", {})
        total_edge = abs(total_data.get("edge", 0))
        total_fire = get_fire_rating(total_edge, total_data.get("win_probability", 0), "total")
        
        pick_type = total_data.get('pick')
        pick_line = total_data.get('pick_line')
        pick_odds = total_data.get('pick_odds', -110)
        
        if pick_type and pick_type != '—':
            total_pick_display = f"{pick_type} {pick_line} ({pick_odds})"
        else:
            total_pick_display = "—"
            
        model_total = round(features.get("predicted_total", 0), 1)
        model_total_display = f"Total: {model_total}"
        market_total_display = f"{odds.get('total', '')} ({total_data.get('market_odds', '')})"
        
        html += f"""
                <tr>
                    <td class="time"></td>
                    <td class="matchup"></td>
                    <td class="pick"><span class="type-badge type-total">FG TOTAL</span> {total_pick_display}</td>
                    <td class="model">{model_total_display}</td>
                    <td class="market">{market_total_display}</td>
                    <td class="edge">{round(total_edge, 1)} pts</td>
                    <td class="fire">{total_fire}</td>
                </tr>
        """

        # --- 1H SPREAD ---
        spread_1h = edge_1h.get("spread", {})
        spread_1h_edge = abs(spread_1h.get("edge", 0))
        spread_1h_fire = get_fire_rating(spread_1h_edge, spread_1h.get("win_probability", 0), "spread_1h")
        
        pick_team = spread_1h.get('pick')
        pick_line = spread_1h.get('pick_line')
        pick_odds = spread_1h.get('pick_odds', -110)
        
        if pick_team and pick_team != '—':
            line_str = f"+{pick_line}" if pick_line is not None and pick_line > 0 else str(pick_line) if pick_line is not None else ""
            spread_1h_pick_display = f"{pick_team} {line_str} ({pick_odds})"
        else:
            spread_1h_pick_display = "—"
            
        # Try to get 1H scores if available, else just show margin
        fh_home_score = features.get("fh_home_expected_pts")
        fh_away_score = features.get("fh_away_expected_pts")
        
        if fh_home_score and fh_away_score:
             model_1h_display = f"{away} {round(fh_away_score, 1)} - {home} {round(fh_home_score, 1)}"
        else:
             # Fallback if specific scores aren't in features (though they should be)
             model_1h_display = "—"

        fh_home_spread = odds.get('fh_home_spread', '')
        market_1h_display = f"{home} {fh_home_spread} ({spread_1h.get('market_odds', '')})"
        
        html += f"""
                <tr>
                    <td class="time"></td>
                    <td class="matchup"></td>
                    <td class="pick"><span class="type-badge type-spread type-1h">1H SPREAD</span> {spread_1h_pick_display}</td>
                    <td class="model">{model_1h_display}</td>
                    <td class="market">{market_1h_display}</td>
                    <td class="edge">{round(spread_1h_edge, 1)} pts</td>
                    <td class="fire">{spread_1h_fire}</td>
                </tr>
        """

        # --- 1H TOTAL ---
        total_1h = edge_1h.get("total", {})
        total_1h_edge = abs(total_1h.get("edge", 0))
        total_1h_fire = get_fire_rating(total_1h_edge, total_1h.get("win_probability", 0), "total_1h")
        
        pick_type = total_1h.get('pick')
        pick_line = total_1h.get('pick_line')
        pick_odds = total_1h.get('pick_odds', -110)
        
        if pick_type and pick_type != '—':
            total_1h_pick_display = f"{pick_type} {pick_line} ({pick_odds})"
        else:
            total_1h_pick_display = "—"
            
        fh_model_total = features.get("fh_predicted_total")
        model_1h_total_display = f"Total: {round(fh_model_total, 1)}" if fh_model_total else "—"
        
        market_1h_total_display = f"{odds.get('fh_total', '')} ({total_1h.get('market_odds', '')})"
        
        html += f"""
                <tr>
                    <td class="time"></td>
                    <td class="matchup"></td>
                    <td class="pick"><span class="type-badge type-total type-1h">1H TOTAL</span> {total_1h_pick_display}</td>
                    <td class="model">{model_1h_total_display}</td>
                    <td class="market">{market_1h_total_display}</td>
                    <td class="edge">{round(total_1h_edge, 1)} pts</td>
                    <td class="fire">{total_1h_fire}</td>
                </tr>
        """

    html += f"""
            </tbody>
        </table>
        
        <div class="footer">
            NBA Prediction System {version} | {timestamp}<br>
            Fire Rating: 🔥 (Good) to 🔥🔥🔥🔥🔥 (Elite)
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
