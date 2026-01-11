#!/usr/bin/env python3
"""
Quick script to fetch today's NBA predictions and generate HTML
"""
import json
import os
import sys
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

# API URL from environment variable with fallback to Azure production
API_PORT = os.getenv("NBA_API_PORT", "8090")
API_URL = os.getenv("NBA_API_URL", f"http://localhost:{API_PORT}")


def fetch_predictions(date="today", api_base=None):
    """Fetch predictions from API with error handling."""
    if api_base is None:
        api_base = API_URL
    url = f"{api_base}/slate/{date}"
    print(f"Fetching predictions from {url}...")
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        print(f"[ERROR] API returned error {e.code}: {e.reason}")
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"[ERROR] Failed to connect to API: {e}")
        print("[HINT] Check if API is running or set NBA_API_URL env var")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON response: {e}")
        sys.exit(1)

def generate_html(data):
    """Generate HTML from prediction data"""
    date = data.get("date", "Unknown")
    predictions = data.get("predictions", [])
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NBA Picks - {date}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            margin-bottom: 10px;
            font-size: 2.5em;
        }}
        .date {{
            color: #666;
            margin-bottom: 30px;
            font-size: 1.2em;
        }}
        .game {{
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            background: #fafafa;
        }}
        .matchup {{
            font-size: 1.5em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 15px;
            text-align: center;
        }}
        .game-time {{
            text-align: center;
            color: #666;
            margin-bottom: 20px;
        }}
        .predictions {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
        }}
        .prediction {{
            background: white;
            border: 1px solid #ddd;
            border-radius: 6px;
            padding: 15px;
        }}
        .prediction-header {{
            font-weight: bold;
            color: #667eea;
            margin-bottom: 10px;
            font-size: 1.1em;
        }}
        .prediction-item {{
            margin: 8px 0;
            padding: 8px;
            background: #f8f9fa;
            border-radius: 4px;
        }}
        .label {{
            font-weight: 600;
            color: #555;
        }}
        .value {{
            color: #333;
            margin-left: 10px;
        }}
        .confidence-high {{
            color: #27ae60;
            font-weight: bold;
        }}
        .confidence-medium {{
            color: #f39c12;
            font-weight: bold;
        }}
        .confidence-low {{
            color: #e74c3c;
        }}
        .passes-filter {{
            background: #d4edda;
            border-left: 4px solid #27ae60;
        }}
        .fails-filter {{
            background: #f8d7da;
            border-left: 4px solid #e74c3c;
        }}
        .no-predictions {{
            text-align: center;
            padding: 40px;
            color: #666;
            font-size: 1.2em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🏀 NBA Picks</h1>
        <div class="date">Date: {date}</div>
"""
    
    if not predictions:
        html += '<div class="no-predictions">No games scheduled for today</div>'
    else:
        for game in predictions:
            matchup = game.get("matchup", "Unknown")
            home_team = game.get("home_team", "Home")
            away_team = game.get("away_team", "Away")
            commence_time = game.get("commence_time", "")
            game_predictions = game.get("predictions", {})
            
            # Format time
            if commence_time:
                try:
                    dt = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
                    time_str = dt.strftime("%I:%M %p %Z")
                except:
                    time_str = commence_time
            else:
                time_str = "TBD"
            
            html += f"""
        <div class="game">
            <div class="matchup">{matchup}</div>
            <div class="game-time">⏰ {time_str}</div>
            <div class="predictions">
"""
            
            # Process each period
            for period_name, period_data in game_predictions.items():
                if not period_data:
                    continue
                
                period_display = period_name.replace("_", " ").title()
                html += f'<div class="prediction"><div class="prediction-header">{period_display}</div>'
                
                # Spread
                if "spread" in period_data:
                    spread = period_data["spread"]
                    passes = spread.get("passes_filter", False)
                    confidence = spread.get("confidence", 0)
                    bet_side = spread.get("bet_side", "N/A")
                    spread_line = spread.get("spread_line", "N/A")
                    edge = spread.get("edge", 0)
                    
                    conf_class = "confidence-high" if confidence >= 0.65 else "confidence-medium" if confidence >= 0.55 else "confidence-low"
                    filter_class = "passes-filter" if passes else "fails-filter"
                    
                    html += f'''
                <div class="prediction-item {filter_class}">
                    <div><span class="label">Spread:</span> <span class="value">{bet_side.upper()} {spread_line}</span></div>
                    <div><span class="label">Confidence:</span> <span class="value {conf_class}">{confidence*100:.1f}%</span></div>
                    <div><span class="label">Edge:</span> <span class="value">{edge:.2f}</span></div>
                    {f'<div><span class="label">Filter:</span> <span class="value">{spread.get("filter_reason", "Passed")}</span></div>' if not passes else ''}
                </div>
'''
                
                # Total
                if "total" in period_data:
                    total = period_data["total"]
                    passes = total.get("passes_filter", False)
                    confidence = total.get("confidence", 0)
                    bet_side = total.get("bet_side", "N/A")
                    total_line = total.get("total_line", "N/A")
                    edge = total.get("edge", 0)
                    
                    conf_class = "confidence-high" if confidence >= 0.65 else "confidence-medium" if confidence >= 0.55 else "confidence-low"
                    filter_class = "passes-filter" if passes else "fails-filter"
                    
                    html += f'''
                <div class="prediction-item {filter_class}">
                    <div><span class="label">Total:</span> <span class="value">{bet_side.upper()} {total_line}</span></div>
                    <div><span class="label">Confidence:</span> <span class="value {conf_class}">{confidence*100:.1f}%</span></div>
                    <div><span class="label">Edge:</span> <span class="value">{edge:.2f}</span></div>
                    {f'<div><span class="label">Filter:</span> <span class="value">{total.get("filter_reason", "Passed")}</span></div>' if not passes else ''}
                </div>
'''
                html += '</div>'
            
            html += '</div></div>'
    
    html += """
    </div>
</body>
</html>"""
    
    return html

def main():
    """Main function"""
    print("Fetching today's NBA predictions...")
    data = fetch_predictions("today")
    
    print(f"Found {len(data.get('predictions', []))} games")
    
    html = generate_html(data)
    
    output_file = Path("nba_picks_today.html")
    output_file.write_text(html, encoding='utf-8')
    
    print(f"\nHTML generated: {output_file.absolute()}")
    print(f"Open in browser: file:///{output_file.absolute()}")

if __name__ == "__main__":
    main()

