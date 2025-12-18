"""
Live Pick Tracker - HTML Dashboard for Microsoft Teams
Returns HTML page for tracking picks in real-time
"""
import logging
import json
import os
from datetime import datetime
import azure.functions as func
from typing import Dict, Any
import sys
from pathlib import Path

FUNCTION_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(FUNCTION_ROOT))
from generate_picks import _generate_picks_async
import asyncio

logger = logging.getLogger(__name__)


def main(req: func.HttpRequest) -> func.HttpResponse:
    """
    Generate live HTML tracker for NBA picks.
    
    Query parameters:
    - date: 'today', 'tomorrow', or YYYY-MM-DD
    - auto_refresh: seconds (default: 60) - auto-refresh interval
    
    Returns HTML page that can be embedded in Teams or opened in browser.
    """
    try:
        date_param = req.params.get('date', 'today')
        auto_refresh = int(req.params.get('auto_refresh', 60))
        
        # Generate picks
        try:
            result = asyncio.run(_generate_picks_async(date_param, None, 'json'))
        except Exception as e:
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>NBA Picks Tracker - Error</title>
                <style>
                    body {{ font-family: Arial, sans-serif; padding: 20px; background: #f5f5f5; }}
                    .error {{ background: #ffebee; padding: 20px; border-radius: 8px; color: #c62828; }}
                </style>
            </head>
            <body>
                <div class="error">
                    <h2>Error Loading Picks</h2>
                    <p>{str(e)}</p>
                </div>
            </body>
            </html>
            """
            return func.HttpResponse(html, mimetype="text/html")
        
        # Generate HTML dashboard
        html = _generate_tracker_html(result, auto_refresh)
        
        return func.HttpResponse(html, mimetype="text/html")
        
    except Exception as e:
        logger.error(f"Error generating tracker: {e}", exc_info=True)
        return func.HttpResponse(
            f"<html><body><h1>Error</h1><p>{str(e)}</p></body></html>",
            status_code=500,
            mimetype="text/html"
        )


def _generate_tracker_html(result: Dict[str, Any], auto_refresh: int = 60) -> str:
    """Generate HTML dashboard for live pick tracking."""
    
    date_str = result.get("date", "N/A")
    total_plays = result.get("total_plays", 0)
    games_count = result.get("games", 0)
    predictions = result.get("predictions", [])
    
    # Calculate status colors
    plays_html = ""
    for pred in predictions:
        matchup = pred.get("matchup", "Unknown")
        plays = pred.get("plays", [])
        commence_time = pred.get("commence_time", "")
        
        plays_list = ""
        for play in plays:
            period = play.get("period", "")
            market = play.get("market", "")
            pick = play.get("pick", "")
            edge = play.get("edge", 0)
            confidence = play.get("confidence", 0)
            line = play.get("line")
            odds = play.get("odds")
            
            # Calculate fire rating
            fire_count = _calculate_fire_count(confidence, abs(edge))
            fire_emoji = "🔥" * fire_count
            
            # Status badge (will be updated via JavaScript when games are live)
            status_badge = '<span class="status-badge pending">⏳ PENDING</span>'
            
            plays_list += f"""
            <tr>
                <td>{period} {market}</td>
                <td><strong>{pick}</strong></td>
                <td>{line if line else "N/A"}</td>
                <td>{odds if odds else "N/A"}</td>
                <td>{edge:+.1f}</td>
                <td>{confidence:.1%}</td>
                <td>{fire_emoji}</td>
                <td>{status_badge}</td>
            </tr>
            """
        
        plays_html += f"""
        <div class="game-card">
            <div class="game-header">
                <h3>{matchup}</h3>
                <span class="game-time">{commence_time}</span>
            </div>
            <table class="picks-table">
                <thead>
                    <tr>
                        <th>Market</th>
                        <th>Pick</th>
                        <th>Line</th>
                        <th>Odds</th>
                        <th>Edge</th>
                        <th>Confidence</th>
                        <th>Rating</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {plays_list}
                </tbody>
            </table>
        </div>
        """
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>NBA Picks Live Tracker - {date_str}</title>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="{auto_refresh}">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header .stats {{
            display: flex;
            justify-content: center;
            gap: 40px;
            margin-top: 20px;
        }}
        
        .stat-item {{
            text-align: center;
        }}
        
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            display: block;
        }}
        
        .stat-label {{
            font-size: 0.9em;
            opacity: 0.9;
            margin-top: 5px;
        }}
        
        .content {{
            padding: 30px;
        }}
        
        .game-card {{
            background: #f8f9fa;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            border: 2px solid #e9ecef;
        }}
        
        .game-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 15px;
            border-bottom: 2px solid #dee2e6;
        }}
        
        .game-header h3 {{
            color: #1e3c72;
            font-size: 1.5em;
        }}
        
        .game-time {{
            color: #6c757d;
            font-weight: 500;
        }}
        
        .picks-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        
        .picks-table th {{
            background: #1e3c72;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        
        .picks-table td {{
            padding: 12px;
            border-bottom: 1px solid #dee2e6;
        }}
        
        .picks-table tbody tr:hover {{
            background: #f1f3f5;
        }}
        
        .status-badge {{
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            display: inline-block;
        }}
        
        .status-badge.pending {{
            background: #fff3cd;
            color: #856404;
        }}
        
        .status-badge.live {{
            background: #d4edda;
            color: #155724;
            animation: pulse 2s infinite;
        }}
        
        .status-badge.win {{
            background: #d1ecf1;
            color: #0c5460;
        }}
        
        .status-badge.loss {{
            background: #f8d7da;
            color: #721c24;
        }}
        
        .status-badge.push {{
            background: #e2e3e5;
            color: #383d41;
        }}
        
        @keyframes pulse {{
            0%, 100% {{
                opacity: 1;
            }}
            50% {{
                opacity: 0.7;
            }}
        }}
        
        .footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #6c757d;
            border-top: 1px solid #dee2e6;
        }}
        
        .refresh-indicator {{
            display: inline-block;
            width: 12px;
            height: 12px;
            background: #28a745;
            border-radius: 50%;
            margin-right: 8px;
            animation: blink 1s infinite;
        }}
        
        @keyframes blink {{
            0%, 100% {{
                opacity: 1;
            }}
            50% {{
                opacity: 0.3;
            }}
        }}
    </style>
    <script>
        // Auto-refresh page every {auto_refresh} seconds
        setTimeout(function() {{
            location.reload();
        }}, {auto_refresh * 1000});
        
        // Update status badges based on game times (simplified - in production, check live scores)
        document.addEventListener('DOMContentLoaded', function() {{
            const now = new Date();
            const commenceTimes = document.querySelectorAll('.game-time');
            
            commenceTimes.forEach(function(timeEl) {{
                // In production, parse commence_time and update status
                // For now, just mark as live if within game window
            }});
        }});
    </script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏀 NBA Picks Live Tracker</h1>
            <p>Date: {date_str}</p>
            <div class="stats">
                <div class="stat-item">
                    <span class="stat-value">{total_plays}</span>
                    <span class="stat-label">Total Plays</span>
                </div>
                <div class="stat-item">
                    <span class="stat-value">{games_count}</span>
                    <span class="stat-label">Games</span>
                </div>
            </div>
        </div>
        
        <div class="content">
            {plays_html if plays_html else '<p style="text-align: center; padding: 40px; color: #6c757d;">No picks available for this date.</p>'}
        </div>
        
        <div class="footer">
            <span class="refresh-indicator"></span>
            Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | Auto-refresh: {auto_refresh}s
        </div>
    </div>
</body>
</html>
"""
    
    return html


def _calculate_fire_count(confidence: float, edge: float) -> int:
    """Calculate fire rating (1-5 fires)."""
    edge_norm = min(edge / 10.0, 1.0)
    combined = (confidence * 0.6) + (edge_norm * 0.4)
    
    if combined >= 0.85:
        return 5
    elif combined >= 0.70:
        return 4
    elif combined >= 0.60:
        return 3
    elif combined >= 0.52:
        return 2
    else:
        return 1