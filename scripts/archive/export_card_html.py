#!/usr/bin/env python3
"""
Export NBA picks Teams card to HTML file for preview.

Usage:
    python scripts/export_card_html.py
    python scripts/export_card_html.py --date 2025-12-19
    python scripts/export_card_html.py --output preview.html
"""
import json
import sys
import os
import argparse
import re
from datetime import datetime
from zoneinfo import ZoneInfo
from urllib.request import Request, urlopen
from urllib.error import URLError

# Configuration
TEAMS_WEBHOOK_URL = os.getenv("TEAMS_WEBHOOK_URL", "")
API_PORT = os.getenv("NBA_API_PORT", "8090")
API_BASE = os.getenv("NBA_API_URL", f"http://localhost:{API_PORT}")

# Team name to 3-letter abbreviation mapping
TEAM_ABBREV = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "LA Clippers": "LAC", "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM", "Miami Heat": "MIA", "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP", "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR", "Utah Jazz": "UTA",
    "Washington Wizards": "WAS"
}


def get_team_abbrev(team_name: str) -> str:
    """Get 3-letter abbreviation for team name."""
    # Remove record from name like "Boston Celtics (25-10)"
    clean_name = team_name.split("(")[0].strip()
    return TEAM_ABBREV.get(clean_name, clean_name[:3].upper())


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


def get_fire_tier(rating) -> str:
    """Convert fire rating to tier name.
    
    Handles both integer ratings (1-5) from API and legacy emoji strings.
    """
    if isinstance(rating, int):
        fire_count = rating
    elif isinstance(rating, str):
        # Legacy: count fire emojis if present
        fire_count = rating.count("\U0001F525")
        # Or try to parse as integer
        if fire_count == 0 and rating.isdigit():
            fire_count = int(rating)
    else:
        fire_count = 0
    
    if fire_count >= 4:
        return "ELITE"
    elif fire_count == 3:
        return "STRONG"
    elif fire_count == 2:
        return "GOOD"
    elif fire_count == 1:
        return "FAIR"
    return "WATCH"


def _get_fire_count(rating) -> int:
    """Extract fire count from rating (int or string)."""
    if isinstance(rating, int):
        return rating
    elif isinstance(rating, str):
        fire_count = rating.count("\U0001F525")
        if fire_count == 0 and rating.isdigit():
            return int(rating)
        return fire_count
    return 0


def _get_fire_emoji(rating) -> str:
    """Convert fire rating to emoji string."""
    count = _get_fire_count(rating)
    return "🔥" * min(count, 5)


def _parse_edge(edge_str) -> float:
    """Parse edge value from string like '+4.2 pts'."""
    if isinstance(edge_str, (int, float)):
        return float(edge_str)
    if not edge_str:
        return 0.0
    match = re.search(r'([+-]?\d+\.?\d*)', str(edge_str))
    return float(match.group(1)) if match else 0.0


def format_teams_message(data: dict) -> dict:
    """Format NBA picks as Teams Adaptive Card."""
    plays = data.get("plays", [])
    
    # Generate title with CST timestamp
    now_cst = datetime.now(ZoneInfo("America/Chicago"))
    title = f"🏀 NBA PICKS - {now_cst.strftime('%m/%d/%Y')} @ {now_cst.strftime('%I:%M %p').lower()} CST"
    
    # Count by tier
    elite = [p for p in plays if get_fire_tier(p.get("fire_rating", 0)) == "ELITE"]
    strong = [p for p in plays if get_fire_tier(p.get("fire_rating", 0)) == "STRONG"]
    good = [p for p in plays if get_fire_tier(p.get("fire_rating", 0)) == "GOOD"]
    
    # Sort by fire rating (most fires first), then by edge
    sorted_plays = sorted(plays, key=lambda p: (
        -_get_fire_count(p.get("fire_rating", 0)),
        -_parse_edge(p.get("edge", "0"))
    ))
    
    # Build Adaptive Card with improved formatting
    card = {
        "type": "AdaptiveCard",
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "version": "1.4",
        "body": [
            # Header with summary
            {
                "type": "Container",
                "style": "emphasis",
                "items": [
                    {
                        "type": "TextBlock",
                        "text": title,
                        "weight": "Bolder",
                        "size": "Large",
                        "color": "Light"
                    },
                    {
                        "type": "TextBlock",
                        "text": f"📊 {len(plays)} Total | 🔥🔥🔥 {len(elite)} ELITE | 🔥🔥 {len(strong)} STRONG | 🔥 {len(good)} GOOD",
                        "spacing": "Small",
                        "color": "Light",
                        "size": "Medium"
                    }
                ],
                "bleed": True
            },
            # Column headers
            {
                "type": "ColumnSet",
                "columns": [
                    {"type": "Column", "width": "40", "items": [{"type": "TextBlock", "text": "PERIOD", "weight": "Bolder", "size": "Small"}]},
                    {"type": "Column", "width": "stretch", "items": [{"type": "TextBlock", "text": "MATCHUP", "weight": "Bolder", "size": "Small"}]},
                    {"type": "Column", "width": "stretch", "items": [{"type": "TextBlock", "text": "PICK | MODEL", "weight": "Bolder", "size": "Small"}]},
                    {"type": "Column", "width": "60", "items": [{"type": "TextBlock", "text": "MARKET", "weight": "Bolder", "size": "Small"}]},
                    {"type": "Column", "width": "50", "items": [{"type": "TextBlock", "text": "EDGE", "weight": "Bolder", "size": "Small"}]}
                ],
                "separator": True,
                "spacing": "Small"
            }
        ]
    }
    
    # Add rows - one per pick
    for p in sorted_plays:
        # Extract data
        matchup_raw = p.get("matchup", "").strip()
        # Parse: "Miami Heat (15-17) @ Boston Celtics (18-12)" -> "MIA @ BOS"
        if " @ " in matchup_raw:
            away_full, home_full = matchup_raw.split(" @ ", 1)
            away = get_team_abbrev(away_full)
            home = get_team_abbrev(home_full)
        else:
            away = matchup_raw[:3].upper()
            home = "TBD"
        
        period = p.get("period", "FG").upper()
        pick_raw = p.get("pick", "").strip()
        confidence = p.get("model_confidence", p.get("confidence", ""))
        
        # Abbreviate team name in pick display
        # "Denver Nuggets +8.5" -> "DEN +8.5", "OVER 228.5" stays as is
        pick = pick_raw
        if not pick.startswith("OVER") and not pick.startswith("UNDER"):
            # It's a spread pick with team name - abbreviate
            for team_name, abbrev in TEAM_ABBREV.items():
                if team_name in pick:
                    pick = pick.replace(team_name, abbrev)
                    break
        
        # Extract confidence percentage
        conf_pct = ""
        if confidence:
            if isinstance(confidence, str):
                # Extract number from string like "55.2%"
                match = re.search(r'(\d+\.?\d*)', str(confidence))
                if match:
                    conf_pct = f"{float(match.group(1)):.0f}%"
            elif isinstance(confidence, (int, float)):
                conf_pct = f"{float(confidence):.0f}%"
        
        market = p.get("market", "").strip()
        market_line = p.get("market_line", "N/A")
        pick_odds = p.get("pick_odds", "-110")
        
        # Edge display - API returns formatted like "+4.2 pts"
        edge_str = p.get("edge", "")
        if isinstance(edge_str, str) and edge_str:
            edge_display = edge_str.replace(" pts", "")  # Keep the +X.X format
        else:
            edge_display = f"{_parse_edge(edge_str):+.1f}"
        
        # Fire rating handling - API returns integer 1-5
        fire_rating = p.get("fire_rating", 0)
        tier = get_fire_tier(fire_rating)
        fire_emoji = _get_fire_emoji(fire_rating)
        
        # Tier color based on rating
        tier_color = "Attention" if tier == "ELITE" else "Warning" if tier == "STRONG" else "Good" if tier == "GOOD" else "Default"
        
        # Combine pick + confidence
        pick_with_conf = f"{pick}" + (f"\n({conf_pct})" if conf_pct else "")
        
        row = {
            "type": "ColumnSet",
            "columns": [
                {"type": "Column", "width": "40", "items": [{"type": "TextBlock", "text": period, "size": "Small", "weight": "Bolder"}]},
                {"type": "Column", "width": "stretch", "items": [{"type": "TextBlock", "text": f"{away}\n@\n{home}", "size": "Small", "weight": "Bolder", "wrap": True}]},
                {"type": "Column", "width": "stretch", "items": [{"type": "TextBlock", "text": pick_with_conf, "size": "Small", "weight": "Bolder", "color": "Accent", "wrap": True}]},
                {"type": "Column", "width": "60", "items": [{"type": "TextBlock", "text": f"{market_line} ({pick_odds})", "size": "Small", "wrap": True}]},
                {"type": "Column", "width": "50", "items": [{"type": "TextBlock", "text": f"{edge_display}\n{fire_emoji}", "size": "Small", "weight": "Bolder", "color": tier_color, "wrap": True}]}
            ],
            "spacing": "Small",
            "separator": True
        }
        card["body"].append(row)
    
    # Footer with data count
    card["body"].append({
        "type": "TextBlock",
        "text": f"Generated from {len(plays)} picks | Model confidence in PICK column",
        "size": "Small",
        "isSubtle": True,
        "spacing": "Large"
    })
    
    return card


def export_to_html(card: dict, output_file: str = "nba_picks_card.html"):
    """Export Adaptive Card to HTML file."""
    card_json_str = json.dumps(card)
    
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NBA Picks Card</title>
    <script src="https://cdn.jsdelivr.net/npm/adaptivecards@latest/dist/adaptivecards.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Roboto", "Oxygen", "Ubuntu", "Cantarell", sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .container {
            width: 100%;
            max-width: 900px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            padding: 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 20px;
            padding-bottom: 20px;
            border-bottom: 2px solid #eee;
        }
        
        .header h1 {
            font-size: 24px;
            color: #333;
            margin-bottom: 10px;
        }
        
        .header p {
            color: #666;
            font-size: 14px;
        }
        
        #adaptiveCardContainer {
            width: 100%;
        }
        
        .error {
            background: #fee;
            border: 1px solid #fcc;
            color: #c33;
            padding: 15px;
            border-radius: 4px;
            margin: 20px 0;
        }
        
        .json-viewer {
            margin-top: 30px;
            padding-top: 20px;
            border-top: 2px solid #eee;
        }
        
        .json-viewer h3 {
            margin-bottom: 10px;
            color: #333;
        }
        
        .json-code {
            background: #f5f5f5;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 15px;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            color: #333;
            max-height: 400px;
            overflow-y: auto;
        }
        
        .button {
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 10px 20px;
            border-radius: 4px;
            text-decoration: none;
            margin-top: 20px;
            border: none;
            cursor: pointer;
            font-size: 14px;
        }
        
        .button:hover {
            background: #764ba2;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏀 NBA Picks Card Preview</h1>
            <p>Adaptive Card for Microsoft Teams</p>
        </div>
        
        <div id="adaptiveCardContainer"></div>
        
        <button class="button" onclick="downloadJSON()">📥 Download JSON</button>
        
        <div class="json-viewer">
            <h3>Card JSON (for troubleshooting):</h3>
            <div class="json-code"><pre id="jsonOutput"></pre></div>
        </div>
    </div>
    
    <script>
        // Parse card data from HTML
        const cardData = """ + card_json_str + """;
        
        // Render the Adaptive Card
        function renderCard() {
            try {
                const cardContainer = document.getElementById("adaptiveCardContainer");
                const adaptiveCard = new AdaptiveCards.AdaptiveCard();
                
                // Set host config for better styling
                adaptiveCard.hostConfig = new AdaptiveCards.HostConfig({
                    containerStyles: {
                        default: {
                            backgroundColor: "#FFFFFF",
                            foregroundColors: {
                                default: {
                                    default: "#333333"
                                }
                            }
                        },
                        emphasis: {
                            backgroundColor: "#08519E",
                            foregroundColors: {
                                default: {
                                    default: "#FFFFFF"
                                }
                            }
                        }
                    },
                    imageSizes: {
                        small: 40,
                        medium: 80,
                        large: 160
                    },
                    actions: {
                        maxActions: 5,
                        spacing: "default",
                        buttonSpacing: 10
                    },
                    adaptiveCard: {
                        allowCustomStyle: true
                    }
                });
                
                adaptiveCard.parse(cardData);
                const renderedCard = adaptiveCard.render();
                cardContainer.appendChild(renderedCard);
                
                document.getElementById("jsonOutput").textContent = JSON.stringify(cardData, null, 2);
            } catch (error) {
                console.error("Error rendering card:", error);
                document.getElementById("adaptiveCardContainer").innerHTML = 
                    '<div class="error">Error rendering card: ' + error.message + '</div>';
            }
        }
        
        function downloadJSON() {
            const dataStr = JSON.stringify(cardData, null, 2);
            const dataBlob = new Blob([dataStr], {type: 'application/json'});
            const url = URL.createObjectURL(dataBlob);
            const link = document.createElement('a');
            link.href = url;
            link.download = 'nba-picks-card.json';
            link.click();
        }
        
        // Render on page load
        window.addEventListener('load', renderCard);
    </script>
</body>
</html>"""
    
    # Write to file with UTF-8 encoding for emoji support
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Export NBA picks card to HTML")
    parser.add_argument("--date", default="today", help="Date for picks (default: today)")
    parser.add_argument("--output", default="nba_picks_card.html", help="Output HTML file")
    parser.add_argument("--local", action="store_true", help="Use localhost API")
    
    args = parser.parse_args()
    
    # Set API source
    api_base = f"http://localhost:8090" if args.local else API_BASE
    
    print(f"[FETCH] Fetching predictions for {args.date}...")
    print(f"[CONFIG] Using API: {api_base}")
    
    data = fetch_executive_data(args.date, api_base)
    plays = data.get("plays", [])
    
    if not plays:
        print("[WARN] No plays found")
        return
    
    print(f"[DATA] Found {len(plays)} plays")
    
    # Format card
    print("[FORMAT] Generating Adaptive Card...")
    card = format_teams_message(data)
    
    # Export to HTML
    print(f"[EXPORT] Exporting to {args.output}...")
    output_path = export_to_html(card, args.output)
    
    print(f"[OK] Card exported to: {output_path}")
    print(f"[INFO] Open in browser: file:///{os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
