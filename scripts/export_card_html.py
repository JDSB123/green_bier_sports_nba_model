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
from datetime import datetime
from zoneinfo import ZoneInfo
from urllib.request import Request, urlopen
from urllib.error import URLError

# Configuration
TEAMS_WEBHOOK_URL = os.getenv("TEAMS_WEBHOOK_URL", "")
API_PORT = os.getenv("NBA_API_PORT", "8090")
API_BASE = os.getenv("NBA_API_URL", f"http://localhost:{API_PORT}")


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


def get_fire_tier(rating: str) -> str:
    """Convert fire emoji rating to tier name."""
    fire_count = rating.count("\U0001F525")  # Count fire emojis
    if fire_count >= 3:
        return "ELITE"
    elif fire_count == 2:
        return "STRONG"
    elif fire_count == 1:
        return "GOOD"
    return "WATCH"


def format_teams_message(data: dict) -> dict:
    """Format NBA picks as Teams Adaptive Card."""
    plays = data.get("plays", [])
    
    # Generate title with CST timestamp
    now_cst = datetime.now(ZoneInfo("America/Chicago"))
    title = f"🏀 NBA PICKS - {now_cst.strftime('%m/%d/%Y')} @ {now_cst.strftime('%I:%M %p').lower()} CST"
    
    # Count by tier
    elite = [p for p in plays if get_fire_tier(p.get("fire_rating", "")) == "ELITE"]
    strong = [p for p in plays if get_fire_tier(p.get("fire_rating", "")) == "STRONG"]
    good = [p for p in plays if get_fire_tier(p.get("fire_rating", "")) == "GOOD"]
    
    # Sort by fire rating (most fires first), then by edge
    sorted_plays = sorted(plays, key=lambda p: (
        -p.get("fire_rating", "").count("\U0001F525"),
        -float(p.get("edge", "0").replace("+", "").replace(" pts", "").replace("%", "") or 0)
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
        # Clean up: "Miami Heat (15-17) @ Boston Celtics (18-12)" -> "MIA @ BOS"
        teams = matchup_raw.split(" @ ") if " @ " in matchup_raw else ["", ""]
        away = teams[0].split("(")[0].strip()[:10] if len(teams) > 0 else ""
        home = teams[1].split("(")[0].strip()[:10] if len(teams) > 1 else ""
        
        period = p.get("period", "FG").upper()
        pick = p.get("pick", "").strip()
        confidence = p.get("model_confidence", p.get("confidence", ""))
        
        # Extract confidence percentage
        conf_pct = ""
        if confidence:
            if isinstance(confidence, str):
                # Extract number from string like "55.2%"
                import re
                match = re.search(r'(\d+\.?\d*)', str(confidence))
                if match:
                    conf_pct = f"{float(match.group(1)):.0f}%"
            elif isinstance(confidence, (int, float)):
                conf_pct = f"{float(confidence):.0f}%"
        
        market = p.get("market", "").strip()
        market_line = p.get("pick_odds", p.get("market_line", "N/A"))
        
        # Edge calculation: model line vs market line
        edge_str = p.get("edge", "")
        # Standardize edge to points: extract numeric value
        import re
        edge_match = re.search(r'([+-]?\d+\.?\d*)', str(edge_str))
        if edge_match:
            edge_val = float(edge_match.group(1))
            if "%" in str(edge_str):
                # If it's a percentage, convert (rough estimate: divide by 10)
                edge_pts = f"+{edge_val/10:.1f}pts" if edge_val > 0 else f"{edge_val/10:.1f}pts"
            else:
                edge_pts = f"+{edge_val:.1f}pts" if edge_val > 0 else f"{edge_val:.1f}pts"
        else:
            edge_pts = "N/A"
        
        fire_rating = p.get("fire_rating", "")
        tier = get_fire_tier(fire_rating)
        
        # Tier color
        tier_color = "Attention" if tier == "ELITE" else "Warning" if tier == "STRONG" else "Good"
        tier_emoji = "🔥🔥🔥" if tier == "ELITE" else "🔥🔥" if tier == "STRONG" else "🔥"
        
        # Combine pick + confidence
        pick_with_conf = f"{pick}" + (f"\n({conf_pct})" if conf_pct else "")
        
        row = {
            "type": "ColumnSet",
            "columns": [
                {"type": "Column", "width": "40", "items": [{"type": "TextBlock", "text": period, "size": "Small", "weight": "Bolder"}]},
                {"type": "Column", "width": "stretch", "items": [{"type": "TextBlock", "text": f"{away}\n@\n{home}", "size": "Small", "weight": "Bolder", "wrap": True}]},
                {"type": "Column", "width": "stretch", "items": [{"type": "TextBlock", "text": pick_with_conf, "size": "Small", "weight": "Bolder", "color": "Accent", "wrap": True}]},
                {"type": "Column", "width": "60", "items": [{"type": "TextBlock", "text": str(market_line)[:15], "size": "Small", "wrap": True}]},
                {"type": "Column", "width": "50", "items": [{"type": "TextBlock", "text": f"{edge_pts}\n{tier_emoji}", "size": "Small", "weight": "Bolder", "color": tier_color, "wrap": True}]}
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
