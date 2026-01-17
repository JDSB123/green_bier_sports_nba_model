#!/usr/bin/env python3
"""
Post NBA betting card to Microsoft Teams via webhook.

Usage:
    python scripts/post_to_teams.py
    python scripts/post_to_teams.py --date 2025-12-19
"""
import json
import sys
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
from urllib.request import Request, urlopen
from urllib.error import URLError

# Configuration from environment (no hardcoded secrets or URLs)
import os

TEAMS_WEBHOOK_URL = os.getenv("TEAMS_WEBHOOK_URL", "")
if not TEAMS_WEBHOOK_URL:
    print("[ERROR] TEAMS_WEBHOOK_URL environment variable is required")
    print("  Set it via: export TEAMS_WEBHOOK_URL='your_webhook_url'")
    sys.exit(1)

# API URL - defaults to localhost, override with NBA_API_URL env var
# For Azure: export NBA_API_URL=$(az containerapp show -n nba-gbsv-api -g nba-gbsv-model-rg --query properties.configuration.ingress.fqdn -o tsv | xargs -I{} echo "https://{}")
API_PORT = os.getenv("NBA_API_PORT", "8090")
API_BASE = os.getenv("NBA_API_URL", f"http://localhost:{API_PORT}")
NBA_MODEL_VERSION = os.getenv("NBA_MODEL_VERSION", "").strip()
MODEL_PACK_PATH = os.getenv("NBA_MODEL_PACK_PATH", "models/production/model_pack.json")


def load_model_timestamp(path: str = MODEL_PACK_PATH) -> str:
    """
    Get the model updated timestamp (CST) from model_pack.json.
    Returns empty string if not available.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ts = data.get("created_at") or ""
        if ts:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(ZoneInfo("America/Chicago"))
            return dt.strftime("%Y-%m-%d %I:%M %p CST")
    except Exception:
        pass
    return ""


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
    return "NONE"


def format_teams_message(data: dict) -> dict:
    """Format betting card as Teams Adaptive Card."""
    plays = data.get("plays", [])
    model_version = NBA_MODEL_VERSION or data.get("version") or "unknown"
    model_updated_cst = load_model_timestamp()

    # Generate title with CST timestamp
    now_cst = datetime.now(ZoneInfo("America/Chicago"))
    title = f"🏀 NBA PICKS - {now_cst.strftime('%m/%d/%Y')} @ {now_cst.strftime('%I:%M %p').lower()} CST"
    subtitle_parts = [f"Model: {model_version}"]
    if model_updated_cst:
        subtitle_parts.append(f"Model updated: {model_updated_cst}")
    subtitle = " | ".join(subtitle_parts)
    
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
                "bleed": True,
                "backgroundImage": {
                    "url": "",
                    "fillMode": "cover",
                    "horizontalAlignment": "center",
                    "verticalAlignment": "center",
                },
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
                    },
                    {
                        "type": "TextBlock",
                        "text": subtitle,
                        "spacing": "Small",
                        "color": "Light",
                        "size": "Small",
                        "wrap": True
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

    # Footer with model/time stamp (CST)
    footer_text = f"Generated {now_cst.strftime('%Y-%m-%d %I:%M %p')} CST | Model {model_version}"
    if model_updated_cst:
        footer_text += f" | Model updated {model_updated_cst}"
    card["body"].append({
        "type": "TextBlock",
        "text": footer_text,
        "spacing": "Medium",
        "isSubtle": True,
        "size": "Small",
        "wrap": True
    })
    
    # Wrap in Teams message format
    message = {
        "type": "message",
        "attachments": [
            {
                "contentType": "application/vnd.microsoft.card.adaptive",
                "contentUrl": None,
                "content": card
            }
        ]
    }
    
    return message


def post_to_teams(message: dict) -> bool:
    """Post message to Teams webhook."""
    try:
        data = json.dumps(message).encode("utf-8")
        req = Request(
            TEAMS_WEBHOOK_URL,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urlopen(req, timeout=30) as resp:
            if resp.status == 200:
                print("[OK] Successfully posted to Teams!")
                return True
            else:
                print(f"[WARN] Teams returned status {resp.status}")
                return False
    except URLError as e:
        print(f"[ERROR] Failed to post to Teams: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Post NBA picks to Teams")
    parser.add_argument("--date", default="today", help="Date (YYYY-MM-DD or 'today')")
    parser.add_argument("--local", action="store_true", help="Use local Docker API instead of Azure")
    args = parser.parse_args()
    
    # Set API source based on flag (or use environment variable)
    if args.local:
        api_base = f"http://localhost:{API_PORT}"
        print(f"[CONFIG] Using LOCAL API: {api_base}")
    else:
        # Use environment variable or default to localhost
        api_base = os.getenv("NBA_API_URL", f"http://localhost:{API_PORT}")
        print(f"[CONFIG] Using API: {api_base}")
    
    print(f"[FETCH] Fetching predictions for {args.date}...")
    data = fetch_executive_data(args.date, api_base)
    
    plays = data.get("plays", [])
    if not plays:
        print("[WARN] No plays found for this date")
        return
    
    print(f"[DATA] Found {len(plays)} plays")
    
    print("[POST] Posting to Teams...")
    message = format_teams_message(data)
    success = post_to_teams(message)
    
    if success:
        elite_count = len([p for p in plays if get_fire_tier(p.get("fire_rating", "")) == "ELITE"])
        strong_count = len([p for p in plays if get_fire_tier(p.get("fire_rating", "")) == "STRONG"])
        print(f"[OK] Posted: {len(plays)} plays ({elite_count} ELITE, {strong_count} STRONG)")


if __name__ == "__main__":
    main()
