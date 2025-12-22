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
# For Azure: export NBA_API_URL=$(az containerapp show -n nba-picks-api -g NBAGBSVMODEL --query properties.configuration.ingress.fqdn -o tsv | xargs -I{} echo "https://{}")
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
    return "NONE"


def format_teams_message(data: dict) -> dict:
    """Format betting card as Teams Adaptive Card."""
    plays = data.get("plays", [])
    
    # Generate title with CST timestamp
    now_cst = datetime.now(ZoneInfo("America/Chicago"))
    title = f"NBA Picks - {now_cst.strftime('%m/%d/%Y')} as of {now_cst.strftime('%I:%M %p').lower()} cst"
    
    # Count by tier
    elite = [p for p in plays if get_fire_tier(p.get("fire_rating", "")) == "ELITE"]
    strong = [p for p in plays if get_fire_tier(p.get("fire_rating", "")) == "STRONG"]
    good = [p for p in plays if get_fire_tier(p.get("fire_rating", "")) == "GOOD"]
    
    # Sort by fire rating (most fires first), then by edge
    sorted_plays = sorted(plays, key=lambda p: (
        -p.get("fire_rating", "").count("\U0001F525"),
        -float(p.get("edge", "0").replace("+", "").replace(" pts", "").replace("%", "") or 0)
    ))
    
    # Build Adaptive Card
    card = {
        "type": "AdaptiveCard",
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "version": "1.4",
        "body": [
            # Header
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
                        "text": f"{len(plays)} Picks: {len(elite)} ELITE | {len(strong)} STRONG | {len(good)} GOOD",
                        "spacing": "None",
                        "color": "Light"
                    }
                ],
                "bleed": True
            },
            # Column headers
            {
                "type": "ColumnSet",
                "columns": [
                    {"type": "Column", "width": "stretch", "items": [{"type": "TextBlock", "text": "MATCHUP", "weight": "Bolder", "size": "Small"}]},
                    {"type": "Column", "width": "auto", "items": [{"type": "TextBlock", "text": "PER", "weight": "Bolder", "size": "Small"}]},
                    {"type": "Column", "width": "stretch", "items": [{"type": "TextBlock", "text": "PICK", "weight": "Bolder", "size": "Small"}]},
                    {"type": "Column", "width": "auto", "items": [{"type": "TextBlock", "text": "ODDS", "weight": "Bolder", "size": "Small"}]},
                    {"type": "Column", "width": "auto", "items": [{"type": "TextBlock", "text": "EDGE", "weight": "Bolder", "size": "Small"}]},
                    {"type": "Column", "width": "auto", "items": [{"type": "TextBlock", "text": "TIER", "weight": "Bolder", "size": "Small"}]}
                ],
                "separator": True
            }
        ]
    }
    
    # Add rows
    for p in sorted_plays:
        # Extract short matchup (team abbreviations)
        matchup_raw = p.get("matchup", "")
        # Try to shorten: "Miami Heat (15-17) @ Boston Celtics (18-12)" -> "MIA @ BOS"
        matchup = matchup_raw[:28]
        
        period = p.get("period", "FG")
        market = p.get("market", "")[:3]
        pick = p.get("pick", "")[:20]
        odds = p.get("pick_odds", "N/A")
        edge = p.get("edge", "N/A")
        tier = get_fire_tier(p.get("fire_rating", ""))
        
        # Tier color
        tier_color = "Attention" if tier == "ELITE" else "Warning" if tier == "STRONG" else "Good"
        
        row = {
            "type": "ColumnSet",
            "columns": [
                {"type": "Column", "width": "stretch", "items": [{"type": "TextBlock", "text": matchup, "size": "Small", "wrap": True}]},
                {"type": "Column", "width": "auto", "items": [{"type": "TextBlock", "text": period, "size": "Small"}]},
                {"type": "Column", "width": "stretch", "items": [{"type": "TextBlock", "text": f"{pick}", "size": "Small", "weight": "Bolder"}]},
                {"type": "Column", "width": "auto", "items": [{"type": "TextBlock", "text": odds, "size": "Small", "weight": "Bolder"}]},
                {"type": "Column", "width": "auto", "items": [{"type": "TextBlock", "text": edge, "size": "Small"}]},
                {"type": "Column", "width": "auto", "items": [{"type": "TextBlock", "text": tier, "size": "Small", "weight": "Bolder", "color": tier_color}]}
            ],
            "spacing": "Small"
        }
        card["body"].append(row)
    
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
