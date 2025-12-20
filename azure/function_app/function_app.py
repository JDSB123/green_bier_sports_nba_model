"""
Azure Function: NBA Picks Trigger

Triggers the NBA prediction API and posts results to Teams.
Can be called via HTTP from anywhere (Teams, browser, Power Automate, etc.)

Endpoints:
  GET /api/nba-picks         - Get today's picks and post to Teams
  GET /api/nba-picks?date=YYYY-MM-DD - Get picks for specific date
  GET /api/health            - Health check
"""
import azure.functions as func
import json
import logging
import os
from datetime import datetime
from urllib.request import Request, urlopen

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# Configuration - SINGLE SOURCE OF TRUTH
# Resource Group: greenbier-enterprise-rg
# Container App: nba-picks-api
# FQDN changes when environment is recreated - always get dynamically:
#   az containerapp show -n nba-picks-api -g greenbier-enterprise-rg --query properties.configuration.ingress.fqdn -o tsv


def _get_nba_api_url() -> str:
    """Resolve NBA API URL from env each request (no restart required).

    REQUIRES: NBA_API_URL set in Azure Function App Settings.
    Get current FQDN: az containerapp show -n nba-picks-api -g greenbier-enterprise-rg --query properties.configuration.ingress.fqdn -o tsv
    """
    url = os.environ.get("NBA_API_URL", "").strip()
    if not url:
        logging.error("NBA_API_URL environment variable is not set. Configure in Azure Function App Settings.")
        raise RuntimeError("NBA_API_URL is required - set in Azure Function App Settings")
    return url


def _get_teams_webhook_url() -> str:
    """Resolve Teams webhook URL from env each request (no restart required)."""
    return os.environ.get("TEAMS_WEBHOOK_URL", "").strip()


def get_fire_tier(rating: str) -> str:
    """Convert fire emoji rating to tier name."""
    fire_count = rating.count("\U0001F525")
    if fire_count >= 3:
        return "ELITE"
    elif fire_count == 2:
        return "STRONG"
    elif fire_count == 1:
        return "GOOD"
    return "NONE"


def fetch_predictions(date: str = "today") -> dict:
    """Fetch predictions from NBA API."""
    url = f"{_get_nba_api_url()}/slate/{date}/executive"
    try:
        req = Request(url, headers={"Accept": "application/json"})
        with urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        logging.error(f"Failed to fetch predictions: {e}")
        return None


def format_teams_card(data: dict) -> dict:
    """Format predictions as Teams Adaptive Card."""
    plays = data.get("plays", [])
    
    # Use API-provided timestamp or generate one
    generated_at = data.get("generated_at", datetime.now().strftime("%m/%d/%Y %I:%M %p CST"))
    title = f"NBA Picks - {generated_at}"
    
    # Count by tier
    elite = [p for p in plays if get_fire_tier(p.get("fire_rating", "")) == "ELITE"]
    strong = [p for p in plays if get_fire_tier(p.get("fire_rating", "")) == "STRONG"]
    good = [p for p in plays if get_fire_tier(p.get("fire_rating", "")) == "GOOD"]
    
    # Sort by fire rating
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
            {
                "type": "Container",
                "style": "emphasis",
                "items": [
                    {"type": "TextBlock", "text": title, "weight": "Bolder", "size": "Large", "color": "Light"},
                    {"type": "TextBlock", "text": f"{len(plays)} Picks: {len(elite)} ELITE | {len(strong)} STRONG | {len(good)} GOOD", "spacing": "None", "color": "Light"}
                ],
                "bleed": True
            },
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
    
    for p in sorted_plays:
        matchup = p.get("matchup", "")[:28]
        period = p.get("period", "FG")
        pick = p.get("pick", "")[:20]
        odds = p.get("pick_odds", "N/A")
        edge = p.get("edge", "N/A")
        tier = get_fire_tier(p.get("fire_rating", ""))
        tier_color = "Attention" if tier == "ELITE" else "Warning" if tier == "STRONG" else "Good"
        
        row = {
            "type": "ColumnSet",
            "columns": [
                {"type": "Column", "width": "stretch", "items": [{"type": "TextBlock", "text": matchup, "size": "Small", "wrap": True}]},
                {"type": "Column", "width": "auto", "items": [{"type": "TextBlock", "text": period, "size": "Small"}]},
                {"type": "Column", "width": "stretch", "items": [{"type": "TextBlock", "text": pick, "size": "Small", "weight": "Bolder"}]},
                {"type": "Column", "width": "auto", "items": [{"type": "TextBlock", "text": odds, "size": "Small", "weight": "Bolder"}]},
                {"type": "Column", "width": "auto", "items": [{"type": "TextBlock", "text": edge, "size": "Small"}]},
                {"type": "Column", "width": "auto", "items": [{"type": "TextBlock", "text": tier, "size": "Small", "weight": "Bolder", "color": tier_color}]}
            ],
            "spacing": "Small"
        }
        card["body"].append(row)
    
    return {
        "type": "message",
        "attachments": [{"contentType": "application/vnd.microsoft.card.adaptive", "contentUrl": None, "content": card}]
    }


def post_to_teams(message: dict) -> bool:
    """Post message to Teams webhook."""
    webhook_url = _get_teams_webhook_url()
    if not webhook_url:
        logging.error("TEAMS_WEBHOOK_URL is not set (App Setting missing).")
        return False
    try:
        data = json.dumps(message).encode("utf-8")
        req = Request(webhook_url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(req, timeout=30) as resp:
            # Teams webhooks typically return 200 + body "1", but treat any 2xx as success.
            status = getattr(resp, "status", None)
            try:
                body = resp.read().decode(errors="replace")
            except Exception:
                body = ""
            if status is None:
                logging.warning("Teams webhook response missing status; treating as failure.")
                return False
            if 200 <= int(status) < 300:
                logging.info(f"Teams webhook accepted message (status={status}).")
                return True
            logging.error(f"Teams webhook rejected message (status={status}, body={body[:200]!r}).")
            return False
    except Exception as e:
        # URLError + any other unexpected exception
        logging.error(f"Failed to post to Teams: {e}")
        return False


@app.route(route="nba-picks", methods=["GET"])
def nba_picks(req: func.HttpRequest) -> func.HttpResponse:
    """
    Main trigger endpoint - fetches predictions and posts to Teams.
    
    Query params:
      - date: YYYY-MM-DD or 'today' (default: today)
      - post: 'true' to post to Teams, 'false' for JSON only (default: true)
    """
    logging.info("NBA Picks trigger invoked")
    
    date = req.params.get("date", "today")
    should_post = req.params.get("post", "true").lower() == "true"
    
    # Fetch predictions
    data = fetch_predictions(date)
    if not data:
        return func.HttpResponse(
            json.dumps({"error": "Failed to fetch predictions from API"}),
            status_code=500,
            mimetype="application/json"
        )
    
    plays = data.get("plays", [])
    if not plays:
        return func.HttpResponse(
            json.dumps({"message": "No picks available for this date", "date": date}),
            status_code=200,
            mimetype="application/json"
        )
    
    # Sort by fire rating
    sorted_plays = sorted(plays, key=lambda p: (
        -p.get("fire_rating", "").count("\U0001F525"),
        -float(p.get("edge", "0").replace("+", "").replace(" pts", "").replace("%", "") or 0)
    ))
    
    # Count tiers
    elite = len([p for p in plays if get_fire_tier(p.get("fire_rating", "")) == "ELITE"])
    strong = len([p for p in plays if get_fire_tier(p.get("fire_rating", "")) == "STRONG"])
    good = len([p for p in plays if get_fire_tier(p.get("fire_rating", "")) == "GOOD"])
    
    # Build simplified picks list for response
    picks_list = []
    for p in sorted_plays:
        picks_list.append({
            "matchup": p.get("matchup", ""),
            "period": p.get("period", "FG"),
            "market": p.get("market", ""),
            "pick": p.get("pick", ""),
            "odds": p.get("pick_odds", "N/A"),
            "edge": p.get("edge", "N/A"),
            "tier": get_fire_tier(p.get("fire_rating", ""))
        })
    
    result = {
        "date": date,
        "total_picks": len(plays),
        "elite": elite,
        "strong": strong,
        "good": good,
        "posted_to_teams": False,
        "picks": picks_list
    }
    
    # Post to Teams if requested
    if should_post:
        message = format_teams_card(data)
        posted = post_to_teams(message)
        result["posted_to_teams"] = posted
        if posted:
            logging.info(f"Posted {len(plays)} picks to Teams")
        else:
            logging.warning("Failed to post to Teams")
    
    return func.HttpResponse(
        json.dumps(result, indent=2),
        status_code=200,
        mimetype="application/json"
    )


@app.route(route="health", methods=["GET"])
def health(req: func.HttpRequest) -> func.HttpResponse:
    """Health check endpoint."""
    return func.HttpResponse(
        json.dumps({
            "status": "ok",
            "service": "nba-picks-trigger",
            "api_url": _get_nba_api_url(),
            "teams_webhook_configured": bool(_get_teams_webhook_url()),
        }),
        status_code=200,
        mimetype="application/json"
    )


@app.route(route="", methods=["GET"])
def dashboard(req: func.HttpRequest) -> func.HttpResponse:
    """Dashboard - fetches picks and posts to Teams SERVER-SIDE (no JS delay)."""
    logging.info("Dashboard accessed - posting to Teams immediately")
    
    # Fetch predictions SERVER-SIDE
    data = fetch_predictions("today")
    
    if not data or not data.get("plays"):
        # Error page
        html = """<!DOCTYPE html>
<html><head><title>NBA Picks</title>
<style>body{font-family:'Segoe UI',sans-serif;background:#1a1a2e;color:#fff;padding:40px;text-align:center;}
.error{background:rgba(255,71,87,0.2);border:2px solid #ff4757;padding:30px;border-radius:10px;max-width:500px;margin:0 auto;}</style>
</head><body><div class="error"><h1>No Picks Available</h1><p>No predictions found for today. Check back later.</p></div></body></html>"""
        return func.HttpResponse(html, status_code=200, mimetype="text/html")
    
    plays = data.get("plays", [])
    
    # Post to Teams IMMEDIATELY (server-side)
    message = format_teams_card(data)
    posted = post_to_teams(message)
    logging.info(f"Posted {len(plays)} picks to Teams: {posted}")
    
    # Sort plays
    sorted_plays = sorted(plays, key=lambda p: (
        -p.get("fire_rating", "").count("\U0001F525"),
        -float(p.get("edge", "0").replace("+", "").replace(" pts", "").replace("%", "") or 0)
    ))
    
    # Count tiers
    elite = len([p for p in plays if get_fire_tier(p.get("fire_rating", "")) == "ELITE"])
    strong = len([p for p in plays if get_fire_tier(p.get("fire_rating", "")) == "STRONG"])
    good = len([p for p in plays if get_fire_tier(p.get("fire_rating", "")) == "GOOD"])
    
    # Build table rows
    rows_html = ""
    for p in sorted_plays:
        tier = get_fire_tier(p.get("fire_rating", ""))
        tier_class = tier
        rows_html += f"""<tr>
            <td>{p.get("matchup", "")}</td>
            <td>{p.get("period", "FG")}</td>
            <td>{p.get("market", "")}</td>
            <td class="pick-cell">{p.get("pick", "")}</td>
            <td class="odds-cell">{p.get("pick_odds", "N/A")}</td>
            <td>{p.get("edge", "N/A")}</td>
            <td class="tier-cell {tier_class}">{tier}</td>
        </tr>"""
    
    status_msg = f"POSTED {len(plays)} picks to Teams!" if posted else f"Found {len(plays)} picks (Teams post failed)"
    status_class = "success" if posted else "error"
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NBA Picks - Green Bier Capital</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #fff; min-height: 100vh; padding: 20px;
        }}
        .header {{ text-align: center; margin-bottom: 20px; }}
        .header h1 {{ font-size: 2rem; color: #00d4aa; margin-bottom: 5px; }}
        .header p {{ color: #888; font-size: 0.9rem; }}
        .status {{
            background: rgba(255,255,255,0.1); border-radius: 10px; padding: 20px;
            text-align: center; margin-bottom: 20px; max-width: 600px; margin-left: auto; margin-right: auto;
        }}
        .status.success {{ border: 2px solid #00d4aa; }}
        .status.error {{ border: 2px solid #ff4757; }}
        .status-text {{ font-size: 1.2rem; margin-bottom: 10px; }}
        .picks-summary {{ display: flex; gap: 15px; justify-content: center; margin-top: 15px; flex-wrap: wrap; }}
        .tier {{ padding: 8px 16px; border-radius: 8px; font-weight: bold; font-size: 1rem; }}
        .tier.elite {{ background: #ff4757; }}
        .tier.strong {{ background: #ffa502; color: #1a1a2e; }}
        .tier.good {{ background: #2ed573; color: #1a1a2e; }}
        .picks-table {{ width: 100%; max-width: 1200px; margin: 0 auto; border-collapse: collapse; }}
        .picks-table th {{
            background: rgba(0, 212, 170, 0.3); padding: 12px 8px; text-align: left;
            font-size: 0.8rem; text-transform: uppercase; border-bottom: 2px solid #00d4aa;
        }}
        .picks-table td {{ padding: 10px 8px; border-bottom: 1px solid rgba(255,255,255,0.1); font-size: 0.85rem; }}
        .picks-table tr:hover {{ background: rgba(255,255,255,0.05); }}
        .tier-cell {{ font-weight: bold; text-align: center; }}
        .tier-cell.ELITE {{ color: #ff4757; }}
        .tier-cell.STRONG {{ color: #ffa502; }}
        .tier-cell.GOOD {{ color: #2ed573; }}
        .pick-cell {{ font-weight: bold; color: #00d4aa; }}
        .odds-cell {{ font-weight: bold; }}
        .footer {{ text-align: center; margin-top: 30px; color: #666; font-size: 0.8rem; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>NBA Picks</h1>
        <p>Green Bier Capital | Powered by Azure</p>
    </div>
    
    <div class="status {status_class}">
        <div class="status-text">{status_msg}</div>
        <div class="picks-summary">
            <span class="tier elite">{elite} ELITE</span>
            <span class="tier strong">{strong} STRONG</span>
            <span class="tier good">{good} GOOD</span>
        </div>
    </div>
    
    <table class="picks-table">
        <thead>
            <tr>
                <th>Matchup</th>
                <th>Per</th>
                <th>Market</th>
                <th>Pick</th>
                <th>Odds</th>
                <th>Edge</th>
                <th>Tier</th>
            </tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>
    
    <div class="footer">NBA Prediction System v5.1 FINAL</div>
</body>
</html>"""
    return func.HttpResponse(html, status_code=200, mimetype="text/html")
