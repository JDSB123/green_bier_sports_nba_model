"""
Azure Function: NBA Picks Trigger - Green Bier Sport Ventures

v6.5 STRICT MODE: Always fetches FRESH data before predictions.

Triggers the NBA prediction API and posts results to Teams.
Supports interactive commands via Teams messages or direct HTTP calls.

IMPORTANT: This function calls /admin/cache/clear BEFORE fetching predictions
to ensure fresh data from ESPN and other sources.

Endpoints:
  GET  /api/nba-picks              - Get today's picks and post to Teams
  GET  /api/nba-picks?date=YYYY-MM-DD - Get picks for specific date
  GET  /api/nba-picks?matchup=LAL  - Filter by team
  GET  /api/nba-picks?elite=true   - Elite picks only (3+ fires)
  POST /api/nba-picks              - Process Teams command (from Power Automate)
  GET  /api/menu                   - Post interactive menu card to Teams
  GET  /api/health                 - Health check
"""
import azure.functions as func
import json
import logging
import os
import re
from datetime import datetime, timedelta
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


def _get_graph_access_token() -> str:
    """Get Microsoft Graph API access token using client credentials flow."""
    tenant_id = os.environ.get("GRAPH_TENANT_ID", "").strip()
    client_id = os.environ.get("GRAPH_CLIENT_ID", "").strip()
    client_secret = os.environ.get("GRAPH_CLIENT_SECRET", "").strip()

    if not all([tenant_id, client_id, client_secret]):
        logging.warning("Graph API credentials not configured")
        return None

    token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    data = f"client_id={client_id}&scope=https%3A%2F%2Fgraph.microsoft.com%2F.default&client_secret={client_secret}&grant_type=client_credentials"

    try:
        req = Request(token_url, data=data.encode(), headers={"Content-Type": "application/x-www-form-urlencoded"})
        with urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
            return result.get("access_token")
    except Exception as e:
        logging.error(f"Failed to get Graph token: {e}")
        return None


def generate_csv_content(plays: list) -> str:
    """Generate CSV content from plays list for download or SharePoint upload."""
    # Sort by edge
    sorted_plays = sorted(plays, key=lambda p: (
        -float(p.get("edge", "0").replace("+", "").replace(" pts", "").replace("%", "") or 0)
    ))

    csv_lines = []
    csv_lines.append("Time (CST),Matchup,Segment,Pick,Model,Market,Edge,Fire Rating")

    for p in sorted_plays:
        fire_rating = p.get("fire_rating", "")
        time_cst = p.get("time_cst", "").replace(",", "")
        matchup = p.get("matchup", "").replace(",", " vs ")
        segment = p.get("period", "FG")
        pick = p.get("pick", "").replace(",", "")
        model = p.get("model_prediction", "")
        market = p.get("market_line", "")
        edge = p.get("edge", "N/A")

        csv_lines.append(f"{time_cst},{matchup},{segment},{pick},{model},{market},{edge},{fire_rating}")

    return "\n".join(csv_lines)


def upload_to_sharepoint(filename: str, content: str) -> bool:
    """Upload file to SharePoint using Microsoft Graph API."""
    from urllib.parse import quote

    token = _get_graph_access_token()
    if not token:
        logging.error("No Graph token available for SharePoint upload")
        return False

    site_name = os.environ.get("SHAREPOINT_SITE_NAME", "").strip()
    folder_path = os.environ.get("SHAREPOINT_FOLDER_PATH", "").strip()

    if not site_name:
        logging.error("SHAREPOINT_SITE_NAME not configured")
        return False

    try:
        # Get site ID
        site_url = f"https://graph.microsoft.com/v1.0/sites/{site_name}"
        req = Request(site_url, headers={"Authorization": f"Bearer {token}"})
        with urlopen(req, timeout=30) as resp:
            site_data = json.loads(resp.read().decode())
            site_id = site_data.get("id")

        logging.info(f"Got SharePoint site ID: {site_id}")

        # Upload file directly using site-based path (more reliable than drive ID)
        # URL-encode the folder path and filename for spaces
        if folder_path:
            encoded_path = quote(folder_path, safe='')
            upload_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive/root:/{encoded_path}/{filename}:/content"
        else:
            upload_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive/root:/{filename}:/content"

        logging.info(f"Uploading to: {upload_url}")

        req = Request(
            upload_url,
            data=content.encode('utf-8'),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "text/csv"
            },
            method="PUT"
        )
        with urlopen(req, timeout=60) as resp:
            if resp.status in [200, 201]:
                logging.info(f"Successfully uploaded {filename} to SharePoint")
                return True

    except Exception as e:
        logging.error(f"SharePoint upload failed: {e}")

    return False


def get_fire_tier(rating: str) -> str:
    """Convert fire rating to tier name. Handles both emoji and string formats."""
    if not rating:
        return "NONE"

    # Handle string format from API (GOOD, STRONG, ELITE)
    rating_upper = rating.upper().strip()
    if rating_upper in ["ELITE", "STRONG", "GOOD"]:
        return rating_upper

    # Handle emoji format (legacy)
    fire_count = rating.count("\U0001F525")
    if fire_count >= 3:
        return "ELITE"
    elif fire_count == 2:
        return "STRONG"
    elif fire_count == 1:
        return "GOOD"
    return "NONE"


def clear_api_cache() -> bool:
    """
    Clear API cache to ensure fresh data - v6.5 STRICT MODE.

    MUST be called before fetching predictions to ensure:
    - Fresh ESPN standings (real-time team records)
    - Fresh game data
    - Fresh statistics
    """
    url = f"{_get_nba_api_url()}/admin/cache/clear"
    try:
        req = Request(url, method="POST", headers={"Accept": "application/json"})
        with urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
            logging.info(f"v6.5 STRICT MODE: Cache cleared - {result}")
            return True
    except Exception as e:
        logging.warning(f"Could not clear cache (will proceed anyway): {e}")
        return False


def fetch_predictions(date: str = "today") -> dict:
    """
    Fetch predictions from NBA API.

    v6.5 STRICT MODE: Always clears cache first to ensure fresh data.
    """
    # STRICT MODE: Clear cache first to force fresh data
    clear_api_cache()

    url = f"{_get_nba_api_url()}/slate/{date}/executive"
    try:
        logging.info(f"v6.5 STRICT MODE: Fetching fresh predictions for {date}")
        req = Request(url, headers={"Accept": "application/json"})
        with urlopen(req, timeout=90) as resp:  # Increased timeout for fresh data fetch
            return json.loads(resp.read().decode())
    except Exception as e:
        logging.error(f"Failed to fetch predictions: {e}")
        return None


def parse_command(text: str) -> dict:
    """Parse natural language commands from Teams messages.

    Examples:
      'picks' or 'today' -> today's picks
      'picks tomorrow' -> tomorrow's picks
      'picks 2024-12-25' -> specific date
      'picks lakers' or 'lakers picks' -> filter by team
      'elite' or 'elite picks' -> only 3+ fire picks
      'menu' -> show interactive menu
    """
    text = text.lower().strip()
    result = {"date": "today", "matchup": None, "elite_only": False, "show_menu": False}

    # Check for menu request
    if "menu" in text or "help" in text or "options" in text:
        result["show_menu"] = True
        return result

    # Check for elite only
    if "elite" in text or "best" in text or "top" in text:
        result["elite_only"] = True

    # Check for tomorrow
    if "tomorrow" in text:
        tomorrow = datetime.now() + timedelta(days=1)
        result["date"] = tomorrow.strftime("%Y-%m-%d")

    # Check for specific date (YYYY-MM-DD or MM/DD)
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', text)
    if date_match:
        result["date"] = date_match.group(1)
    else:
        date_match = re.search(r'(\d{1,2})/(\d{1,2})', text)
        if date_match:
            month, day = int(date_match.group(1)), int(date_match.group(2))
            year = datetime.now().year
            result["date"] = f"{year}-{month:02d}-{day:02d}"

    # Check for team names (common abbreviations and full names)
    teams = {
        "lakers": "LAL", "lal": "LAL", "celtics": "BOS", "bos": "BOS",
        "warriors": "GSW", "gsw": "GSW", "nets": "BKN", "bkn": "BKN",
        "knicks": "NYK", "nyk": "NYK", "heat": "MIA", "mia": "MIA",
        "bucks": "MIL", "mil": "MIL", "sixers": "PHI", "phi": "PHI",
        "suns": "PHX", "phx": "PHX", "mavs": "DAL", "dal": "DAL",
        "nuggets": "DEN", "den": "DEN", "clippers": "LAC", "lac": "LAC",
        "thunder": "OKC", "okc": "OKC", "cavs": "CLE", "cle": "CLE",
        "bulls": "CHI", "chi": "CHI", "hawks": "ATL", "atl": "ATL",
        "raptors": "TOR", "tor": "TOR", "magic": "ORL", "orl": "ORL",
        "pacers": "IND", "ind": "IND", "hornets": "CHA", "cha": "CHA",
        "wizards": "WAS", "was": "WAS", "pistons": "DET", "det": "DET",
        "rockets": "HOU", "hou": "HOU", "spurs": "SAS", "sas": "SAS",
        "kings": "SAC", "sac": "SAC", "blazers": "POR", "por": "POR",
        "jazz": "UTA", "uta": "UTA", "wolves": "MIN", "min": "MIN",
        "pelicans": "NOP", "nop": "NOP", "grizzlies": "MEM", "mem": "MEM",
    }
    for team_name, abbrev in teams.items():
        if team_name in text:
            result["matchup"] = abbrev
            break

    return result


def format_teams_card(data: dict, filter_elite: bool = False, matchup_filter: str = None) -> dict:
    """Format predictions as Teams Adaptive Card matching weekly-lineup dashboard.

    Columns: Time | Matchup (Away vs Home w/records) | Segment | Pick | Model | Market | Edge | Fire
    Limited to top 5 picks to stay within Teams webhook size limits.
    """
    plays = data.get("plays", [])
    total_plays = len(plays)

    # Apply filters
    if filter_elite:
        plays = [p for p in plays if get_fire_tier(p.get("fire_rating", "")) == "ELITE"]
    if matchup_filter:
        matchup_filter = matchup_filter.upper()
        plays = [p for p in plays if matchup_filter in p.get("matchup", "").upper()]

    # Use API-provided timestamp or generate one
    generated_at = data.get("generated_at", datetime.now().strftime("%m/%d/%Y %I:%M %p CST"))

    # Sort by edge (highest first) and limit to top 5 for Teams size limits
    sorted_plays = sorted(plays, key=lambda p: (
        -p.get("fire_rating", "").count("\U0001F525"),
        -float(p.get("edge", "0").replace("+", "").replace(" pts", "").replace("%", "") or 0)
    ))[:5]

    # Filter label
    filter_label = ""
    if filter_elite:
        filter_label = " (ELITE ONLY)"
    elif matchup_filter:
        filter_label = f" ({matchup_filter})"

    # Build Adaptive Card body - matching weekly-lineup dashboard format
    body = [
        {"type": "TextBlock", "text": "GREEN BIER SPORT VENTURES", "weight": "Bolder", "size": "Large", "color": "Good"},
        {"type": "TextBlock", "text": f"Today's Picks - {generated_at}{filter_label}", "size": "Medium"},
        {"type": "TextBlock", "text": f"Top {len(sorted_plays)} of {total_plays} picks | Sorted by Edge", "size": "Small", "isSubtle": True, "spacing": "Small"}
    ]

    # Table header
    body.append({
        "type": "ColumnSet",
        "spacing": "Medium",
        "columns": [
            {"type": "Column", "width": "auto", "items": [{"type": "TextBlock", "text": "Time", "weight": "Bolder", "size": "Small"}]},
            {"type": "Column", "width": "stretch", "items": [{"type": "TextBlock", "text": "Matchup", "weight": "Bolder", "size": "Small"}]},
            {"type": "Column", "width": "auto", "items": [{"type": "TextBlock", "text": "Seg", "weight": "Bolder", "size": "Small"}]},
            {"type": "Column", "width": "auto", "items": [{"type": "TextBlock", "text": "Pick", "weight": "Bolder", "size": "Small"}]},
            {"type": "Column", "width": "auto", "items": [{"type": "TextBlock", "text": "Model", "weight": "Bolder", "size": "Small"}]},
            {"type": "Column", "width": "auto", "items": [{"type": "TextBlock", "text": "Mkt", "weight": "Bolder", "size": "Small"}]},
            {"type": "Column", "width": "auto", "items": [{"type": "TextBlock", "text": "Edge", "weight": "Bolder", "size": "Small"}]},
            {"type": "Column", "width": "auto", "items": [{"type": "TextBlock", "text": "Fire", "weight": "Bolder", "size": "Small"}]}
        ]
    })

    # Add each pick as a table row
    for p in sorted_plays:
        matchup = p.get("matchup", "")  # e.g. "Chicago Bulls (15-17) @ Atlanta Hawks (17-16)"
        time_cst = p.get("time_cst", "")  # e.g. "12/21 02:40 PM"
        segment = p.get("period", "FG")
        pick = p.get("pick", "")
        model_pred = p.get("model_prediction", "")  # e.g. "233.3" or "+1.0 pts"
        market_line = p.get("market_line", "")  # e.g. "248.0" or "-2.5"
        edge = p.get("edge", "N/A")
        fire_rating = p.get("fire_rating", "")

        # Convert fire rating text to emoji display
        fire_map = {"ELITE": "\U0001F525\U0001F525\U0001F525", "STRONG": "\U0001F525\U0001F525", "GOOD": "\U0001F525"}
        fire_display = fire_map.get(fire_rating, "-")

        # Color based on edge value
        try:
            edge_val = float(edge.replace("+", "").replace(" pts", "").replace("%", ""))
            edge_color = "Good" if edge_val > 0 else "Attention"
        except (ValueError, AttributeError):
            edge_color = "Default"

        # Table row
        body.append({
            "type": "ColumnSet",
            "spacing": "Small",
            "columns": [
                {"type": "Column", "width": "auto", "items": [{"type": "TextBlock", "text": time_cst, "size": "Small"}]},
                {"type": "Column", "width": "stretch", "items": [{"type": "TextBlock", "text": matchup, "size": "Small", "wrap": True}]},
                {"type": "Column", "width": "auto", "items": [{"type": "TextBlock", "text": segment, "size": "Small"}]},
                {"type": "Column", "width": "auto", "items": [{"type": "TextBlock", "text": pick, "size": "Small", "wrap": True}]},
                {"type": "Column", "width": "auto", "items": [{"type": "TextBlock", "text": str(model_pred), "size": "Small"}]},
                {"type": "Column", "width": "auto", "items": [{"type": "TextBlock", "text": str(market_line), "size": "Small"}]},
                {"type": "Column", "width": "auto", "items": [{"type": "TextBlock", "text": edge, "size": "Small", "color": edge_color}]},
                {"type": "Column", "width": "auto", "items": [{"type": "TextBlock", "text": fire_display, "size": "Small"}]}
            ]
        })

    card = {
        "type": "AdaptiveCard",
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "version": "1.2",
        "body": body,
        "actions": [
            {"type": "Action.OpenUrl", "title": "View All Picks", "url": "https://nba-picks-trigger.azurewebsites.net/api/dashboard"}
        ]
    }

    return {
        "type": "message",
        "attachments": [{"contentType": "application/vnd.microsoft.card.adaptive", "content": card}]
    }


def format_menu_card() -> dict:
    """Create a menu card for Teams."""
    card = {
        "type": "AdaptiveCard",
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "version": "1.2",
        "body": [
            {"type": "TextBlock", "text": "GREEN BIER SPORT VENTURES", "weight": "Bolder", "size": "Large", "color": "Good"},
            {"type": "TextBlock", "text": "NBA Prediction System v6.4", "size": "Medium"},
            {"type": "TextBlock", "text": "Available Commands:", "weight": "Bolder", "size": "Small", "spacing": "Medium"},
            {"type": "FactSet", "facts": [
                {"title": "picks", "value": "Get today's picks"},
                {"title": "picks tomorrow", "value": "Get tomorrow's picks"},
                {"title": "picks 12/25", "value": "Get picks for specific date"},
                {"title": "picks lakers", "value": "Filter by team"},
                {"title": "elite", "value": "Show only elite picks"}
            ]},
            {"type": "TextBlock", "text": "Powered by Azure Functions", "size": "Small", "isSubtle": True, "spacing": "Medium"}
        ]
    }
    return {
        "type": "message",
        "attachments": [{"contentType": "application/vnd.microsoft.card.adaptive", "content": card}]
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


@app.route(route="nba-picks", methods=["GET", "POST"])
def nba_picks(req: func.HttpRequest) -> func.HttpResponse:
    """
    Main trigger endpoint - fetches predictions and posts to Teams.

    GET Query params:
      - date: YYYY-MM-DD, 'today', or 'tomorrow' (default: today)
      - matchup: Team abbreviation to filter (e.g., LAL, BOS)
      - elite: 'true' to show only elite picks
      - post: 'true' to post to Teams, 'false' for JSON only (default: true)

    POST Body (from Power Automate):
      - text: Natural language command (e.g., "picks lakers", "elite", "picks tomorrow")
    """
    logging.info("NBA Picks trigger invoked")

    # Parse parameters - from query string or POST body
    date = "today"
    matchup_filter = None
    elite_only = False
    should_post = True
    show_menu = False

    if req.method == "POST":
        # Parse command from Teams/Power Automate
        try:
            body = req.get_json()
            command_text = body.get("text", "") or body.get("message", "") or ""
            logging.info(f"Received command: {command_text}")
            parsed = parse_command(command_text)
            date = parsed["date"]
            matchup_filter = parsed["matchup"]
            elite_only = parsed["elite_only"]
            show_menu = parsed["show_menu"]
        except Exception as e:
            logging.warning(f"Failed to parse POST body: {e}")
    else:
        # GET request - use query params
        date = req.params.get("date", "today")
        if date == "tomorrow":
            tomorrow = datetime.now() + timedelta(days=1)
            date = tomorrow.strftime("%Y-%m-%d")
        matchup_filter = req.params.get("matchup")
        elite_only = req.params.get("elite", "false").lower() == "true"
        should_post = req.params.get("post", "true").lower() == "true"
        show_menu = req.params.get("menu", "false").lower() == "true"

    # Show menu if requested
    if show_menu:
        message = format_menu_card()
        posted = post_to_teams(message)
        return func.HttpResponse(
            json.dumps({"action": "menu", "posted_to_teams": posted}),
            status_code=200,
            mimetype="application/json"
        )

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

    # Apply filters for response
    filtered_plays = plays
    if elite_only:
        filtered_plays = [p for p in plays if get_fire_tier(p.get("fire_rating", "")) == "ELITE"]
    if matchup_filter:
        mf = matchup_filter.upper()
        filtered_plays = [p for p in filtered_plays if mf in p.get("matchup", "").upper()]

    # Sort by fire rating
    sorted_plays = sorted(filtered_plays, key=lambda p: (
        -p.get("fire_rating", "").count("\U0001F525"),
        -float(p.get("edge", "0").replace("+", "").replace(" pts", "").replace("%", "") or 0)
    ))

    # Count tiers
    elite_count = len([p for p in filtered_plays if get_fire_tier(p.get("fire_rating", "")) == "ELITE"])
    strong_count = len([p for p in filtered_plays if get_fire_tier(p.get("fire_rating", "")) == "STRONG"])
    good_count = len([p for p in filtered_plays if get_fire_tier(p.get("fire_rating", "")) == "GOOD"])

    # Build simplified picks list for response
    picks_list = [
        {
            "matchup": p.get("matchup", ""),
            "period": p.get("period", "FG"),
            "market": p.get("market", ""),
            "pick": p.get("pick", ""),
            "odds": p.get("pick_odds", "N/A"),
            "edge": p.get("edge", "N/A"),
            "tier": get_fire_tier(p.get("fire_rating", ""))
        }
        for p in sorted_plays
    ]

    result = {
        "date": date,
        "filters": {"elite_only": elite_only, "matchup": matchup_filter},
        "total_picks": len(filtered_plays),
        "elite": elite_count,
        "strong": strong_count,
        "good": good_count,
        "posted_to_teams": False,
        "picks": picks_list
    }

    # Post to Teams if requested
    if should_post:
        message = format_teams_card(data, filter_elite=elite_only, matchup_filter=matchup_filter)
        posted = post_to_teams(message)
        result["posted_to_teams"] = posted
        if posted:
            logging.info(f"Posted {len(filtered_plays)} picks to Teams")
        else:
            logging.warning("Failed to post to Teams")

        # Also upload CSV to SharePoint
        csv_content = generate_csv_content(plays)
        date_str = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"NBA_Picks_{date_str}.csv"
        uploaded = upload_to_sharepoint(filename, csv_content)
        result["uploaded_to_sharepoint"] = uploaded
        if uploaded:
            logging.info(f"Uploaded {filename} to SharePoint")

    return func.HttpResponse(
        json.dumps(result, indent=2),
        status_code=200,
        mimetype="application/json"
    )


@app.route(route="menu", methods=["GET"])
def menu(req: func.HttpRequest) -> func.HttpResponse:
    """Post the interactive menu card to Teams."""
    logging.info("Menu requested")
    message = format_menu_card()
    posted = post_to_teams(message)
    return func.HttpResponse(
        json.dumps({"action": "menu", "posted_to_teams": posted}),
        status_code=200,
        mimetype="application/json"
    )


@app.route(route="health", methods=["GET"])
def health(req: func.HttpRequest) -> func.HttpResponse:
    """Health check endpoint - v6.5 STRICT MODE."""
    return func.HttpResponse(
        json.dumps({
            "status": "ok",
            "version": "6.5",
            "mode": "STRICT",
            "service": "nba-picks-trigger",
            "api_url": _get_nba_api_url(),
            "teams_webhook_configured": bool(_get_teams_webhook_url()),
            "fresh_data": "Always clears cache before predictions",
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
    
    # Build table rows - matching weekly-lineup dashboard format
    rows_html = ""
    for p in sorted_plays:
        fire_rating = p.get("fire_rating", "")
        fire_map = {"ELITE": "\U0001F525\U0001F525\U0001F525", "STRONG": "\U0001F525\U0001F525", "GOOD": "\U0001F525"}
        fire_display = fire_map.get(fire_rating, "-")

        # Edge coloring
        edge = p.get("edge", "N/A")
        try:
            edge_val = float(edge.replace("+", "").replace(" pts", "").replace("%", ""))
            edge_class = "positive" if edge_val > 0 else "negative"
        except (ValueError, AttributeError):
            edge_class = ""

        rows_html += f"""<tr>
            <td>{p.get("time_cst", "")}</td>
            <td>{p.get("matchup", "")}</td>
            <td class="center">{p.get("period", "FG")}</td>
            <td class="pick-cell">{p.get("pick", "")}</td>
            <td class="center">{p.get("model_prediction", "")}</td>
            <td class="center">{p.get("market_line", "")}</td>
            <td class="edge-cell {edge_class}">{edge}</td>
            <td class="fire-cell">{fire_display}</td>
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
        .center {{ text-align: center; }}
        .pick-cell {{ font-weight: bold; color: #00d4aa; }}
        .edge-cell {{ font-weight: bold; text-align: center; }}
        .edge-cell.positive {{ color: #2ed573; }}
        .edge-cell.negative {{ color: #ff4757; }}
        .fire-cell {{ text-align: center; font-size: 1rem; }}
        .actions {{ text-align: center; margin: 30px 0; }}
        .download-btn {{
            display: inline-block; padding: 12px 24px; background: #00d4aa; color: #1a1a2e;
            text-decoration: none; border-radius: 8px; font-weight: bold; font-size: 1rem;
            transition: background 0.2s;
        }}
        .download-btn:hover {{ background: #00b894; }}
        .footer {{ text-align: center; margin-top: 30px; color: #666; font-size: 0.8rem; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>GREEN BIER SPORT VENTURES</h1>
        <p>NBA Picks | {data.get("generated_at", "")}</p>
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
                <th>Time (CST)</th>
                <th>Matchup (Away vs Home)</th>
                <th>Seg</th>
                <th>Pick</th>
                <th>Model</th>
                <th>Market</th>
                <th>Edge</th>
                <th>Fire</th>
            </tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>
    
    <div class="actions">
        <a href="/api/csv" class="download-btn">Download Excel (CSV)</a>
    </div>

    <div class="footer">NBA Prediction System v6.0</div>
</body>
</html>"""
    return func.HttpResponse(html, status_code=200, mimetype="text/html")


def get_bot_token() -> str:
    """Get access token for Bot Framework Connector API."""
    app_id = os.environ.get("MICROSOFT_APP_ID", "").strip()
    app_password = os.environ.get("MICROSOFT_APP_PASSWORD", "").strip()
    tenant_id = os.environ.get("MICROSOFT_APP_TENANT_ID", "").strip()

    if not all([app_id, app_password]):
        logging.error("Bot credentials not configured")
        return None

    # Use tenant-specific endpoint for SingleTenant bot
    token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    data = f"grant_type=client_credentials&client_id={app_id}&client_secret={app_password}&scope=https%3A%2F%2Fapi.botframework.com%2F.default"

    try:
        req = Request(token_url, data=data.encode(), headers={"Content-Type": "application/x-www-form-urlencoded"})
        with urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
            return result.get("access_token")
    except Exception as e:
        logging.error(f"Failed to get bot token: {e}")
        return None


def send_bot_reply(service_url: str, conversation_id: str, activity_id: str, reply_activity: dict) -> bool:
    """Send a reply through the Bot Connector API."""
    token = get_bot_token()
    if not token:
        logging.error("No bot token available")
        return False

    # Ensure service URL ends without trailing slash
    service_url = service_url.rstrip('/')
    reply_url = f"{service_url}/v3/conversations/{conversation_id}/activities/{activity_id}"

    try:
        data = json.dumps(reply_activity).encode('utf-8')
        req = Request(
            reply_url,
            data=data,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            },
            method="POST"
        )
        with urlopen(req, timeout=60) as resp:
            logging.info(f"Bot reply sent successfully: {resp.status}")
            return True
    except Exception as e:
        logging.error(f"Failed to send bot reply: {e}")
        return False


@app.route(route="bot", methods=["POST"])
def teams_bot(req: func.HttpRequest) -> func.HttpResponse:
    """
    Teams bot endpoint - supports BOTH Outgoing Webhook AND Azure Bot Framework.

    Outgoing Webhook: Returns response directly as HTTP response
    Azure Bot: Sends response via Bot Connector API

    Commands:
      @NBA Picks run slate         → Get all today's picks
      @NBA Picks elite             → Elite picks only
      @NBA Picks lakers            → Filter by team
      @NBA Picks help              → Show commands
    """
    logging.info("Teams bot command received")

    try:
        body = req.get_json()
        logging.info(f"Bot request body: {json.dumps(body)[:500]}")
    except Exception as e:
        logging.error(f"Failed to parse bot request: {e}")
        return func.HttpResponse(status_code=200)

    # Extract Bot Framework activity details
    service_url = body.get("serviceUrl", "")
    conversation_id = body.get("conversation", {}).get("id", "")
    activity_id = body.get("id", "")
    raw_text = body.get("text", "")

    # Detect mode: Outgoing Webhook vs Azure Bot
    # Outgoing Webhook has no serviceUrl or has simple format
    is_outgoing_webhook = not service_url or "api.botframework.com" not in service_url
    logging.info(f"Mode: {'Outgoing Webhook' if is_outgoing_webhook else 'Azure Bot'}, Service URL: {service_url}")

    # Remove the @mention tag from the text
    # Teams sends: "<at>Bot Name</at> actual command"
    command_text = re.sub(r'<at>.*?</at>\s*', '', raw_text).strip()
    logging.info(f"Parsed command: {command_text}")

    # Parse the command
    parsed = parse_command(command_text)

    # Helper to send response (works for both modes)
    def send_response(reply: dict) -> func.HttpResponse:
        if is_outgoing_webhook:
            # Outgoing Webhook: Return directly as HTTP response
            return func.HttpResponse(json.dumps(reply), status_code=200, mimetype="application/json")
        else:
            # Azure Bot: Send via Bot Connector API
            send_bot_reply(service_url, conversation_id, activity_id, reply)
            return func.HttpResponse(status_code=200)

    # Handle menu/help
    if parsed["show_menu"] or not command_text or command_text.lower() in ["", "hi", "hello"]:
        help_reply = {
            "type": "message",
            "attachments": [{
                "contentType": "application/vnd.microsoft.card.adaptive",
                "content": {
                    "type": "AdaptiveCard",
                    "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                    "version": "1.2",
                    "body": [
                        {"type": "TextBlock", "text": "GREEN BIER NBA PICKS", "weight": "Bolder", "size": "Large", "color": "Good"},
                        {"type": "TextBlock", "text": "Available Commands:", "weight": "Bolder", "spacing": "Medium"},
                        {"type": "FactSet", "facts": [
                            {"title": "run slate", "value": "Get all today's picks"},
                            {"title": "elite", "value": "Elite picks only (3+ fires)"},
                            {"title": "lakers", "value": "Filter by team name"},
                            {"title": "LAL vs BOS", "value": "Specific matchup"},
                            {"title": "tomorrow", "value": "Tomorrow's picks"},
                            {"title": "12/25", "value": "Picks for specific date"}
                        ]},
                        {"type": "TextBlock", "text": "v6.5 STRICT MODE - Fresh data always", "size": "Small", "isSubtle": True, "spacing": "Medium"}
                    ]
                }
            }]
        }
        return send_response(help_reply)

    # Fetch predictions
    date = parsed["date"]
    matchup_filter = parsed["matchup"]
    elite_only = parsed["elite_only"]

    logging.info(f"Fetching picks: date={date}, matchup={matchup_filter}, elite={elite_only}")
    data = fetch_predictions(date)

    if not data or not data.get("plays"):
        no_picks_reply = {"type": "message", "text": f"No picks available for {date}. Check back closer to game time."}
        return send_response(no_picks_reply)

    plays = data.get("plays", [])

    # Apply filters
    if elite_only:
        plays = [p for p in plays if get_fire_tier(p.get("fire_rating", "")) == "ELITE"]
    if matchup_filter:
        mf = matchup_filter.upper()
        plays = [p for p in plays if mf in p.get("matchup", "").upper()]

    if not plays:
        filter_msg = f"elite " if elite_only else ""
        filter_msg += f"for {matchup_filter}" if matchup_filter else ""
        no_filter_reply = {"type": "message", "text": f"No {filter_msg}picks found. Try different filters."}
        return send_response(no_filter_reply)

    # Sort by edge (no limit - show all picks)
    sorted_plays = sorted(plays, key=lambda p: (
        -p.get("fire_rating", "").count("\U0001F525"),
        -float(p.get("edge", "0").replace("+", "").replace(" pts", "").replace("%", "") or 0)
    ))

    # Build response card body
    generated_at = data.get("generated_at", datetime.now().strftime("%m/%d/%Y %I:%M %p CST"))
    filter_label = ""
    if elite_only:
        filter_label = " (ELITE)"
    elif matchup_filter:
        filter_label = f" ({matchup_filter.upper()})"

    # Build plain text table for cleaner output
    lines = []
    lines.append(f"**GREEN BIER NBA PICKS**")
    lines.append(f"{generated_at}{filter_label} | {len(sorted_plays)} picks")
    lines.append("")
    lines.append("```")
    lines.append(f"{'Time':<12} | {'Matchup':<50} | {'Seg':<3} | {'Pick':<28} | {'Model':<24} | {'Mkt':<8} | {'Edge':<9} | Fire")
    lines.append("-" * 155)

    for p in sorted_plays:
        fire_rating = p.get("fire_rating", "")
        fire_count = 0
        if isinstance(fire_rating, str):
            fire_count = fire_rating.count("\U0001F525")
            if fire_count == 0 and fire_rating.upper() in ["ELITE", "STRONG", "GOOD"]:
                fire_count = {"ELITE": 3, "STRONG": 2, "GOOD": 1}.get(fire_rating.upper(), 0)
        fire_display = "\U0001F525" * fire_count if fire_count else "-"

        time_cst = p.get("time_cst", "")
        matchup = p.get("matchup", "")
        segment = p.get("period", "FG")
        pick = p.get("pick", "")
        pick_odds = p.get("pick_odds", p.get("odds", ""))
        pick_with_odds = f"{pick} ({pick_odds})" if pick_odds else pick
        model_pred = str(p.get("model_prediction", ""))
        market_line = str(p.get("market_line", ""))
        edge = p.get("edge", "")

        lines.append(f"{time_cst:<12} | {matchup:<50} | {segment:<3} | {pick_with_odds:<28} | {model_pred:<24} | {market_line:<8} | {edge:<9} | {fire_display}")

    lines.append("```")

    text_response = "\n".join(lines)

    response_msg = {
        "type": "message",
        "text": text_response
    }

    logging.info(f"Sending {len(sorted_plays)} picks to Teams")
    return send_response(response_msg)


@app.route(route="csv", methods=["GET"])
def csv_download(req: func.HttpRequest) -> func.HttpResponse:
    """Generate CSV file for Excel download."""
    logging.info("CSV download requested")

    data = fetch_predictions("today")

    if not data or not data.get("plays"):
        return func.HttpResponse("No picks available", status_code=404)

    plays = data.get("plays", [])

    # Use shared CSV generation function
    csv_content = generate_csv_content(plays)

    # Generate filename with date
    date_str = datetime.now().strftime("%Y%m%d")
    filename = f"NBA_Picks_{date_str}.csv"

    return func.HttpResponse(
        csv_content,
        status_code=200,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


