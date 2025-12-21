"""
Azure Function: NBA Picks Trigger - Green Bier Sport Ventures

Triggers the NBA prediction API and posts results to Teams.
Supports interactive commands via Teams messages or direct HTTP calls.

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
    """Format predictions as Teams Adaptive Card."""
    plays = data.get("plays", [])

    # Apply filters
    if filter_elite:
        plays = [p for p in plays if get_fire_tier(p.get("fire_rating", "")) == "ELITE"]
    if matchup_filter:
        matchup_filter = matchup_filter.upper()
        plays = [p for p in plays if matchup_filter in p.get("matchup", "").upper()]

    # Use API-provided timestamp or generate one
    generated_at = data.get("generated_at", datetime.now().strftime("%m/%d/%Y %I:%M %p CST"))

    # Sort by edge (highest first)
    sorted_plays = sorted(plays, key=lambda p: (
        -p.get("fire_rating", "").count("\U0001F525"),
        -float(p.get("edge", "0").replace("+", "").replace(" pts", "").replace("%", "") or 0)
    ))

    # Filter label
    filter_label = ""
    if filter_elite:
        filter_label = " (ELITE ONLY)"
    elif matchup_filter:
        filter_label = f" ({matchup_filter})"

    # Build Adaptive Card body
    body = [
        {"type": "TextBlock", "text": "GREEN BIER SPORT VENTURES", "weight": "Bolder", "size": "Large", "color": "Good"},
        {"type": "TextBlock", "text": f"NBA Picks - {generated_at}{filter_label}", "size": "Medium"},
        {"type": "TextBlock", "text": f"{len(plays)} Picks | Sorted by Edge", "size": "Small", "isSubtle": True},
        {
            "type": "ColumnSet",
            "columns": [
                {"type": "Column", "width": "auto", "items": [{"type": "TextBlock", "text": "PER", "weight": "Bolder", "size": "Small"}]},
                {"type": "Column", "width": "stretch", "items": [{"type": "TextBlock", "text": "PICK", "weight": "Bolder", "size": "Small"}]},
                {"type": "Column", "width": "auto", "items": [{"type": "TextBlock", "text": "ODDS", "weight": "Bolder", "size": "Small"}]},
                {"type": "Column", "width": "auto", "items": [{"type": "TextBlock", "text": "EDGE", "weight": "Bolder", "size": "Small"}]}
            ]
        }
    ]

    # Add each pick as a row
    for i, p in enumerate(sorted_plays):
        period = p.get("period", "FG")[:3]
        pick = p.get("pick", "")[:22]
        odds = p.get("pick_odds", "N/A")
        edge = p.get("edge", "N/A")

        # Color based on edge value
        try:
            edge_val = float(edge.replace("+", "").replace(" pts", "").replace("%", ""))
            edge_color = "Good" if edge_val > 0 else "Attention"
        except (ValueError, AttributeError):
            edge_color = "Default"

        row = {
            "type": "ColumnSet",
            "separator": i == 0,
            "columns": [
                {"type": "Column", "width": "auto", "items": [{"type": "TextBlock", "text": period, "size": "Small"}]},
                {"type": "Column", "width": "stretch", "items": [{"type": "TextBlock", "text": pick, "size": "Small", "weight": "Bolder"}]},
                {"type": "Column", "width": "auto", "items": [{"type": "TextBlock", "text": odds, "size": "Small"}]},
                {"type": "Column", "width": "auto", "items": [{"type": "TextBlock", "text": edge, "size": "Small", "color": edge_color}]}
            ]
        }
        body.append(row)

    card = {
        "type": "AdaptiveCard",
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "version": "1.2",
        "body": body
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
    
    <div class="footer">NBA Prediction System v6.0</div>
</body>
</html>"""
    return func.HttpResponse(html, status_code=200, mimetype="text/html")


# =============================================================================
# SCHEDULED TRIGGERS - Automatic daily posts to Teams
# =============================================================================

@app.timer_trigger(schedule="0 0 16 * * *", arg_name="timer", run_on_startup=False)
def daily_picks_10am(timer: func.TimerRequest) -> None:
    """Post today's picks to Teams at 10:00 AM CST daily.

    CRON: 0 0 16 * * * = 16:00 UTC = 10:00 AM CST
    """
    logging.info("Daily picks timer triggered (10 AM)")

    # Fetch predictions
    data = fetch_predictions("today")
    if not data or not data.get("plays"):
        logging.warning("No picks available for daily post")
        return

    # Post to Teams
    message = format_teams_card(data)
    posted = post_to_teams(message)

    plays_count = len(data.get("plays", []))
    if posted:
        logging.info(f"Daily picks posted to Teams: {plays_count} picks")
    else:
        logging.error("Failed to post daily picks to Teams")


@app.timer_trigger(schedule="0 0 0 * * *", arg_name="timer", run_on_startup=False)
def daily_picks_6pm(timer: func.TimerRequest) -> None:
    """Post today's picks to Teams at 6:00 PM CST daily (before evening games).

    CRON: 0 0 0 * * * = 00:00 UTC = 6:00 PM CST (previous day)
    """
    logging.info("Evening picks timer triggered (6 PM)")

    # Fetch predictions
    data = fetch_predictions("today")
    if not data or not data.get("plays"):
        logging.warning("No picks available for evening post")
        return

    # Post to Teams
    message = format_teams_card(data)
    posted = post_to_teams(message)

    plays_count = len(data.get("plays", []))
    if posted:
        logging.info(f"Evening picks posted to Teams: {plays_count} picks")
    else:
        logging.error("Failed to post evening picks to Teams")
