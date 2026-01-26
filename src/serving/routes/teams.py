"""Teams integration routes (Power Automate & Outgoing Webhooks).

Routes:
    GET /teams/workflow       - Power Automate workflow endpoint for Teams channels
    GET /teams/outgoing       - Teams validation & connectivity check
    POST /teams/outgoing      - Teams outgoing webhook handler

These endpoints integrate with Microsoft Teams via:
- Power Automate flows (workflow endpoint)
- Outgoing webhook configuration (outgoing endpoint)
"""

import base64
import hashlib
import hmac
import logging
import os
import re
import time
from typing import Any, Dict, Optional
from urllib.parse import urlencode

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response

from src.ingestion.standardize import normalize_team_to_espn
from src.serving.dependencies import limiter

logger = logging.getLogger(__name__)

# Create router
teams_router = APIRouter(prefix="/teams", tags=["Teams"])

# --- Teams Webhook Picks Cache (5-minute TTL for fast responses) ---
_teams_picks_cache: Dict[str, Any] = {}
_teams_picks_cache_time: Dict[str, float] = {}
_TEAMS_CACHE_TTL_SECONDS = 300  # 5 minutes


def _get_cached_picks(date_key: str) -> Optional[Dict[str, Any]]:
    """Get cached picks if still valid (within TTL)."""
    if date_key in _teams_picks_cache:
        cached_time = _teams_picks_cache_time.get(date_key, 0)
        if time.time() - cached_time < _TEAMS_CACHE_TTL_SECONDS:
            return _teams_picks_cache[date_key]
    return None


def _set_cached_picks(date_key: str, data: Dict[str, Any]) -> None:
    """Cache picks data with timestamp."""
    _teams_picks_cache[date_key] = data
    _teams_picks_cache_time[date_key] = time.time()


async def _validate_teams_outgoing_webhook(request: Request) -> None:
    """Validate Teams outgoing webhook signature if configured."""
    webhook_secret = os.environ.get("TEAMS_WEBHOOK_SECRET", "").strip()
    if not webhook_secret:
        logger.warning("TEAMS_WEBHOOK_SECRET not set - skipping Teams outgoing webhook validation")
        return

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("HMAC "):
        logger.error("Missing HMAC Authorization header for Teams outgoing webhook")
        raise HTTPException(status_code=401, detail="Unauthorized - missing HMAC signature")

    try:
        secret_bytes = base64.b64decode(webhook_secret)
    except Exception:
        logger.error("Invalid TEAMS_WEBHOOK_SECRET - expected base64")
        raise HTTPException(status_code=500, detail="Server configuration error")

    body = await request.body()
    computed_hash = hmac.new(secret_bytes, body, hashlib.sha256)
    computed_signature = base64.b64encode(computed_hash.digest()).decode("utf-8")
    provided_signature = auth_header[5:]
    if not hmac.compare_digest(provided_signature, computed_signature):
        logger.error("HMAC validation failed for Teams outgoing webhook")
        raise HTTPException(status_code=401, detail="Unauthorized - invalid signature")


def _parse_teams_command(text: str) -> dict:
    """Parse Teams outgoing webhook commands into filters."""
    normalized = text.lower().strip()
    result = {"date": "today", "team_filter": None, "elite_only": False, "show_menu": False}

    if not normalized or "help" in normalized or "menu" in normalized or "options" in normalized:
        result["show_menu"] = True
        return result

    # Normalize team name using the canonicalization logic
    team_name, _ = normalize_team_to_espn(normalized, source="teams_webhook")
    result["team_filter"] = team_name

    if "elite" in normalized or "best" in normalized or "top" in normalized:
        result["elite_only"] = True

    if "tomorrow" in normalized:
        result["date"] = "tomorrow"

    return result


# Type alias for executive summary callable
ExecutiveSummaryFunc = None  # Will be set after app initialization


def set_executive_summary_func(func):
    """Set the executive summary function reference for Teams routes.

    This is called from app.py after route initialization to provide
    access to get_executive_summary without circular imports.
    """
    global _get_executive_summary
    _get_executive_summary = func


# Placeholder that will be replaced by set_executive_summary_func
_get_executive_summary = None


@teams_router.get("/workflow")
@teams_router.get("/workflow/")
@limiter.limit("30/minute")
async def teams_workflow_get(
    request: Request,
    date: str = Query("today", description="Date: today, tomorrow, or YYYY-MM-DD"),
    elite: bool = Query(False, description="Only show elite picks (4+ fire rating)"),
    team: Optional[str] = Query(None, description="Filter by team name"),
):
    """
    Power Automate Workflow endpoint - returns formatted picks for Teams.

    Use this with Power Automate to post picks to Teams channels.
    Returns Adaptive Card JSON for rich formatting.

    Example Power Automate flow:
    1. Trigger: Recurrence (daily at 9 AM) or Manual
    2. HTTP action: GET https://nba.greenbiersportventures.com/teams/workflow?date=today
    3. Post to Teams channel using the response
    """
    if _get_executive_summary is None:
        logger.error("Executive summary function not initialized for Teams routes")
        return JSONResponse(
            status_code=500, content={"type": "message", "text": "Server not fully initialized"}
        )

    # Use cache for fast response
    cache_key = f"{date}_{elite}_{team or 'all'}"
    cached_data = _get_cached_picks(date)

    if cached_data is not None:
        data = cached_data
    else:
        try:
            data = await _get_executive_summary(request, date=date, use_splits=True)
            if isinstance(data, dict):
                _set_cached_picks(date, data)
        except HTTPException as e:
            return JSONResponse(
                status_code=200, content={"type": "message", "text": f"❌ Error: {e.detail}"}
            )
        except Exception as e:
            logger.error("Teams workflow request failed: %s", e)
            return JSONResponse(
                status_code=200, content={"type": "message", "text": "❌ Failed to fetch picks."}
            )

    if not isinstance(data, dict):
        return JSONResponse(
            status_code=200, content={"type": "message", "text": "No picks available."}
        )

    plays = data.get("plays", [])
    date_label = data.get("date", date)

    # Apply filters
    if team:
        team_lower = team.lower()
        plays = [p for p in plays if team_lower in p.get("matchup", "").lower()]

    if elite:
        plays = [p for p in plays if int(p.get("fire_rating", 0) or 0) >= 4]

    if not plays:
        filter_desc = []
        if elite:
            filter_desc.append("elite")
        if team:
            filter_desc.append(f"for {team}")
        filter_str = f" ({', '.join(filter_desc)})" if filter_desc else ""
        return JSONResponse(
            status_code=200,
            content={"type": "message", "text": f"📭 No{filter_str} picks for {date_label}."},
        )

    # Build formatted message for Teams
    fire_emoji = {5: "🔥🔥🔥🔥🔥", 4: "🔥🔥🔥🔥", 3: "🔥🔥🔥", 2: "🔥🔥", 1: "🔥"}

    lines = [f"**🏀 NBA Picks for {date_label}**", ""]

    for p in plays[:10]:  # Limit to 10 picks for readability
        matchup = p.get("matchup", "Unknown")
        market = p.get("market", "")
        pick = p.get("pick", "")
        try:
            edge = float(p.get("edge", 0) or 0)
        except (TypeError, ValueError):
            edge = 0.0
        rating = int(p.get("fire_rating", 0) or 0)
        fires = fire_emoji.get(rating, "")

        lines.append(f"**{matchup}**")
        lines.append(f"  {market}: **{pick}** ({edge:+.1f}% edge) {fires}")
        lines.append("")

    if len(plays) > 10:
        lines.append(f"_...and {len(plays) - 10} more picks_")

    lines.append("")
    lines.append(
        f"📊 **Total: {len(plays)} picks** | "
        f"[View Full Lineup](https://nba.greenbiersportventures.com/weekly-lineup/html?date={date_label})"
    )

    return JSONResponse(status_code=200, content={"type": "message", "text": "\n".join(lines)})


@teams_router.api_route(
    "/outgoing",
    methods=["GET", "HEAD", "OPTIONS"],
)
@teams_router.api_route(
    "/outgoing/",
    methods=["GET", "HEAD", "OPTIONS"],
)
async def teams_outgoing_get(request: Request, validationToken: Optional[str] = Query(None)):
    """Handle Teams validation pings (echo validationToken), reachability, and preflight/HEAD."""
    # Teams sends a GET with ?validationToken=... during setup; must echo it as plain text.
    if validationToken:
        return Response(content=validationToken, media_type="text/plain")

    # OPTIONS/HEAD should succeed with 200 to satisfy preflight or connectivity checks
    if request.method in {"OPTIONS", "HEAD"}:
        return Response(status_code=200)

    return JSONResponse(
        status_code=200, content={"text": "Teams webhook is reachable. Use POST for commands."}
    )


@teams_router.post("/outgoing")
@teams_router.post("/outgoing/")
@limiter.limit("30/minute")
async def teams_outgoing_webhook(request: Request):
    """Teams outgoing webhook handler (ACA-hosted).

    Uses in-memory cache (5-min TTL) for fast responses.
    Teams requires response within 5 seconds.
    """
    if _get_executive_summary is None:
        logger.error("Executive summary function not initialized for Teams routes")
        return JSONResponse(status_code=200, content={"text": "Server not fully initialized."})

    await _validate_teams_outgoing_webhook(request)

    try:
        body = await request.json()
    except Exception as e:
        logger.error("Failed to parse Teams outgoing webhook body: %s", e)
        return JSONResponse(status_code=200, content={"text": "Invalid request body."})

    raw_text = body.get("text", "") if isinstance(body, dict) else ""
    command_text = re.sub(r"<at>.*?</at>\s*", "", raw_text).strip()
    parsed = _parse_teams_command(command_text)

    if parsed["show_menu"]:
        help_text = (
            "Commands: picks, picks tomorrow, picks YYYY-MM-DD, " "picks lakers, elite, help"
        )
        return JSONResponse(status_code=200, content={"text": help_text})

    # Try cache first for fast response (Teams has 5-second timeout)
    cache_key = parsed["date"]
    cached_data = _get_cached_picks(cache_key)

    if cached_data is not None:
        logger.info(f"Teams webhook: serving cached picks for {cache_key}")
        data = cached_data
    else:
        # Fetch fresh data and cache it
        try:
            data = await _get_executive_summary(request, date=parsed["date"], use_splits=True)
            if isinstance(data, dict):
                _set_cached_picks(cache_key, data)
        except HTTPException as e:
            logger.error("Teams outgoing webhook request failed: %s", e.detail)
            return JSONResponse(status_code=200, content={"text": f"Error: {e.detail}"})
        except Exception as e:
            logger.error("Teams outgoing webhook request failed: %s", e)
            return JSONResponse(status_code=200, content={"text": "Failed to fetch picks."})

    if not isinstance(data, dict):
        return JSONResponse(status_code=200, content={"text": "No picks available."})

    plays = data.get("plays", [])
    total_plays = data.get("total_plays", len(plays))
    if not plays:
        date_label = data.get("date", parsed["date"])
        return JSONResponse(
            status_code=200, content={"text": f"No picks available for {date_label}."}
        )

    team_filter = parsed["team_filter"]
    if team_filter:
        plays = [p for p in plays if team_filter in p.get("matchup", "").lower()]

    if parsed["elite_only"]:

        def _fire_rating_value(value) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0

        plays = [p for p in plays if _fire_rating_value(p.get("fire_rating")) >= 4]

    if not plays:
        parts = []
        if parsed["elite_only"]:
            parts.append("elite")
        if team_filter:
            parts.append(f"for {team_filter}")
        label = " ".join(parts).strip()
        label = f"{label} " if label else ""
        date_label = data.get("date", parsed["date"])
        return JSONResponse(
            status_code=200, content={"text": f"No {label}picks found for {date_label}."}
        )

    date_label = data.get("date", parsed["date"])
    label_parts = []
    if parsed["elite_only"]:
        label_parts.append("elite only")
    if team_filter:
        label_parts.append(team_filter)
    label = f" ({', '.join(label_parts)})" if label_parts else ""
    lineup_params = {"date": date_label}
    if team_filter:
        lineup_params["team"] = team_filter
    if parsed["elite_only"]:
        lineup_params["elite"] = "true"
    lineup_url = (
        f"{str(request.base_url).rstrip('/')}/weekly-lineup/html?{urlencode(lineup_params)}"
    )
    csv_url = f"{str(request.base_url).rstrip('/')}/weekly-lineup/csv?{urlencode(lineup_params)}"
    public_lineup_base = os.getenv(
        "PUBLIC_WEEKLY_LINEUP_URL",
        "https://www.greenbiersportventures.com/weekly-lineup.html",
    ).strip()

    def _append_query(base_url: str, params: dict) -> str:
        if not base_url:
            return ""
        query = urlencode(params)
        if not query:
            return base_url
        joiner = "&" if "?" in base_url else "?"
        return f"{base_url}{joiner}{query}"

    public_lineup_url = _append_query(public_lineup_base, lineup_params)

    def _fmt_cell(value: str, width: int) -> str:
        text = str(value or "")
        if len(text) <= width:
            return text.ljust(width)
        if width <= 3:
            return text[:width]
        return text[: width - 3] + "..."

    columns = [
        ("TIME", 12),
        ("MATCHUP", 38),
        ("SEG", 9),
        ("PICK", 24),
        ("EDGE", 9),
        ("FIRE", 6),
    ]
    header = " | ".join(_fmt_cell(label, width) for label, width in columns)
    divider = "-+-".join("-" * width for _, width in columns)

    lines_before = [
        f"NBA picks for {date_label}{label}",
        f"Showing {len(plays)} picks (sorted by EV%)",
        "```",
    ]
    lines_after = [
        "```",
        f"CSV: {csv_url}",
        f"HTML: {lineup_url}",
    ]
    if public_lineup_url:
        lines_after.append(f"Website: {public_lineup_url}")
    lines_after.append("Tip: use 'help' for commands.")

    def _fire_display(value: object) -> str:
        try:
            rating = int(value)
        except (TypeError, ValueError):
            rating = 0
        return "\U0001F525" * rating if rating > 0 else "-"

    table_lines = [header, divider]
    for p in plays:
        segment = f"{p.get('period', '')} {p.get('market', '')}".strip()
        row_values = [
            p.get("time_cst", ""),
            p.get("matchup", ""),
            segment,
            p.get("pick", ""),
            p.get("edge", ""),
            _fire_display(p.get("fire_rating", "")),
        ]
        row = " | ".join(_fmt_cell(value, width) for value, (_, width) in zip(row_values, columns))
        table_lines.append(row)

    lines = lines_before + ["\n".join(table_lines)] + lines_after

    return JSONResponse(status_code=200, content={"text": "\n".join(lines)})
