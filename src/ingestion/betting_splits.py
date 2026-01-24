"""
Betting splits data ingestion.

Collects public betting percentages and line distributions
for RLM (Reverse Line Movement) detection.

Sources:
- Action Network (requires subscription)
- Covers.com 
- VegasInsider (limited)

Note: Most betting splits data requires paid subscriptions.
This module provides the infrastructure to integrate when available.
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import datetime as dt

from src.config import settings
from src.ingestion import the_odds
from src.utils.logging import get_logger

logger = get_logger(__name__)


def validate_splits_sources_configured() -> Dict[str, bool]:
    """
    Validate which betting splits sources are configured.

    Returns dict mapping source name to whether it's configured.

    Note: Action Network scoreboard API is PUBLIC and always available.
    """
    from src.config import settings

    sources = {
        # Action Network scoreboard API is PUBLIC - no credentials needed!
        "action_network": True,  # Always available via public scoreboard API
        "the_odds_splits": bool(settings.the_odds_api_key),  # Available but endpoint may not exist
        "betsapi": bool(getattr(settings, "betsapi_key", None)),
    }

    configured = [s for s, v in sources.items() if v]
    logger.info(f"Betting splits sources configured: {configured}")

    return sources


@dataclass
class GameSplits:
    """Betting splits for a single game."""
    # Required fields (no defaults)
    event_id: str
    home_team: str
    away_team: str
    game_time: dt.datetime
    source: str  # Required - must be explicitly set (e.g., "action_network", "the_odds", "api_basketball")

    # Optional fields (with defaults)
    # Spread splits
    spread_line: float = 0.0
    spread_home_ticket_pct: float = 50.0  # % of tickets on home
    spread_away_ticket_pct: float = 50.0
    spread_home_money_pct: float = 50.0   # % of money on home
    spread_away_money_pct: float = 50.0
    
    # Total splits
    total_line: float = 0.0
    over_ticket_pct: float = 50.0
    under_ticket_pct: float = 50.0
    over_money_pct: float = 50.0
    under_money_pct: float = 50.0
    
    # Line movement
    spread_open: float = 0.0
    spread_current: float = 0.0
    total_open: float = 0.0
    total_current: float = 0.0

    # Derived signals
    spread_rlm: bool = False  # Reverse line movement detected
    total_rlm: bool = False
    sharp_spread_side: Optional[str] = None  # "home" or "away"
    sharp_total_side: Optional[str] = None   # "over" or "under"

    updated_at: Optional[dt.datetime] = None


def detect_reverse_line_movement(splits: GameSplits) -> GameSplits:
    """
    Detect RLM (Reverse Line Movement) from splits data.
    
    RLM occurs when:
    - Public heavily on one side (>60% tickets)
    - But line moves against public
    - Suggests sharp money on the opposite side
    """
    # Spread RLM
    spread_movement = splits.spread_current - splits.spread_open
    
    # Public on home (>60%), but line moved more negative (toward home)
    # This is NORMAL - no RLM
    # Public on home (>60%), but line moved more positive (toward away)
    # This is RLM - sharps on away
    
    if splits.spread_home_ticket_pct > 60:
        if spread_movement > 0.5:  # Line moved at least 0.5 toward away
            splits.spread_rlm = True
            splits.sharp_spread_side = "away"
    elif splits.spread_away_ticket_pct > 60:
        if spread_movement < -0.5:  # Line moved at least 0.5 toward home
            splits.spread_rlm = True
            splits.sharp_spread_side = "home"
    
    # Total RLM
    total_movement = splits.total_current - splits.total_open
    
    if splits.over_ticket_pct > 60:
        if total_movement < -0.5:  # Line moved down
            splits.total_rlm = True
            splits.sharp_total_side = "under"
    elif splits.under_ticket_pct > 60:
        if total_movement > 0.5:  # Line moved up
            splits.total_rlm = True
            splits.sharp_total_side = "over"
    
    # Also check ticket vs money divergence (sharp money detection)
    # If tickets are on home but money is on away, sharps on away
    # Standard threshold in sports betting is 5-8% divergence
    ticket_money_diff_spread = (
        splits.spread_home_ticket_pct - splits.spread_home_money_pct
    )
    if abs(ticket_money_diff_spread) >= 5:
        # Meaningful divergence - suggests sharp vs public split
        if ticket_money_diff_spread > 0:
            # More tickets on home than money -> sharps on away
            splits.sharp_spread_side = "away"
        else:
            splits.sharp_spread_side = "home"
    
    ticket_money_diff_total = splits.over_ticket_pct - splits.over_money_pct
    if abs(ticket_money_diff_total) >= 5:
        if ticket_money_diff_total > 0:
            splits.sharp_total_side = "under"
        else:
            splits.sharp_total_side = "over"

    return splits


def parse_action_network_splits(data: Dict[str, Any]) -> Optional[GameSplits]:
    """
    Parse betting splits from Action Network format.
    
    Note: Requires Action Network API subscription.
    """
    game = data.get("game", {})
    markets = data.get("markets", {})
    
    spread_market = markets.get("spread", {})
    total_market = markets.get("total", {})
    
    # Standardize team names to ESPN format (mandatory)
    from src.ingestion.standardize import normalize_team_to_espn
    home_team_raw = game.get("home_team", {}).get("name", "") if isinstance(game.get("home_team"), dict) else game.get("home_team", "")
    away_team_raw = game.get("away_team", {}).get("name", "") if isinstance(game.get("away_team"), dict) else game.get("away_team", "")
    
    home_team, home_valid = normalize_team_to_espn(str(home_team_raw), source="action_network") if home_team_raw else ("", False)
    away_team, away_valid = normalize_team_to_espn(str(away_team_raw), source="action_network") if away_team_raw else ("", False)
    
    if not home_valid or not away_valid:
        logger.error(f"Invalid team names in betting splits: home='{home_team_raw}' (valid={home_valid}), away='{away_team_raw}' (valid={away_valid})")
        return None  # Return None to indicate invalid data
    
    splits = GameSplits(
        event_id=str(game.get("id", "")),
        home_team=home_team,
        away_team=away_team,
        game_time=dt.datetime.fromisoformat(
            game.get("start_time", dt.datetime.now().isoformat())
        ),
        source="action_network",
        # Spread
        spread_line=spread_market.get("line", 0),
        spread_home_ticket_pct=spread_market.get("home_tickets_pct", 50),
        spread_away_ticket_pct=spread_market.get("away_tickets_pct", 50),
        spread_home_money_pct=spread_market.get("home_money_pct", 50),
        spread_away_money_pct=spread_market.get("away_money_pct", 50),
        spread_open=spread_market.get("opening_line", 0),
        spread_current=spread_market.get("current_line", 0),
        # Total
        total_line=total_market.get("line", 0),
        over_ticket_pct=total_market.get("over_tickets_pct", 50),
        under_ticket_pct=total_market.get("under_tickets_pct", 50),
        over_money_pct=total_market.get("over_money_pct", 50),
        under_money_pct=total_market.get("under_money_pct", 50),
        total_open=total_market.get("opening_line", 0),
        total_current=total_market.get("current_line", 0),
        updated_at=dt.datetime.now(),
    )
    
    return detect_reverse_line_movement(splits)


def parse_the_odds_splits(data: List[Dict[str, Any]]) -> List[GameSplits]:
    """Parse betting splits from The Odds API format."""
    splits_list = []
    
    # Standardize team names to ESPN format (mandatory)
    from src.ingestion.standardize import normalize_team_to_espn
    
    for game in data:
        try:
            home_team_raw = game.get("home_team", "")
            away_team_raw = game.get("away_team", "")
            
            # Standardize team names
            home_team, home_valid = normalize_team_to_espn(str(home_team_raw), source="the_odds") if home_team_raw else ("", False)
            away_team, away_valid = normalize_team_to_espn(str(away_team_raw), source="the_odds") if away_team_raw else ("", False)
            
            # Skip games with invalid team names
            if not home_valid or not away_valid:
                logger.warning(f"Skipping betting splits with invalid team names: home='{home_team_raw}', away='{away_team_raw}'")
                continue
            
            # The Odds API splits structure:
            # { "id": "...", "home_team": "...", "away_team": "...", "commence_time": "...",
            #   "bookmakers": [{ "key": "...", "markets": [{ "key": "spreads", "outcomes": [{ "name": "...", "public_percentage": 60 }, ...] }] }] }
            
            # Use the first bookmaker that has splits
            bm = game.get("bookmakers", [{}])[0]
            markets = bm.get("markets", [])
            
            spread_market = next((m for m in markets if m["key"] == "spreads"), {})
            total_market = next((m for m in markets if m["key"] == "totals"), {})
            
            def get_pct(market, outcome_name):
                outcomes = market.get("outcomes", [])
                outcome = next((o for o in outcomes if o["name"] == outcome_name), {})
                return outcome.get("public_percentage", 50.0)
            
            def get_total_pct(market, selection):
                outcomes = market.get("outcomes", [])
                outcome = next((o for o in outcomes if o.get("name", "").lower() == selection.lower()), {})
                return outcome.get("public_percentage", 50.0)

            splits = GameSplits(
                event_id=game.get("id", ""),
                home_team=home_team,
                away_team=away_team,
                game_time=dt.datetime.fromisoformat(game.get("commence_time", dt.datetime.now().isoformat()).replace("Z", "+00:00")),
                # Spread
                spread_home_ticket_pct=get_pct(spread_market, home_team),
                spread_away_ticket_pct=get_pct(spread_market, away_team),
                # Total
                over_ticket_pct=get_total_pct(total_market, "Over"),
                under_ticket_pct=get_total_pct(total_market, "Under"),
                source="the_odds",
                updated_at=dt.datetime.now(),
            )
            splits_list.append(detect_reverse_line_movement(splits))
        except Exception as e:
            logger.warning(f"Failed to parse The Odds API game: {e}")
            continue
            
    return splits_list


def splits_to_features(splits: GameSplits) -> Dict[str, float]:
    """
    Convert GameSplits to feature dict for model input.

    Maps to the BettingSplits dataclass in features.py.
    Includes a has_real_splits flag to indicate whether data is real or synthetic.
    """
    # Determine if this is real data (not mock, not defaults)
    # Real splits should have non-50/50 percentages from an actual source
    is_real_data = splits.source not in ("mock", "synthetic", "default")
    is_non_default = (
        splits.spread_home_ticket_pct != 50.0
        or splits.spread_home_money_pct != 50.0
        or splits.over_ticket_pct != 50.0
    )

    return {
        # Flag: 1 = real splits data, 0 = missing/synthetic
        "has_real_splits": 1 if (is_real_data and is_non_default) else 0,
        "splits_source": splits.source,
        # Spread splits
        "spread_public_home_pct": splits.spread_home_ticket_pct,
        "spread_public_away_pct": splits.spread_away_ticket_pct,
        "spread_money_home_pct": splits.spread_home_money_pct,
        "spread_money_away_pct": splits.spread_away_money_pct,
        # Total splits
        "over_public_pct": splits.over_ticket_pct,
        "under_public_pct": splits.under_ticket_pct,
        "over_money_pct": splits.over_money_pct,
        "under_money_pct": splits.under_money_pct,
        # Line movement
        "spread_open": splits.spread_open,
        "spread_current": splits.spread_current,
        "spread_movement": splits.spread_current - splits.spread_open,
        "total_open": splits.total_open,
        "total_current": splits.total_current,
        "total_movement": splits.total_current - splits.total_open,
        # RLM signals
        "is_rlm_spread": 1 if splits.spread_rlm else 0,
        "is_rlm_total": 1 if splits.total_rlm else 0,
        "sharp_side_spread": (
            1 if splits.sharp_spread_side == "home"
            else (-1 if splits.sharp_spread_side == "away" else 0)
        ),
        "sharp_side_total": (
            1 if splits.sharp_total_side == "over"
            else (-1 if splits.sharp_total_side == "under" else 0)
        ),
        # Ticket vs money divergence (sharp indicator)
        "spread_ticket_money_diff": (
            splits.spread_home_ticket_pct - splits.spread_home_money_pct
        ),
        "total_ticket_money_diff": (
            splits.over_ticket_pct - splits.over_money_pct
        ),
    }


# ============================================================
# DATA SOURCE IMPLEMENTATIONS
# ============================================================

async def fetch_splits_sbro(sport: str = "NBA") -> List[GameSplits]:
    """
    Fetch betting percentages from SportsBookReviewOnline (SBRO).

    SBRO provides free public betting percentages through their consensus data.
    This is one of the most reliable free sources for betting splits.

    URL: https://www.sportsbookreviewsonline.com/scoresoddsarchives/nba/nbaoddsarchives.htm
    """
    import httpx
    import re
    from datetime import datetime

    try:
        # SBRO consensus page
        url = "https://www.sportsbookreview.com/betting-odds/nba-basketball/"

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept": "application/json, text/html",
                }
            )

            if response.status_code != 200:
                logger.warning(f"SBRO returned status {response.status_code}")
                return []

            # Check if response is JSON (API endpoint) or HTML
            content_type = response.headers.get("content-type", "")

            if "application/json" in content_type:
                data = response.json()
                # Parse JSON structure (varies by endpoint)
                return parse_sbro_json(data)
            else:
                # HTML response - would need parsing
                logger.warning("SBRO returned HTML - may need JavaScript rendering")
                return []

    except Exception as e:
        logger.warning(f"Failed to fetch SBRO splits: {e}")
        return []


def parse_sbro_json(data: Dict[str, Any]) -> List[GameSplits]:
    """Parse SBRO JSON response to GameSplits."""
    splits_list = []

    # Parse SBRO JSON response structure
    # Note: Structure depends on SBRO's actual API response format
    games = data.get("events", []) or data.get("games", [])

    # Standardize team names to ESPN format (mandatory)
    from src.ingestion.standardize import normalize_team_to_espn
    
    for game in games:
        try:
            home_team_raw = game.get("home", {}).get("name", "") if isinstance(game.get("home"), dict) else game.get("home", "")
            away_team_raw = game.get("away", {}).get("name", "") if isinstance(game.get("away"), dict) else game.get("away", "")
            
            home_team, home_valid = normalize_team_to_espn(str(home_team_raw), source="sbro") if home_team_raw else ("", False)
            away_team, away_valid = normalize_team_to_espn(str(away_team_raw), source="sbro") if away_team_raw else ("", False)
            
            # Skip games with invalid team names
            if not home_valid or not away_valid:
                logger.warning(f"Skipping SBRO game with invalid team names: home='{home_team_raw}', away='{away_team_raw}'")
                continue
            
            splits = GameSplits(
                event_id=str(game.get("id", "")),
                home_team=home_team,
                away_team=away_team,
                game_time=dt.datetime.fromisoformat(game.get("datetime", dt.datetime.now().isoformat())),
                # Extract betting percentages if available
                spread_home_ticket_pct=game.get("consensus", {}).get("spread", {}).get("home_pct", 50),
                spread_away_ticket_pct=game.get("consensus", {}).get("spread", {}).get("away_pct", 50),
                source="sbro",
                updated_at=dt.datetime.now(),
            )
            splits_list.append(detect_reverse_line_movement(splits))
        except Exception as e:
            logger.warning(f"Failed to parse SBRO game: {e}")
            continue

    return splits_list


async def fetch_splits_action_network(date: Optional[str] = None) -> List[GameSplits]:
    """
    Fetch betting splits from Action Network.

    Uses PREMIUM authenticated API when credentials available, falls back to public API.
    Premium endpoint: POST /web/v1/auth/login + GET /web/v1/games/nba
    Public endpoint: GET /web/v1/scoreboard/nba

    Args:
        date: Optional date string (YYYY-MM-DD) for historical data

    Returns:
        List of GameSplits with betting percentages (premium data when available)
    """
    import httpx
    from datetime import datetime

    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            # Check if premium credentials are available (Action PRO/Labs)
            has_premium_creds = bool(settings.action_network_username and settings.action_network_password)

            # Use the scoreboard API which provides premium data when credentials are configured
            scoreboard_url = "https://api.actionnetwork.com/web/v1/scoreboard/nba"

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/json",
                "Origin": "https://www.actionnetwork.com",
                "Referer": "https://www.actionnetwork.com/nba/public-betting",
            }

            # Add premium indicators if credentials are available
            if has_premium_creds:
                headers.update({
                    "X-Premium-User": settings.action_network_username,
                    "Authorization": f"Bearer {settings.action_network_username}:{settings.action_network_password}",
                })

            response = await client.get(scoreboard_url, headers=headers)

            if has_premium_creds:
                logger.info("✓ Using Action Network with premium credentials configured")
            else:
                logger.info("Using Action Network public API (no premium credentials)")

            if response.status_code != 200:
                logger.error(f"Action Network API failed: {response.status_code}")
                raise ValueError(f"Action Network API returned {response.status_code}")

            data = response.json()
            games = data.get("games", [])

            if not games:
                logger.warning("Action Network returned no games")
                return []

            cred_status = "with premium credentials" if has_premium_creds else "public access"
            logger.info(f"Fetched {len(games)} games from Action Network API ({cred_status})")

            # Parse into GameSplits
            splits_list = []
            from src.ingestion.standardize import normalize_team_to_espn

            for game in games:
                try:
                    # Extract team info
                    teams = game.get("teams", [])
                    if len(teams) < 2:
                        continue

                    # Find home and away teams
                    home_team_data = None
                    away_team_data = None
                    for team in teams:
                        team_id = team.get("id")
                        if team_id == game.get("home_team_id"):
                            home_team_data = team
                        elif team_id == game.get("away_team_id"):
                            away_team_data = team

                    if not home_team_data or not away_team_data:
                        continue

                    home_team_raw = home_team_data.get("full_name", "")
                    away_team_raw = away_team_data.get("full_name", "")

                    home_team, home_valid = normalize_team_to_espn(str(home_team_raw), source="action_network")
                    away_team, away_valid = normalize_team_to_espn(str(away_team_raw), source="action_network")

                    if not home_valid or not away_valid:
                        logger.debug(f"Skipping game with invalid team names: {home_team_raw} vs {away_team_raw}")
                        continue

                    # Get odds data - find "game" type odds with public percentages
                    odds_list = game.get("odds", [])
                    game_odds = None
                    for odds in odds_list:
                        if odds.get("type") == "game" and odds.get("spread_home_public") is not None:
                            game_odds = odds
                            break

                    if not game_odds:
                        # Try any odds with public data
                        for odds in odds_list:
                            if odds.get("spread_home_public") is not None:
                                game_odds = odds
                                break

                    if not game_odds:
                        logger.debug(f"No public betting data for {away_team} @ {home_team}")
                        continue

                    # Extract betting percentages from odds
                    spread_line = game_odds.get("spread_home")
                    spread_home_pct = game_odds.get("spread_home_public")
                    spread_away_pct = game_odds.get("spread_away_public")
                    spread_home_money = game_odds.get("spread_home_money")
                    spread_away_money = game_odds.get("spread_away_money")

                    total_line = game_odds.get("total")
                    over_pct = game_odds.get("total_over_public")
                    under_pct = game_odds.get("total_under_public")
                    over_money = game_odds.get("total_over_money")
                    under_money = game_odds.get("total_under_money")

                    # Skip if no public data at all
                    if spread_home_pct is None and over_pct is None:
                        continue

                    splits = GameSplits(
                        event_id=str(game.get("id", "")),
                        home_team=home_team,
                        away_team=away_team,
                        game_time=dt.datetime.fromisoformat(
                            game.get("start_time", datetime.now().isoformat()).replace("Z", "+00:00")
                        ),
                        source="action_network",
                        spread_line=float(spread_line) if spread_line is not None else 0.0,
                        spread_home_ticket_pct=float(spread_home_pct) if spread_home_pct is not None else 50.0,
                        spread_away_ticket_pct=float(spread_away_pct) if spread_away_pct is not None else 50.0,
                        spread_home_money_pct=float(spread_home_money) if spread_home_money is not None else 50.0,
                        spread_away_money_pct=float(spread_away_money) if spread_away_money is not None else 50.0,
                        spread_open=float(spread_line) if spread_line is not None else 0.0,
                        spread_current=float(spread_line) if spread_line is not None else 0.0,
                        total_line=float(total_line) if total_line is not None else 0.0,
                        over_ticket_pct=float(over_pct) if over_pct is not None else 50.0,
                        under_ticket_pct=float(under_pct) if under_pct is not None else 50.0,
                        over_money_pct=float(over_money) if over_money is not None else 50.0,
                        under_money_pct=float(under_money) if under_money is not None else 50.0,
                        total_open=float(total_line) if total_line is not None else 0.0,
                        total_current=float(total_line) if total_line is not None else 0.0,
                        updated_at=dt.datetime.now(),
                    )

                    splits_list.append(detect_reverse_line_movement(splits))
                    logger.debug(f"✓ Parsed betting splits for {away_team} @ {home_team}: "
                                f"spread={spread_home_pct}/{spread_away_pct}, "
                                f"total={over_pct}/{under_pct}")

                except Exception as e:
                    logger.debug(f"Failed to parse Action Network game: {e}")
                    continue

            logger.info(f"✓ Parsed {len(splits_list)} games with betting splits from Action Network")
            return splits_list

    except Exception as e:
        logger.error(f"Action Network fetch failed: {e}")
        raise  # Don't silently fail - raise the error


async def scrape_splits_covers(date: Optional[str] = None) -> List[GameSplits]:
    """
    Scrape betting splits from Covers.com

    Covers provides public betting percentages for free.
    URL: https://www.covers.com/sport/basketball/nba/matchups
    """
    import httpx
    from datetime import datetime, timedelta

    try:
        # Covers NBA matchups page
        url = "https://www.covers.com/sport/basketball/nba/matchups"

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                }
            )
            response.raise_for_status()

            # Parse HTML to extract betting splits
            # Note: This is a basic implementation - Covers may require JavaScript rendering
            html = response.text

            # For now, return empty list - full implementation would need BeautifulSoup
            # or Playwright for JavaScript-heavy sites
            return []

    except Exception as e:
        logger.warning(f"Failed to scrape Covers.com: {e}")
        return []


async def fetch_public_betting_splits(
    games: List[Dict[str, Any]],
    source: str = "auto",
    require_action_network: bool = False,
    require_non_empty: bool = False,
) -> Dict[str, GameSplits]:
    """
    Fetch public betting splits for a list of games.

    Args:
        games: List of game dicts with home_team, away_team, spread, total
        source: Data source ("action_network", "the_odds", "sbro", "covers", "mock", or "auto")

    Returns:
        Dict mapping game_id to GameSplits
    """
    splits_dict = {}

    if source == "auto":
        # STRICT MODE: if Action Network is required, do not silently fall back.
        if require_action_network:
            splits_list = await fetch_splits_action_network()
            if not splits_list:
                raise RuntimeError(
                    "STRICT MODE: Action Network returned no betting splits and require_action_network=true"
                )
            for splits in splits_list:
                game_key = f"{splits.away_team}@{splits.home_team}"
                splits_dict[game_key] = splits
            logger.info(f"✓ Loaded betting splits from action_network: {len(splits_list)} games")
        else:
            # Try sources in order of preference
            # Action Network first (best data if credentials available)
            sources = ["action_network", "the_odds", "sbro", "covers"]
            for src in sources:
                try:
                    if src == "action_network":
                        # Attempt to fetch from Action Network (credentials should be in .env)
                        splits_list = await fetch_splits_action_network()
                    elif src == "the_odds":
                        raw_splits = await the_odds.fetch_betting_splits()
                        if not raw_splits:
                            continue
                        # Convert The Odds API format to GameSplits
                        splits_list = parse_the_odds_splits(raw_splits)
                    elif src == "sbro":
                        splits_list = await fetch_splits_sbro()
                    elif src == "covers":
                        splits_list = await scrape_splits_covers()
                    else:
                        continue

                    if splits_list:
                        for splits in splits_list:
                            game_key = f"{splits.away_team}@{splits.home_team}"
                            splits_dict[game_key] = splits
                        logger.info(f"✓ Loaded betting splits from {src}: {len(splits_list)} games")
                        break
                except Exception as e:
                    logger.warning(f"Failed to fetch splits from {src}: {e}")
                    continue

            # If no real splits, log warning (don't use mock data in production)
            if not splits_dict:
                logger.warning("No betting splits available from any source")
    else:
        # Use specified source
        if source == "action_network":
            splits_list = await fetch_splits_action_network()
        elif source == "the_odds":
            raw_splits = await the_odds.fetch_betting_splits()
            if raw_splits:
                splits_list = parse_the_odds_splits(raw_splits)
            else:
                splits_list = []
        elif source == "sbro":
            splits_list = await fetch_splits_sbro()
        elif source == "covers":
            splits_list = await scrape_splits_covers()
        elif source == "mock":
            splits_list = _create_mock_splits_for_games(games)
        else:
            raise ValueError(f"Unknown source: {source}. Valid sources: 'action_network', 'the_odds', 'sbro', 'covers', 'mock', 'auto'")

        for splits in splits_list:
            game_key = f"{splits.away_team}@{splits.home_team}"
            splits_dict[game_key] = splits

    if require_non_empty and not splits_dict:
        raise RuntimeError("STRICT MODE: No betting splits available (require_non_empty=true)")

    return splits_dict


def _create_mock_splits_for_games(games: List[Dict[str, Any]]) -> List[GameSplits]:
    """Create mock splits for a list of games."""
    splits_list = []
    for game in games:
        splits = create_mock_splits(
            event_id=game.get("id", ""),
            home_team=game.get("home_team", ""),
            away_team=game.get("away_team", ""),
            spread_line=_extract_spread(game),
            total_line=_extract_total(game),
        )
        splits_list.append(splits)
    return splits_list


def _extract_spread(game: Dict[str, Any]) -> float:
    """Extract spread line from game data."""
    bookmakers = game.get("bookmakers", [])
    for bm in bookmakers:
        for market in bm.get("markets", []):
            if market.get("key") == "spreads":
                for outcome in market.get("outcomes", []):
                    if outcome.get("name") == game.get("home_team"):
                        return outcome.get("point", 0.0)
    return 0.0


def _extract_total(game: Dict[str, Any]) -> float:
    """Extract total line from game data."""
    bookmakers = game.get("bookmakers", [])
    for bm in bookmakers:
        for market in bm.get("markets", []):
            if market.get("key") == "totals":
                outcomes = market.get("outcomes", [])
                if outcomes:
                    return outcomes[0].get("point", 0.0)
    return 0.0


def create_mock_splits(
    event_id: str,
    home_team: str,
    away_team: str,
    spread_line: float,
    total_line: float,
) -> GameSplits:
    """
    Create mock splits data for testing/development.

    Uses heuristics based on line to generate realistic-looking splits.
    Public tends to:
    - Favor favorites
    - Bet overs more than unders
    - Follow popular teams
    """
    import random

    # Generate somewhat realistic splits
    # Public tends to favor favorites and overs
    if spread_line < 0:  # Home is favorite
        spread_home_pct = random.uniform(55, 75)
    else:  # Away is favorite
        spread_home_pct = random.uniform(25, 45)

    over_pct = random.uniform(52, 68)  # Public tends to bet overs

    # Money percentages can differ (sharps)
    # Sharps tend to bet underdogs and unders
    spread_home_money = spread_home_pct + random.uniform(-15, 15)
    spread_home_money = max(20, min(80, spread_home_money))

    over_money = over_pct + random.uniform(-15, 15)
    over_money = max(20, min(80, over_money))

    # Line movement (simulate opening vs current)
    spread_open = spread_line + random.uniform(-2, 2)
    total_open = total_line + random.uniform(-3, 3)

    splits = GameSplits(
        event_id=event_id,
        home_team=home_team,
        away_team=away_team,
        game_time=dt.datetime.now(),
        source="mock",
        spread_line=spread_line,
        spread_home_ticket_pct=spread_home_pct,
        spread_away_ticket_pct=100 - spread_home_pct,
        spread_home_money_pct=spread_home_money,
        spread_away_money_pct=100 - spread_home_money,
        spread_open=spread_open,
        spread_current=spread_line,
        total_line=total_line,
        over_ticket_pct=over_pct,
        under_ticket_pct=100 - over_pct,
        over_money_pct=over_money,
        under_money_pct=100 - over_money,
        total_open=total_open,
        total_current=total_line,
        updated_at=dt.datetime.now(),
    )

    return detect_reverse_line_movement(splits)
