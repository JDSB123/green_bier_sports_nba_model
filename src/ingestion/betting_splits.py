"""
Betting splits data ingestion.

Collects public betting percentages and money line distributions
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


@dataclass
class GameSplits:
    """Betting splits for a single game."""
    event_id: str
    home_team: str
    away_team: str
    game_time: dt.datetime
    
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
    
    # Moneyline splits
    ml_home_ticket_pct: float = 50.0
    ml_away_ticket_pct: float = 50.0
    ml_home_money_pct: float = 50.0
    ml_away_money_pct: float = 50.0
    
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
    
    source: str = "unknown"
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
    
    # Also check ticket vs money divergence
    # If tickets are on home but money is on away, sharps on away
    ticket_money_diff_spread = (
        splits.spread_home_ticket_pct - splits.spread_home_money_pct
    )
    if abs(ticket_money_diff_spread) > 10:
        # Significant divergence
        if ticket_money_diff_spread > 0:
            # More tickets on home than money -> sharps on away
            splits.sharp_spread_side = "away"
        else:
            splits.sharp_spread_side = "home"
    
    ticket_money_diff_total = splits.over_ticket_pct - splits.over_money_pct
    if abs(ticket_money_diff_total) > 10:
        if ticket_money_diff_total > 0:
            splits.sharp_total_side = "under"
        else:
            splits.sharp_total_side = "over"
    
    return splits


def parse_action_network_splits(data: Dict[str, Any]) -> GameSplits:
    """
    Parse betting splits from Action Network format.
    
    Note: Requires Action Network API subscription.
    """
    game = data.get("game", {})
    markets = data.get("markets", {})
    
    spread_market = markets.get("spread", {})
    total_market = markets.get("total", {})
    ml_market = markets.get("moneyline", {})
    
    splits = GameSplits(
        event_id=str(game.get("id", "")),
        home_team=game.get("home_team", {}).get("name", ""),
        away_team=game.get("away_team", {}).get("name", ""),
        game_time=dt.datetime.fromisoformat(
            game.get("start_time", dt.datetime.now().isoformat())
        ),
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
        # ML
        ml_home_ticket_pct=ml_market.get("home_tickets_pct", 50),
        ml_away_ticket_pct=ml_market.get("away_tickets_pct", 50),
        ml_home_money_pct=ml_market.get("home_money_pct", 50),
        ml_away_money_pct=ml_market.get("away_money_pct", 50),
        source="action_network",
        updated_at=dt.datetime.now(),
    )
    
    return detect_reverse_line_movement(splits)


def parse_the_odds_splits(data: List[Dict[str, Any]]) -> List[GameSplits]:
    """Parse betting splits from The Odds API format."""
    splits_list = []
    
    for game in data:
        try:
            home_team = game.get("home_team", "")
            away_team = game.get("away_team", "")
            
            # The Odds API splits structure:
            # { "id": "...", "home_team": "...", "away_team": "...", "commence_time": "...",
            #   "bookmakers": [{ "key": "...", "markets": [{ "key": "spreads", "outcomes": [{ "name": "...", "public_percentage": 60 }, ...] }] }] }
            
            # Use the first bookmaker that has splits
            bm = game.get("bookmakers", [{}])[0]
            markets = bm.get("markets", [])
            
            spread_market = next((m for m in markets if m["key"] == "spreads"), {})
            total_market = next((m for m in markets if m["key"] == "totals"), {})
            ml_market = next((m for m in markets if m["key"] == "h2h"), {})
            
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
                # ML
                ml_home_ticket_pct=get_pct(ml_market, home_team),
                ml_away_ticket_pct=get_pct(ml_market, away_team),
                source="the_odds",
                updated_at=dt.datetime.now(),
            )
            splits_list.append(detect_reverse_line_movement(splits))
        except Exception as e:
            print(f"[WARN] Failed to parse The Odds API game: {e}")
            continue
            
    return splits_list


def splits_to_features(splits: GameSplits) -> Dict[str, float]:
    """
    Convert GameSplits to feature dict for model input.
    
    Maps to the BettingSplits dataclass in features.py
    """
    return {
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
                print(f"[WARN] SBRO returned status {response.status_code}")
                return []

            # Check if response is JSON (API endpoint) or HTML
            content_type = response.headers.get("content-type", "")

            if "application/json" in content_type:
                data = response.json()
                # Parse JSON structure (varies by endpoint)
                return parse_sbro_json(data)
            else:
                # HTML response - would need parsing
                print("[WARN] SBRO returned HTML - may need JavaScript rendering")
                return []

    except Exception as e:
        print(f"[WARN] Failed to fetch SBRO splits: {e}")
        return []


def parse_sbro_json(data: Dict[str, Any]) -> List[GameSplits]:
    """Parse SBRO JSON response to GameSplits."""
    splits_list = []

    # This structure depends on SBRO's actual API response
    # Placeholder implementation
    games = data.get("events", []) or data.get("games", [])

    for game in games:
        try:
            splits = GameSplits(
                event_id=str(game.get("id", "")),
                home_team=game.get("home", {}).get("name", ""),
                away_team=game.get("away", {}).get("name", ""),
                game_time=datetime.fromisoformat(game.get("datetime", datetime.now().isoformat())),
                # Extract betting percentages if available
                spread_home_ticket_pct=game.get("consensus", {}).get("spread", {}).get("home_pct", 50),
                spread_away_ticket_pct=game.get("consensus", {}).get("spread", {}).get("away_pct", 50),
                source="sbro",
                updated_at=datetime.now(),
            )
            splits_list.append(detect_reverse_line_movement(splits))
        except Exception as e:
            print(f"[WARN] Failed to parse SBRO game: {e}")
            continue

    return splits_list


async def fetch_splits_action_network(date: Optional[str] = None):
    """
    Fetch from Action Network API.
    Requires: ACTION_NETWORK_API_KEY env variable

    NOT IMPLEMENTED - requires paid subscription
    """
    raise NotImplementedError(
        "Action Network API requires subscription. "
        "Set ACTION_NETWORK_API_KEY and implement API calls."
    )


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
        print(f"[WARN] Failed to scrape Covers.com: {e}")
        return []


async def fetch_public_betting_splits(
    games: List[Dict[str, Any]],
    source: str = "auto"
) -> Dict[str, GameSplits]:
    """
    Fetch public betting splits for a list of games.

    Args:
        games: List of game dicts with home_team, away_team, spread, total
        source: Data source ("the_odds", "sbro", "covers", "mock", or "auto")

    Returns:
        Dict mapping game_id to GameSplits
    """
    splits_dict = {}

    if source == "auto":
        # Try sources in order of preference
        sources = ["the_odds", "sbro", "covers", "mock"]
        for src in sources:
            try:
                if src == "the_odds":
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
                    # Fall back to mock data
                    splits_list = _create_mock_splits_for_games(games)

                if splits_list:
                    for splits in splits_list:
                        game_key = f"{splits.away_team}@{splits.home_team}"
                        splits_dict[game_key] = splits
                    print(f"[OK] Loaded betting splits from {src}: {len(splits_list)} games")
                    break
            except Exception as e:
                print(f"[WARN] Failed to fetch from {src}: {e}")
                continue
    else:
        # Use specified source
        if source == "the_odds":
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
            raise ValueError(f"Unknown source: {source}. Valid sources: 'the_odds', 'sbro', 'covers', 'mock', 'auto'")

        for splits in splits_list:
            game_key = f"{splits.away_team}@{splits.home_team}"
            splits_dict[game_key] = splits

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
        source="mock",
        updated_at=dt.datetime.now(),
    )

    return detect_reverse_line_movement(splits)
