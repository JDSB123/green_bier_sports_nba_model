"""
NBA v6.0 - Pick Tracker

Dynamic live tracking of predictions with no leakage.
Records predictions at generation time (BEFORE outcomes are known).
Validates against outcomes AFTER games complete.

Key Anti-Leakage Protections:
1. Predictions are recorded with timestamp BEFORE game starts
2. Outcomes are fetched ONLY after game ends (status = "completed")
3. No prediction can be modified after creation
4. All tracked picks include the betting line at time of prediction
"""
from __future__ import annotations

import os
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Literal
from dataclasses import dataclass, field, asdict
import hashlib

from src.config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Storage location for tracked picks
TRACKING_DIR = Path(settings.data_processed_dir) / "tracking"
PICKS_FILE = TRACKING_DIR / "live_picks.jsonl"


@dataclass
class TrackedPick:
    """A single prediction tracked for ROI validation."""
    
    # Identification
    pick_id: str  # SHA256 of (date, home_team, away_team, market, side)
    created_at: str  # ISO timestamp when prediction was made
    
    # Game info
    game_date: str  # YYYY-MM-DD
    home_team: str
    away_team: str
    
    # Market info
    market: Literal["fh_spread", "fh_total", "fg_spread", "fg_total"]
    period: Literal["1h", "fg"]
    market_type: Literal["spread", "total"]
    
    # Prediction details (recorded BEFORE game)
    side: str  # "home_cover", "away_cover", "over", "under"
    line: Optional[float]  # The betting line at time of prediction
    confidence: float  # 0.0-1.0
    passes_filter: bool  # Whether this met our betting threshold
    
    # Outcome details (filled AFTER game completes)
    status: Literal["pending", "win", "loss", "push", "no_line"] = "pending"
    outcome_fetched_at: Optional[str] = None
    actual_result: Optional[float] = None  # Actual margin/total/etc
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrackedPick":
        return cls(**data)


class PickTracker:
    """
    Manages pick tracking with strict anti-leakage protections.
    
    Usage:
        tracker = PickTracker()
        
        # Record prediction BEFORE game starts
        tracker.record_pick(
            game_date="2025-12-20",
            home_team="Lakers",
            away_team="Celtics",
            market="fg_spread",
            side="home_cover",
            line=-3.5,
            confidence=0.65,
            passes_filter=True
        )
        
        # Later, validate outcomes for completed games
        results = await tracker.validate_outcomes(date="2025-12-20")
    """
    
    def __init__(self, tracking_dir: Optional[Path] = None):
        self.tracking_dir = tracking_dir or TRACKING_DIR
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        self.picks_file = self.tracking_dir / "live_picks.jsonl"
        
    def _generate_pick_id(
        self,
        game_date: str,
        home_team: str,
        away_team: str,
        market: str,
        side: str
    ) -> str:
        """Generate unique ID for a pick."""
        key = f"{game_date}|{home_team}|{away_team}|{market}|{side}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]
    
    def record_pick(
        self,
        game_date: str,
        home_team: str,
        away_team: str,
        market: str,
        side: str,
        line: Optional[float],
        confidence: float,
        passes_filter: bool = True
    ) -> TrackedPick:
        """
        Record a prediction BEFORE the game starts.
        
        This is the ONLY way to add picks - ensures no retroactive modifications.
        """
        # Determine period and market type from market name
        if market.startswith("q1_"):
            period = "q1"
            market_type = market[3:]
        elif market.startswith("fh_"):
            period = "1h"
            market_type = market[3:]
        else:
            period = "fg"
            market_type = market[3:] if market.startswith("fg_") else market
        
        pick = TrackedPick(
            pick_id=self._generate_pick_id(game_date, home_team, away_team, market, side),
            created_at=datetime.now(timezone.utc).isoformat(),
            game_date=game_date,
            home_team=home_team,
            away_team=away_team,
            market=market,
            period=period,
            market_type=market_type,
            side=side,
            line=line,
            confidence=confidence,
            passes_filter=passes_filter,
            status="pending"
        )
        
        # Append to JSONL file (immutable append-only)
        with open(self.picks_file, "a") as f:
            f.write(json.dumps(pick.to_dict()) + "\n")
        
        logger.info(f"Recorded pick: {pick.pick_id} - {home_team} vs {away_team} {market} {side}")
        return pick
    
    def get_picks(
        self,
        date: Optional[str] = None,
        status: Optional[str] = None,
        passes_filter_only: bool = False
    ) -> List[TrackedPick]:
        """Get tracked picks with optional filters."""
        picks = []
        
        if not self.picks_file.exists():
            return picks
        
        with open(self.picks_file, "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    pick = TrackedPick.from_dict(data)
                    
                    if date and pick.game_date != date:
                        continue
                    if status and pick.status != status:
                        continue
                    if passes_filter_only and not pick.passes_filter:
                        continue
                    
                    picks.append(pick)
        
        return picks
    
    def get_pending_picks(self, date: Optional[str] = None) -> List[TrackedPick]:
        """Get picks awaiting outcome validation."""
        return self.get_picks(date=date, status="pending")
    
    def _update_pick_outcome(
        self,
        pick_id: str,
        status: str,
        actual_result: Optional[float] = None
    ) -> bool:
        """
        Update a pick's outcome. Internal use only.
        
        This reads all picks, updates the matching one, and rewrites the file.
        The original prediction data is never modified.
        """
        picks = self.get_picks()
        updated = False
        
        for pick in picks:
            if pick.pick_id == pick_id:
                pick.status = status
                pick.outcome_fetched_at = datetime.now(timezone.utc).isoformat()
                pick.actual_result = actual_result
                updated = True
                break
        
        if updated:
            # Rewrite file with updated picks
            with open(self.picks_file, "w") as f:
                for pick in picks:
                    f.write(json.dumps(pick.to_dict()) + "\n")
        
        return updated
    
    async def validate_outcomes(
        self,
        date: Optional[str] = None,
        force_refetch: bool = False
    ) -> Dict[str, Any]:
        """
        Validate pending picks against actual game outcomes.
        
        Fetches completed game data and updates pick statuses.
        Only processes games that are confirmed complete.
        """
        from src.ingestion.api_basketball import fetch_games
        
        pending = self.get_pending_picks(date=date)
        
        if not pending:
            return {"validated": 0, "wins": 0, "losses": 0, "pushes": 0}
        
        # Get unique dates from pending picks
        dates = set(p.game_date for p in pending)
        
        results = {
            "validated": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "details": []
        }
        
        for game_date in dates:
            try:
                # Fetch completed games for this date
                games = await fetch_games(date=game_date)
                
                # Create lookup by teams
                game_lookup = {}
                for game in games:
                    if game.get("status", "") == "completed":
                        key = (
                            game.get("home_team"),
                            game.get("away_team")
                        )
                        game_lookup[key] = game
                
                # Validate picks against outcomes
                for pick in pending:
                    if pick.game_date != game_date:
                        continue
                    
                    game = game_lookup.get((pick.home_team, pick.away_team))
                    if not game:
                        continue
                    
                    # Calculate outcome based on market
                    status, actual = self._determine_outcome(pick, game)
                    
                    if status != "pending":
                        self._update_pick_outcome(pick.pick_id, status, actual)
                        results["validated"] += 1
                        
                        if status == "win":
                            results["wins"] += 1
                        elif status == "loss":
                            results["losses"] += 1
                        elif status == "push":
                            results["pushes"] += 1
                        
                        results["details"].append({
                            "pick_id": pick.pick_id,
                            "game": f"{pick.home_team} vs {pick.away_team}",
                            "market": pick.market,
                            "side": pick.side,
                            "status": status
                        })
                        
            except Exception as e:
                logger.error(f"Error validating outcomes for {game_date}: {e}")
        
        return results
    
    def _determine_outcome(
        self,
        pick: TrackedPick,
        game: Dict[str, Any]
    ) -> tuple[str, Optional[float]]:
        """
        Determine if a pick was a win, loss, or push based on game outcome.
        
        Returns (status, actual_result).
        """
        # Extract scores based on period
        if pick.period == "q1":
            home_score = game.get("home_q1", 0)
            away_score = game.get("away_q1", 0)
        elif pick.period == "1h":
            home_score = game.get("home_1h_score", 0)
            away_score = game.get("away_1h_score", 0)
        else:
            home_score = game.get("home_score", 0)
            away_score = game.get("away_score", 0)
        
        if home_score is None or away_score is None:
            return "pending", None
        
        margin = home_score - away_score
        total = home_score + away_score
        
        # Determine outcome based on market type
        if pick.market_type == "spread":
            if pick.line is None:
                return "no_line", None
            
            covered_margin = margin + pick.line
            
            if pick.side in ["home_cover", "home"]:
                if covered_margin > 0:
                    return "win", covered_margin
                elif covered_margin < 0:
                    return "loss", covered_margin
                else:
                    return "push", covered_margin
            else:  # away_cover
                if covered_margin < 0:
                    return "win", covered_margin
                elif covered_margin > 0:
                    return "loss", covered_margin
                else:
                    return "push", covered_margin
                    
        elif pick.market_type == "total":
            if pick.line is None:
                return "no_line", None
            
            if pick.side == "over":
                if total > pick.line:
                    return "win", total
                elif total < pick.line:
                    return "loss", total
                else:
                    return "push", total
            else:  # under
                if total < pick.line:
                    return "win", total
                elif total > pick.line:
                    return "loss", total
                else:
                    return "push", total
                    
        return "pending", None
    
    def get_roi_summary(
        self,
        date: Optional[str] = None,
        period: Optional[str] = None,
        market_type: Optional[str] = None,
        passes_filter_only: bool = True
    ) -> Dict[str, Any]:
        """
        Calculate ROI summary for tracked picks.
        
        Assumes -110 odds for spread/total.
        """
        picks = self.get_picks(date=date, passes_filter_only=passes_filter_only)
        
        # Apply additional filters
        if period:
            picks = [p for p in picks if p.period == period]
        if market_type:
            picks = [p for p in picks if p.market_type == market_type]
        
        resolved = [p for p in picks if p.status in ["win", "loss", "push"]]
        
        wins = sum(1 for p in resolved if p.status == "win")
        losses = sum(1 for p in resolved if p.status == "loss")
        pushes = sum(1 for p in resolved if p.status == "push")
        total = wins + losses
        
        if total == 0:
            return {
                "total_picks": len(picks),
                "resolved": 0,
                "pending": len([p for p in picks if p.status == "pending"]),
                "wins": 0,
                "losses": 0,
                "pushes": 0,
                "accuracy": 0.0,
                "roi": 0.0,
                "units_won": 0.0
            }
        
        accuracy = wins / total if total > 0 else 0.0
        
        # Calculate ROI assuming 1 unit bets at -110 odds
        # Win pays 0.909 units, loss costs 1 unit
        units_won = (wins * 0.909) - losses
        roi = (units_won / total) * 100 if total > 0 else 0.0
        
        return {
            "total_picks": len(picks),
            "resolved": len(resolved),
            "pending": len([p for p in picks if p.status == "pending"]),
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "accuracy": round(accuracy * 100, 1),
            "roi": round(roi, 1),
            "units_won": round(units_won, 2)
        }
    
    def get_streak(self, passes_filter_only: bool = True) -> Dict[str, Any]:
        """Get current win/loss streak."""
        picks = self.get_picks(passes_filter_only=passes_filter_only)
        resolved = [p for p in picks if p.status in ["win", "loss"]]
        
        if not resolved:
            return {"streak": 0, "type": None}
        
        # Sort by creation time
        resolved.sort(key=lambda p: p.created_at, reverse=True)
        
        streak_type = resolved[0].status
        streak = 0
        
        for pick in resolved:
            if pick.status == streak_type:
                streak += 1
            else:
                break
        
        return {"streak": streak, "type": streak_type}


# Global tracker instance for API use
_tracker: Optional[PickTracker] = None


def get_tracker() -> PickTracker:
    """Get or create the global pick tracker."""
    global _tracker
    if _tracker is None:
        _tracker = PickTracker()
    return _tracker
