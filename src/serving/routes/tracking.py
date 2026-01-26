"""
Tracking routes for ROI summary, picks history, and outcome validation.
"""

from typing import Optional

from fastapi import APIRouter, Query, Request

from src.serving.dependencies import get_tracker, limiter, logger

router = APIRouter(prefix="/tracking", tags=["Tracking"])


@router.get("/summary")
@limiter.limit("30/minute")
async def get_tracking_summary(
    request: Request,
    date: Optional[str] = Query(None, description="Filter by date (YYYY-MM-DD)"),
    period: Optional[str] = Query(None, description="Filter by period (1h, fg)"),
    market_type: Optional[str] = Query(None, description="Filter by market (spread, total)"),
):
    """
    Get ROI summary for tracked picks.

    Provides accuracy, ROI, and win/loss breakdown for live tracked predictions.
    Only includes picks that passed the betting filter.
    """
    tracker = get_tracker(request)
    summary = tracker.get_roi_summary(
        date=date, period=period, market_type=market_type, passes_filter_only=True
    )
    streak = tracker.get_streak(passes_filter_only=True)

    return {
        "summary": summary,
        "current_streak": streak,
        "filters": {"date": date, "period": period, "market_type": market_type},
    }


@router.get("/picks")
@limiter.limit("30/minute")
async def get_tracked_picks(
    request: Request,
    date: Optional[str] = Query(None, description="Filter by date (YYYY-MM-DD)"),
    status: Optional[str] = Query(None, description="Filter by status (pending, win, loss, push)"),
    limit: int = Query(50, ge=1, le=500),
):
    """
    Get list of tracked picks with optional filters.
    """
    tracker = get_tracker(request)
    picks = tracker.get_picks(date=date, status=status, passes_filter_only=True)

    # Sort by creation time, newest first
    picks.sort(key=lambda p: p.created_at, reverse=True)

    return {"total": len(picks), "picks": [p.to_dict() for p in picks[:limit]]}


@router.post("/validate")
@limiter.limit("10/minute")
async def validate_pick_outcomes(
    request: Request,
    date: Optional[str] = Query(None, description="Date to validate (YYYY-MM-DD)"),
):
    """
    Validate pending picks against game outcomes.

    Fetches completed game results and updates pick statuses.
    Only processes games that are confirmed complete.
    """
    tracker = get_tracker(request)
    results = await tracker.validate_outcomes(date=date)

    return {
        "validated": results["validated"],
        "wins": results["wins"],
        "losses": results["losses"],
        "pushes": results["pushes"],
        "details": results.get("details", []),
    }
