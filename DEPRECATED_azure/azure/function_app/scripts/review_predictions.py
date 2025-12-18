#!/usr/bin/env python3
"""
Review model picks against final scores using ingestion data.

Inputs:
- Slate analysis JSON (`data/processed/slate_analysis_<DATE>.json`)
- API-Basketball game results cached under `data/raw/api_basketball/games_*.json`

Outputs:
- JSON summary (`data/processed/pick_review_<DATE>.json`)
- Markdown summary (`data/processed/pick_review_<DATE>.md`)
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import settings
from src.utils.team_names import normalize_team_name, are_same_team

CST = ZoneInfo("America/Chicago")

# #region agent log
DEBUG_LOG_PATH = Path(__file__).parent.parent / ".cursor" / "debug.log"
def _debug_log(location: str, message: str, data: dict):
    """Write debug log entry."""
    try:
        log_entry = {
            "sessionId": "debug-session",
            "runId": "review-debug",
            "hypothesisId": "B",
            "location": location,
            "message": message,
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception:
        pass  # Don't break execution if logging fails
# #endregion


def get_target_date(value: Optional[str]) -> date:
    """Return the review date (defaults to yesterday in CST)."""
    today = datetime.now(CST).date()
    if not value or value.lower() == "yesterday":
        return today - timedelta(days=1)
    if value.lower() == "today":
        return today
    return datetime.strptime(value, "%Y-%m-%d").date()


def load_analysis(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Slate analysis file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def latest_api_basketball_file() -> Path:
    raw_dir = Path(settings.data_raw_dir) / "api_basketball"
    files = sorted(raw_dir.glob("games_*.json"))
    if not files:
        raise FileNotFoundError(
            "No API-Basketball raw games found. "
            "Run `python scripts/collect_api_basketball.py --process-only` first."
        )
    return files[-1]


def load_raw_games() -> List[Dict[str, Any]]:
    path = latest_api_basketball_file()
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload.get("response", [])


def parse_iso_datetime(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


def sum_period(values: List[Optional[int]]) -> int:
    total = 0
    for val in values:
        if val is None:
            continue
        total += int(val)
    return total


def build_scoreboard_map(target_date: date) -> Dict[Tuple[str, str], Dict[str, Any]]:
    games = load_raw_games()
    scoreboard: Dict[Tuple[str, str], Dict[str, Any]] = {}
    # #region agent log
    DEBUG_LOG_PATH = Path(__file__).parent.parent / ".cursor" / "debug.log"
    def _debug_log(location: str, message: str, data: dict, hypothesisId: str = "C"):
        try:
            import json as json_lib
            from datetime import datetime, timezone
            log_entry = {
                "sessionId": "debug-session",
                "runId": "prediction-debug",
                "hypothesisId": hypothesisId,
                "location": location,
                "message": message,
                "data": data,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json_lib.dumps(log_entry) + "\n")
        except Exception:
            pass
    # #endregion
    
    for game in games:
        iso = game.get("date")
        if not iso:
            continue
        try:
            dt = parse_iso_datetime(iso).astimezone(CST)
        except ValueError:
            continue
        if dt.date() != target_date:
            continue
        status = (game.get("status") or {}).get("short")
        if status not in {"FT", "AOT"}:
            continue

        teams = game.get("teams") or {}
        home_name = (teams.get("home") or {}).get("name")
        away_name = (teams.get("away") or {}).get("name")
        # #region agent log
        _debug_log(
            "review_predictions.py:130",
            "Game loaded from API-Basketball",
            {
                "target_date": target_date.isoformat(),
                "game_date": dt.date().isoformat(),
                "home_team_api": home_name,
                "away_team_api": away_name,
                "status": status
            },
            hypothesisId="C"
        )
        # #endregion
        if not home_name or not away_name:
            continue

        scores = game.get("scores") or {}
        home_scores = scores.get("home") or {}
        away_scores = scores.get("away") or {}

        home_total = home_scores.get("total")
        away_total = away_scores.get("total")
        if home_total is None or away_total is None:
            continue

        home_first_half = sum_period(
            [
                home_scores.get("quarter_1"),
                home_scores.get("quarter_2"),
            ]
        )
        away_first_half = sum_period(
            [
                away_scores.get("quarter_1"),
                away_scores.get("quarter_2"),
            ]
        )

        home_id = normalize_team_name(home_name)
        away_id = normalize_team_name(away_name)
        scoreboard[(home_id, away_id)] = {
            "home_team": home_name,
            "away_team": away_name,
            "home_id": home_id,
            "away_id": away_id,
            "home_score": int(home_total),
            "away_score": int(away_total),
            "home_first_half": int(home_first_half),
            "away_first_half": int(away_first_half),
        }
    return scoreboard


def find_game(
    scoreboard: Dict[Tuple[str, str], Dict[str, Any]],
    home_team: str,
    away_team: str,
) -> Optional[Dict[str, Any]]:
    key = (
        normalize_team_name(home_team),
        normalize_team_name(away_team),
    )
    if key in scoreboard:
        return scoreboard[key]

    # Fallback fuzzy match
    for entry in scoreboard.values():
        if are_same_team(entry["home_team"], home_team) and are_same_team(
            entry["away_team"], away_team
        ):
            return entry
    return None


def american_profit(odds: Optional[int]) -> float:
    if odds is None:
        return 0.0
    if odds > 0:
        return odds / 100.0
    return 100.0 / abs(odds)


def evaluate_spread(
    actual_margin: float,
    line: Optional[float],
    pick: Optional[str],
    home_team: str,
    away_team: str,
) -> Optional[str]:
    """
    Evaluate if a spread pick won or lost.
    
    Args:
        actual_margin: home_score - away_score
        line: The spread line (from home team's perspective)
            - Negative means home is favored (e.g., -3.5 = home favored by 3.5)
            - Positive means home is underdog (e.g., +3.5 = home gets 3.5 points)
        pick: The team that was picked to cover
        home_team: Home team name
        away_team: Away team name
    
    Returns:
        "win", "loss", "push", or None
    """
    if line is None or not pick:
        return None
    
    # Check for push: actual_margin exactly equals the line (from home perspective)
    # If line = -3.5, push if actual_margin = 3.5 (home wins by exactly 3.5)
    # If line = +3.5, push if actual_margin = -3.5 (home loses by exactly 3.5)
    if math.isclose(actual_margin, -line, abs_tol=0.05):
        return "push"
    
    # If we picked home team:
    # - Home covers if actual_margin > -line
    # - Example: line = -3.5, home covers if actual_margin > 3.5
    # - Example: line = +3.5, home covers if actual_margin > -3.5 (home can lose by up to 3.5)
    if are_same_team(pick, home_team):
        return "win" if actual_margin > -line else "loss"
    
    # If we picked away team:
    # - Away covers if actual_margin < -line (opposite of home)
    # - Example: line = -3.5, away covers if actual_margin < 3.5 (home doesn't win by more than 3.5)
    # - Example: line = +3.5, away covers if actual_margin < -3.5 (away wins by more than 3.5)
    if are_same_team(pick, away_team):
        return "win" if actual_margin < -line else "loss"
    
    return None


def evaluate_total(
    total: float,
    line: Optional[float],
    pick: Optional[str],
) -> Optional[str]:
    if line is None or not pick:
        return None
    if math.isclose(total, line, abs_tol=0.05):
        return "push"
    pick_upper = pick.upper()
    if pick_upper == "OVER":
        return "win" if total > line else "loss"
    if pick_upper == "UNDER":
        return "win" if total < line else "loss"
    return None


def evaluate_moneyline(
    home_score: int,
    away_score: int,
    pick: Optional[str],
    home_team: str,
    away_team: str,
) -> Optional[str]:
    if not pick:
        return None
    if home_score > away_score and are_same_team(pick, home_team):
        return "win"
    if away_score > home_score and are_same_team(pick, away_team):
        return "win"
    if home_score == away_score:
        return None
    return "loss"


def record_outcome(
    metrics: Dict[str, Dict[str, float]],
    category: str,
    result: Optional[str],
    odds: Optional[int] = None,
) -> None:
    if result is None:
        return
    stats = metrics.setdefault(
        category, {"picks": 0, "wins": 0, "losses": 0, "pushes": 0, "roi": 0.0}
    )
    stats["picks"] += 1
    if result == "win":
        stats["wins"] += 1
        stats["roi"] += american_profit(odds) if odds is not None else 0.0
    elif result == "loss":
        stats["losses"] += 1
        stats["roi"] -= 1.0
    elif result == "push":
        stats["pushes"] += 1


def summarize(metrics: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
    summary = []
    for category, stats in metrics.items():
        picks = stats["picks"]
        wins = stats["wins"]
        losses = stats["losses"]
        pushes = stats["pushes"]
        roi = stats["roi"]
        hit_rate = (wins / picks * 100) if picks else 0.0
        summary.append(
            {
                "category": category,
                "picks": picks,
                "wins": wins,
                "losses": losses,
                "pushes": pushes,
                "hit_rate": round(hit_rate, 1),
                "roi": round(roi, 3),
            }
        )
    summary.sort(key=lambda row: row["category"])
    return summary


def format_markdown(
    target_date: date,
    summary: List[Dict[str, Any]],
    games: List[Dict[str, Any]],
) -> str:
    lines = [
        f"# Slate Review — {target_date.isoformat()}",
        "",
        "## Summary",
        "",
        "| Category | Picks | Wins | Losses | Pushes | Hit Rate | ROI |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary:
        lines.append(
            f"| {row['category']} | {row['picks']} | {row['wins']} | "
            f"{row['losses']} | {row['pushes']} | {row['hit_rate']:.1f}% | "
            f"{row['roi']:+.2f}u |"
        )
    lines.append("")
    lines.append("## Games")
    lines.append("")
    for game in games:
        lines.append(f"### {game['game']}")
        lines.append(
            f"- Final: {game['actual_score']} | 1H: {game['first_half_score']}"
        )
        for pick in game["picks"]:
            detail = f"  - {pick['category']}: {pick.get('pick') or '—'}"
            if pick.get("line") is not None:
                detail += f" (line {pick['line']})"
            if pick.get("odds") is not None:
                detail += f" @ {pick['odds']}"
            if pick.get("result"):
                detail += f" → {pick['result'].upper()}"
            lines.append(detail)
        lines.append("")
    return "\n".join(lines)


def review_slate(target_date: date, analysis: List[Dict[str, Any]]) -> Dict[str, Any]:
    scoreboard = build_scoreboard_map(target_date)
    if not scoreboard:
        raise RuntimeError(f"No completed games found for {target_date.isoformat()}.")

    metrics: Dict[str, Dict[str, float]] = {}
    games_output: List[Dict[str, Any]] = []

    for game in analysis:
        home_team = game.get("home_team")
        away_team = game.get("away_team")
        comp_edge = game.get("comprehensive_edge") or {}
        fg = comp_edge.get("full_game", {})
        fh = comp_edge.get("first_half", {})

        # #region agent log
        DEBUG_LOG_PATH = Path(__file__).parent.parent / ".cursor" / "debug.log"
        def _debug_log(location: str, message: str, data: dict, hypothesisId: str = "C"):
            try:
                import json as json_lib
                from datetime import datetime, timezone
                log_entry = {
                    "sessionId": "debug-session",
                    "runId": "prediction-debug",
                    "hypothesisId": hypothesisId,
                    "location": location,
                    "message": message,
                    "data": data,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
                    f.write(json_lib.dumps(log_entry) + "\n")
            except Exception:
                pass
        # #endregion
        
        actual = find_game(scoreboard, home_team, away_team)
        # #region agent log
        _debug_log(
            "review_predictions.py:411",
            "Team name matching for game review",
            {
                "analysis_home_team": home_team,
                "analysis_away_team": away_team,
                "normalized_home": normalize_team_name(home_team),
                "normalized_away": normalize_team_name(away_team),
                "match_found": actual is not None,
                "scoreboard_keys": list(scoreboard.keys())[:5] if scoreboard else []
            },
            hypothesisId="C"
        )
        # #endregion
        if actual is None:
            print(f"⚠️  Missing scoreboard match for {away_team} @ {home_team}")
            continue

        actual_margin = actual["home_score"] - actual["away_score"]
        actual_total = actual["home_score"] + actual["away_score"]
        actual_first_half_total = actual["home_first_half"] + actual["away_first_half"]

        entry = {
            "game": f"{away_team} @ {home_team}",
            "actual_score": f"{actual['home_score']} - {actual['away_score']}",
            "first_half_score": f"{actual['home_first_half']} - {actual['away_first_half']}",
            "picks": [],
        }

        fg_spread = fg.get("spread") or {}
        pick_line = fg_spread.get("pick_line") or fg_spread.get("market_line")
        pick = fg_spread.get("pick")
        spread_result = evaluate_spread(
            actual_margin,
            pick_line,
            pick,
            home_team,
            away_team,
        )
        # #region agent log
        _debug_log(
            "review_predictions.py:363",
            "FG Spread evaluation",
            {
                "home_team": home_team,
                "away_team": away_team,
                "actual_margin": actual_margin,
                "actual_home_score": actual["home_score"],
                "actual_away_score": actual["away_score"],
                "pick_line": pick_line,
                "pick": pick,
                "result": spread_result,
                "model_margin": fg_spread.get("model_margin"),
                "edge": fg_spread.get("edge"),
                "market_line": fg_spread.get("market_line"),
                "evaluation_logic": {
                    "picked_home": pick == home_team if pick else None,
                    "line_used": pick_line,
                    "check_condition": f"actual_margin >= {-pick_line if pick_line else 0}" if pick == home_team else f"actual_margin <= {-pick_line if pick_line else 0}" if pick == away_team else None
                }
            }
        )
        # #endregion
        record_outcome(metrics, "FG Spread", spread_result, fg_spread.get("pick_odds"))
        entry["picks"].append(
            {
                "category": "FG Spread",
                "pick": fg_spread.get("pick"),
                "line": fg_spread.get("pick_line") or fg_spread.get("market_line"),
                "odds": fg_spread.get("pick_odds"),
                "result": spread_result,
            }
        )

        fg_total = fg.get("total") or {}
        total_result = evaluate_total(
            actual_total,
            fg_total.get("pick_line") or fg_total.get("market_line"),
            fg_total.get("pick"),
        )
        record_outcome(metrics, "FG Total", total_result, fg_total.get("pick_odds"))
        entry["picks"].append(
            {
                "category": "FG Total",
                "pick": fg_total.get("pick"),
                "line": fg_total.get("pick_line") or fg_total.get("market_line"),
                "odds": fg_total.get("pick_odds"),
                "result": total_result,
            }
        )

        fg_moneyline = fg.get("moneyline") or {}
        ml_result = evaluate_moneyline(
            actual["home_score"],
            actual["away_score"],
            fg_moneyline.get("pick"),
            home_team,
            away_team,
        )
        chosen_odds = None
        if ml_result and fg_moneyline.get("pick"):
            if are_same_team(fg_moneyline["pick"], home_team):
                chosen_odds = fg_moneyline.get("market_home_odds")
            elif are_same_team(fg_moneyline["pick"], away_team):
                chosen_odds = fg_moneyline.get("market_away_odds")
        record_outcome(metrics, "FG Moneyline", ml_result, chosen_odds)
        entry["picks"].append(
            {
                "category": "FG Moneyline",
                "pick": fg_moneyline.get("pick"),
                "odds": chosen_odds,
                "result": ml_result,
            }
        )

        fh_spread = fh.get("spread") or {}
        fh_spread_result = evaluate_spread(
            actual["home_first_half"] - actual["away_first_half"],
            fh_spread.get("pick_line") or fh_spread.get("market_line"),
            fh_spread.get("pick"),
            home_team,
            away_team,
        )
        record_outcome(metrics, "1H Spread", fh_spread_result, fh_spread.get("pick_odds"))
        entry["picks"].append(
            {
                "category": "1H Spread",
                "pick": fh_spread.get("pick"),
                "line": fh_spread.get("pick_line") or fh_spread.get("market_line"),
                "odds": fh_spread.get("pick_odds"),
                "result": fh_spread_result,
            }
        )

        fh_total = fh.get("total") or {}
        fh_total_line = fh_total.get("pick_line") or fh_total.get("market_line")
        fh_total_result = evaluate_total(
            actual_first_half_total,
            fh_total_line,
            fh_total.get("pick"),
        )
        entry["picks"].append(
            {
                "category": "1H Total",
                "pick": fh_total.get("pick"),
                "line": fh_total_line,
                "odds": fh_total.get("pick_odds"),
                "result": fh_total_result,
            }
        )
        record_outcome(metrics, "1H Total", fh_total_result, fh_total.get("pick_odds"))

        fh_moneyline = fh.get("moneyline") or {}
        fh_ml_result = evaluate_moneyline(
            actual["home_first_half"],
            actual["away_first_half"],
            fh_moneyline.get("pick"),
            home_team,
            away_team,
        )
        record_outcome(metrics, "1H Moneyline", fh_ml_result, fh_moneyline.get("pick_odds"))
        entry["picks"].append(
            {
                "category": "1H Moneyline",
                "pick": fh_moneyline.get("pick"),
                "odds": fh_moneyline.get("pick_odds"),
                "result": fh_ml_result,
            }
        )

        games_output.append(entry)

    summary = summarize(metrics)
    return {
        "summary": summary,
        "games": games_output,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Review slate performance vs API-Basketball results.")
    parser.add_argument("--date", help="Target date (YYYY-MM-DD, 'today', 'yesterday')", default=None)
    parser.add_argument("--analysis", help="Path to slate analysis JSON")
    parser.add_argument("--output-json", help="Optional JSON output path")
    parser.add_argument("--output-md", help="Optional Markdown output path")
    args = parser.parse_args()

    target_date = get_target_date(args.date)
    default_analysis = (
        Path(settings.data_processed_dir)
        / f"slate_analysis_{target_date.strftime('%Y%m%d')}.json"
    )
    analysis_path = Path(args.analysis) if args.analysis else default_analysis
    analysis = load_analysis(analysis_path)

    print(f"Reviewing slate for {target_date.isoformat()} using {analysis_path.name}")
    results = review_slate(target_date, analysis)

    output_data = {
        "date": target_date.isoformat(),
        **results,
    }

    json_path = Path(args.output_json) if args.output_json else (
        Path(settings.data_processed_dir)
        / f"pick_review_{target_date.strftime('%Y%m%d')}.json"
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(output_data, fh, indent=2)
    print(f"Saved JSON summary to {json_path}")

    md_path = Path(args.output_md) if args.output_md else (
        Path(settings.data_processed_dir)
        / f"pick_review_{target_date.strftime('%Y%m%d')}.md"
    )
    md_path.parent.mkdir(parents=True, exist_ok=True)
    markdown = format_markdown(target_date, results["summary"], results["games"])
    with md_path.open("w", encoding="utf-8") as fh:
        fh.write(markdown)
    print(f"Saved Markdown summary to {md_path}")


if __name__ == "__main__":
    main()

