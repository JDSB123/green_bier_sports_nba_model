#!/usr/bin/env python3
"""
Scheduled Teams poster.

Run on a frequent cadence (ex: every 5 minutes). The script will only post
when the current time is within the posting window, using the NBA API slate
for schedule times:
- Starts 1 hour before the first game of the day
- Posts every hour on that same minute until the last game start

Example (first game 6:30 PM CST): posts at 5:30, 6:30, 7:30, ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from urllib.request import Request, urlopen

from scripts.post_to_teams import fetch_executive_data, format_teams_message, post_to_teams

try:
    from azure.storage.blob import BlobServiceClient
except Exception:  # pragma: no cover - optional dependency for local runs
    BlobServiceClient = None

CST = ZoneInfo("America/Chicago")


def _parse_iso(dt_str: str | None) -> datetime | None:
    if not dt_str:
        return None
    try:
        if dt_str.endswith("Z"):
            dt_str = f"{dt_str[:-1]}+00:00"
        dt = datetime.fromisoformat(dt_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _parse_game_cst(game: dict) -> datetime | None:
    """Return game start time in CST (timezone-aware) or None."""
    commence_time_cst = game.get("commence_time_cst")
    if commence_time_cst:
        dt = _parse_iso(commence_time_cst)
        if dt:
            return dt.astimezone(CST)

    commence_time = game.get("commence_time")
    if commence_time:
        dt = _parse_iso(commence_time)
        if dt:
            return dt.astimezone(CST)

    return None


def _should_post(now: datetime, window_start: datetime, window_end: datetime, interval_minutes: int, tolerance_minutes: int) -> bool:
    if now < window_start or now > window_end:
        return False

    interval_seconds = max(interval_minutes, 1) * 60
    tolerance_seconds = max(tolerance_minutes, 0) * 60
    elapsed = (now - window_start).total_seconds()
    offset = elapsed % interval_seconds
    return offset <= tolerance_seconds or offset >= (interval_seconds - tolerance_seconds)


def _ceil_to_hour(dt: datetime) -> datetime:
    """Ceil to the next top-of-hour (or keep if already on the hour)."""
    if dt.minute == 0 and dt.second == 0 and dt.microsecond == 0:
        return dt
    return (dt + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)


def _resolve_date(date_str: str) -> datetime.date:
    now_cst = datetime.now(CST)
    if date_str is None or date_str.lower() in {"today", ""}:
        return now_cst.date()
    if date_str.lower() == "tomorrow":
        return (now_cst + timedelta(days=1)).date()
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def _fetch_slate(date_label: str, api_base: str) -> dict:
    url = f"{api_base}/slate/{date_label}"
    req = Request(url, headers={"Accept": "application/json"})
    with urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


def _get_game_window(date_label: str, api_base: str) -> tuple[datetime, datetime] | None:
    slate = _fetch_slate(date_label, api_base)
    games = slate.get("predictions", []) or []
    if not games:
        return None

    game_times = []
    for game in games:
        dt = _parse_game_cst(game)
        if dt is not None:
            game_times.append(dt)

    if not game_times:
        return None

    return min(game_times), max(game_times)


def _extract_teams_from_matchup(matchup: str) -> tuple[str | None, str | None]:
    if not matchup or " @ " not in matchup:
        return None, None
    away, home = matchup.split(" @ ", 1)
    away = away.split("(")[0].strip()
    home = home.split("(")[0].strip()
    return away or None, home or None


def _normalize_team(name: str | None) -> str:
    if not name:
        return ""
    return " ".join(name.lower().replace(".", "").split())


def _game_key(away: str | None, home: str | None) -> str:
    return f"{_normalize_team(away)}@{_normalize_team(home)}"


def _play_key(play: dict) -> str:
    away, home = _extract_teams_from_matchup(play.get("matchup", ""))
    return f"{_game_key(away, home)}|{play.get('period')}|{play.get('market')}"


def _snapshot_blob_name(prefix: str, date_label: str) -> str:
    return f"{prefix.rstrip('/')}/{date_label}/latest.json"


def _load_snapshot(date_label: str) -> dict | None:
    prefix = os.getenv("TEAMS_SNAPSHOT_PREFIX", "teams_snapshots")
    container = os.getenv("TEAMS_SNAPSHOT_CONTAINER", "predictions")
    conn = os.getenv("TEAMS_STORAGE_CONNECTION_STRING") or os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if conn and not BlobServiceClient:
        raise RuntimeError("azure-storage-blob is required when AZURE_STORAGE_CONNECTION_STRING is set")
    if conn and BlobServiceClient:
        try:
            client = BlobServiceClient.from_connection_string(conn)
            blob = client.get_blob_client(container=container, blob=_snapshot_blob_name(prefix, date_label))
            data = blob.download_blob().readall()
            return json.loads(data.decode("utf-8"))
        except Exception:
            return None

    # Local fallback (primarily for dev)
    local_path = f"/tmp/teams_snapshot_{date_label}.json"
    if os.path.exists(local_path):
        try:
            with open(local_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _save_snapshot(date_label: str, snapshot: dict) -> None:
    prefix = os.getenv("TEAMS_SNAPSHOT_PREFIX", "teams_snapshots")
    container = os.getenv("TEAMS_SNAPSHOT_CONTAINER", "predictions")
    conn = os.getenv("TEAMS_STORAGE_CONNECTION_STRING") or os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    payload = json.dumps(snapshot).encode("utf-8")
    if conn and not BlobServiceClient:
        raise RuntimeError("azure-storage-blob is required when AZURE_STORAGE_CONNECTION_STRING is set")
    if conn and BlobServiceClient:
        client = BlobServiceClient.from_connection_string(conn)
        blob = client.get_blob_client(container=container, blob=_snapshot_blob_name(prefix, date_label))
        blob.upload_blob(payload, overwrite=True)
        return

    local_path = f"/tmp/teams_snapshot_{date_label}.json"
    with open(local_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)


def _lock_started_games(
    current: dict,
    previous: dict | None,
    started_keys: set[str],
    now_cst: datetime,
) -> dict:
    plays = current.get("plays", []) or []
    prev_map = {}
    prev_snapshot_at = None
    if isinstance(previous, dict):
        prev_snapshot_at = previous.get("snapshot_at_cst") or previous.get("posted_at_cst")
        for play in previous.get("plays", []) or []:
            prev_map[_play_key(play)] = play

    locked_count = 0
    final_plays = []
    for play in plays:
        key = _play_key(play)
        game_key = key.split("|")[0]
        if game_key in started_keys:
            locked = prev_map.get(key)
            if locked:
                locked_play = dict(locked)
                locked_play["locked"] = True
                locked_play["locked_snapshot_at_cst"] = prev_snapshot_at
                final_plays.append(locked_play)
            else:
                play["locked"] = True
                play["locked_snapshot_at_cst"] = prev_snapshot_at or now_cst.strftime("%Y-%m-%d %I:%M %p CST")
                final_plays.append(play)
            locked_count += 1
        else:
            play["locked"] = False
            final_plays.append(play)

    updated = dict(current)
    updated["plays"] = final_plays
    updated["snapshot_at_cst"] = now_cst.strftime("%Y-%m-%d %I:%M %p CST")
    if locked_count > 0 and prev_snapshot_at:
        updated["locked_snapshot_at_cst"] = prev_snapshot_at
    updated["locked_games_count"] = locked_count
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description="Post Teams picks on a schedule window")
    parser.add_argument("--date", default="today", help="Date (YYYY-MM-DD or 'today'/'tomorrow')")
    parser.add_argument("--lead-minutes", type=int, default=int(os.getenv("TEAMS_SCHEDULE_LEAD_MINUTES", "60")))
    parser.add_argument("--interval-minutes", type=int, default=int(os.getenv("TEAMS_SCHEDULE_INTERVAL_MINUTES", "60")))
    parser.add_argument("--tolerance-minutes", type=int, default=int(os.getenv("TEAMS_SCHEDULE_TOLERANCE_MINUTES", "2")))
    parser.add_argument("--force", action="store_true", help="Post regardless of window/timing checks")
    parser.add_argument("--dry-run", action="store_true", help="Compute window but do not post")
    args = parser.parse_args()

    api_base = os.getenv("NBA_API_URL", "").strip()
    if not api_base:
        print("[ERROR] NBA_API_URL environment variable is required")
        return 2
    if not os.getenv("TEAMS_WEBHOOK_URL", "").strip():
        print("[ERROR] TEAMS_WEBHOOK_URL environment variable is required")
        return 2

    now = datetime.now(CST)
    try:
        target_date = _resolve_date(args.date)
    except ValueError:
        print("[ERROR] Invalid date format. Use YYYY-MM-DD, 'today', or 'tomorrow'.")
        return 2

    date_label = target_date.isoformat()
    window = _get_game_window(date_label, api_base)
    if window is None:
        print(f"[INFO] No scheduled games for {target_date} (CST). Skipping.")
        return 0

    first_game, last_game = window
    window_start = first_game - timedelta(minutes=max(args.lead_minutes, 0))
    window_end = last_game
    aligned_start = window_start
    if args.interval_minutes >= 60:
        aligned_start = _ceil_to_hour(window_start)

    print(f"[WINDOW] First game: {first_game.strftime('%Y-%m-%d %I:%M %p CST')}")
    print(f"[WINDOW] Last game:  {last_game.strftime('%Y-%m-%d %I:%M %p CST')}")
    print(f"[WINDOW] Start:      {window_start.strftime('%Y-%m-%d %I:%M %p CST')}")
    if aligned_start != window_start:
        print(f"[WINDOW] Aligned:    {aligned_start.strftime('%Y-%m-%d %I:%M %p CST')}")
    print(f"[WINDOW] End:        {window_end.strftime('%Y-%m-%d %I:%M %p CST')}")
    print(f"[NOW]    {now.strftime('%Y-%m-%d %I:%M %p CST')}")

    if args.dry_run:
        return 0

    if not args.force:
        on_schedule = _should_post(now, aligned_start, window_end, args.interval_minutes, args.tolerance_minutes)
        if not on_schedule:
            print("[SKIP] Outside posting window or not on cadence.")
            return 0

    allow_empty = os.getenv("TEAMS_ALLOW_EMPTY", "true").strip().lower() in {"1", "true", "yes", "y", "on"}

    # Fetch latest executive summary + previous snapshot for lock-in
    try:
        current = fetch_executive_data(date_label, api_base)
    except Exception as exc:
        print(f"[ERROR] Failed to fetch executive summary: {exc}")
        return 1

    previous = _load_snapshot(date_label)

    # Build started game key set from slate times
    try:
        slate = _fetch_slate(date_label, api_base)
    except Exception as exc:
        print(f"[ERROR] Failed to fetch slate for timing: {exc}")
        return 1

    started_keys: set[str] = set()
    for game in slate.get("predictions", []) or []:
        away = game.get("away_team")
        home = game.get("home_team")
        gkey = _game_key(away, home)
        start_dt = _parse_game_cst(game)
        if start_dt and start_dt <= now:
            started_keys.add(gkey)

    data = _lock_started_games(current, previous, started_keys, now)
    plays = data.get("plays", []) or []
    if not plays and not allow_empty:
        print("[INFO] No plays available and TEAMS_ALLOW_EMPTY=false. Skipping.")
        return 0

    message = format_teams_message(data)
    if post_to_teams(message):
        snapshot = {
            "date": date_label,
            "snapshot_at_cst": data.get("snapshot_at_cst"),
            "posted_at_cst": now.strftime("%Y-%m-%d %I:%M %p CST"),
            "plays": data.get("plays", []),
        }
        try:
            _save_snapshot(date_label, snapshot)
        except Exception as exc:
            print(f"[WARN] Failed to persist snapshot: {exc}")
        print(f"[OK] Posted Teams update for {date_label} ({len(plays)} plays)")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
