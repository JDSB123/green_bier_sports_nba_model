#!/usr/bin/env python3
"""
Backfill ESPN box score + player-derived features into training data.

Adds leakage-safe, pre-game features by using ONLY games prior to the target game.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.config import settings
from src.ingestion.espn import fetch_espn_schedule, fetch_espn_box_score
from src.ingestion.standardize import normalize_team_to_espn


def _parse_minutes(value: str) -> float:
    if not value or value in ("--", "DNP"):
        return 0.0
    if ":" in value:
        mins, secs = value.split(":", 1)
        try:
            return float(mins) + float(secs) / 60.0
        except Exception:
            return 0.0
    try:
        return float(value)
    except Exception:
        return 0.0


def _team_key(name: str) -> str:
    norm, _ = normalize_team_to_espn(name, source="espn_backfill")
    return norm


def _cache_path(cache_dir: Path, game_id: str) -> Path:
    return cache_dir / f"{game_id}.json"


async def _build_schedule_map(dates: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Build mapping keyed by ESPN game_id with date and teams.

    Returns:
        {game_id: {"date": "YYYY-MM-DD", "away_team": "...", "home_team": "..."}}
    """
    game_map: Dict[str, Dict[str, Any]] = {}
    for date_str in dates:
        try:
            games = await fetch_espn_schedule(dates=[date_str])
        except Exception:
            continue
        for game in games:
            away = _team_key(game.away_team)
            home = _team_key(game.home_team)
            game_id = str(game.game_id)
            game_map[game_id] = {
                "date": datetime.strptime(date_str, "%Y%m%d").date().isoformat(),
                "away_team": away,
                "home_team": home,
            }
    return game_map


async def _fetch_game_stats(
    game_id: str,
    meta: Dict[str, Any],
    cache_dir: Path,
    sem: asyncio.Semaphore,
) -> Optional[Dict[str, Any]]:
    cache_file = _cache_path(cache_dir, game_id)
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text())
        except Exception:
            pass

    async with sem:
        try:
            box = await fetch_espn_box_score(game_id)
        except Exception:
            return None

        team_stats: Dict[str, Dict[str, float]] = {}
        for team_box in box.get("teams", []):
            team_name = _team_key(team_box.team_name)
            fg_pct = (team_box.fg_made / team_box.fg_attempts * 100.0) if team_box.fg_attempts > 0 else 0.0
            three_pct = (team_box.three_made / team_box.three_attempts * 100.0) if team_box.three_attempts > 0 else 0.0
            ft_pct = (team_box.ft_made / team_box.ft_attempts * 100.0) if team_box.ft_attempts > 0 else 0.0
            efg_pct = ((team_box.fg_made + 0.5 * team_box.three_made) / team_box.fg_attempts * 100.0) if team_box.fg_attempts > 0 else 0.0

            team_stats[team_name] = {
                "fg_pct": fg_pct,
                "three_pct": three_pct,
                "ft_pct": ft_pct,
                "efg_pct": efg_pct,
                "rebounds_total": float(team_box.rebounds_total),
                "rebounds_off": float(team_box.rebounds_off),
                "rebounds_def": float(team_box.rebounds_def),
                "assists": float(team_box.assists),
                "steals": float(team_box.steals),
                "blocks": float(team_box.blocks),
                "turnovers": float(team_box.turnovers),
                "personal_fouls": float(team_box.personal_fouls),
                "points": float(team_box.points),
            }

        player_groups: Dict[str, List[Any]] = defaultdict(list)
        for player in box.get("players", []):
            team_name = _team_key(player.team_name)
            player_groups[team_name].append(player)

        for team_name, players in player_groups.items():
            minutes = sorted(
                [(_parse_minutes(p.minutes), p) for p in players],
                key=lambda x: x[0],
                reverse=True,
            )
            top2 = minutes[:2]
            active_stars = sum(1 for mins, _ in top2 if mins > 20)
            star_avail = active_stars * 0.5

            bench_points = sum(
                float(p.points) for p in players if not getattr(p, "starter", False)
            )

            paint_defense = 0.0
            stats = team_stats.get(team_name)
            if stats:
                paint_defense = stats["blocks"] * 2.0 + stats["rebounds_def"] * 0.5

            if team_name not in team_stats:
                team_stats[team_name] = {}

            team_stats[team_name].update(
                {
                    "star_avail": star_avail,
                    "bench_scoring": bench_points,
                    "paint_defense": paint_defense,
                }
            )

        payload = {
            "game_id": game_id,
            "date": meta["date"],
            "teams": team_stats,
        }

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(payload))
        return payload


def _rolling_mean(series: pd.Series, window: int) -> pd.Series:
    return (
        series
        .rolling(window=window, min_periods=1)
        .mean()
        .shift(1)
    )


def _prepare_team_feature_df(records: List[Dict[str, Any]], window: int) -> pd.DataFrame:
    rows = []
    for rec in records:
        game_date = rec["date"]
        for team, stats in rec["teams"].items():
            if not stats:
                continue
            row = {"team": team, "game_date": game_date}
            row.update(stats)
            rows.append(row)

    team_df = pd.DataFrame(rows)
    if team_df.empty:
        return team_df

    team_df["game_date"] = pd.to_datetime(team_df["game_date"], errors="coerce")
    team_df = team_df.sort_values(["team", "game_date"]).reset_index(drop=True)

    rolling_stats = [
        "fg_pct", "three_pct", "ft_pct", "efg_pct",
        "rebounds_total", "rebounds_off", "rebounds_def",
        "assists", "steals", "blocks", "turnovers",
        "personal_fouls",
    ]

    for stat in rolling_stats:
        if stat in team_df.columns:
            team_df[f"{stat}_avg"] = (
                team_df.groupby("team")[stat]
                .apply(lambda s: _rolling_mean(s, window))
                .reset_index(level=0, drop=True)
            )

    # Previous-game values for player-derived metrics
    for stat in ["star_avail", "bench_scoring", "paint_defense"]:
        if stat in team_df.columns:
            team_df[f"{stat}_prev"] = (
                team_df.groupby("team")[stat]
                .shift(1)
            )

    return team_df


def _merge_features(train_df: pd.DataFrame, team_df: pd.DataFrame) -> pd.DataFrame:
    train_df = train_df.copy()
    train_df["game_date"] = pd.to_datetime(train_df["game_date"], errors="coerce")
    train_df["home_team_norm"] = train_df["home_team"].apply(_team_key)
    train_df["away_team_norm"] = train_df["away_team"].apply(_team_key)

    team_df = team_df.copy()
    team_df["game_date"] = pd.to_datetime(team_df["game_date"], errors="coerce")

    home_map = team_df.rename(
        columns={
            "team": "home_team_norm",
            "fg_pct_avg": "home_fg_pct",
            "three_pct_avg": "home_three_pct",
            "ft_pct_avg": "home_ft_pct",
            "efg_pct_avg": "home_efg_pct",
            "rebounds_total_avg": "home_rebounds",
            "rebounds_off_avg": "home_oreb",
            "rebounds_def_avg": "home_dreb",
            "assists_avg": "home_assists",
            "steals_avg": "home_steals",
            "blocks_avg": "home_blocks",
            "turnovers_avg": "home_turnovers",
            "personal_fouls_avg": "home_fouls",
            "star_avail_prev": "home_star_avail",
            "bench_scoring_prev": "home_bench_scoring",
            "paint_defense_prev": "home_paint_defense",
        }
    )

    away_map = team_df.rename(
        columns={
            "team": "away_team_norm",
            "fg_pct_avg": "away_fg_pct",
            "three_pct_avg": "away_three_pct",
            "ft_pct_avg": "away_ft_pct",
            "efg_pct_avg": "away_efg_pct",
            "rebounds_total_avg": "away_rebounds",
            "rebounds_off_avg": "away_oreb",
            "rebounds_def_avg": "away_dreb",
            "assists_avg": "away_assists",
            "steals_avg": "away_steals",
            "blocks_avg": "away_blocks",
            "turnovers_avg": "away_turnovers",
            "personal_fouls_avg": "away_fouls",
            "star_avail_prev": "away_star_avail",
            "bench_scoring_prev": "away_bench_scoring",
            "paint_defense_prev": "away_paint_defense",
        }
    )

    merged = train_df.merge(
        home_map,
        on=["home_team_norm", "game_date"],
        how="left",
    ).merge(
        away_map,
        on=["away_team_norm", "game_date"],
        how="left",
        suffixes=("", "_awaydup"),
    )

    # Diff features
    diff_pairs = [
        ("fg_pct", "home_fg_pct", "away_fg_pct"),
        ("three_pct", "home_three_pct", "away_three_pct"),
        ("ft_pct", "home_ft_pct", "away_ft_pct"),
        ("efg_pct", "home_efg_pct", "away_efg_pct"),
        ("rebounds", "home_rebounds", "away_rebounds"),
        ("oreb", "home_oreb", "away_oreb"),
        ("assists", "home_assists", "away_assists"),
        ("turnovers", "home_turnovers", "away_turnovers"),
        ("steals", "home_steals", "away_steals"),
        ("blocks", "home_blocks", "away_blocks"),
        ("fouls", "home_fouls", "away_fouls"),
    ]
    for diff_name, home_col, away_col in diff_pairs:
        if home_col in merged.columns and away_col in merged.columns:
            merged[f"{diff_name}_diff"] = merged[home_col] - merged[away_col]

    merged = merged.drop(columns=["home_team_norm", "away_team_norm"])
    return merged


async def main_async(args) -> int:
    training_path = Path(args.training_file)
    if not training_path.exists():
        raise SystemExit(f"Training file not found: {training_path}")

    df = pd.read_csv(training_path, low_memory=False)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    if args.start_date:
        df = df[df["game_date"] >= pd.to_datetime(args.start_date)].copy()
    if args.end_date:
        df = df[df["game_date"] <= pd.to_datetime(args.end_date)].copy()

    if df.empty:
        print("[WARN] No games after date filtering.")
        return 0

    dates = sorted({d.strftime("%Y%m%d") for d in df["game_date"].dropna()})
    schedule_map = await _build_schedule_map(dates)

    sem = asyncio.Semaphore(int(args.concurrency))
    cache_dir = Path(args.cache_dir)
    tasks = []
    for game_id, meta in schedule_map.items():
        tasks.append(_fetch_game_stats(game_id, meta, cache_dir, sem))

    records: List[Dict[str, Any]] = []
    for coro in asyncio.as_completed(tasks):
        rec = await coro
        if rec:
            records.append(rec)

    team_df = _prepare_team_feature_df(records, window=int(args.window))
    if team_df.empty:
        print("[WARN] No ESPN records loaded.")
        return 0

    merged = _merge_features(df, team_df)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"[OK] Wrote backfilled training data to {output_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill ESPN box score features into training data",
    )
    parser.add_argument(
        "--training-file",
        type=str,
        default=str(Path(settings.data_processed_dir) / "training_data.csv"),
        help="Path to training data CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(settings.data_processed_dir) / "training_data.csv"),
        help="Output path (defaults to overwrite training data)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD) filter",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD) filter",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=10,
        help="Rolling window size for box score averages (default: 10)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Concurrency for ESPN box score fetches",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=str(Path(settings.data_raw_dir) / "espn_boxscores"),
        help="Cache directory for ESPN box scores",
    )
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
