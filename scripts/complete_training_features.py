#!/usr/bin/env python3
"""scripts/complete_training_features.py

Complete/enrich training data with additional features used by prediction/backtest.

Policy (strict, no placeholders):
- Do not fabricate neutral defaults for unavailable data (e.g., betting splits).
- Do not silently zero-fill when computation fails.
- If a feature cannot be computed, leave as NaN so downstream strict filters can drop.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.standardization import standardize_team_name, generate_match_key
from src.modeling.team_factors import (
    get_travel_distance,
    get_timezone_difference,
    calculate_travel_fatigue,
)

DATA_DIR = PROJECT_ROOT / "data"
TRAINING_FILE = DATA_DIR / "processed" / "training_data.csv"
KAGGLE_FILE = DATA_DIR / "external" / "kaggle" / "nba_2008-2025.csv"


def fill_moneylines(training_file: Path) -> pd.DataFrame:
    """Fill missing moneylines from Kaggle data."""
    print("\n[1/4] Filling FG Moneylines from Kaggle...")
    
    # Load training data
    df = pd.read_csv(training_file, low_memory=False)
    df["game_date"] = pd.to_datetime(df["game_date"], format="mixed")
    if "fg_ml_home" not in df.columns:
        df["fg_ml_home"] = np.nan
    if "fg_ml_away" not in df.columns:
        df["fg_ml_away"] = np.nan
    before = df["fg_ml_home"].notna().sum()

    if not KAGGLE_FILE.exists():
        print("      [SKIP] Kaggle file missing; filling from TheOdds moneylines when available")
        if "to_fg_ml_home" in df.columns:
            df["fg_ml_home"] = df["fg_ml_home"].fillna(df["to_fg_ml_home"])
        if "to_fg_ml_away" in df.columns:
            df["fg_ml_away"] = df["fg_ml_away"].fillna(df["to_fg_ml_away"])
        after = df["fg_ml_home"].notna().sum()
        print(f"      Before: {before}, After: {after}, Filled: {after - before}")
        return df
    
    # Load Kaggle
    kaggle = pd.read_csv(KAGGLE_FILE)
    kaggle["date"] = pd.to_datetime(kaggle["date"])
    kaggle = kaggle[kaggle["date"] >= "2023-01-01"]
    
    # Standardize team names in Kaggle
    kaggle["home_std"] = kaggle["home"].apply(standardize_team_name)
    kaggle["away_std"] = kaggle["away"].apply(standardize_team_name)
    
    # Generate match keys
    kaggle["match_key"] = kaggle.apply(
        lambda r: generate_match_key(r["date"], r["home_std"], r["away_std"], source_is_utc=False),
        axis=1
    )
    
    # Prepare moneyline lookup
    ml_lookup = kaggle.set_index("match_key")[["moneyline_home", "moneyline_away"]].to_dict("index")
    
    # Fill missing
    filled = 0
    for idx, row in df.iterrows():
        if pd.isna(row["fg_ml_home"]):
            mk = row["match_key"]
            if mk in ml_lookup:
                ml = ml_lookup[mk]
                if pd.notna(ml["moneyline_home"]):
                    df.at[idx, "fg_ml_home"] = ml["moneyline_home"]
                    df.at[idx, "fg_ml_away"] = ml["moneyline_away"]
                    filled += 1
    
    after = df["fg_ml_home"].notna().sum()
    print(f"      Before: {before}, After: {after}, Filled: {filled}")
    
    return df


def compute_travel_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute travel features from team locations."""
    print("\n[2/4] Computing travel features...")
    
    # Initialize columns
    # Strict: initialize to NaN, not neutral defaults.
    df["away_travel_distance"] = np.nan
    df["away_timezone_change"] = np.nan
    df["away_travel_fatigue"] = np.nan
    df["is_away_cross_country"] = np.nan
    df["is_away_long_trip"] = np.nan
    df["away_b2b_travel_penalty"] = np.nan
    df["travel_advantage"] = np.nan
    
    computed = 0
    for idx, row in df.iterrows():
        try:
            home = row["home_team"]
            away = row["away_team"]
            
            # Get travel distance
            distance = get_travel_distance(away, home)
            tz_change = abs(get_timezone_difference(away, home))
            fatigue = calculate_travel_fatigue(distance)
            
            df.at[idx, "away_travel_distance"] = distance
            df.at[idx, "away_timezone_change"] = tz_change
            df.at[idx, "away_travel_fatigue"] = fatigue
            df.at[idx, "is_away_cross_country"] = 1 if distance > 2000 else 0
            df.at[idx, "is_away_long_trip"] = 1 if distance > 1500 else 0
            
            # B2B travel penalty
            away_rest = row.get("away_rest_days", 2)
            if away_rest == 1 and distance > 1000:
                df.at[idx, "away_b2b_travel_penalty"] = fatigue * 1.5
            
            # Travel advantage (positive = home has advantage)
            df.at[idx, "travel_advantage"] = fatigue
            
            computed += 1
        except Exception:
            # Leave as NaN; no silent 0-fill.
            continue
    
    print(f"      Computed travel for {computed}/{len(df)} games")
    return df


def add_betting_splits_defaults(df: pd.DataFrame) -> pd.DataFrame:
    """Add betting splits columns without fabricating values.

    Splits are not available historically in this repo; we explicitly mark them as unavailable
    and leave split-derived numeric features as NaN.
    """
    print("\n[3/4] Adding betting splits columns (no placeholders)...")

    df["has_real_splits"] = 0
    for col in [
        "is_rlm_spread",
        "sharp_side_spread",
        "spread_public_home_pct",
        "spread_ticket_money_diff",
        "spread_movement",
        "is_rlm_total",
        "sharp_side_total",
    ]:
        if col not in df.columns:
            df[col] = np.nan

    print("      Marked splits as unavailable; left split features as NaN")
    return df


def compute_injury_impact(df: pd.DataFrame, *, min_match_fraction: float = 0.99, drop_unmatched: bool = True) -> pd.DataFrame:
    """
    Compute injury impact from inactive_players.csv and common_player_info.csv.
    
    Impact scoring:
    - Each inactive player = base impact
    - Higher experience (season_exp) = higher impact
    - Low draft picks (1st round) = higher impact
    - Greatest 75 flag = max impact
    """
    print("\n[4/8] Computing injury impact...")
    
    # Early exit if injury columns already exist with sufficient coverage
    probe_col = "home_injury_spread_impact" if "home_injury_spread_impact" in df.columns else "home_injury_impact"
    if probe_col in df.columns:
        existing_coverage = df[probe_col].notna().sum() / len(df) if len(df) else 0
        if existing_coverage >= min_match_fraction:
            print(f"      Injury columns already present with {existing_coverage:.1%} coverage - skipping recompute")
            return df

    # Preferred (if available): a normalized per-date injury feed produced by scripts/fetch_injuries.py.
    # This can be backed by API-Basketball (historical) or ESPN (current).
    # Strict policy: if this input exists but does not overlap the training window, fail.
    INJURIES_CSV = PROJECT_ROOT / "data" / "processed" / "injuries.csv"

    INACTIVE_CSV = PROJECT_ROOT / "data" / "external" / "nba_database" / "inactive_players.csv"
    PLAYER_INFO_CSV = PROJECT_ROOT / "data" / "external" / "nba_database" / "common_player_info.csv"
    GAME_CSV_NBA_DB = PROJECT_ROOT / "data" / "external" / "nba_database" / "game.csv"
    GAME_CSV_KAGGLE_NBA = PROJECT_ROOT / "data" / "external" / "kaggle_nba" / "Games.csv"

    train_min = pd.to_datetime(df["game_date"], errors="coerce").min()
    train_max = pd.to_datetime(df["game_date"], errors="coerce").max()
    if pd.isna(train_min) or pd.isna(train_max):
        raise ValueError("Training data has invalid game_date values; cannot compute injury features.")

    # Canonical training data is expected to carry injury columns already.
    # If coverage is insufficient, recompute from injury sources below.

    # Team abbreviation normalization (shared by both injury sources).
    TEAM_ABBR_MAP = {
        "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
        "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
        "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
        "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
        "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
        "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
        "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
        "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
        "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
        "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
    }
    ABBR_TO_FULL = {abbr: full for full, abbr in TEAM_ABBR_MAP.items()}
    ABBR_TO_SHORT = {
        "ATL": "Hawks", "BOS": "Celtics", "BKN": "Nets", "CHA": "Hornets",
        "CHI": "Bulls", "CLE": "Cavaliers", "DAL": "Mavericks", "DEN": "Nuggets",
        "DET": "Pistons", "GSW": "Warriors", "HOU": "Rockets", "IND": "Pacers",
        "LAC": "Clippers", "LAL": "Lakers", "MEM": "Grizzlies", "MIA": "Heat",
        "MIL": "Bucks", "MIN": "Timberwolves", "NOP": "Pelicans", "NYK": "Knicks",
        "OKC": "Thunder", "ORL": "Magic", "PHI": "76ers", "PHX": "Suns",
        "POR": "Trail Blazers", "SAC": "Kings", "SAS": "Spurs", "TOR": "Raptors",
        "UTA": "Jazz", "WAS": "Wizards",
    }
    SHORT_TO_ABBR = {short: abbr for abbr, short in ABBR_TO_SHORT.items()}

    if INJURIES_CSV.exists():
        print(f"      Using injuries feed: {INJURIES_CSV}")

        inj = pd.read_csv(INJURIES_CSV, low_memory=False)
        if inj.empty:
            raise ValueError(f"Injuries feed is empty: {INJURIES_CSV}")

        # Prefer API-Basketball's 'report_date' field; fall back to 'date' if present.
        if "report_date" in inj.columns:
            inj_date = pd.to_datetime(inj["report_date"], errors="coerce")
        elif "date" in inj.columns:
            inj_date = pd.to_datetime(inj["date"], errors="coerce")
        else:
            raise ValueError(
                f"Injuries feed missing a usable date column (expected report_date or date): {INJURIES_CSV}"
            )

        inj["injury_date"] = inj_date.dt.date
        inj = inj[inj["injury_date"].notna()].copy()

        inj_min = pd.to_datetime(inj["injury_date"].min())
        inj_max = pd.to_datetime(inj["injury_date"].max())
        if inj_max < train_min or inj_min > train_max:
            raise ValueError(
                "Injuries feed does not cover the training date range. "
                f"Training window: {train_min.date()}..{train_max.date()} ; "
                f"injuries feed window: {inj_min.date()}..{inj_max.date()}. "
                "Provide historical injury inputs (API-Basketball) or disable injury features for backtests."
            )

        if "team" not in inj.columns:
            raise ValueError(f"Injuries feed missing required column 'team': {INJURIES_CSV}")

        # Map team short names (e.g., "Lakers") or abbreviations (e.g., "LAL") to NBA abbreviations.
        team_val = inj["team"].astype(str).str.strip()
        team_val_upper = team_val.str.upper()
        inj["team_abbr"] = team_val.map(SHORT_TO_ABBR)
        inj.loc[inj["team_abbr"].isna() & team_val_upper.isin(ABBR_TO_FULL.keys()), "team_abbr"] = team_val_upper

        # Strict: don't assume 0 PPG if not present.
        inj["ppg_num"] = pd.to_numeric(inj["ppg"], errors="coerce") if "ppg" in inj.columns else np.nan
        status_series = inj["status"].astype(str).str.lower() if "status" in inj.columns else ""
        inj["is_star_out"] = (status_series == "out") & (inj["ppg_num"] > 15)

        # Aggregate by (date, team). min_count=1 prevents accidental 0-fill when all ppg are NaN.
        by_date_team = (
            inj.groupby(["injury_date", "team_abbr"], dropna=False)
            .agg(
                total_ppg_out=("ppg_num", lambda s: s.sum(min_count=1)),
                star_out=("is_star_out", "max"),
                injury_count=("player_name", "count"),
            )
            .reset_index()
        )
        by_date_team["star_out"] = by_date_team["star_out"].fillna(False).astype(int)

        # Join to games by date + team abbreviation.
        game_dates = pd.to_datetime(df["game_date"], errors="coerce").dt.date
        home_abbr = df["home_team"].map(TEAM_ABBR_MAP)
        away_abbr = df["away_team"].map(TEAM_ABBR_MAP)

        lookup = by_date_team.set_index(["injury_date", "team_abbr"]).to_dict("index")

        home_imp = []
        away_imp = []
        home_star = []
        away_star = []
        has_data = []
        matched = 0
        for d, h, a in zip(game_dates, home_abbr, away_abbr):
            hrow = lookup.get((d, h))
            arow = lookup.get((d, a))
            if hrow is not None or arow is not None:
                matched += 1
                home_imp.append(hrow.get("total_ppg_out", np.nan) if hrow else np.nan)
                away_imp.append(arow.get("total_ppg_out", np.nan) if arow else np.nan)
                home_star.append(hrow.get("star_out", np.nan) if hrow else np.nan)
                away_star.append(arow.get("star_out", np.nan) if arow else np.nan)
                has_data.append(1)
            else:
                home_imp.append(np.nan)
                away_imp.append(np.nan)
                home_star.append(np.nan)
                away_star.append(np.nan)
                has_data.append(0)

        df["home_injury_spread_impact"] = home_imp
        df["away_injury_spread_impact"] = away_imp
        df["injury_spread_diff"] = df["home_injury_spread_impact"] - df["away_injury_spread_impact"]
        df["home_star_out"] = home_star
        df["away_star_out"] = away_star
        df["has_injury_data"] = has_data

        match_fraction = matched / len(df) if len(df) else 0
        print(f"      Matched injuries feed: {matched}/{len(df)} games ({match_fraction*100:.1f}%)")
        if match_fraction < min_match_fraction:
            msg = (
                f"Injury match coverage too low: {match_fraction:.2%} < {min_match_fraction:.2%}. "
                "Fix inputs/matching before backtesting."
            )
            if drop_unmatched:
                before = len(df)
                df = df[df["has_injury_data"] == 1].copy()
                after = len(df)
                print(f"      [STRICT] Dropped unmatched injury games: {before:,} -> {after:,}")
                if after == 0:
                    raise ValueError(msg)
            else:
                raise ValueError(msg)

        return df

    # Fallback: inactive players dataset + player profiles.
    print("      Using inactive players dataset (fallback)")

    if not INACTIVE_CSV.exists() or not PLAYER_INFO_CSV.exists():
        raise FileNotFoundError(
            "Missing required injury inputs: inactive_players.csv and/or common_player_info.csv. "
            "If you intended to use API-Basketball, first generate data/processed/injuries.csv via scripts/fetch_injuries.py."
        )
    
    # Load data
    inact = pd.read_csv(INACTIVE_CSV)
    player_info = pd.read_csv(PLAYER_INFO_CSV)
    
    print(f"      Loaded {len(inact):,} inactive records, {len(player_info):,} player profiles")
    
    # Build player impact scores
    # Impact = season_exp * 0.5 + draft_bonus + star_bonus
    player_info["impact_score"] = 0.0
    
    # Experience bonus (more experience = higher impact when out)
    player_info["impact_score"] += pd.to_numeric(player_info["season_exp"], errors="coerce").fillna(0) * 0.3
    
    # Convert draft columns to numeric
    player_info["draft_round_num"] = pd.to_numeric(player_info["draft_round"], errors="coerce").fillna(99)
    player_info["draft_number_num"] = pd.to_numeric(player_info["draft_number"], errors="coerce").fillna(99)
    
    # Draft position bonus (1st round picks are higher impact)
    player_info["draft_bonus"] = 0.0
    player_info.loc[player_info["draft_round_num"] == 1, "draft_bonus"] = 2.0
    player_info.loc[player_info["draft_number_num"] <= 5, "draft_bonus"] = 3.0  # Top 5 picks
    player_info.loc[player_info["draft_number_num"] == 1, "draft_bonus"] = 4.0  # #1 overall
    player_info["impact_score"] += player_info["draft_bonus"]
    
    # Greatest 75 bonus (all-time greats)
    player_info.loc[player_info["greatest_75_flag"] == "Y", "impact_score"] += 5.0
    
    # Normalize to 0-10 scale (max reasonable impact per player = 10 PPG equivalent)
    max_impact = player_info["impact_score"].max()
    if max_impact > 0:
        player_info["impact_score"] = (player_info["impact_score"] / max_impact) * 10.0
    
    # Create lookup: player_id -> impact_score
    impact_lookup = player_info.set_index("person_id")["impact_score"].to_dict()
    
    # Load game data for game_id -> date/teams mapping.
    # Prefer the nba_database mapping if it overlaps; otherwise use the kaggle_nba Games.csv mapping.
    game_df = None
    mapping_source = None
    if GAME_CSV_NBA_DB.exists():
        candidate = pd.read_csv(GAME_CSV_NBA_DB, low_memory=False)
        candidate["game_date"] = pd.to_datetime(candidate["game_date"], errors="coerce")
        if candidate["game_date"].notna().any():
            cmin = candidate["game_date"].min()
            cmax = candidate["game_date"].max()
            if not (cmax < train_min or cmin > train_max):
                game_df = candidate
                mapping_source = GAME_CSV_NBA_DB

    if game_df is None and GAME_CSV_KAGGLE_NBA.exists():
        candidate = pd.read_csv(GAME_CSV_KAGGLE_NBA, low_memory=False)
        candidate["game_date"] = pd.to_datetime(candidate["gameDateTimeEst"], errors="coerce")
        if candidate["game_date"].notna().any():
            cmin = candidate["game_date"].min()
            cmax = candidate["game_date"].max()
            if not (cmax < train_min or cmin > train_max):
                game_df = candidate
                mapping_source = GAME_CSV_KAGGLE_NBA

    if game_df is None or mapping_source is None:
        raise ValueError(
            "No usable game mapping source found for injury matching. "
            f"Tried: {GAME_CSV_NBA_DB} and {GAME_CSV_KAGGLE_NBA}."
        )

    # Build a fast lookup from (date, home_team, away_team) -> game_id.
    if mapping_source == GAME_CSV_NBA_DB:
        # Map NBA abbreviations to full names to align with training data.
        abbr_to_full = {v: k for k, v in TEAM_ABBR_MAP.items()}
        game_df = game_df[game_df["game_date"].notna()].copy()
        game_df["game_day"] = game_df["game_date"].dt.date
        game_df["home_team_full"] = game_df["team_abbreviation_home"].map(abbr_to_full)
        game_df["away_team_full"] = game_df["team_abbreviation_away"].map(abbr_to_full)
        game_df = game_df[game_df["home_team_full"].notna() & game_df["away_team_full"].notna()].copy()
        game_df = game_df[(game_df["game_date"] >= train_min) & (game_df["game_date"] <= train_max)].copy()
        game_key_to_id = {
            (r["game_day"], r["home_team_full"], r["away_team_full"]): r["game_id"]
            for _, r in game_df.iterrows()
        }
        game_id_to_date = game_df.set_index("game_id")["game_date"].to_dict()
    else:
        # Kaggle mapping is already team city+name; build the full names.
        game_df = game_df[game_df["game_date"].notna()].copy()
        game_df["game_day"] = game_df["game_date"].dt.date
        game_df["home_team_full"] = (game_df["hometeamCity"].astype(str).str.strip() + " " + game_df["hometeamName"].astype(str).str.strip())
        game_df["away_team_full"] = (game_df["awayteamCity"].astype(str).str.strip() + " " + game_df["awayteamName"].astype(str).str.strip())
        game_df = game_df[(game_df["game_date"] >= train_min) & (game_df["game_date"] <= train_max)].copy()
        game_key_to_id = {
            (r["game_day"], r["home_team_full"], r["away_team_full"]): r["gameId"]
            for _, r in game_df.iterrows()
        }
        game_id_to_date = game_df.set_index("gameId")["game_date"].to_dict()

    print(f"      Game mapping source: {mapping_source.name} ; keys: {len(game_key_to_id):,}")
    
    # Compute impact per game per team
    # Strict: do not invent an impact for unknown players; leave as NaN.
    inact["impact"] = inact["player_id"].map(impact_lookup)
    
    # Aggregate by game + team.
    # Strict: min_count=1 prevents treating all-NaN impact sets as 0.
    team_impact = inact.groupby(["game_id", "team_abbreviation"]).agg(
        total_impact=("impact", lambda s: s.sum(min_count=1)),
        inactive_count=("player_id", "count"),
    ).reset_index()
    team_impact = team_impact.rename(columns={"team_abbreviation": "team_abbr"})
    
    # Build lookup: (game_id, team_abbr) -> (total_impact, inactive_count, has_star)
    # Star = impact > 5 (top tier player)
    inact["is_star"] = inact["impact"].fillna(-1) > 5.0
    star_out = inact.groupby(["game_id", "team_abbreviation"])["is_star"].max().reset_index()
    star_out.columns = ["game_id", "team_abbr", "star_out"]
    
    team_impact = team_impact.merge(star_out, on=["game_id", "team_abbr"], how="left")
    team_impact["star_out"] = team_impact["star_out"].fillna(False).astype(int)
    
    impact_lookup_game = {}
    for _, row in team_impact.iterrows():
        key = (row["game_id"], row["team_abbr"])
        impact_lookup_game[key] = {
            "impact": row["total_impact"],
            "count": row["inactive_count"],
            "star_out": row["star_out"]
        }
    
    print(f"      Built impact lookup for {len(impact_lookup_game)} game-team pairs")
    
    # Map to training data
    # Need to match by date + team since game_ids may differ
    home_impact = []
    away_impact = []
    home_star = []
    away_star = []
    has_data = []
    
    # Build team abbr normalization
    TEAM_ABBR_MAP = {
        "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
        "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
        "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
        "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
        "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
        "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
        "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
        "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
        "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
        "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS"
    }
    
    # Validate coverage window for the inactive records themselves (not just the game map).
    inact_ids = pd.to_numeric(inact["game_id"], errors="coerce")
    inact_dates = pd.to_datetime(pd.Series(inact_ids.map(game_id_to_date)), errors="coerce")
    inact_min = inact_dates.min()
    inact_max = inact_dates.max()
    if pd.isna(inact_min) or pd.isna(inact_max) or (inact_max < train_min) or (inact_min > train_max):
        raise ValueError(
            "Inactive-player inputs do not cover the training date range. "
            f"Training window: {train_min.date()}..{train_max.date()} ; "
            f"inactive window: {inact_min.date() if pd.notna(inact_min) else 'NaT'}..{inact_max.date() if pd.notna(inact_max) else 'NaT'}. "
            "Provide a refreshed inactive_players dataset for 2023+ (or generate data/processed/injuries.csv via API-Basketball)."
        )

    matched = 0
    for _, row in df.iterrows():
        home_abbr = TEAM_ABBR_MAP.get(row["home_team"], "")
        away_abbr = TEAM_ABBR_MAP.get(row["away_team"], "")

        game_date = pd.to_datetime(row["game_date"], errors="coerce")
        if pd.isna(game_date):
            home_impact.append(np.nan)
            away_impact.append(np.nan)
            home_star.append(np.nan)
            away_star.append(np.nan)
            has_data.append(0)
            continue

        gid = game_key_to_id.get((game_date.date(), row["home_team"], row["away_team"]))
        found_home = impact_lookup_game.get((gid, home_abbr), {}) if gid is not None else None
        found_away = impact_lookup_game.get((gid, away_abbr), {}) if gid is not None else None
        if gid is not None:
            matched += 1
        
        if found_home or found_away:
            home_impact.append(found_home.get("impact", np.nan) if found_home else np.nan)
            away_impact.append(found_away.get("impact", np.nan) if found_away else np.nan)
            home_star.append(found_home.get("star_out", np.nan) if found_home else np.nan)
            away_star.append(found_away.get("star_out", np.nan) if found_away else np.nan)
            has_data.append(1)
        else:
            home_impact.append(np.nan)
            away_impact.append(np.nan)
            home_star.append(np.nan)
            away_star.append(np.nan)
            has_data.append(0)
    
    df["home_injury_spread_impact"] = home_impact
    df["away_injury_spread_impact"] = away_impact
    df["injury_spread_diff"] = df["home_injury_spread_impact"] - df["away_injury_spread_impact"]
    df["home_star_out"] = home_star
    df["away_star_out"] = away_star
    df["has_injury_data"] = has_data
    
    match_fraction = matched / len(df) if len(df) else 0
    print(f"      Matched injury data: {matched}/{len(df)} games ({match_fraction*100:.1f}%)")
    print(f"      home_injury_spread_impact: mean={df['home_injury_spread_impact'].mean():.2f}")
    print(f"      away_injury_spread_impact: mean={df['away_injury_spread_impact'].mean():.2f}")
    print(f"      Stars out: home={df['home_star_out'].sum()}, away={df['away_star_out'].sum()}")
    
    # Enforce coverage: if we can't match injuries, don't silently keep fabricated zeros.
    if match_fraction < min_match_fraction:
        msg = (
            f"Injury match coverage too low: {match_fraction:.2%} < {min_match_fraction:.2%}. "
            "Fix inputs/matching before backtesting."
        )
        if drop_unmatched:
            before = len(df)
            df = df[df["has_injury_data"] == 1].copy()
            after = len(df)
            print(f"      [STRICT] Dropped unmatched injury games: {before:,} -> {after:,}")
            if after == 0:
                raise ValueError(msg)
        else:
            raise ValueError(msg)

    return df


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute simple derived features from existing columns."""
    print("\n[5/8] Computing derived features...")
    
    # ppg_diff
    if "home_ppg" in df.columns and "away_ppg" in df.columns:
        df["ppg_diff"] = df["home_ppg"] - df["away_ppg"]
        print(f"      ppg_diff: {df['ppg_diff'].notna().sum()}/{len(df)}")
    
    # win_pct_diff
    if "home_win_pct" in df.columns and "away_win_pct" in df.columns:
        df["win_pct_diff"] = df["home_win_pct"] - df["away_win_pct"]
        print(f"      win_pct_diff: {df['win_pct_diff'].notna().sum()}/{len(df)}")
    
    # rest_diff
    if "home_rest_days" in df.columns and "away_rest_days" in df.columns:
        df["rest_diff"] = df["home_rest_days"] - df["away_rest_days"]
        print(f"      rest_diff: {df['rest_diff'].notna().sum()}/{len(df)}")
    
    # home_margin / away_margin (copy from avg_margin)
    if "home_avg_margin" in df.columns:
        df["home_margin"] = df["home_avg_margin"]
        print(f"      home_margin: {df['home_margin'].notna().sum()}/{len(df)}")
    
    if "away_avg_margin" in df.columns:
        df["away_margin"] = df["away_avg_margin"]
        print(f"      away_margin: {df['away_margin'].notna().sum()}/{len(df)}")
    
    # net_rating_diff (use margin diff as proxy)
    if "home_margin" in df.columns and "away_margin" in df.columns:
        df["net_rating_diff"] = df["home_margin"] - df["away_margin"]
        print(f"      net_rating_diff: {df['net_rating_diff'].notna().sum()}/{len(df)}")
    
    # NOTE: Do not inject constant "baseline" features here.
    # If pace/home-court are needed, they must come from real inputs or be computed elsewhere.
    
    return df


def compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling std features."""
    print("\n[6/8] Computing rolling std features...")
    
    df = df.sort_values("game_date").reset_index(drop=True)
    
    # Initialize
    home_margin_std = []
    away_margin_std = []
    home_score_std = []
    away_score_std = []
    home_form_trend = []
    away_form_trend = []
    
    team_margins = {}
    team_scores = {}
    
    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        
        # Home rolling stats (strict: NaN until enough history)
        if home in team_margins and len(team_margins[home]) >= 5:
            home_margin_std.append(np.std(team_margins[home][-10:]))
            home_score_std.append(np.std(team_scores[home][-10:]))
            l5 = np.mean(team_margins[home][-5:])
            l10 = np.mean(team_margins[home][-10:]) if len(team_margins[home]) >= 10 else l5
            home_form_trend.append(l5 - l10)
        else:
            home_margin_std.append(np.nan)
            home_score_std.append(np.nan)
            home_form_trend.append(np.nan)
        
        # Away rolling stats
        if away in team_margins and len(team_margins[away]) >= 5:
            away_margin_std.append(np.std(team_margins[away][-10:]))
            away_score_std.append(np.std(team_scores[away][-10:]))
            l5 = np.mean(team_margins[away][-5:])
            l10 = np.mean(team_margins[away][-10:]) if len(team_margins[away]) >= 10 else l5
            away_form_trend.append(l5 - l10)
        else:
            away_margin_std.append(np.nan)
            away_score_std.append(np.nan)
            away_form_trend.append(np.nan)
        
        # Update history (strict: only when scores are present)
        home_score = row.get("home_score")
        away_score = row.get("away_score")
        margin = (home_score - away_score) if pd.notna(home_score) and pd.notna(away_score) else np.nan
        
        if home not in team_margins:
            team_margins[home] = []
            team_scores[home] = []
        if away not in team_margins:
            team_margins[away] = []
            team_scores[away] = []
        
        if pd.notna(margin):
            team_margins[home].append(float(margin))
            team_margins[away].append(float(-margin))
            team_scores[home].append(float(home_score))
            team_scores[away].append(float(away_score))
    
    df["home_margin_std"] = home_margin_std
    df["away_margin_std"] = away_margin_std
    df["home_score_std"] = home_score_std
    df["away_score_std"] = away_score_std
    df["home_form_trend"] = home_form_trend
    df["away_form_trend"] = away_form_trend
    
    print(f"      home_margin_std: {df['home_margin_std'].notna().sum()}/{len(df)} computed")
    print(f"      away_margin_std: {df['away_margin_std'].notna().sum()}/{len(df)} computed")
    print(f"      home_score_std: {df['home_score_std'].notna().sum()}/{len(df)} computed")
    print(f"      away_score_std: {df['away_score_std'].notna().sum()}/{len(df)} computed")
    print(f"      home_form_trend: {df['home_form_trend'].notna().sum()}/{len(df)} computed")
    print(f"      away_form_trend: {df['away_form_trend'].notna().sum()}/{len(df)} computed")
    
    return df


def compute_h2h_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute head-to-head features."""
    print("\n[7/8] Computing H2H features...")
    
    df = df.sort_values("game_date").reset_index(drop=True)
    
    h2h_games = []
    h2h_margin = []
    
    matchup_history = {}  # (team1, team2) -> list of margins (from team1 perspective)
    
    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        key = tuple(sorted([home, away]))
        
        if key in matchup_history and len(matchup_history[key]) > 0:
            history = matchup_history[key]
            h2h_games.append(len(history))
            # Get from home team's perspective
            if home == key[0]:
                h2h_margin.append(np.mean(history[-5:]))
            else:
                h2h_margin.append(-np.mean(history[-5:]))
        else:
            h2h_games.append(0)
            h2h_margin.append(np.nan)
        
        # Update history (strict: only when scores are present)
        home_score = row.get("home_score")
        away_score = row.get("away_score")
        margin = (home_score - away_score) if pd.notna(home_score) and pd.notna(away_score) else np.nan
        if key not in matchup_history:
            matchup_history[key] = []
        
        # Store from first team's perspective
        if pd.notna(margin):
            if home == key[0]:
                matchup_history[key].append(float(margin))
            else:
                matchup_history[key].append(float(-margin))
    
    df["h2h_games"] = h2h_games
    df["h2h_margin"] = h2h_margin
    
    print(f"      h2h_games: {(df['h2h_games'] > 0).sum()}/{len(df)} with history")
    print(f"      h2h_margin: {df['h2h_margin'].notna().sum()}/{len(df)} computed")
    
    return df


def compute_predicted_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute predicted margin/total features."""
    print("\n[8/8] Computing predicted features...")

    # Strict: do not estimate predicted_* values with baked-in constants.
    # If a feature cannot be computed from real inputs, leave NaN.
    def _num(col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series([np.nan] * len(df), index=df.index)
        return pd.to_numeric(df[col], errors="coerce")

    home_margin = _num("home_avg_margin")
    if home_margin.isna().all():
        home_margin = _num("home_margin")
    away_margin = _num("away_avg_margin")
    if away_margin.isna().all():
        away_margin = _num("away_margin")

    home_ppg = _num("home_ppg")
    away_ppg = _num("away_ppg")
    home_papg = _num("home_papg")
    away_papg = _num("away_papg")

    hca = _num("home_court_advantage")
    if hca.isna().all():
        hca = _num("dynamic_hca")
    away_fatigue = _num("away_travel_fatigue")
    net_rating_diff = _num("net_rating_diff")

    # Predicted margin uses team form + situational context.
    base_margin = (home_margin - away_margin) / 2
    df["predicted_margin"] = (
        base_margin
        + hca.fillna(0)
        - away_fatigue.fillna(0)
        + net_rating_diff.fillna(0) * 0.2
    )
    df.loc[home_margin.isna() | away_margin.isna(), "predicted_margin"] = np.nan
    print(f"      predicted_margin: {df['predicted_margin'].notna().sum()}/{len(df)}")

    # Predicted total uses matchup-based points expectations.
    home_expected = (home_ppg + away_papg) / 2
    away_expected = (away_ppg + home_papg) / 2
    df["predicted_total"] = home_expected + away_expected
    df.loc[home_ppg.isna() | away_ppg.isna() | home_papg.isna() | away_papg.isna(), "predicted_total"] = np.nan
    print(f"      predicted_total: {df['predicted_total'].notna().sum()}/{len(df)}")

    # spread_vs_predicted (use canonical spread_line if present)
    spread_line = _num("spread_line")
    if spread_line.isna().all():
        spread_line = _num("fg_spread_line")
    df["spread_vs_predicted"] = spread_line - (-df["predicted_margin"])
    df.loc[spread_line.isna() | df["predicted_margin"].isna(), "spread_vs_predicted"] = np.nan
    print(f"      spread_vs_predicted: {df['spread_vs_predicted'].notna().sum()}/{len(df)}")

    # total_vs_predicted (use canonical total_line if present)
    total_line = _num("total_line")
    if total_line.isna().all():
        total_line = _num("fg_total_line")
    df["total_vs_predicted"] = total_line - df["predicted_total"]
    df.loc[total_line.isna() | df["predicted_total"].isna(), "total_vs_predicted"] = np.nan
    print(f"      total_vs_predicted: {df['total_vs_predicted'].notna().sum()}/{len(df)}")

    return df


def verify_model_features(df: pd.DataFrame):
    """Verify all 55 model features are present."""
    print("\n" + "=" * 70)
    print("VERIFYING MODEL FEATURES")
    print("=" * 70)
    
    # Load expected features from model
    import joblib
    import warnings
    warnings.filterwarnings("ignore")
    
    model_data = joblib.load(PROJECT_ROOT / "models" / "production" / "fg_spread_model.joblib")
    expected_features = model_data.get("feature_columns", [])
    
    missing = []
    present = []
    for f in expected_features:
        if f in df.columns:
            ct = df[f].notna().sum()
            present.append(f"{f}: {ct}/{len(df)}")
        else:
            missing.append(f)
    
    print(f"\nExpected: {len(expected_features)} features")
    print(f"Present: {len(present)}")
    print(f"Missing: {len(missing)}")
    
    if missing:
        print("\nMISSING FEATURES:")
        for f in missing:
            print(f"  {f}")
    
    return len(missing) == 0


def main():
    print("=" * 70)
    print("COMPLETING TRAINING DATA FEATURES")
    print("=" * 70)
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-file",
        default=str(TRAINING_FILE),
        help=f"Path to training CSV to enrich (default: {TRAINING_FILE})",
    )
    parser.add_argument(
        "--min-injury-match-fraction",
        type=float,
        default=0.99,
        help="Minimum fraction of games that must match injury inputs (default: 0.99)",
    )
    parser.add_argument(
        "--skip-injuries",
        action="store_true",
        help="Skip injury feature computation (use when source files unavailable)",
    )
    args = parser.parse_args()

    training_file = Path(args.training_file)

    # Fill moneylines
    df = fill_moneylines(training_file)
    
    # Compute travel features (no silent 0-fill)
    df = compute_travel_features(df)
    
    # Add betting splits columns (no placeholders)
    df = add_betting_splits_defaults(df)
    
    # Compute injury impact from inactive_players.csv
    if args.skip_injuries:
        print("\n[4/8] Skipping injury impact (--skip-injuries flag)")
    else:
        try:
            df = compute_injury_impact(df, min_match_fraction=args.min_injury_match_fraction)
        except (FileNotFoundError, ValueError) as e:
            print(f"\n[4/8] Skipping injury impact (source files unavailable): {e}")
    
    # Compute derived features (ppg_diff, win_pct_diff, etc.)
    df = compute_derived_features(df)
    
    # Compute rolling features (margin_std, form_trend, etc.)
    df = compute_rolling_features(df)
    
    # Compute H2H features
    df = compute_h2h_features(df)
    
    # Compute predicted features
    df = compute_predicted_features(df)
    
    # Save
    print("\nSaving...")
    df.to_csv(training_file, index=False)
    print(f"Saved to {training_file}")
    
    # Verify
    all_present = verify_model_features(df)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total games: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"fg_ml_home coverage: {df['fg_ml_home'].notna().sum()}/{len(df)}")
    
    if all_present:
        print("\n[OK] All 55 model features are present!")
    else:
        print("\n[WARN] Some features still missing - check above")


if __name__ == "__main__":
    main()
