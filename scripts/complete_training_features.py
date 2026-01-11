#!/usr/bin/env python3
"""
Complete training data with all features used by prediction models.

Fills in:
1. FG Moneylines from Kaggle
2. Travel features (computed from team locations)
3. Betting splits defaults (not available historically)
4. Injury feature defaults (not available historically)

This ensures training data has all 55 features the model expects.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.standardization import standardize_team_name, generate_match_key
from src.modeling.team_factors import (
    get_travel_distance,
    get_timezone_difference,
    calculate_travel_fatigue,
)

DATA_DIR = PROJECT_ROOT / "data"
TRAINING_FILE = DATA_DIR / "processed" / "training_data_complete_2023.csv"
KAGGLE_FILE = DATA_DIR / "external" / "kaggle" / "nba_2008-2025.csv"


def fill_moneylines():
    """Fill missing moneylines from Kaggle data."""
    print("\n[1/4] Filling FG Moneylines from Kaggle...")
    
    # Load training data
    df = pd.read_csv(TRAINING_FILE, low_memory=False)
    df["game_date"] = pd.to_datetime(df["game_date"], format="mixed")
    before = df["fg_ml_home"].notna().sum()
    
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
    df["away_travel_distance"] = 0.0
    df["away_timezone_change"] = 0
    df["away_travel_fatigue"] = 0.0
    df["is_away_cross_country"] = 0
    df["is_away_long_trip"] = 0
    df["away_b2b_travel_penalty"] = 0.0
    df["travel_advantage"] = 0.0
    
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
        except Exception as e:
            pass
    
    print(f"      Computed travel for {computed}/{len(df)} games")
    return df


def add_betting_splits_defaults(df: pd.DataFrame) -> pd.DataFrame:
    """Add default values for betting splits features (not available historically)."""
    print("\n[3/4] Adding betting splits defaults...")
    
    # These are real-time features not available for backtesting
    # Set to neutral defaults so model doesn't get confused
    df["has_real_splits"] = 0  # Indicates no real splits data
    df["is_rlm_spread"] = 0
    df["sharp_side_spread"] = 0
    df["spread_public_home_pct"] = 50  # Neutral
    df["spread_ticket_money_diff"] = 0
    df["spread_movement"] = 0.0
    df["is_rlm_total"] = 0
    df["sharp_side_total"] = 0
    
    print("      Set all splits features to neutral defaults")
    return df


def compute_injury_impact(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute injury impact from inactive_players.csv and common_player_info.csv.
    
    Impact scoring:
    - Each inactive player = base impact
    - Higher experience (season_exp) = higher impact
    - Low draft picks (1st round) = higher impact
    - Greatest 75 flag = max impact
    """
    print("\n[4/8] Computing injury impact from inactive players...")
    
    INACTIVE_CSV = PROJECT_ROOT / "data" / "external" / "nba_database" / "inactive_players.csv"
    PLAYER_INFO_CSV = PROJECT_ROOT / "data" / "external" / "nba_database" / "common_player_info.csv"
    GAME_CSV = PROJECT_ROOT / "data" / "external" / "nba_database" / "game.csv"
    
    if not INACTIVE_CSV.exists() or not PLAYER_INFO_CSV.exists():
        print("      [SKIP] Missing inactive_players.csv or common_player_info.csv")
        df["home_injury_spread_impact"] = 0.0
        df["away_injury_spread_impact"] = 0.0
        df["injury_spread_diff"] = 0.0
        df["home_star_out"] = 0
        df["away_star_out"] = 0
        df["has_injury_data"] = 0
        return df
    
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
    
    # Load game data for game_id -> date/teams mapping
    if GAME_CSV.exists():
        game_df = pd.read_csv(GAME_CSV, low_memory=False)
        game_df["game_date"] = pd.to_datetime(game_df["game_date"])
        game_df = game_df[game_df["game_date"] >= "2023-01-01"]
        
        # Build game lookup: game_id -> (date, home_abbr, away_abbr)
        game_lookup = {}
        for _, row in game_df.iterrows():
            game_lookup[row["game_id"]] = {
                "date": row["game_date"],
                "home": row["team_abbreviation_home"],
                "away": row["team_abbreviation_away"]
            }
        print(f"      Game lookup built: {len(game_lookup)} games")
    else:
        game_lookup = {}
    
    # Compute impact per game per team
    inact["impact"] = inact["player_id"].map(impact_lookup).fillna(0.5)  # Default 0.5 for unknown
    
    # Aggregate by game + team
    team_impact = inact.groupby(["game_id", "team_abbreviation"]).agg({
        "impact": "sum",
        "player_id": "count"
    }).reset_index()
    team_impact.columns = ["game_id", "team_abbr", "total_impact", "inactive_count"]
    
    # Build lookup: (game_id, team_abbr) -> (total_impact, inactive_count, has_star)
    # Star = impact > 5 (top tier player)
    inact["is_star"] = inact["impact"] > 5.0
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
    
    matched = 0
    for _, row in df.iterrows():
        home_abbr = TEAM_ABBR_MAP.get(row["home_team"], "")
        away_abbr = TEAM_ABBR_MAP.get(row["away_team"], "")
        
        # Try to match by game_id from game_lookup via date
        game_date = pd.to_datetime(row["game_date"])
        found_home = None
        found_away = None
        
        # Search game_lookup for matching date + teams
        for gid, ginfo in game_lookup.items():
            if ginfo["date"].date() == game_date.date():
                if ginfo["home"] == home_abbr and ginfo["away"] == away_abbr:
                    found_home = impact_lookup_game.get((gid, home_abbr), {})
                    found_away = impact_lookup_game.get((gid, away_abbr), {})
                    matched += 1
                    break
        
        if found_home or found_away:
            home_impact.append(found_home.get("impact", 0) if found_home else 0)
            away_impact.append(found_away.get("impact", 0) if found_away else 0)
            home_star.append(found_home.get("star_out", 0) if found_home else 0)
            away_star.append(found_away.get("star_out", 0) if found_away else 0)
            has_data.append(1)
        else:
            home_impact.append(0)
            away_impact.append(0)
            home_star.append(0)
            away_star.append(0)
            has_data.append(0)
    
    df["home_injury_spread_impact"] = home_impact
    df["away_injury_spread_impact"] = away_impact
    df["injury_spread_diff"] = df["home_injury_spread_impact"] - df["away_injury_spread_impact"]
    df["home_star_out"] = home_star
    df["away_star_out"] = away_star
    df["has_injury_data"] = has_data
    
    print(f"      Matched injury data: {matched}/{len(df)} games ({matched/len(df)*100:.1f}%)")
    print(f"      home_injury_spread_impact: mean={df['home_injury_spread_impact'].mean():.2f}")
    print(f"      away_injury_spread_impact: mean={df['away_injury_spread_impact'].mean():.2f}")
    print(f"      Stars out: home={df['home_star_out'].sum()}, away={df['away_star_out'].sum()}")
    
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
    
    # home_court_advantage (constant baseline)
    df["home_court_advantage"] = 3.5  # Historical average
    print(f"      home_court_advantage: {len(df)} (constant 3.5)")
    
    # dynamic_hca (adjust by team strength)
    df["dynamic_hca"] = 3.5 + df.get("win_pct_diff", pd.Series([0]*len(df))).fillna(0) * 2.0
    print(f"      dynamic_hca: {df['dynamic_hca'].notna().sum()}/{len(df)}")
    
    # pace (default for historical)
    df["home_pace"] = 100.0  # League average
    df["away_pace"] = 100.0
    print(f"      home_pace / away_pace: {len(df)} (default 100)")
    
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
        
        # Home margin std (from last 10 games)
        if home in team_margins and len(team_margins[home]) >= 5:
            home_margin_std.append(np.std(team_margins[home][-10:]))
            home_score_std.append(np.std(team_scores[home][-10:]))
            l5 = np.mean(team_margins[home][-5:])
            l10 = np.mean(team_margins[home][-10:]) if len(team_margins[home]) >= 10 else l5
            home_form_trend.append(l5 - l10)
        else:
            home_margin_std.append(6.0)  # League average std
            home_score_std.append(8.0)
            home_form_trend.append(0.0)
        
        # Away margin std
        if away in team_margins and len(team_margins[away]) >= 5:
            away_margin_std.append(np.std(team_margins[away][-10:]))
            away_score_std.append(np.std(team_scores[away][-10:]))
            l5 = np.mean(team_margins[away][-5:])
            l10 = np.mean(team_margins[away][-10:]) if len(team_margins[away]) >= 10 else l5
            away_form_trend.append(l5 - l10)
        else:
            away_margin_std.append(6.0)
            away_score_std.append(8.0)
            away_form_trend.append(0.0)
        
        # Update history
        margin = row["home_score"] - row["away_score"] if pd.notna(row.get("home_score")) else 0
        
        if home not in team_margins:
            team_margins[home] = []
            team_scores[home] = []
        if away not in team_margins:
            team_margins[away] = []
            team_scores[away] = []
        
        team_margins[home].append(margin)
        team_margins[away].append(-margin)
        team_scores[home].append(row.get("home_score", 110))
        team_scores[away].append(row.get("away_score", 110))
    
    df["home_margin_std"] = home_margin_std
    df["away_margin_std"] = away_margin_std
    df["home_score_std"] = home_score_std
    df["away_score_std"] = away_score_std
    df["home_form_trend"] = home_form_trend
    df["away_form_trend"] = away_form_trend
    
    print(f"      home_margin_std: {(df['home_margin_std'] != 6.0).sum()}/{len(df)} computed")
    print(f"      away_margin_std: {(df['away_margin_std'] != 6.0).sum()}/{len(df)} computed")
    print(f"      home_score_std: {(df['home_score_std'] != 8.0).sum()}/{len(df)} computed")
    print(f"      away_score_std: {(df['away_score_std'] != 8.0).sum()}/{len(df)} computed")
    print(f"      home_form_trend: {(df['home_form_trend'] != 0.0).sum()}/{len(df)} computed")
    print(f"      away_form_trend: {(df['away_form_trend'] != 0.0).sum()}/{len(df)} computed")
    
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
            h2h_margin.append(0.0)
        
        # Update history
        margin = row["home_score"] - row["away_score"] if pd.notna(row.get("home_score")) else 0
        if key not in matchup_history:
            matchup_history[key] = []
        
        # Store from first team's perspective
        if home == key[0]:
            matchup_history[key].append(margin)
        else:
            matchup_history[key].append(-margin)
    
    df["h2h_games"] = h2h_games
    df["h2h_margin"] = h2h_margin
    
    print(f"      h2h_games: {(df['h2h_games'] > 0).sum()}/{len(df)} with history")
    print(f"      h2h_margin: {(df['h2h_margin'] != 0.0).sum()}/{len(df)} computed")
    
    return df


def compute_predicted_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute predicted margin/total features."""
    print("\n[8/8] Computing predicted features...")
    
    # Predicted margin: use spread line negated, or estimate from team stats
    if "fg_spread_line" in df.columns:
        df["predicted_margin"] = -df["fg_spread_line"].fillna(df.get("ppg_diff", 0) + 3.5)
        print(f"      predicted_margin: {df['predicted_margin'].notna().sum()}/{len(df)}")
    else:
        df["predicted_margin"] = df.get("ppg_diff", pd.Series([0]*len(df))).fillna(0) + 3.5
        print(f"      predicted_margin (estimated): {len(df)}")
    
    # Predicted total: use total line, or estimate from PPG
    if "fg_total_line" in df.columns:
        df["predicted_total"] = df["fg_total_line"].fillna(
            df.get("home_ppg", 110).fillna(110) + df.get("away_ppg", 110).fillna(110)
        )
        print(f"      predicted_total: {df['predicted_total'].notna().sum()}/{len(df)}")
    else:
        df["predicted_total"] = df.get("home_ppg", 110) + df.get("away_ppg", 110)
        print(f"      predicted_total (estimated): {len(df)}")
    
    # spread_vs_predicted
    df["spread_vs_predicted"] = df["fg_spread_line"].fillna(0) - (-df["predicted_margin"].fillna(0))
    print(f"      spread_vs_predicted: {df['spread_vs_predicted'].notna().sum()}/{len(df)}")
    
    # total_vs_predicted
    df["total_vs_predicted"] = df["fg_total_line"].fillna(220) - df["predicted_total"].fillna(220)
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
    
    # Fill moneylines
    df = fill_moneylines()
    
    # Compute travel features (silently fail if team_factors not available)
    try:
        df = compute_travel_features(df)
    except Exception as e:
        print(f"\n[2/8] Computing travel features...")
        print(f"      [SKIP] Travel module error: {e}")
        df["travel_advantage"] = 0.0
    
    # Add betting splits defaults
    df = add_betting_splits_defaults(df)
    
    # Compute injury impact from inactive_players.csv
    df = compute_injury_impact(df)
    
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
    df.to_csv(TRAINING_FILE, index=False)
    print(f"Saved to {TRAINING_FILE}")
    
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
        print("\n✅ All 55 model features are present!")
    else:
        print("\n⚠️ Some features still missing - check above")


if __name__ == "__main__":
    main()
