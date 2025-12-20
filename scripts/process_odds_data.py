#!/usr/bin/env python3
"""
Process The Odds API data to extract betting splits and first-half lines.

Reads the raw JSON files from data/raw/the_odds/YYYY-MM-DD/ and extracts:
1. Betting splits data (line movement, opening lines for RLM detection)
2. First-half lines from markets endpoint
3. Team totals lines

Outputs:
- data/processed/betting_splits.csv
- data/processed/first_half_lines.csv
- data/processed/team_totals_lines.csv

Usage:
    python scripts/process_odds_data.py [--date YYYY-MM-DD]
"""
import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.ingestion.betting_splits import GameSplits, detect_reverse_line_movement


def find_latest_odds_dir():
    """Find the most recent odds data directory."""
    odds_dir = Path('data/raw/the_odds')
    if not odds_dir.exists():
        return None

    # Get all date directories
    date_dirs = [d for d in odds_dir.iterdir() if d.is_dir() and d.name.count('-') == 2]
    if not date_dirs:
        return None

    # Sort by date (directory name is YYYY-MM-DD)
    date_dirs.sort(reverse=True)
    return date_dirs[0]


def extract_betting_splits(odds_data_dir: Path):
    """
    Extract betting splits from odds data.

    Line movement is inferred from comparing multiple bookmakers.
    True public betting percentages require separate data source.
    """
    splits_records = []

    # Read sport-level odds file
    sport_odds_files = list(odds_data_dir.glob('sport_odds_*.json'))
    if not sport_odds_files:
        print("[WARN] No sport_odds files found")
        return pd.DataFrame()

    sport_odds_file = sport_odds_files[-1]  # Use most recent
    print(f"Reading {sport_odds_file.name}...")

    with open(sport_odds_file, 'r') as f:
        events = json.load(f)

    for event in events:
        event_id = event.get('id')
        home_team = event.get('home_team')
        away_team = event.get('away_team')
        commence_time = event.get('commence_time')

        bookmakers = event.get('bookmakers', [])
        if not bookmakers:
            continue

        # Extract spread and total lines from all bookmakers
        spread_lines = []
        total_lines = []

        for bm in bookmakers:
            bm_name = bm.get('key')
            last_update = bm.get('last_update')

            for market in bm.get('markets', []):
                market_key = market.get('key')

                if market_key == 'spreads':
                    # Find home team spread
                    for outcome in market.get('outcomes', []):
                        if outcome.get('name') == home_team:
                            spread_lines.append({
                                'bookmaker': bm_name,
                                'line': outcome.get('point'),
                                'price': outcome.get('price'),
                                'update_time': last_update
                            })

                elif market_key == 'totals':
                    # Total line
                    for outcome in market.get('outcomes', []):
                        if outcome.get('name') == 'Over':
                            total_lines.append({
                                'bookmaker': bm_name,
                                'line': outcome.get('point'),
                                'price': outcome.get('price'),
                                'update_time': last_update
                            })

        if not spread_lines or not total_lines:
            continue

        # Calculate consensus lines and line disagreement
        spread_values = [s['line'] for s in spread_lines if s['line'] is not None]
        total_values = [t['line'] for t in total_lines if t['line'] is not None]

        if not spread_values or not total_values:
            continue

        spread_consensus = sum(spread_values) / len(spread_values)
        spread_std = pd.Series(spread_values).std() if len(spread_values) > 1 else 0

        total_consensus = sum(total_values) / len(total_values)
        total_std = pd.Series(total_values).std() if len(total_values) > 1 else 0

        # Infer opening lines (earliest update time = opening)
        spread_sorted = sorted(spread_lines, key=lambda x: x.get('update_time', ''))
        total_sorted = sorted(total_lines, key=lambda x: x.get('update_time', ''))

        spread_open = spread_sorted[0]['line'] if spread_sorted else spread_consensus
        spread_current = spread_sorted[-1]['line'] if spread_sorted else spread_consensus

        total_open = total_sorted[0]['line'] if total_sorted else total_consensus
        total_current = total_sorted[-1]['line'] if total_sorted else total_consensus

        # Create splits record (without actual public betting percentages)
        # These would need to be fetched from a splits provider like Action Network
        splits = GameSplits(
            event_id=event_id,
            home_team=home_team,
            away_team=away_team,
            game_time=datetime.fromisoformat(commence_time.replace('Z', '+00:00')),
            spread_line=spread_consensus,
            spread_open=spread_open,
            spread_current=spread_current,
            total_line=total_consensus,
            total_open=total_open,
            total_current=total_current,
            source='the_odds_api_inferred'
        )

        # Detect RLM (will only work if we had real public betting percentages)
        splits = detect_reverse_line_movement(splits)

        splits_records.append({
            'event_id': splits.event_id,
            'home_team': splits.home_team,
            'away_team': splits.away_team,
            'game_time': splits.game_time,
            'spread_line': splits.spread_line,
            'spread_open': splits.spread_open,
            'spread_current': splits.spread_current,
            'spread_movement': splits.spread_current - splits.spread_open,
            'spread_line_std': spread_std,
            'total_line': splits.total_line,
            'total_open': splits.total_open,
            'total_current': splits.total_current,
            'total_movement': splits.total_current - splits.total_open,
            'total_line_std': total_std,
            'is_rlm_spread': splits.spread_rlm,
            'is_rlm_total': splits.total_rlm,
            'sharp_spread_side': splits.sharp_spread_side or '',
            'sharp_total_side': splits.sharp_total_side or '',
            'bookmaker_count': len(bookmakers),
            'source': splits.source
        })

    return pd.DataFrame(splits_records)


def extract_first_half_lines(odds_data_dir: Path):
    """
    Extract first-half lines from markets endpoint.

    The markets endpoint may have 1H spreads and totals.
    """
    fh_records = []

    # Read event markets files
    markets_files = list(odds_data_dir.glob('event_*_markets_*.json'))
    print(f"Processing {len(markets_files)} markets files...")

    for markets_file in markets_files:
        # Extract event ID from filename
        filename = markets_file.stem
        event_id = filename.split('_')[1]

        with open(markets_file, 'r') as f:
            try:
                markets_data = json.load(f)
            except:
                continue

        if not isinstance(markets_data, dict):
            continue

        bookmakers = markets_data.get('bookmakers', [])
        home_team = markets_data.get('home_team')
        away_team = markets_data.get('away_team')
        commence_time = markets_data.get('commence_time')

        # Look for first-half markets
        fh_spread_lines = []
        fh_total_lines = []

        for bm in bookmakers:
            bm_name = bm.get('key')

            for market in bm.get('markets', []):
                market_key = market.get('key')

                # First-half spread (various naming conventions)
                if 'h1' in market_key.lower() or 'first_half' in market_key.lower() or '1h' in market_key.lower():
                    if 'spread' in market_key.lower() or 'handicap' in market_key.lower():
                        for outcome in market.get('outcomes', []):
                            if outcome.get('name') == home_team:
                                fh_spread_lines.append({
                                    'bookmaker': bm_name,
                                    'line': outcome.get('point'),
                                    'price': outcome.get('price')
                                })

                    # First-half total
                    elif 'total' in market_key.lower() or 'over_under' in market_key.lower():
                        for outcome in market.get('outcomes', []):
                            if outcome.get('name') == 'Over' or outcome.get('description') == 'Over':
                                fh_total_lines.append({
                                    'bookmaker': bm_name,
                                    'line': outcome.get('point'),
                                    'price': outcome.get('price')
                                })

        if not fh_spread_lines and not fh_total_lines:
            continue  # No first-half markets for this game

        record = {
            'event_id': event_id,
            'home_team': home_team,
            'away_team': away_team,
            'commence_time': commence_time,
        }

        # Calculate consensus first-half lines
        if fh_spread_lines:
            spread_values = [s['line'] for s in fh_spread_lines if s['line'] is not None]
            if spread_values:
                record['fh_spread_line'] = sum(spread_values) / len(spread_values)
                record['fh_spread_line_std'] = pd.Series(spread_values).std() if len(spread_values) > 1 else 0
                record['fh_spread_bookmaker_count'] = len(fh_spread_lines)

        if fh_total_lines:
            total_values = [t['line'] for t in fh_total_lines if t['line'] is not None]
            if total_values:
                record['fh_total_line'] = sum(total_values) / len(total_values)
                record['fh_total_line_std'] = pd.Series(total_values).std() if len(total_values) > 1 else 0
                record['fh_total_bookmaker_count'] = len(fh_total_lines)

        if 'fh_spread_line' in record or 'fh_total_line' in record:
            fh_records.append(record)

    return pd.DataFrame(fh_records)


def extract_team_totals_lines(odds_data_dir: Path):
    """
    Extract team-specific totals lines from markets endpoint.

    Some books offer team totals (e.g., Lakers team total over/under 112.5).
    """
    team_totals_records = []

    # Read event markets files
    markets_files = list(odds_data_dir.glob('event_*_markets_*.json'))

    for markets_file in markets_files:
        filename = markets_file.stem
        event_id = filename.split('_')[1]

        with open(markets_file, 'r') as f:
            try:
                markets_data = json.load(f)
            except:
                continue

        if not isinstance(markets_data, dict):
            continue

        bookmakers = markets_data.get('bookmakers', [])
        home_team = markets_data.get('home_team')
        away_team = markets_data.get('away_team')
        commence_time = markets_data.get('commence_time')

        home_totals = []
        away_totals = []

        for bm in bookmakers:
            bm_name = bm.get('key')

            for market in bm.get('markets', []):
                market_key = market.get('key')
                market_desc = market.get('description', '').lower()

                # Team totals (various naming)
                if 'team_total' in market_key or 'team_over_under' in market_key:
                    for outcome in market.get('outcomes', []):
                        outcome_name = outcome.get('name')
                        if outcome_name == home_team or home_team in str(outcome.get('description', '')):
                            if outcome.get('description', '').lower() == 'over' or outcome.get('name') == 'Over':
                                home_totals.append({
                                    'bookmaker': bm_name,
                                    'line': outcome.get('point'),
                                    'price': outcome.get('price')
                                })
                        elif outcome_name == away_team or away_team in str(outcome.get('description', '')):
                            if outcome.get('description', '').lower() == 'over' or outcome.get('name') == 'Over':
                                away_totals.append({
                                    'bookmaker': bm_name,
                                    'line': outcome.get('point'),
                                    'price': outcome.get('price')
                                })

        if not home_totals and not away_totals:
            continue

        record = {
            'event_id': event_id,
            'home_team': home_team,
            'away_team': away_team,
            'commence_time': commence_time,
        }

        if home_totals:
            home_values = [t['line'] for t in home_totals if t['line'] is not None]
            if home_values:
                record['home_team_total_line'] = sum(home_values) / len(home_values)
                record['home_team_total_bookmaker_count'] = len(home_totals)

        if away_totals:
            away_values = [t['line'] for t in away_totals if t['line'] is not None]
            if away_values:
                record['away_team_total_line'] = sum(away_values) / len(away_values)
                record['away_team_total_bookmaker_count'] = len(away_totals)

        if 'home_team_total_line' in record or 'away_team_total_line' in record:
            team_totals_records.append(record)

    return pd.DataFrame(team_totals_records)


def main():
    parser = argparse.ArgumentParser(description='Process The Odds API data')
    parser.add_argument('--date', help='Date directory (YYYY-MM-DD), defaults to latest')
    args = parser.parse_args()

    print("="*80)
    print("THE ODDS API DATA PROCESSOR")
    print("="*80)

    # Find odds data directory
    if args.date:
        odds_dir = Path(f'data/raw/the_odds/{args.date}')
        if not odds_dir.exists():
            print(f"[ERROR] Directory not found: {odds_dir}")
            sys.exit(1)
    else:
        odds_dir = find_latest_odds_dir()
        if not odds_dir:
            print("[ERROR] No odds data directories found in data/raw/the_odds/")
            sys.exit(1)

    print(f"\nProcessing data from: {odds_dir}")

    # Extract betting splits
    print("\n=== Extracting Betting Splits ===")
    splits_df = extract_betting_splits(odds_dir)
    if not splits_df.empty:
        out_path = Path(settings.data_processed_dir) / 'betting_splits.csv'
        splits_df.to_csv(out_path, index=False)
        print(f"[OK] Saved {len(splits_df)} records to {out_path}")
        print(f"     Average spread movement: {splits_df['spread_movement'].mean():.2f}")
        print(f"     Average total movement: {splits_df['total_movement'].mean():.2f}")
    else:
        print("[WARN] No betting splits data extracted")

    # Extract first-half lines
    print("\n=== Extracting First-Half Lines ===")
    fh_df = extract_first_half_lines(odds_dir)
    if not fh_df.empty:
        out_path = Path(settings.data_processed_dir) / 'first_half_lines.csv'
        fh_df.to_csv(out_path, index=False)
        print(f"[OK] Saved {len(fh_df)} records to {out_path}")
        if 'fh_spread_line' in fh_df.columns:
            has_spread = fh_df['fh_spread_line'].notna().sum()
            print(f"     Games with 1H spread: {has_spread}")
        if 'fh_total_line' in fh_df.columns:
            has_total = fh_df['fh_total_line'].notna().sum()
            print(f"     Games with 1H total: {has_total}")
    else:
        print("[WARN] No first-half lines found in markets data")

    # Extract team totals
    print("\n=== Extracting Team Totals Lines ===")
    team_totals_df = extract_team_totals_lines(odds_dir)
    if not team_totals_df.empty:
        out_path = Path(settings.data_processed_dir) / 'team_totals_lines.csv'
        team_totals_df.to_csv(out_path, index=False)
        print(f"[OK] Saved {len(team_totals_df)} records to {out_path}")
        if 'home_team_total_line' in team_totals_df.columns:
            has_home = team_totals_df['home_team_total_line'].notna().sum()
            print(f"     Games with home team total: {has_home}")
        if 'away_team_total_line' in team_totals_df.columns:
            has_away = team_totals_df['away_team_total_line'].notna().sum()
            print(f"     Games with away team total: {has_away}")
    else:
        print("[WARN] No team totals lines found in markets data")

    print("\n" + "="*80)
    print("Done!")
    print("="*80)


if __name__ == '__main__':
    main()
