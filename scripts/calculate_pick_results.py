"""
Calculate hit/miss/push results for picks based on scoreboard data.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings


def load_scoreboard(scoreboard_file: Path) -> Dict:
    """Load scoreboard data from JSON file."""
    if not scoreboard_file.exists():
        raise FileNotFoundError(f"Scoreboard file not found: {scoreboard_file}")
    with open(scoreboard_file, 'r') as f:
        return json.load(f)


def find_game(scoreboard: Dict, away_team: str, home_team: str) -> Optional[Dict]:
    """Find game in scoreboard by team names."""
    # Normalize team names for matching
    team_mapping = {
        "TN State": "Tennessee State",
        "Tennessee State": "Tennessee State",
    }
    
    away_norm = team_mapping.get(away_team, away_team)
    home_norm = team_mapping.get(home_team, home_team)
    
    # Try exact match
    key = f"{away_norm}@{home_norm}"
    if key in scoreboard:
        return scoreboard[key]
    
    # Try reverse
    key = f"{home_norm}@{away_norm}"
    if key in scoreboard:
        game = scoreboard[key].copy()
        # Swap teams and adjust margins
        game['away_team'], game['home_team'] = game['home_team'], game['away_team']
        game['away_1h'], game['home_1h'] = game['home_1h'], game['away_1h']
        game['away_2h'], game['home_2h'] = game['home_2h'], game['away_2h']
        game['away_final'], game['home_final'] = game['home_final'], game['away_final']
        game['margin_fg'] = -game['margin_fg']
        game['margin_1h'] = -game['margin_1h']
        game['margin_2h'] = -game['margin_2h']
        return game
    
    # Try fuzzy matching
    for key, game_data in scoreboard.items():
        if away_norm.lower() in key.lower() and home_norm.lower() in key.lower():
            # Check if we need to swap
            if key.startswith(home_norm):
                game = game_data.copy()
                game['away_team'], game['home_team'] = game['home_team'], game['away_team']
                game['away_1h'], game['home_1h'] = game['home_1h'], game['away_1h']
                game['away_2h'], game['home_2h'] = game['home_2h'], game['away_2h']
                game['away_final'], game['home_final'] = game['home_final'], game['away_final']
                game['margin_fg'] = -game['margin_fg']
                game['margin_1h'] = -game['margin_1h']
                game['margin_2h'] = -game['margin_2h']
                return game
            return game_data
    
    return None


def calculate_result(pick_type: str, segment: str, pick_team: str, line: float, 
                    game_data: Dict, matchup: str) -> Tuple[str, str]:
    """
    Calculate hit/miss/push for a pick.
    
    Returns: (result_description, hit/miss/push)
    """
    away_team, home_team = matchup.split(' @ ')
    
    if pick_type == "Spread":
        if segment == "FG":
            actual_margin = game_data['margin_fg']
            if pick_team == away_team:
                # Away team spread: positive line means they're getting points
                if line > 0:  # e.g., +17
                    result_margin = actual_margin + line  # Memphis +17: actual -26, so -26 + 17 = -9 (lost by 9 more than line)
                else:  # e.g., -3.5
                    result_margin = actual_margin - line  # Ohio State -3.5: actual -1, so -1 - (-3.5) = 2.5 (won by 2.5)
            else:  # home team
                # actual_margin = away - home, so home_margin = -actual_margin
                home_margin = -actual_margin
                if line > 0:  # e.g., +4.5
                    result_margin = home_margin + line  # Magic +4.5: home_margin = -12, so -12 + 4.5 = -7.5 (lost)
                else:  # e.g., -3.5
                    # For -3.5, home needs to win by 3.5+, so result = home_margin - 3.5
                    result_margin = home_margin + line  # line is negative, so this is home_margin - 3.5
            
            if result_margin > 0:
                return f"Won by {abs(result_margin):.1f}", "HIT"
            elif result_margin < 0:
                return f"Lost by {abs(result_margin):.1f}", "MISS"
            else:
                return "Push", "PUSH"
        
        elif segment == "1H":
            actual_margin = game_data['margin_1h']
            if pick_team == away_team:
                if line > 0:
                    result_margin = actual_margin + line
                else:
                    result_margin = actual_margin - line
            else:  # home team
                home_margin = -actual_margin
                result_margin = home_margin + line  # Works for both positive and negative lines
            
            if result_margin > 0:
                return f"Won 1H by {abs(result_margin):.1f}", "HIT"
            elif result_margin < 0:
                return f"Lost 1H by {abs(result_margin):.1f}", "MISS"
            else:
                return "Push", "PUSH"
        
        elif segment == "2H":
            actual_margin = game_data['margin_2h']
            # For 2H spread, we need to check if the picked team is home or away
            # If Gonzaga is home and we picked Gonzaga -5, we need to check if home team won 2H by 5+
            if pick_team == away_team:
                # Picked away team
                if line > 0:  # e.g., +3
                    result_margin = actual_margin + line
                else:  # e.g., -5 (shouldn't happen for away, but handle it)
                    result_margin = actual_margin - line
            else:  # picked home team
                # actual_margin is away_score - home_score
                # For home team, margin is -actual_margin (home_score - away_score)
                home_margin = -actual_margin
                if line > 0:  # e.g., +3
                    result_margin = home_margin + line
                else:  # e.g., -5 (Gonzaga -5 means home team needs to win by 5+)
                    # home_margin - abs(line) = home_margin - 5
                    # If home_margin = 5 and line = -5, then 5 - 5 = 0 (push)
                    result_margin = home_margin + line  # line is negative, so this is home_margin - 5
            
            if result_margin > 0:
                return f"Won 2H by {abs(result_margin):.1f}", "HIT"
            elif result_margin < 0:
                return f"Lost 2H by {abs(result_margin):.1f}", "MISS"
            else:
                return "Push", "PUSH"
    
    elif pick_type == "Total":
        if segment == "FG":
            actual_total = game_data['total_fg']
            if "Over" in pick_team:
                if actual_total > line:
                    return f"Total {actual_total}", "HIT"
                elif actual_total < line:
                    return f"Total {actual_total}", "MISS"
                else:
                    return f"Total {actual_total}", "PUSH"
            else:  # Under
                if actual_total < line:
                    return f"Total {actual_total}", "HIT"
                elif actual_total > line:
                    return f"Total {actual_total}", "MISS"
                else:
                    return f"Total {actual_total}", "PUSH"
        
        elif segment == "1H":
            actual_total = game_data['total_1h']
            if "Over" in pick_team:
                if actual_total > line:
                    return f"1H Total {actual_total}", "HIT"
                elif actual_total < line:
                    return f"1H Total {actual_total}", "MISS"
                else:
                    return f"1H Total {actual_total}", "PUSH"
            else:  # Under
                if actual_total < line:
                    return f"1H Total {actual_total}", "HIT"
                elif actual_total > line:
                    return f"1H Total {actual_total}", "MISS"
                else:
                    return f"1H Total {actual_total}", "PUSH"
        
        elif segment == "2H":
            actual_total = game_data['total_2h']
            if "Over" in pick_team:
                if actual_total > line:
                    return f"2H Total {actual_total}", "HIT"
                elif actual_total < line:
                    return f"2H Total {actual_total}", "MISS"
                else:
                    return f"2H Total {actual_total}", "PUSH"
            else:  # Under
                if actual_total < line:
                    return f"2H Total {actual_total}", "HIT"
                elif actual_total > line:
                    return f"2H Total {actual_total}", "MISS"
                else:
                    return f"2H Total {actual_total}", "PUSH"
    
    elif pick_type == "Moneyline":
        if segment == "FG":
            if pick_team == away_team:
                won = game_data['away_final'] > game_data['home_final']
            else:
                won = game_data['home_final'] > game_data['away_final']
            
            if won:
                return "Won", "HIT"
            else:
                return "Lost", "MISS"
        
        elif segment == "1H":
            if pick_team == away_team:
                won = game_data['away_1h'] > game_data['home_1h']
            else:
                won = game_data['home_1h'] > game_data['away_1h']
            
            if won:
                return "Won 1H", "HIT"
            else:
                return "Lost 1H", "MISS"
    
    elif pick_type == "Team Total":
        # Team total - need to get the team's score
        if pick_team == away_team:
            team_score = game_data['away_final']
        else:
            team_score = game_data['home_final']
        
        if "Over" in str(line) or "o" in str(line).lower():
            # Extract the number
            line_num = float(str(line).replace('Over', '').replace('o', '').replace('O', '').strip())
            if team_score > line_num:
                return f"{team_score} points", "HIT"
            elif team_score < line_num:
                return f"{team_score} points", "MISS"
            else:
                return f"{team_score} points", "PUSH"
    
    return "Unknown", "TBD"


def load_picks(picks_file: Path) -> List[Dict]:
    """
    Load picks from JSON file. REQUIRED - no default picks.

    Picks file should be a JSON array of pick dictionaries with keys:
    - league, matchup, segment, pick, odds, stake, pick_type, pick_team, line

    Raises:
        FileNotFoundError: If picks file does not exist
        ValueError: If picks file is not provided
    """
    if not picks_file:
        raise ValueError("--picks-file is REQUIRED. No default picks allowed.")

    if not picks_file.exists():
        raise FileNotFoundError(f"Picks file not found: {picks_file}")

    with open(picks_file, 'r') as f:
        picks = json.load(f)

    if not picks:
        raise ValueError(f"Picks file is empty: {picks_file}")

    return picks


def calculate_risk_to_win(odds: int, stake: int) -> Tuple[float, float]:
    """Calculate risk and to_win amounts based on odds and stake (in thousands)."""
    base_unit = stake * 1000  # Convert to actual dollars
    
    if odds < 0:
        # Negative odds: to win base_unit, risk = (abs(odds) / 100) * base_unit
        risk = (abs(odds) / 100) * base_unit
        to_win = base_unit
    else:
        # Positive odds: risk base_unit, to win = (odds / 100) * base_unit
        risk = base_unit
        to_win = (odds / 100) * base_unit
    
    return risk, to_win


def main():
    parser = argparse.ArgumentParser(
        description="Calculate hit/miss/push results for picks based on scoreboard data."
    )
    parser.add_argument(
        "--scoreboard",
        type=Path,
        default=None,
        help="Path to scoreboard JSON file (default: data/processed/scoreboard_YYYYMMDD.json)",
    )
    parser.add_argument(
        "--picks-file",
        type=Path,
        required=True,
        help="Path to JSON file containing picks array (REQUIRED - no defaults)",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date string in YYYYMMDD format for default scoreboard filename (default: today)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to output results JSON file (optional)",
    )
    args = parser.parse_args()

    # Determine scoreboard file
    if args.scoreboard:
        scoreboard_file = args.scoreboard
    else:
        if args.date:
            date_str = args.date
        else:
            date_str = datetime.now().strftime("%Y%m%d")
        scoreboard_file = Path(settings.data_processed_dir) / f"scoreboard_{date_str}.json"

    try:
        scoreboard = load_scoreboard(scoreboard_file)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print(f"   Please provide a valid scoreboard file with --scoreboard")
        sys.exit(1)

    # Load picks - REQUIRED, no defaults
    try:
        all_picks = load_picks(args.picks_file)
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ {e}")
        sys.exit(1)

    # NBA scores (from previous data - can be removed if scoreboard has all data)
    nba_scores = {
        "New York Knicks@Orlando Magic": {
            "away_team": "New York Knicks",
            "home_team": "Orlando Magic",
            "away_1h": 33 + 38,  # Actually need quarter scores, but we have 1H total
            "home_1h": 36 + 28,
            "away_final": 132,
            "home_final": 120,
            "total_1h": 71 + 64,  # Actually 71-64 Knicks
            "total_fg": 252,
            "margin_fg": 12,  # Knicks won by 12
            "margin_1h": 7,  # Knicks won 1H by 7
        },
        "San Antonio Spurs@Oklahoma City Thunder": {
            "away_team": "San Antonio Spurs",
            "home_team": "Oklahoma City Thunder",
            "away_1h": 20 + 26,
            "home_1h": 31 + 18,
            "away_final": 111,
            "home_final": 109,
            "total_1h": 46 + 49,  # Actually 49-46 Thunder
            "total_fg": 220,
            "margin_fg": 2,  # Spurs won by 2
            "margin_1h": -3,  # Spurs lost 1H by 3
        }
    }
    
    # Add NBA scores to scoreboard (if not already present)
    for key, value in nba_scores.items():
        if key not in scoreboard:
            scoreboard[key] = value

    results = []

    for pick in all_picks:
        matchup = pick['matchup']
        away_team, home_team = matchup.split(' @ ')
        
        # Get game data
        if pick['league'] == "NCAAF":
            game_data = None  # No scores available
        else:
            game_data = find_game(scoreboard, away_team, home_team)
        
        # Calculate risk/to_win
        risk, to_win = calculate_risk_to_win(pick['odds'], pick['stake'])
        
        # Calculate result
        if game_data is None:
            result_desc = "TBD"
            result_status = "TBD"
        else:
            # Determine segment (FG, 1H, 2H)
            segment_type = "FG"
            if "1H" in pick['segment']:
                segment_type = "1H"
            elif "2H" in pick['segment']:
                segment_type = "2H"
            
            result_desc, result_status = calculate_result(
                pick['pick_type'],
                segment_type,
                pick['pick_team'],
                pick['line'],
                game_data,
                matchup
            )
        
        results.append({
            **pick,
            'risk': risk,
            'to_win': to_win,
            'result': result_desc,
            'status': result_status
        })
    
    # Save results to JSON if output specified
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Saved results to {args.output}")

    # Print results
    date_str = args.date or datetime.now().strftime("%Y%m%d")
    print("=" * 120)
    print(f"CONSOLIDATED {date_str} PICKS WITH RESULTS")
    print("=" * 120)
    print()

    # Get unique leagues from picks
    leagues = sorted(set(pick.get('league', 'UNKNOWN') for pick in all_picks))
    
    for league in leagues:
        league_picks = [r for r in results if r['league'] == league]
        if not league_picks:
            continue
        
        print(f"\n### {league} Picks:\n")
        print("| Date & Time (CST) | League | Matchup (Away @ Home) | Segment | Pick (Odds) | Risk ($) | To Win ($) | Result | Hit/Miss/Push |")
        print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        
        for r in league_picks:
            pick_str = f"{r['pick']} ({r['odds']:+d})"
            date_display = datetime.strptime(date_str, "%Y%m%d").strftime("%m/%d/%Y")
            print(f"| {date_display} TBD | {r['league']} | {r['matchup']} | {r['segment']} | {pick_str} | {r['risk']:,.2f} | {r['to_win']:,.2f} | {r['result']} | **{r['status']}** |")

        print()


if __name__ == "__main__":
    main()




