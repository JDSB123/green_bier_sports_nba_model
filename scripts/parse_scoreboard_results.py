"""
Parse HTML scoreboard file to extract game scores for NCAAM games.
"""
import argparse
import re
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings


def parse_scoreboard_html(html_file: str) -> Dict[str, Dict]:
    """
    Parse HTML scoreboard and extract game scores.
    
    Returns a dictionary mapping game keys (e.g., "Arizona@Alabama") to score data.
    """
    with open(html_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    games = {}
    
    # Pattern to find game sections
    # Each game has: team names, 1H scores, 2H scores, final scores
    # Look for ScoreboardScoreCell sections
    
    # Find all game sections
    game_sections = re.findall(
        r'<section class="Scoreboard[^>]*>.*?</section>',
        content,
        re.DOTALL
    )
    
    for section in game_sections:
        # Extract team names
        team_names = re.findall(
            r'<div class="ScoreCell__TeamName[^>]*>([^<]+)</div>',
            section
        )
        
        if len(team_names) < 2:
            continue
        
        away_team = team_names[0].strip()
        home_team = team_names[1].strip()
        
        # Extract scores - look for ScoreboardScoreCell__Value (1H and 2H) and ScoreCell__Score (final)
        # Scores appear in order: away 1H, away 2H, away final, home 1H, home 2H, home final
        score_values = re.findall(
            r'<div class="ScoreboardScoreCell__Value[^>]*>(\d+)</div>',
            section
        )
        final_scores = re.findall(
            r'<div class="ScoreCell__Score[^>]*>(\d+)</div>',
            section
        )
        
        if len(score_values) >= 4 and len(final_scores) >= 2:
            away_1h = int(score_values[0])
            away_2h = int(score_values[1])
            home_1h = int(score_values[2])
            home_2h = int(score_values[3])
            away_final = int(final_scores[0])
            home_final = int(final_scores[1])
            
            # Create game key
            game_key = f"{away_team}@{home_team}"
            
            games[game_key] = {
                'away_team': away_team,
                'home_team': home_team,
                'away_1h': away_1h,
                'away_2h': away_2h,
                'home_1h': home_1h,
                'home_2h': home_2h,
                'away_final': away_final,
                'home_final': home_final,
                'total_1h': away_1h + home_1h,
                'total_2h': away_2h + home_2h,
                'total_fg': away_final + home_final,
                'margin_fg': away_final - home_final,
                'margin_1h': away_1h - home_1h,
                'margin_2h': away_2h - home_2h,
            }
    
    return games


def normalize_team_name(name: str) -> str:
    """Normalize team names for matching."""
    name = name.strip()
    # Common variations
    replacements = {
        'Ohio St': 'Ohio State',
        'Ohio St.': 'Ohio State',
        'Cal State Bakersfield': 'Cal State Bakersfield',
        'CSU Bakersfield': 'Cal State Bakersfield',
        'Mississippi St': 'Mississippi State',
        'Mississippi St.': 'Mississippi State',
        'Miss St': 'Mississippi State',
        'Miss St.': 'Mississippi State',
        'TN State': 'Tennessee State',
        'Tennessee State': 'Tennessee State',
    }
    return replacements.get(name, name)


def find_game_scores(games: Dict, away_team: str, home_team: str) -> Optional[Dict]:
    """Find game scores by team names with fuzzy matching."""
    away_norm = normalize_team_name(away_team)
    home_norm = normalize_team_name(home_team)
    
    # Try exact match first
    game_key = f"{away_norm}@{home_norm}"
    if game_key in games:
        return games[game_key]
    
    # Try reverse (home @ away)
    game_key = f"{home_norm}@{away_norm}"
    if game_key in games:
        result = games[game_key].copy()
        # Swap the scores since we reversed the teams
        result['away_team'], result['home_team'] = result['home_team'], result['away_team']
        result['away_1h'], result['home_1h'] = result['home_1h'], result['away_1h']
        result['away_2h'], result['home_2h'] = result['home_2h'], result['away_2h']
        result['away_final'], result['home_final'] = result['home_final'], result['away_final']
        result['margin_fg'] = -result['margin_fg']
        result['margin_1h'] = -result['margin_1h']
        result['margin_2h'] = -result['margin_2h']
        return result
    
    # Try fuzzy matching
    for key, game_data in games.items():
        if (away_norm.lower() in key.lower() or key.lower().startswith(away_norm.lower())) and \
           (home_norm.lower() in key.lower() or key.lower().endswith(home_norm.lower())):
            return game_data
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Parse HTML scoreboard and extract game scores.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to HTML scoreboard file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to output JSON file (default: data/processed/scoreboard_YYYYMMDD.json)",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date string in YYYYMMDD format for output filename (default: today)",
    )
    args = parser.parse_args()

    html_file = args.input
    if not html_file.exists():
        print(f"❌ Input file not found: {html_file}")
        sys.exit(1)

    # Determine output file
    if args.output:
        output_file = args.output
    else:
        from datetime import datetime
        if args.date:
            date_str = args.date
        else:
            date_str = datetime.now().strftime("%Y%m%d")
        output_file = Path(settings.data_processed_dir) / f"scoreboard_{date_str}.json"

    print(f"Parsing scoreboard HTML from: {html_file}")
    games = parse_scoreboard_html(str(html_file))

    print(f"\nFound {len(games)} games:")
    for key, data in games.items():
        print(f"\n{key}:")
        print(f"  Final: {data['away_team']} {data['away_final']} - {data['home_final']} {data['home_team']}")
        print(f"  1H: {data['away_team']} {data['away_1h']} - {data['home_1h']} {data['home_team']} (Total: {data['total_1h']})")
        print(f"  2H: {data['away_team']} {data['away_2h']} - {data['home_2h']} {data['home_team']} (Total: {data['total_2h']})")

    # Save to JSON for easy access
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(games, f, indent=2)

    print(f"\n✅ Saved results to {output_file}")


if __name__ == "__main__":
    main()




