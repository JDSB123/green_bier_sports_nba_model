#!/usr/bin/env python3
"""Display executive summary from the API in a formatted table."""
import json
import sys
import urllib.request

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def emoji_to_text(fire_rating: str) -> str:
    """Convert fire emoji rating to text for Windows compatibility."""
    fire_count = fire_rating.count('🔥')
    if fire_count >= 3:
        return "***ELITE***"
    elif fire_count >= 2:
        return "**STRONG**"
    else:
        return "*GOOD*"


def main():
    url = "http://localhost:8090/slate/today/executive"
    
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            data = json.loads(response.read().decode())
    except Exception as e:
        print(f"Error fetching data: {e}")
        sys.exit(1)
    
    print()
    print("=" * 140)
    print(f"NBA EXECUTIVE BETTING CARD - {data['date']} | Generated: {data['generated_at']}")
    print(f"v{data['version']} | Total Plays: {data['total_plays']}")
    print("=" * 140)
    print()
    
    # Header
    header = f"{'TIME CST':<16} {'MATCHUP':<55} {'PER':<4} {'MKT':<7} {'PICK':<28} {'ODDS':<7} {'MODEL':<12} {'LINE':<8} {'EDGE':<10} {'FIRE'}"
    print(header)
    print("-" * 140)
    
    # Plays
    for play in data['plays']:
        matchup = play['matchup'][:53]
        pick = play['pick'][:26]
        fire = emoji_to_text(play['fire_rating'])
        
        row = f"{play['time_cst']:<16} {matchup:<55} {play['period']:<4} {play['market']:<7} {pick:<28} {play['pick_odds']:<7} {play['model_prediction']:<12} {play['market_line']:<8} {play['edge']:<10} {fire}"
        print(row)
    
    print()
    print("=" * 140)
    print("LEGEND:")
    print("  ***ELITE***  = 70%+ confidence AND 5+ pt edge")
    print("  **STRONG**   = 60%+ confidence AND 3+ pt edge")
    print("  *GOOD*       = Passes all filters")
    print()
    print("PERIODS: FG = Full Game, 1H = First Half")
    print("MARKETS: SPREAD = Point Spread, TOTAL = Over/Under, ML = Moneyline")
    print("=" * 140)

if __name__ == "__main__":
    main()
