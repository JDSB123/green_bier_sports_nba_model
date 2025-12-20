#!/usr/bin/env python3
"""Display executive summary from the API in a formatted table."""
import json
import sys
import urllib.request
from collections import defaultdict

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


def get_fire_tier(fire_rating: str) -> int:
    """Get numeric tier for sorting (3=ELITE, 2=STRONG, 1=GOOD)."""
    fire_count = fire_rating.count('🔥')
    return fire_count if fire_count >= 1 else 1


def main():
    import os
    api_port = os.getenv("NBA_API_PORT", "8090")
    api_base = os.getenv("NBA_API_URL", f"http://localhost:{api_port}")
    url = f"{api_base}/slate/today/executive"
    
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            data = json.loads(response.read().decode())
    except Exception as e:
        print(f"Error fetching data: {e}")
        sys.exit(1)
    
    plays = data['plays']
    
    # ==================== CONSOLIDATED SUMMARY ====================
    print()
    print("=" * 100)
    print(f"CONSOLIDATED SUMMARY - {data['date']} | {data['generated_at']}")
    print("=" * 100)
    
    # Count by tier
    elite_plays = [p for p in plays if get_fire_tier(p['fire_rating']) >= 3]
    strong_plays = [p for p in plays if get_fire_tier(p['fire_rating']) == 2]
    good_plays = [p for p in plays if get_fire_tier(p['fire_rating']) == 1]
    
    print(f"\nTOTAL PLAYS: {len(plays)}  |  ELITE: {len(elite_plays)}  |  STRONG: {len(strong_plays)}  |  GOOD: {len(good_plays)}")
    print()
    
    # Top picks by tier
    if elite_plays:
        print("-" * 100)
        print(">>> ELITE PICKS (Best Plays) <<<")
        print("-" * 100)
        for p in elite_plays:
            matchup_short = p['matchup'].split('@')[0].strip()[:20] + " @ " + p['matchup'].split('@')[1].strip()[:20] if '@' in p['matchup'] else p['matchup'][:42]
            print(f"  {p['time_cst']:<14} {matchup_short:<42} {p['period']:<3} {p['market']:<7} >> {p['pick']:<26} {p['pick_odds']:<6} | Edge: {p['edge']}")
    
    if strong_plays:
        print()
        print("-" * 100)
        print(">>> STRONG PICKS <<<")
        print("-" * 100)
        for p in strong_plays:
            matchup_short = p['matchup'].split('@')[0].strip()[:20] + " @ " + p['matchup'].split('@')[1].strip()[:20] if '@' in p['matchup'] else p['matchup'][:42]
            print(f"  {p['time_cst']:<14} {matchup_short:<42} {p['period']:<3} {p['market']:<7} >> {p['pick']:<26} {p['pick_odds']:<6} | Edge: {p['edge']}")
    
    # Games summary
    games = defaultdict(list)
    for p in plays:
        games[p['matchup']].append(p)
    
    print()
    print("-" * 100)
    print("PLAYS BY GAME:")
    print("-" * 100)
    for matchup, game_plays in games.items():
        elite_count = len([p for p in game_plays if get_fire_tier(p['fire_rating']) >= 3])
        strong_count = len([p for p in game_plays if get_fire_tier(p['fire_rating']) == 2])
        time_cst = game_plays[0]['time_cst']
        print(f"  {time_cst:<14} {matchup[:60]:<60} | {len(game_plays)} plays ({elite_count} ELITE, {strong_count} STRONG)")
    
    print()
    print("=" * 100)
    print()
    
    # ==================== DETAILED TABLE ====================
    print("=" * 140)
    print(f"DETAILED BETTING CARD")
    print("=" * 140)
    print()
    
    # Header
    header = f"{'TIME CST':<16} {'MATCHUP':<55} {'PER':<4} {'MKT':<7} {'PICK':<28} {'ODDS':<7} {'MODEL':<12} {'LINE':<8} {'EDGE':<10} {'FIRE'}"
    print(header)
    print("-" * 140)
    
    # Plays
    for play in plays:
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
