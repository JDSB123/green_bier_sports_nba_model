#!/usr/bin/env python3
"""
Display NBA predictions from predictions.json file.

This script parses and formats NBA predictions for display in
GitHub Actions workflows and other automated contexts.

Usage:
    python scripts/display_predictions.py predictions.json
"""

import json
import sys
from datetime import datetime
from pathlib import Path


def display_predictions(predictions_file: str = "predictions.json"):
    """
    Parse and display predictions from JSON file.
    
    Args:
        predictions_file: Path to JSON file containing NBA predictions.
                         Expected structure:
                         {
                             "analysis": [
                                 {
                                     "home_team": str,
                                     "away_team": str,
                                     "time_cst": str,
                                     "comprehensive_edge": {
                                         "full_game": {"spread": {...}, "total": {...}},
                                         "first_half": {"spread": {...}, "total": {...}}
                                     }
                                 }
                             ]
                         }
    
    Returns:
        int: 0 on success, 1 on error
        
    Side Effects:
        - Prints formatted predictions to stdout
        - Creates 'predictions_summary.json' with game count and team names
    """
    
    try:
        with open(predictions_file, 'r') as f:
            data = json.load(f)
        
        analysis = data.get('analysis', [])
        
        if not analysis:
            print("No games found for this date")
            return 0
        
        print("\n" + "="*100)
        print(f"NBA PREDICTIONS - {datetime.now().strftime('%Y-%m-%d %H:%M CST')}")
        print("="*100)
        print(f"\nFound {len(analysis)} game(s)\n")
        
        for game in analysis:
            home = game.get('home_team', '')
            away = game.get('away_team', '')
            time_cst = game.get('time_cst', 'TBD')
            edge_data = game.get('comprehensive_edge', {})
            
            print(f"\n{'='*100}")
            print(f"GAME: {away} @ {home}")
            print(f"TIME: {time_cst}")
            
            # Full Game picks
            fg = edge_data.get('full_game', {})
            if fg:
                print("\n  FULL GAME:")
                
                spread = fg.get('spread', {})
                if spread.get('pick'):
                    print(f"    SPREAD: {spread['pick']} {spread.get('pick_line', 0):+.1f}")
                    print(f"       Edge: {spread.get('edge', 0):+.1f} pts")
                    print(f"       Confidence: {spread.get('confidence', 0):.0%}")
                
                total = fg.get('total', {})
                if total.get('pick'):
                    print(f"    TOTAL: {total['pick']} {total.get('market_line', 0):.1f}")
                    print(f"       Edge: {abs(total.get('edge', 0)):.1f} pts")
                    print(f"       Confidence: {total.get('confidence', 0):.0%}")
            
            # First Half picks
            fh = edge_data.get('first_half', {})
            if fh:
                has_picks = any(fh.get(m, {}).get('pick') for m in ['spread', 'total'])
                if has_picks:
                    print("\n  FIRST HALF:")
                    
                    spread = fh.get('spread', {})
                    if spread.get('pick'):
                        print(f"    1H SPREAD: {spread['pick']} {spread.get('pick_line', 0):+.1f}")
                        print(f"       Edge: {spread.get('edge', 0):+.1f} pts")
                        print(f"       Confidence: {spread.get('confidence', 0):.0%}")
                    
                    total = fh.get('total', {})
                    if total.get('pick'):
                        print(f"    1H TOTAL: {total['pick']} {total.get('market_line', 0):.1f}")
                        print(f"       Edge: {abs(total.get('edge', 0)):.1f} pts")
                        print(f"       Confidence: {total.get('confidence', 0):.0%}")
        
        print("\n" + "="*100)
        
        # Save summary
        summary = {
            'date': datetime.now().isoformat(),
            'games_count': len(analysis),
            'games': [{'home': g.get('home_team'), 'away': g.get('away_team')} for g in analysis]
        }
        with open('predictions_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        return 0
            
    except FileNotFoundError:
        print(f"Error: File '{predictions_file}' not found")
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{predictions_file}': {e}")
        return 1
    except Exception as e:
        print(f"Error processing predictions: {e}")
        return 1


if __name__ == "__main__":
    predictions_file = sys.argv[1] if len(sys.argv) > 1 else "predictions.json"
    sys.exit(display_predictions(predictions_file))
