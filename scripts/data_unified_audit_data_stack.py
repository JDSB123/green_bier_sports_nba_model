#!/usr/bin/env python3
"""
COMPREHENSIVE DATA STACK AUDIT
Verifies all data sources are properly loaded, merged, and flowing through the pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def print_header(text):
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80)

def print_section(text):
    print(f"\n[{text}]")

def safe_read_csv(path, **kwargs):
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as e:
        return None

def main():
    print_header("COMPREHENSIVE DATA STACK AUDIT")
    issues = []
    
    # =========================================================================
    # PART 1: SOURCE DATA INVENTORY
    # =========================================================================
    print_header("PART 1: SOURCE DATA INVENTORY")
    
    # 1. Kaggle betting data
    print_section("1. KAGGLE BETTING DATA")
    kg = safe_read_csv("data/external/kaggle/nba_2008-2025.csv")
    if kg is not None:
        print(f"   Rows: {len(kg):,}")
        date_col = 'date' if 'date' in kg.columns else 'game_date'
        if date_col in kg.columns:
            print(f"   Date range: {kg[date_col].min()} to {kg[date_col].max()}")
        for col in ['spread', 'total', 'moneyline_home', 'moneyline_away']:
            if col in kg.columns:
                pct = kg[col].notna().mean() * 100
                print(f"   {col}: {pct:.1f}% coverage")
    else:
        issues.append("Kaggle data not found")
        print("   ERROR: Not found")
    
    # 2. TheOdds Derived
    print_section("2. THEODDS DERIVED LINES")
    to = safe_read_csv("data/historical/derived/theodds_lines.csv")
    if to is not None:
        print(f"   Rows: {len(to):,}")
        date_col = next((c for c in ['game_date', 'line_date', 'commence_time'] if c in to.columns), None)
        if date_col:
            print(f"   Date range: {to[date_col].min()} to {to[date_col].max()}")
        print(f"   Columns: {list(to.columns)}")
        for col in ['fg_spread_line', 'fg_total_line', 'fh_spread_line', 'fh_total_line']:
            if col in to.columns:
                pct = to[col].notna().mean() * 100
                print(f"   {col}: {pct:.1f}% coverage")
    else:
        issues.append("TheOdds derived not found")
        print("   ERROR: Not found")
    
    # 3. TheOdds 2025-26
    print_section("3. THEODDS 2025-26 ALL MARKETS")
    to26 = safe_read_csv("data/historical/the_odds/2025-2026/2025-2026_all_markets.csv")
    if to26 is not None:
        print(f"   Rows: {len(to26):,}")
        print(f"   Date range: {to26['commence_time'].min()} to {to26['commence_time'].max()}")
        
        # Check key columns
        key_cols = {
            'fg_spread': 'FG Spread',
            'fg_total': 'FG Total',
            'h1_spread': '1H Spread',
            'h1_total': '1H Total',
            'q1_spread': 'Q1 Spread',
            'q1_total': 'Q1 Total',
        }
        for col, name in key_cols.items():
            if col in to26.columns:
                pct = to26[col].notna().mean() * 100
                print(f"   {name}: {pct:.1f}% coverage")
            else:
                print(f"   {name}: NOT EXTRACTED")
        
        # Check h2h columns
        h2h_cols = [c for c in to26.columns if c.startswith('h2h_') and '_price' in c]
        h2h_h1_cols = [c for c in h2h_cols if 'h1' in c]
        h2h_fg_cols = [c for c in h2h_cols if 'h1' not in c]
        print(f"   FG h2h columns: {len(h2h_fg_cols)}")
        print(f"   1H h2h columns: {len(h2h_h1_cols)}")
    else:
        issues.append("TheOdds 2025-26 not found")
        print("   ERROR: Not found")
    
    # 4. 1H Exports
    print_section("4. 1H ODDS EXPORTS")
    for season in ['2023-2024', '2024-2025']:
        df = safe_read_csv(f"data/historical/exports/{season}_odds_1h.csv")
        if df is not None:
            print(f"   {season}: {len(df):,} rows")
        else:
            print(f"   {season}: NOT FOUND")
    
    # 5. Box Scores
    print_section("5. BOX SCORES (nba_api)")
    for f in ['box_scores_2023_24.csv', 'box_scores_2024_25.csv', 'box_scores_2025_26.csv']:
        df = safe_read_csv(f"data/raw/nba_api/{f}")
        if df is not None:
            print(f"   {f}: {len(df):,} games")
        else:
            print(f"   {f}: NOT FOUND")
            issues.append(f"Box scores {f} not found")
    
    # 6. Quarter Scores
    print_section("6. QUARTER SCORES (nba_api)")
    qs = safe_read_csv("data/raw/nba_api/quarter_scores_2025_26.csv")
    if qs is not None:
        print(f"   quarter_scores_2025_26.csv: {len(qs):,} games")
        cols = [c for c in qs.columns if 'q1' in c.lower() or 'q2' in c.lower() or '1h' in c.lower()]
        print(f"   Score columns: {cols}")
    else:
        issues.append("Quarter scores not found")
        print("   ERROR: Not found")
    
    # 7. NBA Database (wyattowalsh)
    print_section("7. NBA DATABASE (wyattowalsh)")
    gm = safe_read_csv("data/external/nba_database/game.csv")
    if gm is not None:
        print(f"   game.csv: {len(gm):,} games")
        date_col = next((c for c in ['GAME_DATE', 'game_date', 'date'] if c in gm.columns), None)
        if date_col:
            print(f"   Date range: {gm[date_col].min()} to {gm[date_col].max()}")
        else:
            print(f"   Columns: {list(gm.columns)[:10]}...")
    else:
        print("   game.csv: NOT FOUND (expected for injury matching)")
    
    # =========================================================================
    # PART 2: TRAINING DATA COVERAGE
    # =========================================================================
    print_header("PART 2: TRAINING DATA COVERAGE")
    
    td = safe_read_csv("data/processed/training_data.csv", low_memory=False)
    if td is not None:
        print(f"\nTotal games: {len(td):,}")
        print(f"Total columns: {len(td.columns)}")
        print(f"Date range: {td['game_date'].min()} to {td['game_date'].max()}")
        
        # Season breakdown
        print("\nBy Season:")
        for season in sorted(td['season'].unique()):
            cnt = len(td[td['season'] == season])
            print(f"   {season}: {cnt:,} games")
        
        # 9-Market Coverage
        print_section("9-MARKET COVERAGE")
        markets = {
            'FG Spread': ('fg_spread_line', 'fg_spread_covered'),
            'FG Total': ('fg_total_line', 'fg_total_over'),
            'FG Moneyline': ('fg_ml_home', 'fg_home_win'),
            '1H Spread': ('1h_spread_line', '1h_spread_covered'),
            '1H Total': ('1h_total_line', '1h_total_over'),
            '1H Moneyline': ('1h_ml_home', '1h_home_win'),
            'Q1 Spread': ('q1_spread_line', None),
            'Q1 Total': ('q1_total_line', None),
            'Q1 Moneyline': ('q1_ml_home', None),
        }
        
        for market, (line_col, label_col) in markets.items():
            line_cov = td[line_col].notna().sum() if line_col in td.columns else 0
            line_pct = line_cov / len(td) * 100
            
            if label_col and label_col in td.columns:
                label_cov = td[label_col].notna().sum()
                label_pct = label_cov / len(td) * 100
                print(f"   {market}: Line={line_pct:.1f}%, Label={label_pct:.1f}%")
            else:
                print(f"   {market}: Line={line_pct:.1f}%")
                
            if line_pct < 80:
                issues.append(f"{market} line coverage only {line_pct:.1f}%")
        
        # Key Features
        print_section("KEY FEATURE COVERAGE")
        features = [
            'home_elo', 'away_elo', 'elo_diff',
            'home_efg_pct', 'away_efg_pct',
            'home_off_rtg', 'away_off_rtg',
            'home_def_rtg', 'away_def_rtg',
            'home_rest_days', 'away_rest_days',
            'spread_move', 'total_move',
            'home_q1', 'away_q1', 'home_1h', 'away_1h',
            'home_score', 'away_score',
        ]
        
        for feat in features:
            if feat in td.columns:
                cov = td[feat].notna().sum()
                pct = cov / len(td) * 100
                status = "OK" if pct >= 95 else "WARN" if pct >= 80 else "LOW"
                print(f"   {feat}: {pct:.1f}% [{status}]")
            else:
                print(f"   {feat}: MISSING")
                issues.append(f"Feature {feat} missing from training data")
        
        # Score columns check
        print_section("SCORE & RESULT COLUMNS")
        score_cols = ['home_score', 'away_score', 'fg_total_actual', 'fg_margin']
        for col in score_cols:
            if col in td.columns:
                cov = td[col].notna().sum()
                pct = cov / len(td) * 100
                print(f"   {col}: {pct:.1f}%")
            else:
                print(f"   {col}: MISSING")
                issues.append(f"Score column {col} missing")
        
        # 1H Score columns
        print_section("1H SCORE COLUMNS")
        h1_cols = ['home_1h', 'away_1h', 'home_q1', 'home_q2', 'away_q1', 'away_q2']
        for col in h1_cols:
            if col in td.columns:
                cov = td[col].notna().sum()
                pct = cov / len(td) * 100
                print(f"   {col}: {pct:.1f}%")
            else:
                print(f"   {col}: MISSING")
        
    else:
        issues.append("Training data not found!")
        print("ERROR: Training data not found")
    
    # =========================================================================
    # PART 3: DATA FLOW VERIFICATION
    # =========================================================================
    print_header("PART 3: DATA FLOW VERIFICATION")
    
    if td is not None:
        # Check if data from each source is making it into training data
        print_section("Source -> Training Data Flow")
        
        # Kaggle data check (pre-2023 seasons would only come from Kaggle)
        pre2023 = td[td['season'].isin(['2020-2021', '2021-2022', '2022-2023'])]
        print(f"   Pre-2023 games (Kaggle): {len(pre2023):,}")
        
        # TheOdds derived check
        to_spread = td['fg_spread_line'].notna().sum()
        print(f"   Games with FG spread: {to_spread:,} ({to_spread/len(td)*100:.1f}%)")
        
        # Box scores check
        box_cov = td['home_efg_pct'].notna().sum() if 'home_efg_pct' in td.columns else 0
        print(f"   Games with box scores: {box_cov:,} ({box_cov/len(td)*100:.1f}%)")
        
        # ELO check
        elo_cov = td['home_elo'].notna().sum() if 'home_elo' in td.columns else 0
        print(f"   Games with ELO: {elo_cov:,} ({elo_cov/len(td)*100:.1f}%)")
        
        # 1H data check by season
        print_section("1H Data Coverage by Season")
        for season in sorted(td['season'].unique()):
            s_df = td[td['season'] == season]
            h1_spread = s_df['1h_spread_line'].notna().sum() if '1h_spread_line' in s_df.columns else 0
            h1_score = s_df['home_1h'].notna().sum() if 'home_1h' in s_df.columns else 0
            print(f"   {season}: 1H spread={h1_spread}/{len(s_df)} ({h1_spread/len(s_df)*100:.1f}%), 1H score={h1_score}/{len(s_df)} ({h1_score/len(s_df)*100:.1f}%)")
        
        # Q1 data check
        print_section("Q1 Data Coverage by Season")
        for season in sorted(td['season'].unique()):
            s_df = td[td['season'] == season]
            q1_spread = s_df['q1_spread_line'].notna().sum() if 'q1_spread_line' in s_df.columns else 0
            q1_score = s_df['home_q1'].notna().sum() if 'home_q1' in s_df.columns else 0
            print(f"   {season}: Q1 spread={q1_spread}/{len(s_df)} ({q1_spread/len(s_df)*100:.1f}%), Q1 score={q1_score}/{len(s_df)} ({q1_score/len(s_df)*100:.1f}%)")
    
    # =========================================================================
    # PART 4: BUILD SCRIPT VERIFICATION
    # =========================================================================
    print_header("PART 4: BUILD SCRIPT DATA LOADER CHECK")
    
    # Run the loaders individually to verify they work
    print_section("Testing Individual Loaders")
    
    sys.path.insert(0, 'scripts')
    try:
        from build_training_data_complete import (
            load_kaggle_betting,
            load_theodds_derived,
            load_theodds_2025_26,
            load_h1_exports,
            load_line_movement,
            load_box_scores_old,
            load_box_scores_new,
            load_quarter_scores_2526,
        )
        
        loaders = [
            ("load_kaggle_betting", load_kaggle_betting),
            ("load_theodds_derived", load_theodds_derived),
            ("load_theodds_2025_26", load_theodds_2025_26),
            ("load_h1_exports", load_h1_exports),
            ("load_line_movement", load_line_movement),
            ("load_box_scores_old", load_box_scores_old),
            ("load_box_scores_new", load_box_scores_new),
            ("load_quarter_scores_2526", load_quarter_scores_2526),
        ]
        
        for name, loader in loaders:
            try:
                df = loader()
                if df is not None and not df.empty:
                    print(f"   {name}: OK ({len(df):,} rows)")
                else:
                    print(f"   {name}: EMPTY")
                    issues.append(f"Loader {name} returns empty")
            except Exception as e:
                print(f"   {name}: ERROR - {str(e)[:50]}")
                issues.append(f"Loader {name} failed: {e}")
                
    except ImportError as e:
        print(f"   Could not import loaders: {e}")
    
    # =========================================================================
    # PART 5: ISSUES SUMMARY
    # =========================================================================
    print_header("AUDIT SUMMARY")
    
    if issues:
        print(f"\n⚠️  ISSUES FOUND: {len(issues)}")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
    else:
        print("\n✅ NO CRITICAL ISSUES FOUND")
    
    # Final coverage summary
    if td is not None:
        print("\n" + "-" * 40)
        print("FINAL COVERAGE MATRIX (9 Markets)")
        print("-" * 40)
        print(f"{'Market':<20} {'Line':<10} {'Label':<10}")
        print("-" * 40)
        
        markets_final = [
            ('FG Spread', 'fg_spread_line', 'fg_spread_covered'),
            ('FG Total', 'fg_total_line', 'fg_total_over'),
            ('FG ML', 'fg_ml_home', 'fg_home_win'),
            ('1H Spread', '1h_spread_line', '1h_spread_covered'),
            ('1H Total', '1h_total_line', '1h_total_over'),
            ('1H ML', '1h_ml_home', '1h_home_win'),
            ('Q1 Spread', 'q1_spread_line', None),
            ('Q1 Total', 'q1_total_line', None),
            ('Q1 ML', 'q1_ml_home', None),
        ]
        
        for name, line_col, label_col in markets_final:
            line_pct = td[line_col].notna().mean() * 100 if line_col in td.columns else 0
            label_pct = td[label_col].notna().mean() * 100 if label_col and label_col in td.columns else 0
            line_str = f"{line_pct:.1f}%" if line_pct > 0 else "N/A"
            label_str = f"{label_pct:.1f}%" if label_pct > 0 else "N/A"
            print(f"{name:<20} {line_str:<10} {label_str:<10}")

if __name__ == "__main__":
    main()
