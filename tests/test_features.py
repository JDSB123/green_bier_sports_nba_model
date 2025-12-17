import unittest
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modeling.features import FeatureEngineer

class TestFeatureEngineer(unittest.TestCase):
    def setUp(self):
        self.engineer = FeatureEngineer(lookback=5)
        
        # Create sample games data
        dates = [
            datetime(2023, 10, 25),
            datetime(2023, 10, 27),
            datetime(2023, 10, 29),
            datetime(2023, 11, 1),
            datetime(2023, 11, 3),
        ]
        
        self.games_data = [
            # Game 1: BOS (Home) vs NYK (Away) - BOS wins 108-104
            {"date": dates[0], "home_team": "BOS", "away_team": "NYK", "home_score": 108, "away_score": 104},
            # Game 2: BOS (Home) vs MIA (Away) - BOS wins 119-111
            {"date": dates[1], "home_team": "BOS", "away_team": "MIA", "home_score": 119, "away_score": 111},
            # Game 3: WAS (Home) vs BOS (Away) - BOS wins 126-107
            {"date": dates[2], "home_team": "WAS", "away_team": "BOS", "home_score": 107, "away_score": 126},
            # Game 4: BOS (Home) vs IND (Away) - BOS wins 155-104
            {"date": dates[3], "home_team": "BOS", "away_team": "IND", "home_score": 155, "away_score": 104},
            # Game 5: BKN (Home) vs BOS (Away) - BOS wins 124-114
            {"date": dates[4], "home_team": "BKN", "away_team": "BOS", "home_score": 114, "away_score": 124},
        ]
        
        self.games_df = pd.DataFrame(self.games_data)
        self.games_df["date"] = pd.to_datetime(self.games_df["date"])

    def test_compute_team_rolling_stats(self):
        # Test stats for BOS as of Nov 4th (after all 5 games)
        as_of_date = pd.Timestamp("2023-11-04")
        stats = self.engineer.compute_team_rolling_stats(self.games_df, "BOS", as_of_date)
        
        self.assertIsNotNone(stats)
        self.assertEqual(stats["games_played"], 5)
        self.assertEqual(stats["win_pct"], 1.0)  # Won all 5
        
        # Points scored: 108, 119, 126, 155, 124 -> Avg: 126.4
        expected_ppg = (108 + 119 + 126 + 155 + 124) / 5
        self.assertAlmostEqual(stats["ppg"], expected_ppg)
        
        # Points allowed: 104, 111, 107, 104, 114 -> Avg: 108.0
        expected_papg = (104 + 111 + 107 + 104 + 114) / 5
        self.assertAlmostEqual(stats["papg"], expected_papg)
        
        # Avg margin: 126.4 - 108.0 = 18.4
        self.assertAlmostEqual(stats["margin"], 18.4)

    def test_compute_rest_days(self):
        # Test rest days for BOS before Game 2 (Oct 27)
        # Previous game was Oct 25. Rest days = (27 - 25) - 1 = 1 day
        game_date = pd.Timestamp("2023-10-27")
        rest = self.engineer.compute_rest_days(self.games_df, "BOS", game_date)
        self.assertEqual(rest, 1)
        
        # Test rest days for BOS before Game 3 (Oct 29)
        # Previous game was Oct 27. Rest days = (29 - 27) - 1 = 1 day
        game_date = pd.Timestamp("2023-10-29")
        rest = self.engineer.compute_rest_days(self.games_df, "BOS", game_date)
        self.assertEqual(rest, 1)
        
        # Test back-to-back scenario
        # Add a game on Oct 30
        b2b_game = pd.DataFrame([{
            "date": pd.Timestamp("2023-10-30"),
            "home_team": "BOS", "away_team": "DET",
            "home_score": 100, "away_score": 90
        }])
        df_with_b2b = pd.concat([self.games_df, b2b_game])
        
        # Check rest for Oct 30 game (prev game Oct 29)
        # Rest = (30 - 29) - 1 = 0
        rest = self.engineer.compute_rest_days(df_with_b2b, "BOS", pd.Timestamp("2023-10-30"))
        self.assertEqual(rest, 0)

    def test_insufficient_data(self):
        # Test with only 1 game (returns empty dict when <3 games)
        short_df = self.games_df.head(1)
        as_of_date = pd.Timestamp("2023-11-04")
        stats = self.engineer.compute_team_rolling_stats(short_df, "BOS", as_of_date)
        self.assertEqual(stats, {})

if __name__ == '__main__':
    unittest.main()
