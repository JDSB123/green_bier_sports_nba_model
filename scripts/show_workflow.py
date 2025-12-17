#!/usr/bin/env python3
"""
Visual workflow diagram showing the complete NBA V4.0 data pipeline.

Usage:
    python scripts/show_workflow.py
"""

def print_workflow():
    """Print visual workflow diagram."""
    
    workflow = """
================================================================================
                        NBA V4.0 COMPLETE WORKFLOW
================================================================================

DATA COLLECTION (Step 1)
------------------------
run_the_odds_tomorrow.py - Fetches ALL The Odds API endpoints
  Output: data/raw/the_odds/YYYY-MM-DD/
    - events_*.json, sport_odds_*.json, participants_*.json
    - scores_*.json, upcoming_odds_*.json
    - event_{id}_odds_*.json, event_{id}_markets_*.json
    - summary_*.json

fetch_injuries.py - Fetches injury data from ESPN + API-Basketball
  Output: data/processed/injuries.csv
    - player_name, team, status, injury_type
    - ppg, minutes_per_game, usage_rate, source

DATA PROCESSING (Step 2)
-------------------------
process_odds_data.py - Extracts betting splits and line movement
  Input: data/raw/the_odds/YYYY-MM-DD/*.json
  Outputs:
    - data/processed/betting_splits.csv (line movement, RLM detection)
    - data/processed/first_half_lines.csv (1H spreads/totals)
    - data/processed/team_totals_lines.csv

collect_api_basketball.py - Normalizes API-Basketball game outcomes
build_training_dataset.py - Links odds + outcomes into training_data.csv

MODEL TRAINING (Step 3)
------------------------
train_models.py - Trains base models
  Output: data/processed/models/
    - spreads_model.joblib
    - totals_model.joblib
    - moneyline_model.joblib

train_ensemble_models.py - Trains ensemble models (+2-5% accuracy)
  Models: Logistic Regression + Gradient Boosting
  Output: data/processed/models/
    - spreads_ensemble.joblib
    - totals_ensemble.joblib

PREDICTIONS (Step 4)
--------------------
predict.py - Generates predictions for upcoming games
  Inputs: models, injuries, betting_splits, recent games
  Output: data/processed/predictions.csv
    - spread_pred, spread_prob, spread_value
    - total_pred, total_prob, total_value
    - ml_pred, ml_prob, ml_value
review_predictions.py - Grades picks vs results (full game + 1H)

AUTOMATED PIPELINE
------------------
full_pipeline.py - Runs entire workflow
  Steps: 1. Fetch odds 2. Fetch injuries 3. Process odds
         4. Archive cache 5. Build training data 6. Train models
         7. Predict next slate 8. Review previous slate

  Usage: python scripts/full_pipeline.py [--skip-odds] [--skip-train]

================================================================================
DATA SEGMENTS SUMMARY
================================================================================

Segment              Source                What It Contains
-------------------- --------------------- ---------------------------------
Raw Odds             The Odds API          All endpoints for tomorrow
(data/raw/the_odds)                        - Events, odds, markets
                                           - Bookmaker lines
                                           - Home/away verification

Injuries             ESPN + API-Basketball Current injury reports
(processed/)                               - Player stats (PPG, MPG)
                                           - Impact assessment

Betting Splits       Processed from odds   Line movement analysis
(processed/)                               - Opening vs current
                                           - RLM detection
                                           - Bookmaker disagreement

Training Data        The Odds + API-B      Linked odds/outcomes + engineered features
(processed/)                               - Team stats / rolling windows
                                           - Odds consensus & line movement
                                           - Spread/total targets

Models               Trained from data     Pickled scikit-learn models
(processed/models/)                        - Base models
                                           - Ensemble models
                                           - Manifest with versions

Predictions          Generated from        Tomorrow's predictions
(processed/)         models + features     - Spreads, totals, ML
                                           - Probabilities and value

================================================================================
EXAMPLE: Lakers @ Celtics - 2025-12-04 7:30 PM EST
================================================================================

1. DATA COLLECTION
   The Odds API: Event id=abc123, Celtics -6.5, Total 223.5, ML -280/+230
   ESPN: Lakers - LeBron OUT, AD questionable | Celtics - Tatum probable

2. PROCESSING
   Betting Splits: Spread opened -6.0, moved to -6.5 (sharp on Celtics)
   Total: Opened 224.5, moved to 223.5
   RLM: Line moved with Celtics despite public on Lakers
   Injury Impact: Lakers missing 28 PPG (LeBron), questionable 24 PPG (AD)

3. FEATURE GENERATION
   Lakers L10: 112.3 PPG, 48.2 FG%, 110.2 OFF_RTG
   Celtics L10: 118.7 PPG, 49.8 FG%, 118.5 OFF_RTG
   H2H: Celtics won last 3, avg margin +8.3
   Injury adjustment: Lakers -8 points expected

4. PREDICTIONS
   Spread: Celtics cover -6.5 (68% prob, +0.05 value)
   Total: Under 223.5 (55% prob, -0.10 value)
   ML: Celtics win (78% prob, value on Lakers ML +230)

================================================================================
QUICK START
================================================================================

Daily Usage:
  1. Set API key: $env:THE_ODDS_API_KEY = 'your_key'
  2. Run: python scripts/full_pipeline.py
  3. Review: cat data/processed/pick_review_<DATE>.md
  4. View: cat data/processed/predictions.csv

Check Model Performance:
  python -c "from src.modeling.model_tracker import ModelTracker; print(ModelTracker().generate_report())"

================================================================================
"""
    
    print(workflow)


if __name__ == '__main__':
    print_workflow()
