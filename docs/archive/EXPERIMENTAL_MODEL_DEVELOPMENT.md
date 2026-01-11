# Experimental Model Development Framework

## Overview

This document outlines a safe approach for developing improved NBA prediction models using ALL available historical data **without affecting production**.

```
┌─────────────────────────────────────────────────────────────────────┐
│                   PARALLEL DEVELOPMENT TRACKS                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  PRODUCTION (frozen)              EXPERIMENTAL (active development) │
│  ├── models/production/           ├── models/experimental/          │
│  ├── Version: NBA_v33.0.11.0      ├── Version: NBA_vX.0.0.0-exp     │
│  ├── Training: 80 days            ├── Training: 17 seasons          │
│  ├── Deployed to Azure            ├── Local testing only            │
│  └── DO NOT MODIFY                └── Safe to iterate               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Current State

### Production Model (DO NOT TOUCH)
```
Version: NBA_v33.0.11.0
Location: models/production/
Training Data: 2024-10-02 to 2025-12-20 (~80 days)
Features: 55 per model
Performance:
  - FG Spread: 60.6% accuracy, 15.7% ROI
  - FG Total: 59.2% accuracy, 13.1% ROI
  - 1H Spread: 55.9% accuracy, 8.2% ROI
  - 1H Total: 58.1% accuracy, 11.4% ROI
```

### Available Historical Data (UNUSED)
```
data/external/kaggle/nba_2008-2025.csv     # 17 seasons, ~20,000 games
data/historical/exports/*_odds_*.csv       # 2023-2025 with betting lines
data/historical/elo/                       # ELO ratings (1946-2015)
data/historical/derived/theodds_lines.csv  # Consensus lines
```

---

## Directory Structure

```
models/
├── production/                    # FROZEN - deployed to Azure
│   ├── fg_spread_model.joblib
│   ├── fg_total_model.joblib
│   ├── 1h_spread_model.joblib
│   ├── 1h_total_model.joblib
│   └── model_pack.json
│
└── experimental/                  # NEW - safe for experimentation
    ├── v1_expanded_data/          # Experiment: More training data
    │   ├── fg_spread_model.joblib
    │   └── experiment.json        # Hypothesis, results, notes
    ├── v2_feature_engineering/    # Experiment: New features
    └── v3_algorithm_tuning/       # Experiment: XGBoost, etc.

data/
├── processed/                     # Current season (regenerated)
└── experimental/                  # NEW - expanded training data
    ├── training_data_full.csv     # All 17 seasons
    ├── training_data_2020_2025.csv # 5 recent seasons
    └── manifest.json              # Data lineage
```

---

## Step-by-Step Approach

### Phase 1: Create Experimental Infrastructure

```powershell
# Create directories
mkdir models/experimental
mkdir data/experimental

# Create git branch for experimentation
git checkout -b experiment/expanded-training-data
```

### Phase 2: Build Expanded Training Data

```python
# scripts/build_experimental_training_data.py

"""
Build training data from ALL available historical sources.

Data Sources:
1. Kaggle (2008-2025): Game outcomes, quarter scores
2. The Odds API (2023-2025): Betting lines
3. ELO ratings (historical): Team strength

Output:
- data/experimental/training_data_full.csv
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

def build_experimental_training_data():
    """Build expanded training dataset from all historical sources."""
    
    # 1. Load Kaggle data (17 seasons)
    kaggle = pd.read_csv(DATA_DIR / "external/kaggle/nba_2008-2025.csv")
    print(f"Kaggle: {len(kaggle):,} games (2008-2025)")
    
    # 2. Load The Odds API lines (2023-2025)
    theodds_lines = pd.read_csv(DATA_DIR / "historical/derived/theodds_lines.csv")
    print(f"The Odds API lines: {len(theodds_lines):,} events")
    
    # 3. Load ELO ratings
    elo = pd.read_csv(DATA_DIR / "historical/elo/fivethirtyeight_elo_historical.csv")
    print(f"ELO ratings: {len(elo):,} rows")
    
    # 4. Merge: Kaggle + ELO (for 2008-2015)
    # Note: ELO data ends at 2015, so we can't use it for recent games
    
    # 5. Merge: Kaggle + The Odds lines (for 2023-2025)
    # This gives us games WITH betting lines for training
    
    # 6. Feature engineering
    # - Calculate rolling stats (PPG, PAPG, win%)
    # - Calculate rest days
    # - Calculate H2H history
    # - Calculate 1H outcomes from quarter scores
    
    # 7. Create labels
    # - spread_covered (based on actual margin vs line)
    # - total_over (based on actual total vs line)
    # - 1h_spread_covered, 1h_total_over
    
    # 8. Save
    output_path = DATA_DIR / "experimental/training_data_full.csv"
    # training_df.to_csv(output_path, index=False)
    
    return output_path
```

### Phase 3: Train Experimental Models

```python
# scripts/train_experimental_models.py

"""
Train models on expanded historical data.

Key differences from production training:
- Uses models/experimental/ directory
- Trains on 17 seasons (not 80 days)
- Validates using walk-forward across seasons
- Does NOT overwrite production models
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from src.modeling.models import SpreadsModel, TotalsModel

EXPERIMENTAL_DIR = Path("models/experimental")

def train_experimental(experiment_name: str, training_data_path: str):
    """Train experimental models safely."""
    
    exp_dir = EXPERIMENTAL_DIR / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Load expanded training data
    import pandas as pd
    df = pd.read_csv(training_data_path)
    
    # Walk-forward validation across seasons
    # Train on seasons 1-N, validate on season N+1
    seasons = sorted(df['season'].unique())
    
    results = []
    for i in range(len(seasons) - 1):
        train_seasons = seasons[:i+1]
        test_season = seasons[i+1]
        
        train_df = df[df['season'].isin(train_seasons)]
        test_df = df[df['season'] == test_season]
        
        # Train model
        model = SpreadsModel()
        model.train(train_df)
        
        # Evaluate
        accuracy = model.evaluate(test_df)
        results.append({
            'train_seasons': train_seasons,
            'test_season': test_season,
            'accuracy': accuracy
        })
    
    # Save final model (trained on all but last season)
    final_model = SpreadsModel()
    final_model.train(df[df['season'] != seasons[-1]])
    final_model.save(exp_dir / "fg_spread_model.joblib")
    
    # Save experiment metadata
    import json
    with open(exp_dir / "experiment.json", "w") as f:
        json.dump({
            "name": experiment_name,
            "training_data": training_data_path,
            "seasons_used": seasons,
            "walk_forward_results": results,
            "hypothesis": "More training data improves generalization",
        }, f, indent=2)
    
    return results
```

### Phase 4: Compare to Production

```python
# scripts/compare_experimental_to_production.py

"""
Compare experimental models against production on held-out data.

Uses current season (2024-2025) as hold-out test set.
Both models evaluated on SAME games for fair comparison.
"""

def compare_models():
    """Compare experimental vs production on held-out data."""
    
    from src.modeling.models import SpreadsModel
    import joblib
    
    # Load production model (DO NOT RETRAIN)
    prod_model = joblib.load("models/production/fg_spread_model.joblib")
    
    # Load experimental model
    exp_model = joblib.load("models/experimental/v1_expanded_data/fg_spread_model.joblib")
    
    # Load held-out test data (current season only)
    test_df = pd.read_csv("data/processed/current_season_games.csv")
    
    # Evaluate both
    prod_accuracy = prod_model.evaluate(test_df)
    exp_accuracy = exp_model.evaluate(test_df)
    
    print(f"Production Model: {prod_accuracy:.1%}")
    print(f"Experimental Model: {exp_accuracy:.1%}")
    print(f"Difference: {(exp_accuracy - prod_accuracy)*100:+.1f}%")
    
    # Statistical significance test
    from scipy import stats
    # ... bootstrap or paired t-test
```

### Phase 5: Promote to Production (IF BETTER)

```powershell
# Only if experimental model is significantly better

# 1. Update version
$NEW_VERSION = "NBA_v34.0.0.0"

# 2. Copy experimental to production
Copy-Item models/experimental/v1_expanded_data/*.joblib models/production/

# 3. Update model_pack.json with new metadata

# 4. Commit and tag
git add models/production/
git commit -m "Promote experimental model to production - $NEW_VERSION"
git tag -a $NEW_VERSION -m "Trained on 17 seasons"

# 5. Deploy to Azure
pwsh ./infra/nba/deploy.ps1 -Tag $NEW_VERSION
```

---

## Experiment Ideas

### Experiment 1: Expanded Training Data
```
Hypothesis: Training on 17 seasons instead of 80 days improves generalization
Data: data/external/kaggle/nba_2008-2025.csv
Challenge: Older seasons have different rules/pace, may not be relevant
Mitigation: Weight recent seasons more heavily
```

### Experiment 2: Feature Engineering
```
Hypothesis: Adding new features improves accuracy
Features to try:
- Team pace (possessions per game)
- 3-point shooting rate trends
- Opponent-adjusted stats
- Travel distance / timezone changes
- Back-to-back fatigue with travel
Data: Same as production
```

### Experiment 3: Algorithm Comparison
```
Hypothesis: Gradient boosting outperforms logistic regression
Algorithms to try:
- XGBoost
- LightGBM
- CatBoost
- Neural network (simple MLP)
Data: Same as production
```

### Experiment 4: Recent Seasons Only
```
Hypothesis: Only last 5 seasons (2020-2025) are relevant
Rationale: Game has changed (3-point revolution, pace, etc.)
Data: Filter to 2020-2025 only
```

---

## Safety Checklist

### Before Starting Experiments
- [ ] Production models backed up to Azure Blob
- [ ] Git tag for current production version exists
- [ ] Working on feature branch (not main)
- [ ] Experimental directories created

### During Development
- [ ] Never modify files in `models/production/`
- [ ] All experimental models go to `models/experimental/`
- [ ] Document hypothesis in experiment.json
- [ ] Log all results for comparison

### Before Promotion
- [ ] Experimental model beats production by >2% accuracy
- [ ] Statistical significance confirmed (p < 0.05)
- [ ] Backtest on multiple seasons shows consistency
- [ ] No data leakage in training pipeline
- [ ] Code reviewed by second person

---

## Quick Commands

```powershell
# Start experiment branch
git checkout -b experiment/expanded-training-data

# Build experimental training data
python scripts/build_experimental_training_data.py

# Train experimental models
python scripts/train_experimental_models.py --experiment v1_expanded_data

# Compare to production
python scripts/compare_experimental_to_production.py

# If better, promote
# (Only after thorough validation)
```
