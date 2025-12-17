# Spread Prediction Improvement Recommendations

## Executive Summary
Current spread model achieves 54.5% accuracy (+4.1% ROI), but analysis reveals critical issues that, when addressed, could significantly improve performance.

## Critical Issues Identified

### 1. **MODEL CALIBRATION FAILURE** ⚠️ HIGHEST PRIORITY
**Problem:** Model is severely miscalibrated at high confidence
- Predictions >60% confidence: Expected 78.3% → Actual 47.6% win rate
- Model is overconfident and wrong when it's most certain
- This is WORSE than random guessing

**Root Cause:** Likely due to:
- Class imbalance in training data
- Overfitting on small sample sizes
- Lack of probability calibration post-training

**Solutions:**
```python
# 1. Add Platt Scaling / Isotonic Regression for calibration
from sklearn.calibration import CalibratedClassifierCV

spreads_model = CalibratedClassifierCV(
    LogisticRegression(max_iter=1000, C=0.1),
    method='isotonic',  # or 'sigmoid' for Platt scaling
    cv=5
)

# 2. Use proper calibration metrics during validation
from sklearn.calibration import calibration_curve
# Plot and monitor calibration curves

# 3. Add sample weights to balance home/away predictions
sample_weights = compute_sample_weight('balanced', y_train)
spreads_model.fit(X_train, y_train, sample_weight=sample_weights)
```

**Expected Impact:** +3-5% accuracy improvement on high-confidence bets

---

### 2. **SMALL SPREAD PERFORMANCE** (3-6 points: 42.2% accuracy)
**Problem:** Model performs WORSE than random on competitive games

**Root Cause:**
- These are the hardest games to predict (most efficient market)
- Model lacks features to differentiate close matchups
- Noise dominates in close games

**Solutions:**

#### A. Add Clutch Performance Features
```python
def add_clutch_features(team, historical_df):
    """Add features for close game performance."""
    close_games = historical_df[abs(historical_df['home_margin']) <= 5]

    team_close = close_games[
        (close_games['home_team'] == team) |
        (close_games['away_team'] == team)
    ]

    features = {}
    features['close_game_win_pct'] = calculate_win_pct(team_close, team)
    features['clutch_ppg'] = calculate_clutch_scoring(team_close, team)
    features['close_game_margin'] = calculate_avg_margin(team_close, team)

    return features
```

#### B. Add Lineup Strength Features
```python
def add_lineup_features(team, game_date):
    """Factor in starting lineup quality and availability."""
    features = {}
    features['star_availability'] = check_star_players(team, game_date)
    features['bench_depth_rating'] = calculate_bench_strength(team)
    features['starting_5_net_rating'] = get_lineup_net_rating(team)

    return features
```

#### C. **SIMPLE FIX: Don't bet small spreads (3-6)**
- Filter out these games entirely
- Focus betting on pick-ems (0-3) and larger spreads where model performs well

**Expected Impact:** +5-8% accuracy if you avoid 3-6 point spreads

---

### 3. **HOME BIAS** (Home: 51.7% vs Away: 56.8%)
**Problem:** Model predicts home covers poorly

**Solutions:**

#### A. Separate Models by Bet Type
```python
# Train separate models for home and away predictions
home_model = train_model(features[features['bet_side'] == 'home'])
away_model = train_model(features[features['bet_side'] == 'away'])

# Use appropriate model based on prediction
if predicting_home_cover:
    prob = home_model.predict_proba(X)
else:
    prob = away_model.predict_proba(X)
```

#### B. Add Home Court Advantage Variance
```python
def get_dynamic_hca(home_team, away_team, season_date):
    """Calculate dynamic home court advantage."""
    base_hca = get_home_court_advantage(home_team)  # Current: 3.0 fixed

    # Adjust for factors
    if is_back_to_back(home_team, season_date):
        base_hca -= 1.5  # Tired home team

    if is_rivalry_game(home_team, away_team):
        base_hca += 1.0  # Extra energy

    if is_early_season(season_date):
        base_hca *= 0.8  # HCA weaker early

    return base_hca
```

**Expected Impact:** +2-3% accuracy on home predictions

---

### 4. **EARLY SEASON WEAKNESS** (51.8% → 57.3% over time)
**Problem:** Model performs poorly with limited data

**Solutions:**

#### A. Use Prior Season Data for Training
```python
def load_training_data_with_carryover():
    """Include last N games from prior season."""
    current_season = load_current_season_games()
    prior_season = load_prior_season_games().tail(200)  # Last 200 games

    # Combine with time decay weight
    prior_season['sample_weight'] = 0.5  # Downweight old data
    current_season['sample_weight'] = 1.0

    return pd.concat([prior_season, current_season])
```

#### B. Require More Training Games Early
```python
# In backtest, use adaptive minimum:
if games_played < 100:
    min_training_games = 150  # Require more historical data
else:
    min_training_games = 50
```

**Expected Impact:** +3-4% early season accuracy

---

## Additional Feature Improvements

### 5. Add Advanced Features

#### A. Opponent-Adjusted Metrics
```python
def calculate_opponent_adjusted_stats(team, historical_df):
    """Adjust team stats for opponent strength."""
    team_games = get_team_games(team, historical_df)

    features = {}
    features['adj_offensive_rating'] = calculate_adj_ortg(team_games)
    features['adj_defensive_rating'] = calculate_adj_drtg(team_games)
    features['strength_of_schedule'] = calculate_sos(team_games)

    return features
```

#### B. Situational Context
```python
def add_situational_features(game):
    """Add game-specific context."""
    features = {}

    # Playoff implications
    features['playoff_race_game'] = is_playoff_implications(game)
    features['standings_gap'] = get_standings_difference(game)

    # Motivation factors
    features['revenge_game'] = is_revenge_game(game)
    features['statement_game'] = is_statement_game(game)

    # Schedule context
    features['days_since_last_loss'] = get_days_since_loss(game)
    features['winning_streak'] = get_current_streak(game)

    return features
```

#### C. Betting Market Features (if available)
```python
def add_market_features(game):
    """Incorporate betting market intelligence."""
    features = {}

    # Line movement
    features['line_movement'] = get_line_movement(game)
    features['opening_vs_closing'] = get_line_diff(game)

    # Public betting
    features['public_betting_pct'] = get_public_betting(game)
    features['sharp_money_indicator'] = detect_sharp_action(game)

    # Reverse line movement
    features['is_rlm'] = detect_rlm(game)

    return features
```

---

## Model Architecture Improvements

### 6. Advanced ML Techniques

#### A. Ensemble Models
```python
from sklearn.ensemble import VotingClassifier
import xgboost as xgb

# Combine multiple models
spreads_ensemble = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(max_iter=1000, C=0.1)),
        ('gb', GradientBoostingClassifier(n_estimators=100)),
        ('xgb', xgb.XGBClassifier(n_estimators=100)),
    ],
    voting='soft',  # Use probability averaging
    weights=[1, 2, 2]  # Weight stronger models more
)
```

#### B. Neural Network for Complex Patterns
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

def build_spread_nn(input_dim):
    """Neural network for spread prediction."""
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(32, activation='relu'),
        Dropout(0.2),

        Dense(1, activation='sigmoid')  # Binary classification
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model
```

#### C. Hyperparameter Tuning
```python
from sklearn.model_selection import RandomizedSearchCV

param_distributions = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'min_samples_split': [2, 5, 10],
}

search = RandomizedSearchCV(
    GradientBoostingClassifier(),
    param_distributions,
    n_iter=50,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

search.fit(X_train, y_train)
best_model = search.best_estimator_
```

---

## Betting Strategy Improvements

### 7. Smarter Bet Selection

#### A. Edge-Based Filtering
```python
def filter_value_bets(predictions, min_edge=0.05):
    """Only bet when model has significant edge."""
    predictions['edge'] = abs(predictions['model_prob'] - 0.5)

    # Only bet when model disagrees with 50/50 by 5%+
    value_bets = predictions[predictions['edge'] >= min_edge]

    return value_bets
```

#### B. Avoid Problem Areas
```python
def apply_filters(predictions):
    """Filter out low-performing segments."""
    filtered = predictions.copy()

    # Remove small spreads (3-6 points)
    filtered = filtered[~filtered['spread_line'].between(3, 6)]

    # Remove miscalibrated high confidence bets
    # (until calibration is fixed)
    filtered = filtered[filtered['spread_prob'].between(0.45, 0.65)]

    # Remove early season games (first 10 games per team)
    filtered = filtered[filtered['games_played'] >= 10]

    return filtered
```

#### C. Kelly Criterion Sizing
```python
def calculate_kelly_stake(model_prob, odds=-110, bankroll=100):
    """Optimal bet sizing using Kelly Criterion."""
    # Convert -110 to decimal odds
    decimal_odds = 1 + (100 / 110)

    # Kelly formula: f = (bp - q) / b
    # b = decimal odds - 1, p = win probability, q = 1 - p
    b = decimal_odds - 1
    p = model_prob
    q = 1 - p

    kelly_fraction = (b * p - q) / b

    # Use fractional Kelly (25% of full Kelly for safety)
    conservative_fraction = kelly_fraction * 0.25

    # Limit to max 5% of bankroll
    stake = min(conservative_fraction * bankroll, bankroll * 0.05)

    return max(stake, 0)  # Never negative
```

---

## Implementation Priority

### Phase 1: Quick Wins (Week 1)
1. ✅ **Add probability calibration** (CalibratedClassifierCV)
2. ✅ **Filter out 3-6 point spreads**
3. ✅ **Add edge-based filtering** (only bet edge > 5%)
4. ✅ **Use prior season data** for early season

**Expected Impact:** +8-10% ROI improvement

### Phase 2: Feature Engineering (Week 2-3)
5. ✅ Add clutch performance features
6. ✅ Add opponent-adjusted metrics
7. ✅ Add situational context features
8. ✅ Implement dynamic home court advantage

**Expected Impact:** +5-7% accuracy improvement

### Phase 3: Advanced Models (Week 4+)
9. ✅ Implement ensemble models (XGBoost + LightGBM + Neural Net)
10. ✅ Hyperparameter tuning with cross-validation
11. ✅ Separate models for home vs away predictions
12. ✅ Add betting market features (if data available)

**Expected Impact:** +3-5% accuracy improvement

---

## Measurement & Validation

### Track These Metrics
```python
def evaluate_improvements(predictions, actuals):
    """Comprehensive evaluation metrics."""
    from sklearn.metrics import log_loss, brier_score_loss
    from sklearn.calibration import calibration_curve

    metrics = {}

    # Accuracy
    metrics['accuracy'] = (predictions['pred'] == actuals).mean()

    # ROI
    metrics['roi'] = calculate_roi(predictions['pred'], actuals)

    # Calibration
    metrics['brier_score'] = brier_score_loss(actuals, predictions['prob'])
    metrics['log_loss'] = log_loss(actuals, predictions['prob'])

    # Calibration curve (visual)
    prob_true, prob_pred = calibration_curve(
        actuals,
        predictions['prob'],
        n_bins=10
    )

    # Sharpness (how often model is confident)
    metrics['avg_confidence'] = abs(predictions['prob'] - 0.5).mean()

    return metrics
```

### A/B Testing Framework
```python
def compare_models(baseline_model, improved_model, test_games):
    """Compare model performance side-by-side."""
    baseline_preds = baseline_model.predict(test_games)
    improved_preds = improved_model.predict(test_games)

    print(f"Baseline: {evaluate(baseline_preds)}")
    print(f"Improved: {evaluate(improved_preds)}")
    print(f"Lift: {calculate_lift(baseline_preds, improved_preds)}")
```

---

## Expected Overall Impact

If all Phase 1-2 improvements are implemented:

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| Accuracy | 54.5% | 62-65% | +7-10 pts |
| ROI | +4.1% | +12-15% | +8-11 pts |
| Calibration Error | High | Low | 50% reduction |
| Small Spread Acc | 42.2% | N/A | Filter out |
| High Conf Acc | 55.4% | 65%+ | +10 pts |

**Conservative Estimate:** +10-15% ROI improvement
**Optimistic Estimate:** +20-25% ROI improvement

---

## Next Steps

1. **Start with Phase 1** (calibration + filtering) - immediate wins
2. **Validate on holdout set** before deploying
3. **Implement one feature at a time** and measure impact
4. **Monitor calibration curves** weekly
5. **Keep detailed logs** of what works and what doesn't

Good luck! 🚀
