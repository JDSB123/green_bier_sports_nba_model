# Rationale Generation Requirements

## Overview
This document specifies the requirements for generating betting pick rationales in the NBA v4.0 model. All rationales must follow these guidelines to ensure consistency, transparency, and actionable insights.

## Format Requirements

### Output Format
- **Length**: 1-3 sentences (bullet points)
- **Format**: Newline-separated bullet points (for Excel cell formatting)
- **Tone**: Analytical, neutral, data-driven
- **Content**: Pre-game data only (no post-game data, no subjective language)

### Category Requirements
Each rationale **must include at least 3 of the following 6 categories**, with priority given to timestamped market movement and sharp action indicators.

---

## Category 1: 📈 Market Context (High Priority)

### Required Elements:
- **Market Type**: e.g., 1H Spread, Full Game Total, Moneyline
- **Opening Line vs. Current Line**: Include both values
- **Line Movement Direction & Magnitude**: e.g., "moved from -2.5 to -4.0"
- **Timestamp of Movement**: Highlight if movement occurred within 24 hours of tipoff
- **Steam Movement**: Sudden, sharp line changes with low public betting volume
- **Reverse Line Movement**: Line moves opposite to public betting percentage

### Example:
```
📈 Line moved from -2.5 to -4.0 in the 12 hours before tipoff, suggesting sharp action on the favorite.
```

### Implementation Notes:
- Extract opening line from `betting_splits.spread_open` or `betting_splits.total_open`
- Calculate movement: `current_line - opening_line`
- Check if movement occurred within 24 hours: compare `game_time` to current time
- Movement magnitude threshold: ≥1.0 points for spreads, ≥0.5 for totals

---

## Category 2: 🧮 Team Fundamentals (High Priority)

### NBA-Specific Metrics:
- **Points Per Game (PPG)**: Offensive output
- **Points Allowed Per Game (PAPG)**: Defensive efficiency
- **ELO Rating**: Team strength metric
- **Offensive/Defensive Rating**: Advanced efficiency metrics
- **Pace Factor**: Expected game tempo

### Example:
```
🧮 Team A averages 115.2 PPG and faces a defense allowing 108.5 PPG.
🧮 Team A holds significant ELO advantage (75 points).
```

### Implementation Notes:
- Use `features.home_ppg`, `features.away_ppg`, `features.home_papg`, `features.away_papg`
- Use `features.home_elo`, `features.away_elo` for ELO comparisons
- For spreads: Compare pick team's PPG vs opponent's PAPG
- For totals: Show combined offensive output

---

## Category 3: 🌍 Situational Factors (Medium Priority)

### Required Elements:
- **Home/Away Status**: Venue advantage
- **Rest Days**: e.g., "Team B is on short rest (5 days)"
- **Travel Distance**: Especially for cross-country or back-to-back road games
- **Back-to-Back Games**: Fatigue factor
- **Weather Forecast**: (if applicable for outdoor venues - not typically for NBA)

### Example:
```
🌍 Team A has rest advantage: 2 days rest vs Team B's 1 day (1.5 pt edge).
🌍 High pace expected (factor: 1.05), favoring higher scoring.
```

### Implementation Notes:
- Use `features.rest_margin_adj` for rest advantage calculations
- Use `features.home_days_rest`, `features.away_days_rest` for specific rest days
- Use `features.expected_pace_factor` for pace analysis (totals only)
- Rest advantage threshold: ≥1.5 points

---

## Category 4: 💸 Market Sentiment & Sharp Action (Very High Priority)

### Required Elements:
- **Betting Splits**:
  - % of bets vs. % of money
  - Public vs. sharp money
- **Steam Indicator**: Detected via sudden line movement without corresponding public volume
- **Reverse Line Movement (RLM)**: Line moves against public betting direction
- **Line Stability Score**: Volatility in odds over time

### Example:
```
💸 Reverse line movement detected: despite 70% of bets on the underdog, the line moved in favor of the favorite — a classic sharp money signal.
💸 Sharp money indicator: only 35% tickets but 58% money on Team A, suggesting professional action.
```

### Implementation Notes:
- Use `betting_splits.spread_home_ticket_pct`, `betting_splits.spread_home_money_pct`
- Use `betting_splits.spread_rlm`, `betting_splits.sharp_spread_side` for RLM detection
- Ticket vs Money divergence threshold: ≥10 percentage points
- Sharp signal: <45% tickets but >55% money
- Contrarian: <35% tickets (fading public)

---

## Category 5: 📊 Model Confidence (High Priority)

### Required Elements:
- **Model Probability**: e.g., "Model assigns 63% probability to cover"
- **Expected Value (EV)**: e.g., "+14% EV based on no-vig odds"
- **Confidence Score**: Normalized score from 0 to 1
- **Recommendation Flag**: Only include rationale if `is_recommended = True`

### Example:
```
📊 Model assigns 63% probability to cover with a +2.5 pt edge, exceeding the confidence threshold.
📊 Expected value: +14.2% based on current odds (-110).
📊 High-confidence play: probability exceeds 65% threshold.
```

### Implementation Notes:
- Always include model probability (required)
- Calculate EV: `(model_prob * profit) - (1 - model_prob)`
- Include EV if `abs(ev_pct) >= 5%`
- High-confidence threshold: `model_prob >= 0.65` or `model_prob <= 0.35`

---

## Category 6: 🕰️ Historical Context (Low Priority)

### Required Elements:
- **ATS Trends**: Last 3 years (if available)
- **Performance in Similar Situations**: e.g., "Team A is 5–1 ATS as a road underdog"
- **Head-to-Head Matchup History**: Recent meetings

### Example:
```
🕰️ Team A has won 75% of recent head-to-head meetings.
🕰️ Team A is 4-1 ATS in last 5 games as road underdog.
```

### Implementation Notes:
- Use `features.h2h_win_rate` for head-to-head data
- H2H threshold: `h2h_win_rate > 0.6` or `h2h_win_rate < 0.4`
- Only include if significant deviation from 0.5

---

## Assembly Logic

### Priority Order:
1. **Model Confidence** (always included - required)
2. **Market Context** (high priority)
3. **Sharp Action** (very high priority)
4. **Team Fundamentals** (high priority)
5. **Situational Factors** (medium priority)
6. **Historical Context** (low priority)

### Minimum Requirements:
- Must include at least **3 categories**
- Must include **Model Confidence** (Category 5)
- Prioritize **Market Context** and **Sharp Action** when available
- Fill remaining slots with other categories in priority order

### Implementation:
```python
# Always include Model Confidence
rationale_bullets.extend(model_confidence[:1])

# High priority: Market Context and Sharp Action
if market_context:
    rationale_bullets.extend(market_context[:1])
if sharp_action:
    rationale_bullets.extend(sharp_action[:1])

# Fill to minimum 3 categories
# Add Team Fundamentals, Situational, Historical as needed
```

---

## Excel Formatting

### Cell Format:
- Each bullet point on its own line within the cell
- Use newline character (`\n`) to separate bullets
- Excel will automatically wrap and display as bullet points

### Example Excel Cell Content:
```
📊 Model assigns 63% probability to cover with a +2.5 pt edge.
📈 Line moved from -2.5 to -4.0 in the 12 hours before tipoff.
💸 Sharp money indicator: only 35% tickets but 58% money on Team A.
```

---

## Validation Checklist

Before generating a rationale, ensure:
- [ ] At least 3 categories are included
- [ ] Model Confidence is always included
- [ ] Market Context or Sharp Action is included when data is available
- [ ] All data is pre-game only (no post-game results)
- [ ] Tone is analytical and neutral
- [ ] Format is newline-separated bullet points
- [ ] Length is 1-3 sentences (3 bullet points max)

---

## Data Sources

### Required Data Objects:
- `features`: Dict containing team statistics and metrics
- `betting_splits`: GameSplits object with betting percentages and line movement
- `game`: Dict containing game metadata (commence_time, etc.)
- `odds`: Dict containing current market lines

### Key Fields:
- `betting_splits.spread_open`, `betting_splits.spread_current`
- `betting_splits.total_open`, `betting_splits.total_current`
- `betting_splits.spread_rlm`, `betting_splits.sharp_spread_side`
- `features.home_ppg`, `features.away_ppg`, `features.home_papg`, `features.away_papg`
- `features.rest_margin_adj`, `features.home_days_rest`, `features.away_days_rest`
- `features.h2h_win_rate`
- `game.commence_time`

---

## Version History

- **v1.0** (2025-12-13): Initial requirements specification
  - Defined 6 rationale categories
  - Established priority order and minimum requirements
  - Specified Excel formatting requirements

---

## References

- Implementation: `scripts/analyze_todays_slate.py::generate_rationale()`
- Data Structures: `src/ingestion/betting_splits.py::GameSplits`
- Feature Building: `scripts/build_rich_features.py::RichFeatureBuilder`
