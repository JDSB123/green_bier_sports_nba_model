# NBA v5.0 Backtest Quick Start

## Overview

The backtest system is fully containerized and uses **only fresh data** from APIs.

**Key Principles:**
- ✅ **Single source of truth** - One container, one pipeline
- ✅ **No placeholder data** - All data comes from real APIs
- ✅ **No silent failures** - Errors are raised, not ignored
- ✅ **Strict validation** - Data integrity checked at every step

---

## Prerequisites

### 1. API Keys (Required)

You need two API keys:

| API | Purpose | Free Tier | Get Key |
|-----|---------|-----------|---------|
| The Odds API | Betting lines (spreads, totals) | 500 req/month | [the-odds-api.com](https://the-odds-api.com/) |
| API-Basketball | Game outcomes, Q1-Q4 scores | 100 req/day | [api-sports.io](https://api-sports.io/) |

### 2. Environment Setup

```bash
# Copy example env file
cp env.example .env

# Edit with your API keys
# THE_ODDS_API_KEY=your_key_here
# API_BASKETBALL_KEY=your_key_here
```

---

## Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# Full pipeline: fetch data → build training set → run backtest
docker-compose -f docker-compose.backtest.yml up backtest-full

# Just fetch and build training data (no backtest)
docker-compose -f docker-compose.backtest.yml up backtest-data

# Run backtest on existing data (no API calls)
docker-compose -f docker-compose.backtest.yml up backtest-only

# Validate existing training data
docker-compose -f docker-compose.backtest.yml up backtest-validate

# Interactive debugging shell
docker-compose -f docker-compose.backtest.yml run --rm backtest-shell
```

### Option 2: Single Docker Command

```bash
# Build the image
docker build -f Dockerfile.backtest -t nba-backtest .

# Run full pipeline
docker run --env-file .env -v $(pwd)/data:/app/data nba-backtest full

# Run specific command
docker run --env-file .env -v $(pwd)/data:/app/data nba-backtest data
docker run -v $(pwd)/data:/app/data nba-backtest backtest
docker run -v $(pwd)/data:/app/data nba-backtest validate
```

### Option 3: Local (No Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Build training data
python scripts/build_fresh_training_data.py --seasons 2024-2025,2025-2026

# Run backtest
python scripts/backtest.py --markets all --strict
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SEASONS` | `2024-2025,2025-2026` | Seasons to backtest |
| `MARKETS` | `all` | Markets to test (see below) |
| `MIN_TRAINING` | `80` | Min games before first prediction |

### Available Markets

| Market | Description | Requires |
|--------|-------------|----------|
| `fg_spread` | Full game spreads | `spread_line` column |
| `fg_total` | Full game totals | `total_line` column |
| `fg_moneyline` | Full game moneyline | Scores only |
| `1h_spread` | First half spreads | Q1-Q2 scores + lines |
| `1h_total` | First half totals | Q1-Q2 scores + lines |
| `1h_moneyline` | First half moneyline | Q1-Q2 scores |
| `q1_spread` | First quarter spreads | Q1 scores + lines |
| `q1_total` | First quarter totals | Q1 scores + lines |
| `q1_moneyline` | First quarter moneyline | Q1 scores |

---

## Output Files

After a successful run, results are saved to:

```
data/
├── processed/
│   ├── training_data.csv          # Built training data
│   └── all_markets_backtest_results.csv
├── results/
│   ├── backtest_results_YYYYMMDD_HHMMSS.csv
│   └── backtest_report_YYYYMMDD_HHMMSS.md
└── raw/
    ├── api_basketball/            # Raw API responses
    └── the_odds/
```

---

## Interpreting Results

### Production Readiness Status

| Status | Criteria | Recommendation |
|--------|----------|----------------|
| ✅ PRODUCTION READY | Accuracy ≥ 55%, ROI > 5% | Use for live betting |
| ⚠️ NEEDS MONITORING | Accuracy ≥ 52%, ROI > 0% | Use with caution |
| ❌ NOT RECOMMENDED | Below thresholds | Do not use |

### Example Output

```
📊 PRODUCTION READINESS SUMMARY:
----------------------------------------
  Full Game Spreads: 60.6% acc, +15.7% ROI (203 bets) - ✅ PRODUCTION READY
  Full Game Totals: 59.2% acc, +13.1% ROI (341 bets) - ✅ PRODUCTION READY
  Full Game Moneyline: 58.4% acc, +10.2% ROI (422 bets) - ✅ PRODUCTION READY
  First Half Spreads: 55.9% acc, +6.7% ROI (189 bets) - ⚠️  NEEDS MONITORING
```

---

## Troubleshooting

### "API_BASKETBALL_KEY is not set"

```bash
# Make sure .env exists and has your key
cat .env | grep API_BASKETBALL_KEY
```

### "Historical odds endpoint not available"

The Odds API historical endpoint requires a paid plan (Group 2+). Without it:
- Full game moneyline still works (no lines needed)
- Spread/total backtests require historical lines

Options:
1. Upgrade to The Odds API Group 2+
2. Collect odds going forward with regular data collection
3. Use alternative historical data source

### "Training data not found"

Run the data pipeline first:
```bash
docker-compose -f docker-compose.backtest.yml up backtest-data
```

### "Not enough data"

Backtest requires at least `MIN_TRAINING + 50` games. Either:
- Fetch more seasons: `SEASONS=2023-2024,2024-2025,2025-2026`
- Lower MIN_TRAINING: `MIN_TRAINING=50`

---

## Advanced Usage

### Custom Backtest Parameters

```bash
# Specific markets only
MARKETS=fg_spread,fg_total docker-compose -f docker-compose.backtest.yml up backtest-full

# More training data
MIN_TRAINING=150 docker-compose -f docker-compose.backtest.yml up backtest-full

# Historical seasons
SEASONS=2020-2021,2021-2022,2022-2023,2023-2024,2024-2025 \
  docker-compose -f docker-compose.backtest.yml up backtest-full
```

### Strict Mode (Fail on Any Error)

```bash
# In container
docker run --env-file .env -v $(pwd)/data:/app/data nba-backtest full

# Locally
python scripts/backtest.py --strict --markets all
```

### Debug Shell

```bash
docker-compose -f docker-compose.backtest.yml run --rm backtest-shell

# Inside container:
python scripts/build_fresh_training_data.py --validate-only
python scripts/backtest.py --markets fg_moneyline --min-training 50
```

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    BACKTEST CONTAINER                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. FETCH DATA                                              │
│     ┌──────────────┐      ┌──────────────┐                  │
│     │ API-Basketball│      │ The Odds API │                  │
│     │  (Outcomes)   │      │   (Lines)    │                  │
│     └──────┬───────┘      └──────┬───────┘                  │
│            │                      │                          │
│            ▼                      ▼                          │
│  2. BUILD TRAINING DATA                                     │
│     ┌─────────────────────────────────────────┐             │
│     │ Merge outcomes + lines → training_data.csv            │
│     │ • Standardize team names                 │             │
│     │ • Compute labels (spread_covered, etc.) │             │
│     │ • Validate data integrity               │             │
│     └─────────────────────────────────────────┘             │
│                        │                                     │
│                        ▼                                     │
│  3. RUN BACKTEST                                            │
│     ┌─────────────────────────────────────────┐             │
│     │ Walk-forward validation (no leakage)    │             │
│     │ • Train on past games                   │             │
│     │ • Predict next game                     │             │
│     │ • Calculate accuracy & ROI              │             │
│     └─────────────────────────────────────────┘             │
│                        │                                     │
│                        ▼                                     │
│  4. OUTPUT RESULTS                                          │
│     • backtest_results.csv                                  │
│     • backtest_report.md                                    │
│     • Production readiness summary                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```
