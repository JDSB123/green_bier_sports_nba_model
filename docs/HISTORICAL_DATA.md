# Historical Odds Data System

This document describes the system for ingesting, storing, and exporting historical NBA odds data from The Odds API.

## Overview

The historical data system is designed to:
- Ingest NBA betting odds data from past seasons (2023-2024 and beyond)
- Store data in a **separate location** from live prediction data
- Maintain model integrity by isolating historical data
- Support backtesting and historical analysis

## Storage Structure

Historical data is stored separately from live data in `data/historical/`:

```
data/historical/
├── the_odds/                    # Raw JSON from The Odds API
│   ├── events/                  # Historical events by date
│   │   ├── 2023-2024/
│   │   │   ├── events_2023-10-24.json
│   │   │   ├── events_2023-10-25.json
│   │   │   └── ...
│   │   └── 2024-2025/
│   │       └── ...
│   ├── odds/                    # Historical odds snapshots
│   │   ├── 2023-2024/
│   │   │   ├── odds_2023-10-24_featured.json
│   │   │   ├── odds_2023-10-24_periods.json
│   │   │   └── ...
│   │   └── 2024-2025/
│   │       └── ...
│   ├── player_props/            # Player props (if enabled)
│   │   └── 2023-2024/
│   │       └── props_2023-10-24.json
│   └── metadata/                # Ingestion tracking
│       ├── progress_2023-2024.json
│       └── progress_2024-2025.json
└── exports/                     # Normalized exports
    ├── 2023-2024_events.csv
    ├── 2023-2024_events.parquet
    ├── 2023-2024_odds_featured.csv
    ├── 2023-2024_odds_featured.parquet
    ├── 2023-2024_odds_periods.csv
    └── manifest.json
```

## Supported Markets

### Featured Markets (Always Available)
- `h2h` - Head-to-head / Moneyline
- `spreads` - Point spread
- `totals` - Over/Under

### Game Period Markets
- First Half: `h2h_h1`, `spreads_h1`, `totals_h1`
- Second Half: `h2h_h2`, `spreads_h2`, `totals_h2`
- Quarters: `h2h_q1-q4`, `spreads_q1-q4`, `totals_q1-q4`
- 3-Way: `h2h_3_way`, `h2h_3_way_h1`, `h2h_3_way_q1`, etc.

### Alternate Markets
- `alternate_spreads`, `alternate_totals`
- Quarter/half variants: `alternate_spreads_h1`, etc.

### Team Totals
- `team_totals`, `team_totals_h1`, etc.
- `alternate_team_totals`

### Player Props (NBA-specific)
- Points: `player_points`, `player_points_q1`, `player_points_alternate`
- Rebounds: `player_rebounds`, `player_rebounds_q1`, `player_rebounds_alternate`
- Assists: `player_assists`, `player_assists_q1`, `player_assists_alternate`
- Combined: `player_points_rebounds_assists`, `player_points_rebounds`, etc.
- Special: `player_first_basket`, `player_double_double`, `player_triple_double`

## Usage

### 1. Ingesting Historical Data

```bash
# Full 2023-2024 season (featured + period markets)
python scripts/ingest_historical_period_odds.py --season 2023-2024

# Both seasons
python scripts/ingest_historical_period_odds.py --season 2023-2024
python scripts/ingest_historical_period_odds.py --season 2024-2025

# Specific date range
python scripts/ingest_historical_period_odds.py --start-date 2024-01-01 --end-date 2024-01-31

# Include all markets (more API credits)
python scripts/ingest_historical_period_odds.py --season 2023-2024 --markets all

# Include player props (expensive)
python scripts/ingest_historical_period_odds.py --season 2023-2024 --include-props

# Dry run to estimate costs
python scripts/ingest_historical_period_odds.py --season 2023-2024 --dry-run

# Resume interrupted ingestion
python scripts/ingest_historical_period_odds.py --season 2023-2024 --resume

# Check current status
python scripts/ingest_historical_period_odds.py --season 2023-2024 --show-summary
```

### 2. Exporting Data

```bash
# Export all ingested data to CSV and Parquet
python scripts/export_historical_odds.py

# Export specific season
python scripts/export_historical_odds.py --season 2023-2024

# Export CSV only
python scripts/export_historical_odds.py --format csv

# Include all bookmakers (not just primary US books)
python scripts/export_historical_odds.py --include-all-bookmakers

# Show what data is available
python scripts/export_historical_odds.py --show-summary
```

## API Costs

The Odds API uses a credit-based system for historical data:

| Endpoint | Cost |
|----------|------|
| Historical Events | 1 credit per request |
| Historical Odds | 10 credits per region per market |
| Historical Event Odds | 10 credits per region per market |

### Cost Estimation Example

For a full NBA season (~180 game days):
- Events: 180 credits
- Featured markets (3): 180 × 10 × 3 = 5,400 credits
- Period markets (20+): 180 × 10 × 20 = 36,000 credits

**Recommended approach:**
1. Start with `featured` markets only (~5,580 credits)
2. Add `periods` if needed for first-half/quarter analysis
3. Only add player props if specifically required

### Dry Run

Always use `--dry-run` first to estimate costs:

```bash
python scripts/ingest_historical_period_odds.py --season 2023-2024 --dry-run
```

## Data Isolation (Model Integrity)

Historical data is intentionally stored separately from live prediction data:

| Data Type | Location | Purpose |
|-----------|----------|---------|
| Live Odds | `data/raw/the_odds/` | Current predictions |
| Live Processed | `data/processed/` | Model input |
| Historical Raw | `data/historical/the_odds/` | Backtesting |
| Historical Exports | `data/historical/exports/` | Analysis |

**Key Points:**
- Historical data never flows into the live prediction pipeline
- Different directory structures prevent accidental mixing
- Model training should explicitly choose data source

## Progress Tracking

The system tracks ingestion progress in `metadata/progress_{season}.json`:

```json
{
  "season": "2023-2024",
  "last_date_processed": "2024-01-15",
  "dates_completed": ["2023-10-24", "2023-10-25", ...],
  "dates_failed": [],
  "total_events_fetched": 1234,
  "total_api_calls": 567,
  "estimated_credits_used": 5678,
  "started_at": "2024-01-01T12:00:00Z",
  "last_updated_at": "2024-01-15T18:30:00Z"
}
```

Use `--resume` to continue from the last checkpoint if interrupted.

## Exported Data Schema

### Events Export

| Column | Type | Description |
|--------|------|-------------|
| event_id | string | Unique event identifier |
| sport_key | string | Always "basketball_nba" |
| commence_time | datetime | Game start time (UTC) |
| home_team | string | Home team name |
| away_team | string | Away team name |
| completed | boolean | Whether game is finished |
| game_date | date | Game date |

### Odds Export

| Column | Type | Description |
|--------|------|-------------|
| snapshot_timestamp | datetime | When odds were captured |
| event_id | string | Event identifier |
| home_team | string | Home team |
| away_team | string | Away team |
| commence_time | datetime | Game start time |
| bookmaker_key | string | Bookmaker identifier |
| market_key | string | Market type (h2h, spreads, etc.) |
| outcome_name | string | Outcome (team name, Over/Under) |
| outcome_price | int | American odds |
| outcome_point | float | Line/spread (if applicable) |
| last_update | datetime | When bookmaker updated |
| game_date | date | Game date |

### Player Props Export

| Column | Type | Description |
|--------|------|-------------|
| event_id | string | Event identifier |
| player_name | string | Player name |
| market_key | string | Prop type (player_points, etc.) |
| outcome_name | string | Over/Under |
| outcome_price | int | American odds |
| outcome_point | float | Line (points, rebounds, etc.) |
| bookmaker_key | string | Bookmaker |
| game_date | date | Game date |

## Best Practices

### 1. Start Small
```bash
# Test with a small date range first
python scripts/ingest_historical_period_odds.py --start-date 2024-01-01 --end-date 2024-01-07 --dry-run
```

### 2. Use Resume
If ingestion is interrupted, always use `--resume`:
```bash
python scripts/ingest_historical_period_odds.py --season 2023-2024 --resume
```

### 3. Rate Limiting
The default rate limit is 1 request/second. Adjust if needed:
```bash
python scripts/ingest_historical_period_odds.py --season 2023-2024 --rate-limit 2.0
```

### 4. Export After Ingestion
Always export after ingestion for easier analysis:
```bash
python scripts/export_historical_odds.py --season 2023-2024
```

### 5. Verify Exports
Check the manifest to confirm exports:
```bash
cat data/historical/exports/manifest.json
```

## Troubleshooting

### "THE_ODDS_API_KEY not set"
Ensure your API key is configured:
```bash
# Option 1: Environment variable
export THE_ODDS_API_KEY=your_key_here

# Option 2: Local secrets file
echo "your_key_here" > secrets/THE_ODDS_API_KEY
```

### "403 Forbidden"
Historical endpoints require a paid plan. Check your Odds API subscription.

### "No events found for date"
- Verify the date had NBA games scheduled
- Check if you're querying outside the NBA season
- Some dates (All-Star break, off days) have no games

### Rate Limit Exceeded
Increase the rate limit interval:
```bash
python scripts/ingest_historical_period_odds.py --season 2023-2024 --rate-limit 2.0
```

### Incomplete Data
Use `--show-summary` to check progress:
```bash
python scripts/ingest_historical_period_odds.py --season 2023-2024 --show-summary
```

## Integration with Model

Historical data can be used for backtesting without affecting the live model:

```python
# In backtest scripts
import pandas as pd

# Load historical odds
odds_df = pd.read_parquet("data/historical/exports/2023-2024_odds_featured.parquet")

# Load historical events
events_df = pd.read_parquet("data/historical/exports/2023-2024_events.parquet")

# Your backtesting logic here
# ...
```

**Important:** Keep historical data separate from `data/processed/` which feeds the live model.
