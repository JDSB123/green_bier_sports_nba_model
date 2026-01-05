# Historical NBA Odds Data

**⚠️ PROTECTED DATA - DO NOT DELETE ⚠️**

This directory contains historical NBA betting odds data from The Odds API.

## Contents

- **the_odds/**: Raw JSON data from The Odds API
  - `events/`: Historical game events by season
  - `odds/`: Historical full game odds (h2h, spreads, totals)
  - `period_odds/`: Historical period odds (1H, quarters)
  - `metadata/`: Ingestion progress tracking

- **exports/**: Normalized CSV/Parquet exports for analysis

## Data Coverage

- **2021-2022**: Events + Full game odds
- **2022-2023**: Events + Full game odds  
- **2023-2024**: Events + Full game odds + 1H odds
- **2024-2025**: Events + Full game odds + 1H odds

## Protection

This data is:
- ✅ Committed to git repository
- ✅ Tagged as `historical-data-v1`
- ✅ Protected from deletion
- ✅ Isolated from live model pipeline

**DO NOT DELETE OR MODIFY** - This is valuable historical data for backtesting and analysis.

## Usage

See `docs/HISTORICAL_DATA.md` for full documentation on using this data.
