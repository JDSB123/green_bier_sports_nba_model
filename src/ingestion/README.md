**NBA Data Ingestion Modules**

This directory contains modules for ingesting NBA data from various sources:
- The Odds API (betting odds)
- API-Basketball (game outcomes, statistics)
- ESPN (injuries, schedule)
- Betting splits (public betting percentages)

**Files:**
- `src/ingestion/the_odds.py` — primary ingest helper for The Odds API data
- `src/ingestion/api_basketball.py` — API-Basketball client for game data
- `src/ingestion/espn.py` — ESPN schedule and standings fetcher
- `src/ingestion/injuries.py` — Injury report fetcher
- `src/ingestion/betting_splits.py` — public betting percentages
- `src/ingestion/standardize.py` — team name standardization utilities
- `scripts/data_unified_fetch_the_odds.py` — fetch current odds
- `scripts/data_unified_fetch_api_basketball.py` — fetch game data

**Environment / Secrets:**
- API keys are read from environment variables: `THE_ODDS_API_KEY`, `API_BASKETBALL_KEY`
- For local testing, copy `.env.example` to `.env` and fill in the keys
- Production uses Azure Key Vault

**Security / Logging:**
- API keys are masked in logs to avoid leaking secrets
- After running in a shared terminal, remove environment variables or close the shell

**Historical endpoints & cost:**
- Historical endpoints require a paid plan and consume more API credits
- Legacy historical data scripts have been archived to `scripts/_archive/backtest/`
