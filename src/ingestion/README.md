**NBA Data Ingestion Modules**

This directory contains modules for ingesting NBA data from various sources:
- The Odds API (betting odds)
- API-Basketball (game outcomes, statistics)
- ESPN (injuries, schedule)
- GitHub (open-source datasets)
- Betting splits (public betting percentages)

**Files:**
- `src/ingestion/the_odds.py` — primary ingest helper for The Odds API data
- `src/ingestion/api_basketball.py` — API-Basketball client for game data
- `src/ingestion/injuries.py` — ESPN injury report fetcher
- `src/ingestion/betting_splits.py` — public betting percentages
- `src/ingestion/github_data.py` — GitHub-hosted open-source data fetcher
- `src/ingestion/standardize.py` — team name standardization utilities
- `scripts/test_the_odds_endpoints.py` — test harness for The Odds API endpoints
- `scripts/fetch_github_data.py` — utility script for fetching GitHub-hosted data

**Environment / Secrets:**
- The test harness reads the API key from the environment variable `THE_ODDS_API_KEY`.
- It will also load any `.env*` files found in the repository root and the `scripts/` directory, but it will NOT override environment variables already set in your shell (so an explicit `$env:THE_ODDS_API_KEY` takes precedence).
- For local testing, you can use the included `.env.example` as a template; copy it to `.env` and fill in `THE_ODDS_API_KEY`.

**Security / Logging:**
- The test harness masks the `apiKey` query parameter in any printed URLs (it prints `[REDACTED]`) to avoid leaking secrets into logs.
- After running tests in a shared terminal/session, remove the environment variable or close the shell if you set the key there.

**Historical endpoints & cost:**
- Historical endpoints (snapshots and historical event odds) typically require a paid plan and consume more API credits. The harness will report `403 Forbidden` when access is not enabled for the API key.

**How to run (PowerShell):**
```
$env:THE_ODDS_API_KEY = '<your_api_key>'
python .\scripts\test_the_odds_endpoints.py
Remove-Item Env:\THE_ODDS_API_KEY
```

**What the harness does:**
- Calls: `/v4/sports`, `/v4/sports/basketball_nba/events`, `/v4/sports/basketball_nba/odds`, `/v4/sports/basketball_nba/events/{eventId}/odds`, `/v4/sports/basketball_nba/events/{eventId}/markets`, `/v4/sports/basketball_nba/scores`, `/v4/sports/basketball_nba/participants`, `/v4/historical/...` endpoints and `/v4/sports/upcoming/odds`.
- Performs presence/shape checks and prints a concise summary with HTTP status codes.

**GitHub Data Fetcher:**

The `github_data.py` module provides utilities for fetching open-source NBA datasets from GitHub repositories (e.g., FiveThirtyEight ELO data).

**Usage:**
```python
from src.ingestion.github_data import fetch_fivethirtyeight_elo

# Fetch FiveThirtyEight ELO historical data
df = await fetch_fivethirtyeight_elo("elo_historical")
```

**Command-line usage:**
```powershell
# Fetch FiveThirtyEight ELO data
python scripts/fetch_github_data.py --source fivethirtyeight --dataset elo_historical

# List all available sources
python scripts/fetch_github_data.py --list-sources
```

**Notes / Next steps (optional):**
- Add a CI job or GitHub Action to run the harness periodically (be mindful of API quota and historical endpoint cost).
- Optionally add the test harness to project test runners under an opt-in flag to avoid consuming API credits during routine CI runs.

If you'd like, I can also:
- Add a short GitHub Actions workflow that runs this script on-demand (manual workflow) or on a schedule, with guidance to store the API key in repository secrets.
- Update `requirements.txt` and `.gitignore` as a follow-up.
