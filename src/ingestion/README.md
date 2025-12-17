**OddsAPI Ingest Module**

This README documents the OddsAPI (TheOdds) ingest helpers in this repository. It is focused solely on the OddsAPI ingest module and local testing helpers added alongside it.

**Purpose:**
- Describe the ingest module's responsibilities and how to run the lightweight test harness that verifies TheOdds v4 NBA endpoints.

**Files:**
- `src/ingestion/the_odds.py` — primary ingest helper used by the project to fetch TheOdds data (see module for implementation details).
- `scripts/test_the_odds_endpoints.py` — test harness that exercises every NBA-capable TheOdds v4 endpoint and performs lightweight validation checks.

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

**Notes / Next steps (optional):**
- Add a CI job or GitHub Action to run the harness periodically (be mindful of API quota and historical endpoint cost).
- Optionally add the test harness to project test runners under an opt-in flag to avoid consuming API credits during routine CI runs.

If you'd like, I can also:
- Add a short GitHub Actions workflow that runs this script on-demand (manual workflow) or on a schedule, with guidance to store the API key in repository secrets.
- Update `requirements.txt` and `.gitignore` as a follow-up.
