# NBA v4.0 Model — Single Source of Truth

Canonical entrypoint: `scripts/nba_v4_0_model.py` (wraps `analyze_todays_slate.py` with rich features).

## What it does
- Fetches today’s (or specified) NBA slate from The Odds API.
- Builds rich features via `scripts/build_rich_features.py` (tempo-free efficiency, rest, form, H2H).
- Produces model predictions (margin/total/ELO) and compares to market odds for edges.
- Outputs: text report, JSON, and optional visualization PNG.

## Usage
```bash
pip install -r requirements.txt
python scripts/nba_v4_0_model.py --date 2025-12-07 --output weekly_lineup.png
```
Options:
- `--date`: `YYYY-MM-DD`, `today`, or `tomorrow`
- `--output`: path for visualization PNG
- `--no-api`: skip API-Basketball (falls back to odds-only features)
- Single-game: add `--home "Home Team"` and `--away "Away Team"` to filter to one matchup on the given date.

## Key dependencies
- API keys: `API_BASKETBALL_KEY`, The Odds API key (via settings).
- Requirements: see `requirements.txt` (includes pandas, numpy, sklearn, matplotlib, etc.).

## Data flow (high level)
1) Odds: `src/ingestion/the_odds.py` → consensus odds.  
2) Features: `scripts/build_rich_features.py` → tempo-free + rest/form/H2H.  
3) Analysis: `scripts/analyze_todays_slate.py` → aggregates odds + features, computes edges, renders report/PNG.

## Outputs
- Text: `data/processed/slate_analysis_<DATE>.txt`
- JSON: `data/processed/slate_analysis_<DATE>.json`
- PNG: `data/processed/weekly_lineup_<DATE>.png` (when matplotlib available)

## Notes
- Rest is computed relative to game date when provided; pace/form adjustments are pace-scaled.
- Only this doc + `scripts/nba_v4_0_model.py` should be treated as canonical; legacy/archived entrypoints have been removed.

