# Data Guide

**Last Updated:** 2026-01-23
**Status:** Consolidated from DATA_SINGLE_SOURCE_OF_TRUTH, DATA_SOURCE_OF_TRUTH, DATA_INVENTORY, DATA_INGESTION_METHODOLOGY

---

## Canonical Training Data

```
data/processed/training_data.csv
```

**This is THE ONLY training data file.**

| Attribute | Value |
|-----------|-------|
| Rows | ~4,000 games |
| Columns | 327+ |
| Date Range | 2023-01-01 to present |
| Injury Data | 100% coverage |
| FG Labels | 100% coverage |
| 1H Labels | 81%+ overall (99%+ since 2023-05-01) |

---

## Data Sources

| Data Type | Primary Source | Purpose |
|-----------|----------------|---------|
| **Betting Odds** | The Odds API | Spreads, totals, moneylines |
| **Game Outcomes** | API-Basketball | Scores, Q1-Q4 breakdowns |
| **Team Statistics** | API-Basketball | PPG, PAPG, standings |
| **Injuries** | ESPN + API-Basketball | Injury impact features |
| **Betting Splits** | Action Network / The Odds API | Public betting % |
| **Elo Ratings** | FiveThirtyEight (historical) | Team strength |

---

## Single Source Functions

Each data type has ONE primary aggregation function:

| Data Type | Function | Location |
|-----------|----------|----------|
| **Injuries** | `fetch_all_injuries()` | `src/ingestion/injuries.py` |
| **Betting Splits** | `fetch_public_betting_splits()` | `src/ingestion/betting_splits.py` |
| **Game Odds** | `fetch_odds()` | `src/ingestion/the_odds.py` |
| **Game Outcomes** | `APIBasketballClient.ingest_essential()` | `src/ingestion/api_basketball.py` |

---

## No Mock Data Policy

- ✅ **Production code NEVER uses mock data**
- ✅ If a data source fails, return empty data
- ✅ Empty data is acceptable - predictions proceed without optional features
- ❌ Mock data corrupts predictions

---

## Team Name Standardization

All team names are normalized to a **canonical format** (historically called "ESPN format"):

```python
from src.ingestion.standardize import normalize_team_name

# Normalizes various formats to canonical:
# "LA Lakers" → "Los Angeles Lakers"
# "LAL" → "Los Angeles Lakers"
```

The canonical module is `src/ingestion/standardize.py`. Other modules delegate to it.

---

## Azure Blob Storage

Training data mirrors to Azure for versioning and backup:

**Storage Account:** `nbagbsvstrg`
**Container:** `nbahistoricaldata`

```
training_data/
├── vYYYY.MM.DD/           # Versioned snapshots
│   ├── training_data.csv
│   └── manifest.json
└── latest/                # Points to validated version
    ├── training_data.csv
    └── manifest.json
```

### Upload/Download

```bash
# Upload validated data
python scripts/upload_training_data_to_azure.py --force --version v2026.01.XX

# Download canonical data
python scripts/download_training_data_from_azure.py --version latest --verify

# List versions
python scripts/download_training_data_from_azure.py --list
```

---

## Quality Gates

Before upload to Azure, data must pass:

- ✅ Minimum 3,000 games
- ✅ Minimum 50 features/columns
- ✅ 80%+ injury coverage
- ✅ 90%+ odds coverage
- ✅ No nulls in critical columns
- ✅ Score ranges validated (50-200)
- ✅ SHA256 checksum in manifest

---

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      DATA SOURCES (APIs)                        │
├─────────────────────────────────────────────────────────────────┤
│  The Odds API │ API-Basketball │ ESPN │ Action Network         │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                  ┌─────────▼─────────┐
                  │ STANDARDIZATION   │
                  │ (normalize names) │
                  └─────────┬─────────┘
                            │
                  ┌─────────▼─────────┐
                  │ RAW STORAGE       │
                  │ (data/raw/)       │
                  └─────────┬─────────┘
                            │
                  ┌─────────▼─────────┐
                  │ TRAINING DATA     │
                  │ (data/processed/) │
                  └─────────┬─────────┘
                            │
                  ┌─────────▼─────────┐
                  │ AZURE BLOB        │
                  │ (versioned)       │
                  └───────────────────┘
```

---

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `data_unified_build_training_complete.py` | Build training data from raw |
| `data_unified_validate_training.py` | Validate training data |
| `data_unified_ingest_all.py` | Full ingestion pipeline |
| `upload_training_data_to_azure.py` | Upload to Azure |
| `download_training_data_from_azure.py` | Download from Azure |

---

## Backtest Policy

Backtests must use the audited canonical dataset:

```
data/processed/training_data.csv
```

Rules:
- Do NOT rebuild or merge raw data for backtests
- If file missing, download from Azure and verify checksum
- Backtest phase is consume-only
