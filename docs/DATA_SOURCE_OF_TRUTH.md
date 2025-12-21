# Data Source of Truth - NBA v6.0

**Last Updated:** 2025-12-20
**Status:** ✅ Production Ready - No Mock Data

---

## Overview

This document defines the **single source of truth** for all data ingestion in the NBA prediction system. All data flows through standardized modules that **NEVER use mock/fake data in production**.

---

## Core Principles

### 1. No Mock Data Policy
- ✅ **Production code NEVER uses mock data**
- ✅ If a data source fails, return empty data, not mock data
- ✅ Empty data is acceptable - predictions will proceed without optional features
- ❌ Mock data corrupts predictions and misleads users

### 2. Single Source of Truth Functions
Each data type has ONE primary aggregation function that should be called:

| Data Type | Single Source Function | Location |
|-----------|----------------------|----------|
| **Injuries** | `fetch_all_injuries()` | `src/ingestion/injuries.py` |
| **Betting Splits** | `fetch_public_betting_splits(source="auto")` | `src/ingestion/betting_splits.py` |
| **Game Odds** | `the_odds.fetch_odds()` | `src/ingestion/the_odds.py` |
| **Game Outcomes** | `APIBasketballClient.ingest_essential()` | `src/ingestion/api_basketball.py` |

### 3. Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   DATA SOURCES (APIs)                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ The Odds API │  │API-Basketball│  │ ESPN (Free)  │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                  │                  │              │
└─────────┼──────────────────┼──────────────────┼──────────────┘
          │                  │                  │
          └──────────────────┼──────────────────┘
                             │
                  ┌──────────▼──────────┐
                  │ SINGLE SOURCE       │
                  │ FUNCTIONS           │
                  │                     │
                  │ • fetch_all_injuries│
                  │ • fetch_odds()      │
                  │ • fetch_splits()    │
                  │ • ingest_games()    │
                  └──────────┬──────────┘
                             │
                  ┌──────────▼──────────┐
                  │ DATA PROCESSING     │
                  │                     │
                  │ • Standardization   │
                  │ • Validation        │
                  │ • Feature Building  │
                  └──────────┬──────────┘
                             │
                  ┌──────────▼──────────┐
                  │ STORAGE             │
                  │                     │
                  │ • CSV files         │
                  │ • Database          │
                  │ • Models            │
                  └─────────────────────┘
```

---

## Injury Data - Single Source of Truth

### Primary Function: `fetch_all_injuries()`

**Location:** `src/ingestion/injuries.py`

**What It Does:**
- Aggregates injuries from multiple sources (ESPN + API-Basketball)
- Merges duplicate injuries (same player)
- Returns standardized `InjuryReport` objects
- **NEVER returns mock data** - returns empty list if all sources fail

**Sources (in order):**
1. **ESPN** (free, always tried)
2. **API-Basketball** (if API key configured)

**Usage:**
```python
from src.ingestion.injuries import fetch_all_injuries

# This is the ONLY way to get injury data
injuries = await fetch_all_injuries()

if not injuries:
    # This is OK - predictions can proceed without injury data
    logger.warning("No injury data available")
```

**❌ DO NOT:**
- Call `fetch_injuries_espn()` directly
- Call `fetch_injuries_api_basketball()` directly
- Use mock data as fallback

**✅ DO:**
- Always use `fetch_all_injuries()` in production code
- Handle empty results gracefully
- Log when no injury data is available

---

## Betting Splits - Single Source of Truth

### Primary Function: `fetch_public_betting_splits(source="auto")`

**Location:** `src/ingestion/betting_splits.py`

**What It Does:**
- Tries multiple sources in order (Action Network → The Odds → SBRO → Covers)
- Returns empty dict if all sources fail
- **NEVER uses mock data** unless explicitly requested with `source="mock"`

**Usage:**
```python
from src.ingestion.betting_splits import fetch_public_betting_splits

# Production usage - auto tries all sources
splits = await fetch_public_betting_splits(games, source="auto")

if not splits:
    # This is OK - predictions can proceed without betting splits
    logger.warning("No betting splits available")
```

**❌ DO NOT:**
- Use `source="mock"` in production
- Call individual source functions directly

**✅ DO:**
- Use `source="auto"` for production
- Handle empty results gracefully

---

## Game Odds - Single Source of Truth

### Primary Function: `the_odds.fetch_odds()`

**Location:** `src/ingestion/the_odds.py`

**What It Does:**
- Fetches live odds from The Odds API
- Standardizes team names to ESPN format
- Filters invalid games (returns empty string for invalid teams)

**Usage:**
```python
from src.ingestion import the_odds

odds_data = await the_odds.fetch_odds()
# Already filtered - only valid games returned
```

---

## Game Outcomes - Single Source of Truth

### Primary Function: `APIBasketballClient.ingest_essential()`

**Location:** `src/ingestion/api_basketball.py`

**What It Does:**
- Fetches game outcomes, scores, and statistics
- Standardizes team names to ESPN format
- Returns standardized game data

**Usage:**
```python
from src.ingestion.api_basketball import APIBasketballClient

client = APIBasketballClient()
games = await client.ingest_essential()
```

---

## Containerization & Services

All services in `docker-compose.yml` use the same single source functions:

### Prediction Service
- Uses `fetch_all_injuries()` for injury data
- Uses `fetch_public_betting_splits(source="auto")` for splits
- Uses `the_odds.fetch_odds()` for odds

### Data Ingestion Services
- All services use the standardized ingestion modules
- No mock data fallbacks in containerized services

---

## Validation & Testing

### How to Verify No Mock Data

1. **Check for mock fallbacks:**
   ```bash
   grep -r "mock.*production\|fake.*production\|mock.*fallback" src/
   ```

2. **Check injury data source:**
   ```python
   # Should only call fetch_all_injuries()
   grep -r "fetch_injuries_espn\|fetch_injuries_api_basketball" scripts/
   ```

3. **Check betting splits source:**
   ```python
   # Should use source="auto" or real sources
   grep -r "source.*=.*mock" scripts/
   ```

### Production Readiness Checklist

- [x] ✅ No mock data in `fetch_all_injuries()`
- [x] ✅ No mock data in `fetch_public_betting_splits()` when `source="auto"`
- [x] ✅ All services use single source functions
- [x] ✅ Empty data handled gracefully
- [x] ✅ Proper logging for data source failures

---

## Error Handling

### When Data Sources Fail

**Correct Approach:**
```python
# Return empty data, log warning, continue
injuries = await fetch_all_injuries()
if not injuries:
    logger.warning("No injury data available - proceeding without injury features")
    # Continue prediction without injury data
```

**Incorrect Approach:**
```python
# DO NOT DO THIS
if not injuries:
    injuries = get_mock_injuries()  # ❌ NEVER USE MOCK DATA
```

---

## Migration Guide

If you find code using mock data or calling individual source functions directly:

1. **Replace individual calls with single source function:**
   ```python
   # OLD (don't do this)
   espn_injuries = await fetch_injuries_espn()
   api_injuries = await fetch_injuries_api_basketball()
   injuries = espn_injuries + api_injuries
   
   # NEW (do this)
   injuries = await fetch_all_injuries()
   ```

2. **Remove mock fallbacks:**
   ```python
   # OLD (don't do this)
   if not injuries:
       injuries = mock_injuries
   
   # NEW (do this)
   if not injuries:
       logger.warning("No injury data available")
       # Continue without injuries
   ```

---

## Summary

✅ **Single Source Functions:**
- Injuries: `fetch_all_injuries()`
- Betting Splits: `fetch_public_betting_splits(source="auto")`
- Odds: `the_odds.fetch_odds()`
- Games: `APIBasketballClient.ingest_essential()`

✅ **No Mock Data Policy:**
- Empty data is acceptable
- Mock data corrupts predictions
- Always use real sources or handle empty gracefully

✅ **Containerization:**
- All services use standardized ingestion modules
- No mock data in production containers

---

**Questions or Issues?**
- Check individual module docstrings
- Review `docs/DATA_INGESTION_METHODOLOGY.md`
- Check logs for data source failures
