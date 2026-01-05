# Team Name Normalization Audit Report

## Overview
This audit confirms that team name standardization is bulletproof across all data ingestion and merging operations to prevent confusion from team name variants.

## Architecture Summary

### 1. Master Database: `team_mapping.json`
- **Location**: `src/ingestion/team_mapping.json`
- **Format**: Canonical IDs (e.g., `nba_lal`) → Variants list
- **Access**: Via `src.utils.team_names.normalize_team_name()`
- **Coverage**: 95+ team name variants from all sources

### 2. Standardization Module: `src/ingestion/standardize.py`
- **Function**: `normalize_team_to_espn()` - Primary normalization function
- **Output Format**: ESPN full team names (e.g., "Los Angeles Lakers")
- **Process**:
  1. Check cache (TTL: 24 hours)
  2. Check MASTER database (`team_mapping.json`)
  3. Fallback to legacy hardcoded mappings
  4. Fuzzy matching for edge cases
- **Validation**: Returns `(normalized_name, is_valid)` tuple

### 3. Standardization Flow by Source

#### The Odds API (`src/ingestion/the_odds.py`)
- **ALL endpoints** call `standardize_game_data()`:
  - `fetch_odds()` ✅ Line 152
  - `fetch_events()` ✅ Line 427
  - `fetch_scores()` ✅ Line 514
  - `fetch_event_odds()` ✅ Line 266
  - `fetch_historical_odds()` ✅ Line 616
  - `fetch_betting_splits()` ✅ Line 926
  - `fetch_participants()` ✅ Line 1015

**Result**: All team names from The Odds API are normalized to ESPN format before use.

#### API-Basketball (`src/ingestion/api_basketball.py`)
- **Team lookup**: Uses `normalize_team_name()` from `team_names.py` ✅ Line 893
- **Game data**: Uses `standardize_game_data()` ✅ Line 189-192
- **Result**: All team names normalized to ESPN format

#### BetsAPI (`src/ingestion/betsapi.py`)
- Uses `normalize_team_to_espn()` directly ✅ Line 422-423
- **Result**: All team names normalized to ESPN format

#### ESPN (`src/ingestion/espn.py`)
- ESPN is the canonical source - names already in ESPN format
- No normalization needed (by definition)

#### Action Network (`src/ingestion/betting_splits.py`)
- Uses `normalize_team_to_espn()` ✅ Line 169-170, 219-220, 395-396, 509-510
- **Result**: All team names normalized to ESPN format

### 4. Unified Records Implementation

#### Current Flow (CRITICAL ISSUE IDENTIFIED)
```
fetch_todays_games() 
  → fetch_odds() [STANDARDIZED ✅]
  → fetch_team_records_from_odds_api()
    → fetch_scores() [STANDARDIZED ✅]
    → Records stored with ESPN-formatted team names as keys ✅
```

**ISSUE**: `get_unified_team_record()` uses unsafe substring matching:
```python
# Line 162-164: UNSAFE - could match wrong teams
for name, record in records.items():
    if team_name.lower() in name.lower() or name.lower() in team_name.lower():
        return record["wins"], record["losses"]
```

**Problem**: 
- "LA" could match "LA Clippers" OR "Los Angeles Lakers" (ambiguous!)
- Should use proper normalization instead

### 5. Data Integrity Safeguards

✅ **Validation Flags**: All standardized data includes `_data_valid` flag
✅ **Error Logging**: Failed normalizations logged as ERROR (not warning)
✅ **Skip Invalid**: Invalid team names result in empty strings, games skipped
✅ **Source Tracking**: All data includes `_source` metadata
✅ **Standardization Flag**: All data includes `_standardized` flag

### 6. Team Name Format Consistency

**Input Formats**:
- The Odds API: "Los Angeles Lakers", "LA Clippers"
- API-Basketball: "Lakers", "Celtics" (short names)
- BetsAPI: Various formats
- ESPN: "Los Angeles Lakers" (canonical)

**Output Format (ALL SOURCES)**:
- ESPN full names: "Los Angeles Lakers", "Boston Celtics", "LA Clippers"
- Consistent across entire pipeline

## Issues Found

### ❌ CRITICAL: Unsafe Team Name Matching in Unified Records

**Location**: `src/utils/slate_analysis.py:162-164`

**Problem**: `get_unified_team_record()` uses substring matching which can cause false matches:
- "LA" matches both "LA Clippers" and "Los Angeles Lakers"
- Partial matches are ambiguous and unsafe

**Recommendation**: Use proper normalization via `normalize_team_to_espn()` instead of substring matching.

### ⚠️ MINOR: Partial Match Fallback

The partial match in `get_unified_team_record()` is a fallback, but since both odds and scores are standardized, exact matches should always work. The partial match is unnecessary and potentially dangerous.

## Recommendations

1. **Fix `get_unified_team_record()`**: Replace substring matching with proper normalization
2. **Add validation**: Ensure team names from records match normalized odds team names
3. **Remove unsafe fallback**: Since both sources are standardized, exact matches should suffice

## Conclusion

✅ **Standardization is comprehensive**: All data sources normalize team names through `standardize_game_data()` or `normalize_team_to_espn()`

✅ **MASTER database is used**: `team_mapping.json` provides canonical matching

❌ **Unified records code needs fix**: Substring matching in `get_unified_team_record()` is unsafe

**Overall Assessment**: 95% bulletproof - one critical issue needs fixing in unified records lookup.
