# NBA v6.0 Model Review - Executive Summary

**Review Date:** December 22, 2025  
**Scope:** Complete end-to-end model audit for single source of truth violations  
**Status:** ✅ **AUDIT COMPLETE** - 3 Violations Identified & Documented

---

## Key Findings

Your NBA model has **3 critical violations** of the single source of truth principle you documented. These create data inconsistencies between training and production:

| # | Violation | Location | Severity | Fix Time |
|---|-----------|----------|----------|----------|
| 1 | Direct ESPN injury call bypasses aggregator | `src/ingestion/comprehensive.py:616` | 🔴 Critical | 5 min |
| 2 | Three duplicate team name normalization implementations | `src/modeling/` (3 files) | 🔴 Critical | 30 min |
| 3 | Dual odds paths (historical + current) | `scripts/build_fresh_training_data.py:247-355` | 🟡 High | 20 min |

---

## Detailed Report

**Full audit with code fixes:** 📄 [docs/SINGLE_SOURCE_OF_TRUTH_AUDIT.md](docs/SINGLE_SOURCE_OF_TRUTH_AUDIT.md)

This document includes:
- ✅ Executive summary of violations
- ✅ Detailed explanation of each violation
- ✅ Architectural impact analysis
- ✅ Before/after code fixes for each violation
- ✅ Validation test suite to prevent regressions
- ✅ Post-fix checklist

---

## Violation #1: Injury Data Bypass

### Problem
`comprehensive.py` calls `fetch_injuries_espn()` directly instead of using `fetch_all_injuries()` which aggregates ESPN + API-Basketball.

### Why It Matters
- If ESPN fails, no fallback to API-Basketball
- Inconsistent injury data compared to other modules
- Violates your documented single source principle

### Fix
Replace line 616 in `comprehensive.py`:
```python
# ❌ WRONG
from src.ingestion.injuries import fetch_injuries_espn
data = await api_cache.get_or_fetch(fetch_fn=fetch_injuries_espn, ...)

# ✅ RIGHT
from src.ingestion.injuries import fetch_all_injuries
data = await api_cache.get_or_fetch(fetch_fn=fetch_all_injuries, ...)
```

---

## Violation #2: Team Name Duplication

### Problem
Three separate `normalize_team_name()` implementations:
1. `src/utils/team_names.py` → returns `"nba_lal"`
2. `src/modeling/team_factors.py` → returns `"Los Angeles Lakers"`
3. `src/modeling/dataset.py` → has its own TEAM_NAME_MAP

### Why It Matters
- Same team may be represented 3 different ways in features
- Hard to maintain (changes needed in 3 places)
- Feature mismatch between training and prediction
- Causes cascading bugs when team lists are updated

### Fix
1. Remove `normalize_team_name` from `team_factors.py`
2. Remove `TEAM_NAME_MAP` from `dataset.py`
3. Import from `src/utils/team_names.py` everywhere
4. Update HCA logic to work with canonical IDs

---

## Violation #3: Odds Collection Paths

### Problem
`build_fresh_training_data.py` imports and calls BOTH:
- `fetch_historical_odds()` (lines 266-293)
- `fetch_odds()` (line 355)

### Why It Matters
- Training may use historical odds, prediction uses current odds
- Inconsistent odds standardization
- Two different failure modes
- Violates documented single source of truth

### Fix
Use only `fetch_odds()` which internally handles fallback logic:
```python
# ❌ WRONG
from the_odds import fetch_historical_odds, fetch_odds
lines_data = await fetch_historical_odds(...)  # Path A
lines_data = await fetch_odds(...)  # Path B

# ✅ RIGHT
from the_odds import fetch_odds
lines_data = await fetch_odds(markets="spreads,totals,h2h")
# fetch_odds internally tries historical then falls back to current
```

---

## Architectural Impact

```
With Violations:
─────────────────────────────────────────
Training Data Pipeline    Prediction Pipeline
└─ Injury Source A        └─ Injury Source A+B (different!)
└─ Team Name Format 1     └─ Team Name Format 2 (different!)
└─ Odds Path A            └─ Odds Path B (different!)
            ↓
     FEATURE MISMATCH
     Model overfits to training distribution
```

---

## What's Working Well ✅

1. **Betting Splits:** Correctly uses single source `fetch_public_betting_splits(source="auto")`
2. **Core Data Flow:** Production entry point (`docker-entrypoint-backtest.sh`) → `build_fresh_training_data.py` is clear
3. **Documentation:** Your DATA_SOURCE_OF_TRUTH.md is excellent and comprehensive
4. **API Abstraction:** Ingestion modules have clean interfaces
5. **No Mock Data:** Production code doesn't use mock data (good!)

---

## What Needs Fixing ❌

1. **comprehensive.py** - Uses `fetch_injuries_espn()` directly (should use aggregator)
2. **Team Names** - 3 duplicate implementations scattered across codebase
3. **Odds Pipeline** - Mixes historical and current odds paths
4. **Testing** - No validation that single source functions are used

---

## Next Steps

### 🔴 Immediate (This Session)
1. Review this audit report
2. Understand the 3 violations and their impact
3. Decide on fix strategy

### 🟡 Soon (Next 1-2 hours)
1. Apply fixes to `comprehensive.py` (5 min)
2. Consolidate team name implementations (30 min)
3. Fix odds pipeline in `build_fresh_training_data.py` (20 min)
4. Add validation tests (15 min)

### 🟢 Follow-up
1. Update documentation in `comprehensive.py` docstring
2. Consider making these checks part of CI/CD pipeline
3. Add enforcement rules to prevent future violations

---

## Files Affected

### Violations (Need Fixes)
- `src/ingestion/comprehensive.py` - Violation #1
- `src/modeling/team_factors.py` - Violation #2
- `src/modeling/dataset.py` - Violation #2
- `scripts/build_fresh_training_data.py` - Violation #3

### Correct Implementations (Reference)
- `src/ingestion/injuries.py` - `fetch_all_injuries()` ✅
- `src/utils/team_names.py` - `normalize_team_name()` ✅
- `src/ingestion/the_odds.py` - `fetch_odds()` ✅
- `src/ingestion/betting_splits.py` - `fetch_public_betting_splits()` ✅

---

## Testing

After fixes, verify with:

```bash
# Run new validation test
pytest tests/test_single_source_of_truth.py -v

# Check for violations manually
grep -r "fetch_injuries_espn\|fetch_injuries_api_basketball" src/ scripts/
grep -r "def normalize_team_name" src/modeling/
grep -r "fetch_historical_odds" scripts/

# All three commands should return NO results
```

---

## Summary

Your model is **well-designed with excellent documentation**, but **3 implementations have drifted from the single source principle**. These are clean, straightforward fixes that will:

- ✅ Ensure consistent data across all components
- ✅ Improve model reliability and reduce bugs
- ✅ Make maintenance easier (changes in one place)
- ✅ Eliminate feature mismatches between training and production
- ✅ Make debugging and tracing easier

**Total fix time: ~1 hour**  
**Risk level: Very Low** (no API changes, only internal cleanup)  
**Benefit: High** (prevents subtle but critical data inconsistencies)

---

**For detailed fixes with code examples, see:**  
📄 [docs/SINGLE_SOURCE_OF_TRUTH_AUDIT.md](docs/SINGLE_SOURCE_OF_TRUTH_AUDIT.md)

