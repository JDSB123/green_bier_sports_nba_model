# Training Data Consolidation Report

**Date:** 2026-01-15
**Status:** RESOLVED - Single canonical dataset enforced

**Canonical (as of 2026-01-17):** Azure `nbagbsvstrg/nbahistoricaldata/training_data/latest/` (version `v20260117_175446`, sha256 `0eea968d…e91d1`) mirrored locally at `data/processed/training_data.csv`.

---

## Issue Identified

Historically, two training data files existed with different contents. This has been resolved.

| File | Rows | Columns | Date Range | FG Coverage | 1H Coverage |
|------|------|---------|------------|-------------|-------------|
| **training_data.csv** | 3,969 | 327 | 2023-01-01 to 2026-01-08 | 100% | 81.2% |
| **master_training_data.csv** | (deprecated) | — | — | — | — |

**Resolution:** Backtests and optimization now consume only `training_data.csv` (Azure `latest`).

---

## Which File Is Correct?

### training_data.csv (3,969 games)
**Pros:**
- ✅ More comprehensive (includes partial 2022-23 season)
- ✅ Documented as canonical source in DATA_SINGLE_SOURCE_OF_TRUTH.md
- ✅ Used by backtest_production.py
- ✅ More training data = better model generalization
- ✅ 100% FG spread/total/ML coverage

**Cons:**
- ⚠️ Lower 1H coverage overall (81.2%), driven by early-2023 segment
- ⚠️ Includes partial season data (Jan-Apr 2023)

### master_training_data.csv (deprecated)
**Pros:**
-- (Historical note) This was used in some earlier optimization work, but it is no longer a supported input.

**Replacement:** use `training_data.csv` and apply coverage windowing (e.g., since 2023-05-01) when evaluating 1H markets.

**Cons:**
- ❌ Fewer games (774 fewer than training_data.csv)
- ❌ Not documented as official canonical source
- ❌ Creates confusion with two files

---

## Decision: SINGLE SOURCE OF TRUTH

**SELECTED:** `data/processed/training_data.csv` (3,969 games)

**Rationale:**
1. Documented as canonical source in official docs
2. More comprehensive training set
3. 100% FG coverage across all markets
4. 81.2% 1H coverage is still acceptable (3,221 games)
5. More data generally improves model robustness

---

## Action Plan

### 1. Rename/Archive master_training_data.csv
Status: ✅ Completed (master_training_data.csv is deprecated; do not use for backtests/optimizations)

### 2. Update All References

**Scripts to Update:**
- ✅ `scripts/backtest_extended.py` - Change default to `training_data.csv`
- ✅ `audit_market_coverage.py` - Change to `training_data.csv`
- ✅ Recent optimization scripts - Verify they use correct file

**Documentation to Update:**
- ✅ Confirm DATA_SINGLE_SOURCE_OF_TRUTH.md is accurate
- ✅ Add note about master_training_data.csv deprecation

### 3. Verify Model Training

Optional check: confirm production models were trained against canonical features.

```bash
# Check model metadata
python -c "import joblib; m=joblib.load('models/production/fg_spread_model.joblib'); print(m.get('metadata', {}))"
```

### 4. Re-run Optimizations (if needed)

If optimization agents used wrong file, re-run with correct file:
- Spreads optimization
- Totals optimization
- Moneylines optimization

---

## Single Source of Truth - FINAL

**Canonical Training Data:**
```
data/processed/training_data.csv
```

**Stats:**
- Games: 3,969
- Columns: 327
- Date Range: 2023-01-01 to 2026-01-08
- FG Spread Coverage: 100% (3,969 games)
- FG Total Coverage: 100% (3,969 games)
- FG Moneyline Coverage: 100% (3,969 games)
- 1H Spread Coverage: 81.2% (3,221 games)
- 1H Total Coverage: 81.2% (3,221 games)
- 1H Moneyline Coverage: 81.2% (3,221 games)

**Azure Blob Backup:**
- Storage Account: `nbagbsvstrg`
- Container: `nbahistoricaldata`
- Path: `training_data/latest/training_data.csv`

**Version Control:**
- Committed in Git: YES
- Tagged: v33.0.17.0 (last major data update)

---

## Verification Checklist

Before deployment, verify:
- [x] training_data.csv is the ONLY canonical file
- [x] master_training_data.csv is archived/removed
- [x] All scripts reference training_data.csv
- [x] Azure blob has latest version uploaded
- [x] Documentation is updated

---

## Impact on Recent Work

**Optimization Framework (v33.0.20.0):**
- Historical note: early optimization work referenced `master_training_data.csv`.
- Current policy: all optimizations/backtests consume only canonical `training_data.csv` (Azure `training_data/latest`).

**Recommendation:**
1. Use canonical `training_data.csv` for any future optimization runs.
2. When comparing results across runs, always record the Azure manifest version + sha256.

---

*Generated: 2026-01-15*
*Priority: CRITICAL - Must resolve before production deployment*
