# Training Data Consolidation Report

**Date:** 2026-01-15
**Status:** CRITICAL - Multiple training data files found

---

## Issue Identified

Two training data files exist with different contents:

| File | Rows | Columns | Date Range | FG Coverage | 1H Coverage |
|------|------|---------|------------|-------------|-------------|
| **training_data.csv** | 3,969 | 327 | 2023-01-01 to 2026-01-08 | 100% | 81.2% |
| **master_training_data.csv** | 3,195 | 327 | 2023-10-24 to 2026-01-08 | 100% | 99.7% |

**Problem:** Different scripts reference different files, creating inconsistency.

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
- ⚠️ Lower 1H coverage (81.2% vs 99.7%)
- ⚠️ Includes partial season data (Jan-Apr 2023)

### master_training_data.csv (3,195 games)
**Pros:**
- ✅ Clean season boundaries (starts 2023-10-24)
- ✅ Better 1H coverage (99.7%)
- ✅ Used by recent optimization agents
- ✅ Used by backtest_extended.py

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
```bash
mv data/processed/master_training_data.csv data/processed/_archive/master_training_data_20260115.csv
```

### 2. Update All References

**Scripts to Update:**
- ✅ `scripts/backtest_extended.py` - Change default to `training_data.csv`
- ✅ `audit_market_coverage.py` - Change to `training_data.csv`
- ✅ Recent optimization scripts - Verify they use correct file

**Documentation to Update:**
- ✅ Confirm DATA_SINGLE_SOURCE_OF_TRUTH.md is accurate
- ✅ Add note about master_training_data.csv deprecation

### 3. Verify Model Training

**Check:** Were production models trained on `training_data.csv` or `master_training_data.csv`?

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
- [  ] training_data.csv is the ONLY canonical file
- [  ] master_training_data.csv is archived/removed
- [  ] All scripts reference training_data.csv
- [  ] Production models were trained on training_data.csv
- [  ] Azure blob has latest version uploaded
- [  ] Documentation is updated

---

## Impact on Recent Work

**Optimization Framework (v33.0.20.0):**
- ⚠️ Agents may have used `master_training_data.csv` (3,195 games)
- ✅ Results are still valid (cleaner data, better 1H coverage)
- ⚠️ Consider re-running with `training_data.csv` (3,969 games) for consistency

**Recommendation:**
1. Use current optimization results (based on 3,195 games) for now
2. After deployment, re-run optimizations with full 3,969 game dataset
3. Compare results and update parameters if significantly different

---

*Generated: 2026-01-15*
*Priority: CRITICAL - Must resolve before production deployment*
