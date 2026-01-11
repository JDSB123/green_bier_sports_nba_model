# Data Gaps and Required Actions

## Current State (as of 2026-01-10)

### Training Data Coverage

| Data | Source | Coverage | Date Range |
|------|--------|----------|------------|
| Games/Scores | Kaggle | 100% | 2008-2025 |
| FG Betting Lines | Kaggle + TheOdds | 100% | 2008-2025 |
| 1H Betting Lines | TheOdds API | 77% | May 2023+ |
| Box Scores | wyattowalsh/basketball | 22.5% | 1946-June 2023 |
| ELO Ratings | FiveThirtyEight | 0% | 1947-2015 (OUTDATED!) |

### Feature Count: 291

- Rolling stats: 199
- Advanced (Four Factors): 45
- Ratings: 34
- Situational: 8
- Lines: 10
- Labels: 7

---

## CRITICAL GAPS

### 1. Box Scores for 2023-24 and 2024-25 Seasons

**Problem:** wyattowalsh/basketball dataset ends June 2023.  
**Impact:** No Four Factors, offensive/defensive ratings, or pace data for recent games.

**Solution:** Use API-Basketball to fetch box scores.

```powershell
# Fetch box scores from API-Basketball
python scripts/collect_api_basketball.py --season 2024
python scripts/collect_api_basketball.py --season 2025
```

### 2. ELO Ratings

**Problem:** FiveThirtyEight ELO data ends in 2015.  
**Impact:** No team strength ratings for recent games.

**Solutions:**
1. **Compute our own ELO** from historical game results (preferred)
2. Use alternative sources (NBA.com power rankings, betting market implied ratings)

### 3. Q1 Betting Lines

**Problem:** Not currently extracting Q1 period odds from TheOdds API.  
**Impact:** Cannot train Q1 spread/total models.

**Solution:** Extract from existing period_odds files (data already exists).

---

## ACTION ITEMS

### Immediate (High Priority)

1. **[DONE]** Create standardization module (`src/data/standardization.py`)
2. **[DONE]** Build maximum training data with all available features
3. **[TODO]** Fetch 2023-24 and 2024-25 box scores from API-Basketball
4. **[TODO]** Compute rolling ELO from game results

### Short-Term

5. **[TODO]** Extract Q1 betting lines from TheOdds period_odds
6. **[TODO]** Add line movement features (opening vs closing)
7. **[TODO]** Integrate player injury data

### Long-Term

8. **[TODO]** Add play-by-play derived features (clutch, momentum)
9. **[TODO]** Add conference/division strength features
10. **[TODO]** Add travel/timezone features

---

## Verification

Run these scripts to verify data quality:

```powershell
# Verify standardization
python scripts/verify_data_standardization.py

# Audit all available data
python scripts/audit_all_available_data.py

# Build maximum training data
python scripts/build_training_data_complete.py --start-date 2023-01-01
```

---

## File Locations

| File | Purpose |
|------|---------|
| `src/data/standardization.py` | SINGLE SOURCE OF TRUTH for team names, CST timezone |
| `scripts/build_training_data_complete.py` | Builds training data with all features |
| `scripts/verify_data_standardization.py` | Verifies data integrity |
| `scripts/audit_all_available_data.py` | Audits all available data sources |
| `data/processed/training_data_maximum_2023.csv` | Output training data |
