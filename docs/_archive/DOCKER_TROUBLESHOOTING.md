# Docker Backtest Troubleshooting Guide

## Common Issues and Solutions

### Issue: Container fails to start or exits immediately

**Symptoms:**
- Container exits with code 1
- No output or minimal error messages
- "Model struggling to kick off"

**Diagnosis Steps:**

1. **Run diagnostics:**
   ```powershell
   docker-compose -f docker-compose.backtest.yml up backtest-diagnose
   ```

2. **Optional: Check environment variables (only needed for live ingestion):**
   ```powershell
   # If you use live ingestion elsewhere
   cat .env | Select-String "API_KEY"
   ```

3. **Test Python imports manually:**
   ```powershell
   docker-compose -f docker-compose.backtest.yml run --rm backtest-shell
   # Inside container:
   python -c "from src.config import settings; print('OK')"
   ```

### Note: API Keys

Backtests that rely on canonical `data/processed/training_data.csv` do not require API keys.
Keys are only needed for live ingestion or raw rebuild workflows outside the backtest container.

### Issue: Python Import Errors

**Error:**
```
CRITICAL IMPORT ERRORS:
  - src.config: No module named 'src'
  - src.modeling.models: No module named 'src.modeling'
```

**Possible Causes:**
1. **Missing dependencies:**
   ```powershell
   # Rebuild the image
   docker-compose -f docker-compose.backtest.yml build --no-cache
   ```

2. **PYTHONPATH not set:**
   - The Dockerfile sets `ENV PYTHONPATH=/app`
   - If this isn't working, check the Dockerfile

3. **Source files not copied:**
   - Verify `src/` directory exists in the container:
   ```powershell
   docker-compose -f docker-compose.backtest.yml run --rm backtest-shell
   # Inside container:
   ls -la /app/src/
   ```

### Issue: Training Data Not Found

**Error:**
```
ERROR: Training data not found at /app/data/processed/training_data.csv
```

**Solution:**
1. **Ensure canonical data exists:**
   - `data/processed/training_data.csv` (audited 2023+ dataset)
   - You can run `backtest-data` to verify presence:
   ```powershell
   docker-compose -f docker-compose.backtest.yml up backtest-data
   ```

2. **Check volume mount:**
   - Verify `./data:/app/data` is mounted correctly
   - Check that `data/processed/` directory exists locally

3. **Verify file was created:**
   ```powershell
   # After running backtest-data
   ls -lh data/processed/training_data.csv
   ```

### Issue: Model Training Fails During Backtest

**Note:** The backtest script trains models on-the-fly (walk-forward validation). You don't need pre-trained models.

**If training fails:**
1. **Check training data quality:**
   ```powershell
   docker-compose -f docker-compose.backtest.yml up backtest-validate
   ```

2. **Check for sufficient data:**
   - Minimum training games: 80 (default)
   - Need at least 30 historical games per feature calculation
   - Need at least 50 games for model training

3. **Check error messages:**
   - The improved entrypoint script now shows detailed error messages
   - Look for specific feature calculation or model training errors

### Issue: Container Build Fails

**Error during `docker build`:**

1. **Check Dockerfile syntax:**
   ```powershell
   docker build -f Dockerfile.backtest -t nba-backtest-test . 2>&1 | Select-Object -Last 20
   ```

2. **Common build issues:**
   - Missing `requirements.txt`
   - Network issues downloading packages
   - Disk space issues

3. **Clean rebuild:**
   ```powershell
   docker system prune -f
   docker-compose -f docker-compose.backtest.yml build --no-cache
   ```

## Quick Diagnostic Commands

### Check container status
```powershell
docker ps -a --filter "name=backtest"
```

### View logs
```powershell
docker-compose -f docker-compose.backtest.yml logs backtest-full
```

### Run diagnostics
```powershell
docker-compose -f docker-compose.backtest.yml up backtest-diagnose
```

### Interactive debugging
```powershell
docker-compose -f docker-compose.backtest.yml run --rm backtest-shell
```

### Test individual steps
```powershell
# Just confirm canonical data
docker-compose -f docker-compose.backtest.yml up backtest-data

# Just validate
docker-compose -f docker-compose.backtest.yml up backtest-validate

# Just run backtest (requires existing data)
docker-compose -f docker-compose.backtest.yml up backtest-only
```

## What the Entrypoint Script Does

The `docker-entrypoint-backtest.sh` script:

1. **Validates Python** - Tests critical imports
2. **Confirms data** (if `data` or `full` command) - Requires `data/processed/training_data.csv`
3. **Runs backtest** (if `backtest` or `full` command) - Calls `historical_backtest_production.py`
4. **Validates data** (if `validate` command) - Calls `data_unified_validate_training.py`
5. **Cleans cache** (if `backtest` or `full` command) - Removes ephemeral data

## Recent Improvements

The entrypoint script has been improved with:
- ✅ Better error messages with context
- ✅ File existence and size checks
- ✅ Diagnostic command for troubleshooting
- ✅ More detailed Python import validation
- ✅ Directory structure verification

## Still Having Issues?

1. Run the diagnostic command and share the output
2. Check Docker logs: `docker-compose -f docker-compose.backtest.yml logs`
3. Try the interactive shell to debug manually
4. Verify your `.env` file has valid API keys
5. Check that you have sufficient disk space
