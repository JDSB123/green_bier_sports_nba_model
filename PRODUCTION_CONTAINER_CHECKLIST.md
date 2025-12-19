# Production Container Deployment Checklist

**Last Updated:** 2025-12-18  
**Purpose:** Ensure production container deploys successfully every time

---

## Pre-Deployment Verification

### 1. Model Files Check ✅
Before building, verify all required models exist:

```powershell
# Check models/production/ has all 6 required files
Get-ChildItem models/production/*.joblib, models/production/*.pkl
```

**Required Files (6 files for 4 required models):**
- ✅ `spreads_model.joblib` - Full Game Spread
- ✅ `totals_model.joblib` - Full Game Total
- ✅ `first_half_spread_model.pkl` - First Half Spread
- ✅ `first_half_spread_features.pkl` - First Half Spread Features
- ✅ `first_half_total_model.pkl` - First Half Total
- ✅ `first_half_total_features.pkl` - First Half Total Features


### 2. Environment Variables Check ✅

Verify `.env` file has required API keys:
- `THE_ODDS_API_KEY` (required)
- `API_BASKETBALL_KEY` (required)
- `REQUIRE_API_AUTH` (optional, default: false)
- `SERVICE_API_KEY` (required if `REQUIRE_API_AUTH=true`)

---

## Build & Deploy

### Step 1: Build Container
```powershell
docker compose build strict-api
```

**What the build does:**
1. ✅ Installs Python dependencies
2. ✅ Copies application code
3. ✅ Copies models from `models/production/` to `/app/data/processed/models/`
4. ✅ **Verifies all 6 model files exist** (build fails if missing!)
5. ✅ Sets correct file permissions for appuser
6. ✅ Creates health check configuration

### Step 2: Start Container
```powershell
docker compose up -d
```

### Step 3: Verify Startup
```powershell
# Check container logs for errors
docker compose logs strict-api

# Check health endpoint
curl http://localhost:8090/health

# Should return:
# {
#   "status": "ok",
#   "engine_loaded": true,
#   "markets": 6,
#   ...
# }
```

---

## Common Failures & Fixes

### ❌ Failure: "Security validation failed"

**Cause:** Missing required API keys in `.env`

**Fix:**
1. Check `.env` file has `THE_ODDS_API_KEY` and `API_BASKETBALL_KEY`
2. Verify no typos in variable names
3. Restart container: `docker compose restart strict-api`

### ❌ Failure: "Permission denied" on model files

**Cause:** File permissions issue (shouldn't happen with fixed Dockerfile)

**Fix:**
1. Rebuild container (Dockerfile now sets correct permissions)
2. Check logs: `docker compose logs strict-api`

### ❌ Failure: Container exits immediately

**Cause:** Startup validation failed (missing models or API keys)

**Fix:**
1. Check logs: `docker compose logs strict-api`
2. Look for error message indicating what's missing
3. Fix the issue and rebuild/restart

---

## Health Check

The container includes a health check that:
- ✅ Runs every 30 seconds
- ✅ Tests `/health` endpoint
- ✅ Fails after 3 retries
- ✅ Has 15s startup grace period

Check health status:
```powershell
docker compose ps
# Look for "healthy" status
```

---

## Verification Script

Run the verification script inside the container:

```powershell
docker compose exec strict-api python /app/scripts/verify_container_startup.py
```

This will:
- ✅ List all required model files
- ✅ Check file sizes
- ✅ Verify files are readable
- ✅ Show clear error messages if anything is missing

---

## Quick Debug Commands

```powershell
# Check container status
docker compose ps

# View logs
docker compose logs -f strict-api

# Execute command in container
docker compose exec strict-api bash

# List model files in container
docker compose exec strict-api ls -lah /app/data/processed/models/

# Test health endpoint
docker compose exec strict-api curl http://localhost:8080/health

# Check environment variables
docker compose exec strict-api env | grep -E "API_KEY|DATA_PROCESSED_DIR"
```

---

## Success Criteria

✅ Container builds without errors  
✅ Container starts and stays running  
✅ Health check passes: `docker compose ps` shows "healthy"  
✅ `/health` endpoint returns `engine_loaded: true`  
✅ `/slate/today` endpoint returns predictions  

---

## What Changed (Fix Summary)

**Fixed Issues:**
1. ✅ Removed all moneyline model references
2. ✅ Clarified model requirements (4 required models: spreads and totals for FG and 1H)
3. ✅ Updated all comments for consistency (4 markets total)
4. ✅ Added explicit healthcheck to docker-compose.yml
5. ✅ Added diagnostic logging on startup (lists found files)
6. ✅ Created verification script for troubleshooting

**The container will now FAIL FAST with clear error messages if:**
- Any model file is missing
- File permissions are wrong
- API keys are missing

**No more silent failures!** 🎯
