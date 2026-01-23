# Teams Webhook Integration Setup

**Status:** ✅ Script Ready | ⚠️ Environment Variables Required
**Script:** [scripts/post_to_teams.py](scripts/post_to_teams.py)

---

## 🎯 What It Does

The Teams webhook integration automatically posts daily NBA picks to a Microsoft Teams channel using an Adaptive Card format.

**Features:**
- Beautiful adaptive card layout with fire ratings
- Automatic tier classification (ELITE, STRONG, GOOD)
- Model version and timestamp tracking
- Sorted by fire rating and edge
- CST timezone formatting

---

## ⚙️ Setup Instructions

### Step 1: Create Teams Webhook

1. Open your Microsoft Teams channel
2. Click the **"..."** menu next to the channel name
3. Select **"Connectors"**
4. Find **"Incoming Webhook"** and click **"Configure"**
5. Give it a name (e.g., "NBA Picks Bot")
6. Copy the webhook URL (starts with `https://outlook.office.com/webhook/...`)

### Step 2: Configure Environment Variables

**Option A: Temporary (Current Session)**
```bash
# Set Teams webhook URL
export TEAMS_WEBHOOK_URL="https://outlook.office.com/webhook/YOUR_WEBHOOK_URL_HERE"

# Set NBA API URL (production)
export NBA_API_URL="https://nba-gbsv-api.livelycoast-b48c3cb0.eastus.azurecontainerapps.io"

# Optional: Set model version (reads from model_pack.json if not set)
export NBA_MODEL_VERSION="NBA_v33.0.21.0"
```

**Option B: Permanent (Add to .bashrc or .zshrc)**
```bash
# Add to ~/.bashrc or ~/.zshrc
echo 'export TEAMS_WEBHOOK_URL="https://outlook.office.com/webhook/YOUR_WEBHOOK_URL_HERE"' >> ~/.bashrc
echo 'export NBA_API_URL="https://nba-gbsv-api.livelycoast-b48c3cb0.eastus.azurecontainerapps.io"' >> ~/.bashrc
source ~/.bashrc
```

**Option C: Environment File (Recommended)**
Create a `.env` file in the project root:
```bash
TEAMS_WEBHOOK_URL=https://outlook.office.com/webhook/YOUR_WEBHOOK_URL_HERE
NBA_API_URL=https://nba-gbsv-api.livelycoast-b48c3cb0.eastus.azurecontainerapps.io
NBA_MODEL_VERSION=NBA_v33.0.21.0
```

Then source it before running:
```bash
set -a; source .env; set +a
python scripts/post_to_teams.py
```

### Step 3: Test the Integration

```bash
# Post today's picks to Teams
python scripts/post_to_teams.py

# Post picks for specific date
python scripts/post_to_teams.py --date 2026-01-16

# Use local API instead of production (for testing)
python scripts/post_to_teams.py --local
```

---

## 📋 Usage Examples

### Daily Production Use
```bash
# Set environment variables (one time per session)
export TEAMS_WEBHOOK_URL="your_webhook_url"
export NBA_API_URL="https://nba-gbsv-api.livelycoast-b48c3cb0.eastus.azurecontainerapps.io"

# Post today's picks
python scripts/post_to_teams.py
```

### Scheduled Automation (Cron Job)
```bash
# Add to crontab (runs daily at 10 AM CST)
# 0 10 * * * cd /path/to/nba_gbsv_local && export TEAMS_WEBHOOK_URL="..." && export NBA_API_URL="..." && python scripts/post_to_teams.py
```

### Manual Testing
```bash
# Test with specific date
python scripts/post_to_teams.py --date 2026-01-16

# Test with local API (Docker)
python scripts/post_to_teams.py --local
```

---

## 🎨 Card Format

The Teams message includes:

**Header:**
- Title: "🏀 NBA PICKS - MM/DD/YYYY @ HH:MM am/pm CST"
- Summary: Total picks count and tier breakdown
- Model version and last updated timestamp

**Pick Details (Per Row):**
- **Period:** FG (Full Game) or 1H (First Half)
- **Matchup:** Away @ Home team names
- **Pick:** Bet recommendation with confidence %
- **Market:** Market line/odds
- **Edge:** Model edge in points with fire rating emoji

**Footer:**
- Generation timestamp (CST)
- Model version
- Model last updated timestamp

---

## 🔍 Verification

### Check Environment Variables
```bash
# Verify variables are set
echo "TEAMS_WEBHOOK_URL: ${TEAMS_WEBHOOK_URL:0:50}..."  # Show first 50 chars
echo "NBA_API_URL: $NBA_API_URL"
echo "NBA_MODEL_VERSION: $NBA_MODEL_VERSION"
```

### Test API Connection
```bash
# Test production API
curl "https://nba-gbsv-api.livelycoast-b48c3cb0.eastus.azurecontainerapps.io/slate/today/executive"

# Should return JSON with picks
```

### Test Teams Webhook
```bash
# Send test message
python scripts/post_to_teams.py --date today
```

Expected output:
```
[CONFIG] Using API: https://nba-gbsv-api.livelycoast-b48c3cb0.eastus.azurecontainerapps.io
[FETCH] Fetching predictions for today...
[DATA] Found 14 plays
[POST] Posting to Teams...
[OK] Successfully posted to Teams!
[OK] Posted: 14 plays (2 ELITE, 5 STRONG)
```

---

## 🛠️ Troubleshooting

### Error: "TEAMS_WEBHOOK_URL environment variable is required"
**Solution:** Set the environment variable before running:
```bash
export TEAMS_WEBHOOK_URL="your_webhook_url_here"
```

### Error: "Failed to fetch from API"
**Solution:** Check API URL is correct and accessible:
```bash
# Test API connectivity
curl "https://nba-gbsv-api.livelycoast-b48c3cb0.eastus.azurecontainerapps.io/health"

# Set correct API URL
export NBA_API_URL="https://nba-gbsv-api.livelycoast-b48c3cb0.eastus.azurecontainerapps.io"
```

### Error: "Failed to post to Teams"
**Possible causes:**
1. Invalid webhook URL
2. Webhook expired/disabled
3. Network connectivity issues

**Solution:**
1. Verify webhook URL is correct (copy again from Teams)
2. Recreate webhook in Teams if expired
3. Test connectivity: `curl -X POST $TEAMS_WEBHOOK_URL -H "Content-Type: application/json" -d '{"text":"test"}'`

### No Picks Returned
**Solution:** Check if picks exist for the date:
```bash
curl "https://nba-gbsv-api.livelycoast-b48c3cb0.eastus.azurecontainerapps.io/slate/today"
```

---

## 🔄 Integration with Monitoring

You can combine Teams posting with Week 1 monitoring:

```bash
# Daily workflow
# 1. Save picks locally
python scripts/predict_unified_save_daily_picks.py

# 2. Post to Teams
python scripts/post_to_teams.py

# 3. View monitoring report
python scripts/monitor_week1_performance.py --date $(date +%Y-%m-%d)
```

Or create a combined script:
```bash
#!/bin/bash
# daily_update.sh

# Set environment
export TEAMS_WEBHOOK_URL="your_webhook_url"
export NBA_API_URL="https://nba-gbsv-api.livelycoast-b48c3cb0.eastus.azurecontainerapps.io"

# Save picks
python scripts/predict_unified_save_daily_picks.py

# Post to Teams
python scripts/post_to_teams.py

# Show monitoring report
python scripts/monitor_week1_performance.py --date $(date +%Y-%m-%d)
```

---

## 📚 Script Details

**Location:** [scripts/post_to_teams.py](scripts/post_to_teams.py)

**Required Environment Variables:**
- `TEAMS_WEBHOOK_URL` - Microsoft Teams incoming webhook URL (REQUIRED)
- `NBA_API_URL` - NBA API base URL (optional, defaults to localhost:8090)
- `NBA_MODEL_VERSION` - Model version string (optional, reads from model_pack.json)
- `NBA_MODEL_PACK_PATH` - Path to model_pack.json (optional, defaults to models/production/model_pack.json)

**Command Line Arguments:**
- `--date` - Date to fetch (YYYY-MM-DD or "today"), default: "today"
- `--local` - Use local API (http://localhost:8090) instead of production

**API Endpoints Used:**
- `/slate/{date}/executive` - Executive summary with formatted plays

---

## ✅ Current Status

**Script Status:** ✅ Ready and tested
**Integration:** ⚠️ Requires environment variable setup
**Production API:** ✅ Operational
**Model Version:** NBA_v33.0.21.0

**To Complete Setup:**
1. Get Teams webhook URL from your Teams channel
2. Set `TEAMS_WEBHOOK_URL` environment variable
3. Set `NBA_API_URL` to production endpoint
4. Run `python scripts/post_to_teams.py` to test

---

*Last Updated: 2026-01-16*
*Production API: https://nba-gbsv-api.livelycoast-b48c3cb0.eastus.azurecontainerapps.io*
