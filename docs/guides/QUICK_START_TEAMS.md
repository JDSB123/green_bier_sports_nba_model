# Quick Start: Teams Webhook Setup

**Time Required:** 2 minutes
**Status:** ✅ Tools Ready | ⏳ Awaiting Your Webhook URL

---

## 🚀 Super Quick Setup (3 Steps)

### Step 1: Get Your Webhook URL (1 minute)

1. Open your Microsoft Teams channel where you want picks posted
2. Click the **"..."** (three dots) next to the channel name
3. Select **"Connectors"**
4. Search for **"Incoming Webhook"**
5. Click **"Configure"**
6. Name it: **"NBA Picks Bot"**
7. Click **"Create"**
8. **Copy the webhook URL** (starts with `https://outlook.office.com/webhook/...`)

### Step 2: Run Interactive Setup (1 minute)

```bash
python scripts/setup_teams_webhook.py
```

The script will:
- ✅ Prompt you to paste the webhook URL
- ✅ Confirm production API URL
- ✅ Create .env file automatically
- ✅ Test the connection
- ✅ Post a test message to Teams

### Step 3: Daily Usage

```bash
# Load environment (once per terminal session)
# PowerShell:
.\load_env.ps1

# CMD:
load_env.bat

# Then post to Teams:
python scripts/post_to_teams.py
```

---

## 📋 What the Interactive Setup Does

When you run `python scripts/setup_teams_webhook.py`, it will:

1. **Prompt for webhook URL** - Paste the URL you copied from Teams
2. **Confirm API URL** - Press Enter to use production (default)
3. **Create .env file** - Saves your configuration
4. **Test connection** - Sends a test post to your Teams channel
5. **Show next steps** - Tells you how to use it daily

---

## 🎯 After Setup

Once setup is complete, posting picks to Teams is just:

```bash
# Option 1: Manual (load env first)
.\load_env.ps1                    # Load environment
python scripts/post_to_teams.py   # Post to Teams

# Option 2: One-liner (PowerShell)
.\load_env.ps1; python scripts/post_to_teams.py
```

---

## 📊 What Gets Posted to Teams

A beautiful adaptive card with:

- **Header**: Date, time (CST), pick counts (ELITE/STRONG/GOOD)
- **Picks Table**:
  - Period (FG/1H)
  - Matchup (Away @ Home)
  - Pick with confidence %
  - Market line
  - Edge + Fire rating (🔥🔥🔥 for ELITE)
- **Footer**: Model version, timestamp

**Example:**
```
🏀 NBA PICKS - 01/16/2026 @ 10:00 am CST
📊 14 Total | 🔥🔥🔥 2 ELITE | 🔥🔥 5 STRONG | 🔥 3 GOOD

PERIOD | MATCHUP          | PICK           | MARKET | EDGE
FG     | CLE @ PHI        | PHI -5.5 (67%) | -5.5   | +2.5pts 🔥🔥
1H     | NOP @ IND        | UNDER (72%)    | 124.5  | +4.6pts 🔥🔥🔥
...

Generated 2026-01-16 10:00 AM CST | Model NBA_v33.0.21.0
```

---

## 🔧 Troubleshooting

### "TEAMS_WEBHOOK_URL environment variable is required"

**Solution:** Run the setup script first:
```bash
python scripts/setup_teams_webhook.py
```

### "Failed to post to Teams"

**Possible causes:**
1. Webhook URL expired (recreate in Teams)
2. Wrong URL copied (copy again)
3. Network connectivity

**Fix:** Rerun setup with correct URL:
```bash
python scripts/setup_teams_webhook.py
```

### Environment variables not loading

**PowerShell:**
```powershell
.\load_env.ps1
```

**CMD:**
```cmd
load_env.bat
```

**Manual (PowerShell):**
```powershell
$env:TEAMS_WEBHOOK_URL = "your_webhook_url_here"
$env:NBA_API_URL = "https://nba-gbsv-api.livelycoast-b48c3cb0.eastus.azurecontainerapps.io"
```

---

## 📁 Files Created

After running setup:

- **`.env`** - Your configuration (NOT committed to git, kept secret)
- **`.env.template`** - Template for reference (committed to git)
- **`load_env.ps1`** - PowerShell loader (committed to git)
- **`load_env.bat`** - CMD loader (committed to git)

---

## 🔒 Security Note

Your `.env` file contains the webhook URL and is automatically excluded from git commits (via `.gitignore`). This keeps your webhook URL private.

**Never share your .env file or webhook URL publicly!**

---

## ✅ Complete Workflow Example

```powershell
# One-time setup (2 minutes)
python scripts/setup_teams_webhook.py
# (Paste webhook URL when prompted, test succeeds)

# Daily use (10 seconds)
.\load_env.ps1
python scripts/post_to_teams.py
# (Picks posted to Teams channel)

# Optional: Combine with monitoring
python scripts/predict_unified_save_daily_picks.py                           # Save picks
python scripts/post_to_teams.py                              # Post to Teams
python scripts/monitor_week1_performance.py --date 2026-01-16 # View report
```

---

## 🎯 Current Status

**Setup Tools:** ✅ Ready (commit fcbec9a)
**Production API:** ✅ Operational
**Model Version:** NBA_v33.0.21.0
**Your Status:** ⏳ **Run `python scripts/setup_teams_webhook.py` to complete setup**

---

*For detailed documentation, see [TEAMS_WEBHOOK_SETUP.md](TEAMS_WEBHOOK_SETUP.md)*
*For monitoring guide, see [WEEK1_MONITORING_GUIDE.md](WEEK1_MONITORING_GUIDE.md)*
