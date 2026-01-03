# Running NBA Predictions for Tonight's Slate

This guide covers multiple ways to generate predictions for tonight's NBA games.

## Quick Start (Recommended)

### Method 1: One-Command Script (Local)

The fastest way to get tonight's predictions:

```bash
./run_tonight_predictions.sh
```

This script automatically:
- ✅ Checks if Docker is running
- ✅ Verifies API keys are configured
- ✅ Starts the prediction API container
- ✅ Fetches and displays predictions
- ✅ Saves output to files (TXT and HTML)

**Options:**
```bash
./run_tonight_predictions.sh              # Today's games
./run_tonight_predictions.sh tomorrow     # Tomorrow's games  
./run_tonight_predictions.sh "Lakers"     # Filter by team
```

**Output:**
- Console: Formatted predictions with fire ratings
- File: `data/processed/slate_output_YYYYMMDD_HHMMSS.txt`
- HTML: `data/processed/slate_output_YYYYMMDD_HHMMSS.html`
- Archive: `archive/slate_outputs/`

---

### Method 2: Python Script (Local)

Use the main Python orchestration script directly:

```bash
python scripts/run_slate.py
```

**Options:**
```bash
python scripts/run_slate.py --date today
python scripts/run_slate.py --date tomorrow
python scripts/run_slate.py --date 2025-12-31
python scripts/run_slate.py --matchup "Lakers"
python scripts/run_slate.py --date tomorrow --matchup "Celtics"
```

This is the same command the shell script uses internally.

---

### Method 3: GitHub Actions Workflow (Remote)

Run predictions from GitHub using the cloud infrastructure:

1. **Navigate to Actions tab** in GitHub repository
2. **Select workflow**: "Run NBA Predictions"
3. **Click "Run workflow"**
4. **Choose options**:
   - Date: today or tomorrow
   - Matchup: (optional) filter by team
5. **Click "Run workflow"** button

**Output:**
- Predictions displayed in workflow logs
- Results saved as downloadable artifact
- Artifact retained for 30 days

**Advantages:**
- No local setup required
- Uses production Azure infrastructure
- Can be scheduled or triggered remotely
- Results archived automatically

---

## Prerequisites

### For Local Execution (Methods 1 & 2):

1. **Docker Desktop** - Must be running
   ```bash
   # Verify Docker is running
   docker info
   ```

2. **API Keys** - Configure via secrets or .env file

   **Option A: Docker Secrets (Recommended)**
   ```bash
   mkdir -p secrets
   echo 'your_the_odds_api_key' > secrets/THE_ODDS_API_KEY
   echo 'your_api_basketball_key' > secrets/API_BASKETBALL_KEY
   ```

   **Option B: Environment File**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys:
   # THE_ODDS_API_KEY=your_key_here
   # API_BASKETBALL_KEY=your_key_here
   ```

3. **Python 3.x** - For running scripts
   ```bash
   python3 --version
   ```

### For GitHub Actions (Method 3):

- Azure credentials configured in repository secrets:
  - `AZURE_CLIENT_ID`
  - `AZURE_TENANT_ID`
  - `AZURE_SUBSCRIPTION_ID`
- Container app deployed and running (`nba-gbsv-api`)

---

## How It Works

### Architecture

```
┌─────────────────────────────────────────────┐
│  scripts/run_slate.py (Orchestrator)        │
│  - Checks Docker status                     │
│  - Starts containers if needed              │
│  - Waits for API health                     │
│  - Fetches predictions                      │
│  - Formats and saves output                 │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  Docker Container: nba-v33                  │
│  Port: 8090                                 │
│                                             │
│  ┌─────────────────────────────────────┐   │
│  │  FastAPI Prediction Service         │   │
│  │  - Loads 4 trained models           │   │
│  │  - Fetches live odds                │   │
│  │  - Computes predictions             │   │
│  │  - Returns comprehensive analysis   │   │
│  └─────────────────────────────────────┘   │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  External APIs                              │
│  - The Odds API (betting lines)            │
│  - API-Basketball (game data)              │
└─────────────────────────────────────────────┘
```

### Prediction Process

1. **Container Startup**
   - Docker compose starts the `nba-v33` container
   - API loads 4 trained models (1H spread, 1H total, FG spread, FG total)
   - Health check confirms engine is loaded

2. **Data Collection**
   - Fetches today's NBA schedule
   - Retrieves current betting lines (spread, total, moneyline)
   - Collects team statistics and recent performance

3. **Feature Engineering**
   - Computes rolling averages (last 5, 10 games)
   - Calculates rest days
   - Computes offensive/defensive ratings
   - Applies betting market features

4. **Model Prediction**
   - Each market uses independent model
   - Predicts game margin and total points
   - Calculates confidence scores
   - Computes edge vs market lines

5. **Output Filtering**
   - Filters picks by minimum thresholds:
     - Spread: 55% confidence, 1.0 pt edge
     - Total: 55% confidence, 1.5 pt edge
   - Calculates fire ratings (1-5 🔥)
   - Formats for display

---

## Understanding the Output

### Fire Ratings

🔥🔥🔥🔥🔥 (5 fires) - Elite pick (85%+ combined score)
🔥🔥🔥🔥 (4 fires) - Strong pick (70-85%)
🔥🔥🔥 (3 fires) - Good pick (60-70%)
🔥🔥 (2 fires) - Moderate pick (52-60%)
🔥 (1 fire) - Marginal pick (<52%)

**Combined Score = (Confidence × 0.6) + (Edge Normalized × 0.4)**

### Pick Format

**Spread Picks:**
```
FG Spread: Cleveland Cavaliers -7.5 (-110)
   Model predicts: CLE wins by 9.2 pts
   Market line: -7.5
   Edge: +1.7 pts of value
   Confidence: 62% | 🔥🔥🔥
```

**Total Picks:**
```
FG Total: OVER 223.5 (-110)
   Model predicts: 227.3 total points
   Market line: 223.5
   Edge: 3.8 pts of value
   Confidence: 68% | 🔥🔥🔥🔥
```

### Edge Calculation

**For Spreads:**
- Edge = |Model Margin - Market Line|
- Positive edge = Model gives more value than market

**For Totals:**
- Edge = |Model Total - Market Line|
- Positive edge = Model projects more separation from line

---

## Output Files

### Text File (`slate_output_YYYYMMDD_HHMMSS.txt`)

Contains:
- Summary table of all picks
- Detailed analysis for each game
- Fire ratings and confidence scores
- EV% and Kelly sizing

### HTML File (`slate_output_YYYYMMDD_HHMMSS.html`)

Features:
- Responsive design
- Color-coded picks
- Sortable tables
- Fire rating visualization
- Mobile-friendly

### Archive

Both files are automatically copied to `archive/slate_outputs/` for historical tracking.

---

## Troubleshooting

### "Docker is not running"
```bash
# Start Docker Desktop
# Then retry
./run_tonight_predictions.sh
```

### "API did not become ready in time"
```bash
# Check container logs
docker compose logs nba-v33

# Verify API health manually
curl http://localhost:8090/health

# Restart container
docker compose restart
```

### "No API keys found"
```bash
# Create secrets directory
mkdir -p secrets
echo 'your_key' > secrets/THE_ODDS_API_KEY
echo 'your_key' > secrets/API_BASKETBALL_KEY

# Or create .env file
cp .env.example .env
# Edit .env with your keys
```

### "No games found for this date"

This is normal if:
- Running very early in the day (odds not published yet)
- No games scheduled for that date
- Off-season period

Try:
```bash
./run_tonight_predictions.sh tomorrow
```

### Models Not Loaded

```bash
# Verify model files exist
ls -lh models/production/

# Should see:
# - fg_spread_model_v*.pkl
# - fg_total_model_v*.pkl
# - 1h_spread_model_v*.pkl
# - 1h_total_model_v*.pkl

# If missing, train models:
python scripts/train_models.py
```

---

## Advanced Usage

### Scheduling Predictions

**Linux/Mac (cron):**
```bash
# Run at 12:00 PM daily
0 12 * * * cd /path/to/nba_model && ./run_tonight_predictions.sh >> logs/predictions.log 2>&1
```

**Windows (Task Scheduler):**
1. Create task: "NBA Predictions"
2. Trigger: Daily at 12:00 PM
3. Action: Run program
   - Program: `C:\path\to\run_tonight_predictions.sh`
   - Start in: `C:\path\to\nba_model`

### Filtering Picks

**By team:**
```bash
./run_tonight_predictions.sh today "Lakers"
```

**By multiple teams:**
```bash
./run_tonight_predictions.sh today "Lakers, Celtics, Warriors"
```

**By matchup:**
```bash
./run_tonight_predictions.sh today "Lakers vs Celtics"
```

### Custom API URL

For connecting to remote API:
```bash
export NBA_API_URL="https://nba-gbsv-api.ambitiouscoast-4bcd4cd8.eastus.azurecontainerapps.io"
python scripts/run_slate.py
```

---

## Performance Notes

### System Requirements
- **CPU**: 2+ cores recommended
- **RAM**: 2GB minimum (models loaded in memory)
- **Disk**: ~500MB for models and data
- **Network**: Stable internet for API calls

### Execution Time
- **Container startup**: 15-30 seconds (first time)
- **API health check**: 5-10 seconds
- **Prediction fetch**: 10-30 seconds per game
- **Total time**: ~1-2 minutes for full slate

### API Rate Limits
- **The Odds API**: 500 requests/month (free tier)
- **API-Basketball**: 100 requests/day (free tier)
- Consider paid plans for higher volume

---

## Getting Help

1. **Check documentation**:
   - `docs/DOCKER_TROUBLESHOOTING.md`
   - `docs/STACK_FLOW_AND_VERIFICATION.md`
   - `README.md`

2. **View logs**:
   ```bash
   docker compose logs -f nba-v33
   ```

3. **Verify setup**:
   ```bash
   python scripts/validate_production_readiness.py
   ```

4. **Test API**:
   ```bash
   python scripts/test_all_api_endpoints.py
   ```

---

## Related Commands

```bash
# Build training data
python scripts/build_fresh_training_data.py

# Train models
python scripts/train_models.py

# Run backtest
docker compose -f docker-compose.backtest.yml up backtest-full

# Check data quality
python scripts/check_data_quality.py

# Validate models
python scripts/validate_model.py
```

---

## Version

This guide is for **NBA Model v33.0.8.0** (4 independent markets: 1H + FG)

Last updated: 2025-12-31
