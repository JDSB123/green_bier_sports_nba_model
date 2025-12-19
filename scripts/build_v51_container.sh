#!/bin/bash
# =============================================================================
# NBA v5.1 FINAL - Container Build & Export Script
# =============================================================================
# 
# Builds the production container and exports to target path.
# 
# Usage:
#   ./scripts/build_v51_container.sh [OUTPUT_DIR]
#
# Default output: /mnt/c/Users/JB/green-bier-ventures/nba_v5.1_model_FINAL
# =============================================================================

set -e  # Exit on error

# Configuration
IMAGE_NAME="nba-v51-final"
IMAGE_TAG="latest"
FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

# Output directory (WSL path to Windows)
DEFAULT_OUTPUT="/mnt/c/Users/JB/green-bier-ventures/nba_v5.1_model_FINAL"
OUTPUT_DIR="${1:-$DEFAULT_OUTPUT}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}NBA v5.1 FINAL - Container Build${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "3 PROVEN ROE Full Game Markets:"
echo "  - Spread: 60.6% accuracy, +15.7% ROI"
echo "  - Total: 59.2% accuracy, +13.1% ROI"
echo "  - Moneyline: 65.5% accuracy, +25.1% ROI"
echo ""

# Step 1: Verify models exist
echo -e "${YELLOW}[1/5] Verifying production models...${NC}"
MODELS_DIR="models/production"
REQUIRED_MODELS=("spreads_model.joblib" "totals_model.joblib" "moneyline_model.joblib")

for model in "${REQUIRED_MODELS[@]}"; do
    if [ -f "$MODELS_DIR/$model" ]; then
        echo -e "  ${GREEN}✓${NC} $model"
    else
        echo -e "  ${RED}✗${NC} $model - MISSING!"
        echo -e "${RED}ERROR: Required model file missing. Aborting.${NC}"
        exit 1
    fi
done
echo ""

# Step 2: Build the container
echo -e "${YELLOW}[2/5] Building Docker image...${NC}"
docker build -f Dockerfile.v51 -t "$FULL_IMAGE" .
echo -e "${GREEN}✓ Image built successfully${NC}"
echo ""

# Step 3: Create output directory
echo -e "${YELLOW}[3/5] Creating output directory...${NC}"
mkdir -p "$OUTPUT_DIR"
echo -e "${GREEN}✓ Output directory: $OUTPUT_DIR${NC}"
echo ""

# Step 4: Export the container image
echo -e "${YELLOW}[4/5] Exporting container image...${NC}"
EXPORT_FILE="$OUTPUT_DIR/nba_v5.1_model_FINAL.tar.gz"
docker save "$FULL_IMAGE" | gzip > "$EXPORT_FILE"
echo -e "${GREEN}✓ Image exported to: $EXPORT_FILE${NC}"
echo ""

# Step 5: Create deployment manifest
echo -e "${YELLOW}[5/5] Creating deployment manifest...${NC}"
MANIFEST_FILE="$OUTPUT_DIR/manifest.json"
BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
IMAGE_ID=$(docker images -q "$FULL_IMAGE")
IMAGE_SIZE=$(docker images --format "{{.Size}}" "$FULL_IMAGE")

cat > "$MANIFEST_FILE" << EOF
{
  "name": "NBA v5.1 FINAL - Production Picks Model",
  "version": "5.1-FINAL",
  "build_date": "$BUILD_DATE",
  "image": {
    "name": "$IMAGE_NAME",
    "tag": "$IMAGE_TAG",
    "id": "$IMAGE_ID",
    "size": "$IMAGE_SIZE"
  },
  "markets": {
    "count": 3,
    "period": "full_game",
    "list": ["spread", "total", "moneyline"]
  },
  "performance": {
    "spread": {"accuracy": 0.606, "roi": 0.157},
    "total": {"accuracy": 0.592, "roi": 0.131},
    "moneyline": {"accuracy": 0.655, "roi": 0.251}
  },
  "models": [
    "spreads_model.joblib",
    "totals_model.joblib",
    "moneyline_model.joblib"
  ],
  "security": {
    "read_only_filesystem": true,
    "non_root_user": true,
    "no_new_privileges": true
  },
  "ports": {
    "api": 8080,
    "external": 8090
  },
  "deployment": {
    "load_command": "docker load -i nba_v5.1_model_FINAL.tar.gz",
    "run_command": "docker run -d -p 8090:8080 --read-only --tmpfs /tmp --env-file .env --name nba-v51 nba-v51-final:latest",
    "health_check": "curl http://localhost:8090/health"
  }
}
EOF
echo -e "${GREEN}✓ Manifest created: $MANIFEST_FILE${NC}"
echo ""

# Copy docker-compose and env template
cp docker-compose.v51.yml "$OUTPUT_DIR/"
if [ -f ".env.example" ]; then
    cp .env.example "$OUTPUT_DIR/.env.example"
fi

# Create README
cat > "$OUTPUT_DIR/README.md" << 'EOF'
# NBA v5.1 FINAL - Production Picks Model

## 3 Proven ROE Full Game Markets

| Market | Accuracy | ROI |
|--------|----------|-----|
| Spread | 60.6% | +15.7% |
| Total | 59.2% | +13.1% |
| Moneyline | 65.5% | +25.1% |

## Quick Start

### 1. Load the container image

```bash
docker load -i nba_v5.1_model_FINAL.tar.gz
```

### 2. Create environment file

```bash
cp .env.example .env
# Edit .env with your API keys
```

Required API keys:
- `THE_ODDS_API_KEY` - From the-odds-api.com
- `API_BASKETBALL_KEY` - From api-basketball.com

### 3. Run the container

```bash
# Using docker-compose (recommended)
docker compose -f docker-compose.v51.yml up -d

# Or using docker run
docker run -d -p 8090:8080 \
  --read-only \
  --tmpfs /tmp \
  --env-file .env \
  --name nba-v51 \
  nba-v51-final:latest
```

### 4. Verify it's running

```bash
# Check health
curl http://localhost:8090/health

# Get today's predictions
curl http://localhost:8090/slate/today
```

## Security Features

- **Read-only filesystem**: Container filesystem is immutable
- **Non-root user**: Runs as unprivileged `appuser` (UID 1000)
- **No new privileges**: Prevents privilege escalation
- **Baked-in models**: Models are immutable in the container image

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check with model info |
| `GET /slate/{date}` | Get predictions for a date |
| `GET /slate/{date}/comprehensive` | Full analysis with edge calculations |
| `POST /predict/game` | Predict single game |
| `GET /metrics` | Prometheus metrics |
| `GET /verify` | Verify model integrity |

## Files in this package

- `nba_v5.1_model_FINAL.tar.gz` - Docker image
- `docker-compose.v51.yml` - Docker compose configuration
- `.env.example` - Environment template
- `manifest.json` - Build metadata
- `README.md` - This file
EOF

echo -e "${GREEN}✓ README created${NC}"
echo ""

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Build Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Contents:"
ls -lh "$OUTPUT_DIR"
echo ""
echo -e "${GREEN}To deploy:${NC}"
echo "  1. docker load -i $EXPORT_FILE"
echo "  2. docker compose -f docker-compose.v51.yml up -d"
echo "  3. curl http://localhost:8090/health"
echo ""

