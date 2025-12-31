#!/bin/bash
#
# Quick runner script for tonight's NBA predictions
# 
# This script starts the Docker container (if not running) and fetches predictions
# for tonight's slate using the Python orchestration script.
#
# Usage:
#   ./run_tonight_predictions.sh              # Today's predictions
#   ./run_tonight_predictions.sh tomorrow     # Tomorrow's predictions
#   ./run_tonight_predictions.sh "Lakers"     # Filter by team
#
# Requirements:
#   - Docker Desktop running
#   - API keys configured in secrets/ directory or .env file
#   - Python 3.x installed
#
# Output:
#   - Console output with predictions
#   - Files saved to data/processed/slate_output_*.txt and *.html
#   - Archived to archive/slate_outputs/

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}🏀 NBA Prediction System - Tonight's Slate${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running${NC}"
    echo "Please start Docker Desktop and try again"
    exit 1
fi
echo -e "${GREEN}✅ Docker is running${NC}"

# Check for API keys
if [ ! -f "secrets/THE_ODDS_API_KEY" ] && [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠️  No API keys found${NC}"
    echo ""
    echo "Please configure API keys either by:"
    echo "  1. Creating secrets directory:"
    echo "     mkdir -p secrets"
    echo "     echo 'your_key_here' > secrets/THE_ODDS_API_KEY"
    echo "     echo 'your_key_here' > secrets/API_BASKETBALL_KEY"
    echo ""
    echo "  2. Or create .env file from .env.example:"
    echo "     cp .env.example .env"
    echo "     # Edit .env with your API keys"
    echo ""
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python 3 found${NC}"

# Parse arguments
DATE_ARG="${1:-today}"
MATCHUP_ARG="${2:-}"

echo ""
echo -e "${BLUE}Running predictions...${NC}"

# Build command array (safe from injection)
CMD_ARGS=(python3 scripts/run_slate.py --date "$DATE_ARG")
if [ -n "$MATCHUP_ARG" ]; then
    CMD_ARGS+=(--matchup "$MATCHUP_ARG")
fi

echo -e "${BLUE}Command: ${CMD_ARGS[*]}${NC}"
echo ""

# Run the prediction script (no eval - direct execution)
"${CMD_ARGS[@]}"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}✅ Predictions completed successfully!${NC}"
    echo -e "${GREEN}============================================${NC}"
    echo ""
    echo -e "${BLUE}Output files:${NC}"
    echo "  📄 Latest predictions: data/processed/slate_output_*.txt"
    echo "  🌐 HTML report: data/processed/slate_output_*.html"
    echo "  📦 Archived to: archive/slate_outputs/"
    echo ""
else
    echo ""
    echo -e "${RED}============================================${NC}"
    echo -e "${RED}❌ Prediction run failed${NC}"
    echo -e "${RED}============================================${NC}"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check Docker logs: docker compose logs nba-v33"
    echo "  2. Verify API health: curl http://localhost:8090/health"
    echo "  3. See docs/DOCKER_TROUBLESHOOTING.md for common issues"
    echo ""
    exit $EXIT_CODE
fi
