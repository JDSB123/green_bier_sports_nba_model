#!/bin/bash
set -e  # Exit on any error - NO SILENT FAILURES

# ============================================================================
# NBA v5.0 Backtest Entrypoint
# ============================================================================
# This script is the SINGLE ENTRY POINT for all backtest operations.
# It enforces strict validation and never uses placeholder data.
#
# Commands:
#   full      - Full pipeline: fetch data + build training + run backtest
#   data      - Fetch and build training data only
#   backtest  - Run backtest on existing data
#   validate  - Validate existing training data
# ============================================================================

COMMAND=${1:-full}
SEASONS=${SEASONS:-"2024-2025,2025-2026"}
MARKETS=${MARKETS:-"all"}
MIN_TRAINING=${MIN_TRAINING:-80}

echo "============================================================"
echo "NBA v5.0 BACKTEST CONTAINER"
echo "============================================================"
echo "Command: $COMMAND"
echo "Seasons: $SEASONS"
echo "Markets: $MARKETS"
echo "============================================================"
echo ""

# Function to validate required environment variables
validate_env() {
    local errors=0
    
    if [ -z "$API_BASKETBALL_KEY" ]; then
        echo "ERROR: API_BASKETBALL_KEY is not set"
        errors=$((errors + 1))
    fi
    
    if [ -z "$THE_ODDS_API_KEY" ]; then
        echo "ERROR: THE_ODDS_API_KEY is not set"
        errors=$((errors + 1))
    fi
    
    if [ $errors -gt 0 ]; then
        echo ""
        echo "Missing required API keys. Please set them in your .env file:"
        echo "  API_BASKETBALL_KEY=your_key_here"
        echo "  THE_ODDS_API_KEY=your_key_here"
        echo ""
        exit 1
    fi
    
    echo "✓ Environment validated"
}

# Function to validate Python environment
validate_python() {
    echo "Validating Python environment..."
    
    python -c "
import sys
errors = []

# Check critical imports
try:
    from src.config import settings
except ImportError as e:
    errors.append(f'src.config: {e}')

try:
    from src.modeling.models import SpreadsModel, TotalsModel
except ImportError as e:
    errors.append(f'src.modeling.models: {e}')

try:
    from src.modeling.features import FeatureEngineer
except ImportError as e:
    errors.append(f'src.modeling.features: {e}')

try:
    from src.ingestion.api_basketball import APIBasketballClient
except ImportError as e:
    errors.append(f'src.ingestion.api_basketball: {e}')

if errors:
    print('CRITICAL IMPORT ERRORS:')
    for err in errors:
        print(f'  - {err}')
    sys.exit(1)

print('✓ All critical modules imported successfully')
"
}

# Function to fetch fresh data
fetch_data() {
    echo ""
    echo "============================================================"
    echo "STEP 1: FETCHING FRESH DATA"
    echo "============================================================"
    
    python scripts/build_fresh_training_data.py \
        --seasons "$SEASONS" \
        --output training_data.csv
    
    if [ $? -ne 0 ]; then
        echo "✗ Data fetch failed!"
        exit 1
    fi
    
    echo "✓ Fresh data fetched and training data built"
}

# Function to run backtest
run_backtest() {
    echo ""
    echo "============================================================"
    echo "STEP 2: RUNNING BACKTEST"
    echo "============================================================"
    
    # Validate training data exists
    if [ ! -f "/app/data/processed/training_data.csv" ]; then
        echo "ERROR: Training data not found at /app/data/processed/training_data.csv"
        echo "Run with 'data' or 'full' command first to build training data."
        exit 1
    fi
    
    # Run the backtest
    python scripts/backtest.py \
        --markets "$MARKETS" \
        --min-training "$MIN_TRAINING" \
        --data "/app/data/processed/training_data.csv"
    
    if [ $? -ne 0 ]; then
        echo "✗ Backtest failed!"
        exit 1
    fi
    
    # Copy results to results directory with timestamp
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    mkdir -p /app/data/results
    
    if [ -f "/app/data/processed/all_markets_backtest_results.csv" ]; then
        cp /app/data/processed/all_markets_backtest_results.csv \
           /app/data/results/backtest_results_${TIMESTAMP}.csv
        echo "✓ Results saved to /app/data/results/backtest_results_${TIMESTAMP}.csv"
    fi
    
    if [ -f "/app/ALL_MARKETS_BACKTEST_RESULTS.md" ]; then
        cp /app/ALL_MARKETS_BACKTEST_RESULTS.md \
           /app/data/results/backtest_report_${TIMESTAMP}.md
        echo "✓ Report saved to /app/data/results/backtest_report_${TIMESTAMP}.md"
    fi
    
    echo "✓ Backtest completed successfully"
}

# Function to validate existing data
validate_data() {
    echo ""
    echo "============================================================"
    echo "VALIDATING TRAINING DATA"
    echo "============================================================"
    
    python scripts/build_fresh_training_data.py --validate-only
    
    if [ $? -ne 0 ]; then
        echo "✗ Validation failed!"
        exit 1
    fi
    
    echo "✓ Training data validation passed"
}

# Main execution
case "$COMMAND" in
    full)
        validate_env
        validate_python
        fetch_data
        run_backtest
        echo ""
        echo "============================================================"
        echo "FULL PIPELINE COMPLETED SUCCESSFULLY"
        echo "============================================================"
        ;;
    
    data)
        validate_env
        validate_python
        fetch_data
        echo ""
        echo "============================================================"
        echo "DATA PIPELINE COMPLETED SUCCESSFULLY"
        echo "============================================================"
        ;;
    
    backtest)
        validate_python
        run_backtest
        echo ""
        echo "============================================================"
        echo "BACKTEST COMPLETED SUCCESSFULLY"
        echo "============================================================"
        ;;
    
    validate)
        validate_python
        validate_data
        ;;
    
    shell)
        echo "Starting interactive shell..."
        exec /bin/bash
        ;;
    
    *)
        echo "Unknown command: $COMMAND"
        echo ""
        echo "Available commands:"
        echo "  full      - Full pipeline: fetch data + build training + run backtest"
        echo "  data      - Fetch and build training data only"
        echo "  backtest  - Run backtest on existing data"
        echo "  validate  - Validate existing training data"
        echo "  shell     - Interactive shell for debugging"
        exit 1
        ;;
esac

