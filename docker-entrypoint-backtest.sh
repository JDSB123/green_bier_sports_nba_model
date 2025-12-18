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
    echo "Python version: $(python --version)"
    echo "Python path: $(python -c 'import sys; print(sys.executable)')"
    echo "Working directory: $(pwd)"
    echo "PYTHONPATH: ${PYTHONPATH:-not set}"
    echo ""
    
    python -c "
import sys
import os
errors = []

# Check Python path
print(f'Python executable: {sys.executable}')
print(f'Python version: {sys.version}')
print(f'Current directory: {os.getcwd()}')
print(f'PYTHONPATH: {os.environ.get(\"PYTHONPATH\", \"not set\")}')
print('')

# Check if src directory exists
if not os.path.exists('/app/src'):
    errors.append('src directory not found at /app/src')
    print('ERROR: /app/src directory does not exist')
    print('Available directories in /app:')
    for item in os.listdir('/app'):
        print(f'  - {item}')

# Check critical imports
try:
    from src.config import settings
    print('✓ src.config imported successfully')
except ImportError as e:
    errors.append(f'src.config: {e}')
    print(f'✗ src.config import failed: {e}')

try:
    from src.modeling.models import SpreadsModel, TotalsModel
    print('✓ src.modeling.models imported successfully')
except ImportError as e:
    errors.append(f'src.modeling.models: {e}')
    print(f'✗ src.modeling.models import failed: {e}')

try:
    from src.modeling.features import FeatureEngineer
    print('✓ src.modeling.features imported successfully')
except ImportError as e:
    errors.append(f'src.modeling.features: {e}')
    print(f'✗ src.modeling.features import failed: {e}')

try:
    from src.ingestion.api_basketball import APIBasketballClient
    print('✓ src.ingestion.api_basketball imported successfully')
except ImportError as e:
    errors.append(f'src.ingestion.api_basketball: {e}')
    print(f'✗ src.ingestion.api_basketball import failed: {e}')

if errors:
    print('')
    print('CRITICAL IMPORT ERRORS:')
    for err in errors:
        print(f'  - {err}')
    print('')
    print('Troubleshooting:')
    print('  1. Check that all dependencies are installed: pip install -r requirements.txt')
    print('  2. Verify src/ directory structure is correct')
    print('  3. Check PYTHONPATH is set to /app')
    sys.exit(1)

print('')
print('✓ All critical modules imported successfully')
"
    
    if [ $? -ne 0 ]; then
        echo ""
        echo "Python validation failed. Check the errors above."
        exit 1
    fi
}

# Function to fetch fresh data
fetch_data() {
    echo ""
    echo "============================================================"
    echo "STEP 1: FETCHING FRESH DATA"
    echo "============================================================"
    echo "Seasons: $SEASONS"
    echo "Output: /app/data/processed/training_data.csv"
    echo ""
    
    # Ensure output directory exists
    mkdir -p /app/data/processed
    
    python scripts/build_fresh_training_data.py \
        --seasons "$SEASONS" \
        --output /app/data/processed/training_data.csv
    
    if [ $? -ne 0 ]; then
        echo ""
        echo "✗ Data fetch failed!"
        echo "Check the error messages above for details."
        exit 1
    fi
    
    # Verify output file was created
    if [ ! -f "/app/data/processed/training_data.csv" ]; then
        echo "ERROR: Training data file was not created at /app/data/processed/training_data.csv"
        exit 1
    fi
    
    echo "✓ Fresh data fetched and training data built"
    echo "  File: /app/data/processed/training_data.csv"
    echo "  Size: $(du -h /app/data/processed/training_data.csv | cut -f1)"
}

# Function to run backtest
run_backtest() {
    echo ""
    echo "============================================================"
    echo "STEP 2: RUNNING BACKTEST"
    echo "============================================================"
    echo "Markets: $MARKETS"
    echo "Min training games: $MIN_TRAINING"
    echo ""
    
    # Validate training data exists
    if [ ! -f "/app/data/processed/training_data.csv" ]; then
        echo "ERROR: Training data not found at /app/data/processed/training_data.csv"
        echo ""
        echo "Available files in /app/data/processed:"
        ls -lh /app/data/processed/ | head -20
        echo ""
        echo "Run with 'data' or 'full' command first to build training data."
        exit 1
    fi
    
    echo "Training data found:"
    echo "  File: /app/data/processed/training_data.csv"
    echo "  Size: $(du -h /app/data/processed/training_data.csv | cut -f1)"
    echo "  Lines: $(wc -l < /app/data/processed/training_data.csv)"
    echo ""
    
    # Run the backtest
    echo "Starting backtest..."
    python scripts/backtest.py \
        --markets "$MARKETS" \
        --min-training "$MIN_TRAINING" \
        --data "/app/data/processed/training_data.csv"
    
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo ""
        echo "✗ Backtest failed with exit code: $exit_code"
        echo "Check the error messages above for details."
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
    
    diagnose)
        echo ""
        echo "============================================================"
        echo "DIAGNOSTIC INFORMATION"
        echo "============================================================"
        echo ""
        echo "Environment:"
        echo "  Python: $(python --version)"
        echo "  PYTHONPATH: ${PYTHONPATH:-not set}"
        echo "  Working directory: $(pwd)"
        echo ""
        echo "Directory structure:"
        echo "  /app exists: $([ -d /app ] && echo 'yes' || echo 'NO')"
        echo "  /app/src exists: $([ -d /app/src ] && echo 'yes' || echo 'NO')"
        echo "  /app/scripts exists: $([ -d /app/scripts ] && echo 'yes' || echo 'NO')"
        echo "  /app/data exists: $([ -d /app/data ] && echo 'yes' || echo 'NO')"
        echo ""
        echo "Environment variables:"
        echo "  API_BASKETBALL_KEY: $([ -n "$API_BASKETBALL_KEY" ] && echo 'set' || echo 'NOT SET')"
        echo "  THE_ODDS_API_KEY: $([ -n "$THE_ODDS_API_KEY" ] && echo 'set' || echo 'NOT SET')"
        echo "  SEASONS: ${SEASONS:-not set}"
        echo "  MARKETS: ${MARKETS:-not set}"
        echo ""
        echo "Data files:"
        if [ -f "/app/data/processed/training_data.csv" ]; then
            echo "  training_data.csv: exists ($(du -h /app/data/processed/training_data.csv | cut -f1), $(wc -l < /app/data/processed/training_data.csv) lines)"
        else
            echo "  training_data.csv: NOT FOUND"
        fi
        echo ""
        echo "Python imports:"
        validate_python
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
        echo "  diagnose  - Show diagnostic information"
        exit 1
        ;;
esac


