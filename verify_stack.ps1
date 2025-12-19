# NBA v5.0 Stack Verification Script
# Verifies that the entire stack flow works correctly

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "NBA v5.0 STACK VERIFICATION" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$errors = 0

# Test 1: Container Entry Point
Write-Host "[1/6] Testing container entry point..." -ForegroundColor Yellow
try {
    $result = docker compose -f docker-compose.backtest.yml run --rm backtest-shell python -c "from src.modeling.models import SpreadsModel, TotalsModel; print('OK')" 2>&1
    if ($result -match "OK") {
        Write-Host "  ✓ Container entry point OK" -ForegroundColor Green
    } else {
        Write-Host "  ✗ Container entry point FAILED" -ForegroundColor Red
        Write-Host "  Output: $result" -ForegroundColor Red
        $errors++
    }
} catch {
    Write-Host "  ✗ Container entry point FAILED: $_" -ForegroundColor Red
    $errors++
}

# Test 2: Data Pipeline (quick check - just validate it can start)
Write-Host "[2/6] Testing data pipeline imports..." -ForegroundColor Yellow
try {
    $result = docker compose -f docker-compose.backtest.yml run --rm backtest-shell python -c "from scripts.build_fresh_training_data import FreshDataPipeline; print('OK')" 2>&1
    if ($result -match "OK") {
        Write-Host "  ✓ Data pipeline imports OK" -ForegroundColor Green
    } else {
        Write-Host "  ✗ Data pipeline imports FAILED" -ForegroundColor Red
        $errors++
    }
} catch {
    Write-Host "  ✗ Data pipeline imports FAILED: $_" -ForegroundColor Red
    $errors++
}

# Test 3: Feature Engineering (check 1H features exist)
Write-Host "[3/6] Testing feature engineering (1H features)..." -ForegroundColor Yellow
try {
    $result = docker compose -f docker-compose.backtest.yml run --rm backtest-shell python -c @"
import pandas as pd
from datetime import datetime
from src.modeling.features import FeatureEngineer

fe = FeatureEngineer()
game = pd.Series({
    'home_team': 'LAL',
    'away_team': 'BOS',
    'date': pd.Timestamp('2025-12-18'),
    'spread_line': -5.0,
    'total_line': 220.0
})
historical = pd.DataFrame([
    {'date': pd.Timestamp('2025-12-15'), 'home_team': 'LAL', 'away_team': 'MIA', 'home_score': 110, 'away_score': 105},
])

features = fe.build_game_features(game, historical)
if 'predicted_margin_1h' in features and 'predicted_total_1h' in features:
    print('OK')
else:
    print('MISSING_1H_FEATURES')
"@ 2>&1
    
    if ($result -match "OK") {
        Write-Host "  ✓ Feature engineering OK (1H features present)" -ForegroundColor Green
    } elseif ($result -match "MISSING_1H_FEATURES") {
        Write-Host "  ✗ Feature engineering FAILED - Missing 1H features!" -ForegroundColor Red
        $errors++
    } else {
        Write-Host "  ✗ Feature engineering FAILED" -ForegroundColor Red
        Write-Host "  Output: $result" -ForegroundColor Red
        $errors++
    }
} catch {
    Write-Host "  ✗ Feature engineering FAILED: $_" -ForegroundColor Red
    $errors++
}

# Test 4: Prediction Logic (check spread/total predictors work)
Write-Host "[4/6] Testing prediction logic..." -ForegroundColor Yellow
try {
    $result = docker compose -f docker-compose.backtest.yml run --rm backtest-shell python -c @"
from src.prediction.spreads.predictor import SpreadPredictor
from src.prediction.totals.predictor import TotalPredictor
print('OK')
"@ 2>&1
    
    if ($result -match "OK") {
        Write-Host "  ✓ Prediction logic imports OK" -ForegroundColor Green
    } else {
        Write-Host "  ✗ Prediction logic imports FAILED" -ForegroundColor Red
        $errors++
    }
} catch {
    Write-Host "  ✗ Prediction logic imports FAILED: $_" -ForegroundColor Red
    $errors++
}

# Test 5: API Endpoint Functions Exist
Write-Host "[5/6] Testing API endpoint functions..." -ForegroundColor Yellow
try {
    $result = docker compose -f docker-compose.backtest.yml run --rm backtest-shell python -c @"
from src.ingestion.the_odds import fetch_participants, fetch_betting_splits, fetch_event_odds
from src.ingestion.api_basketball import APIBasketballClient

client = APIBasketballClient()
if hasattr(client, 'ingest_essential'):
    print('OK')
else:
    print('MISSING_INGEST_ESSENTIAL')
"@ 2>&1
    
    if ($result -match "OK") {
        Write-Host "  ✓ API endpoint functions OK" -ForegroundColor Green
    } elseif ($result -match "MISSING_INGEST_ESSENTIAL") {
        Write-Host "  ✗ Missing ingest_essential() function!" -ForegroundColor Red
        $errors++
    } else {
        Write-Host "  ✗ API endpoint functions FAILED" -ForegroundColor Red
        $errors++
    }
} catch {
    Write-Host "  ✗ API endpoint functions FAILED: $_" -ForegroundColor Red
    $errors++
}

# Test 6: Backtest Script Imports
Write-Host "[6/6] Testing backtest script..." -ForegroundColor Yellow
try {
    $result = docker compose -f docker-compose.backtest.yml run --rm backtest-shell python -c "import scripts.backtest; print('OK')" 2>&1
    if ($result -match "OK") {
        Write-Host "  ✓ Backtest script OK" -ForegroundColor Green
    } else {
        Write-Host "  ✗ Backtest script FAILED" -ForegroundColor Red
        $errors++
    }
} catch {
    Write-Host "  ✗ Backtest script FAILED: $_" -ForegroundColor Red
    $errors++
}

# Summary
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
if ($errors -eq 0) {
    Write-Host "✅ ALL CHECKS PASSED" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "  1. Run full pipeline: docker compose -f docker-compose.backtest.yml up backtest-full" -ForegroundColor White
    Write-Host "  2. Check results: cat data/results/backtest_report_*.md" -ForegroundColor White
} else {
    Write-Host "❌ $errors CHECK(S) FAILED" -ForegroundColor Red
    Write-Host ""
    Write-Host "Review the errors above and fix before proceeding." -ForegroundColor Yellow
}
Write-Host "============================================================" -ForegroundColor Cyan

exit $errors
