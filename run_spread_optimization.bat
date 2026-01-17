@echo off
REM Spread Parameter Optimization Script
REM This script runs backtests for FG and 1H spreads with different configurations

echo ================================================================================
echo SPREAD PARAMETER OPTIMIZATION - INDEPENDENT ANALYSIS
echo ================================================================================
echo.

set OUTPUT_DIR=data\backtest_results\spread_optimization
set DATA=data\processed\training_data.csv
set MODELS=models\production

REM Create output directory
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo Starting optimization runs...
echo.

REM ============================================================================
REM FG SPREAD OPTIMIZATION
REM ============================================================================

echo.
echo [1/24] FG Spread: conf=0.55, juice=-110
python scripts\backtest_production.py --data=%DATA% --models-dir=%MODELS% --markets=fg_spread --spread-juice=-110 --total-juice=-110 --output-json=%OUTPUT_DIR%\fg_spread_conf55_j110.json > %OUTPUT_DIR%\fg_spread_conf55_j110.log 2>&1

echo [2/24] FG Spread: conf=0.60, juice=-110
python scripts\backtest_production.py --data=%DATA% --models-dir=%MODELS% --markets=fg_spread --spread-juice=-110 --total-juice=-110 --output-json=%OUTPUT_DIR%\fg_spread_conf60_j110.json > %OUTPUT_DIR%\fg_spread_conf60_j110.log 2>&1

echo [3/24] FG Spread: conf=0.62, juice=-110
python scripts\backtest_production.py --data=%DATA% --models-dir=%MODELS% --markets=fg_spread --spread-juice=-110 --total-juice=-110 --output-json=%OUTPUT_DIR%\fg_spread_conf62_j110.json > %OUTPUT_DIR%\fg_spread_conf62_j110.log 2>&1

echo [4/24] FG Spread: conf=0.65, juice=-110
python scripts\backtest_production.py --data=%DATA% --models-dir=%MODELS% --markets=fg_spread --spread-juice=-110 --total-juice=-110 --output-json=%OUTPUT_DIR%\fg_spread_conf65_j110.json > %OUTPUT_DIR%\fg_spread_conf65_j110.log 2>&1

echo [5/24] FG Spread: conf=0.68, juice=-110
python scripts\backtest_production.py --data=%DATA% --models-dir=%MODELS% --markets=fg_spread --spread-juice=-110 --total-juice=-110 --output-json=%OUTPUT_DIR%\fg_spread_conf68_j110.json > %OUTPUT_DIR%\fg_spread_conf68_j110.log 2>&1

echo [6/24] FG Spread: conf=0.70, juice=-110
python scripts\backtest_production.py --data=%DATA% --models-dir=%MODELS% --markets=fg_spread --spread-juice=-110 --total-juice=-110 --output-json=%OUTPUT_DIR%\fg_spread_conf70_j110.json > %OUTPUT_DIR%\fg_spread_conf70_j110.log 2>&1

echo [7/24] FG Spread: conf=0.55, juice=-105
python scripts\backtest_production.py --data=%DATA% --models-dir=%MODELS% --markets=fg_spread --spread-juice=-105 --total-juice=-110 --output-json=%OUTPUT_DIR%\fg_spread_conf55_j105.json > %OUTPUT_DIR%\fg_spread_conf55_j105.log 2>&1

echo [8/24] FG Spread: conf=0.60, juice=-105
python scripts\backtest_production.py --data=%DATA% --models-dir=%MODELS% --markets=fg_spread --spread-juice=-105 --total-juice=-110 --output-json=%OUTPUT_DIR%\fg_spread_conf60_j105.json > %OUTPUT_DIR%\fg_spread_conf60_j105.log 2>&1

echo [9/24] FG Spread: conf=0.62, juice=-105
python scripts\backtest_production.py --data=%DATA% --models-dir=%MODELS% --markets=fg_spread --spread-juice=-105 --total-juice=-110 --output-json=%OUTPUT_DIR%\fg_spread_conf62_j105.json > %OUTPUT_DIR%\fg_spread_conf62_j105.log 2>&1

echo [10/24] FG Spread: conf=0.65, juice=-105
python scripts\backtest_production.py --data=%DATA% --models-dir=%MODELS% --markets=fg_spread --spread-juice=-105 --total-juice=-110 --output-json=%OUTPUT_DIR%\fg_spread_conf65_j105.json > %OUTPUT_DIR%\fg_spread_conf65_j105.log 2>&1

echo [11/24] FG Spread: conf=0.68, juice=-105
python scripts\backtest_production.py --data=%DATA% --models-dir=%MODELS% --markets=fg_spread --spread-juice=-105 --total-juice=-110 --output-json=%OUTPUT_DIR%\fg_spread_conf68_j105.json > %OUTPUT_DIR%\fg_spread_conf68_j105.log 2>&1

echo [12/24] FG Spread: conf=0.70, juice=-105
python scripts\backtest_production.py --data=%DATA% --models-dir=%MODELS% --markets=fg_spread --spread-juice=-105 --total-juice=-110 --output-json=%OUTPUT_DIR%\fg_spread_conf70_j105.json > %OUTPUT_DIR%\fg_spread_conf70_j105.log 2>&1

REM ============================================================================
REM 1H SPREAD OPTIMIZATION
REM ============================================================================

echo.
echo [13/24] 1H Spread: conf=0.55, juice=-110
python scripts\backtest_production.py --data=%DATA% --models-dir=%MODELS% --markets=1h_spread --spread-juice=-110 --total-juice=-110 --output-json=%OUTPUT_DIR%\1h_spread_conf55_j110.json > %OUTPUT_DIR%\1h_spread_conf55_j110.log 2>&1

echo [14/24] 1H Spread: conf=0.60, juice=-110
python scripts\backtest_production.py --data=%DATA% --models-dir=%MODELS% --markets=1h_spread --spread-juice=-110 --total-juice=-110 --output-json=%OUTPUT_DIR%\1h_spread_conf60_j110.json > %OUTPUT_DIR%\1h_spread_conf60_j110.log 2>&1

echo [15/24] 1H Spread: conf=0.62, juice=-110
python scripts\backtest_production.py --data=%DATA% --models-dir=%MODELS% --markets=1h_spread --spread-juice=-110 --total-juice=-110 --output-json=%OUTPUT_DIR%\1h_spread_conf62_j110.json > %OUTPUT_DIR%\1h_spread_conf62_j110.log 2>&1

echo [16/24] 1H Spread: conf=0.65, juice=-110
python scripts\backtest_production.py --data=%DATA% --models-dir=%MODELS% --markets=1h_spread --spread-juice=-110 --total-juice=-110 --output-json=%OUTPUT_DIR%\1h_spread_conf65_j110.json > %OUTPUT_DIR%\1h_spread_conf65_j110.log 2>&1

echo [17/24] 1H Spread: conf=0.68, juice=-110
python scripts\backtest_production.py --data=%DATA% --models-dir=%MODELS% --markets=1h_spread --spread-juice=-110 --total-juice=-110 --output-json=%OUTPUT_DIR%\1h_spread_conf68_j110.json > %OUTPUT_DIR%\1h_spread_conf68_j110.log 2>&1

echo [18/24] 1H Spread: conf=0.70, juice=-110
python scripts\backtest_production.py --data=%DATA% --models-dir=%MODELS% --markets=1h_spread --spread-juice=-110 --total-juice=-110 --output-json=%OUTPUT_DIR%\1h_spread_conf70_j110.json > %OUTPUT_DIR%\1h_spread_conf70_j110.log 2>&1

echo [19/24] 1H Spread: conf=0.55, juice=-105
python scripts\backtest_production.py --data=%DATA% --models-dir=%MODELS% --markets=1h_spread --spread-juice=-105 --total-juice=-110 --output-json=%OUTPUT_DIR%\1h_spread_conf55_j105.json > %OUTPUT_DIR%\1h_spread_conf55_j105.log 2>&1

echo [20/24] 1H Spread: conf=0.60, juice=-105
python scripts\backtest_production.py --data=%DATA% --models-dir=%MODELS% --markets=1h_spread --spread-juice=-105 --total-juice=-110 --output-json=%OUTPUT_DIR%\1h_spread_conf60_j105.json > %OUTPUT_DIR%\1h_spread_conf60_j105.log 2>&1

echo [21/24] 1H Spread: conf=0.62, juice=-105
python scripts\backtest_production.py --data=%DATA% --models-dir=%MODELS% --markets=1h_spread --spread-juice=-105 --total-juice=-110 --output-json=%OUTPUT_DIR%\1h_spread_conf62_j105.json > %OUTPUT_DIR%\1h_spread_conf62_j105.log 2>&1

echo [22/24] 1H Spread: conf=0.65, juice=-105
python scripts\backtest_production.py --data=%DATA% --models-dir=%MODELS% --markets=1h_spread --spread-juice=-105 --total-juice=-110 --output-json=%OUTPUT_DIR%\1h_spread_conf65_j105.json > %OUTPUT_DIR%\1h_spread_conf65_j105.log 2>&1

echo [23/24] 1H Spread: conf=0.68, juice=-105
python scripts\backtest_production.py --data=%DATA% --models-dir=%MODELS% --markets=1h_spread --spread-juice=-105 --total-juice=-110 --output-json=%OUTPUT_DIR%\1h_spread_conf68_j105.json > %OUTPUT_DIR%\1h_spread_conf68_j105.log 2>&1

echo [24/24] 1H Spread: conf=0.70, juice=-105
python scripts\backtest_production.py --data=%DATA% --models-dir=%MODELS% --markets=1h_spread --spread-juice=-105 --total-juice=-110 --output-json=%OUTPUT_DIR%\1h_spread_conf70_j105.json > %OUTPUT_DIR%\1h_spread_conf70_j105.log 2>&1

echo.
echo ================================================================================
echo OPTIMIZATION COMPLETE
echo ================================================================================
echo Results saved to: %OUTPUT_DIR%
echo.
echo Run the analysis script to view results:
echo python scripts\analyze_spread_optimization.py
echo.
