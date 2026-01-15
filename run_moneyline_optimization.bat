@echo off
REM Batch script to run moneyline optimization for FG and 1H markets independently

echo ======================================================================
echo NBA Moneyline Optimization - Independent FG and 1H Markets
echo ======================================================================
echo.

REM Create output directory if it doesn't exist
if not exist "data\backtest_results" mkdir "data\backtest_results"

echo Running optimization for BOTH markets...
echo.

REM Run optimization for both FG and 1H
python scripts/train_moneyline_models.py --market all --test-cutoff 2025-01-01

echo.
echo ======================================================================
echo Optimization Complete!
echo ======================================================================
echo.
echo Results saved to:
echo   - data/backtest_results/fg_moneyline_optimization_results.json
echo   - data/backtest_results/1h_moneyline_optimization_results.json
echo.
echo Press any key to exit...
pause > nul
