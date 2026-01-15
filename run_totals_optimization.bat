@echo off
REM Totals Market Optimization Script
REM This script runs comprehensive backtesting and parameter optimization for 1H and FG totals markets

echo ================================================================================
echo TOTALS MARKET OPTIMIZATION
echo ================================================================================
echo.

cd /d "%~dp0"

echo Running totals optimization...
python scripts\optimize_totals_only.py

echo.
echo ================================================================================
echo OPTIMIZATION COMPLETE
echo ================================================================================
echo Check data/backtest_results/ for the results JSON file
echo.

pause
