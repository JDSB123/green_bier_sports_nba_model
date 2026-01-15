@echo off
REM Alternative Totals Market Optimization using existing optimize_confidence_thresholds.py
REM This runs FG and 1H totals optimization separately

echo ================================================================================
echo TOTALS MARKET OPTIMIZATION (Alternative Method)
echo ================================================================================
echo.

cd /d "%~dp0"

REM Create output directory if it doesn't exist
if not exist "data\backtest_results" mkdir "data\backtest_results"

echo [1/2] Optimizing FG (Full Game) Totals...
echo ----------------------------------------
python scripts\optimize_confidence_thresholds.py --markets fg_total --spread-juice -110 --total-juice -110 --confidence-min 0.55 --confidence-max 0.79 --confidence-step 0.02 --edge-min 0.0 --edge-max 6.0 --edge-step 0.5 --min-bets 30 --objective roi --top 10 --output-json data\backtest_results\fg_total_optimization.json

echo.
echo.
echo [2/2] Optimizing 1H (First Half) Totals...
echo ----------------------------------------
python scripts\optimize_confidence_thresholds.py --markets 1h_total --spread-juice -110 --total-juice -110 --confidence-min 0.55 --confidence-max 0.79 --confidence-step 0.02 --edge-min 0.0 --edge-max 6.0 --edge-step 0.5 --min-bets 30 --objective roi --top 10 --output-json data\backtest_results\1h_total_optimization.json

echo.
echo ================================================================================
echo OPTIMIZATION COMPLETE
echo ================================================================================
echo.
echo Results saved to:
echo   - data\backtest_results\fg_total_optimization.json
echo   - data\backtest_results\1h_total_optimization.json
echo.

pause
