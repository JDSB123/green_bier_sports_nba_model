@echo off
REM Quick Teams Webhook Setup
REM This creates a .env file with production settings
REM You just need to add your TEAMS_WEBHOOK_URL

echo ================================================================================
echo TEAMS WEBHOOK QUICK SETUP
echo ================================================================================
echo.
echo This will create a .env file with production API settings.
echo You'll need to add your Teams webhook URL to the file.
echo.

if exist .env (
    echo WARNING: .env file already exists!
    echo.
    set /p overwrite="Overwrite existing .env? (y/n): "
    if /i not "%overwrite%"=="y" (
        echo Setup cancelled.
        exit /b 0
    )
)

echo Creating .env file...
echo.

(
echo # NBA Model Environment Variables
echo # Created by setup_teams_quick.bat
echo.
echo # REQUIRED: Add your Teams webhook URL here
echo # Get from: Teams Channel ^> ... ^> Connectors ^> Incoming Webhook
echo TEAMS_WEBHOOK_URL=PASTE_YOUR_WEBHOOK_URL_HERE
echo.
echo # Production API URL ^(configured^)
echo NBA_API_URL=https://nba-gbsv-api.livelycoast-b48c3cb0.eastus.azurecontainerapps.io
echo.
echo # Model version
echo NBA_MODEL_VERSION=NBA_v33.0.21.0
echo.
echo # Local API port ^(for testing^)
echo NBA_API_PORT=8090
echo.
echo # Model pack path
echo NBA_MODEL_PACK_PATH=models/production/model_pack.json
) > .env

echo ================================================================================
echo .env file created successfully!
echo ================================================================================
echo.
echo NEXT STEPS:
echo.
echo 1. Get your Teams webhook URL:
echo    - Open Teams channel
echo    - Click "..." menu ^> Connectors
echo    - Find "Incoming Webhook" ^> Configure
echo    - Name it "NBA Picks Bot"
echo    - Copy the webhook URL
echo.
echo 2. Edit .env file and replace PASTE_YOUR_WEBHOOK_URL_HERE with your actual URL
echo    - Open: .env
echo    - Find: TEAMS_WEBHOOK_URL=PASTE_YOUR_WEBHOOK_URL_HERE
echo    - Replace with: TEAMS_WEBHOOK_URL=https://outlook.office.com/webhook/...
echo.
echo 3. Test the setup:
echo    load_env.bat
echo    python scripts/post_to_teams.py
echo.
echo ================================================================================

start notepad .env
