@echo off
REM Post NBA picks to Teams using existing API endpoint
REM Your Teams webhook is already configured in Azure

echo Posting today's picks to Teams...
curl -X POST "https://nba-gbsv-api.livelycoast-b48c3cb0.eastus.azurecontainerapps.io/teams/outgoing" -H "Content-Type: application/json" -d "{\"text\":\"picks\"}"
echo.
echo.
echo Done! Check your Teams channel for the picks.
