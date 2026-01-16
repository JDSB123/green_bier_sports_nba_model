@echo off
REM Load environment variables from .env file (Windows batch version)

if not exist .env (
    echo ERROR: .env file not found
    echo Run: python scripts/setup_teams_webhook.py
    exit /b 1
)

echo Loading environment variables from .env...

for /f "tokens=1,2 delims==" %%a in (.env) do (
    REM Skip comments and empty lines
    echo %%a | findstr /r /c:"^#" >nul && (
        REM Skip comment
    ) || (
        if not "%%a"=="" (
            set "%%a=%%b"
            echo Set %%a
        )
    )
)

echo.
echo Environment variables loaded!
echo.
echo You can now run:
echo   python scripts/post_to_teams.py
echo.
