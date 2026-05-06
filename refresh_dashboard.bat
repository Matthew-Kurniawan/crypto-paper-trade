@echo off
REM ============================================================
REM  Pulls the latest from GitHub and opens the dashboard.
REM  Pin a SHORTCUT to this file on your desktop or taskbar.
REM  Double-click -> done.
REM ============================================================

REM Move to the directory this script lives in (the project root).
cd /d "%~dp0"

echo.
echo Pulling latest from GitHub...
echo ============================================================
git pull --ff-only
if errorlevel 1 (
    echo.
    echo ============================================================
    echo  Pull failed. Either:
    echo    - You have local changes git wants to merge
    echo    - The remote moved in a way fast-forward can't handle
    echo    - You don't have internet
    echo  Dashboard may be stale. Check above for the error.
    echo ============================================================
    pause
    exit /b 1
)

echo.
echo Opening dashboard...
start "" "data\processed\live\dashboard.html"
echo Done. (You can close this window.)
timeout /t 3 >nul
