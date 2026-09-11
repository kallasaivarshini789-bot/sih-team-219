@echo off
echo ======================================================================
echo   Satellite Cloud Removal ^& Analysis System - Setup Script (SIH 2026)
echo ======================================================================
echo.

echo [1/3] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found on PATH. Please install Python 3.10+ and re-run.
    pause
    exit /b 1
)

echo [2/3] Installing required Python libraries...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

echo [3/3] Pre-generating satellite sample scenes...
python sample_generator.py

echo.
echo ======================================================================
echo   Setup Completed Successfully! Run 'run.bat' to launch the app.
echo ======================================================================
echo.
pause
