@echo off
setlocal EnableDelayedExpansion

title Relievia - Smart Bed Simulation Launcher

echo ======================================================================
echo           RELIEVIA: Autonomous Pressure Injury Prevention
echo                    One-Click System Launcher
echo ======================================================================
echo.

:: Step 0: Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python was not found in your system PATH!
    echo Please install Python 3.10+ from https://www.python.org/
    echo During installation, be sure to check "Add Python to PATH".
    echo.
    pause
    exit /b 1
)

:: Step 0.5: Verify Python dependencies
python -c "import flask, flask_cors, sklearn, joblib, requests" >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [INFO] Required Python packages missing. Installing dependencies...
    python -m pip install -r ml_backend\requirements.txt
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to install dependencies.
        echo Please run manually in PowerShell:
        echo   python -m pip install -r ml_backend\requirements.txt
        echo.
        pause
        exit /b 1
    )
)

:: Step 0.7: Verify trained model artifact exists
if not exist "ml_backend\model.pkl" (
    echo [INFO] Trained model artifact not found. Training model now...
    python ml_backend\train_model.py
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Model training failed.
        echo.
        pause
        exit /b 1
    )
)

:: Step 1: Start the Flask ML backend (if not already running)
powershell -NoProfile -Command "if (Get-NetTCPConnection -LocalPort 5000 -ErrorAction SilentlyContinue) { exit 10 } else { exit 0 }"
if %ERRORLEVEL% EQU 10 (
    echo [INFO] Relievia ML backend is already running on port 5000.
) else (
    echo [1/3] Starting Relievia ML Backend...
    start "Relievia ML Server" /B python ml_backend\app.py
)

:: Step 2: Poll /health until ML backend is fully ready
echo [2/3] Waiting for ML API to become ready at http://127.0.0.1:5000/health...
powershell -NoProfile -Command ^
  "$ready = $false; for ($i=0; $i -lt 30; $i++) { try { $r = Invoke-RestMethod -Uri 'http://127.0.0.1:5000/health' -TimeoutSec 1; if ($r.status -eq 'ok') { $ready = $true; break } } catch { Start-Sleep -Milliseconds 400 } }; if (-not $ready) { exit 1 }"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Timed out waiting for ML API backend to respond!
    echo Please verify that port 5000 is available and python can bind to it.
    echo.
    pause
    exit /b 1
)

echo [OK] ML Backend is online and responding!
echo.

:: Step 3: Launch simulation in default browser
echo [3/3] Opening Relievia...
start "" "relievia.html"

echo.
echo ======================================================================
echo    Relievia Simulation is ACTIVE!
echo    - ML Risk API:   http://127.0.0.1:5000/predict
echo    - Health Probe:  http://127.0.0.1:5000/health
echo    - Digital Twin:  Opened in default browser (relievia.html)
echo ======================================================================
echo.
echo Keep this terminal window open while using Relievia.
echo To shut down the ML backend:
echo   - Close this window, or
echo   - Double-click stop_relievia.bat
echo.

:: Maintain process loop while running
powershell -NoProfile -Command ^
  "$host.UI.RawUI.WindowTitle = 'Relievia ML Backend - ACTIVE (Port 5000)'; while ($true) { Start-Sleep -Seconds 2 }"
