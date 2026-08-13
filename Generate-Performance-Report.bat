@echo off
setlocal EnableExtensions
title CryptoAgent Performance Report
cd /d "%~dp0" || exit /b 1

set "MT5_TERMINAL_PATH=D:\MT5IntelliTrade\terminal64.exe"
set "MT5_BTC_SYMBOL=BTCUSD"
set "MT5_XAU_SYMBOL=XAUUSD+"
set "REQUIRE_DEMO_ACCOUNT=true"
set "TRADING_ENABLED=false"
set "DRY_RUN=true"

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Python environment not found.
    exit /b 1
)

echo Reconciling MT5 history and generating reports...
".venv\Scripts\python.exe" performance_report.py --sync
if errorlevel 1 (
    echo Report generation failed.
    pause
    exit /b 1
)

echo.
echo Report created at reports\performance_report.html
start "" "reports\performance_report.html"
endlocal
