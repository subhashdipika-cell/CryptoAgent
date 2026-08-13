@echo off
setlocal EnableExtensions
title CryptoAgent Predictive Validation
cd /d "%~dp0" || exit /b 1

set "MT5_TERMINAL_PATH=D:\MT5IntelliTrade\terminal64.exe"
set "MT5_BTC_SYMBOL=BTCUSD"
set "MT5_XAU_SYMBOL=XAUUSD+"
set "REQUIRE_DEMO_ACCOUNT=true"
set "TRADING_ENABLED=false"
set "DRY_RUN=true"
set "PREDICTIVE_MODE=shadow"

echo Generating leakage-free BTC and Gold walk-forward validation...
".venv\Scripts\python.exe" predictive_validation.py --bars 3000
if errorlevel 1 (
    echo Validation failed.
    pause
    exit /b 1
)
start "" "reports\predictive_validation.html"
endlocal
