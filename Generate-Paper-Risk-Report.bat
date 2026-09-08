@echo off
setlocal EnableExtensions
title CryptoAgent XAU Paper Risk Report
cd /d "%~dp0" || exit /b 1

set "MT5_TERMINAL_PATH=D:\MT5IntelliTrade\terminal64.exe"
set "MT5_BTC_SYMBOL=BTCUSD"
set "MT5_XAU_SYMBOL=XAUUSD+"
set "REQUIRE_DEMO_ACCOUNT=true"
set "TRADING_ENABLED=false"
set "DRY_RUN=true"

echo Generating PAPER_ONLY XAUUSD 0.01-lot report at a fixed 1%% reference cap...
".venv\Scripts\python.exe" paper_risk_report.py
if errorlevel 1 (
    echo Report failed.
    pause
    exit /b 1
)
start "" "reports\xau_minimum_equity_risk_report.html"
endlocal
