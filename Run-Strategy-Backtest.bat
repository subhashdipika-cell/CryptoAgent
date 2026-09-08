@echo off
setlocal EnableExtensions
title CryptoAgent Strategy Backtest
cd /d "%~dp0" || exit /b 1

set "MT5_TERMINAL_PATH=D:\MT5IntelliTrade\terminal64.exe"
set "MT5_BTC_SYMBOL=BTCUSD"
set "MT5_XAU_SYMBOL=XAUUSD+"
set "REQUIRE_DEMO_ACCOUNT=true"
set "TRADING_ENABLED=false"
set "DRY_RUN=true"
set "PREDICTIVE_MODE=calibrated"
set "MAX_RISK_FRACTION=0.02"

echo Running out-of-sample CryptoAgent backtest...
".venv\Scripts\python.exe" strategy_backtest.py --bars 3000 --commission-per-side 0.03 --slippage-points 10
if errorlevel 1 (
    echo Backtest failed.
    pause
    exit /b 1
)
start "" "reports\strategy_backtest.html"
endlocal
