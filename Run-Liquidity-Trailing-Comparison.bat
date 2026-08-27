@echo off
setlocal EnableExtensions
title CryptoAgent Liquidity Trailing Comparison
cd /d "%~dp0" || exit /b 1

set "MT5_TERMINAL_PATH=D:\MT5IntelliTrade\terminal64.exe"
set "MT5_BTC_SYMBOL=BTCUSD"
set "MT5_XAU_SYMBOL=XAUUSD+"
set "REQUIRE_DEMO_ACCOUNT=true"
set "TRADING_ENABLED=false"
set "DRY_RUN=true"
set "STRATEGY_MODE=liquidity_breakout"
set "TRADING_STRATEGY=LiquidityBreakout"
set "MAX_RISK_FRACTION=0.02"

echo Comparing liquidity trailing rules over paginated MT5 history...
".venv\Scripts\python.exe" liquidity_trailing_comparison.py --m3-bars 99999 --starting-equity 1000 --commission-per-lot-side 0.03 --slippage-points 10
if errorlevel 1 (
    echo Trailing comparison failed.
    pause
    exit /b 1
)
start "" "reports\liquidity_trailing_comparison.html"
endlocal
