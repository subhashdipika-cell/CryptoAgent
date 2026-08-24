@echo off
setlocal EnableExtensions
title CryptoAgent - Liquidity Breakout - MT5 DEMO

cd /d "%~dp0" || (
    echo ERROR: Unable to open the CryptoAgent project directory.
    exit /b 1
)

rem Opt-in DEMO-only H4/M15/M3 institutional-liquidity strategy.
set "MT5_TERMINAL_PATH=D:\MT5IntelliTrade\terminal64.exe"
set "MT5_BTC_SYMBOL=BTCUSD"
set "MT5_XAU_SYMBOL=XAUUSD+"
set "MT5_MAGIC=26081301"
set "MT5_APP_NAME=CryptoAgent"
set "TRADING_STRATEGY=LiquidityBreakout"
set "STRATEGY_MODE=liquidity_breakout"
set "PREDICTIVE_MODE=calibrated"

set "REQUIRE_DEMO_ACCOUNT=true"
set "TRADING_ENABLED=true"
set "DRY_RUN=false"
set "MAX_RISK_FRACTION=0.02"
set "AUTOMATIC_REVALIDATION=false"

set "LIQUIDITY_MIN_RRR=2.5"
set "LIQUIDITY_MIN_TOUCHES=3"
set "LIQUIDITY_VOLUME_EXPANSION=1.2"
set "LIQUIDITY_MOMENTUM_BODY_FRACTION=0.60"
set "LIQUIDITY_MAX_TRADES_PER_DAY=3"
set "LIQUIDITY_DAILY_ACTIVE_CAPITAL=1000"
set "LIQUIDITY_DAILY_TARGET_FRACTION=0.25"
set "LIQUIDITY_DAILY_TIMEZONE=Asia/Kolkata"

if not exist "%MT5_TERMINAL_PATH%" (
    echo ERROR: MT5 terminal not found: %MT5_TERMINAL_PATH%
    exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Python environment not found.
    exit /b 1
)

if /I "%~1"=="--check" (
    echo CryptoAgent liquidity-breakout startup check passed.
    echo Terminal:   %MT5_TERMINAL_PATH%
    echo Symbols:    %MT5_BTC_SYMBOL%, %MT5_XAU_SYMBOL%
    echo Mode:       DEMO only - order routing ENABLED
    echo Strategy:   H4 liquidity target, M15 structure, M3 volume breakout
    echo Entry gate: Minimum 1:2.5 RRR, 3 H4 touches, volume expansion
    echo Risk:       Maximum 2%% of account equity per trade
    echo Daily lock: Maximum 3 entries or 25%% of up to $1,000 active capital
    echo Management: Fixed SL/TP; breakeven at 2R
    exit /b 0
)

echo Starting opt-in Liquidity Breakout strategy on MT5 DEMO...
echo The agent will stop if the connected account is not DEMO.
echo Close this window or press Ctrl+C to stop the agent cleanly.
echo.

".venv\Scripts\python.exe" main.py
set "CRYPTO_AGENT_EXIT=%ERRORLEVEL%"

if not "%CRYPTO_AGENT_EXIT%"=="0" (
    echo.
    echo CryptoAgent stopped with exit code %CRYPTO_AGENT_EXIT%.
    pause
)

endlocal & exit /b %CRYPTO_AGENT_EXIT%
