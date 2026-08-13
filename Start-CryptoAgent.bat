@echo off
setlocal EnableExtensions
title CryptoAgent - MT5 DEMO Dry Run

cd /d "%~dp0" || (
    echo ERROR: Unable to open the CryptoAgent project directory.
    exit /b 1
)

rem Verified Vantage DEMO terminal and broker symbol aliases.
set "MT5_TERMINAL_PATH=D:\MT5IntelliTrade\terminal64.exe"
set "MT5_BTC_SYMBOL=BTCUSD"
set "MT5_XAU_SYMBOL=XAUUSD+"

rem Fail-closed trading controls. This launcher cannot route real orders.
set "REQUIRE_DEMO_ACCOUNT=true"
set "TRADING_ENABLED=false"
set "DRY_RUN=true"

if not exist "%MT5_TERMINAL_PATH%" (
    echo ERROR: MT5 terminal not found: %MT5_TERMINAL_PATH%
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Python environment not found.
    echo Run setup_trading_env.py before starting CryptoAgent.
    exit /b 1
)

if not exist "models\chronos-2-base\config.json" (
    echo ERROR: Offline Chronos-2 model is not staged.
    echo Run: .venv\Scripts\python.exe stage_chronos_model.py
    exit /b 1
)

if /I "%~1"=="--check" (
    echo CryptoAgent startup check passed.
    echo Terminal: %MT5_TERMINAL_PATH%
    echo Symbols:  %MT5_BTC_SYMBOL%, %MT5_XAU_SYMBOL%
    echo Mode:     DEMO dry-run - order routing disabled
    exit /b 0
)

echo Starting CryptoAgent in MT5 DEMO dry-run mode...
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
