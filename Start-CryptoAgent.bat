@echo off
setlocal EnableExtensions
title CryptoAgent - MT5 DEMO Order Routing

cd /d "%~dp0" || (
    echo ERROR: Unable to open the CryptoAgent project directory.
    exit /b 1
)

rem Verified Vantage DEMO terminal and broker symbol aliases.
set "MT5_TERMINAL_PATH=D:\MT5IntelliTrade\terminal64.exe"
set "MT5_BTC_SYMBOL=BTCUSD"
set "MT5_XAU_SYMBOL=XAUUSD+"
set "MT5_MAGIC=26081301"
set "MT5_APP_NAME=CryptoAgent"
set "TRADING_STRATEGY=AssetCalibrated"
set "PREDICTIVE_MODE=calibrated"

rem Order routing is enabled, but the runtime must verify a DEMO account.
set "REQUIRE_DEMO_ACCOUNT=true"
set "TRADING_ENABLED=true"
set "DRY_RUN=false"
set "MAX_RISK_FRACTION=0.02"
set "AUTOMATIC_REVALIDATION=true"
set "REVALIDATION_NEW_M15_BARS=500"

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
    echo Expert ID: %MT5_MAGIC% ^(CryptoAgent^)
    echo Comment:   %MT5_APP_NAME%^|%TRADING_STRATEGY%
    echo Mode:     DEMO only - order routing ENABLED
    echo Predictor: Dedicated BTC/XAU models with holdout-calibrated policy
    echo Risk:     Maximum 2%% of equity per trade
    echo Revalidation: Candidate-only after 500 new completed M15 bars
    echo Promotion:    Manual approval and restart required
    exit /b 0
)

echo Starting CryptoAgent with MT5 DEMO order routing enabled...
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
