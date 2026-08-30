@echo off
setlocal EnableExtensions
title Close CryptoAgent
cd /d "%~dp0"

powershell.exe -NoLogo -NoProfile -Command "$process=Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {$_.ExecutablePath -like 'D:\Projects\CryptoAgent\.venv\*' -and $_.CommandLine -match '[\\/]main[.]py'} | Select-Object -First 1; if($process){exit 0}; exit 2" >nul 2>&1
if errorlevel 2 (
    echo CryptoAgent is not running.
    exit /b 0
)

rem Fail closed: a restart must not disable management for CryptoAgent-owned MT5 positions.
".venv\Scripts\python.exe" -c "import sys, MetaTrader5 as mt5; ok=mt5.initialize(path=r'D:\MT5IntelliTrade\terminal64.exe'); positions=mt5.positions_get() if ok else None; code=12 if not ok or positions is None else (10 if any(int(getattr(p,'magic',0))==26081301 for p in positions) else 0); mt5.shutdown() if ok else None; sys.exit(code)" >nul 2>&1
if errorlevel 12 (
    echo [ERROR] MT5 position safety check failed. CryptoAgent was not stopped.
    exit /b 1
)
if errorlevel 10 (
    echo [BLOCKED] CryptoAgent-owned MT5 positions are open. Restart is unsafe.
    exit /b 1
)

powershell.exe -NoLogo -NoProfile -Command ^
  "$targets=Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {$_.ExecutablePath -like 'D:\Projects\CryptoAgent\.venv\*' -and $_.CommandLine -match '[\\/]main[.]py'};" ^
  "if(-not $targets){Write-Host 'CryptoAgent is not running.'; exit 0};" ^
  "foreach($process in $targets){Write-Host \"Stopping CryptoAgent PID $($process.ProcessId)...\"; & taskkill.exe /PID $process.ProcessId /T /F | Out-Null; if($LASTEXITCODE -ne 0){exit $LASTEXITCODE}}; exit 0"

set "CLOSE_EXIT=%ERRORLEVEL%"
if not "%CLOSE_EXIT%"=="0" echo [ERROR] CryptoAgent could not be stopped cleanly.
if /i not "%TRADING_LAB_HIDDEN%"=="1" timeout /t 2 /nobreak >nul
endlocal & exit /b %CLOSE_EXIT%
