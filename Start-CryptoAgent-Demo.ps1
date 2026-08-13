param(
    [string]$TerminalPath = "D:\MT5IntelliTrade\terminal64.exe",
    [string]$BitcoinSymbol = "BTCUSD",
    [string]$GoldSymbol = "XAUUSD+"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $TerminalPath)) {
    throw "MT5 terminal not found: $TerminalPath"
}
if (-not (Test-Path -LiteralPath $python)) {
    throw "Project environment not found. Run setup_trading_env.py first."
}

$env:MT5_TERMINAL_PATH = $TerminalPath
$env:MT5_BTC_SYMBOL = $BitcoinSymbol
$env:MT5_XAU_SYMBOL = $GoldSymbol
$env:REQUIRE_DEMO_ACCOUNT = "true"
$env:TRADING_ENABLED = "false"
$env:DRY_RUN = "true"
$env:PREDICTIVE_MODE = "calibrated"

& $python (Join-Path $projectRoot "main.py")
