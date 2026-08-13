# Hybrid Chronos + FinBERT MT5 Agent

An asynchronous Windows trading service for BTCUSD and XAUUSD. Chronos-2 runs locally on CPU from pre-staged files; news and FinBERT sentiment are isolated behind an asynchronous, neutral-on-failure adapter. MT5 is the only execution interface.

## Safety defaults

- Order routing is off (`TRADING_ENABLED=false`) and dry-run logging is on.
- A DEMO account is required by default.
- Every order plan includes its initial SL and TP.
- Risk is capped at 1% of current equity and rounded down to broker lot steps.
- A separate margin policy limits each order to 25% of free margin.
- One managed position per symbol prevents repeated entries each loop.
- Cloud/API failures return neutral sentiment (`0.5`) and never stop position management.

These controls prevent accidental routing; they do not establish that the strategy is profitable. Forward-test on DEMO with broker-specific spreads, slippage, symbol names, and contract sizes before considering any change in account mode.

## Installation

Use 64-bit Python on Windows, ideally the same interpreter architecture as MT5:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For CPU-only PyTorch, install the matching wheel from PyTorch's CPU index before installing the remaining requirements if needed.

Amazon publishes the 120M-parameter model as `amazon/chronos-2`; there is no official `amazon/chronos-2-base` model ID. The local directory retains the requested name. Stage it once while online:

```powershell
.\.venv\Scripts\python.exe stage_chronos_model.py
```

After staging, the runtime sets both `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, and passes `local_files_only=True`. Missing weights cause startup to fail before MT5 connects.

Copy `.env.example` values into process environment variables (the program intentionally does not parse or commit `.env`). At minimum, configure the MT5 terminal path and credentials. `HF_API_KEY` and `CRYPTOPANIC_API_KEY` are optional; without them sentiment safely remains neutral.

## Run and validate

Keep MT5 open, authenticated to DEMO, and Algo Trading enabled. Start in dry-run mode:

```powershell
$env:REQUIRE_DEMO_ACCOUNT = "true"
$env:TRADING_ENABLED = "false"
$env:DRY_RUN = "true"
.\.venv\Scripts\python.exe main.py
```

This workstation's verified Vantage DEMO terminal uses `BTCUSD` and the broker-suffixed
`XAUUSD+`. The convenience launcher pins that terminal and forces the safety flags:

```powershell
.\Start-CryptoAgent-Demo.ps1
```

For another broker, pass `-TerminalPath`, `-BitcoinSymbol`, and `-GoldSymbol`, or set
`MT5_BTC_SYMBOL` and `MT5_XAU_SYMBOL` directly.

For a bounded deployment check, set the same environment variables and call
`TradingApplication().run_once()`. This method refuses to run unless dry-run is enabled.

Run deterministic tests without a terminal connection:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Logs rotate under `logs/trading.log`. The loop requests 500 completed bars (the forming bar is excluded) for both M15 and H1. Both timeframe forecasts must agree before sentiment can contribute to a trade decision. Trailing activates after +1.5 ATR and maintains a 1.0 ATR distance.

## AutoGen implementation team

`autogen_orchestration.py` defines an optional Microsoft AutoGen round-robin implementer/reviewer team using `Qwen2.5-Coder-7B-Instruct` through a local OpenAI-compatible endpoint. It is deliberately outside the trading runtime and has no MT5 tools. Set `QWEN_BASE_URL`, start the local Qwen server, and call `review_task()` from a maintenance script when code review is wanted.

## Module map

- `config.py`: offline flags, credentials, account/risk gates, paths.
- `quant_engine.py`: compact OHLC conversion, ATR, Chronos inference, directional mapping.
- `sentiment_engine.py`: pooled async RSS/API collection and FinBERT fallback.
- `execution_agent.py`: MT5 state, sizing, initial SL/TP order payload, trailing stops.
- `main.py`: scheduling, logs, signal consensus, clean shutdown.
- `autogen_orchestration.py`: isolated Qwen-backed implementation/review agents.
