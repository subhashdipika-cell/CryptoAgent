# Asset-Specific BTC/Gold + Chronos + FinBERT MT5 Agent

An asynchronous Windows trading service for BTCUSD and XAUUSD. Chronos-2 runs locally on CPU from pre-staged files; news and FinBERT sentiment are isolated behind an asynchronous, neutral-on-failure adapter. MT5 is the only execution interface.

Dedicated `BTC-DirectRidge` and `XAU-DirectRidge` pipelines are fitted independently
on completed broker OHLC bars. They use different context lengths, regularization,
minimum edge, and cost assumptions. The default `PREDICTIVE_MODE=shadow` records
their forecasts without allowing them to alter Chronos decisions or orders.

## Safety defaults

- Order routing is off (`TRADING_ENABLED=false`) and dry-run logging is on.
- A DEMO account is required by default.
- Every order plan includes its initial SL and TP.
- Risk is capped at 1% of current equity and rounded down to broker lot steps.
- A separate margin policy limits each order to 25% of free margin.
- One managed position per symbol prevents repeated entries each loop.
- Entry orders use Expert ID `26081301` and the Comment
  `CryptoAgent|ChronosFinBERT`, identifying both application and strategy.
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
.\Start-CryptoAgent.bat
```

Double-click `Start-CryptoAgent.bat` to enable order routing to the verified DEMO
terminal. The launcher forces `REQUIRE_DEMO_ACCOUNT=true`, and the runtime refuses
to connect if the selected account is not DEMO. It retains the 1% equity risk cap,
initial SL/TP requirement, and ATR trailing controls. Validate its local terminal,
environment, model, and symbol configuration without starting the loop:

```powershell
.\Start-CryptoAgent.bat --check
```

The PowerShell launcher remains the dry-run option when custom path or symbol
parameters are needed:

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

## Dedicated model validation

Generate a leakage-free walk-forward report from 3,000 completed MT5 bars by
double-clicking `Generate-Predictive-Validation.bat`, or run:

```powershell
.\.venv\Scripts\python.exe predictive_validation.py --bars 3000
```

The report writes `predictive_validation.html`, JSON methodology/results, and
fold-level CSV evidence under `reports/`. It subtracts explicit asset cost
assumptions, but does not reproduce tick-level fills. A numerical PASS is advisory:
the report always marks every candidate `SHADOW_ONLY`; deployment requires an
explicit configuration change after reviewing forward evidence.

Two general-purpose foundation challengers are supported through strictly offline
adapters:

- `ibm-granite/granite-timeseries-ttm-r3`: preferred CPU challenger because its
  family is much smaller. Install `requirements-candidates.txt`, then stage `ttm`.
- `google/timesfm-2.5-200m-transformers`: PyTorch Transformers equivalent of the
  200M TimesFM 2.5 checkpoint. It is an accuracy challenger with a substantially
  larger RAM footprint; stage `timesfm` only for bounded validation.

```powershell
.\.venv\Scripts\python.exe stage_candidate_models.py ttm
.\.venv\Scripts\python.exe stage_candidate_models.py timesfm
```

Neither checkpoint is finance-specific. It must beat the asset-specific baseline
on BTC and Gold walk-forward/forward data before it can influence execution.

## AutoGen implementation team

`autogen_orchestration.py` defines an optional Microsoft AutoGen round-robin implementer/reviewer team using `Qwen2.5-Coder-7B-Instruct` through a local OpenAI-compatible endpoint. It is deliberately outside the trading runtime and has no MT5 tools. Set `QWEN_BASE_URL`, start the local Qwen server, and call `review_task()` from a maintenance script when code review is wanted.

## Trade journal and performance report

CryptoAgent continuously writes an ignored local SQLite journal to
`data/trade_journal.db`. It stores account/equity snapshots, forecast decisions,
submission plans, and reconciled broker orders/deals. Reconciliation is idempotent
and recognizes both legacy Expert ID `84010310` and current ID `26081301`.

Generate fresh reports by double-clicking `Generate-Performance-Report.bat`, or run:

```powershell
.\.venv\Scripts\python.exe performance_report.py --sync
```

Outputs under `reports/` include:

- `performance_report.html`: overall metrics and strategy/asset breakdown.
- `completed_trades.csv`: one row per completed position with net P/L and exit reason.
- `deals.csv` and `orders.csv`: reconciled broker records.
- `submissions.csv`: requested/executed price, planned risk, ATR, SL/TP, and errors.
- `signals.csv`: M15/H1 forecasts, sentiment, ATR, and BUY/SELL/HOLD decisions.
- `model_forecasts.csv`: shadow/active per-model predictions, confidence, and edge.
- `equity_snapshots.csv`: account equity, free margin, leverage, and position count.

Realized metrics come from MT5 deals and include profit, commission, swap, and fees.
Open positions and unfilled orders are deliberately excluded. Broker comments are
descriptive and can be overwritten on SL/TP exits, so Expert ID is the primary
attribution key. Back up the SQLite file if the journal must survive machine loss.

## Module map

- `config.py`: offline flags, credentials, account/risk gates, paths.
- `quant_engine.py`: compact OHLC conversion, ATR, Chronos inference, directional mapping.
- `asset_predictive_engine.py`: separate locally fitted BTC and Gold direct-return models.
- `foundation_backends.py`: offline IBM TTM-R3 and Google TimesFM 2.5 adapters.
- `predictive_validation.py`: completed-bar walk-forward validation and reports.
- `sentiment_engine.py`: pooled async RSS/API collection and FinBERT fallback.
- `execution_agent.py`: MT5 state, sizing, initial SL/TP order payload, trailing stops.
- `main.py`: scheduling, logs, signal consensus, clean shutdown.
- `autogen_orchestration.py`: isolated Qwen-backed implementation/review agents.
- `trade_journal.py`: SQLite decision, account, submission, order, and deal journal.
- `performance_report.py`: MT5 reconciliation and HTML/CSV performance exports.
