# Asset-Specific BTC/Gold + Chronos + FinBERT MT5 Agent

An asynchronous Windows trading service for BTCUSD and XAUUSD. Chronos-2 runs locally on CPU from pre-staged files; news and FinBERT sentiment are isolated behind an asynchronous, neutral-on-failure adapter. MT5 is the only execution interface.

Dedicated `BTC-DirectRidge` and `XAU-DirectRidge` pipelines are fitted independently
on completed broker OHLC bars. They use different context lengths, regularization,
minimum edge, and cost assumptions. The production launcher uses
`PREDICTIVE_MODE=calibrated`: only a policy that passes a chronological holdout
gate may influence DEMO orders. Failed or missing policies fail closed with
`UNVALIDATED_MODEL`.

## Safety defaults

- Order routing is off (`TRADING_ENABLED=false`) and dry-run logging is on.
- A DEMO account is required by default.
- Every order plan includes its initial SL and TP.
- Risk is capped at 2% of current equity and rounded down to broker lot steps.
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

`Start-CryptoAgent.bat` requests order routing to the verified DEMO terminal, but
startup also requires the selected strategy to be `DEMO_READY` in
`policies/strategy_readiness.json`. Missing, rejected, revoked, research-only, or
configuration-stale readiness evidence fails closed before forecast or execution
engines initialize. Readiness and its successful DEMO reconciliation proof expire
after 15 minutes by default. The launcher also forces `REQUIRE_DEMO_ACCOUNT=true`; the
runtime refuses any non-DEMO account and retains the 2% equity risk cap, initial
SL/TP requirement, and ATR trailing controls. Validate its local terminal,
environment, model, symbol, and readiness configuration without starting the loop:

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
assumptions, but does not reproduce tick-level fills. Gold's complete M15+H1
policy uses the earlier 65% of chronological folds to select thresholds and the
untouched later 35% for its deployment gate. BTC uses a dedicated H1-only policy:
M15 is still journaled for diagnostics but cannot block, authorize, or change the
H1 direction. Each completed H1 bar is evaluated at most once, preventing repeated
entries from the same hourly forecast. BTC H1 validation samples up to 360
chronological folds. The threshold is selected only on the earlier 65%, and DEMO
eligibility requires at least five later holdout trades, positive net returns
after assumed costs, at least 52% directional accuracy, and profit factor 1.10.

Every signal records a decision reason: `ENTRY_SIGNAL`, `TIMEFRAME_DISAGREEMENT`,
`INSUFFICIENT_EDGE`, `UNVALIDATED_MODEL`, `MODEL_POLICY_MISMATCH`,
`H1_BAR_ALREADY_EVALUATED`, or `POSITION_ALREADY_OPEN`. A signal that cannot satisfy broker size, margin, or
risk constraints is recorded as `ORDER_PLAN_REJECTED`. When sentiment is degraded, its weight is omitted and
the quantitative timeframes are renormalized instead of assigning permanent
neutral weight.

### XAUUSD minimum-equity paper report

Run `Generate-Paper-Risk-Report.bat` to create read-only HTML and JSON evidence for
XAUUSD 0.01 lot under a fixed 1% reference risk cap. The report reads the current
DEMO account equity, broker contract metadata, and completed-bar ATR, then shows
the minimum equity required at the current ATR stop and the maximum ATR/stop
distance supported by current equity. It forcibly disables order routing and does
not change the configured runtime sizing cap.

Each `ORDER_PLAN_REJECTED` decision is also written to the
`order_plan_rejections` journal table with the original error, 1% risk/equity
shortfalls, minimum equity, and maximum ATR/stop conditions. These rows are
exported to `reports/order_plan_rejections.csv` and shown in the performance report.

### Scheduled revalidation and manual promotion

The production launcher counts unique completed M15 bars persistently under
`data/revalidation_state.json`. After at least 500 new bars, it starts the
validation script in an isolated hidden process while trading/position management
continues. Revalidation writes only
`reports/candidate_asset_decision_policy.json`; it never changes the active policy.

Inspect active and candidate policies:

```powershell
.\.venv\Scripts\python.exe policy_admin.py status
```

After reviewing a passing BTC candidate, promotion requires this explicit command:

```powershell
.\.venv\Scripts\python.exe policy_admin.py approve BTCUSD
```

A failed candidate cannot be approved. A successful manual approval is recorded
in the active policy audit, and CryptoAgent must be restarted before the approved
policy can influence DEMO orders. Gold's current active approval is preserved by
candidate revalidation.

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

## Strategy backtest

Double-click `Run-Strategy-Backtest.bat` to replay the locked calibrated policy
over its chronological policy-holdout period. The broker-aware proxy uses MT5
historical spread, next-M15-bar entry, 0.03 commission per side, 10 points of
adverse slippage per fill, dynamic 2% risk sizing, broker volume steps, hard
SL/TP, and conservative stop-first resolution for ambiguous OHLC bars.

Outputs are `strategy_backtest.html`, `strategy_backtest.json`, and
`strategy_backtest_trades.csv` under `reports/`. This is a policy-holdout replay,
not a second untouched test or forward evidence; retain DEMO execution until a
separate forward sample is large enough to assess.

## AutoGen implementation team

`autogen_orchestration.py` defines an optional Microsoft AutoGen round-robin implementer/reviewer team using `Qwen2.5-Coder-7B-Instruct` through a local OpenAI-compatible endpoint. It is deliberately outside the trading runtime and has no MT5 tools. Set `QWEN_BASE_URL`, start the local Qwen server, and call `review_task()` from a maintenance script when code review is wanted.

## Institutional liquidity breakout strategy (opt-in DEMO)

The existing `Start-CryptoAgent.bat` calibrated route is unchanged. To select the
new strategy explicitly, run the non-live launcher check first:

```powershell
.\Start-CryptoAgent-Liquidity-Demo.bat --check
```

Then double-click `Start-CryptoAgent-Liquidity-Demo.bat` to permit orders only on
the configured MT5 DEMO account. The strategy uses completed candles exclusively:

1. H4: requires at least three repeated touches of a recent range boundary and
   locates the nearest older swing beyond that boundary as the liquidity target.
2. M15: uses the preceding structure boundary for breakout confirmation and the
   nearest eight-bar structural extreme for the fixed stop.
3. M3: requires a directional breakout candle with at least 60% body/range and
   tick volume at least 1.2 times the prior 20-bar average.
4. Broker-price planning rechecks that reward/risk is at least 1:2.5 before sizing.

No retail-positioning feed is available, so retail bait is a deterministic
repeated-touch price-structure proxy, not a claim about measured crowd positions.
The 25% of up-to-$1,000 daily target is an anti-greed stop ceiling, not an expected
or guaranteed return. Per-trade loss remains capped at 2% of account equity, broker
minimum-stop and margin limits remain enforced, and every order has a fixed SL/TP.

The Asia/Kolkata daily gate counts only `LiquidityBreakout` position IDs for Expert
ID `26081301`, includes commission, swap, and fees in realized P/L, permits at most
three entries, and stops new entries once the daily target is realized. Only
positions carrying `CryptoAgent|LiquidityBreakout` are moved to breakeven at 2R;
minor pullbacks do not cause discretionary closes.

Each evaluation is logged as `LIQUIDITY_EVALUATION` JSON and persisted to
`liquidity_signals`, including the H4 bias/target, bait level, M3 entry, M15 stop,
RRR, risk, projected reward, status, and notes. This is a new unvalidated strategy:
keep it on DEMO until deterministic replay and reconciled forward evidence support
any later promotion decision.

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
- `forward_evidence.csv`: per-asset reconciled DEMO results entered inside the
  policy-active window, including sample size, net P/L after costs, drawdown, win rate, and profit factor.
- `completed_trades.csv`: one row per completed position with net P/L and exit reason.
- `deals.csv` and `orders.csv`: reconciled broker records.
- `submissions.csv`: requested/executed price, planned risk, ATR, SL/TP, and errors.
- `signals.csv`: M15/H1 forecasts, sentiment, ATR, and BUY/SELL/HOLD decisions.
- `model_forecasts.csv`: shadow/active per-model predictions, confidence, and edge.
- `order_plan_rejections.csv`: one row per rejected entry plan with 1% paper shortfalls.
- `equity_snapshots.csv`: account equity, free margin, leverage, and position count.

Realized metrics come from MT5 deals and include profit, commission, swap, and fees.
Open positions and unfilled orders are deliberately excluded. Broker comments are
descriptive and can be overwritten on SL/TP exits, so Expert ID is the primary
attribution key. Back up the SQLite file if the journal must survive machine loss.

Forward evidence uses only the current routed Expert ID, requires the latest
account/server snapshot to prove MT5 DEMO mode, and requires 30 completed trades
per asset before leaving `INSUFFICIENT_FORWARD_EVIDENCE`. The threshold indicates
sample availability, not profitability or promotion eligibility. Reporting is
read-only and never changes policy, eligibility, sizing, or order routing.

## Strategy readiness dashboard and deployment gate

Refresh reconciled DEMO evidence first, then generate the auditable readiness
registry and dashboard:

```powershell
$env:TRADING_ENABLED = "false"
$env:DRY_RUN = "true"
.\.venv\Scripts\python.exe performance_report.py --sync
.\.venv\Scripts\python.exe strategy_readiness.py generate
```

Open `reports/strategy_readiness.html` for the human-readable dashboard. The
tracked `policies/strategy_readiness.json` is the runtime control record. A
strategy becomes `DEMO_READY` only when its own reconciled forward DEMO evidence
has at least 30 closed trades over 10 sessions, positive net expectancy after
costs, profit factor of at least 1.20, maximum drawdown no greater than 5%, and an
enabled, approved policy. Backtests and replay remain separately labelled and
cannot make a strategy deployable. Any strategy or policy-code change invalidates
the stored configuration hash and blocks routing until evidence is regenerated.
This gate authorizes DEMO routing only and never authorizes LIVE trading or
guarantees future profit.

## Module map

- `config.py`: offline flags, credentials, account/risk gates, paths.
- `quant_engine.py`: compact OHLC conversion, ATR, Chronos inference, directional mapping.
- `asset_predictive_engine.py`: separate locally fitted BTC and Gold direct-return models.
- `foundation_backends.py`: offline IBM TTM-R3 and Google TimesFM 2.5 adapters.
- `predictive_validation.py`: completed-bar walk-forward validation and reports.
- `strategy_backtest.py`: broker-aware calibrated-policy execution replay.
- `strategy_readiness.py`: readiness registry, HTML dashboard, and fail-closed DEMO deployment gate.
- `paper_risk_report.py`: read-only XAUUSD 0.01-lot minimum-equity and ATR report.
- `sentiment_engine.py`: pooled async RSS/API collection and FinBERT fallback.
- `execution_agent.py`: MT5 state, sizing, initial SL/TP order payload, trailing stops.
- `main.py`: scheduling, logs, signal consensus, clean shutdown.
- `autogen_orchestration.py`: isolated Qwen-backed implementation/review agents.
- `trade_journal.py`: SQLite decision, account, submission, order, and deal journal.
- `performance_report.py`: MT5 reconciliation and HTML/CSV performance exports.
