"""Broker-aware out-of-sample backtest for the deployed calibrated policy."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import tempfile
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from asset_predictive_engine import ASSET_SPECS, DedicatedAssetForecastEngine, asset_key
from config import SETTINGS, Settings
from decision_engine import CalibratedDecisionEngine
from execution_agent import MT5ExecutionAgent, Side
from predictive_validation import composite_walk_forward, walk_forward
from quant_engine import true_range_atr


@dataclass(slots=True)
class Position:
    symbol: str
    side: Side
    volume: float
    entry_time: int
    entry: float
    stop: float
    target: float
    atr: float
    initial_risk: float


@dataclass(frozen=True, slots=True)
class BacktestTrade:
    symbol: str
    side: str
    entry_time: str
    exit_time: str
    volume: float
    entry: float
    exit: float
    initial_stop: float
    target: float
    exit_reason: str
    gross_profit: float
    commission: float
    net_profit: float
    r_multiple: float
    equity_after: float


def prepare_research_policy(settings: Settings, source: Path) -> tuple[Settings, Path]:
    """Copy an enabled candidate into an isolated replay-only approved policy."""
    payload = json.loads(source.read_text(encoding="utf-8"))
    enabled = [row for row in payload.get("policies", []) if row.get("enabled", False)]
    if not enabled:
        raise ValueError("research policy has no enabled candidate")
    for row in enabled:
        row["approved"] = True
    take_profit_atr = float(
        payload.get("research_take_profit_atr_multiple", settings.take_profit_atr_multiple)
    )
    if take_profit_atr <= 0:
        raise ValueError("research take-profit ATR multiple must be positive")
    max_risk_fraction = float(
        payload.get("research_max_risk_fraction", settings.max_risk_fraction)
    )
    if max_risk_fraction <= 0 or max_risk_fraction > settings.max_risk_fraction:
        raise ValueError("research max-risk fraction must be positive and cannot exceed runtime risk")
    payload["research_only_replay"] = True
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    try:
        json.dump(payload, handle, indent=2, allow_nan=False)
    finally:
        handle.close()
    path = Path(handle.name)
    return (
        replace(
            settings,
            decision_policy_path=path,
            trading_enabled=False,
            dry_run=True,
            take_profit_atr_multiple=take_profit_atr,
            max_risk_fraction=max_risk_fraction,
        ),
        path,
    )


def append_research_experiment(
    source: Path,
    summaries: list[dict[str, object]],
    commission_per_lot_side: float,
    slippage_points: float,
) -> Path:
    policy = json.loads(source.read_text(encoding="utf-8"))
    digest = hashlib.sha256()
    digest.update(source.read_bytes())
    for filename in (
        "asset_predictive_engine.py",
        "decision_engine.py",
        "strategy_backtest.py",
    ):
        digest.update((Path(__file__).resolve().parent / filename).read_bytes())
    configuration_hash = digest.hexdigest()
    backtested = [row for row in summaries if row.get("status") == "BACKTESTED"]
    basic_metrics_pass = bool(backtested) and all(
        int(row.get("trades", 0)) >= 30
        and float(row.get("expectancy", 0.0)) > 0
        and float(row.get("profit_factor", 0.0)) >= 1.20
        and float(row.get("max_closed_equity_drawdown_pct", 999.0)) <= 5.0
        for row in backtested
    )
    record: dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "strategy_id": "asset_calibrated",
        "timeframes": ["H1", "M15"],
        "configuration_hash": configuration_hash,
        "evidence_class": "BROKER_AWARE_POLICY_REPLAY_NOT_FORWARD_EVIDENCE",
        "candidate_policy": [
            row for row in policy.get("policies", []) if row.get("enabled", False)
        ],
        "research_constraints": {
            "side_filter": str(policy.get("research_side_filter", "BOTH")).upper(),
            "take_profit_atr_multiple": float(
                policy.get("research_take_profit_atr_multiple", SETTINGS.take_profit_atr_multiple)
            ),
            "max_risk_fraction": float(
                policy.get("research_max_risk_fraction", SETTINGS.max_risk_fraction)
            ),
            "h1_trend_ema_period": int(policy.get("research_h1_trend_ema_period", 0)),
        },
        "costs": {
            "commission_per_lot_side": commission_per_lot_side,
            "slippage_points_per_fill": slippage_points,
        },
        "replay": summaries,
        "result": (
            "BASIC_REPLAY_METRICS_PASS_UNTOUCHED_NOT_PROVEN"
            if basic_metrics_pass
            else "REPLAY_GATE_REJECTED"
        ),
        "promotion_eligible": False,
        "promotion_blockers": [
            "WALK_FORWARD_STABILITY_NOT_PROVEN",
            "UNTOUCHED_TEST_NOT_PROVEN",
            "FORWARD_DEMO_NOT_STARTED",
        ],
        "validation_windows": [
            {
                "symbol": row.get("symbol"),
                "classification": row.get(
                    "replay_window_classification", "PREVIOUSLY_OBSERVED_POLICY_REPLAY"
                ),
                "fold_id": row.get("replay_fold"),
                "period_start_utc": row.get("period_start_utc"),
                "period_end_utc": row.get("period_end_utc"),
                "trades": row.get("trades", 0),
            }
            for row in backtested
        ],
        "routing_changed": False,
        "forward_demo_status": "NOT_STARTED",
    }
    experiment_id = hashlib.sha256(
        json.dumps(record, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    ledger = Path(__file__).resolve().parent / "research" / "strategy_experiments.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"experiment_id": experiment_id, **record}) + "\n")
    return ledger


def _research_side_allowed(side: Side, side_filter: str) -> bool:
    normalized = str(side_filter).upper()
    if normalized not in {"BOTH", "BUY", "SELL"}:
        raise ValueError(f"unsupported research side filter: {side_filter}")
    return normalized == "BOTH" or side.value == normalized


def _research_replay_bounds(
    length: int,
    default_start: int,
    minimum_history_start: int,
    policy_payload: dict[str, object],
) -> tuple[int, int, str, str | None]:
    window = policy_payload.get("research_replay_window")
    if window is None:
        return default_start, length, "PREVIOUSLY_OBSERVED_POLICY_REPLAY", None
    if not isinstance(window, dict):
        raise ValueError("research replay window must be an object")
    classification = str(window.get("classification", ""))
    if classification != "DEVELOPMENT_WALK_FORWARD":
        raise ValueError("research replay window cannot claim untouched classification")
    start_fraction = float(window.get("start_fraction", -1))
    end_fraction = float(window.get("end_fraction", -1))
    if not 0 <= start_fraction < end_fraction <= 1:
        raise ValueError("research replay fractions must satisfy 0 <= start < end <= 1")
    start = max(minimum_history_start, int(length * start_fraction))
    end = min(length, int(length * end_fraction))
    if end - start < 3:
        raise ValueError("research replay window is too short after history requirements")
    return start, end, classification, str(window.get("fold_id", "UNSPECIFIED"))


def _research_trend_allowed(side: Side, closes: np.ndarray, ema_period: int) -> bool:
    if ema_period == 0:
        return True
    if ema_period < 2:
        raise ValueError("research H1 trend EMA period must be zero or at least two")
    values = np.asarray(closes, dtype=float)
    if len(values) < ema_period:
        return False
    alpha = 2.0 / (ema_period + 1.0)
    ema = float(values[0])
    for value in values[1:]:
        ema = alpha * float(value) + (1.0 - alpha) * ema
    return float(values[-1]) > ema if side is Side.BUY else float(values[-1]) < ema


def _timestamp(value: int) -> str:
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()


def _bars_completed_by_observation(
    rates: Any,
    timeframe_minutes: int,
    observed_at: datetime,
) -> Any:
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise ValueError("historical observation time must be timezone-aware")
    if timeframe_minutes <= 0:
        raise ValueError("timeframe minutes must be positive")
    cutoff = int(observed_at.astimezone(timezone.utc).timestamp())
    closes_at = rates["time"].astype(np.int64) + timeframe_minutes * 60
    completed = rates[closes_at <= cutoff]
    if len(completed) == 0:
        raise RuntimeError("no bars were completed by the historical observation time")
    return completed


def _round_volume(value: float, minimum: float, maximum: float, step: float) -> float:
    steps = math.floor((min(value, maximum) - minimum + 1e-12) / step)
    return round(minimum + max(0, steps) * step, 8) if value >= minimum else 0.0


def _profit(mt5: Any, symbol: str, side: Side, volume: float, entry: float, exit_price: float) -> float:
    order_type = mt5.ORDER_TYPE_BUY if side is Side.BUY else mt5.ORDER_TYPE_SELL
    value = mt5.order_calc_profit(order_type, symbol, volume, entry, exit_price)
    if value is None:
        raise RuntimeError(f"order_calc_profit failed for historical {symbol}: {mt5.last_error()}")
    return float(value)


def _round_trip_commission(volume: float, commission_per_lot_side: float) -> float:
    if volume < 0 or commission_per_lot_side < 0:
        raise ValueError("volume and commission rate cannot be negative")
    return 2.0 * commission_per_lot_side * volume


def _plan(
    mt5: Any, settings: Settings, symbol: str, side: Side, entry: float, atr: float,
    equity: float,
) -> Position | None:
    info = mt5.symbol_info(symbol)
    sign = 1.0 if side is Side.BUY else -1.0
    stop = entry - sign * settings.stop_atr_multiple * atr
    target = entry + sign * settings.take_profit_atr_multiple * atr
    loss_per_lot = abs(_profit(mt5, symbol, side, 1.0, entry, stop))
    if loss_per_lot <= 0:
        return None
    volume = _round_volume(
        equity * settings.max_risk_fraction / loss_per_lot,
        float(info.volume_min), float(info.volume_max), float(info.volume_step),
    )
    if volume <= 0:
        return None
    risk = abs(_profit(mt5, symbol, side, volume, entry, stop))
    return Position(symbol, side, volume, 0, entry, stop, target, atr, risk)


def _exit_for_bar(
    position: Position, row: Any, point: float, slippage_points: float
) -> tuple[float, str] | None:
    spread = float(row["spread"]) * point if "spread" in (row.dtype.names or ()) else 0.0
    slip = slippage_points * point
    if position.side is Side.BUY:
        stop_hit = float(row["low"]) <= position.stop
        target_hit = float(row["high"]) >= position.target
        if stop_hit:
            gap_price = min(position.stop, float(row["open"]))
            return gap_price - slip, "STOP_LOSS" if not target_hit else "AMBIGUOUS_STOP_FIRST"
        if target_hit:
            return position.target - slip, "TAKE_PROFIT"
    else:
        ask_open = float(row["open"]) + spread
        ask_high = float(row["high"]) + spread
        ask_low = float(row["low"]) + spread
        stop_hit = ask_high >= position.stop
        target_hit = ask_low <= position.target
        if stop_hit:
            gap_price = max(position.stop, ask_open)
            return gap_price + slip, "STOP_LOSS" if not target_hit else "AMBIGUOUS_STOP_FIRST"
        if target_hit:
            return position.target + slip, "TAKE_PROFIT"
    return None


def backtest_symbol(
    symbol: str,
    m15_rates: np.ndarray,
    h1_rates: np.ndarray,
    execution: MT5ExecutionAgent,
    settings: Settings,
    starting_equity: float,
    commission_per_lot_side: float,
    slippage_points: float,
) -> tuple[list[BacktestTrade], dict[str, object]]:
    policy_payload = json.loads(settings.decision_policy_path.read_text(encoding="utf-8"))
    policy = next(row for row in policy_payload["policies"] if row["symbol"] == symbol)
    research_side_filter = str(policy_payload.get("research_side_filter", "BOTH")).upper()
    research_h1_trend_ema_period = int(policy_payload.get("research_h1_trend_ema_period", 0))
    if not policy["enabled"] or not policy.get("approved", False):
        return [], {"symbol": symbol, "status": "DISABLED_BY_HOLDOUT_POLICY"}

    m15 = np.asarray(m15_rates)
    h1 = np.asarray(h1_rates)
    if policy.get("decision_mode", "M15_H1") == "H1_ONLY":
        validation_folds = walk_forward(symbol, h1, "1h", settings)
    else:
        validation_folds = composite_walk_forward(symbol, m15, h1, settings)
    cutoff = validation_folds[max(1, int(len(validation_folds) * 0.65))].origin_time
    start_index = int(np.searchsorted(m15["time"], cutoff, side="left"))
    spec = ASSET_SPECS[asset_key(symbol)]
    minimum_history_start = spec.context + spec.minimum_samples + settings.prediction_length
    start_index = max(start_index, minimum_history_start)
    start_index, evaluation_end_index, replay_classification, replay_fold = _research_replay_bounds(
        len(m15), start_index, minimum_history_start, policy_payload
    )
    engine = DedicatedAssetForecastEngine(settings)
    decisions = CalibratedDecisionEngine(settings.decision_policy_path)
    info = execution.mt5.symbol_info(symbol)
    point = float(info.point)
    equity = starting_equity
    peak = equity
    max_drawdown = 0.0
    position: Position | None = None
    pending: Position | None = None
    trades: list[BacktestTrade] = []
    evaluated = entries_blocked_by_risk = entries_blocked_by_side_filter = 0
    entries_blocked_by_trend_filter = 0
    last_h1_decision_index = -1

    for index in range(start_index, evaluation_end_index - 1):
        row = m15[index]
        if pending is not None:
            spread = float(row["spread"]) * point if "spread" in (row.dtype.names or ()) else 0.0
            slip = slippage_points * point
            entry = float(row["open"]) + (spread + slip if pending.side is Side.BUY else -slip)
            planned = _plan(execution.mt5, settings, symbol, pending.side, entry, pending.atr, equity)
            if planned is None:
                entries_blocked_by_risk += 1
            else:
                planned.entry_time = int(row["time"])
                position = planned
            pending = None

        if position is not None:
            exit_event = _exit_for_bar(position, row, point, slippage_points)
            if exit_event is not None:
                exit_price, reason = exit_event
                gross = _profit(
                    execution.mt5, symbol, position.side, position.volume,
                    position.entry, exit_price,
                )
                commission = _round_trip_commission(
                    position.volume, commission_per_lot_side
                )
                net = gross - commission
                equity += net
                peak = max(peak, equity)
                max_drawdown = max(max_drawdown, peak - equity)
                trades.append(
                    BacktestTrade(
                        symbol, position.side.value, _timestamp(position.entry_time),
                        _timestamp(int(row["time"])), position.volume, position.entry,
                        exit_price, position.entry - (1 if position.side is Side.BUY else -1)
                        * settings.stop_atr_multiple * position.atr,
                        position.target, reason, gross, commission, net,
                        net / position.initial_risk, equity,
                    )
                )
                position = None
            else:
                spread = float(row["spread"]) * point if "spread" in (row.dtype.names or ()) else 0.0
                close = float(row["close"]) if position.side is Side.BUY else float(row["close"]) + spread
                favorable = close - position.entry if position.side is Side.BUY else position.entry - close
                if favorable >= settings.trailing_trigger_atr * position.atr:
                    candidate = (
                        close - settings.trailing_distance_atr * position.atr
                        if position.side is Side.BUY
                        else close + settings.trailing_distance_atr * position.atr
                    )
                    position.stop = max(position.stop, candidate) if position.side is Side.BUY else min(position.stop, candidate)

        if position is None and pending is None:
            h1_last = int(np.searchsorted(h1["time"], int(row["time"]) - 2700, side="right") - 1)
            if h1_last < spec.context + spec.minimum_samples:
                continue
            if policy.get("decision_mode", "M15_H1") == "H1_ONLY":
                if h1_last == last_h1_decision_index:
                    continue
                last_h1_decision_index = h1_last
            m15_training = m15[max(0, index - settings.bar_count + 1) : index + 1]
            h1_training = h1[max(0, h1_last - settings.bar_count + 1) : h1_last + 1]
            atr = true_range_atr(m15_training, settings.atr_period)
            m15_forecast = engine.forecast(symbol, m15_training, "15min")
            h1_forecast = engine.forecast(symbol, h1_training, "1h")
            outcome = decisions.evaluate(symbol, m15_forecast, h1_forecast, 0.5, True)
            evaluated += 1
            if outcome.side:
                if not _research_side_allowed(outcome.side, research_side_filter):
                    entries_blocked_by_side_filter += 1
                elif not _research_trend_allowed(
                    outcome.side, h1_training["close"], research_h1_trend_ema_period
                ):
                    entries_blocked_by_trend_filter += 1
                else:
                    pending = Position(symbol, outcome.side, 0.0, 0, 0.0, 0.0, 0.0, atr, 0.0)

    if position is not None:
        row = m15[evaluation_end_index - 1]
        spread = float(row["spread"]) * point if "spread" in (row.dtype.names or ()) else 0.0
        exit_price = float(row["close"]) if position.side is Side.BUY else float(row["close"]) + spread
        gross = _profit(execution.mt5, symbol, position.side, position.volume, position.entry, exit_price)
        commission = _round_trip_commission(position.volume, commission_per_lot_side)
        net = gross - commission
        equity += net
        trades.append(
            BacktestTrade(
                symbol, position.side.value, _timestamp(position.entry_time), _timestamp(int(row["time"])),
                position.volume, position.entry, exit_price,
                position.entry - (1 if position.side is Side.BUY else -1) * settings.stop_atr_multiple * position.atr,
                position.target, "END_OF_TEST", gross, commission, net,
                net / position.initial_risk, equity,
            )
        )

    returns = [trade.net_profit for trade in trades]
    wins = [value for value in returns if value > 0]
    losses = [-value for value in returns if value < 0]
    consecutive_losses = maximum_consecutive_losses = 0
    for value in returns:
        consecutive_losses = consecutive_losses + 1 if value < 0 else 0
        maximum_consecutive_losses = max(maximum_consecutive_losses, consecutive_losses)
    summary = {
        "symbol": symbol,
        "status": "BACKTESTED",
        "period_start_utc": _timestamp(int(m15[start_index]["time"])),
        "period_end_utc": _timestamp(int(m15[evaluation_end_index - 1]["time"])),
        "replay_window_classification": replay_classification,
        "replay_fold": replay_fold,
        "starting_equity": starting_equity,
        "ending_equity": equity,
        "net_profit": equity - starting_equity,
        "gross_profit_after_spread_slippage": sum(trade.gross_profit for trade in trades),
        "total_commission": sum(trade.commission for trade in trades),
        "return_pct": (equity / starting_equity - 1.0) * 100.0,
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": len(wins) / len(trades) * 100.0 if trades else 0.0,
        "profit_factor": sum(wins) / sum(losses) if losses else (999.0 if wins else 0.0),
        "expectancy": sum(returns) / len(trades) if trades else 0.0,
        "average_r": sum(trade.r_multiple for trade in trades) / len(trades) if trades else 0.0,
        "max_closed_equity_drawdown": max_drawdown,
        "max_closed_equity_drawdown_pct": max_drawdown / starting_equity * 100.0,
        "maximum_consecutive_losses": maximum_consecutive_losses,
        "maximum_volume": max((trade.volume for trade in trades), default=0.0),
        "evaluated_entry_bars": evaluated,
        "entries_blocked_by_risk": entries_blocked_by_risk,
        "entries_blocked_by_side_filter": entries_blocked_by_side_filter,
        "entries_blocked_by_trend_filter": entries_blocked_by_trend_filter,
        "research_side_filter": research_side_filter,
        "research_h1_trend_ema_period": research_h1_trend_ema_period,
        "risk_cap_pct": settings.max_risk_fraction * 100.0,
        "commission_per_lot_per_side": commission_per_lot_side,
        "slippage_points_per_fill": slippage_points,
        "intrabar_policy": "stop first when stop and target are both touched",
    }
    return trades, summary


def write_report(settings: Settings, summaries: list[dict[str, object]], trades: list[BacktestTrade]) -> None:
    settings.report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "scope": "chronological policy-holdout execution replay for locked calibrated thresholds",
        "limitations": [
            "M15 OHLC cannot reproduce tick ordering or actual liquidity",
            "historical spread is used where MT5 supplies it",
            "trailing updates occur at bar close and apply from the following bar",
            "results are not forward trading evidence",
            "the replay period is previously observed research evidence and is not an untouched test set",
            "basic replay metric passes remain ineligible for promotion until walk-forward stability and a new untouched window are proven",
        ],
        "summaries": summaries,
    }
    (settings.report_dir / "strategy_backtest.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    with (settings.report_dir / "strategy_backtest_trades.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(BacktestTrade.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(trade) for trade in trades)
    headers = "".join(f"<th>{html.escape(key)}</th>" for key in summaries[0])
    rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row.values()) + "</tr>"
        for row in summaries
    )
    trade_rows = "".join(
        f"<tr><td>{trade.entry_time}</td><td>{trade.symbol}</td><td>{trade.side}</td><td>{trade.volume}</td><td>{trade.exit_reason}</td><td>{trade.net_profit:.2f}</td><td>{trade.r_multiple:.2f}</td></tr>"
        for trade in trades
    )
    document = f"""<!doctype html><meta charset='utf-8'><title>CryptoAgent Strategy Backtest</title>
<style>body{{font:14px Segoe UI,Arial;margin:32px}}table{{border-collapse:collapse;font-size:12px}}th,td{{border:1px solid #ccc;padding:6px;text-align:right}}th{{background:#eee}}.warning{{background:#fff3cd;padding:12px}}</style>
<h1>CryptoAgent AssetCalibrated Backtest</h1><p class='warning'>Policy-holdout OHLC replay, not an independent second test or proof of future performance. Ambiguous bars are resolved stop-first.</p>
<table><thead><tr>{headers}</tr></thead><tbody>{rows}</tbody></table>
<h2>Trades</h2><table><tr><th>Entry UTC</th><th>Symbol</th><th>Side</th><th>Volume</th><th>Exit</th><th>Net</th><th>R</th></tr>{trade_rows}</table>"""
    (settings.report_dir / "strategy_backtest.html").write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bars", type=int, default=3000)
    parser.add_argument(
        "--commission-per-lot-side", "--commission-per-side",
        dest="commission_per_lot_side", type=float, default=3.0,
    )
    parser.add_argument("--slippage-points", type=float, default=10.0)
    parser.add_argument("--starting-equity", type=float)
    parser.add_argument(
        "--research-policy",
        type=Path,
        help="Replay an enabled, unapproved candidate through an isolated temporary policy.",
    )
    args = parser.parse_args()
    settings = SETTINGS
    settings.validate()
    temporary_policy: Path | None = None
    if args.research_policy is not None:
        settings, temporary_policy = prepare_research_policy(
            settings, args.research_policy
        )
    execution: MT5ExecutionAgent | None = None
    summaries: list[dict[str, object]] = []
    all_trades: list[BacktestTrade] = []
    try:
        execution = MT5ExecutionAgent(settings)
        execution.connect()
        account = execution.mt5.account_info()
        if account is None:
            raise RuntimeError("MT5 account unavailable")
        starting_equity = (
            float(args.starting_equity)
            if args.starting_equity is not None
            else float(account.balance)
        )
        if starting_equity <= 0:
            raise ValueError("starting equity must be positive")
        observed_at = datetime.now(timezone.utc)
        for symbol in settings.symbols:
            m15 = execution.bars(symbol, execution.mt5.TIMEFRAME_M15, args.bars)
            h1 = execution.bars(symbol, execution.mt5.TIMEFRAME_H1, args.bars)
            m15 = _bars_completed_by_observation(m15, 15, observed_at)
            h1 = _bars_completed_by_observation(h1, 60, observed_at)
            trades, summary = backtest_symbol(
                symbol, m15, h1, execution, settings, starting_equity,
                args.commission_per_lot_side, args.slippage_points,
            )
            all_trades.extend(trades)
            summaries.append(summary)
    finally:
        if execution is not None:
            execution.shutdown()
        if temporary_policy is not None:
            temporary_policy.unlink(missing_ok=True)
    write_report(settings, summaries, all_trades)
    if args.research_policy is not None:
        ledger = append_research_experiment(
            args.research_policy,
            summaries,
            args.commission_per_lot_side,
            args.slippage_points,
        )
        print(ledger)
    print(settings.report_dir / "strategy_backtest.html")


if __name__ == "__main__":
    main()
