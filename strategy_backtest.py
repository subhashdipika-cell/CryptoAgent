"""Broker-aware out-of-sample backtest for the deployed calibrated policy."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from asset_predictive_engine import ASSET_SPECS, DedicatedAssetForecastEngine, asset_key
from config import SETTINGS, Settings
from decision_engine import CalibratedDecisionEngine
from execution_agent import MT5ExecutionAgent, Side
from predictive_validation import composite_walk_forward
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


def _timestamp(value: int) -> str:
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()


def _round_volume(value: float, minimum: float, maximum: float, step: float) -> float:
    steps = math.floor((min(value, maximum) - minimum + 1e-12) / step)
    return round(minimum + max(0, steps) * step, 8) if value >= minimum else 0.0


def _profit(mt5: Any, symbol: str, side: Side, volume: float, entry: float, exit_price: float) -> float:
    order_type = mt5.ORDER_TYPE_BUY if side is Side.BUY else mt5.ORDER_TYPE_SELL
    value = mt5.order_calc_profit(order_type, symbol, volume, entry, exit_price)
    if value is None:
        raise RuntimeError(f"order_calc_profit failed for historical {symbol}: {mt5.last_error()}")
    return float(value)


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
    commission_per_side: float,
    slippage_points: float,
) -> tuple[list[BacktestTrade], dict[str, object]]:
    policy_payload = json.loads(settings.decision_policy_path.read_text(encoding="utf-8"))
    policy = next(row for row in policy_payload["policies"] if row["symbol"] == symbol)
    if not policy["enabled"]:
        return [], {"symbol": symbol, "status": "DISABLED_BY_HOLDOUT_POLICY"}

    m15 = np.asarray(m15_rates)
    h1 = np.asarray(h1_rates)
    agreement_folds = composite_walk_forward(symbol, m15, h1, settings)
    cutoff = agreement_folds[max(1, int(len(agreement_folds) * 0.65))].origin_time
    start_index = int(np.searchsorted(m15["time"], cutoff, side="left"))
    spec = ASSET_SPECS[asset_key(symbol)]
    start_index = max(start_index, spec.context + spec.minimum_samples + settings.prediction_length)
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
    evaluated = entries_blocked_by_risk = 0

    for index in range(start_index, len(m15) - 1):
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
                commission = 2.0 * commission_per_side
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
            m15_training = m15[max(0, index - settings.bar_count + 1) : index + 1]
            h1_training = h1[max(0, h1_last - settings.bar_count + 1) : h1_last + 1]
            atr = true_range_atr(m15_training, settings.atr_period)
            m15_forecast = engine.forecast(symbol, m15_training, "15min")
            h1_forecast = engine.forecast(symbol, h1_training, "1h")
            outcome = decisions.evaluate(symbol, m15_forecast, h1_forecast, 0.5, True)
            evaluated += 1
            if outcome.side:
                pending = Position(symbol, outcome.side, 0.0, 0, 0.0, 0.0, 0.0, atr, 0.0)

    if position is not None:
        row = m15[-1]
        spread = float(row["spread"]) * point if "spread" in (row.dtype.names or ()) else 0.0
        exit_price = float(row["close"]) if position.side is Side.BUY else float(row["close"]) + spread
        gross = _profit(execution.mt5, symbol, position.side, position.volume, position.entry, exit_price)
        commission = 2.0 * commission_per_side
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
        "period_end_utc": _timestamp(int(m15[-1]["time"])),
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
        "risk_cap_pct": settings.max_risk_fraction * 100.0,
        "commission_per_side": commission_per_side,
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
            "the replay period was used to decide whether the policy passed its deployment gate and is not a second untouched test set",
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
    parser.add_argument("--commission-per-side", type=float, default=0.03)
    parser.add_argument("--slippage-points", type=float, default=10.0)
    args = parser.parse_args()
    settings = SETTINGS
    settings.validate()
    execution = MT5ExecutionAgent(settings)
    execution.connect()
    account = execution.mt5.account_info()
    if account is None:
        raise RuntimeError("MT5 account unavailable")
    summaries: list[dict[str, object]] = []
    all_trades: list[BacktestTrade] = []
    try:
        for symbol in settings.symbols:
            m15 = execution.bars(symbol, execution.mt5.TIMEFRAME_M15, args.bars)
            h1 = execution.bars(symbol, execution.mt5.TIMEFRAME_H1, args.bars)
            trades, summary = backtest_symbol(
                symbol, m15, h1, execution, settings, float(account.balance),
                args.commission_per_side, args.slippage_points,
            )
            all_trades.extend(trades)
            summaries.append(summary)
    finally:
        execution.shutdown()
    write_report(settings, summaries, all_trades)
    print(settings.report_dir / "strategy_backtest.html")


if __name__ == "__main__":
    main()
