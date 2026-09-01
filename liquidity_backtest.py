"""Chronological broker-aware replay for the opt-in liquidity-breakout strategy."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from config import SETTINGS, Settings
from execution_agent import MT5ExecutionAgent, Side
from liquidity_breakout import LiquidityBreakoutDecision, LiquidityBreakoutEngine, daily_lock_reason


@dataclass(slots=True)
class ReplayPosition:
    symbol: str
    side: Side
    volume: float
    entry_time: int
    entry: float
    initial_stop: float
    stop: float
    target: float
    initial_risk_usd: float
    breakeven_armed: bool = False


@dataclass(frozen=True, slots=True)
class ReplayTrade:
    symbol: str
    side: str
    entry_time_utc: str
    exit_time_utc: str
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


@dataclass(slots=True)
class PendingEntry:
    decision: LiquidityBreakoutDecision
    signal_time: int


def _iso(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def _day(timestamp: int, timezone_name: str) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).astimezone(
        ZoneInfo(timezone_name)
    ).date().isoformat()


def _profit(mt5: Any, symbol: str, side: Side, volume: float, entry: float, exit_price: float) -> float:
    order_type = mt5.ORDER_TYPE_BUY if side is Side.BUY else mt5.ORDER_TYPE_SELL
    result = mt5.order_calc_profit(order_type, symbol, volume, entry, exit_price)
    if result is None:
        raise RuntimeError(f"order_calc_profit failed for {symbol}: {mt5.last_error()}")
    return float(result)


def _round_volume(raw: float, minimum: float, maximum: float, step: float) -> float:
    if raw < minimum or step <= 0:
        return 0.0
    units = math.floor((min(raw, maximum) + 1e-12) / step)
    decimals = max(0, int(round(-math.log10(step)))) if step < 1 else 0
    return round(units * step, decimals)


def _fetch_closed(mt5: Any, symbol: str, timeframe: int, count: int) -> np.ndarray:
    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"symbol_select({symbol}) failed: {mt5.last_error()}")
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, count)
    if rates is None:
        raise RuntimeError(f"copy_rates_from_pos({symbol}) failed: {mt5.last_error()}")
    return np.asarray(rates)


def _closed_slice(rates: np.ndarray, trigger_open: int, timeframe_seconds: int, limit: int) -> np.ndarray:
    """Return bars known closed when the three-minute trigger bar closes."""
    latest_open = trigger_open + 180 - timeframe_seconds
    end = int(np.searchsorted(rates["time"], latest_open, side="right"))
    return rates[max(0, end - limit) : end]


def _entry_plan(
    mt5: Any,
    settings: Settings,
    symbol: str,
    decision: LiquidityBreakoutDecision,
    row: Any,
    equity: float,
    slippage_points: float,
) -> tuple[ReplayPosition | None, str]:
    assert decision.side is not None
    assert decision.stop_loss_15m is not None and decision.take_profit_4h is not None
    info = mt5.symbol_info(symbol)
    point = float(info.point)
    spread = float(row["spread"]) * point if "spread" in (row.dtype.names or ()) else 0.0
    slip = slippage_points * point
    entry = float(row["open"]) + (spread + slip if decision.side is Side.BUY else -slip)
    stop, target = float(decision.stop_loss_15m), float(decision.take_profit_4h)
    risk_distance = entry - stop if decision.side is Side.BUY else stop - entry
    reward_distance = target - entry if decision.side is Side.BUY else entry - target
    if risk_distance <= 0 or reward_distance <= 0:
        return None, "BROKER_PRICE_INVALID_STRUCTURE"
    if reward_distance / risk_distance < settings.liquidity_min_rrr:
        return None, "BROKER_PRICE_INSUFFICIENT_RRR"
    if min(risk_distance, reward_distance) < float(info.trade_stops_level) * point:
        return None, "BROKER_MINIMUM_STOP"
    loss_per_lot = abs(_profit(mt5, symbol, decision.side, 1.0, entry, stop))
    if loss_per_lot <= 0:
        return None, "INVALID_TICK_VALUE"
    volume = _round_volume(
        equity * settings.max_risk_fraction / loss_per_lot,
        float(info.volume_min),
        float(info.volume_max),
        float(info.volume_step),
    )
    if volume <= 0:
        return None, "MINIMUM_LOT_EXCEEDS_RISK"
    initial_risk = abs(_profit(mt5, symbol, decision.side, volume, entry, stop))
    return ReplayPosition(
        symbol, decision.side, volume, int(row["time"]), entry, stop, stop, target, initial_risk
    ), "ENTERED"


def _position_event(
    position: ReplayPosition,
    row: Any,
    point: float,
    slippage_points: float,
) -> tuple[float, str] | None:
    """Resolve stops first; arm breakeven only after a completed non-exit bar."""
    spread = float(row["spread"]) * point if "spread" in (row.dtype.names or ()) else 0.0
    slip = slippage_points * point
    risk_distance = abs(position.entry - position.initial_stop)
    if position.side is Side.BUY:
        stop_hit = float(row["low"]) <= position.stop
        target_hit = float(row["high"]) >= position.target
        if stop_hit:
            return min(position.stop, float(row["open"])) - slip, (
                "AMBIGUOUS_STOP_FIRST" if target_hit else ("BREAKEVEN" if position.breakeven_armed else "STOP_LOSS")
            )
        if target_hit:
            return position.target - slip, "TAKE_PROFIT"
        if float(row["high"]) >= position.entry + 2.0 * risk_distance:
            position.stop = position.entry
            position.breakeven_armed = True
    else:
        ask_open = float(row["open"]) + spread
        ask_high = float(row["high"]) + spread
        ask_low = float(row["low"]) + spread
        stop_hit = ask_high >= position.stop
        target_hit = ask_low <= position.target
        if stop_hit:
            return max(position.stop, ask_open) + slip, (
                "AMBIGUOUS_STOP_FIRST" if target_hit else ("BREAKEVEN" if position.breakeven_armed else "STOP_LOSS")
            )
        if target_hit:
            return position.target + slip, "TAKE_PROFIT"
        if ask_low <= position.entry - 2.0 * risk_distance:
            position.stop = position.entry
            position.breakeven_armed = True
    return None


def _summaries(
    symbols: tuple[str, ...], trades: list[ReplayTrade], starting_equity: float,
    ending_equity: float, maximum_drawdown: float, status_counts: dict[str, Counter[str]],
    period_start: int, period_end: int, settings: Settings, commission_per_lot_side: float,
    slippage_points: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for symbol in (*symbols, "PORTFOLIO"):
        selected = trades if symbol == "PORTFOLIO" else [trade for trade in trades if trade.symbol == symbol]
        profits = [trade.net_profit for trade in selected]
        wins = [value for value in profits if value > 0]
        losses = [-value for value in profits if value < 0]
        sample_state = "BACKTESTED" if len(selected) >= 30 else "INSUFFICIENT_BACKTEST_SAMPLE"
        rows.append({
            "symbol": symbol,
            "state": sample_state,
            "period_start_utc": _iso(period_start),
            "period_end_utc": _iso(period_end),
            "starting_equity": starting_equity,
            "ending_equity": ending_equity if symbol == "PORTFOLIO" else None,
            "net_profit": sum(profits),
            "return_pct_on_starting_equity": sum(profits) / starting_equity * 100.0,
            "trades": len(selected),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate_pct": len(wins) / len(selected) * 100.0 if selected else 0.0,
            "profit_factor": sum(wins) / sum(losses) if losses else (999.0 if wins else 0.0),
            "expectancy": sum(profits) / len(selected) if selected else 0.0,
            "average_r": sum(trade.r_multiple for trade in selected) / len(selected) if selected else 0.0,
            "max_closed_equity_drawdown": maximum_drawdown if symbol == "PORTFOLIO" else None,
            "max_closed_equity_drawdown_pct": maximum_drawdown / starting_equity * 100.0 if symbol == "PORTFOLIO" else None,
            "signal_status_counts": dict(status_counts.get(symbol, {})) if symbol != "PORTFOLIO" else {},
            "risk_cap_pct": settings.max_risk_fraction * 100.0,
            "daily_maximum_entries": settings.liquidity_max_trades_per_day,
            "daily_target_pct_of_active_capital": settings.liquidity_daily_target_fraction * 100.0,
            "commission_per_lot_per_side": commission_per_lot_side,
            "slippage_points_per_fill": slippage_points,
        })
    return rows


def run_backtest(
    execution: MT5ExecutionAgent,
    settings: Settings,
    m3_bars: int,
    starting_equity: float,
    commission_per_lot_side: float,
    slippage_points: float,
    *,
    histories: dict[str, dict[str, np.ndarray]] | None = None,
    engine_parameters: dict[str, float | int] | None = None,
) -> tuple[list[ReplayTrade], list[dict[str, object]]]:
    mt5 = execution.mt5
    if histories is None:
        histories = load_histories(execution, settings, m3_bars)

    parameters: dict[str, float | int] = {
        "minimum_rrr": settings.liquidity_min_rrr,
        "minimum_touches": settings.liquidity_min_touches,
        "volume_expansion": settings.liquidity_volume_expansion,
        "momentum_body_fraction": settings.liquidity_momentum_body_fraction,
    }
    parameters.update(engine_parameters or {})
    engine = LiquidityBreakoutEngine(
        **parameters,
    )
    event_rows: dict[int, list[tuple[str, Any, int]]] = defaultdict(list)
    for symbol, history in histories.items():
        for index in range(21, len(history["m3"])):
            event_rows[int(history["m3"][index]["time"])].append((symbol, history["m3"][index], index))
    if not event_rows:
        raise RuntimeError("No replay events were available")

    equity = peak = starting_equity
    maximum_drawdown = 0.0
    trades: list[ReplayTrade] = []
    positions: dict[str, ReplayPosition] = {}
    pending: dict[str, PendingEntry] = {}
    daily_entries: Counter[str] = Counter()
    daily_net: Counter[str] = Counter()
    status_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for timestamp in sorted(event_rows):
        for symbol, row, index in sorted(event_rows[timestamp], key=lambda item: item[0]):
            day = _day(timestamp, settings.liquidity_daily_timezone)
            if symbol in pending:
                planned, plan_status = _entry_plan(
                    mt5, settings, symbol, pending.pop(symbol).decision, row, equity, slippage_points
                )
                status_counts[symbol][plan_status] += 1
                if planned is not None:
                    positions[symbol] = planned
                    daily_entries[day] += 1

            if symbol in positions:
                position = positions[symbol]
                event = _position_event(
                    position, row, float(mt5.symbol_info(symbol).point), slippage_points
                )
                if event is not None:
                    exit_price, reason = event
                    gross = _profit(mt5, symbol, position.side, position.volume, position.entry, exit_price)
                    commission = 2.0 * commission_per_lot_side * position.volume
                    net = gross - commission
                    equity += net
                    peak = max(peak, equity)
                    maximum_drawdown = max(maximum_drawdown, peak - equity)
                    daily_net[day] += net
                    trades.append(ReplayTrade(
                        symbol, position.side.value, _iso(position.entry_time), _iso(timestamp),
                        position.volume, position.entry, exit_price, position.initial_stop,
                        position.target, reason, gross, commission, net,
                        net / position.initial_risk_usd, equity,
                    ))
                    del positions[symbol]

            if symbol in positions or symbol in pending:
                continue
            target_profit = min(equity, settings.liquidity_daily_active_capital) * settings.liquidity_daily_target_fraction
            lock = daily_lock_reason(
                daily_entries[day], daily_net[day], target_profit,
                settings.liquidity_max_trades_per_day,
            )
            if lock:
                status_counts[symbol][lock] += 1
                continue
            history = histories[symbol]
            h4 = _closed_slice(history["h4"], timestamp, 14_400, 96)
            m15 = _closed_slice(history["m15"], timestamp, 900, 40)
            m3 = history["m3"][max(0, index - 40) : index + 1]
            decision = engine.evaluate(symbol, h4, m15, m3, has_position=False)
            status_counts[symbol][decision.trade_status] += 1
            if decision.side is not None:
                pending[symbol] = PendingEntry(decision, timestamp)

    for symbol, position in list(positions.items()):
        row = histories[symbol]["m3"][-1]
        point = float(mt5.symbol_info(symbol).point)
        spread = float(row["spread"]) * point if "spread" in (row.dtype.names or ()) else 0.0
        exit_price = float(row["close"]) if position.side is Side.BUY else float(row["close"]) + spread
        gross = _profit(mt5, symbol, position.side, position.volume, position.entry, exit_price)
        commission = 2.0 * commission_per_lot_side * position.volume
        net = gross - commission
        equity += net
        trades.append(ReplayTrade(
            symbol, position.side.value, _iso(position.entry_time), _iso(int(row["time"])),
            position.volume, position.entry, exit_price, position.initial_stop, position.target,
            "END_OF_TEST", gross, commission, net, net / position.initial_risk_usd, equity,
        ))

    start = min(int(history["m3"][21]["time"]) for history in histories.values())
    end = max(int(history["m3"][-1]["time"]) for history in histories.values())
    summaries = _summaries(
        settings.symbols, trades, starting_equity, equity, maximum_drawdown,
        status_counts, start, end, settings, commission_per_lot_side, slippage_points,
    )
    return trades, summaries


def load_histories(
    execution: MT5ExecutionAgent, settings: Settings, m3_bars: int,
) -> dict[str, dict[str, np.ndarray]]:
    mt5 = execution.mt5
    histories: dict[str, dict[str, np.ndarray]] = {}
    for symbol in settings.symbols:
        histories[symbol] = {
            "m3": _fetch_closed(mt5, symbol, mt5.TIMEFRAME_M3, m3_bars),
            "m15": _fetch_closed(
                mt5, symbol, mt5.TIMEFRAME_M15, max(500, m3_bars // 5 + 120)
            ),
            "h4": _fetch_closed(
                mt5, symbol, mt5.TIMEFRAME_H4, max(200, m3_bars // 80 + 120)
            ),
        }
        if (
            len(histories[symbol]["m3"]) < 100
            or len(histories[symbol]["m15"]) < 30
            or len(histories[symbol]["h4"]) < 44
        ):
            raise RuntimeError(
                f"{symbol} does not have enough synchronized M3/M15/H4 history"
            )
    return histories


def write_report(report_dir: Path, trades: list[ReplayTrade], summaries: list[dict[str, object]]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "scope": "chronological closed-bar replay of the opt-in liquidity-breakout strategy",
        "evidence_class": "HISTORICAL_BACKTEST_NOT_FORWARD_EVIDENCE",
        "limitations": [
            "Historical M3 OHLC and tick volume cannot reproduce tick ordering or executable liquidity.",
            "Entries occur at the next M3 open with historical spread and configured adverse slippage.",
            "If stop and target occur in one bar, the stop is applied first.",
            "A 2R breakeven stop becomes active only after the triggering M3 bar closes.",
            "Margin competition and cross-asset intrabar ordering are not modeled.",
            "Results do not establish profitability and are excluded from DEMO forward evidence.",
        ],
        "summaries": summaries,
    }
    (report_dir / "liquidity_backtest.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    with (report_dir / "liquidity_backtest_trades.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ReplayTrade.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(trade) for trade in trades)
    table_headers = list(summaries[0])
    summary_head = "".join(f"<th>{html.escape(key)}</th>" for key in table_headers)
    summary_rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(row[key]))}</td>" for key in table_headers) + "</tr>"
        for row in summaries
    )
    trade_rows = "".join(
        f"<tr><td>{trade.entry_time_utc}</td><td>{trade.symbol}</td><td>{trade.side}</td>"
        f"<td>{trade.volume}</td><td>{trade.exit_reason}</td><td>{trade.net_profit:.2f}</td>"
        f"<td>{trade.r_multiple:.2f}</td></tr>" for trade in trades
    )
    document = f"""<!doctype html><meta charset='utf-8'><title>Liquidity Breakout Backtest</title>
<style>body{{font:14px Segoe UI,Arial;margin:32px}}table{{border-collapse:collapse;font-size:12px}}th,td{{border:1px solid #ccc;padding:6px;text-align:right}}th{{background:#eee}}.warning{{background:#fff3cd;padding:12px}}</style>
<h1>CryptoAgent Liquidity Breakout Backtest</h1>
<p class='warning'>Historical OHLC replay only—not DEMO forward evidence or proof of future returns. Ambiguous bars are resolved stop-first.</p>
<table><thead><tr>{summary_head}</tr></thead><tbody>{summary_rows}</tbody></table>
<h2>Trades</h2><table><tr><th>Entry UTC</th><th>Symbol</th><th>Side</th><th>Volume</th><th>Exit</th><th>Net</th><th>R</th></tr>{trade_rows}</table>"""
    (report_dir / "liquidity_backtest.html").write_text(document, encoding="utf-8")


def append_experiment(
    ledger: Path,
    settings: Settings,
    m3_bars: int,
    starting_equity: float,
    commission_per_lot_side: float,
    slippage_points: float,
    summaries: list[dict[str, object]],
) -> Path:
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    configuration = {
        "minimum_rrr": settings.liquidity_min_rrr,
        "minimum_touches": settings.liquidity_min_touches,
        "volume_expansion": settings.liquidity_volume_expansion,
        "momentum_body_fraction": settings.liquidity_momentum_body_fraction,
        "maximum_trades_per_day": settings.liquidity_max_trades_per_day,
        "daily_active_capital": settings.liquidity_daily_active_capital,
        "daily_target_fraction": settings.liquidity_daily_target_fraction,
        "daily_timezone": settings.liquidity_daily_timezone,
        "m3_bars": m3_bars,
        "starting_equity": starting_equity,
        "commission_per_lot_side": commission_per_lot_side,
        "slippage_points_per_fill": slippage_points,
    }
    configuration_hash = hashlib.sha256(
        json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    record = {
        "experiment_id": hashlib.sha256(
            f"{generated_at}:{configuration_hash}".encode("utf-8")
        ).hexdigest()[:16],
        "generated_at": generated_at,
        "strategy_id": "liquidity_breakout",
        "timeframes": ["H4", "M15", "M3"],
        "configuration_hash": configuration_hash,
        "configuration": configuration,
        "evidence_class": "HISTORICAL_BACKTEST_NOT_FORWARD_EVIDENCE",
        "summaries": summaries,
        "result": "RESEARCH_ONLY_INSUFFICIENT_STABLE_DEVELOPMENT_EVIDENCE",
        "promotion_eligible": False,
        "promotion_blockers": [
            "MINIMUM_TRADES_PER_SEGMENT_NOT_PROVEN",
            "WALK_FORWARD_STABILITY_NOT_PROVEN",
            "UNTOUCHED_TEST_NOT_PROVEN",
            "FORWARD_DEMO_NOT_STARTED",
        ],
        "routing_changed": False,
        "forward_demo_status": "NOT_STARTED",
    }
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, allow_nan=False) + "\n")
    return ledger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m3-bars", type=int, default=30_000)
    parser.add_argument("--starting-equity", type=float, default=1_000.0)
    parser.add_argument("--commission-per-lot-side", type=float, default=3.0)
    parser.add_argument("--slippage-points", type=float, default=10.0)
    args = parser.parse_args()
    if args.m3_bars < 500 or args.starting_equity <= 0:
        raise ValueError("m3-bars must be >= 500 and starting equity must be positive")
    settings = SETTINGS
    settings.validate()
    execution = MT5ExecutionAgent(settings)
    execution.connect()
    try:
        trades, summaries = run_backtest(
            execution, settings, args.m3_bars, args.starting_equity,
            args.commission_per_lot_side, args.slippage_points,
        )
    finally:
        execution.shutdown()
    write_report(settings.report_dir, trades, summaries)
    ledger = append_experiment(
        Path("research") / "liquidity_experiments.jsonl",
        settings,
        args.m3_bars,
        args.starting_equity,
        args.commission_per_lot_side,
        args.slippage_points,
        summaries,
    )
    print(json.dumps(summaries, indent=2, allow_nan=False))
    print(ledger)
    print(settings.report_dir / "liquidity_backtest.html")


if __name__ == "__main__":
    main()
