"""Counterfactual trailing-stop comparison for the liquidity backtest.

This module deliberately does not modify routing. It reuses the deployed signal,
entry, sizing, daily-lock, and cost model while swapping only stop management.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

import liquidity_backtest as replay
from config import SETTINGS, Settings
from execution_agent import MT5ExecutionAgent, Side


@dataclass(frozen=True, slots=True)
class TrailingVariant:
    name: str
    one_r_stop: float | None
    one_point_five_r_stop: float | None
    two_r_trailing_distance: float | None
    description: str


VARIANTS = (
    TrailingVariant(
        "BASELINE_2R_BREAKEVEN", None, None, None,
        "Original stop until 2R, then breakeven from the next completed M3 bar.",
    ),
    TrailingVariant(
        "BREAKEVEN_AT_1R", 0.0, None, None,
        "Original stop until 1R, then breakeven from the next completed M3 bar.",
    ),
    TrailingVariant(
        "STAGED_1R_TRAIL", -0.25, 0.0, 1.0,
        "At 1R cap risk at -0.25R, at 1.5R move to breakeven, and after 2R trail 1R behind the best completed-M3 excursion.",
    ),
)


def fetch_closed_paginated(mt5: Any, symbol: str, timeframe: int, count: int) -> np.ndarray:
    """Read MT5 history in accepted 50k pages and return chronological unique bars."""
    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"symbol_select({symbol}) failed: {mt5.last_error()}")
    pages: list[np.ndarray] = []
    remaining, offset = count, 1
    while remaining > 0:
        requested = min(50_000, remaining)
        values = mt5.copy_rates_from_pos(symbol, timeframe, offset, requested)
        if values is None:
            if pages:
                break
            raise RuntimeError(f"copy_rates_from_pos({symbol}) failed: {mt5.last_error()}")
        page = np.asarray(values)
        if len(page) == 0:
            break
        pages.append(page)
        if len(page) < requested:
            break
        remaining -= len(page)
        offset += len(page)
    if not pages:
        raise RuntimeError(f"{symbol} returned no closed bars")
    combined = np.concatenate(pages)
    combined = combined[np.argsort(combined["time"])]
    _, indices = np.unique(combined["time"], return_index=True)
    return combined[np.sort(indices)]


def make_position_event(variant: TrailingVariant) -> Callable[..., tuple[float, str] | None]:
    best_excursion: dict[int, float] = {}

    def position_event(
        position: replay.ReplayPosition,
        row: Any,
        point: float,
        slippage_points: float,
    ) -> tuple[float, str] | None:
        spread = float(row["spread"]) * point if "spread" in (row.dtype.names or ()) else 0.0
        slip = slippage_points * point
        risk = abs(position.entry - position.initial_stop)
        key = id(position)
        if position.side is Side.BUY:
            stop_hit = float(row["low"]) <= position.stop
            target_hit = float(row["high"]) >= position.target
            if stop_hit:
                reason = _stop_reason(position)
                best_excursion.pop(key, None)
                return min(position.stop, float(row["open"])) - slip, (
                    "AMBIGUOUS_STOP_FIRST" if target_hit else reason
                )
            if target_hit:
                best_excursion.pop(key, None)
                return position.target - slip, "TAKE_PROFIT"
            favorable = max(best_excursion.get(key, 0.0), float(row["high"]) - position.entry)
            best_excursion[key] = favorable
            candidate = _candidate_stop(variant, position.entry, risk, favorable / risk, True)
            if candidate is not None and candidate > position.stop:
                position.stop = candidate
                position.breakeven_armed = candidate >= position.entry
        else:
            ask_open = float(row["open"]) + spread
            ask_high = float(row["high"]) + spread
            ask_low = float(row["low"]) + spread
            stop_hit = ask_high >= position.stop
            target_hit = ask_low <= position.target
            if stop_hit:
                reason = _stop_reason(position)
                best_excursion.pop(key, None)
                return max(position.stop, ask_open) + slip, (
                    "AMBIGUOUS_STOP_FIRST" if target_hit else reason
                )
            if target_hit:
                best_excursion.pop(key, None)
                return position.target + slip, "TAKE_PROFIT"
            favorable = max(best_excursion.get(key, 0.0), position.entry - ask_low)
            best_excursion[key] = favorable
            candidate = _candidate_stop(variant, position.entry, risk, favorable / risk, False)
            if candidate is not None and candidate < position.stop:
                position.stop = candidate
                position.breakeven_armed = candidate <= position.entry
        return None

    return position_event


def _candidate_stop(
    variant: TrailingVariant, entry: float, risk: float, favorable_r: float, is_buy: bool
) -> float | None:
    locked_r: float | None = None
    if variant.one_r_stop is not None and favorable_r >= 1.0:
        locked_r = variant.one_r_stop
    if variant.one_point_five_r_stop is not None and favorable_r >= 1.5:
        locked_r = max(locked_r if locked_r is not None else -1.0, variant.one_point_five_r_stop)
    if variant.name == "BASELINE_2R_BREAKEVEN" and favorable_r >= 2.0:
        locked_r = 0.0
    if variant.two_r_trailing_distance is not None and favorable_r >= 2.0:
        locked_r = max(
            locked_r if locked_r is not None else -1.0,
            favorable_r - variant.two_r_trailing_distance,
        )
    if locked_r is None:
        return None
    return entry + risk * locked_r if is_buy else entry - risk * locked_r


def _stop_reason(position: replay.ReplayPosition) -> str:
    tolerance = abs(position.entry) * 1e-12 + 1e-9
    favorable = (
        position.stop - position.entry if position.side is Side.BUY
        else position.entry - position.stop
    )
    if favorable > tolerance:
        return "TRAILING_STOP"
    if abs(favorable) <= tolerance:
        return "BREAKEVEN"
    initial_distance = abs(position.entry - position.initial_stop)
    current_distance = abs(position.entry - position.stop)
    return "REDUCED_LOSS" if current_distance < initial_distance - tolerance else "STOP_LOSS"


def trade_excursions(
    trades: list[replay.ReplayTrade], histories: dict[tuple[str, int, int], np.ndarray],
    mt5: Any, requested_m3_bars: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for trade in trades:
        history = histories[(trade.symbol, mt5.TIMEFRAME_M3, requested_m3_bars)]
        entry_time = int(datetime.fromisoformat(trade.entry_time_utc).timestamp())
        exit_time = int(datetime.fromisoformat(trade.exit_time_utc).timestamp())
        start = int(np.searchsorted(history["time"], entry_time, side="left"))
        end = int(np.searchsorted(history["time"], exit_time, side="right"))
        bars = history[start:end]
        risk = abs(trade.entry - trade.initial_stop)
        point = float(mt5.symbol_info(trade.symbol).point)
        if len(bars) == 0 or risk <= 0:
            mfe_r = mae_r = 0.0
        elif trade.side == Side.BUY.value:
            mfe_r = max(0.0, (float(np.max(bars["high"])) - trade.entry) / risk)
            mae_r = max(0.0, (trade.entry - float(np.min(bars["low"]))) / risk)
        else:
            spreads = bars["spread"].astype(float) * point if "spread" in (bars.dtype.names or ()) else 0.0
            ask_high = bars["high"] + spreads
            ask_low = bars["low"] + spreads
            mfe_r = max(0.0, (trade.entry - float(np.min(ask_low))) / risk)
            mae_r = max(0.0, (float(np.max(ask_high)) - trade.entry) / risk)
        rows.append({
            **asdict(trade),
            "mfe_r": mfe_r,
            "mae_r": mae_r,
        })
    return rows


def _portfolio_summary(summaries: list[dict[str, object]]) -> dict[str, object]:
    return next(row for row in summaries if row["symbol"] == "PORTFOLIO")


def _selection(comparisons: list[dict[str, object]]) -> dict[str, object]:
    baseline = comparisons[0]
    if int(baseline["trades"]) < 30:
        return {
            "state": "INSUFFICIENT_COMPARISON_EVIDENCE",
            "minimum_required_trades": 30,
            "selected_for_demo": None,
            "routing_changed": False,
            "reason": "Fewer than 30 baseline trades; no trailing variant may change DEMO management.",
        }
    candidates = []
    for row in comparisons[1:]:
        retains_net = float(row["net_profit"]) >= max(0.0, float(baseline["net_profit"]) * 0.90)
        reduces_drawdown = float(row["max_closed_equity_drawdown"]) <= float(baseline["max_closed_equity_drawdown"]) * 0.85
        if retains_net and reduces_drawdown and float(row["expectancy"]) > 0 and float(row["profit_factor"]) >= 1.2:
            candidates.append(row)
    selected = max(candidates, key=lambda row: (float(row["net_profit"]), -float(row["max_closed_equity_drawdown"])), default=None)
    return {
        "state": "CANDIDATE_IDENTIFIED" if selected else "NO_TRAILING_IMPROVEMENT",
        "minimum_required_trades": 30,
        "selected_for_demo": selected["variant"] if selected else None,
        "routing_changed": False,
        "reason": "Selection is advisory; explicit implementation and DEMO verification remain required.",
    }


def write_comparison(
    report_dir: Path, comparisons: list[dict[str, object]], trade_rows: list[dict[str, object]],
    selection: dict[str, object], variants: tuple[TrailingVariant, ...],
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "scope": "counterfactual stop-management comparison using identical liquidity strategy entry logic",
        "evidence_class": "HISTORICAL_BACKTEST_NOT_FORWARD_EVIDENCE",
        "selection": selection,
        "variants": [asdict(variant) for variant in variants],
        "comparisons": comparisons,
        "limitations": [
            "M3 OHLC cannot determine tick order; stop/target collisions are stop-first.",
            "Stop changes become active only after the M3 bar that triggered them closes.",
            "MFE and MAE are bar extrema, not executable tick paths.",
            "Reports do not alter routing or count as reconciled DEMO forward evidence.",
        ],
    }
    (report_dir / "liquidity_trailing_comparison.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    if trade_rows:
        with (report_dir / "liquidity_trailing_comparison_trades.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(trade_rows[0]))
            writer.writeheader()
            writer.writerows(trade_rows)
    headers = list(comparisons[0])
    head = "".join(f"<th>{html.escape(key)}</th>" for key in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(row[key]))}</td>" for key in headers) + "</tr>"
        for row in comparisons
    )
    document = f"""<!doctype html><meta charset='utf-8'><title>Liquidity Trailing Comparison</title>
<style>body{{font:14px Segoe UI,Arial;margin:32px}}table{{border-collapse:collapse;font-size:12px}}th,td{{border:1px solid #ccc;padding:6px;text-align:right}}th{{background:#eee}}.warning{{background:#fff3cd;padding:12px}}</style>
<h1>Liquidity Trailing-Stop Comparison</h1><p class='warning'>{html.escape(str(selection['state']))}: {html.escape(str(selection['reason']))}</p>
<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"""
    (report_dir / "liquidity_trailing_comparison.html").write_text(document, encoding="utf-8")


def run_comparison(
    execution: MT5ExecutionAgent, settings: Settings, m3_bars: int,
    starting_equity: float, commission_per_lot_side: float, slippage_points: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    original_fetch, original_event = replay._fetch_closed, replay._position_event
    cache: dict[tuple[str, int, int], np.ndarray] = {}

    def cached_fetch(mt5: Any, symbol: str, timeframe: int, count: int) -> np.ndarray:
        key = (symbol, timeframe, count)
        if key not in cache:
            cache[key] = fetch_closed_paginated(mt5, symbol, timeframe, count)
        return cache[key]

    comparisons: list[dict[str, object]] = []
    all_trade_rows: list[dict[str, object]] = []
    try:
        replay._fetch_closed = cached_fetch
        for variant in VARIANTS:
            replay._position_event = make_position_event(variant)
            trades, summaries = replay.run_backtest(
                execution, settings, m3_bars, starting_equity,
                commission_per_lot_side, slippage_points,
            )
            portfolio = _portfolio_summary(summaries)
            excursions = trade_excursions(trades, cache, execution.mt5, m3_bars)
            losing_excursions = [row for row in excursions if float(row["net_profit"]) < 0]
            comparison = {
                "variant": variant.name,
                "state": portfolio["state"],
                "period_start_utc": portfolio["period_start_utc"],
                "period_end_utc": portfolio["period_end_utc"],
                "trades": portfolio["trades"],
                "net_profit": portfolio["net_profit"],
                "return_pct": portfolio["return_pct_on_starting_equity"],
                "win_rate_pct": portfolio["win_rate_pct"],
                "profit_factor": portfolio["profit_factor"],
                "expectancy": portfolio["expectancy"],
                "average_r": portfolio["average_r"],
                "max_closed_equity_drawdown": portfolio["max_closed_equity_drawdown"],
                "max_closed_equity_drawdown_pct": portfolio["max_closed_equity_drawdown_pct"],
                "losing_trades_reaching_1r_first": sum(float(row["mfe_r"]) >= 1.0 for row in losing_excursions),
                "full_stop_losses": sum(str(row["exit_reason"]) in {"STOP_LOSS", "AMBIGUOUS_STOP_FIRST"} for row in excursions),
            }
            comparisons.append(comparison)
            all_trade_rows.extend({"variant": variant.name, **row} for row in excursions)
    finally:
        replay._fetch_closed, replay._position_event = original_fetch, original_event
    selection = _selection(comparisons)
    return comparisons, all_trade_rows, selection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m3-bars", type=int, default=99_999)
    parser.add_argument("--starting-equity", type=float, default=1_000.0)
    parser.add_argument("--commission-per-lot-side", type=float, default=0.03)
    parser.add_argument("--slippage-points", type=float, default=10.0)
    args = parser.parse_args()
    if args.m3_bars < 500 or args.m3_bars > 99_999 or args.starting_equity <= 0:
        raise ValueError("m3-bars must be in [500, 99999] and starting equity must be positive")
    settings = SETTINGS
    settings.validate()
    execution = MT5ExecutionAgent(settings)
    execution.connect()
    try:
        comparisons, trades, selection = run_comparison(
            execution, settings, args.m3_bars, args.starting_equity,
            args.commission_per_lot_side, args.slippage_points,
        )
    finally:
        execution.shutdown()
    write_comparison(settings.report_dir, comparisons, trades, selection, VARIANTS)
    print(json.dumps({"selection": selection, "comparisons": comparisons}, indent=2, allow_nan=False))
    print(settings.report_dir / "liquidity_trailing_comparison.html")


if __name__ == "__main__":
    main()
