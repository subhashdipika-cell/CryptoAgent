"""Leakage-aware H4 liquidity-entry parameter search; never changes routing."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

from config import BASE_DIR, SETTINGS, Settings
from execution_agent import MT5ExecutionAgent
from liquidity_backtest import ReplayTrade, load_histories, run_backtest


TOUCHES = (2, 3)
ZONE_BARS = (12, 18, 24)
HISTORY_BARS = (72, 96)


def segment_metrics(profits: list[float], starting_equity: float) -> dict[str, float | int]:
    wins = [value for value in profits if value > 0]
    losses = [-value for value in profits if value < 0]
    equity = peak = starting_equity
    drawdown = 0.0
    for value in profits:
        equity += value
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    return {
        "trades": len(profits),
        "net_profit": sum(profits),
        "expectancy": sum(profits) / len(profits) if profits else 0.0,
        "profit_factor": sum(wins) / sum(losses) if losses else (999.0 if wins else 0.0),
        "max_drawdown_pct": drawdown / starting_equity * 100.0,
    }


def parameter_distance(left: dict[str, int], right: dict[str, int]) -> int:
    axes = (TOUCHES, ZONE_BARS, HISTORY_BARS)
    names = ("minimum_touches", "h4_zone_bars", "h4_history_bars")
    return sum(abs(axis.index(left[name]) - axis.index(right[name])) for axis, name in zip(axes, names))


def _passes_development(row: dict[str, object]) -> bool:
    train = row["calibration"]
    validation = row["walk_forward"]
    assert isinstance(train, dict) and isinstance(validation, dict)
    return (
        train["trades"] >= 10
        and validation["trades"] >= 5
        and train["expectancy"] > 0
        and validation["expectancy"] > 0
        and train["profit_factor"] >= 1.05
        and validation["profit_factor"] >= 1.10
        and validation["max_drawdown_pct"] <= 10.0
    )


def _trade_time(trade: ReplayTrade) -> int:
    return int(datetime.fromisoformat(trade.entry_time_utc).timestamp())


def _window_metrics(
    trades: list[ReplayTrade], start: int, end: int, starting_equity: float,
) -> dict[str, float | int]:
    return segment_metrics(
        [trade.net_profit for trade in trades if start <= _trade_time(trade) < end],
        starting_equity,
    )


def search(
    execution: MT5ExecutionAgent,
    settings: Settings,
    m3_bars: int,
    starting_equity: float,
    commission_per_lot_side: float,
    slippage_points: float,
) -> dict[str, object]:
    histories = load_histories(execution, settings, m3_bars)
    payload: dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evidence_class": "HISTORICAL_RESEARCH_NOT_FORWARD_EVIDENCE",
        "methodology": "50% calibration, 25% walk-forward selection, 25% untouched test revealed only for the selected stable configuration",
        "costs": {
            "commission_per_lot_side": commission_per_lot_side,
            "slippage_points_per_fill": slippage_points,
        },
        "symbols": {},
    }
    for symbol in settings.symbols:
        history = histories[symbol]
        start = int(history["m3"][21]["time"])
        end = int(history["m3"][-1]["time"]) + 1
        calibration_end = start + (end - start) // 2
        validation_end = start + 3 * (end - start) // 4
        symbol_settings = replace(settings, symbols=(symbol,))
        rows: list[dict[str, object]] = []
        for touches, zone_bars, history_bars in itertools.product(
            TOUCHES, ZONE_BARS, HISTORY_BARS
        ):
            parameters = {
                "minimum_touches": touches,
                "h4_zone_bars": zone_bars,
                "h4_history_bars": history_bars,
            }
            trades, _ = run_backtest(
                execution,
                symbol_settings,
                m3_bars,
                starting_equity,
                commission_per_lot_side,
                slippage_points,
                histories={symbol: history},
                engine_parameters=parameters,
            )
            rows.append(
                {
                    "parameters": parameters,
                    "calibration": _window_metrics(
                        trades, start, calibration_end, starting_equity
                    ),
                    "walk_forward": _window_metrics(
                        trades, calibration_end, validation_end, starting_equity
                    ),
                    "total_trade_count": len(trades),
                    "_trades": trades,
                }
            )
        development = [row for row in rows if _passes_development(row)]
        stable = [
            row
            for row in development
            if any(
                other is not row
                and parameter_distance(row["parameters"], other["parameters"]) == 1
                for other in development
            )
        ]
        selected = max(
            stable,
            key=lambda row: (
                row["walk_forward"]["expectancy"],
                row["walk_forward"]["profit_factor"],
                row["walk_forward"]["trades"],
            ),
            default=None,
        )
        public_rows = [
            {key: value for key, value in row.items() if key != "_trades"}
            for row in rows
        ]
        if selected is None:
            selection: dict[str, object] = {
                "state": "NO_STABLE_DEVELOPMENT_CANDIDATE",
                "routing_changed": False,
            }
        else:
            untouched = _window_metrics(
                selected["_trades"], validation_end, end, starting_equity
            )
            passed = (
                untouched["trades"] >= 30
                and untouched["expectancy"] > 0
                and untouched["profit_factor"] >= 1.20
                and untouched["max_drawdown_pct"] <= 5.0
            )
            selection = {
                "state": "UNTOUCHED_GATE_PASS" if passed else "UNTOUCHED_GATE_REJECTED",
                "parameters": selected["parameters"],
                "calibration": selected["calibration"],
                "walk_forward": selected["walk_forward"],
                "untouched_test": untouched,
                "routing_changed": False,
            }
        payload["symbols"][symbol] = {
            "period_start_utc": datetime.fromtimestamp(start, timezone.utc).isoformat(),
            "period_end_utc": datetime.fromtimestamp(end - 1, timezone.utc).isoformat(),
            "experiments": public_rows,
            "selection": selection,
        }
    return payload


def write_results(payload: dict[str, object], report_dir: Path) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "liquidity_parameter_search.json"
    report_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    ledger_dir = BASE_DIR / "research"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = ledger_dir / "strategy_experiments.jsonl"
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"experiment_id": digest, **payload}, allow_nan=False) + "\n")
    return report_path, ledger_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m3-bars", type=int, default=50_000)
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
        payload = search(
            execution,
            settings,
            args.m3_bars,
            args.starting_equity,
            args.commission_per_lot_side,
            args.slippage_points,
        )
    finally:
        execution.shutdown()
    report, ledger = write_results(payload, settings.report_dir)
    print(json.dumps({symbol: row["selection"] for symbol, row in payload["symbols"].items()}, indent=2))
    print(report)
    print(ledger)


if __name__ == "__main__":
    main()
