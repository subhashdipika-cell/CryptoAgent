"""Leakage-free walk-forward reports for the dedicated BTC and Gold models."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

import numpy as np

from asset_predictive_engine import ASSET_SPECS, DedicatedAssetForecastEngine, asset_key
from config import SETTINGS, Settings
from execution_agent import MT5ExecutionAgent
from foundation_backends import candidate_inventory
from quant_engine import rates_to_ohlc


@dataclass(frozen=True, slots=True)
class FoldResult:
    origin_time: int
    predicted_edge_bps: float
    confidence: float
    actual_return_bps: float
    direction_correct: bool
    traded: bool
    net_return_bps: float


def walk_forward(
    symbol: str,
    rates: np.ndarray,
    frequency: str,
    settings: Settings,
    max_folds: int = 60,
) -> list[FoldResult]:
    ohlc = rates_to_ohlc(rates)
    spec = ASSET_SPECS[asset_key(symbol)]
    first_origin = spec.context + spec.minimum_samples + settings.prediction_length
    last_origin = len(ohlc) - settings.prediction_length - 1
    if last_origin < first_origin:
        raise ValueError(f"{symbol} {frequency} has insufficient bars for walk-forward validation")
    origins = np.unique(
        np.linspace(first_origin, last_origin, min(max_folds, last_origin - first_origin + 1), dtype=int)
    )
    engine = DedicatedAssetForecastEngine(settings)
    folds: list[FoldResult] = []
    for origin in origins:
        training = ohlc[max(0, origin - settings.bar_count + 1) : origin + 1]
        prediction = engine.forecast(symbol, training, frequency)
        actual = math.log(float(ohlc["close"][origin + settings.prediction_length]) / float(ohlc["close"][origin]))
        actual_bps = actual * 10_000.0
        predicted_sign = 1.0 if prediction.edge_bps >= 0 else -1.0
        traded = prediction.probability >= 0.55 and abs(prediction.edge_bps) >= spec.minimum_edge_bps
        net = predicted_sign * actual_bps - spec.round_trip_cost_bps if traded else 0.0
        folds.append(
            FoldResult(
                int(ohlc["time"][origin]), prediction.edge_bps, prediction.probability,
                actual_bps, predicted_sign == (1.0 if actual_bps >= 0 else -1.0), traded, net,
            )
        )
    return folds


def summarize(symbol: str, timeframe: str, folds: list[FoldResult], settings: Settings) -> dict[str, object]:
    traded = [fold for fold in folds if fold.traded]
    wins = [fold.net_return_bps for fold in traded if fold.net_return_bps > 0]
    losses = [-fold.net_return_bps for fold in traded if fold.net_return_bps < 0]
    direction_accuracy = mean(float(fold.direction_correct) for fold in folds)
    profit_factor = sum(wins) / sum(losses) if losses else (999.0 if wins else 0.0)
    net_bps = sum(fold.net_return_bps for fold in traded)
    passed = (
        len(folds) >= settings.validation_min_folds
        and len(traded) >= 10
        and direction_accuracy >= 0.52
        and net_bps > 0
        and profit_factor >= 1.10
    )
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "model": f"{asset_key(symbol)}-DirectRidge",
        "folds": len(folds),
        "trades": len(traded),
        "direction_accuracy": direction_accuracy,
        "net_return_bps_after_assumed_costs": net_bps,
        "profit_factor": profit_factor,
        "validation_gate": "PASS" if passed else "FAIL",
        "deployment": "SHADOW_ONLY",
    }


def write_reports(summaries: list[dict[str, object]], folds: dict[str, list[FoldResult]], settings: Settings) -> None:
    settings.report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "methodology": "anchored walk-forward; completed bars only; five-bar horizon; no execution fills",
        "promotion_policy": "PASS is advisory; every candidate remains SHADOW_ONLY until explicitly approved",
        "foundation_candidates": candidate_inventory(settings.ttm_model_path, settings.timesfm_model_path),
        "results": summaries,
    }
    (settings.report_dir / "predictive_validation.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    with (settings.report_dir / "predictive_validation_folds.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["series", *FoldResult.__dataclass_fields__])
        for key, rows in folds.items():
            for row in rows:
                writer.writerow([key, *asdict(row).values()])
    table_rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row.values()) + "</tr>"
        for row in summaries
    )
    headers = "".join(f"<th>{html.escape(key)}</th>" for key in summaries[0])
    candidate_rows = "".join(
        f"<li>{html.escape(str(item['candidate']))}: staged={item['staged']}, runtime={item['runtime_installed']} &mdash; {html.escape(str(item['role']))}</li>"
        for item in payload["foundation_candidates"]
    )
    document = f"""<!doctype html><meta charset='utf-8'><title>CryptoAgent Predictive Validation</title>
<style>body{{font:14px Segoe UI,Arial;margin:32px;max-width:1200px}}table{{border-collapse:collapse}}th,td{{border:1px solid #ccc;padding:8px;text-align:right}}th{{background:#eee}}.warning{{padding:12px;background:#fff3cd}}</style>
<h1>CryptoAgent Predictive Validation</h1><p class='warning'>Research evidence only. PASS does not enable trading; all candidates remain SHADOW_ONLY.</p>
<p>Anchored walk-forward testing on completed MT5 bars, five-bar horizon. Net returns subtract configured cost assumptions but do not reproduce tick-level spread, slippage, or fills.</p>
<table><thead><tr>{headers}</tr></thead><tbody>{table_rows}</tbody></table>
<h2>Foundation candidates</h2><ul>{candidate_rows}</ul>"""
    (settings.report_dir / "predictive_validation.html").write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bars", type=int, default=SETTINGS.validation_bars)
    args = parser.parse_args()
    settings = SETTINGS
    settings.validate()
    execution = MT5ExecutionAgent(settings)
    execution.connect()
    try:
        datasets = []
        for symbol in settings.symbols:
            for timeframe, label, frequency in (
                (execution.mt5.TIMEFRAME_M15, "M15", "15min"),
                (execution.mt5.TIMEFRAME_H1, "H1", "1h"),
            ):
                datasets.append((symbol, label, frequency, execution.bars(symbol, timeframe, args.bars)))
    finally:
        execution.shutdown()
    all_folds: dict[str, list[FoldResult]] = {}
    summaries: list[dict[str, object]] = []
    for symbol, label, frequency, rates in datasets:
        key = f"{symbol}-{label}"
        all_folds[key] = walk_forward(symbol, rates, frequency, settings)
        summaries.append(summarize(symbol, label, all_folds[key], settings))
    write_reports(summaries, all_folds, settings)
    print(settings.report_dir / "predictive_validation.html")


if __name__ == "__main__":
    main()
