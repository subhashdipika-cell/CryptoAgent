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


@dataclass(frozen=True, slots=True)
class CompositeFold:
    origin_time: int
    direction: str
    confidence: float
    m15_edge_bps: float
    h1_edge_bps: float
    actual_return_bps: float


def _profit_factor(returns: list[float]) -> float:
    wins = sum(value for value in returns if value > 0)
    losses = -sum(value for value in returns if value < 0)
    return wins / losses if losses else (999.0 if wins else 0.0)


def composite_walk_forward(
    symbol: str,
    m15_rates: np.ndarray,
    h1_rates: np.ndarray,
    settings: Settings,
    max_folds: int = 80,
) -> list[CompositeFold]:
    """Reproduce the live M15 plus latest fully closed H1 chronology."""
    m15 = rates_to_ohlc(m15_rates)
    h1 = rates_to_ohlc(h1_rates)
    spec = ASSET_SPECS[asset_key(symbol)]
    first_origin = spec.context + spec.minimum_samples + settings.prediction_length
    last_origin = len(m15) - settings.prediction_length - 1
    origins = np.unique(np.linspace(first_origin, last_origin, max_folds, dtype=int))
    engine = DedicatedAssetForecastEngine(settings)
    folds: list[CompositeFold] = []
    for origin in origins:
        h1_last = int(np.searchsorted(h1["time"], int(m15["time"][origin]) - 2700, side="right") - 1)
        if h1_last < spec.context + spec.minimum_samples:
            continue
        m15_training = m15[max(0, origin - settings.bar_count + 1) : origin + 1]
        h1_training = h1[max(0, h1_last - settings.bar_count + 1) : h1_last + 1]
        m15_prediction = engine.forecast(symbol, m15_training, "15min")
        h1_prediction = engine.forecast(symbol, h1_training, "1h")
        if m15_prediction.direction != h1_prediction.direction:
            continue
        actual = math.log(
            float(m15["close"][origin + settings.prediction_length]) / float(m15["close"][origin])
        ) * 10_000.0
        folds.append(
            CompositeFold(
                int(m15["time"][origin]), m15_prediction.direction,
                0.5 * (m15_prediction.probability + h1_prediction.probability),
                m15_prediction.edge_bps, h1_prediction.edge_bps, actual,
            )
        )
    return folds


def calibrate_policy(symbol: str, folds: list[CompositeFold]) -> tuple[dict[str, object], dict[str, object]]:
    """Select thresholds on early folds and enable only if later folds pass."""
    spec = ASSET_SPECS[asset_key(symbol)]
    split = max(1, int(len(folds) * 0.65))
    calibration, holdout = folds[:split], folds[split:]
    best: tuple[float, float, float, list[float]] | None = None
    for confidence in (0.52, 0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70):
        for m15_edge in (spec.minimum_edge_bps, spec.minimum_edge_bps * 1.5, spec.minimum_edge_bps * 2.0):
            for h1_edge in (spec.minimum_edge_bps, spec.minimum_edge_bps * 1.5, spec.minimum_edge_bps * 2.0):
                selected = [
                    fold for fold in calibration
                    if fold.confidence >= confidence
                    and abs(fold.m15_edge_bps) >= m15_edge
                    and abs(fold.h1_edge_bps) >= h1_edge
                ]
                returns = [
                    (1.0 if fold.direction == "BULLISH" else -1.0) * fold.actual_return_bps
                    - spec.round_trip_cost_bps for fold in selected
                ]
                if len(returns) >= 5 and (best is None or sum(returns) > sum(best[3])):
                    best = (confidence, m15_edge, h1_edge, returns)
    if best is None:
        best = (0.70, spec.minimum_edge_bps * 2, spec.minimum_edge_bps * 2, [])
    confidence, m15_edge, h1_edge, calibration_returns = best
    selected_holdout = [
        fold for fold in holdout
        if fold.confidence >= confidence
        and abs(fold.m15_edge_bps) >= m15_edge
        and abs(fold.h1_edge_bps) >= h1_edge
    ]
    holdout_returns = [
        (1.0 if fold.direction == "BULLISH" else -1.0) * fold.actual_return_bps
        - spec.round_trip_cost_bps for fold in selected_holdout
    ]
    correct = sum(
        (fold.direction == "BULLISH") == (fold.actual_return_bps >= 0)
        for fold in selected_holdout
    )
    accuracy = correct / len(selected_holdout) if selected_holdout else 0.0
    profit_factor = _profit_factor(holdout_returns)
    enabled = (
        len(selected_holdout) >= 5 and accuracy >= 0.52
        and sum(holdout_returns) > 0 and profit_factor >= 1.10
    )
    policy = {
        "symbol": symbol,
        "model_name": f"{asset_key(symbol)}-DirectRidge",
        "decision_mode": "M15_H1",
        "enabled": enabled,
        "approved": False,
        "confidence_threshold": confidence,
        "m15_edge_bps": m15_edge,
        "h1_edge_bps": h1_edge,
        "calibration_trades": len(calibration_returns),
        "holdout_trades": len(holdout_returns),
        "holdout_net_bps": sum(holdout_returns),
        "holdout_profit_factor": profit_factor,
    }
    diagnostics = {
        **policy,
        "composite_agreement_folds": len(folds),
        "holdout_direction_accuracy": accuracy,
        "deployment": "DEMO_ELIGIBLE" if enabled else "SHADOW_ONLY",
    }
    return policy, diagnostics


def calibrate_h1_policy(
    symbol: str, folds: list[FoldResult]
) -> tuple[dict[str, object], dict[str, object]]:
    """Select an H1-only threshold early and gate it on untouched later folds."""
    spec = ASSET_SPECS[asset_key(symbol)]
    split = max(1, int(len(folds) * 0.65))
    calibration, holdout = folds[:split], folds[split:]
    best: tuple[float, float, list[float]] | None = None
    for confidence in (
        0.52, 0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70,
        0.75, 0.80, 0.85, 0.90,
    ):
        for h1_edge in (
            spec.minimum_edge_bps,
            spec.minimum_edge_bps * 1.5,
            spec.minimum_edge_bps * 2.0,
            spec.minimum_edge_bps * 3.0,
        ):
            selected = [
                fold for fold in calibration
                if fold.confidence >= confidence
                and abs(fold.predicted_edge_bps) >= h1_edge
            ]
            returns = [
                (1.0 if fold.predicted_edge_bps >= 0 else -1.0)
                * fold.actual_return_bps
                - spec.round_trip_cost_bps
                for fold in selected
            ]
            if len(returns) >= 10 and (best is None or sum(returns) > sum(best[2])):
                best = (confidence, h1_edge, returns)
    if best is None:
        best = (0.70, spec.minimum_edge_bps * 2.0, [])
    confidence, h1_edge, calibration_returns = best
    selected_holdout = [
        fold for fold in holdout
        if fold.confidence >= confidence
        and abs(fold.predicted_edge_bps) >= h1_edge
    ]
    holdout_returns = [
        (1.0 if fold.predicted_edge_bps >= 0 else -1.0)
        * fold.actual_return_bps
        - spec.round_trip_cost_bps
        for fold in selected_holdout
    ]
    correct = sum(fold.direction_correct for fold in selected_holdout)
    accuracy = correct / len(selected_holdout) if selected_holdout else 0.0
    profit_factor = _profit_factor(holdout_returns)
    enabled = (
        len(selected_holdout) >= 5
        and accuracy >= 0.52
        and sum(holdout_returns) > 0
        and profit_factor >= 1.10
    )
    policy = {
        "symbol": symbol,
        "model_name": f"{asset_key(symbol)}-DirectRidge",
        "decision_mode": "H1_ONLY",
        "enabled": enabled,
        "approved": False,
        "confidence_threshold": confidence,
        "m15_edge_bps": 0.0,
        "h1_edge_bps": h1_edge,
        "calibration_trades": len(calibration_returns),
        "holdout_trades": len(holdout_returns),
        "holdout_net_bps": sum(holdout_returns),
        "holdout_profit_factor": profit_factor,
    }
    diagnostics = {
        **policy,
        "h1_folds": len(folds),
        "holdout_direction_accuracy": accuracy,
        "deployment": "DEMO_ELIGIBLE" if enabled else "SHADOW_ONLY",
    }
    return policy, diagnostics


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


def write_reports(
    summaries: list[dict[str, object]], folds: dict[str, list[FoldResult]],
    policies: list[dict[str, object]], policy_diagnostics: list[dict[str, object]],
    settings: Settings,
) -> None:
    settings.report_dir.mkdir(parents=True, exist_ok=True)
    settings.candidate_policy_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "methodology": "anchored walk-forward; completed bars only; five-bar horizon; no execution fills",
        "promotion_policy": "PASS is advisory; every candidate remains SHADOW_ONLY until explicitly approved",
        "foundation_candidates": candidate_inventory(settings.ttm_model_path, settings.timesfm_model_path),
        "results": summaries,
        "complete_decision_policy": policy_diagnostics,
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
    settings.candidate_policy_path.write_text(
        json.dumps(
            {
                "methodology": "chronological 65% calibration and 35% untouched holdout",
                "policies": policies,
            },
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    table_rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row.values()) + "</tr>"
        for row in summaries
    )
    headers = "".join(f"<th>{html.escape(key)}</th>" for key in summaries[0])
    policy_headers = "".join(f"<th>{html.escape(key)}</th>" for key in policy_diagnostics[0])
    policy_rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row.values()) + "</tr>"
        for row in policy_diagnostics
    )
    candidate_rows = "".join(
        f"<li>{html.escape(str(item['candidate']))}: staged={item['staged']}, runtime={item['runtime_installed']} &mdash; {html.escape(str(item['role']))}</li>"
        for item in payload["foundation_candidates"]
    )
    document = f"""<!doctype html><meta charset='utf-8'><title>CryptoAgent Predictive Validation</title>
<style>body{{font:14px Segoe UI,Arial;margin:32px;max-width:1200px}}table{{border-collapse:collapse}}th,td{{border:1px solid #ccc;padding:8px;text-align:right}}th{{background:#eee}}.warning{{padding:12px;background:#fff3cd}}</style>
<h1>CryptoAgent Predictive Validation</h1><p class='warning'>Research evidence only. PASS does not enable trading; all candidates remain SHADOW_ONLY.</p>
<p>Anchored walk-forward testing on completed MT5 bars, five-bar horizon. Net returns subtract configured cost assumptions but do not reproduce tick-level spread, slippage, or fills.</p>
<table><thead><tr>{headers}</tr></thead><tbody>{table_rows}</tbody></table>
<h2>Complete M15 + H1 decision-policy holdout</h2>
<table><thead><tr>{policy_headers}</tr></thead><tbody>{policy_rows}</tbody></table>
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
        datasets: dict[str, dict[str, tuple[str, np.ndarray]]] = {}
        for symbol in settings.symbols:
            datasets[symbol] = {}
            for timeframe, label, frequency in (
                (execution.mt5.TIMEFRAME_M15, "M15", "15min"),
                (execution.mt5.TIMEFRAME_H1, "H1", "1h"),
            ):
                datasets[symbol][label] = (
                    frequency, execution.bars(symbol, timeframe, args.bars)
                )
    finally:
        execution.shutdown()
    all_folds: dict[str, list[FoldResult]] = {}
    summaries: list[dict[str, object]] = []
    policies: list[dict[str, object]] = []
    policy_diagnostics: list[dict[str, object]] = []
    for symbol, by_timeframe in datasets.items():
        for label, (frequency, rates) in by_timeframe.items():
            key = f"{symbol}-{label}"
            max_folds = 360 if asset_key(symbol) == "BTC" and label == "H1" else 60
            all_folds[key] = walk_forward(
                symbol, rates, frequency, settings, max_folds=max_folds
            )
            summaries.append(summarize(symbol, label, all_folds[key], settings))
        if asset_key(symbol) == "BTC":
            policy, diagnostics = calibrate_h1_policy(
                symbol, all_folds[f"{symbol}-H1"]
            )
        else:
            composite = composite_walk_forward(
                symbol, by_timeframe["M15"][1], by_timeframe["H1"][1], settings
            )
            policy, diagnostics = calibrate_policy(symbol, composite)
        policies.append(policy)
        policy_diagnostics.append(diagnostics)
    write_reports(summaries, all_folds, policies, policy_diagnostics, settings)
    print(settings.report_dir / "predictive_validation.html")


if __name__ == "__main__":
    main()
