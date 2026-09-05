"""Asset-specific, locally fitted BTC and Gold directional forecasting.

These models are intentionally distinct from zero-shot foundation models.  Each
timeframe is fitted only on the selected asset's completed OHLC bars and predicts
the next five cumulative log returns directly.  No future bar is used in a fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt
from typing import Any, Iterable

import numpy as np

from config import Settings
from quant_engine import ForecastResult, rates_to_ohlc


@dataclass(frozen=True, slots=True)
class AssetModelSpec:
    asset: str
    context: int
    ridge_alpha: float
    minimum_samples: int
    minimum_edge_bps: float
    round_trip_cost_bps: float


ASSET_SPECS: dict[str, AssetModelSpec] = {
    "BTC": AssetModelSpec("BTC", 64, 12.0, 220, 10.0, 8.0),
    "XAU": AssetModelSpec("XAU", 96, 18.0, 220, 3.0, 3.0),
}


def asset_key(symbol: str) -> str:
    name = symbol.upper()
    if "BTC" in name:
        return "BTC"
    if "XAU" in name or "GOLD" in name:
        return "XAU"
    raise ValueError(f"no dedicated predictive specification for {symbol}")


def _series_features(ohlc: np.ndarray, context: int) -> tuple[np.ndarray, np.ndarray]:
    close = ohlc["close"].astype(np.float64)
    log_returns = np.diff(np.log(close))
    candle_range = (ohlc["high"] - ohlc["low"]) / close
    candle_body = (ohlc["close"] - ohlc["open"]) / close
    if len(log_returns) < context:
        raise ValueError(f"at least {context + 1} bars are required")
    lags = log_returns[-context:]
    momentum = np.array([np.sum(log_returns[-span:]) for span in (3, 6, 12, 24)])
    volatility = np.array([np.std(log_returns[-span:], ddof=1) for span in (6, 12, 24)])
    current = np.array([candle_range[-1], candle_body[-1]])
    return np.concatenate((lags, momentum, volatility, current)), log_returns


def supervised_matrix(
    rates: Iterable[Any], spec: AssetModelSpec, horizon: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Create leakage-free features and cumulative-return targets."""
    ohlc = rates_to_ohlc(rates)
    rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for origin in range(spec.context, len(ohlc) - horizon):
        feature, _ = _series_features(ohlc[: origin + 1], spec.context)
        base = float(ohlc["close"][origin])
        future = np.log(ohlc["close"][origin + 1 : origin + horizon + 1] / base)
        rows.append(feature)
        targets.append(future)
    if len(rows) < spec.minimum_samples:
        raise ValueError(
            f"{spec.asset} requires {spec.minimum_samples} training samples; found {len(rows)}"
        )
    return np.asarray(rows, dtype=np.float64), np.asarray(targets, dtype=np.float64)


@dataclass(slots=True)
class _RidgeState:
    x_mean: np.ndarray
    x_scale: np.ndarray
    coefficients: np.ndarray
    intercept: np.ndarray
    residual_scale: np.ndarray


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> _RidgeState:
    x_mean = x.mean(axis=0)
    x_scale = x.std(axis=0)
    x_scale[x_scale < 1e-10] = 1.0
    standardized = (x - x_mean) / x_scale
    y_mean = y.mean(axis=0)
    centered_y = y - y_mean
    gram = standardized.T @ standardized
    coefficients = np.linalg.solve(
        gram + np.eye(gram.shape[0], dtype=np.float64) * alpha,
        standardized.T @ centered_y,
    )
    residual = centered_y - standardized @ coefficients
    residual_scale = residual.std(axis=0, ddof=1)
    residual_scale[residual_scale < 1e-8] = 1e-8
    return _RidgeState(x_mean, x_scale, coefficients, y_mean, residual_scale)


def _predict(state: _RidgeState, feature: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    standardized = (feature - state.x_mean) / state.x_scale
    return state.intercept + standardized @ state.coefficients, state.residual_scale


class DedicatedAssetForecastEngine:
    """Maintains independent BTC/XAU models per timeframe on CPU."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._cache: dict[tuple[str, str, int], tuple[_RidgeState, AssetModelSpec]] = {}

    def forecast(self, symbol: str, rates: Iterable[Any], frequency: str) -> ForecastResult:
        ohlc = rates_to_ohlc(rates)
        spec = ASSET_SPECS[asset_key(symbol)]
        last_timestamp = int(ohlc["time"][-1])
        cache_key = (spec.asset, frequency, last_timestamp)
        cached = self._cache.get(cache_key)
        if cached is None:
            x, y = supervised_matrix(ohlc, spec, self.settings.prediction_length)
            state = _fit_ridge(x, y, spec.ridge_alpha)
            self._cache = {
                key: value for key, value in self._cache.items() if key[:2] != cache_key[:2]
            }
            self._cache[cache_key] = (state, spec)
        else:
            state, _ = cached
        feature, _ = _series_features(ohlc, spec.context)
        cumulative_returns, residual_scale = _predict(state, feature)
        last_close = float(ohlc["close"][-1])
        predictions = last_close * np.exp(cumulative_returns)
        edge_bps = float(cumulative_returns[-1] * 10_000.0)
        z_score = float(cumulative_returns[-1] / residual_scale[-1])
        bullish_probability = 0.5 * (1.0 + erf(z_score / sqrt(2.0)))
        direction = "BULLISH" if edge_bps >= 0 else "BEARISH"
        directional_confidence = bullish_probability if direction == "BULLISH" else 1.0 - bullish_probability
        if abs(edge_bps) < spec.minimum_edge_bps:
            directional_confidence = min(directional_confidence, 0.5)
        return ForecastResult(
            predictions=np.asarray(predictions, dtype=np.float32),
            direction=direction,
            probability=float(np.clip(directional_confidence, 0.5, 0.99)),
            normalized_slope=float(np.polyfit(np.arange(1, 6), predictions, 1)[0] / last_close),
            model_name=f"{spec.asset}-DirectRidge",
            edge_bps=edge_bps,
        )
