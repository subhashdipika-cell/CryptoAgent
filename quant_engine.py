"""CPU-bounded, strictly offline Chronos-2 forecasting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from config import Settings


OHLC_DTYPE = np.dtype(
    [("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"), ("close", "f8")]
)


@dataclass(frozen=True, slots=True)
class ForecastResult:
    predictions: np.ndarray
    direction: str
    probability: float
    normalized_slope: float
    model_name: str = "Chronos-2"
    edge_bps: float = 0.0


def rates_to_ohlc(rates: Iterable[Any]) -> np.ndarray:
    """Copy only required fields from MT5 records into a compact contiguous array."""
    source = np.asarray(rates)
    if source.size == 0:
        raise ValueError("MT5 returned no bars")
    required = {"time", "open", "high", "low", "close"}
    names = set(source.dtype.names or ())
    if not required.issubset(names):
        raise ValueError(f"bars are missing fields: {sorted(required - names)}")
    result = np.empty(source.size, dtype=OHLC_DTYPE)
    for name in required:
        result[name] = source[name]
    if not np.all(np.isfinite(result["close"])) or np.any(result["close"] <= 0):
        raise ValueError("close series contains invalid prices")
    return np.ascontiguousarray(result)


def close_prices(rates: Iterable[Any], max_bars: int = 500) -> np.ndarray:
    ohlc = rates_to_ohlc(rates)
    return np.ascontiguousarray(ohlc["close"][-max_bars:], dtype=np.float32)


def true_range_atr(rates: Iterable[Any], period: int = 14) -> float:
    ohlc = rates_to_ohlc(rates)
    if len(ohlc) < period + 1:
        raise ValueError(f"at least {period + 1} bars are required for ATR")
    high, low, previous_close = ohlc["high"][1:], ohlc["low"][1:], ohlc["close"][:-1]
    tr = np.maximum(high - low, np.maximum(np.abs(high - previous_close), np.abs(low - previous_close)))
    atr = float(np.mean(tr[-period:], dtype=np.float64))
    if not np.isfinite(atr) or atr <= 0:
        raise ValueError("ATR is not positive")
    return atr


def trend_from_predictions(last_close: float, predictions: np.ndarray) -> ForecastResult:
    values = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if values.size != 5 or not np.all(np.isfinite(values)) or last_close <= 0:
        raise ValueError("exactly five finite predictions and a positive last close are required")
    x = np.arange(1, 6, dtype=np.float64)
    slope = float(np.polyfit(x, values, 1)[0]) / last_close
    terminal_return = float(values[-1] - last_close) / last_close
    # Smoothly maps combined relative trend strength into [0, 1].
    bullish_probability = float(1.0 / (1.0 + np.exp(-80.0 * (0.6 * terminal_return + 0.4 * slope))))
    return ForecastResult(
        predictions=values.astype(np.float32),
        direction="BULLISH" if bullish_probability >= 0.5 else "BEARISH",
        probability=bullish_probability if bullish_probability >= 0.5 else 1.0 - bullish_probability,
        normalized_slope=slope,
        model_name="Chronos-2",
        edge_bps=terminal_return * 10_000.0,
    )


class ChronosForecastEngine:
    """Lazy-load one Chronos-2 model, on CPU, from local files only."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._pipeline: Any | None = None

    def _check_process_memory(self) -> None:
        """Fail if the Windows process working set exceeds the configured budget."""
        import os

        if os.name != "nt":
            return
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        process = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
            if counters.WorkingSetSize > self.settings.max_model_ram_bytes:
                raise MemoryError(
                    f"process working set {counters.WorkingSetSize / 1024**3:.2f} GiB exceeds "
                    f"{self.settings.max_model_ram_bytes / 1024**3:.2f} GiB budget"
                )

    def load(self) -> None:
        if self._pipeline is not None:
            return
        if not self.settings.model_path.is_dir():
            raise FileNotFoundError(
                f"Chronos model directory not found: {self.settings.model_path}. "
                "Download amazon/chronos-2 into this directory before enabling offline mode."
            )
        import torch
        from chronos import Chronos2Pipeline

        torch.set_num_threads(self.settings.torch_threads)
        torch.set_num_interop_threads(1)
        self._pipeline = Chronos2Pipeline.from_pretrained(
            str(self.settings.model_path),
            device_map="cpu",
            dtype=torch.float32,
            local_files_only=True,
        )
        self._check_process_memory()

    def forecast(self, closes: np.ndarray, frequency: str = "15min") -> ForecastResult:
        self.load()
        import pandas as pd

        values = np.ascontiguousarray(closes[-self.settings.bar_count :], dtype=np.float32)
        if values.nbytes > self.settings.max_model_ram_bytes:
            raise MemoryError("input array exceeds configured model memory budget")
        context = pd.DataFrame(
            {
                "id": "series",
                "timestamp": pd.date_range("2020-01-01", periods=len(values), freq=frequency),
                "target": values,
            }
        )
        output = self._pipeline.predict_df(
            context,
            prediction_length=self.settings.prediction_length,
            quantile_levels=[0.5],
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        )
        prediction_column = "predictions" if "predictions" in output else "0.5"
        predictions = output[prediction_column].to_numpy(dtype=np.float32)[-5:]
        self._check_process_memory()
        return trend_from_predictions(float(values[-1]), predictions)
