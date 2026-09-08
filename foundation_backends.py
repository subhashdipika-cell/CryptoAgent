"""Strictly offline adapters for optional time-series foundation challengers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class GraniteTTMR3Backend:
    """IBM Granite TTM-R3 adapter using a pre-staged local checkpoint."""

    name = "ibm-granite/granite-timeseries-ttm-r3"

    def __init__(self, model_path: Path, horizon: int = 5):
        self.model_path = model_path
        self.horizon = horizon
        self._model: Any | None = None

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"TTM-R3 checkpoint is not staged: {self.model_path}")
        try:
            from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
        except ImportError as error:
            raise RuntimeError("install requirements-candidates.txt before validating TTM-R3") from error
        self._model = TinyTimeMixerForPrediction.from_pretrained(
            str(self.model_path),
            local_files_only=True,
            prediction_filter_length=self.horizon,
        )
        self._model.eval()

    def forecast(self, closes: np.ndarray) -> np.ndarray:
        self.load()
        import torch

        context_length = int(self._model.config.context_length)
        values = np.asarray(closes[-context_length:], dtype=np.float32)
        tensor = torch.from_numpy(values).reshape(1, -1, 1)
        with torch.inference_mode():
            output = self._model(past_values=tensor)
        predictions = output.prediction_outputs[0, : self.horizon, 0]
        return predictions.detach().cpu().numpy().astype(np.float32)


class TimesFM25Backend:
    """Google TimesFM 2.5 adapter via its PyTorch Transformers checkpoint."""

    name = "google/timesfm-2.5-200m-transformers"

    def __init__(self, model_path: Path, horizon: int = 5, context: int = 512):
        self.model_path = model_path
        self.horizon = horizon
        self.context = context
        self._model: Any | None = None

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"TimesFM checkpoint is not staged: {self.model_path}")
        from transformers import TimesFm2_5ModelForPrediction

        self._model = TimesFm2_5ModelForPrediction.from_pretrained(
            str(self.model_path), local_files_only=True
        )
        self._model.eval()

    def forecast(self, closes: np.ndarray) -> np.ndarray:
        self.load()
        import torch

        values = torch.as_tensor(
            np.asarray(closes[-self.context :], dtype=np.float32), dtype=torch.float32
        )
        with torch.inference_mode():
            output = self._model(
                past_values=[values],
                forecast_context_len=min(self.context, len(values)),
                truncate_negative=True,
            )
        return (
            output.mean_predictions[0, : self.horizon]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )


def candidate_inventory(ttm_path: Path, timesfm_path: Path) -> list[dict[str, object]]:
    import importlib.util

    return [
        {
            "candidate": GraniteTTMR3Backend.name,
            "staged": ttm_path.is_dir(),
            "runtime_installed": importlib.util.find_spec("tsfm_public") is not None,
            "role": "CPU-primary foundation challenger",
        },
        {
            "candidate": TimesFM25Backend.name,
            "staged": timesfm_path.is_dir(),
            "runtime_installed": importlib.util.find_spec("transformers") is not None,
            "role": "accuracy challenger; validate RAM and latency",
        },
    ]
