"""Point-forecast adapters for Adaptime extraction."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from timebench.evaluation.utils import normalize_tsicl_quantiles
from timebench.paths import foundation_weight_path


MODEL_ALIASES = ("chronos_bolt", "chronos2", "ts_icl", "seasonal_naive")


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value, dtype=np.float32)


def _point_batch(values: Any, *, channels: int, horizon: int) -> np.ndarray:
    items = values if isinstance(values, (list, tuple)) else list(values)
    normalized: list[np.ndarray] = []
    for value in items:
        array = _numpy(value)
        if array.ndim == 1:
            array = array[None, :]
        if array.shape == (horizon, channels):
            array = array.T
        if array.shape != (channels, horizon):
            raise ValueError(
                f"foundation point forecast has shape {array.shape}, expected {(channels, horizon)}"
            )
        normalized.append(array)
    return np.stack(normalized)


class _ChronosAdapter:
    def __init__(
        self,
        model_name: str,
        pipeline: Any,
        *,
        horizon: int,
        weights_id: str,
        supports_multivariate: bool,
        supports_retrieval_context: bool,
    ) -> None:
        self.model_name = model_name
        self.pipeline = pipeline
        self.horizon = int(horizon)
        self.weights_id = weights_id
        self.supports_multivariate = supports_multivariate
        self.supports_retrieval_context = supports_retrieval_context

    def represent(self, context: np.ndarray) -> np.ndarray:
        raise ValueError(f"{self.model_name} does not expose a stable retrieval representation")

    def forecast(
        self,
        context: np.ndarray,
        *,
        retrieval_context: np.ndarray | None = None,
    ) -> np.ndarray:
        import torch

        context = np.asarray(context, dtype=np.float32)
        if retrieval_context is not None and not self.supports_retrieval_context:
            raise ValueError(f"{self.model_name} does not consume retrieval context")
        inputs: list[Any] = []
        for row, target in enumerate(context):
            target_tensor = torch.from_numpy(
                target if self.supports_multivariate else target[0]
            )
            if retrieval_context is None:
                inputs.append(target_tensor)
                continue
            retrieved = np.asarray(retrieval_context[row], dtype=np.float32)
            flattened = retrieved.reshape(-1, retrieved.shape[-1])
            lookback = target.shape[-1]
            inputs.append(
                {
                    "target": target_tensor,
                    "past_covariates": {
                        f"retrieved_{index}": torch.from_numpy(values[:lookback])
                        for index, values in enumerate(flattened)
                    },
                    "future_covariates": {
                        f"retrieved_{index}": torch.from_numpy(values[lookback:])
                        for index, values in enumerate(flattened)
                    },
                }
            )
        with torch.inference_mode():
            quantiles, means = self.pipeline.predict_quantiles(
                inputs=inputs,
                prediction_length=self.horizon,
                quantile_levels=[0.5],
            )
        if means is not None:
            return _point_batch(means, channels=context.shape[1], horizon=self.horizon)
        median = []
        for value in quantiles:
            array = _numpy(value)
            if array.ndim == 3 and array.shape[-1] == 1:
                array = array[..., 0]
            elif array.ndim == 3 and array.shape[0] == 1:
                array = array[0]
            median.append(array)
        return _point_batch(median, channels=context.shape[1], horizon=self.horizon)


class _TSICLAdapter:
    model_name = "ts_icl"
    supports_multivariate = False
    supports_retrieval_context = True

    def __init__(self, model: Any, *, horizon: int, device: str, weights_id: str) -> None:
        self.model = model
        self.horizon = int(horizon)
        self.device = device
        self.weights_id = weights_id

    def represent(self, context: np.ndarray) -> np.ndarray:
        raise ValueError("ts_icl does not expose a stable retrieval representation")

    def forecast(
        self,
        context: np.ndarray,
        *,
        retrieval_context: np.ndarray | None = None,
    ) -> np.ndarray:
        import torch

        context = np.asarray(context, dtype=np.float32)
        if context.shape[1] != 1:
            raise ValueError("ts_icl extraction requires univariate prepared windows")
        inputs = torch.from_numpy(np.moveaxis(context, 1, 2))
        kwargs: dict[str, Any] = {}
        if retrieval_context is not None:
            retrieved = np.asarray(retrieval_context, dtype=np.float32)
            flattened = retrieved.reshape(retrieved.shape[0], -1, retrieved.shape[-1])
            kwargs.update(
                {
                    "covars": torch.from_numpy(np.moveaxis(flattened, 1, 2)),
                    "allow_auto_complete": False,
                    "allow_covar_forecast": False,
                }
            )
        with torch.inference_mode():
            _, quantiles = self.model.forecast(
                inputs=inputs,
                prediction_length=self.horizon,
                batch_size=len(context),
                quantile_levels=[0.5],
                context_length=context.shape[-1],
                device=torch.device(self.device),
                denormalize=True,
                squeeze_output=False,
                **kwargs,
            )
        values = normalize_tsicl_quantiles(quantiles)
        if values.ndim != 4 or values.shape[1] != 1:
            raise ValueError(f"unexpected TS-ICL quantile shape {values.shape}")
        return np.asarray(values[:, 0], dtype=np.float32)


class _SeasonalNaiveAdapter:
    model_name = "seasonal_naive"
    weights_id = "none"
    supports_multivariate = True
    supports_retrieval_context = False

    def __init__(self, *, horizon: int, period: int) -> None:
        self.horizon = int(horizon)
        self.period = int(period)

    def represent(self, context: np.ndarray) -> np.ndarray:
        raise ValueError("seasonal_naive has no model representation")

    def forecast(
        self,
        context: np.ndarray,
        *,
        retrieval_context: np.ndarray | None = None,
    ) -> np.ndarray:
        if retrieval_context is not None:
            raise ValueError("seasonal_naive does not consume retrieval context")
        context = np.asarray(context, dtype=np.float32)
        period = min(self.period, context.shape[-1])
        repeats = int(np.ceil(self.horizon / period))
        return np.tile(context[..., -period:], (1, 1, repeats))[..., : self.horizon]


@lru_cache(maxsize=None)
def _load_chronos_pipeline(path: str, device: str) -> Any:
    from chronos import BaseChronosPipeline

    return BaseChronosPipeline.from_pretrained(
        path,
        device_map=device,
        local_files_only=True,
    )


@lru_cache(maxsize=None)
def _load_tsicl_model(path: str) -> Any:
    from tsicl import TSICL

    return TSICL(model_path=path, allow_auto_download=False)


def load_adaptime_forecaster(
    name: str,
    *,
    horizon: int,
    period: int,
    model_path: str | Path | None = None,
    weights_id: str | None = None,
    device: str = "cuda",
) -> Any:
    """Load one exact maintained alias without online downloads."""

    if name not in MODEL_ALIASES:
        raise ValueError(f"model must be one of {MODEL_ALIASES}; aliases are case-sensitive")
    if name == "seasonal_naive":
        return _SeasonalNaiveAdapter(horizon=horizon, period=period)

    if name == "ts_icl":
        path = foundation_weight_path(
            "tsicl/tsicl-v1.ckpt",
            explicit=model_path,
            directory=False,
        )
        model = _load_tsicl_model(str(path))
        return _TSICLAdapter(
            model,
            horizon=horizon,
            device=device,
            weights_id=weights_id or path.name,
        )

    default_path = "chronos2" if name == "chronos2" else "chronos-bolt-base"
    path = foundation_weight_path(default_path, explicit=model_path, directory=True)
    pipeline = _load_chronos_pipeline(str(path), device)
    return _ChronosAdapter(
        name,
        pipeline,
        horizon=horizon,
        weights_id=weights_id or path.name,
        supports_multivariate=name == "chronos2",
        supports_retrieval_context=name == "chronos2",
    )
