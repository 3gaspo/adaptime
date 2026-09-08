"""Unconditional vanilla forecasts for every official TIME test row."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from timebench.evaluation.adaptation_data import PreparedDataset
from timebench.evaluation.timing import EvaluationTimer
from timebench.pipeline.adaptime_extraction import AdaptimeForecaster


VANILLA_SCHEMA = 1


@dataclass(frozen=True)
class VanillaConfig:
    model_batch_size: int = 64
    arrow_cache_items: int = 2

    def validate(self) -> None:
        if int(self.model_batch_size) <= 0:
            raise ValueError("model_batch_size must be positive")
        if int(self.arrow_cache_items) <= 0:
            raise ValueError("arrow_cache_items must be positive")


def _canonical_hash(value: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _memmap(path: Path, shape: tuple[int, ...], dtype: object) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", shape=shape, dtype=dtype)


def extract_vanilla_test_forecasts(
    prepared_path: str | Path,
    forecaster: AdaptimeForecaster,
    config: VanillaConfig,
    output_dir: str | Path,
) -> Path:
    """Forecast all official test rows with as much past context as is available."""

    config.validate()
    prepared = PreparedDataset(prepared_path)
    references = prepared.indices("test")
    if not len(references):
        raise ValueError("the official TIME test grid is empty")
    available = np.minimum(
        np.asarray(references[:, 2], dtype=np.int64), prepared.context_length
    )
    if np.any(available <= 0):
        raise ValueError("an official TIME test row has no past observation")

    identity = {
        "schema_version": VANILLA_SCHEMA,
        "prepared_signature": prepared.signature,
        "model": forecaster.model_name,
        "weights_id": forecaster.weights_id,
        "maximum_context_length": prepared.context_length,
        "context_policy": "all_available_history_capped_at_model_limit",
        "config": asdict(config),
    }
    signature = _canonical_hash(identity)
    root = Path(output_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("signature") == signature
            and existing.get("status") == "completed"
            and all(
                (root / relative).is_file()
                for relative in dict(existing.get("arrays", {})).values()
            )
        ):
            return manifest_path
        raise FileExistsError(f"vanilla directory already differs: {root}")

    reader = prepared.reader(cache_items=config.arrow_cache_items)
    first = reader.read(references[:1], context_length=int(available[0]))
    predictions = _memmap(
        root / "predictions.npy",
        (len(references), first.context.shape[1], prepared.prediction_length),
        np.float32,
    )
    context_lengths = _memmap(
        root / "context_length.npy", (len(references),), np.int64
    )
    context_lengths[:] = available
    forecast_seconds = 0.0
    for context_length in np.unique(available):
        positions = np.flatnonzero(available == context_length)
        for start in range(0, len(positions), config.model_batch_size):
            selected = positions[start : start + config.model_batch_size]
            batch = reader.read(
                references[selected], context_length=int(context_length)
            )
            timer = EvaluationTimer()
            timer.start()
            values = np.asarray(
                forecaster.forecast(batch.context, retrieval_context=None),
                dtype=np.float32,
            )
            forecast_seconds += timer.stop()
            if values.shape != predictions[selected].shape:
                raise ValueError(
                    f"vanilla forecaster returned {values.shape}, expected "
                    f"{predictions[selected].shape}"
                )
            predictions[selected] = values
    predictions.flush()
    context_lengths.flush()

    manifest: dict[str, Any] = {
        **identity,
        "format": "adaptime_vanilla_test_forecasts",
        "signature": signature,
        "status": "completed",
        "forecast_type": "point",
        "prediction_length": prepared.prediction_length,
        "test_windows": int(len(references)),
        "context_length_range": {
            "minimum": int(available.min()),
            "maximum": int(available.max()),
        },
        "timing_seconds": {"vanilla_model_forecast_seconds": float(forecast_seconds)},
        "arrays": {
            "predictions": "predictions.npy",
            "context_length": "context_length.npy",
        },
    }
    _atomic_json(manifest_path, manifest)
    return manifest_path


def open_vanilla_test_forecasts(path: str | Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = Path(path).expanduser().resolve()
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != VANILLA_SCHEMA
        or manifest.get("format") != "adaptime_vanilla_test_forecasts"
        or manifest.get("status") != "completed"
    ):
        raise ValueError("not a completed Adaptime vanilla test artifact")
    return manifest_path.parent, manifest
