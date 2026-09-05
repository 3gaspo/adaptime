"""Untouched TIME test comparison for frozen Adaptime ridge models."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from timebench.adaptime.ridge import full_ridge_design, full_ridge_predict
from timebench.evaluation.adaptation_data import PreparedDataset
from timebench.pipeline.adaptime_extraction import open_extraction
from timebench.pipeline.adaptime_training import ExtractionArrays, open_adaptation_model


ADAPTATION_RESULT_SCHEMA = 1
METHODS = ("vanilla", "covariate", "adaptime")
METRICS = ("mse", "mae", "nmse", "nmae")


@dataclass(frozen=True)
class AdaptimeTestingConfig:
    chunk_size: int = 1024

    def validate(self) -> None:
        if int(self.chunk_size) <= 0:
            raise ValueError("chunk_size must be positive")


def _canonical_hash(value: dict[str, object]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _memmap(path: Path, shape: tuple[int, ...], dtype: object) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", shape=shape, dtype=dtype)


def _aggregate_metrics(
    references: np.ndarray,
    metrics: dict[str, np.ndarray],
) -> dict[str, object]:
    rows, channels = metrics["mse"].shape
    item = np.repeat(np.asarray(references)[:, 0], channels)
    if channels == 1 and np.all(np.asarray(references)[:, 1] >= 0):
        channel = np.asarray(references)[:, 1]
    else:
        channel = np.tile(np.arange(channels, dtype=np.int64), rows)
    user_keys = np.stack((item, channel), axis=1)
    unique_users, user_inverse = np.unique(user_keys, axis=0, return_inverse=True)
    user_counts = np.bincount(user_inverse)
    result: dict[str, object] = {
        "windows": int(rows),
        "channels": int(channels),
        "users": int(len(unique_users)),
    }
    for name, values in metrics.items():
        flat = np.asarray(values).reshape(-1)
        per_user = np.bincount(user_inverse, weights=flat) / user_counts
        result[name] = {
            "equal_window_mean": float(flat.mean(dtype=np.float64)),
            "equal_window_std": float(flat.std(dtype=np.float64)),
            "equal_user_mean": float(per_user.mean()),
            "equal_user_std": float(per_user.std()),
        }
    return result


def _metric_values(
    prediction: np.ndarray,
    target: np.ndarray,
    scale: np.ndarray,
) -> dict[str, np.ndarray]:
    error = np.asarray(prediction) - np.asarray(target)
    selected_scale = np.maximum(np.asarray(scale), 1e-8)[..., None]
    return {
        "mse": np.mean(np.square(error), axis=-1),
        "mae": np.mean(np.abs(error), axis=-1),
        "nmse": np.mean(np.square(error / selected_scale), axis=-1),
        "nmae": np.mean(np.abs(error) / selected_scale, axis=-1),
    }


def _inference_timing(
    extraction: dict[str, object],
    selected_k: int,
    windows: int,
    ridge_seconds: float,
) -> dict[str, object]:
    measured = dict(extraction["timing_seconds"])
    representation = float(measured["test.representation_seconds"])
    retrieval = float(measured["test.retrieval_seconds"])
    context_construction = float(
        measured[f"test.context_construction_k{selected_k}_seconds"]
    )
    vanilla_forecast = float(measured["test.vanilla_forecast_seconds"])
    covariate_forecast = float(
        measured[f"test.context_forecast_k{selected_k}_seconds"]
    )
    shared_retrieval = representation + retrieval + context_construction
    totals = {
        "vanilla": vanilla_forecast,
        "covariate": shared_retrieval + covariate_forecast,
        "adaptime": (
            shared_retrieval
            + vanilla_forecast
            + covariate_forecast
            + float(ridge_seconds)
        ),
    }
    return {
        "unit": "seconds",
        "test_windows": int(windows),
        "methods": {
            method: {
                "total_seconds": seconds,
                "seconds_per_window": seconds / int(windows),
            }
            for method, seconds in totals.items()
        },
        "components": {
            "query_representation_seconds": representation,
            "retrieval_seconds": retrieval,
            "context_construction_seconds": context_construction,
            "vanilla_model_forecast_seconds": vanilla_forecast,
            "covariate_model_forecast_seconds": covariate_forecast,
            "ridge_design_and_adjustment_seconds": float(ridge_seconds),
        },
        "precomputed_extraction": {
            "datastore_representation_seconds": float(
                measured["datastore.representation_seconds"]
            ),
            "neighbor_forecast_seconds": float(
                measured["offline.neighbor_forecast_seconds"]
            ),
            "complete_extraction_seconds": float(
                measured["extraction_total_seconds"]
            ),
        },
    }


def evaluate_frozen_adaptation(
    prepared_path: str | Path,
    extraction_path: str | Path,
    model_path: str | Path,
    config: AdaptimeTestingConfig,
    output_dir: str | Path,
) -> Path:
    """Compare V, retrieval-context C, and frozen full_ridge_shared on TIME test."""

    config.validate()
    prepared = PreparedDataset(prepared_path)
    extraction_root, extraction_manifest = open_extraction(extraction_path)
    model_root, model_manifest = open_adaptation_model(model_path)
    if extraction_manifest["prepared_signature"] != prepared.signature:
        raise ValueError("extraction and prepared TIME windows do not match")
    if model_manifest["extraction_signature"] != extraction_manifest["signature"]:
        raise ValueError("frozen Adaptime model and extraction do not match")

    identity = {
        "schema_version": ADAPTATION_RESULT_SCHEMA,
        "timing_contract": "test_method_seconds_per_window",
        "prepared_signature": prepared.signature,
        "extraction_signature": extraction_manifest["signature"],
        "model_signature": model_manifest["signature"],
        "comparison": list(METHODS),
    }
    signature = _canonical_hash(identity)
    root = Path(output_dir).expanduser().resolve()
    manifest_path = root / "result_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = dict(existing.get("files", {}))
        expected = [files.get("comparison_summary")]
        expected.extend(dict(files.get("predictions", {})).values())
        expected.extend(dict(files.get("metrics", {})).values())
        if (
            existing.get("signature") == signature
            and existing.get("status") == "completed"
            and all(relative and (root / relative).is_file() for relative in expected)
        ):
            return manifest_path
        raise FileExistsError(f"test directory already contains a different run: {root}")
    root.mkdir(parents=True, exist_ok=True)

    arrays = ExtractionArrays(extraction_root, extraction_manifest)
    selected_k = int(model_manifest["selected"]["k"])
    coefficients = np.load(
        model_root / model_manifest["files"]["coefficients"], allow_pickle=False
    )
    vanilla = arrays.open("test.vanilla")
    context = arrays.open(f"test.context_forecast_k{selected_k}")
    target = arrays.open("test.target")
    scale = arrays.open("test.query_scale")
    neighbor_ids = arrays.open("test.neighbor_id")
    prediction_stores = {
        method: _memmap(root / "predictions" / f"{method}.npy", target.shape, np.float32)
        for method in METHODS
    }
    metric_stores = {
        (method, metric): _memmap(
            root / "metrics" / f"{method}_{metric}.npy",
            target.shape[:-1],
            np.float32,
        )
        for method in METHODS
        for metric in METRICS
    }
    ridge_seconds = 0.0

    for start in range(0, len(target), config.chunk_size):
        stop = min(start + config.chunk_size, len(target))
        ridge_started = perf_counter()
        selected = np.asarray(neighbor_ids[start:stop, :selected_k])
        design, _ = full_ridge_design(
            vanilla[start:stop],
            context[start:stop],
            arrays.datastore_target[selected],
            arrays.neighbor_forecast(selected),
            target[start:stop],
        )
        adapted = full_ridge_predict(vanilla[start:stop], design, coefficients)
        ridge_seconds += perf_counter() - ridge_started
        predictions = {
            "vanilla": np.asarray(vanilla[start:stop]),
            "covariate": np.asarray(context[start:stop]),
            "adaptime": adapted,
        }
        for method, values in predictions.items():
            prediction_stores[method][start:stop] = values
            computed = _metric_values(values, target[start:stop], scale[start:stop])
            for metric, metric_values in computed.items():
                metric_stores[(method, metric)][start:stop] = metric_values

    for store in (*prediction_stores.values(), *metric_stores.values()):
        store.flush()

    summaries = {
        method: _aggregate_metrics(
            prepared.indices("test"),
            {
                metric: np.asarray(metric_stores[(method, metric)])
                for metric in METRICS
            },
        )
        for method in METHODS
    }
    vanilla_mse = np.asarray(metric_stores[("vanilla", "mse")])
    wins = {
        method: float(
            np.mean(np.asarray(metric_stores[(method, "mse")]) < vanilla_mse)
        )
        for method in ("covariate", "adaptime")
    }
    timing = _inference_timing(
        extraction_manifest,
        selected_k,
        len(target),
        ridge_seconds,
    )
    _atomic_json(
        root / "comparison_summary.json",
        {
            "methods": summaries,
            "mse_win_rate_vs_vanilla": wins,
            "selected": model_manifest["selected"],
            "timing": timing,
        },
    )

    result: dict[str, object] = {
        **identity,
        "format": "adaptime_time_comparison",
        "signature": signature,
        "status": "completed",
        "protocol": "frozen_model_evaluate_untouched_official_time_test",
        "testing_config": asdict(config),
        "selected": model_manifest["selected"],
        "timing": timing,
        "feature_names": model_manifest["feature_names"],
        "files": {
            "predictions": {
                method: f"predictions/{method}.npy" for method in METHODS
            },
            "metrics": {
                f"{method}.{metric}": f"metrics/{method}_{metric}.npy"
                for method in METHODS
                for metric in METRICS
            },
            "comparison_summary": "comparison_summary.json",
        },
    }
    _atomic_json(manifest_path, result)
    return manifest_path
