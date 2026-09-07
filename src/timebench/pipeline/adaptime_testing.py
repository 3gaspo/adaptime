"""Untouched TIME test comparison for frozen Adaptime ridge models."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from timebench.adaptime.ridge import (
    full_ridge_design,
    full_ridge_predict_with_fallback,
)
from timebench.evaluation.adaptation_data import PreparedDataset
from timebench.evaluation.timing import EvaluationTimer
from timebench.pipeline.adaptime_extraction import AdaptimeForecaster, open_extraction
from timebench.pipeline.adaptime_training import ExtractionArrays, open_adaptation_model


ADAPTATION_RESULT_SCHEMA = 1
METHODS = ("vanilla", "covariate", "adaptime")
SCORE_METHODS = ("seasonal_naive", *METHODS)
METRICS = ("mse", "mae", "mase", "msse")


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


def _write_array(path: Path, values: np.ndarray, dtype: object) -> None:
    store = _memmap(path, values.shape, dtype)
    store[:] = values
    store.flush()


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
    result: dict[str, object] = {
        "windows": int(rows),
        "channels": int(channels),
        "users": int(len(unique_users)),
    }
    for name, values in metrics.items():
        flat = np.asarray(values).reshape(-1)
        finite = np.isfinite(flat)
        finite_user_counts = np.bincount(user_inverse, weights=finite.astype(np.int64))
        per_user = np.divide(
            np.bincount(user_inverse, weights=np.where(finite, flat, 0.0)),
            finite_user_counts,
            out=np.full(len(unique_users), np.nan, dtype=np.float64),
            where=finite_user_counts > 0,
        )
        finite_values = flat[finite]
        finite_users = per_user[np.isfinite(per_user)]
        result[name] = {
            "equal_window_mean": (
                float(finite_values.mean(dtype=np.float64)) if len(finite_values) else np.nan
            ),
            "equal_window_std": (
                float(finite_values.std(dtype=np.float64)) if len(finite_values) else np.nan
            ),
            "equal_user_mean": float(finite_users.mean()) if len(finite_users) else np.nan,
            "equal_user_std": float(finite_users.std()) if len(finite_users) else np.nan,
            "finite_windows": int(len(finite_values)),
            "total_windows": int(len(flat)),
        }
    return result


def _metric_values(
    prediction: np.ndarray,
    target: np.ndarray,
    mase_scale: np.ndarray,
    msse_scale: np.ndarray,
) -> dict[str, np.ndarray]:
    def valid_target_mean(values: np.ndarray) -> np.ndarray:
        valid = np.isfinite(target)
        count = valid.sum(axis=-1)
        return np.divide(
            np.where(valid, values, 0.0).sum(axis=-1),
            count,
            out=np.full(values.shape[:-1], np.nan, dtype=np.float64),
            where=count > 0,
        )

    prediction = np.asarray(prediction)
    target = np.asarray(target)
    error = prediction - target
    mse = valid_target_mean(np.square(error))
    mae = valid_target_mean(np.abs(error))
    mase_scale = np.where(np.asarray(mase_scale) > 0, mase_scale, np.nan)
    msse_scale = np.where(np.asarray(msse_scale) > 0, msse_scale, np.nan)
    return {
        "mse": mse,
        "mae": mae,
        "mase": mae / mase_scale,
        "msse": mse / np.square(msse_scale),
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


def evaluate_vanilla_fallback(
    hf_dataset: object,
    forecaster: AdaptimeForecaster,
    *,
    dataset: str,
    term: str,
    frequency: str,
    target_mode: str,
    context_limit: int,
    prediction_length: int,
    test_length: int,
    seasonality: int,
    model_batch_size: int,
    fallback_reason: str,
    output_dir: str | Path,
) -> Path:
    """Evaluate TIME test windows with vanilla forecasts when adaptation is infeasible."""

    if target_mode != "univariate":
        raise ValueError("the vanilla-only Adaptime fallback currently requires univariate mode")
    identity = {
        "schema_version": ADAPTATION_RESULT_SCHEMA,
        "format": "adaptime_time_comparison",
        "protocol": "vanilla_fallback_insufficient_adaptation_history",
        "timing_contract": "test_method_seconds_per_window",
        "dataset_fingerprint": str(hf_dataset._fingerprint),
        "model": forecaster.model_name,
        "weights_id": forecaster.weights_id,
        "dataset": dataset,
        "frequency": frequency,
        "term": term,
        "target_mode": target_mode,
        "context_limit": int(context_limit),
        "prediction_length": int(prediction_length),
        "test_length": int(test_length),
        "metric_seasonality": int(seasonality),
        "comparison": list(SCORE_METHODS),
        "metrics": list(METRICS),
        "performance_metric": "task_mase_divided_by_matching_seasonal_naive_mase",
        "fallback_reason": fallback_reason,
    }
    signature = _canonical_hash(identity)
    root = Path(output_dir).expanduser().resolve()
    manifest_path = root / "result_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = dict(existing.get("files", {}))
        expected = [
            files.get("comparison_summary"),
            files.get("references"),
            files.get("target"),
            files.get("mase_scale"),
            files.get("msse_scale"),
        ]
        expected.extend(dict(files.get("predictions", {})).values())
        expected.extend(dict(files.get("metrics", {})).values())
        expected.extend((files.get("rag_eligible"), files.get("fallback_reason")))
        if (
            existing.get("signature") == signature
            and existing.get("status") == "completed"
            and all(relative and (root / relative).is_file() for relative in expected)
        ):
            return manifest_path
        raise FileExistsError(f"test directory already contains a different run: {root}")

    references: list[tuple[int, int, int]] = []
    contexts_by_length: dict[int, list[tuple[int, np.ndarray]]] = {}
    targets: list[np.ndarray] = []
    mase_scales: list[np.ndarray] = []
    msse_scales: list[np.ndarray] = []
    seasonal_predictions: list[np.ndarray] = []
    context_lengths: list[int] = []
    windows = int(test_length) // int(prediction_length)
    if windows <= 0:
        raise ValueError("test_length must contain at least one complete horizon")

    for item in range(len(hf_dataset)):
        values = np.asarray(hf_dataset[item]["target"])
        if values.ndim == 1:
            values = values[None, :]
        elif values.ndim != 2:
            raise ValueError(f"TIME targets must have one or two dimensions, got {values.shape}")
        test_start = int(values.shape[-1]) - int(test_length)
        if test_start <= 0:
            raise ValueError(
                f"item {item} cannot provide the configured test interval of {test_length} values"
            )
        for channel in range(int(values.shape[0])):
            raw_series = np.asarray(values[channel])
            series = np.asarray(raw_series, dtype=np.float32)
            for window in range(windows):
                origin = test_start + window * int(prediction_length)
                context_start = max(0, origin - int(context_limit))
                context = series[None, context_start:origin]
                target = series[None, origin : origin + int(prediction_length)]
                if context.shape[-1] == 0 or target.shape[-1] != int(prediction_length):
                    raise ValueError(f"invalid vanilla fallback window {(item, channel, origin)}")
                if origin < int(seasonality):
                    raise ValueError(
                        f"window {(item, channel, origin)} lacks one seasonal-naive period"
                    )
                prefix = np.asarray(raw_series[:origin], dtype=np.float64)
                left = prefix[: -int(seasonality)]
                right = prefix[int(seasonality) :]
                valid = np.isfinite(left) & np.isfinite(right)
                differences = right[valid] - left[valid]
                mase_scale = (
                    float(np.mean(np.abs(differences))) if len(differences) else np.nan
                )
                msse_scale = (
                    float(np.sqrt(np.mean(np.square(differences))))
                    if len(differences)
                    else np.nan
                )
                period_values = series[origin - int(seasonality) : origin]
                repeats = int(np.ceil(int(prediction_length) / int(seasonality)))
                seasonal = np.tile(period_values, repeats)[: int(prediction_length)]

                position = len(references)
                references.append((item, channel, origin))
                targets.append(target)
                mase_scales.append(np.asarray([mase_scale], dtype=np.float32))
                msse_scales.append(np.asarray([msse_scale], dtype=np.float32))
                seasonal_predictions.append(seasonal[None, :])
                context_lengths.append(int(context.shape[-1]))
                contexts_by_length.setdefault(int(context.shape[-1]), []).append(
                    (position, context)
                )

    target_values = np.stack(targets)
    vanilla = np.empty_like(target_values, dtype=np.float32)
    timer = EvaluationTimer()
    timer.start()
    for length in sorted(contexts_by_length):
        grouped = contexts_by_length[length]
        for start in range(0, len(grouped), int(model_batch_size)):
            batch = grouped[start : start + int(model_batch_size)]
            positions = np.asarray([position for position, _ in batch], dtype=np.int64)
            contexts = np.stack([context for _, context in batch])
            forecast = np.asarray(
                forecaster.forecast(contexts, retrieval_context=None), dtype=np.float32
            )
            expected = (len(batch), 1, int(prediction_length))
            if forecast.shape != expected:
                raise ValueError(
                    f"forecaster returned {forecast.shape}, expected {expected}"
                )
            vanilla[positions] = forecast
    forecast_seconds = timer.stop()

    references_array = np.asarray(references, dtype=np.int64)
    mase_scale_values = np.stack(mase_scales)
    msse_scale_values = np.stack(msse_scales)
    seasonal_naive = np.stack(seasonal_predictions)
    predictions = {
        "seasonal_naive": seasonal_naive,
        "vanilla": vanilla,
        "covariate": vanilla,
        "adaptime": vanilla,
    }
    metric_values = {
        method: _metric_values(
            values,
            target_values,
            mase_scale_values,
            msse_scale_values,
        )
        for method, values in predictions.items()
    }
    summaries = {
        method: _aggregate_metrics(references_array, metrics)
        for method, metrics in metric_values.items()
    }
    seasonal_summary = summaries["seasonal_naive"]
    for method in SCORE_METHODS:
        scaled_mase: dict[str, float] = {}
        for key in ("equal_window_mean", "equal_user_mean"):
            denominator = float(seasonal_summary["mase"][key])
            if not np.isfinite(denominator) or denominator <= 0:
                raise ValueError(
                    "scaled MASE requires a positive matching Seasonal Naive MASE"
                )
            scaled_mase[key] = float(summaries[method]["mase"][key]) / denominator
        summaries[method]["scaled_mase"] = scaled_mase

    root.mkdir(parents=True, exist_ok=True)
    _write_array(root / "references.npy", references_array, np.int64)
    _write_array(root / "target.npy", target_values, np.float32)
    _write_array(root / "mase_scale.npy", mase_scale_values, np.float32)
    _write_array(root / "msse_scale.npy", msse_scale_values, np.float32)
    for method, values in predictions.items():
        _write_array(root / "predictions" / f"{method}.npy", values, np.float32)
        for metric, metric_array in metric_values[method].items():
            _write_array(
                root / "metrics" / f"{method}_{metric}.npy",
                metric_array,
                np.float32,
            )
    rag_eligible = np.zeros(len(references_array), dtype=bool)
    fallback_codes = np.full(len(references_array), 5, dtype=np.uint8)
    _write_array(root / "predictions" / "rag_eligible.npy", rag_eligible, bool)
    _write_array(root / "predictions" / "fallback_reason.npy", fallback_codes, np.uint8)

    timing = {
        "unit": "seconds",
        "test_windows": int(len(references_array)),
        "methods": {
            method: {
                "total_seconds": float(forecast_seconds),
                "seconds_per_window": float(forecast_seconds / len(references_array)),
            }
            for method in METHODS
        },
        "components": {
            "vanilla_model_forecast_seconds": float(forecast_seconds),
            "retrieval_seconds": 0.0,
            "covariate_model_forecast_seconds": 0.0,
            "ridge_design_and_adjustment_seconds": 0.0,
        },
        "precomputed_extraction": {"complete_extraction_seconds": 0.0},
    }
    selected = {"k": None, "alpha": None, "validation_msse": None}
    _atomic_json(
        root / "comparison_summary.json",
        {
            "methods": summaries,
            "scaled_mase_win_rate_vs_vanilla": {
                "covariate": 0.0,
                "adaptime": 0.0,
            },
            "selected": selected,
            "rag_coverage": {
                "eligible_windows": 0,
                "fallback_windows": int(len(references_array)),
                "eligible_fraction": 0.0,
            },
            "timing": timing,
            "fallback_reason": fallback_reason,
        },
    )
    result: dict[str, object] = {
        **identity,
        "signature": signature,
        "status": "completed",
        "selected": selected,
        "timing": timing,
        "feature_names": [],
        "rag_coverage": {
            "eligible_windows": 0,
            "fallback_windows": int(len(references_array)),
            "eligible_fraction": 0.0,
        },
        "context_lengths": {
            "minimum": int(min(context_lengths)),
            "maximum": int(max(context_lengths)),
            "limit": int(context_limit),
        },
        "fallback_reason_codes": {"5": "insufficient_adaptation_history"},
        "files": {
            "references": "references.npy",
            "target": "target.npy",
            "mase_scale": "mase_scale.npy",
            "msse_scale": "msse_scale.npy",
            "predictions": {
                method: f"predictions/{method}.npy" for method in SCORE_METHODS
            },
            "metrics": {
                f"{method}.{metric}": f"metrics/{method}_{metric}.npy"
                for method in SCORE_METHODS
                for metric in METRICS
            },
            "comparison_summary": "comparison_summary.json",
            "rag_eligible": "predictions/rag_eligible.npy",
            "fallback_reason": "predictions/fallback_reason.npy",
        },
    }
    _atomic_json(manifest_path, result)
    return manifest_path


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
        "comparison": list(SCORE_METHODS),
        "metrics": list(METRICS),
        "metric_seasonality": prepared.seasonality,
        "performance_metric": "task_mase_divided_by_matching_seasonal_naive_mase",
        "testing_config": asdict(config),
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
        expected.extend((files.get("rag_eligible"), files.get("fallback_reason")))
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
    mase_scale = arrays.open("test.mase_scale")
    msse_scale = arrays.open("test.msse_scale")
    seasonal_naive = arrays.open("test.seasonal_naive")
    neighbor_ids = arrays.open("test.neighbor_id")
    base_eligible = arrays.open("test.rag_eligible")
    base_reason = arrays.open("test.fallback_reason")
    prediction_stores = {
        method: _memmap(root / "predictions" / f"{method}.npy", target.shape, np.float32)
        for method in SCORE_METHODS
    }
    metric_stores = {
        (method, metric): _memmap(
            root / "metrics" / f"{method}_{metric}.npy",
            target.shape[:-1],
            np.float32,
        )
        for method in SCORE_METHODS
        for metric in METRICS
    }
    eligibility_store = _memmap(root / "predictions" / "rag_eligible.npy", (len(target),), bool)
    reason_store = _memmap(root / "predictions" / "fallback_reason.npy", (len(target),), np.uint8)
    ridge_seconds = 0.0
    for start in range(0, len(target), config.chunk_size):
        stop = min(start + config.chunk_size, len(target))
        ridge_started = perf_counter()
        chunk_vanilla = np.asarray(vanilla[start:stop])
        chunk_context = np.asarray(context[start:stop])
        selected = np.asarray(neighbor_ids[start:stop, :selected_k])
        candidate = np.asarray(base_eligible[start:stop], dtype=bool) & np.all(
            selected >= 0, axis=1
        )
        design = np.zeros(
            (*chunk_vanilla.shape, len(coefficients)), dtype=np.float32
        )
        final_eligible = np.zeros(stop - start, dtype=bool)
        candidate_positions = np.flatnonzero(candidate)
        if len(candidate_positions):
            candidate_ids = selected[candidate_positions]
            candidate_design, _ = full_ridge_design(
                chunk_vanilla[candidate_positions],
                chunk_context[candidate_positions],
                arrays.datastore_target[candidate_ids],
                arrays.neighbor_forecast(candidate_ids),
                np.zeros_like(chunk_vanilla[candidate_positions]),
            )
            complete = np.isfinite(candidate_design).reshape(
                len(candidate_design), -1
            ).all(axis=1)
            complete &= np.isfinite(chunk_vanilla[candidate_positions]).reshape(
                len(candidate_positions), -1
            ).all(axis=1)
            complete_positions = candidate_positions[complete]
            design[complete_positions] = candidate_design[complete]
            final_eligible[complete_positions] = True
        adapted = full_ridge_predict_with_fallback(
            chunk_vanilla, design, coefficients, final_eligible
        )
        covariate = np.array(chunk_context, copy=True)
        covariate[~final_eligible] = chunk_vanilla[~final_eligible]
        final_reason = np.asarray(base_reason[start:stop]).copy()
        final_reason[candidate & ~final_eligible] = 4
        eligibility_store[start:stop] = final_eligible
        reason_store[start:stop] = final_reason
        ridge_seconds += perf_counter() - ridge_started
        predictions = {
            "seasonal_naive": np.asarray(seasonal_naive[start:stop]),
            "vanilla": chunk_vanilla,
            "covariate": covariate,
            "adaptime": adapted,
        }
        for method, values in predictions.items():
            prediction_stores[method][start:stop] = values
            computed = _metric_values(
                values,
                target[start:stop],
                mase_scale[start:stop],
                msse_scale[start:stop],
            )
            for metric, metric_values in computed.items():
                metric_stores[(method, metric)][start:stop] = metric_values

    for store in (
        *prediction_stores.values(),
        *metric_stores.values(),
        eligibility_store,
        reason_store,
    ):
        store.flush()

    summaries = {
        method: _aggregate_metrics(
            prepared.indices("test"),
            {
                metric: np.asarray(metric_stores[(method, metric)])
                for metric in METRICS
            },
        )
        for method in SCORE_METHODS
    }
    seasonal_naive_summary = summaries["seasonal_naive"]
    task_baseline_mase = float(seasonal_naive_summary["mase"]["equal_user_mean"])
    if not np.isfinite(task_baseline_mase) or task_baseline_mase <= 0:
        raise ValueError("scaled MASE requires a positive matching Seasonal Naive MASE")
    for method in SCORE_METHODS:
        scaled_mase: dict[str, float] = {}
        for key in ("equal_window_mean", "equal_user_mean"):
            denominator = float(seasonal_naive_summary["mase"][key])
            if not np.isfinite(denominator) or denominator <= 0:
                raise ValueError(
                    "scaled MASE requires a positive matching Seasonal Naive MASE"
                )
            scaled_mase[key] = float(summaries[method]["mase"][key]) / denominator
        summaries[method]["scaled_mase"] = scaled_mase
    vanilla_scaled_mase = (
        np.asarray(metric_stores[("vanilla", "mase")]) / task_baseline_mase
    )
    wins = {
        method: (
            lambda compared, finite: float(
                np.mean(compared[finite] < vanilla_scaled_mase[finite])
            )
            if np.any(finite)
            else np.nan
        )(
            np.asarray(metric_stores[(method, "mase")]) / task_baseline_mase,
            np.isfinite(np.asarray(metric_stores[(method, "mase")]))
            & np.isfinite(vanilla_scaled_mase),
        )
        for method in ("covariate", "adaptime")
    }
    rag_windows = int(np.count_nonzero(eligibility_store))
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
            "scaled_mase_win_rate_vs_vanilla": wins,
            "selected": model_manifest["selected"],
            "rag_coverage": {
                "eligible_windows": rag_windows,
                "fallback_windows": int(len(target) - rag_windows),
                "eligible_fraction": float(rag_windows / len(target)),
            },
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
        "rag_coverage": {
            "eligible_windows": rag_windows,
            "fallback_windows": int(len(target) - rag_windows),
            "eligible_fraction": float(rag_windows / len(target)),
        },
        "fallback_reason_codes": {
            "0": "rag_eligible",
            "1": "insufficient_finite_query_context",
            "2": "insufficient_valid_neighbors",
            "4": "nonfinite_ridge_features",
        },
        "files": {
            "predictions": {
                method: f"predictions/{method}.npy" for method in SCORE_METHODS
            },
            "metrics": {
                f"{method}.{metric}": f"metrics/{method}_{metric}.npy"
                for method in SCORE_METHODS
                for metric in METRICS
            },
            "comparison_summary": "comparison_summary.json",
            "rag_eligible": "predictions/rag_eligible.npy",
            "fallback_reason": "predictions/fallback_reason.npy",
        },
    }
    _atomic_json(manifest_path, result)
    return manifest_path
