"""Frozen Adaptime wrapper inference without metric computation."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from timebench.adaptime.ridge import (
    RIDGE_VARIANTS,
    full_ridge_design,
    full_ridge_predict_with_fallback,
    ridge_feature_indices,
)
from timebench.evaluation.adaptation_data import PreparedDataset
from timebench.pipeline.adaptime_extraction import (
    FALLBACK_REASONS,
    open_eval_extraction,
    open_extraction,
)
from timebench.pipeline.adaptime_training import ExtractionArrays, open_adaptation_model
from timebench.pipeline.adaptime_vanilla import open_vanilla_test_forecasts


POINT_PREDICTION_SCHEMA = 1
ADAPTATION_METHODS = (
    "vanilla",
    "covariate_prediction",
    "bayes_covariate_prediction",
    "cov_ridge_shared",
    "y_ridge_shared",
    "full_ridge_shared",
    "selected_adaptation",
)


@dataclass(frozen=True)
class PredictionConfig:
    chunk_size: int = 1024

    def validate(self) -> None:
        if int(self.chunk_size) <= 0:
            raise ValueError("chunk_size must be positive")


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


def _inference_timings(
    vanilla: dict[str, object],
    evaluation_extraction: dict[str, object],
    selected_ks: tuple[int, ...],
    selected_method: str,
    ridge_seconds: float,
    bayes_seconds: float,
) -> tuple[dict[str, float], dict[str, float]]:
    vanilla_seconds = float(
        vanilla["timing_seconds"]["vanilla_model_forecast_seconds"]
    )
    if not selected_ks:
        return (
            {method: vanilla_seconds for method in ADAPTATION_METHODS},
            {"vanilla_model_forecast_seconds": vanilla_seconds},
        )
    measured = dict(evaluation_extraction["timing_seconds"])
    components: dict[str, float] = {
        "query_representation_seconds": float(measured["test.representation_seconds"]),
        "retrieval_seconds": float(measured["test.retrieval_seconds"]),
        "context_construction_seconds": float(measured["test.context_construction_seconds"]),
        "vanilla_model_forecast_seconds": vanilla_seconds,
        "neighbor_model_forecast_seconds": float(
            measured["test.neighbor_forecast_seconds"]
        ),
        "covariate_model_forecast_seconds": float(measured["test.context_forecast_seconds"]),
        "ridge_design_and_adjustment_seconds": float(ridge_seconds),
        "bayes_mixture_seconds": float(bayes_seconds),
    }
    context_path = sum(
        components[name]
        for name in (
            "query_representation_seconds",
            "retrieval_seconds",
            "context_construction_seconds",
            "covariate_model_forecast_seconds",
        )
    )
    timings = {
            "vanilla": vanilla_seconds,
            "covariate_prediction": vanilla_seconds + context_path,
            "bayes_covariate_prediction": (
                vanilla_seconds + context_path + bayes_seconds
            ),
            **{
                method: (
                vanilla_seconds
                + context_path
                + components["neighbor_model_forecast_seconds"]
                + ridge_seconds
                )
                for method in RIDGE_VARIANTS
            },
        }
    timings["selected_adaptation"] = timings.get(selected_method, vanilla_seconds)
    return timings, components


def predict_adaptation_family(
    prepared_path: str | Path,
    fit_extraction_path: str | Path,
    model_path: str | Path,
    eval_extraction_path: str | Path,
    vanilla_path: str | Path,
    config: PredictionConfig,
    output_dir: str | Path,
) -> Path:
    """Produce aligned candidate forecasts and the validation-selected forecast."""

    config.validate()
    prepared = PreparedDataset(prepared_path)
    extraction_root, extraction = open_extraction(fit_extraction_path)
    model_root, model = open_adaptation_model(model_path)
    eval_root, eval_extraction = open_eval_extraction(eval_extraction_path)
    vanilla_root, vanilla_manifest = open_vanilla_test_forecasts(vanilla_path)
    if extraction["prepared_signature"] != prepared.signature:
        raise ValueError("extraction and shared TIME data do not match")
    if model["extraction_signature"] != extraction["signature"]:
        raise ValueError("frozen ridge model and extraction do not match")
    if eval_extraction["adaptation_signature"] != model["signature"]:
        raise ValueError("evaluation extraction and frozen ridge model do not match")
    if vanilla_manifest["prepared_signature"] != prepared.signature:
        raise ValueError("vanilla forecasts and shared TIME data do not match")
    identity = {
        "schema_version": POINT_PREDICTION_SCHEMA,
        "methods": list(ADAPTATION_METHODS),
        "prepared_signature": prepared.signature,
        "extraction_signature": extraction["signature"],
        "adaptation_signature": model["signature"],
        "eval_extraction_signature": eval_extraction["signature"],
        "vanilla_signature": vanilla_manifest["signature"],
        "config": asdict(config),
    }
    signature = _canonical_hash(identity)
    root = Path(output_dir).expanduser().resolve()
    manifest_path = root / "prediction_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected: list[str] = []
        for value in dict(existing.get("files", {})).values():
            if isinstance(value, dict):
                expected.extend(map(str, value.values()))
            else:
                expected.append(str(value))
        if (
            existing.get("signature") == signature
            and existing.get("status") == "completed"
            and all((root / relative).is_file() for relative in expected)
        ):
            return manifest_path
        raise FileExistsError(f"prediction directory already differs: {root}")

    arrays = ExtractionArrays(extraction_root, extraction)
    vanilla = np.load(
        vanilla_root / vanilla_manifest["arrays"]["predictions"], mmap_mode="r"
    )
    prediction_stores = {
        method: _memmap(root / f"{method}.npy", vanilla.shape, np.float32)
        for method in ADAPTATION_METHODS
    }
    eligible_store = _memmap(root / "rag_eligible.npy", (len(vanilla),), bool)
    reason_store = _memmap(root / "fallback_reason.npy", (len(vanilla),), np.uint8)
    fallback_reason = model.get("fallback_reason")
    ridge_seconds = 0.0
    bayes_seconds = 0.0
    selected_ks = tuple(map(int, model.get("evaluation_k_values", ())))
    selected_method = str(model["selected"]["method"])
    method_selections = dict(model.get("method_selections", {}))
    probability = 0.0
    if fallback_reason is not None:
        for store in prediction_stores.values():
            store[:] = vanilla
        eligible_store[:] = False
        reason_store[:] = 6
    else:
        bayes = json.loads(
            (model_root / model["files"]["bayes_mixture"]).read_text(
                encoding="utf-8"
            )
        )
        probability = float(bayes["probability_covariate_better"])
        coefficient_files = dict(model["files"]["coefficients"])
        coefficients = {
            method: np.load(model_root / relative, allow_pickle=False)
            for method, relative in coefficient_files.items()
        }
        eval_arrays = dict(eval_extraction["arrays"])
        base_eligible = np.load(
            eval_root / eval_arrays["test.rag_eligible"], mmap_mode="r"
        )
        base_reason = np.load(
            eval_root / eval_arrays["test.fallback_reason"], mmap_mode="r"
        )
        if selected_ks:
            contexts = {
                k: np.load(
                    eval_root / eval_arrays[f"test.context_forecast_k{k}"],
                    mmap_mode="r",
                )
                for k in selected_ks
            }
            neighbor_ids = np.load(
                eval_root / eval_arrays["test.neighbor_id"], mmap_mode="r"
            )
            new_forecast_ids = np.load(
                eval_root / eval_arrays["datastore.selected_forecast_id"],
                mmap_mode="r",
            )
            new_forecast_values = np.load(
                eval_root / eval_arrays["datastore.selected_forecast"],
                mmap_mode="r",
            )
            forecast_ids = np.concatenate(
                (np.asarray(arrays.forecast_ids), np.asarray(new_forecast_ids))
            )
            forecast_values = np.concatenate(
                (np.asarray(arrays.forecast_values), np.asarray(new_forecast_values)),
                axis=0,
            )
            order = np.argsort(forecast_ids)
            forecast_ids = forecast_ids[order]
            forecast_values = forecast_values[order]
        control_k = int(model["selected"]["k"])
        if control_k == 0 and selected_ks:
            bayes_k = int(method_selections["bayes_covariate_prediction"]["k"])
            control_k = bayes_k if bayes_k > 0 else selected_ks[0]
        for start in range(0, len(vanilla), config.chunk_size):
            stop = min(start + config.chunk_size, len(vanilla))
            chunk_vanilla = np.asarray(vanilla[start:stop])
            chunk_predictions = {
                method: np.array(chunk_vanilla, copy=True)
                for method in ADAPTATION_METHODS
            }
            chunk_eligible = np.zeros(stop - start, dtype=bool)
            for k in selected_ks:
                started = perf_counter()
                selected_ids = np.asarray(neighbor_ids[start:stop, :k])
                candidate = np.asarray(base_eligible[start:stop], dtype=bool) & np.all(
                    selected_ids >= 0, axis=1
                )
                design = np.zeros(
                    (*chunk_vanilla.shape, 2 + 2 * k), dtype=np.float32
                )
                final_eligible = np.zeros(stop - start, dtype=bool)
                positions = np.flatnonzero(candidate)
                if len(positions):
                    ids = selected_ids[positions]
                    forecast_positions = np.searchsorted(forecast_ids, ids)
                    if (
                        np.any(forecast_positions >= len(forecast_ids))
                        or not np.array_equal(forecast_ids[forecast_positions], ids)
                    ):
                        raise ValueError(
                            "evaluation extraction is missing a selected neighbor forecast"
                        )
                    candidate_design, _ = full_ridge_design(
                        chunk_vanilla[positions],
                        np.asarray(contexts[k][start:stop])[positions],
                        arrays.datastore_target[ids],
                        forecast_values[forecast_positions],
                        np.zeros_like(chunk_vanilla[positions]),
                    )
                    complete = np.isfinite(candidate_design).reshape(
                        len(candidate_design), -1
                    ).all(axis=1)
                    accepted = positions[complete]
                    design[accepted] = candidate_design[complete]
                    final_eligible[accepted] = True
                chunk_eligible |= final_eligible
                if k == control_k:
                    chunk_predictions["covariate_prediction"][final_eligible] = (
                        np.asarray(contexts[k][start:stop])[final_eligible]
                    )
                bayes_selection = method_selections["bayes_covariate_prediction"]
                if int(bayes_selection["k"]) == k:
                    bayes_started = perf_counter()
                    chunk_predictions["bayes_covariate_prediction"][final_eligible] = (
                        (1.0 - probability) * chunk_vanilla[final_eligible]
                        + probability
                        * np.asarray(contexts[k][start:stop])[final_eligible]
                    )
                    bayes_seconds += perf_counter() - bayes_started
                for method in RIDGE_VARIANTS:
                    selection = method_selections[method]
                    if int(selection["k"]) != k:
                        continue
                    indices = ridge_feature_indices(method, k)
                    chunk_predictions[method] = full_ridge_predict_with_fallback(
                        chunk_vanilla,
                        design[..., indices],
                        coefficients[method],
                        final_eligible,
                    )
                ridge_seconds += perf_counter() - started
            chunk_predictions["selected_adaptation"] = np.array(
                chunk_predictions.get(selected_method, chunk_vanilla), copy=True
            )
            for method, values in chunk_predictions.items():
                prediction_stores[method][start:stop] = values
            final_reason = np.asarray(base_reason[start:stop]).copy()
            final_reason[np.asarray(base_eligible[start:stop]) & ~chunk_eligible] = 5
            eligible_store[start:stop] = chunk_eligible
            reason_store[start:stop] = final_reason
    for store in prediction_stores.values():
        store.flush()
    eligible_store.flush()
    reason_store.flush()
    inference_seconds, components = _inference_timings(
        vanilla_manifest,
        eval_extraction,
        selected_ks,
        selected_method,
        ridge_seconds,
        bayes_seconds,
    )
    manifest: dict[str, Any] = {
        **identity,
        "format": "adaptime_family_predictions",
        "signature": signature,
        "status": "completed",
        "forecast_type": "point",
        "context_length": prepared.context_length,
        "context_policy": "vanilla_flexible_adaptation_fixed",
        "prediction_length": prepared.prediction_length,
        "selected": model["selected"],
        "method_selections": method_selections,
        "evaluation_k_values": list(selected_ks),
        "bayes_probability_covariate_better": probability,
        "fallback_reason": fallback_reason,
        "rag_coverage": {
            "eligible_windows": int(np.count_nonzero(eligible_store)),
            "total_windows": int(len(eligible_store)),
        },
        "fallback_reason_codes": {
            str(code): label for code, label in FALLBACK_REASONS.items()
        },
        "inference_seconds": inference_seconds,
        "timing_components": components,
        "files": {
            "predictions": {
                method: f"{method}.npy" for method in ADAPTATION_METHODS
            },
            "rag_eligible": "rag_eligible.npy",
            "fallback_reason": "fallback_reason.npy",
        },
    }
    _atomic_json(manifest_path, manifest)
    return manifest_path


def open_point_predictions(path: str | Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = Path(path).expanduser().resolve()
    if manifest_path.is_dir():
        manifest_path = manifest_path / "prediction_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != POINT_PREDICTION_SCHEMA
        or manifest.get("status") != "completed"
        or manifest.get("format")
        not in {"adaptime_point_predictions", "adaptime_family_predictions"}
    ):
        raise ValueError("not a completed Adaptime prediction artifact")
    return manifest_path.parent, manifest
