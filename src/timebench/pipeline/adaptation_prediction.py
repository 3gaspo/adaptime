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

from timebench.adaptime.ridge import full_ridge_design, full_ridge_predict_with_fallback
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
    "full_ridge_shared",
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
    selected_k: int | None,
    ridge_seconds: float,
    bayes_seconds: float,
) -> tuple[dict[str, float], dict[str, float]]:
    vanilla_seconds = float(
        vanilla["timing_seconds"]["vanilla_model_forecast_seconds"]
    )
    if selected_k is None:
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
    return (
        {
            "vanilla": vanilla_seconds,
            "covariate_prediction": vanilla_seconds + context_path,
            "bayes_covariate_prediction": (
                vanilla_seconds + context_path + bayes_seconds
            ),
            "full_ridge_shared": (
                vanilla_seconds
                + context_path
                + components["neighbor_model_forecast_seconds"]
                + ridge_seconds
            ),
        },
        components,
    )


def predict_adaptation_family(
    prepared_path: str | Path,
    fit_extraction_path: str | Path,
    model_path: str | Path,
    eval_extraction_path: str | Path,
    vanilla_path: str | Path,
    config: PredictionConfig,
    output_dir: str | Path,
) -> Path:
    """Produce the four aligned Adaptime comparison forecasts."""

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
    if fallback_reason is not None:
        for store in prediction_stores.values():
            store[:] = vanilla
        eligible_store[:] = False
        reason_store[:] = 6
        selected_k: int | None = None
        probability = 0.0
    else:
        selected_k = int(model["selected"]["k"])
        bayes = json.loads(
            (model_root / model["files"]["bayes_mixture"]).read_text(
                encoding="utf-8"
            )
        )
        probability = float(bayes["probability_covariate_better"])
        coefficients = np.load(
            model_root / model["files"]["coefficients"], allow_pickle=False
        )
        eval_arrays = dict(eval_extraction["arrays"])
        context = np.load(
            eval_root / eval_arrays["test.context_forecast"], mmap_mode="r"
        )
        neighbor_ids = np.load(
            eval_root / eval_arrays["test.neighbor_id"], mmap_mode="r"
        )
        base_eligible = np.load(
            eval_root / eval_arrays["test.rag_eligible"], mmap_mode="r"
        )
        base_reason = np.load(
            eval_root / eval_arrays["test.fallback_reason"], mmap_mode="r"
        )
        new_forecast_ids = np.load(
            eval_root / eval_arrays["datastore.selected_forecast_id"], mmap_mode="r"
        )
        new_forecast_values = np.load(
            eval_root / eval_arrays["datastore.selected_forecast"], mmap_mode="r"
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
        for start in range(0, len(vanilla), config.chunk_size):
            stop = min(start + config.chunk_size, len(vanilla))
            started = perf_counter()
            chunk_vanilla = np.asarray(vanilla[start:stop])
            selected = np.asarray(neighbor_ids[start:stop, :selected_k])
            candidate = np.asarray(base_eligible[start:stop], dtype=bool) & np.all(
                selected >= 0, axis=1
            )
            design = np.zeros(
                (*chunk_vanilla.shape, len(coefficients)), dtype=np.float32
            )
            final_eligible = np.zeros(stop - start, dtype=bool)
            positions = np.flatnonzero(candidate)
            if len(positions):
                ids = selected[positions]
                forecast_positions = np.searchsorted(forecast_ids, ids)
                if (
                    np.any(forecast_positions >= len(forecast_ids))
                    or not np.array_equal(
                        forecast_ids[forecast_positions], ids
                    )
                ):
                    raise ValueError(
                        "evaluation extraction is missing a selected neighbor forecast"
                    )
                candidate_design, _ = full_ridge_design(
                    chunk_vanilla[positions],
                    np.asarray(context[start:stop])[positions],
                    arrays.datastore_target[ids],
                    forecast_values[forecast_positions],
                    np.zeros_like(chunk_vanilla[positions]),
                )
                complete = np.isfinite(candidate_design).reshape(
                    len(candidate_design), -1
                ).all(axis=1)
                complete &= np.isfinite(chunk_vanilla[positions]).reshape(
                    len(positions), -1
                ).all(axis=1)
                accepted = positions[complete]
                design[accepted] = candidate_design[complete]
                final_eligible[accepted] = True
            ridge_prediction = full_ridge_predict_with_fallback(
                chunk_vanilla, design, coefficients, final_eligible
            )
            ridge_seconds += perf_counter() - started
            covariate_prediction = np.array(chunk_vanilla, copy=True)
            covariate_prediction[final_eligible] = np.asarray(
                context[start:stop]
            )[final_eligible]
            bayes_started = perf_counter()
            bayes_prediction = np.array(chunk_vanilla, copy=True)
            bayes_prediction[final_eligible] = (
                (1.0 - probability) * chunk_vanilla[final_eligible]
                + probability * covariate_prediction[final_eligible]
            )
            bayes_seconds += perf_counter() - bayes_started
            prediction_stores["vanilla"][start:stop] = chunk_vanilla
            prediction_stores["covariate_prediction"][start:stop] = (
                covariate_prediction
            )
            prediction_stores["bayes_covariate_prediction"][start:stop] = (
                bayes_prediction
            )
            prediction_stores["full_ridge_shared"][start:stop] = ridge_prediction
            final_reason = np.asarray(base_reason[start:stop]).copy()
            final_reason[np.asarray(base_eligible[start:stop]) & ~candidate] = 2
            final_reason[candidate & ~final_eligible] = 5
            eligible_store[start:stop] = final_eligible
            reason_store[start:stop] = final_reason
    for store in prediction_stores.values():
        store.flush()
    eligible_store.flush()
    reason_store.flush()
    inference_seconds, components = _inference_timings(
        vanilla_manifest,
        eval_extraction,
        selected_k,
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
