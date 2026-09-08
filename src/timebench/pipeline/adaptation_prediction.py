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
from timebench.pipeline.adaptime_extraction import open_extraction
from timebench.pipeline.adaptime_training import ExtractionArrays, open_adaptation_model


POINT_PREDICTION_SCHEMA = 1


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


def _effective_inference_seconds(
    extraction: dict[str, object], selected_k: int | None, ridge_seconds: float
) -> tuple[float, dict[str, float]]:
    measured = dict(extraction["timing_seconds"])
    vanilla = float(measured["test.vanilla_forecast_seconds"])
    if selected_k is None:
        return vanilla, {"vanilla_model_forecast_seconds": vanilla}
    components = {
        "query_representation_seconds": float(
            measured["test.representation_seconds"]
        ),
        "retrieval_seconds": float(measured["test.retrieval_seconds"]),
        "context_construction_seconds": float(
            measured[f"test.context_construction_k{selected_k}_seconds"]
        ),
        "vanilla_model_forecast_seconds": vanilla,
        "covariate_model_forecast_seconds": float(
            measured[f"test.context_forecast_k{selected_k}_seconds"]
        ),
        "ridge_design_and_adjustment_seconds": float(ridge_seconds),
    }
    return float(sum(components.values())), components


def predict_frozen_ridge(
    prepared_path: str | Path,
    extraction_path: str | Path,
    model_path: str | Path,
    config: PredictionConfig,
    output_dir: str | Path,
) -> Path:
    """Produce one point forecast per official TIME test row."""

    config.validate()
    prepared = PreparedDataset(prepared_path)
    extraction_root, extraction = open_extraction(extraction_path)
    model_root, model = open_adaptation_model(model_path)
    if extraction["prepared_signature"] != prepared.signature:
        raise ValueError("extraction and shared TIME data do not match")
    if model["extraction_signature"] != extraction["signature"]:
        raise ValueError("frozen ridge model and extraction do not match")
    identity = {
        "schema_version": POINT_PREDICTION_SCHEMA,
        "method": "full_ridge_shared",
        "prepared_signature": prepared.signature,
        "extraction_signature": extraction["signature"],
        "adaptation_signature": model["signature"],
        "config": asdict(config),
    }
    signature = _canonical_hash(identity)
    root = Path(output_dir).expanduser().resolve()
    manifest_path = root / "prediction_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = dict(existing.get("files", {})).values()
        if (
            existing.get("signature") == signature
            and existing.get("status") == "completed"
            and all((root / relative).is_file() for relative in expected)
        ):
            return manifest_path
        raise FileExistsError(f"prediction directory already differs: {root}")

    arrays = ExtractionArrays(extraction_root, extraction)
    vanilla = arrays.open("test.vanilla")
    prediction = _memmap(root / "predictions.npy", vanilla.shape, np.float32)
    eligible_store = _memmap(root / "rag_eligible.npy", (len(vanilla),), bool)
    reason_store = _memmap(root / "fallback_reason.npy", (len(vanilla),), np.uint8)
    fallback_reason = model.get("fallback_reason")
    ridge_seconds = 0.0
    if fallback_reason is not None:
        prediction[:] = vanilla
        eligible_store[:] = False
        reason_store[:] = 2
        selected_k: int | None = None
    else:
        selected_k = int(model["selected"]["k"])
        coefficients = np.load(
            model_root / model["files"]["coefficients"], allow_pickle=False
        )
        context = arrays.open(f"test.context_forecast_k{selected_k}")
        neighbor_ids = arrays.open("test.neighbor_id")
        base_eligible = arrays.open("test.rag_eligible")
        base_reason = arrays.open("test.fallback_reason")
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
                candidate_design, _ = full_ridge_design(
                    chunk_vanilla[positions],
                    np.asarray(context[start:stop])[positions],
                    arrays.datastore_target[ids],
                    arrays.neighbor_forecast(ids),
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
            prediction[start:stop] = full_ridge_predict_with_fallback(
                chunk_vanilla, design, coefficients, final_eligible
            )
            final_reason = np.asarray(base_reason[start:stop]).copy()
            final_reason[np.asarray(base_eligible[start:stop]) & ~candidate] = 2
            final_reason[candidate & ~final_eligible] = 4
            eligible_store[start:stop] = final_eligible
            reason_store[start:stop] = final_reason
            ridge_seconds += perf_counter() - started
    prediction.flush()
    eligible_store.flush()
    reason_store.flush()
    inference_seconds, components = _effective_inference_seconds(
        extraction, selected_k, ridge_seconds
    )
    manifest: dict[str, Any] = {
        **identity,
        "format": "adaptime_point_predictions",
        "signature": signature,
        "status": "completed",
        "forecast_type": "point",
        "context_length": prepared.context_length,
        "prediction_length": prepared.prediction_length,
        "selected": model["selected"],
        "fallback_reason": fallback_reason,
        "rag_coverage": {
            "eligible_windows": int(np.count_nonzero(eligible_store)),
            "total_windows": int(len(eligible_store)),
        },
        "inference_seconds": inference_seconds,
        "timing_components": components,
        "files": {
            "predictions": "predictions.npy",
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
        or manifest.get("format") != "adaptime_point_predictions"
    ):
        raise ValueError("not a completed Adaptime point-prediction artifact")
    return manifest_path.parent, manifest
