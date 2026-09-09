"""Task-level vanilla fallback around the unchanged TS-RAG pipeline."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from timebench.evaluation.adaptation_data import (
    InsufficientAdaptationHistory,
    PreparedDataset,
    validate_global_datastore_requirement,
)
from timebench.pipeline.adaptation_prediction import POINT_PREDICTION_SCHEMA
from timebench.pipeline.adaptime_vanilla import open_vanilla_test_forecasts
from timebench.pipeline.tsrag import TSRAG_SOURCE_COMMIT


TSRAG_FALLBACK_EXTRACTION_FORMAT = "adaptime_tsrag_task_fallback"
TSRAG_FALLBACK_REASON = "insufficient_global_datastore"


def _canonical_hash(value: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def tsrag_task_fallback(prepared_path: str | Path) -> dict[str, str] | None:
    """Return the supported task-level fallback, or validate TS-RAG support."""

    prepared = PreparedDataset(prepared_path)
    try:
        validate_global_datastore_requirement(prepared.manifest)
    except InsufficientAdaptationHistory as error:
        return {"code": TSRAG_FALLBACK_REASON, "message": str(error)}
    return None


def write_tsrag_fallback_extraction(
    prepared_path: str | Path,
    fallback: dict[str, str],
    output_dir: str | Path,
) -> Path:
    """Record that native extraction was intentionally skipped for one task."""

    prepared = PreparedDataset(prepared_path)
    identity = {
        "schema_version": 1,
        "prepared_signature": prepared.signature,
        "source_commit": TSRAG_SOURCE_COMMIT,
        "fallback": dict(fallback),
    }
    signature = _canonical_hash(identity)
    root = Path(output_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("signature") == signature and existing.get("status") == "completed":
            return manifest_path
        raise FileExistsError(f"TS-RAG fallback extraction already differs: {root}")
    _atomic_json(
        manifest_path,
        {
            **identity,
            "format": TSRAG_FALLBACK_EXTRACTION_FORMAT,
            "signature": signature,
            "status": "completed",
            "protocol": "skip_native_tsrag_and_reuse_exact_vanilla_task_forecast",
            "counts": {
                "datastore": int(len(prepared.indices("datastore"))),
                "test": int(len(prepared.indices("test"))),
            },
            "arrays": {},
            "timing_seconds": {
                "datastore_representation_seconds": 0.0,
                "datastore_index_construction_seconds": 0.0,
                "test_representation_seconds": 0.0,
                "test_retrieval_seconds": 0.0,
                "extraction_total_seconds": 0.0,
            },
        },
    )
    return manifest_path


def open_tsrag_task_fallback(path: str | Path) -> dict[str, str] | None:
    manifest_path = Path(path).expanduser().resolve()
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != TSRAG_FALLBACK_EXTRACTION_FORMAT:
        return None
    if manifest.get("schema_version") != 1 or manifest.get("status") != "completed":
        raise ValueError("TS-RAG task fallback is not a completed schema-1 artifact")
    return {str(key): str(value) for key, value in dict(manifest["fallback"]).items()}


def predict_tsrag_vanilla_fallback(
    prepared_path: str | Path,
    vanilla_path: str | Path,
    fallback: dict[str, str],
    output_dir: str | Path,
) -> Path:
    """Copy an existing exact vanilla forecast into a TS-RAG prediction artifact."""

    prepared = PreparedDataset(prepared_path)
    vanilla_root, vanilla = open_vanilla_test_forecasts(vanilla_path)
    if vanilla["prepared_signature"] != prepared.signature:
        raise ValueError("vanilla fallback and shared TIME data do not match")
    identity = {
        "schema_version": POINT_PREDICTION_SCHEMA,
        "method": "tsrag",
        "prepared_signature": prepared.signature,
        "source_commit": TSRAG_SOURCE_COMMIT,
        "vanilla_signature": vanilla["signature"],
        "fallback": dict(fallback),
    }
    signature = _canonical_hash(identity)
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "prediction_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("signature") == signature
            and existing.get("status") == "completed"
            and (root / "predictions.npy").is_file()
        ):
            return manifest_path
        raise FileExistsError(f"TS-RAG fallback prediction already differs: {root}")

    source = np.load(vanilla_root / vanilla["arrays"]["predictions"], mmap_mode="r")
    expected = (len(prepared.indices("test")), 1, prepared.prediction_length)
    if source.shape != expected:
        raise ValueError(f"vanilla fallback has shape {source.shape}, expected {expected}")
    destination = np.lib.format.open_memmap(
        root / "predictions.npy", mode="w+", shape=source.shape, dtype=np.float32
    )
    destination[:] = source
    destination.flush()

    vanilla_seconds = float(
        vanilla["timing_seconds"]["vanilla_model_forecast_seconds"]
    )
    manifest: dict[str, Any] = {
        **identity,
        "format": "adaptime_point_predictions",
        "signature": signature,
        "status": "completed",
        "forecast_type": "point",
        "protocol": "tsrag_task_fallback_to_existing_vanilla_forecast",
        "context_length": prepared.context_length,
        "context_policy": "reused_vanilla_all_available_history_capped_at_model_limit",
        "prediction_length": prepared.prediction_length,
        "seasonality": prepared.seasonality,
        "fallback_reason": dict(fallback),
        "inference_seconds": vanilla_seconds,
        "timing": {
            "unit": "seconds",
            "total_seconds": vanilla_seconds,
            "reused_prediction_artifact": str(Path(vanilla_path).resolve()),
            "vanilla_recomputed": False,
        },
        "files": {"predictions": "predictions.npy"},
    }
    _atomic_json(manifest_path, manifest)
    return manifest_path
