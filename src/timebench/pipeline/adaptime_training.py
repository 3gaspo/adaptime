"""Streaming fit and validation selection for frozen Adaptime ridge models."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from timebench.adaptime.ridge import (
    FullRidgeStatistics,
    full_ridge_design,
    full_ridge_feature_names,
)
from timebench.evaluation.adaptation_data import PreparedDataset
from timebench.pipeline.adaptime_extraction import open_extraction


ADAPTATION_MODEL_SCHEMA = 1
PRIMARY_K = 10
PRIMARY_ALPHA = 1e-2


@dataclass(frozen=True)
class RidgeTrainingConfig:
    """Selection grid for the one shared, no-intercept full ridge."""

    k_values: tuple[int, ...] = (1, 5, 10, 15)
    alpha_values: tuple[float, ...] = (1e-3, 1e-2, 1e-1)
    chunk_size: int = 1024
    seed: int = 1
    minimum_training_date_ratio: float = 1.0
    minimum_validation_date_ratio: float = 0.1
    default_k: int = PRIMARY_K
    default_alpha: float = PRIMARY_ALPHA

    def validate(self) -> None:
        if not self.k_values or any(int(k) <= 0 for k in self.k_values):
            raise ValueError("k_values must contain positive integers")
        if tuple(sorted(set(self.k_values))) != self.k_values:
            raise ValueError("k_values must be sorted and unique")
        if not self.alpha_values or any(float(alpha) < 0 for alpha in self.alpha_values):
            raise ValueError("alpha_values must contain non-negative values")
        if len(set(map(float, self.alpha_values))) != len(self.alpha_values):
            raise ValueError("alpha_values must be unique")
        if int(self.chunk_size) <= 0:
            raise ValueError("chunk_size must be positive")
        if float(self.minimum_training_date_ratio) < 0:
            raise ValueError("minimum_training_date_ratio must be non-negative")
        if float(self.minimum_validation_date_ratio) < 0:
            raise ValueError("minimum_validation_date_ratio must be non-negative")
        if int(self.default_k) not in self.k_values:
            raise ValueError("default_k must be included in k_values")
        if float(self.default_alpha) not in set(map(float, self.alpha_values)):
            raise ValueError("default_alpha must be included in alpha_values")


def _canonical_hash(value: dict[str, object]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


class ExtractionArrays:
    """Memory-mapped access to one immutable extraction artifact."""

    def __init__(self, root: Path, manifest: dict[str, object]) -> None:
        self.root = root
        self.manifest = manifest
        self.paths = dict(manifest["arrays"])
        self.datastore_target = self.open("datastore.target")
        self.forecast_ids = self.open("datastore.selected_forecast_id")
        self.forecast_values = self.open("datastore.selected_forecast")

    def open(self, name: str) -> np.ndarray:
        return np.load(self.root / self.paths[name], mmap_mode="r")

    def neighbor_forecast(self, neighbor_ids: np.ndarray) -> np.ndarray:
        positions = np.searchsorted(self.forecast_ids, neighbor_ids)
        if np.any(positions >= len(self.forecast_ids)) or not np.array_equal(
            np.asarray(self.forecast_ids[positions]), np.asarray(neighbor_ids)
        ):
            raise ValueError("extraction is missing a selected neighbor forecast")
        return np.asarray(self.forecast_values[positions])


def split_statistics_grid(
    arrays: ExtractionArrays,
    split: str,
    k_values: tuple[int, ...],
    chunk_size: int,
    *,
    date_ticks: np.ndarray,
    include_vanilla_fallback: bool = False,
) -> tuple[dict[int, FullRidgeStatistics], dict[int, dict[str, int]]]:
    """Accumulate every K from one bounded pass over a split's neighbors."""

    vanilla = arrays.open(f"{split}.vanilla")
    contexts = {
        k: arrays.open(f"{split}.context_forecast_k{k}") for k in k_values
    }
    target = arrays.open(f"{split}.target")
    scale = arrays.open(f"{split}.msse_scale")
    neighbor_ids = arrays.open(f"{split}.neighbor_id")
    rag_eligible = arrays.open(f"{split}.rag_eligible")
    dates = np.asarray(date_ticks, dtype=np.int64)
    if len(dates) != len(target):
        raise ValueError(f"{split} calendar dates do not match extracted rows")
    statistics = {
        k: FullRidgeStatistics(features=2 + 2 * int(k)) for k in k_values
    }
    coverage = {
        k: {
            "total_windows": int(len(target)),
            "label_complete_windows": 0,
            "adapted_windows": 0,
            "vanilla_fallback_windows": 0,
            "excluded_label_windows": 0,
            "total_dates": int(len(np.unique(dates))),
            "adapted_dates": 0,
        }
        for k in k_values
    }
    adapted_dates = {k: set() for k in k_values}
    for start in range(0, len(target), int(chunk_size)):
        stop = min(start + int(chunk_size), len(target))
        chunk_target = np.asarray(target[start:stop])
        chunk_vanilla = np.asarray(vanilla[start:stop])
        chunk_scale = np.asarray(scale[start:stop])
        label_complete = (
            np.isfinite(chunk_target).reshape(stop - start, -1).all(axis=1)
            & np.isfinite(chunk_vanilla).reshape(stop - start, -1).all(axis=1)
            & np.isfinite(chunk_scale).reshape(stop - start, -1).all(axis=1)
        )
        for k in k_values:
            selected = np.asarray(neighbor_ids[start:stop, :k])
            candidate = (
                np.asarray(rag_eligible[start:stop], dtype=bool)
                & label_complete
                & np.all(selected >= 0, axis=1)
            )
            candidate_positions = np.flatnonzero(candidate)
            adapted = np.zeros(stop - start, dtype=bool)
            if len(candidate_positions):
                candidate_ids = selected[candidate_positions]
                design, residual = full_ridge_design(
                    chunk_vanilla[candidate_positions],
                    np.asarray(contexts[k][start:stop])[candidate_positions],
                    arrays.datastore_target[candidate_ids],
                    arrays.neighbor_forecast(candidate_ids),
                    chunk_target[candidate_positions],
                )
                complete_design = (
                    np.isfinite(design).reshape(len(design), -1).all(axis=1)
                    & np.isfinite(residual).reshape(len(residual), -1).all(axis=1)
                )
                complete_positions = candidate_positions[complete_design]
                if len(complete_positions):
                    statistics[k].update(
                        design[complete_design],
                        residual[complete_design],
                        scale=chunk_scale[complete_positions],
                    )
                    adapted[complete_positions] = True
                    adapted_dates[k].update(
                        map(int, dates[start:stop][complete_positions])
                    )
            fallback = label_complete & ~adapted
            if include_vanilla_fallback and np.any(fallback):
                fallback_residual = chunk_target[fallback] - chunk_vanilla[fallback]
                zero_design = np.zeros(
                    (*fallback_residual.shape, statistics[k].features),
                    dtype=np.float64,
                )
                statistics[k].update(
                    zero_design,
                    fallback_residual,
                    scale=chunk_scale[fallback],
                )
            coverage[k]["label_complete_windows"] += int(np.count_nonzero(label_complete))
            coverage[k]["adapted_windows"] += int(np.count_nonzero(adapted))
            coverage[k]["vanilla_fallback_windows"] += (
                int(np.count_nonzero(fallback)) if include_vanilla_fallback else 0
            )
            coverage[k]["excluded_label_windows"] += int(
                len(label_complete) - np.count_nonzero(label_complete)
            )
    for k in k_values:
        coverage[k]["adapted_dates"] = len(adapted_dates[k])
    return statistics, coverage


def fit_full_ridge(
    prepared_path: str | Path,
    extraction_path: str | Path,
    config: RidgeTrainingConfig,
    output_dir: str | Path,
) -> Path:
    """Fit on adaptation-train, select on validation, and freeze coefficients."""

    config.validate()
    prepared = PreparedDataset(prepared_path)
    extraction_root, extraction_manifest = open_extraction(extraction_path)
    if extraction_manifest["prepared_signature"] != prepared.signature:
        raise ValueError("extraction and prepared TIME windows do not match")
    extracted_config = dict(extraction_manifest["config"])
    if max(config.k_values) > int(extracted_config["max_k"]):
        raise ValueError("requested K exceeds the extracted neighbor count")
    context_k = set(map(int, extracted_config["context_k"]))
    missing_context = sorted(set(config.k_values) - context_k)
    if missing_context:
        raise ValueError(f"context forecasts were not extracted for K={missing_context}")

    identity = {
        "schema_version": ADAPTATION_MODEL_SCHEMA,
        "extraction_signature": extraction_manifest["signature"],
        "method": "full_ridge_shared",
        "config": asdict(config),
    }
    signature = _canonical_hash(identity)
    root = Path(output_dir).expanduser().resolve()
    manifest_path = root / "model_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = dict(existing.get("files", {}))
        if (
            existing.get("signature") == signature
            and existing.get("status") == "completed"
            and files.get("selection")
            and all((root / relative).is_file() for relative in files.values())
        ):
            return manifest_path
        raise FileExistsError(f"training directory already contains a different run: {root}")
    root.mkdir(parents=True, exist_ok=True)

    arrays = ExtractionArrays(extraction_root, extraction_manifest)
    selection_rows: list[dict[str, object]] = []
    best: tuple[float, int, float, np.ndarray] | None = None
    train_by_k, train_coverage = split_statistics_grid(
        arrays,
        "adaptation_train",
        config.k_values,
        config.chunk_size,
        date_ticks=prepared.calendar_ticks("adaptation_train"),
    )
    validation_by_k, validation_coverage = split_statistics_grid(
        arrays,
        "adaptation_validation",
        config.k_values,
        config.chunk_size,
        date_ticks=prepared.calendar_ticks("adaptation_validation"),
        include_vanilla_fallback=True,
    )
    test_dates = int(len(np.unique(prepared.calendar_ticks("test"))))
    training_date_limit = float(config.minimum_training_date_ratio) * test_dates
    validation_date_limit = float(config.minimum_validation_date_ratio) * test_dates
    default_k = int(config.default_k)
    default_alpha = float(config.default_alpha)
    default_training_dates = int(train_coverage[default_k]["adapted_dates"])
    if default_training_dates <= training_date_limit:
        reason = (
            f"K={default_k} has {default_training_dates} valid training dates; "
            f"requires more than {training_date_limit:g} for {test_dates} test dates"
        )
        _atomic_json(
            root / "selection.json",
            {
                "criterion": "vanilla_fallback",
                "fallback_reason": reason,
                "primary_configuration": {"k": default_k, "alpha": default_alpha},
                "date_support": {
                    "test_dates": test_dates,
                    "minimum_training_dates_exclusive": training_date_limit,
                    "minimum_validation_dates_exclusive": validation_date_limit,
                    "default_k_training_dates": default_training_dates,
                    "default_k_validation_dates": int(
                        validation_coverage[default_k]["adapted_dates"]
                    ),
                },
                "candidates": [],
            },
        )
        model = {
            **identity,
            "format": "adaptime_full_ridge_model",
            "signature": signature,
            "status": "completed",
            "protocol": "vanilla_fallback_when_valid_training_dates_are_insufficient",
            "selected": {"k": None, "alpha": None, "validation_msse": None},
            "fallback_reason": reason,
            "primary_configuration": {"k": default_k, "alpha": default_alpha},
            "date_support": {
                "test_dates": test_dates,
                "minimum_training_dates_exclusive": training_date_limit,
                "minimum_validation_dates_exclusive": validation_date_limit,
            },
            "feature_names": [],
            "coverage": {
                "adaptation_train": {
                    str(k): value for k, value in train_coverage.items()
                },
                "adaptation_validation": {
                    str(k): value for k, value in validation_coverage.items()
                },
            },
            "files": {"selection": "selection.json"},
        }
        _atomic_json(manifest_path, model)
        return manifest_path

    default_validation_dates = int(validation_coverage[default_k]["adapted_dates"])
    if default_validation_dates <= validation_date_limit:
        selected_k = default_k
        selected_alpha = default_alpha
        validation_msse: float | None = None
        coefficients = train_by_k[selected_k].solve(selected_alpha)
        selection_criterion = "default_sparse_validation"
        selection_rows.append(
            {
                "k": selected_k,
                "alpha": selected_alpha,
                "validation_msse": None,
            }
        )
    else:
        selection_criterion = "adaptation_validation_msse"
        candidate_ks = [
            k
            for k in config.k_values
            if int(train_coverage[k]["adapted_dates"]) > training_date_limit
            and int(validation_coverage[k]["adapted_dates"]) > validation_date_limit
        ]
        for k in candidate_ks:
            train_statistics = train_by_k[k]
            validation_statistics = validation_by_k[k]
            for alpha in config.alpha_values:
                candidate_coefficients = train_statistics.solve(alpha)
                candidate_msse = validation_statistics.mean_squared_error(
                    candidate_coefficients
                )
                selection_rows.append(
                    {
                        "k": int(k),
                        "alpha": float(alpha),
                        "validation_msse": float(candidate_msse),
                    }
                )
                candidate = (
                    float(candidate_msse),
                    int(k),
                    float(alpha),
                    candidate_coefficients,
                )
                if best is None or candidate[:3] < best[:3]:
                    best = candidate
        assert best is not None
        validation_msse, selected_k, selected_alpha, coefficients = best

    coefficient_path = root / "coefficients.npy"
    with coefficient_path.open("wb") as stream:
        np.save(stream, coefficients, allow_pickle=False)
    _atomic_json(
        root / "selection.json",
        {
            "criterion": selection_criterion,
            "selected_k": selected_k,
            "selected_alpha": selected_alpha,
            "selected_validation_msse": validation_msse,
            "primary_configuration": {"k": default_k, "alpha": default_alpha},
            "date_support": {
                "test_dates": test_dates,
                "minimum_training_dates_exclusive": training_date_limit,
                "minimum_validation_dates_exclusive": validation_date_limit,
                "selected_k_training_dates": int(
                    train_coverage[selected_k]["adapted_dates"]
                ),
                "selected_k_validation_dates": int(
                    validation_coverage[selected_k]["adapted_dates"]
                ),
            },
            "candidates": selection_rows,
        },
    )

    model: dict[str, object] = {
        **identity,
        "format": "adaptime_full_ridge_model",
        "signature": signature,
        "status": "completed",
        "protocol": (
            "fit_once_valid_adaptation_train_select_when_validation_sufficient_"
            "otherwise_use_default_freeze_before_time_test"
        ),
        "selected": {
            "k": selected_k,
            "alpha": selected_alpha,
            "validation_msse": validation_msse,
        },
        "primary_configuration": {"k": default_k, "alpha": default_alpha},
        "date_support": {
            "test_dates": test_dates,
            "minimum_training_dates_exclusive": training_date_limit,
            "minimum_validation_dates_exclusive": validation_date_limit,
        },
        "feature_names": full_ridge_feature_names(selected_k),
        "coverage": {
            "adaptation_train": {str(k): value for k, value in train_coverage.items()},
            "adaptation_validation": {
                str(k): value for k, value in validation_coverage.items()
            },
        },
        "files": {
            "coefficients": coefficient_path.name,
            "selection": "selection.json",
        },
    }
    _atomic_json(manifest_path, model)
    return manifest_path


def open_adaptation_model(path: str | Path) -> tuple[Path, dict[str, object]]:
    manifest_path = Path(path).expanduser().resolve()
    if manifest_path.is_dir():
        manifest_path = manifest_path / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != ADAPTATION_MODEL_SCHEMA
        or manifest.get("format") != "adaptime_full_ridge_model"
        or manifest.get("status") != "completed"
    ):
        raise ValueError("Adaptime model is not a completed schema-1 full ridge artifact")
    return manifest_path.parent, manifest
