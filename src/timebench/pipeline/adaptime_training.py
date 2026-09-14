"""Streaming fit and validation selection for frozen Adaptime ridge models."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from timebench.adaptime.ridge import (
    PER_VARIATE_RIDGE,
    RIDGE_VARIANTS,
    FullRidgeStatistics,
    full_ridge_design,
    ridge_feature_indices,
    ridge_feature_names,
)
from timebench.evaluation.adaptation_data import PreparedDataset
from timebench.pipeline.adaptime_extraction import open_extraction


ADAPTATION_MODEL_SCHEMA = 1
PRIMARY_K = 10
PRIMARY_ALPHA = 1e-2


@dataclass(frozen=True)
class RidgeTrainingConfig:
    """Selection grid for shared and per-variate no-intercept Ridge fits."""

    k_values: tuple[int, ...] = (1, 5, 10, 15)
    alpha_values: tuple[float, ...] = (1e-3, 1e-2, 1e-1)
    chunk_size: int = 1024
    seed: int = 1
    minimum_training_window_ratio: float = 1.0
    minimum_validation_window_ratio: float = 0.1
    default_k: int = PRIMARY_K
    default_alpha: float = PRIMARY_ALPHA
    fitting_scopes: tuple[str, ...] = ("all", "same_series")

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
        if float(self.minimum_training_window_ratio) < 0:
            raise ValueError("minimum_training_window_ratio must be non-negative")
        if float(self.minimum_validation_window_ratio) < 0:
            raise ValueError("minimum_validation_window_ratio must be non-negative")
        if int(self.default_k) not in self.k_values:
            raise ValueError("default_k must be included in k_values")
        if float(self.default_alpha) not in set(map(float, self.alpha_values)):
            raise ValueError("default_alpha must be included in alpha_values")
        if (
            not self.fitting_scopes
            or tuple(dict.fromkeys(self.fitting_scopes)) != self.fitting_scopes
            or any(value not in {"all", "same_series"} for value in self.fitting_scopes)
        ):
            raise ValueError("fitting_scopes must contain unique all/same_series values")


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
    row_positions: np.ndarray | None = None,
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
    rows = (
        np.arange(len(target), dtype=np.int64)
        if row_positions is None
        else np.asarray(row_positions, dtype=np.int64)
    )
    statistics = {
        k: FullRidgeStatistics(features=2 + 2 * int(k)) for k in k_values
    }
    coverage = {
        k: {
            "total_windows": int(len(rows)),
            "label_complete_windows": 0,
            "adapted_windows": 0,
            "vanilla_fallback_windows": 0,
            "excluded_label_windows": 0,
            "total_dates": int(len(np.unique(dates[rows]))),
            "adapted_dates": 0,
        }
        for k in k_values
    }
    adapted_dates = {k: set() for k in k_values}
    for start in range(0, len(rows), int(chunk_size)):
        stop = min(start + int(chunk_size), len(rows))
        selected_rows = rows[start:stop]
        chunk_target = np.asarray(target[selected_rows])
        chunk_vanilla = np.asarray(vanilla[selected_rows])
        chunk_scale = np.asarray(scale[selected_rows])
        label_complete = (
            np.isfinite(chunk_target).reshape(stop - start, -1).all(axis=1)
            & np.isfinite(chunk_vanilla).reshape(stop - start, -1).all(axis=1)
            & np.isfinite(chunk_scale).reshape(stop - start, -1).all(axis=1)
        )
        for k in k_values:
            selected = np.asarray(neighbor_ids[selected_rows, :k])
            candidate = (
                np.asarray(rag_eligible[selected_rows], dtype=bool)
                & label_complete
                & np.all(selected >= 0, axis=1)
            )
            candidate_positions = np.flatnonzero(candidate)
            adapted = np.zeros(stop - start, dtype=bool)
            if len(candidate_positions):
                candidate_ids = selected[candidate_positions]
                design, residual = full_ridge_design(
                    chunk_vanilla[candidate_positions],
                    np.asarray(contexts[k][selected_rows])[candidate_positions],
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
                        map(int, dates[selected_rows][complete_positions])
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


def covariate_win_evidence(
    arrays: ExtractionArrays,
    split: str,
    k: int,
    chunk_size: int,
) -> dict[str, float | int]:
    """Accumulate paired per-window MSSE wins for C against vanilla V."""

    vanilla = arrays.open(f"{split}.vanilla")
    context = arrays.open(f"{split}.context_forecast_k{k}")
    target = arrays.open(f"{split}.target")
    scale = arrays.open(f"{split}.msse_scale")
    neighbor_ids = arrays.open(f"{split}.neighbor_id")
    rag_eligible = arrays.open(f"{split}.rag_eligible")
    trials = 0
    wins = 0.0
    vanilla_loss_sum = 0.0
    covariate_loss_sum = 0.0
    for start in range(0, len(target), int(chunk_size)):
        stop = min(start + int(chunk_size), len(target))
        chunk_vanilla = np.asarray(vanilla[start:stop])
        chunk_context = np.asarray(context[start:stop])
        chunk_target = np.asarray(target[start:stop])
        chunk_scale = np.asarray(scale[start:stop])
        selected = np.asarray(neighbor_ids[start:stop, : int(k)])
        valid = (
            np.asarray(rag_eligible[start:stop], dtype=bool)
            & np.all(selected >= 0, axis=1)
            & np.isfinite(chunk_vanilla).reshape(stop - start, -1).all(axis=1)
            & np.isfinite(chunk_context).reshape(stop - start, -1).all(axis=1)
            & np.isfinite(chunk_target).reshape(stop - start, -1).all(axis=1)
            & np.isfinite(chunk_scale).reshape(stop - start, -1).all(axis=1)
        )
        if not np.any(valid):
            continue
        denominator = np.maximum(chunk_scale[valid], 1e-8)[..., None]
        vanilla_loss = np.mean(
            np.square((chunk_vanilla[valid] - chunk_target[valid]) / denominator),
            axis=(1, 2),
        )
        covariate_loss = np.mean(
            np.square((chunk_context[valid] - chunk_target[valid]) / denominator),
            axis=(1, 2),
        )
        wins += float(np.count_nonzero(covariate_loss < vanilla_loss))
        wins += 0.5 * float(np.count_nonzero(covariate_loss == vanilla_loss))
        trials += int(len(vanilla_loss))
        vanilla_loss_sum += float(vanilla_loss.sum(dtype=np.float64))
        covariate_loss_sum += float(covariate_loss.sum(dtype=np.float64))
    return {
        "trials": trials,
        "covariate_wins_including_half_ties": wins,
        "vanilla_msse_sum": vanilla_loss_sum,
        "covariate_msse_sum": covariate_loss_sum,
    }


def past_target_win_evidence(
    arrays: ExtractionArrays,
    split: str,
    chunk_size: int,
) -> dict[str, float | int]:
    """Accumulate paired MSSE wins for all-other-variate past covariates."""

    vanilla = arrays.open(f"{split}.vanilla")
    candidate = arrays.open(f"{split}.past_target_covariate_forecast")
    available = arrays.open(f"{split}.past_target_covariate_available")
    target = arrays.open(f"{split}.target")
    scale = arrays.open(f"{split}.msse_scale")
    trials = 0
    wins = 0.0
    vanilla_loss_sum = 0.0
    candidate_loss_sum = 0.0
    for start in range(0, len(target), int(chunk_size)):
        stop = min(start + int(chunk_size), len(target))
        chunk_vanilla = np.asarray(vanilla[start:stop])
        chunk_candidate = np.asarray(candidate[start:stop])
        chunk_target = np.asarray(target[start:stop])
        chunk_scale = np.asarray(scale[start:stop])
        valid = (
            np.asarray(available[start:stop], dtype=bool)
            & np.isfinite(chunk_vanilla).reshape(stop - start, -1).all(axis=1)
            & np.isfinite(chunk_candidate).reshape(stop - start, -1).all(axis=1)
            & np.isfinite(chunk_target).reshape(stop - start, -1).all(axis=1)
            & np.isfinite(chunk_scale).reshape(stop - start, -1).all(axis=1)
        )
        if not np.any(valid):
            continue
        denominator = np.maximum(chunk_scale[valid], 1e-8)[..., None]
        vanilla_loss = np.mean(
            np.square((chunk_vanilla[valid] - chunk_target[valid]) / denominator),
            axis=(1, 2),
        )
        candidate_loss = np.mean(
            np.square((chunk_candidate[valid] - chunk_target[valid]) / denominator),
            axis=(1, 2),
        )
        wins += float(np.count_nonzero(candidate_loss < vanilla_loss))
        wins += 0.5 * float(np.count_nonzero(candidate_loss == vanilla_loss))
        trials += int(len(vanilla_loss))
        vanilla_loss_sum += float(vanilla_loss.sum(dtype=np.float64))
        candidate_loss_sum += float(candidate_loss.sum(dtype=np.float64))
    return {
        "trials": trials,
        "past_target_covariate_wins_including_half_ties": wins,
        "vanilla_msse_sum": vanilla_loss_sum,
        "past_target_covariate_msse_sum": candidate_loss_sum,
    }


def past_target_validation_msse(
    arrays: ExtractionArrays,
    split: str,
    probability: float,
    chunk_size: int,
) -> float:
    vanilla = arrays.open(f"{split}.vanilla")
    candidate = arrays.open(f"{split}.past_target_covariate_forecast")
    available = arrays.open(f"{split}.past_target_covariate_available")
    target = arrays.open(f"{split}.target")
    scale = arrays.open(f"{split}.msse_scale")
    squared_error = 0.0
    observations = 0
    for start in range(0, len(target), int(chunk_size)):
        stop = min(start + int(chunk_size), len(target))
        chunk_vanilla = np.asarray(vanilla[start:stop], dtype=np.float64)
        prediction = np.array(chunk_vanilla, copy=True)
        usable = np.asarray(available[start:stop], dtype=bool)
        prediction[usable] = (
            (1.0 - float(probability)) * chunk_vanilla[usable]
            + float(probability)
            * np.asarray(candidate[start:stop], dtype=np.float64)[usable]
        )
        chunk_target = np.asarray(target[start:stop], dtype=np.float64)
        chunk_scale = np.asarray(scale[start:stop], dtype=np.float64)
        valid = (
            np.isfinite(prediction).reshape(stop - start, -1).all(axis=1)
            & np.isfinite(chunk_target).reshape(stop - start, -1).all(axis=1)
            & np.isfinite(chunk_scale).reshape(stop - start, -1).all(axis=1)
        )
        if np.any(valid):
            residual = (prediction[valid] - chunk_target[valid]) / np.maximum(
                chunk_scale[valid], 1e-8
            )[..., None]
            squared_error += float(np.square(residual).sum(dtype=np.float64))
            observations += int(residual.size)
    if observations == 0:
        raise ValueError("cannot score empty past-target covariate validation data")
    return squared_error / observations


def _series_positions(references: np.ndarray, key: tuple[int, int]) -> np.ndarray:
    refs = np.asarray(references, dtype=np.int64).reshape(-1, 3)
    return np.flatnonzero((refs[:, 0] == int(key[0])) & (refs[:, 1] == int(key[1])))


def fit_per_variate_full_ridge(
    arrays: ExtractionArrays,
    prepared: PreparedDataset,
    config: RidgeTrainingConfig,
) -> tuple[
    dict[str, object],
    np.ndarray,
    np.ndarray,
    dict[str, object],
    list[dict[str, object]],
]:
    """Select one K/alpha globally while fitting one coefficient vector per variate."""

    train_refs = prepared.indices("adaptation_train")
    validation_refs = prepared.indices("adaptation_validation")
    test_refs = prepared.indices("test")
    keys = [tuple(map(int, value)) for value in np.unique(test_refs[:, :2], axis=0)]
    statistics: dict[
        tuple[int, int],
        tuple[
            dict[int, FullRidgeStatistics],
            dict[int, dict[str, int]],
            dict[int, FullRidgeStatistics],
            dict[int, dict[str, int]],
            int,
        ],
    ] = {}
    coverage: dict[str, object] = {}
    for key in keys:
        train_positions = _series_positions(train_refs, key)
        validation_positions = _series_positions(validation_refs, key)
        test_windows = int(len(_series_positions(test_refs, key)))
        train_by_k, train_coverage = split_statistics_grid(
            arrays,
            "adaptation_train",
            config.k_values,
            config.chunk_size,
            date_ticks=prepared.calendar_ticks("adaptation_train"),
            row_positions=train_positions,
        )
        validation_by_k, validation_coverage = split_statistics_grid(
            arrays,
            "adaptation_validation",
            config.k_values,
            config.chunk_size,
            date_ticks=prepared.calendar_ticks("adaptation_validation"),
            include_vanilla_fallback=True,
            row_positions=validation_positions,
        )
        statistics[key] = (
            train_by_k,
            train_coverage,
            validation_by_k,
            validation_coverage,
            test_windows,
        )
        coverage[f"{key[0]}:{key[1]}"] = {
            "test_windows": test_windows,
            "adaptation_train": {str(k): value for k, value in train_coverage.items()},
            "adaptation_validation": {
                str(k): value for k, value in validation_coverage.items()
            },
        }

    candidates: list[dict[str, object]] = []
    default_k = int(config.default_k)
    vanilla_error = 0.0
    vanilla_observations = 0
    for _, _, validation_by_k, _, _ in statistics.values():
        validation = validation_by_k[default_k]
        if validation.observations:
            mse = validation.mean_squared_error(
                np.zeros(validation.features, dtype=np.float64)
            )
            vanilla_error += mse * validation.observations
            vanilla_observations += validation.observations
    if vanilla_observations == 0:
        raise ValueError("per-variate Ridge has no complete validation observations")
    best: dict[str, object] = {
        "method": "vanilla",
        "k": 0,
        "alpha": None,
        "validation_msse": vanilla_error / vanilla_observations,
    }
    candidates.append(dict(best))
    fitted_by_candidate: dict[tuple[int, float], dict[tuple[int, int], np.ndarray]] = {}
    for k in config.k_values:
        indices = ridge_feature_indices(PER_VARIATE_RIDGE, int(k))
        for alpha in config.alpha_values:
            error = 0.0
            observations = 0
            fitted: dict[tuple[int, int], np.ndarray] = {}
            fitted_series = 0
            for key, (
                train_by_k,
                train_coverage,
                validation_by_k,
                validation_coverage,
                test_windows,
            ) in statistics.items():
                validation = validation_by_k[int(k)].select_features(indices)
                if not validation.observations:
                    continue
                train_limit = float(config.minimum_training_window_ratio) * test_windows
                validation_limit = float(config.minimum_validation_window_ratio) * test_windows
                if (
                    int(train_coverage[int(k)]["adapted_windows"]) > train_limit
                    and int(validation_coverage[int(k)]["adapted_windows"])
                    > validation_limit
                ):
                    coefficients = train_by_k[int(k)].select_features(indices).solve(
                        float(alpha)
                    )
                    fitted[key] = coefficients
                    fitted_series += 1
                else:
                    coefficients = np.zeros(len(indices), dtype=np.float64)
                error += validation.mean_squared_error(coefficients) * validation.observations
                observations += validation.observations
            if observations == 0:
                continue
            candidate = {
                "method": PER_VARIATE_RIDGE,
                "k": int(k),
                "alpha": float(alpha),
                "validation_msse": error / observations,
                "fitted_variates": fitted_series,
                "total_variates": len(keys),
            }
            candidates.append(candidate)
            fitted_by_candidate[(int(k), float(alpha))] = fitted
            if not fitted_series:
                continue
            candidate_key = (
                float(candidate["validation_msse"]),
                int(candidate["k"]),
                float(candidate["alpha"]),
            )
            best_key = (
                float(best["validation_msse"]),
                int(best["k"]),
                -1.0 if best["alpha"] is None else float(best["alpha"]),
            )
            if candidate_key < best_key:
                best = candidate

    if best["method"] == "vanilla":
        return best, np.empty((0, 2), np.int64), np.empty((0, 0), np.float64), coverage, candidates
    selected_fits = fitted_by_candidate[(int(best["k"]), float(best["alpha"]))]
    selected_keys = np.asarray(sorted(selected_fits), dtype=np.int64).reshape(-1, 2)
    selected_coefficients = np.stack(
        [selected_fits[tuple(map(int, key))] for key in selected_keys]
    )
    return best, selected_keys, selected_coefficients, coverage, candidates


def fit_full_ridge(
    prepared_path: str | Path,
    extraction_path: str | Path,
    config: RidgeTrainingConfig,
    output_dir: str | Path,
) -> Path:
    """Fit shared and per-variate candidates and select each against vanilla."""

    config.validate()
    prepared = PreparedDataset(prepared_path)
    extraction_root, extraction_manifest = open_extraction(extraction_path)
    if extraction_manifest["prepared_signature"] != prepared.signature:
        raise ValueError("extraction and prepared TIME windows do not match")
    extracted_config = dict(extraction_manifest["config"])
    if max(config.k_values) > int(extracted_config["max_k"]):
        raise ValueError("requested K exceeds the extracted neighbor count")
    missing_context = sorted(
        set(config.k_values) - set(map(int, extracted_config["context_k"]))
    )
    if missing_context:
        raise ValueError(f"context forecasts were not extracted for K={missing_context}")

    selection_methods = [
        "vanilla",
        "bayes_covariate_prediction",
        "bayes_past_targets_prediction",
        *RIDGE_VARIANTS,
        PER_VARIATE_RIDGE,
    ]
    identity = {
        "schema_version": ADAPTATION_MODEL_SCHEMA,
        "extraction_signature": extraction_manifest["signature"],
        "method": "adaptime_validation_selector",
        "config": asdict(config),
        "bayes_covariate_protocol": {
            "fit_evidence": "paired_adaptation_train_window_msse_wins",
            "selection_evidence": "adaptation_validation_msse",
            "tie_weight": 0.5,
            "prior": {"alpha": 1.0, "beta": 1.0},
        },
        "selection_methods": selection_methods,
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
            and files.get("bayes_mixture")
            and files.get("bayes_past_targets_mixture")
            and all(
                (root / relative).is_file()
                for value in files.values()
                for relative in (value.values() if isinstance(value, dict) else (value,))
            )
        ):
            return manifest_path
        raise FileExistsError(f"training directory already contains a different run: {root}")
    root.mkdir(parents=True, exist_ok=True)

    arrays = ExtractionArrays(extraction_root, extraction_manifest)
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
    test_windows = int(len(prepared.indices("test")))
    training_window_limit = float(config.minimum_training_window_ratio) * test_windows
    validation_window_limit = float(config.minimum_validation_window_ratio) * test_windows
    default_k = int(config.default_k)
    default_alpha = float(config.default_alpha)
    vanilla_msse = validation_by_k[default_k].mean_squared_error(
        np.zeros(validation_by_k[default_k].features, dtype=np.float64)
    )
    vanilla_candidate: dict[str, object] = {
        "method": "vanilla",
        "k": 0,
        "alpha": None,
        "validation_msse": float(vanilla_msse),
    }
    selection_rows: list[dict[str, object]] = [dict(vanilla_candidate)]
    best_by_method: dict[str, tuple[dict[str, object], np.ndarray | None]] = {
        method: (dict(vanilla_candidate), None)
        for method in ("bayes_covariate_prediction", *RIDGE_VARIANTS)
    }

    candidate_ks = [
        int(k)
        for k in config.k_values
        if "all" in config.fitting_scopes
        and int(train_coverage[k]["adapted_windows"]) > training_window_limit
        and int(validation_coverage[k]["adapted_windows"]) > validation_window_limit
    ]
    bayes_by_k: dict[int, dict[str, object]] = {}
    for k in candidate_ks:
        evidence = covariate_win_evidence(
            arrays, "adaptation_train", int(k), config.chunk_size
        )
        trials = int(evidence["trials"])
        wins = float(evidence["covariate_wins_including_half_ties"])
        posterior_alpha = 1.0 + wins
        posterior_beta = 1.0 + trials - wins
        probability = posterior_alpha / (posterior_alpha + posterior_beta)
        bayes_by_k[k] = {
            "training": evidence,
            "posterior": {"alpha": posterior_alpha, "beta": posterior_beta},
            "probability_covariate_better": probability,
        }
        full_validation = validation_by_k[k]
        mixture_coefficients = np.zeros(full_validation.features, dtype=np.float64)
        mixture_coefficients[0] = -probability
        mixture_coefficients[1] = probability
        bayes_msse = full_validation.mean_squared_error(mixture_coefficients)
        bayes_candidate = {
            "method": "bayes_covariate_prediction",
            "k": k,
            "alpha": None,
            "validation_msse": float(bayes_msse),
            "probability_covariate_better": probability,
        }
        selection_rows.append(bayes_candidate)
        if float(bayes_msse) < float(
            best_by_method["bayes_covariate_prediction"][0]["validation_msse"]
        ):
            best_by_method["bayes_covariate_prediction"] = (bayes_candidate, None)

        for method in RIDGE_VARIANTS:
            indices = ridge_feature_indices(method, k)
            train_statistics = train_by_k[k].select_features(indices)
            validation_statistics = full_validation.select_features(indices)
            for alpha in config.alpha_values:
                candidate_coefficients = train_statistics.solve(alpha)
                candidate_msse = validation_statistics.mean_squared_error(
                    candidate_coefficients
                )
                candidate = {
                    "method": method,
                    "k": k,
                    "alpha": float(alpha),
                    "validation_msse": float(candidate_msse),
                }
                selection_rows.append(candidate)
                current = best_by_method[method][0]
                if (float(candidate_msse), k, float(alpha)) < (
                    float(current["validation_msse"]),
                    int(current["k"]),
                    -1.0 if current["alpha"] is None else float(current["alpha"]),
                ):
                    best_by_method[method] = (candidate, candidate_coefficients)

    method_selections = {
        method: dict(candidate) for method, (candidate, _) in best_by_method.items()
    }
    coefficients_by_method = {
        method: coefficients
        for method, (candidate, coefficients) in best_by_method.items()
        if candidate["method"] != "vanilla" and coefficients is not None
    }

    if "same_series" in config.fitting_scopes:
        per_selection, per_keys, per_coefficients, per_coverage, per_candidates = (
            fit_per_variate_full_ridge(arrays, prepared, config)
        )
    else:
        per_selection = dict(vanilla_candidate)
        per_keys = np.empty((0, 2), dtype=np.int64)
        per_coefficients = np.empty((0, 0), dtype=np.float64)
        per_coverage = {}
        per_candidates = []
    method_selections[PER_VARIATE_RIDGE] = dict(per_selection)
    selection_rows.extend(
        {**candidate, "selection_family": PER_VARIATE_RIDGE}
        for candidate in per_candidates
    )

    past_evidence = past_target_win_evidence(
        arrays, "adaptation_train", config.chunk_size
    )
    past_trials = int(past_evidence["trials"])
    past_wins = float(
        past_evidence["past_target_covariate_wins_including_half_ties"]
    )
    past_probability = (
        (1.0 + past_wins) / (2.0 + past_trials) if past_trials else 0.0
    )
    past_msse = (
        past_target_validation_msse(
            arrays,
            "adaptation_validation",
            past_probability,
            config.chunk_size,
        )
        if past_trials
        else float(vanilla_msse)
    )
    past_candidate = {
        "method": "bayes_past_targets_prediction",
        "k": 0,
        "alpha": None,
        "validation_msse": float(past_msse),
        "probability_past_targets_better": past_probability,
    }
    selection_rows.append(past_candidate)
    method_selections["bayes_past_targets_prediction"] = (
        past_candidate
        if float(past_msse) < float(vanilla_msse)
        else dict(vanilla_candidate)
    )

    selected = dict(vanilla_candidate)
    for method in (
        "bayes_covariate_prediction",
        "bayes_past_targets_prediction",
        *RIDGE_VARIANTS,
        PER_VARIATE_RIDGE,
    ):
        candidate = method_selections[method]
        if float(candidate["validation_msse"]) < float(selected["validation_msse"]):
            selected = dict(candidate)

    coefficient_files: dict[str, str] = {}
    for method, coefficients in coefficients_by_method.items():
        coefficient_path = root / f"{method}_coefficients.npy"
        with coefficient_path.open("wb") as stream:
            np.save(stream, coefficients, allow_pickle=False)
        coefficient_files[method] = coefficient_path.name
    per_series_file: str | None = None
    per_coefficients_file: str | None = None
    if per_selection["method"] != "vanilla":
        per_series_path = root / f"{PER_VARIATE_RIDGE}_series.npy"
        per_coefficients_path = root / f"{PER_VARIATE_RIDGE}_coefficients.npy"
        with per_series_path.open("wb") as stream:
            np.save(stream, per_keys, allow_pickle=False)
        with per_coefficients_path.open("wb") as stream:
            np.save(stream, per_coefficients, allow_pickle=False)
        per_series_file = per_series_path.name
        per_coefficients_file = per_coefficients_path.name

    selected_bayes = method_selections["bayes_covariate_prediction"]
    selected_bayes_k = int(selected_bayes["k"])
    selected_bayes_fit = bayes_by_k.get(selected_bayes_k)
    _atomic_json(
        root / "bayes_mixture.json",
        {
            "method": "bayes_covariate_prediction",
            "status": "vanilla_selected" if selected_bayes_k == 0 else "fitted",
            "selected_k": selected_bayes_k,
            "loss": "paired_per_window_msse",
            "success": "covariate_prediction_msse_below_vanilla_msse",
            "tie_weight": 0.5,
            "prior": {"alpha": 1.0, "beta": 1.0},
            "fit_split": "adaptation_train",
            "selection_split": "adaptation_validation",
            "posterior": None if selected_bayes_fit is None else selected_bayes_fit["posterior"],
            "training": None if selected_bayes_fit is None else selected_bayes_fit["training"],
            "probability_covariate_better": 0.0
            if selected_bayes_fit is None
            else selected_bayes_fit["probability_covariate_better"],
            "validation_msse": selected_bayes["validation_msse"],
            "fits_by_k": {str(k): value for k, value in bayes_by_k.items()},
        },
    )
    _atomic_json(
        root / "bayes_past_targets_mixture.json",
        {
            "method": "bayes_past_targets_prediction",
            "status": (
                "fitted"
                if method_selections["bayes_past_targets_prediction"]["method"]
                != "vanilla"
                else "vanilla_selected"
            ),
            "loss": "paired_per_window_msse",
            "candidate": "chronos2_with_all_other_variates_as_past_covariates",
            "tie_weight": 0.5,
            "prior": {"alpha": 1.0, "beta": 1.0},
            "fit_split": "adaptation_train",
            "selection_split": "adaptation_validation",
            "training": past_evidence,
            "probability_past_targets_better": past_probability,
            "validation_msse": float(past_msse),
        },
    )

    evaluation_k_values = sorted(
        {
            int(value["k"])
            for method, value in method_selections.items()
            if method != "bayes_past_targets_prediction" and int(value["k"]) > 0
        }
    )
    selection_criterion = "adaptation_validation_msse_with_virtual_k0"
    _atomic_json(
        root / "selection.json",
        {
            "criterion": selection_criterion,
            "selected": selected,
            "method_selections": method_selections,
            "evaluation_k_values": evaluation_k_values,
            "primary_configuration": {"k": default_k, "alpha": default_alpha},
            "window_support": {
                "test_windows": test_windows,
                "minimum_training_windows_exclusive": training_window_limit,
                "minimum_validation_windows_exclusive": validation_window_limit,
            },
            "candidates": selection_rows,
        },
    )

    files: dict[str, object] = {
        "coefficients": coefficient_files,
        "selection": "selection.json",
        "bayes_mixture": "bayes_mixture.json",
        "bayes_past_targets_mixture": "bayes_past_targets_mixture.json",
    }
    if per_series_file is not None and per_coefficients_file is not None:
        files["per_variate_series"] = per_series_file
        files["per_variate_coefficients"] = per_coefficients_file
    model: dict[str, object] = {
        **identity,
        "format": "adaptime_full_ridge_model",
        "signature": signature,
        "status": "completed",
        "protocol": (
            "fit_on_adaptation_train_select_every_family_against_virtual_"
            "vanilla_on_validation_freeze_before_time_test"
        ),
        "selected": selected,
        "method_selections": method_selections,
        "evaluation_k_values": evaluation_k_values,
        "primary_configuration": {"k": default_k, "alpha": default_alpha},
        "window_support": {
            "test_windows": test_windows,
            "minimum_training_windows_exclusive": training_window_limit,
            "minimum_validation_windows_exclusive": validation_window_limit,
        },
        "feature_names": {
            method: (
                []
                if method_selections[method]["method"] == "vanilla"
                else ridge_feature_names(method, int(method_selections[method]["k"]))
            )
            for method in (*RIDGE_VARIANTS, PER_VARIATE_RIDGE)
        },
        "coverage": {
            "adaptation_train": {str(k): value for k, value in train_coverage.items()},
            "adaptation_validation": {
                str(k): value for k, value in validation_coverage.items()
            },
            "per_variate": per_coverage,
        },
        "files": files,
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
