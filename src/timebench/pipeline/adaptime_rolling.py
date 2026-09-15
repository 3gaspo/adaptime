"""Causal per-variate rolling horizon Ridge over vanilla and retrieved targets."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from timebench.adaptime.retrieval import blockwise_topk, context_representation
from timebench.adaptime.ridge import query_scale
from timebench.evaluation.adaptation_data import PreparedDataset
from timebench.evaluation.grid import (
    EVALUATION_GRID_DEFINITION,
    flatten_univariate_grid,
    load_evaluation_grid,
)
from timebench.pipeline.adaptime_cache import SharedWindowCache
from timebench.pipeline.adaptime_vanilla import open_vanilla_test_forecasts


ROLLING_RIDGE_SCHEMA = 1
ROLLING_RIDGE_METHOD = "rolling_y_ridge_horizon"


@dataclass(frozen=True)
class RollingRidgeConfig:
    k: int = 15
    alpha: float = 1.0
    n_fitting_dates: int = 100
    minimum_fitting_dates: int = 64
    fitting_stride: int | None = None
    max_datastore_windows: int = 10_000
    datastore_stride: int | None = None
    representation: str = "instance"
    distance_metric: str = "euclidean"
    minimum_overlap_fraction: float = 0.8
    model_batch_size: int = 64
    datastore_block_size: int = 512
    arrow_cache_items: int = 2

    def validate(self, period: int) -> None:
        positive = {
            "k": self.k,
            "n_fitting_dates": self.n_fitting_dates,
            "minimum_fitting_dates": self.minimum_fitting_dates,
            "max_datastore_windows": self.max_datastore_windows,
            "model_batch_size": self.model_batch_size,
            "datastore_block_size": self.datastore_block_size,
            "arrow_cache_items": self.arrow_cache_items,
        }
        invalid = [name for name, value in positive.items() if int(value) <= 0]
        if invalid:
            raise ValueError(f"positive rolling settings required: {', '.join(invalid)}")
        if int(self.minimum_fitting_dates) > int(self.n_fitting_dates):
            raise ValueError("minimum_fitting_dates cannot exceed n_fitting_dates")
        if float(self.alpha) < 0:
            raise ValueError("alpha must be non-negative")
        if self.representation not in {"raw", "instance", "model"}:
            raise ValueError("unsupported rolling representation")
        if self.distance_metric not in {"euclidean", "cosine"}:
            raise ValueError("unsupported rolling distance metric")
        if not 0.0 < float(self.minimum_overlap_fraction) <= 1.0:
            raise ValueError("minimum_overlap_fraction must be in (0, 1]")
        for name, value in (
            ("fitting_stride", self.fitting_stride or period),
            ("datastore_stride", self.datastore_stride or period),
        ):
            if int(value) <= 0 or int(value) % int(period):
                raise ValueError(f"{name} must be a positive multiple of the period")


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


def _represent(
    shared_cache: SharedWindowCache, contexts: np.ndarray, mode: str
) -> tuple[np.ndarray, float]:
    """Compute cheap retrieval representations without persisting them."""

    started = perf_counter()
    values = (
        np.asarray(shared_cache.forecaster.represent(contexts), dtype=np.float32)
        if mode == "model"
        else context_representation(contexts, mode)
    )
    return values, perf_counter() - started


@dataclass(frozen=True)
class _Series:
    item: int
    channel: int
    start_tick: int
    length: int


def _series(prepared: PreparedDataset) -> list[_Series]:
    result: list[_Series] = []
    for item in range(len(prepared.hf_dataset)):
        entry = prepared.hf_dataset[item]
        target = np.asarray(entry["target"])
        channels = int(target.shape[0]) if target.ndim > 1 else 1
        length = int(target.shape[-1])
        start_tick = int(
            pd.Period(pd.Timestamp(entry["start"]), freq=entry["freq"]).ordinal
        )
        result.extend(
            _Series(item, channel, start_tick, length)
            for channel in range(channels)
        )
    return result


def _recent_origins(
    series: _Series,
    *,
    before_tick: int,
    alignment_tick: int,
    period: int,
    stride: int,
    count: int,
    context_length: int,
    horizon: int,
) -> np.ndarray:
    last_tick = min(
        int(before_tick),
        int(series.start_tick + series.length - horizon),
    )
    last_tick -= (last_tick - int(alignment_tick)) % int(period)
    last_origin = int(last_tick - series.start_tick)
    if last_origin < int(context_length):
        return np.empty(0, dtype=np.int64)
    values = np.arange(
        last_origin,
        int(context_length) - 1,
        -int(stride),
        dtype=np.int64,
    )
    return values[: int(count)][::-1]


def _reference(series: _Series, origin: int) -> tuple[int, int, int]:
    return (int(series.item), int(series.channel), int(origin))


class _HorizonStatistics:
    def __init__(self, horizon: int, features: int) -> None:
        self.horizon = int(horizon)
        self.features = int(features)
        self.windows = 0
        self.sum_squares = np.zeros((horizon, features), dtype=np.float64)
        self.xtx = np.zeros((horizon, features, features), dtype=np.float64)
        self.xty = np.zeros((horizon, features), dtype=np.float64)

    def update(
        self,
        design: np.ndarray,
        residual: np.ndarray,
        scale: float,
        sign: int,
    ) -> None:
        x = np.asarray(design, dtype=np.float64) / max(float(scale), 1e-8)
        y = np.asarray(residual, dtype=np.float64) / max(float(scale), 1e-8)
        self.sum_squares += int(sign) * np.square(x)
        self.xtx += int(sign) * np.einsum("hf,hg->hfg", x, x)
        self.xty += int(sign) * x * y[:, None]
        self.windows += int(sign)

    def solve(self, alpha: float) -> np.ndarray:
        rms = np.maximum(
            np.sqrt(np.maximum(self.sum_squares, 0.0) / self.windows), 1e-12
        )
        result = np.empty((self.horizon, self.features), dtype=np.float64)
        for horizon in range(self.horizon):
            matrix = (
                self.xtx[horizon]
                / np.outer(rms[horizon], rms[horizon])
                / self.windows
                + float(alpha) * np.eye(self.features)
            )
            target = self.xty[horizon] / rms[horizon] / self.windows
            try:
                standardized = np.linalg.solve(matrix, target)
            except np.linalg.LinAlgError:
                standardized = np.linalg.lstsq(matrix, target, rcond=None)[0]
            result[horizon] = standardized / rms[horizon]
        return result


def _write_fallback(
    root: Path,
    identity: dict[str, object],
    vanilla: np.ndarray,
    evaluation_grid_cells: np.ndarray,
    reason: str,
    *,
    context_length: int,
    prediction_length: int,
    started: float,
) -> Path:
    prediction_path = root / "predictions.npy"
    eligibility_path = root / "rolling_eligible.npy"
    nonfinite_path = root / "nonfinite_prediction_fallback.npy"
    predictions = _memmap(prediction_path, vanilla.shape, np.float32)
    eligibility = _memmap(eligibility_path, (len(vanilla),), bool)
    nonfinite = _memmap(nonfinite_path, (len(vanilla),), bool)
    predictions[:] = vanilla
    eligibility[:] = False
    nonfinite[:] = False
    predictions.flush()
    eligibility.flush()
    nonfinite.flush()
    manifest_path = root / "prediction_manifest.json"
    _atomic_json(
        manifest_path,
        {
            **identity,
            "format": "adaptime_point_predictions",
            "status": "completed",
            "method": ROLLING_RIDGE_METHOD,
            "context_length": context_length,
            "prediction_length": prediction_length,
            "inference_seconds": perf_counter() - started,
            "fallback_reason": reason,
            "rolling_coverage": {"eligible_windows": 0, "total_windows": len(vanilla)},
            "nonfinite_prediction_fallback": {
                "policy": "not_applicable_task_level_vanilla_fallback",
                "count": 0,
                "eligible_evaluation_windows": int(
                    np.count_nonzero(evaluation_grid_cells)
                ),
            },
            "files": {
                "predictions": prediction_path.name,
                "rolling_eligible": eligibility_path.name,
                "nonfinite_prediction_fallback": nonfinite_path.name,
            },
        },
    )
    return manifest_path


def predict_rolling_ridge(
    prepared_path: str | Path,
    vanilla_path: str | Path,
    shared_cache: SharedWindowCache,
    config: RollingRidgeConfig,
    output_dir: str | Path,
    *,
    evaluation_grid_path: str | Path,
) -> Path:
    """Fit one causal horizon Ridge per variate at every official test query."""

    started = perf_counter()
    prepared = PreparedDataset(prepared_path)
    if prepared.target_mode != "univariate":
        raise ValueError("rolling per-variate Ridge requires univariate prepared rows")
    period = int(prepared.config["retrieval_period"])
    config.validate(period)
    fitting_stride = int(config.fitting_stride or period)
    datastore_stride = int(config.datastore_stride or period)
    vanilla_root, vanilla_manifest = open_vanilla_test_forecasts(vanilla_path)
    vanilla = np.load(
        vanilla_root / vanilla_manifest["arrays"]["predictions"], mmap_mode="r"
    )
    grid_targets, grid_cells = flatten_univariate_grid(
        *load_evaluation_grid(evaluation_grid_path)
    )
    if grid_targets.shape != (len(vanilla), prepared.prediction_length):
        raise ValueError("shared evaluation grid does not match rolling Ridge test rows")
    vanilla_finite = np.all(
        ~grid_targets | np.isfinite(np.asarray(vanilla[:, 0])), axis=1
    )
    if np.any(grid_cells & ~vanilla_finite):
        raise ValueError("vanilla forecast is non-finite on the shared evaluation grid")
    identity = {
        "schema_version": ROLLING_RIDGE_SCHEMA,
        "signature": _canonical_hash(
            {
                "prepared_signature": prepared.signature,
                "vanilla_signature": vanilla_manifest["signature"],
                "forecast_cache_config": shared_cache.config,
                "evaluation_grid": EVALUATION_GRID_DEFINITION,
                "method": ROLLING_RIDGE_METHOD,
                "config": asdict(config),
                "resolved_fitting_stride": fitting_stride,
                "resolved_datastore_stride": datastore_stride,
            }
        ),
        "prepared_signature": prepared.signature,
        "vanilla_signature": vanilla_manifest["signature"],
        "forecast_cache_config": shared_cache.config,
        "evaluation_grid": EVALUATION_GRID_DEFINITION,
        "evaluation_grid_source": str(
            Path(evaluation_grid_path).expanduser().resolve()
        ),
        "config": asdict(config),
        "resolved_fitting_stride": fitting_stride,
        "resolved_datastore_stride": datastore_stride,
    }
    root = Path(output_dir).expanduser().resolve()
    manifest_path = root / "prediction_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("signature") == identity["signature"] and all(
            (root / relative).is_file()
            for relative in dict(existing.get("files", {})).values()
        ):
            return manifest_path
        raise FileExistsError(f"rolling Ridge prediction directory differs: {root}")
    root.mkdir(parents=True, exist_ok=True)

    all_series = _series(prepared)
    if not all_series:
        return _write_fallback(
            root,
            identity,
            vanilla,
            grid_cells,
            "no_source_series",
            context_length=prepared.context_length,
            prediction_length=prepared.prediction_length,
            started=started,
        )
    series_by_key = {(value.item, value.channel): value for value in all_series}
    test_refs = np.asarray(prepared.indices("test"), dtype=np.int64)
    test_ticks = np.asarray(prepared.calendar_ticks("test"), dtype=np.int64)
    fitting_by_test: list[np.ndarray] = []
    for reference, query_tick in zip(test_refs, test_ticks, strict=True):
        current = series_by_key[(int(reference[0]), int(reference[1]))]
        origins = _recent_origins(
            current,
            before_tick=int(query_tick) - prepared.prediction_length,
            alignment_tick=int(query_tick),
            period=period,
            stride=fitting_stride,
            count=config.n_fitting_dates,
            context_length=prepared.reference_context_length,
            horizon=prepared.prediction_length,
        )
        fitting_by_test.append(
            np.asarray(
                [_reference(current, origin) for origin in origins], dtype=np.int64
            ).reshape(-1, 3)
        )
    fitting_support = np.asarray(list(map(len, fitting_by_test)), dtype=np.int64)
    fitting_eligible = fitting_support >= int(config.minimum_fitting_dates)
    if not np.any(fitting_eligible):
        return _write_fallback(
            root,
            identity,
            vanilla,
            grid_cells,
            "fewer_than_minimum_period_aligned_fitting_dates",
            context_length=prepared.context_length,
            prediction_length=prepared.prediction_length,
            started=started,
        )
    fitting_by_test = [
        values[-min(len(values), int(config.n_fitting_dates)) :]
        if eligible
        else np.empty((0, 3), dtype=np.int64)
        for values, eligible in zip(fitting_by_test, fitting_eligible, strict=True)
    ]
    retrieval_refs = np.unique(
        np.concatenate((test_refs[fitting_eligible], *fitting_by_test), axis=0), axis=0
    )
    retrieval_ticks = np.asarray(
        [
            series_by_key[(int(item), int(channel))].start_tick + int(origin)
            for item, channel, origin in retrieval_refs
        ],
        dtype=np.int64,
    )

    max_dates_per_variate = int(config.max_datastore_windows) // len(all_series)
    if max_dates_per_variate <= 0:
        return _write_fallback(
            root,
            identity,
            vanilla,
            grid_cells,
            "datastore_cap_smaller_than_number_of_variates",
            context_length=prepared.context_length,
            prediction_length=prepared.prediction_length,
            started=started,
        )
    datastore_capacity = max_dates_per_variate
    for retrieval_tick in retrieval_ticks:
        for candidate_series in all_series:
            available = _recent_origins(
                candidate_series,
                before_tick=int(retrieval_tick) - prepared.prediction_length,
                alignment_tick=int(retrieval_tick),
                period=period,
                stride=datastore_stride,
                count=max_dates_per_variate,
                context_length=prepared.reference_context_length,
                horizon=prepared.prediction_length,
            )
            datastore_capacity = min(datastore_capacity, len(available))
    if datastore_capacity * len(all_series) < int(config.k):
        return _write_fallback(
            root,
            identity,
            vanilla,
            grid_cells,
            "rolling_datastore_cannot_supply_k_neighbors",
            context_length=prepared.context_length,
            prediction_length=prepared.prediction_length,
            started=started,
        )

    reader = prepared.reader(cache_items=config.arrow_cache_items)
    reference_to_row = {
        tuple(map(int, reference)): row for row, reference in enumerate(retrieval_refs)
    }
    total_retrieval = len(retrieval_refs)
    horizon = prepared.prediction_length
    vanilla_rows = _memmap(root / "retrieval_vanilla.npy", (total_retrieval, 1, horizon), np.float32)
    target_rows = _memmap(root / "retrieval_target.npy", (total_retrieval, 1, horizon), np.float32)
    scale_rows = _memmap(root / "retrieval_scale.npy", (total_retrieval,), np.float32)
    neighbor_refs = _memmap(root / "neighbor_reference.npy", (total_retrieval, config.k, 3), np.int64)
    neighbor_distance = _memmap(root / "neighbor_distance.npy", (total_retrieval, config.k), np.float32)
    retrieval_eligible = _memmap(root / "retrieval_eligible.npy", (total_retrieval,), bool)
    neighbor_refs[:] = -1
    neighbor_distance[:] = np.inf
    retrieval_eligible[:] = False
    cache_forecast_seconds = 0.0
    representation_seconds = 0.0
    query_representations: np.ndarray | None = None
    for start in range(0, total_retrieval, config.model_batch_size):
        stop = min(start + config.model_batch_size, total_retrieval)
        selected_refs = retrieval_refs[start:stop]
        retrieval_batch = reader.read(
            selected_refs,
            context_length=prepared.retrieval_context_length,
        )
        values, seconds = shared_cache.forecasts_for_references(
            selected_refs, reader=reader
        )
        vanilla_rows[start:stop] = values
        cache_forecast_seconds += seconds
        target_rows[start:stop] = retrieval_batch.target
        scale_rows[start:stop] = query_scale(retrieval_batch.context)[:, 0]
        values, seconds = _represent(
            shared_cache, retrieval_batch.context, config.representation
        )
        if query_representations is None:
            query_representations = np.empty(
                (total_retrieval, values.shape[1]), dtype=np.float32
            )
        query_representations[start:stop] = values
        representation_seconds += seconds
    assert query_representations is not None

    retrieval_seconds = 0.0
    for row, (reference, retrieval_tick) in enumerate(
        zip(retrieval_refs, retrieval_ticks, strict=True)
    ):
        candidates: list[tuple[int, int, int]] = []
        candidate_ticks: list[int] = []
        for candidate_series in all_series:
            origins = _recent_origins(
                candidate_series,
                before_tick=int(retrieval_tick) - horizon,
                alignment_tick=int(retrieval_tick),
                period=period,
                stride=datastore_stride,
                count=datastore_capacity,
                context_length=prepared.reference_context_length,
                horizon=horizon,
            )
            candidates.extend(_reference(candidate_series, origin) for origin in origins)
            candidate_ticks.extend(candidate_series.start_tick + int(origin) for origin in origins)
        candidate_refs = np.asarray(candidates, dtype=np.int64)
        representation_parts = []
        for start in range(0, len(candidate_refs), config.datastore_block_size):
            stop = min(start + config.datastore_block_size, len(candidate_refs))
            candidate_batch = reader.read(
                candidate_refs[start:stop],
                context_length=prepared.retrieval_context_length,
            )
            values, seconds = _represent(
                shared_cache, candidate_batch.context, config.representation
            )
            representation_parts.append(values)
            representation_seconds += seconds
        candidate_representations = np.concatenate(representation_parts, axis=0)
        retrieval_started = perf_counter()
        distances, positions = blockwise_topk(
            query_representations[row : row + 1],
            candidate_representations,
            reference[None, :],
            candidate_refs,
            query_calendar_ticks=np.asarray([retrieval_tick], dtype=np.int64),
            datastore_calendar_ticks=np.asarray(candidate_ticks, dtype=np.int64),
            k=config.k,
            stride=period,
            horizon=horizon,
            scope="all",
            metric=config.distance_metric,
            minimum_overlap_fraction=config.minimum_overlap_fraction,
            query_block_size=1,
            datastore_block_size=config.datastore_block_size,
            require_complete_k=False,
        )
        retrieval_seconds += perf_counter() - retrieval_started
        valid = positions[0] >= 0
        if np.count_nonzero(valid) == config.k:
            selected = candidate_refs[positions[0]]
            targets = reader.read(
                selected, context_length=prepared.retrieval_context_length
            ).target
            if np.isfinite(targets).all():
                neighbor_refs[row] = selected
                neighbor_distance[row] = distances[0]
                retrieval_eligible[row] = True

    fitting_rows_by_test = [
        np.asarray(
            [reference_to_row[tuple(map(int, reference))] for reference in values],
            dtype=np.int64,
        )
        for values in fitting_by_test
    ]
    fitting_offsets = np.zeros(len(test_refs) + 1, dtype=np.int64)
    fitting_offsets[1:] = np.cumsum(list(map(len, fitting_rows_by_test)))
    fitting_index = np.concatenate(
        [values for values in fitting_rows_by_test if len(values)], axis=0
    )
    test_index = np.full(len(test_refs), -1, dtype=np.int64)
    test_index[fitting_eligible] = np.asarray(
        [
            reference_to_row[tuple(map(int, reference))]
            for reference in test_refs[fitting_eligible]
        ],
        dtype=np.int64,
    )
    predictions = _memmap(root / "predictions.npy", vanilla.shape, np.float32)
    rolling_eligible = _memmap(root / "rolling_eligible.npy", (len(test_refs),), bool)
    nonfinite_fallback = _memmap(
        root / "nonfinite_prediction_fallback.npy", (len(test_refs),), bool
    )
    predictions[:] = vanilla
    rolling_eligible[:] = False
    nonfinite_fallback[:] = False
    trackers: dict[tuple[int, int, int], tuple[_HorizonStatistics, set[int]]] = {}
    feature_cache: dict[int, np.ndarray] = {}

    def features(row: int) -> np.ndarray:
        if row not in feature_cache:
            selected_targets = reader.read(
                np.asarray(neighbor_refs[row]),
                context_length=prepared.retrieval_context_length,
            ).target[:, 0]
            vanilla_value = np.asarray(vanilla_rows[row, 0], dtype=np.float64)
            feature_cache[row] = np.column_stack(
                (vanilla_value, selected_targets.T)
            )
        return feature_cache[row]

    def fitting_example(row: int) -> tuple[np.ndarray, np.ndarray]:
        x = features(row)
        y = np.asarray(target_rows[row, 0], dtype=np.float64) - np.asarray(
            vanilla_rows[row, 0], dtype=np.float64
        )
        return x, y

    fit_seconds = 0.0
    for output_row, reference in enumerate(test_refs):
        if not fitting_eligible[output_row]:
            continue
        current = int(test_index[output_row])
        fitting_rows = fitting_rows_by_test[output_row]
        desired = set(map(int, fitting_rows))
        if not bool(retrieval_eligible[current]) or not np.all(
            np.asarray(retrieval_eligible)[list(desired)]
        ):
            continue
        last_fit_tick = int(retrieval_ticks[fitting_rows[-1]])
        key = (
            int(reference[0]),
            int(reference[1]),
            last_fit_tick % fitting_stride,
        )
        statistics, active = trackers.setdefault(
            key, (_HorizonStatistics(horizon, config.k + 1), set())
        )
        fit_started = perf_counter()
        for expired in active - desired:
            x, y = fitting_example(expired)
            statistics.update(x, y, float(scale_rows[expired]), -1)
        for added in desired - active:
            x, y = fitting_example(added)
            statistics.update(x, y, float(scale_rows[added]), +1)
        active.clear()
        active.update(desired)
        coefficients = statistics.solve(config.alpha)
        current_x = features(current)
        value = np.asarray(vanilla_rows[current, 0], dtype=np.float64) + np.einsum(
            "hf,hf->h", current_x, coefficients
        )
        fit_seconds += perf_counter() - fit_started
        finite_when_expected = np.all(
            ~grid_targets[output_row] | np.isfinite(value)
        )
        if finite_when_expected:
            predictions[output_row, 0] = value
            rolling_eligible[output_row] = True
        elif grid_cells[output_row]:
            nonfinite_fallback[output_row] = True

    for store in (
        vanilla_rows,
        target_rows,
        scale_rows,
        neighbor_refs,
        neighbor_distance,
        retrieval_eligible,
        predictions,
        rolling_eligible,
        nonfinite_fallback,
    ):
        store.flush()
    np.save(root / "retrieval_reference.npy", retrieval_refs, allow_pickle=False)
    np.save(root / "fitting_index.npy", fitting_index, allow_pickle=False)
    np.save(root / "fitting_offsets.npy", fitting_offsets, allow_pickle=False)
    np.save(root / "fitting_support.npy", fitting_support, allow_pickle=False)
    np.save(root / "test_index.npy", test_index, allow_pickle=False)
    _atomic_json(
        manifest_path,
        {
            **identity,
            "format": "adaptime_point_predictions",
            "status": "completed",
            "method": ROLLING_RIDGE_METHOD,
            "context_length": prepared.context_length,
            "prediction_length": horizon,
            "inference_seconds": perf_counter() - started,
            "fallback_reason": None,
            "rolling_protocol": {
                "fitting_scope": "same_series",
                "retrieval_scope": "all_series",
                "features": ["V", "Y_1..Y_K"],
                "coefficient_scope": "per_series_and_horizon",
                "selection": "fixed_k_and_alpha_without_validation",
                "effective_n_fitting_dates": {
                    "minimum": int(fitting_support[fitting_eligible].min()),
                    "maximum": int(fitting_support[fitting_eligible].max()),
                },
                "datastore_dates_per_variate": datastore_capacity,
                "total_datastore_windows": datastore_capacity * len(all_series),
            },
            "rolling_coverage": {
                "eligible_windows": int(np.count_nonzero(rolling_eligible)),
                "total_windows": len(test_refs),
            },
            "nonfinite_prediction_fallback": {
                "policy": "replace_complete_forecast_with_vanilla_on_shared_evaluation_grid",
                "count": int(np.count_nonzero(nonfinite_fallback)),
                "eligible_evaluation_windows": int(np.count_nonzero(grid_cells)),
            },
            "timing_seconds": {
                "new_vanilla_forecast_seconds": cache_forecast_seconds,
                "representation_seconds": representation_seconds,
                "retrieval_seconds": retrieval_seconds,
                "rolling_fit_and_prediction_seconds": fit_seconds,
            },
            "files": {
                "predictions": "predictions.npy",
                "rolling_eligible": "rolling_eligible.npy",
                "nonfinite_prediction_fallback": "nonfinite_prediction_fallback.npy",
                "retrieval_reference": "retrieval_reference.npy",
                "retrieval_vanilla": "retrieval_vanilla.npy",
                "retrieval_target": "retrieval_target.npy",
                "retrieval_scale": "retrieval_scale.npy",
                "neighbor_reference": "neighbor_reference.npy",
                "neighbor_distance": "neighbor_distance.npy",
                "retrieval_eligible": "retrieval_eligible.npy",
                "fitting_index": "fitting_index.npy",
                "fitting_offsets": "fitting_offsets.npy",
                "fitting_support": "fitting_support.npy",
                "test_index": "test_index.npy",
            },
        },
    )
    return manifest_path
