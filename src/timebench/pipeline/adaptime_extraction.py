"""Disk-backed fit-grid and selected-K evaluation extraction for Adaptime."""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol

import numpy as np

from timebench.adaptime.retrieval import blockwise_topk, context_representation
from timebench.adaptime.ridge import query_scale
from timebench.evaluation.adaptation_data import FIT_QUERY_SPLITS, PreparedDataset
from timebench.evaluation.timing import EvaluationTimer


EXTRACTION_SCHEMA = 1
EVAL_EXTRACTION_SCHEMA = 1
FALLBACK_REASONS = {
    0: "rag_eligible",
    1: "insufficient_finite_query_context",
    2: "insufficient_valid_neighbors",
    3: "nonfinite_datastore_target",
    4: "insufficient_past_context",
    5: "nonfinite_adaptation_design",
    6: "insufficient_training_support",
}


class AdaptimeForecaster(Protocol):
    """Minimal model boundary used by the extraction pipeline.

    Implementations return point forecasts, not quantile axes. Retrieval context
    has shape ``(batch, K, channels, lookback + horizon)`` and is already
    expressed in the query window's level and scale.
    """

    model_name: str
    weights_id: str
    supports_multivariate: bool
    supports_retrieval_context: bool

    def forecast(
        self,
        context: np.ndarray,
        *,
        retrieval_context: np.ndarray | None = None,
    ) -> np.ndarray: ...

    def represent(self, context: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class ExtractionConfig:
    representation: str = "instance"
    distance_metric: str = "euclidean"
    retrieval_scope: str = "all"
    minimum_overlap_fraction: float = 0.8
    minimum_query_finite_fraction: float = 0.8
    max_k: int = 15
    context_k: tuple[int, ...] = (1, 5, 10, 15)
    model_batch_size: int = 64
    query_block_size: int = 256
    datastore_block_size: int = 4096
    arrow_cache_items: int = 2

    def validate(self) -> None:
        if self.representation not in {"raw", "instance", "model"}:
            raise ValueError("representation must be raw, instance, or model")
        if self.distance_metric not in {"euclidean", "cosine"}:
            raise ValueError("distance_metric must be euclidean or cosine")
        if self.retrieval_scope not in {"all", "same_series", "other_series"}:
            raise ValueError("unsupported retrieval_scope")
        if not 0.0 < float(self.minimum_overlap_fraction) <= 1.0:
            raise ValueError("minimum_overlap_fraction must be in (0, 1]")
        if not 0.0 < float(self.minimum_query_finite_fraction) <= 1.0:
            raise ValueError("minimum_query_finite_fraction must be in (0, 1]")
        positive = {
            "max_k": self.max_k,
            "model_batch_size": self.model_batch_size,
            "query_block_size": self.query_block_size,
            "datastore_block_size": self.datastore_block_size,
            "arrow_cache_items": self.arrow_cache_items,
        }
        invalid = [name for name, value in positive.items() if int(value) <= 0]
        if invalid:
            raise ValueError(f"positive extraction settings required: {', '.join(invalid)}")
        if not self.context_k or any(int(k) <= 0 or int(k) > self.max_k for k in self.context_k):
            raise ValueError("context_k must contain values between 1 and max_k")
        if tuple(sorted(set(self.context_k))) != self.context_k:
            raise ValueError("context_k must be sorted and unique")


def _canonical_hash(value: dict[str, object]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _normalize_extraction_config(value: dict[str, object]) -> dict[str, object]:
    normalized = dict(value)
    normalized["context_k"] = tuple(normalized["context_k"])
    return normalized


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _memmap(path: Path, shape: tuple[int, ...], dtype: object) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", shape=shape, dtype=dtype)


def _forecast(
    forecaster: AdaptimeForecaster,
    context: np.ndarray,
    *,
    horizon: int,
    retrieval_context: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(
        forecaster.forecast(context, retrieval_context=retrieval_context),
        dtype=np.float32,
    )
    expected = (context.shape[0], context.shape[1], int(horizon))
    if values.shape != expected:
        raise ValueError(f"forecaster returned {values.shape}, expected {expected}")
    return values


def _record_seconds(timings: dict[str, float], name: str, seconds: float) -> None:
    timings[name] = timings.get(name, 0.0) + float(seconds)


def _timed_forecast(
    forecaster: AdaptimeForecaster,
    context: np.ndarray,
    timings: dict[str, float],
    name: str,
    *,
    horizon: int,
    retrieval_context: np.ndarray | None = None,
) -> np.ndarray:
    timer = EvaluationTimer()
    timer.start()
    values = _forecast(
        forecaster,
        context,
        horizon=horizon,
        retrieval_context=retrieval_context,
    )
    _record_seconds(timings, name, timer.stop())
    return values


def _represent(
    forecaster: AdaptimeForecaster,
    context: np.ndarray,
    mode: str,
) -> np.ndarray:
    values = (
        np.asarray(forecaster.represent(context), dtype=np.float32)
        if mode == "model"
        else context_representation(context, mode)
    )
    if values.ndim != 2 or values.shape[0] != context.shape[0]:
        raise ValueError("representations must have shape (batch, features)")
    return np.ascontiguousarray(values)


def _timed_represent(
    forecaster: AdaptimeForecaster,
    context: np.ndarray,
    mode: str,
    timings: dict[str, float],
    name: str,
) -> np.ndarray:
    timer = EvaluationTimer()
    timer.start()
    values = _represent(forecaster, context, mode)
    _record_seconds(timings, name, timer.stop())
    return values


def _query_scaled_retrieval_context(
    query: np.ndarray,
    neighbor_context: np.ndarray,
    neighbor_target: np.ndarray,
) -> np.ndarray:
    query = np.asarray(query, dtype=np.float32)
    neighbor_context = np.asarray(neighbor_context, dtype=np.float32)
    neighbor_target = np.asarray(neighbor_target, dtype=np.float32)
    if neighbor_context.ndim != 4 or neighbor_target.ndim != 4:
        raise ValueError("retrieved arrays must have shape (batch, K, channels, time)")
    if neighbor_context.shape[:3] != neighbor_target.shape[:3]:
        raise ValueError("retrieved context and target axes must align")
    if query.shape[0] != neighbor_context.shape[0] or query.shape[1] != neighbor_context.shape[2]:
        raise ValueError("query and retrieved channel axes must align")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        query_mean = np.nanmean(query, axis=-1, keepdims=True)[:, None]
        query_std = np.maximum(np.nanstd(query, axis=-1, keepdims=True), 1e-8)[:, None]
        neighbor_mean = np.nanmean(neighbor_context, axis=-1, keepdims=True)
        neighbor_std = np.maximum(
            np.nanstd(neighbor_context, axis=-1, keepdims=True), 1e-8
        )
    scaled_context = (neighbor_context - neighbor_mean) / neighbor_std * query_std + query_mean
    scaled_target = (neighbor_target - neighbor_mean) / neighbor_std * query_std + query_mean
    return np.concatenate((scaled_context, scaled_target), axis=-1)


def _source_eligibility(
    context: np.ndarray,
    target: np.ndarray,
    split: str,
    minimum_finite_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify rows without imputing histories or retrieved futures."""

    context = np.asarray(context)
    target = np.asarray(target)
    if np.isinf(context).any() or np.isinf(target).any():
        raise ValueError("Adaptime source windows must not contain infinite values")
    fraction = np.isfinite(context).reshape(len(context), -1).mean(axis=1)
    eligible = fraction >= float(minimum_finite_fraction)
    reason = np.where(eligible, 0, 1).astype(np.uint8)
    if split == "datastore":
        complete_target = np.isfinite(target).reshape(len(target), -1).all(axis=1)
        reason[eligible & ~complete_target] = 3
        eligible &= complete_target
    return fraction.astype(np.float32), eligible, reason


def _materialize_source_rows(
    prepared: PreparedDataset,
    split: str,
    forecaster: AdaptimeForecaster,
    config: ExtractionConfig,
    root: Path,
    arrays: dict[str, str],
    timings: dict[str, float],
    *,
    include_vanilla: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    references = prepared.indices(split)
    first_stop = min(len(references), config.model_batch_size)
    reader = prepared.reader(cache_items=config.arrow_cache_items)
    representation_key = f"{split}.representation_seconds"
    if first_stop:
        first = reader.read(references[:first_stop])
        first_representation = _timed_represent(
            forecaster,
            first.context,
            config.representation,
            timings,
            representation_key,
        )
        channels = int(first.context.shape[1])
        representation_width = int(first_representation.shape[1])
    else:
        channels = 1 if prepared.target_mode == "univariate" else int(
            np.asarray(prepared.hf_dataset[0]["target"]).shape[0]
        )
        if config.representation in {"raw", "instance"}:
            representation_width = channels * prepared.context_length
        else:
            probe = np.zeros((1, channels, prepared.context_length), dtype=np.float32)
            representation_width = int(
                _timed_represent(
                    forecaster,
                    probe,
                    config.representation,
                    timings,
                    representation_key,
                ).shape[1]
            )
    horizon = prepared.prediction_length
    target_path = root / split / "target.npy"
    representation_path = root / split / "representation.npy"
    scale_path = root / split / "query_scale.npy"
    finite_fraction_path = root / split / "context_finite_fraction.npy"
    eligible_path = root / split / "rag_eligible.npy"
    fallback_reason_path = root / split / "fallback_reason.npy"
    target = _memmap(target_path, (len(references), channels, horizon), np.float32)
    representation = _memmap(
        representation_path,
        (len(references), representation_width),
        np.float32,
    )
    scale = _memmap(scale_path, (len(references), channels), np.float32)
    finite_fraction = _memmap(finite_fraction_path, (len(references),), np.float32)
    rag_eligible = _memmap(eligible_path, (len(references),), bool)
    fallback_reason = _memmap(fallback_reason_path, (len(references),), np.uint8)
    vanilla: np.memmap | None = None
    mase_scale: np.memmap | None = None
    msse_scale: np.memmap | None = None
    seasonal_naive: np.memmap | None = None
    if include_vanilla:
        vanilla_path = root / split / "vanilla.npy"
        vanilla = _memmap(
            vanilla_path,
            (len(references), channels, horizon),
            np.float32,
        )
        arrays[f"{split}.vanilla"] = str(vanilla_path.relative_to(root))
        mase_scale_path = root / split / "mase_scale.npy"
        msse_scale_path = root / split / "msse_scale.npy"
        seasonal_naive_path = root / split / "seasonal_naive.npy"
        mase_scale = _memmap(mase_scale_path, (len(references), channels), np.float32)
        msse_scale = _memmap(msse_scale_path, (len(references), channels), np.float32)
        seasonal_naive = _memmap(
            seasonal_naive_path,
            (len(references), channels, horizon),
            np.float32,
        )
        arrays[f"{split}.mase_scale"] = str(mase_scale_path.relative_to(root))
        arrays[f"{split}.msse_scale"] = str(msse_scale_path.relative_to(root))
        arrays[f"{split}.seasonal_naive"] = str(
            seasonal_naive_path.relative_to(root)
        )
    arrays[f"{split}.target"] = str(target_path.relative_to(root))
    arrays[f"{split}.representation"] = str(representation_path.relative_to(root))
    arrays[f"{split}.query_scale"] = str(scale_path.relative_to(root))
    arrays[f"{split}.context_finite_fraction"] = str(
        finite_fraction_path.relative_to(root)
    )
    arrays[f"{split}.rag_eligible"] = str(eligible_path.relative_to(root))
    arrays[f"{split}.fallback_reason"] = str(fallback_reason_path.relative_to(root))

    if first_stop:
        target[:first_stop] = first.target
        representation[:first_stop] = first_representation
        scale[:first_stop] = query_scale(first.context)
        first_fraction, first_eligible, first_reason = _source_eligibility(
            first.context,
            first.target,
            split,
            config.minimum_query_finite_fraction,
        )
        finite_fraction[:first_stop] = first_fraction
        rag_eligible[:first_stop] = first_eligible
        fallback_reason[:first_stop] = first_reason
    if vanilla is not None and first_stop:
        assert mase_scale is not None and msse_scale is not None and seasonal_naive is not None
        first_mase_scale, first_msse_scale = reader.seasonal_scales(
            references[:first_stop]
        )
        mase_scale[:first_stop] = first_mase_scale
        msse_scale[:first_stop] = first_msse_scale
        seasonal_naive[:first_stop] = reader.seasonal_naive_forecast(
            references[:first_stop]
        )
        vanilla[:first_stop] = _timed_forecast(
            forecaster,
            first.context,
            timings,
            f"{split}.vanilla_forecast_seconds",
            horizon=horizon,
        )
    for start in range(first_stop, len(references), config.model_batch_size):
        stop = min(start + config.model_batch_size, len(references))
        batch = reader.read(references[start:stop])
        target[start:stop] = batch.target
        representation[start:stop] = _timed_represent(
            forecaster,
            batch.context,
            config.representation,
            timings,
            representation_key,
        )
        scale[start:stop] = query_scale(batch.context)
        batch_fraction, batch_eligible, batch_reason = _source_eligibility(
            batch.context,
            batch.target,
            split,
            config.minimum_query_finite_fraction,
        )
        finite_fraction[start:stop] = batch_fraction
        rag_eligible[start:stop] = batch_eligible
        fallback_reason[start:stop] = batch_reason
        if vanilla is not None:
            assert mase_scale is not None and msse_scale is not None and seasonal_naive is not None
            batch_mase_scale, batch_msse_scale = reader.seasonal_scales(
                references[start:stop]
            )
            mase_scale[start:stop] = batch_mase_scale
            msse_scale[start:stop] = batch_msse_scale
            seasonal_naive[start:stop] = reader.seasonal_naive_forecast(
                references[start:stop]
            )
            vanilla[start:stop] = _timed_forecast(
                forecaster,
                batch.context,
                timings,
                f"{split}.vanilla_forecast_seconds",
                horizon=horizon,
            )
    target.flush()
    representation.flush()
    scale.flush()
    finite_fraction.flush()
    rag_eligible.flush()
    fallback_reason.flush()
    if vanilla is not None:
        vanilla.flush()
        assert mase_scale is not None and msse_scale is not None and seasonal_naive is not None
        mase_scale.flush()
        msse_scale.flush()
        seasonal_naive.flush()
    return target, representation


def _materialize_neighbors(
    prepared: PreparedDataset,
    split: str,
    datastore_representation: np.ndarray,
    config: ExtractionConfig,
    root: Path,
    arrays: dict[str, str],
    timings: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    query_representation = np.load(
        root / arrays[f"{split}.representation"], mmap_mode="r"
    )
    query_eligible = np.load(
        root / arrays[f"{split}.rag_eligible"], mmap_mode="r+"
    )
    fallback_reason = np.load(
        root / arrays[f"{split}.fallback_reason"], mmap_mode="r+"
    )
    datastore_eligible = np.load(
        root / arrays["datastore.rag_eligible"], mmap_mode="r"
    )
    query_positions = np.flatnonzero(query_eligible)
    datastore_positions = np.flatnonzero(datastore_eligible)
    started = perf_counter()
    distances = np.full((len(query_representation), config.max_k), np.inf, np.float32)
    ids = np.full((len(query_representation), config.max_k), -1, np.int64)
    if len(query_positions) and len(datastore_positions):
        retrieval_k = min(config.max_k, len(datastore_positions))
        selected_distances, selected_ids = blockwise_topk(
            query_representation[query_positions],
            datastore_representation[datastore_positions],
            prepared.indices(split)[query_positions],
            prepared.indices("datastore")[datastore_positions],
            query_calendar_ticks=prepared.calendar_ticks(split)[query_positions],
            datastore_calendar_ticks=prepared.calendar_ticks("datastore")[
                datastore_positions
            ],
            retrieval_period=int(prepared.config["retrieval_period"]),
            datastore_end_ticks_by_item=prepared.datastore_end_ticks_by_item,
            k=retrieval_k,
            stride=int(prepared.config["datastore_stride"]),
            horizon=prepared.prediction_length,
            scope=config.retrieval_scope,
            metric=config.distance_metric,
            minimum_overlap_fraction=config.minimum_overlap_fraction,
            query_block_size=config.query_block_size,
            datastore_block_size=config.datastore_block_size,
            require_complete_k=False,
        )
        distances[query_positions, :retrieval_k] = selected_distances
        valid = selected_ids >= 0
        mapped_ids = np.full_like(selected_ids, -1)
        mapped_ids[valid] = datastore_positions[selected_ids[valid]]
        ids[query_positions, :retrieval_k] = mapped_ids
        enough = np.count_nonzero(valid, axis=1) >= config.max_k
        query_eligible[query_positions[~enough]] = False
        fallback_reason[query_positions[~enough]] = 2
    else:
        query_eligible[query_positions] = False
        fallback_reason[query_positions] = 2
    query_eligible.flush()
    fallback_reason.flush()
    _record_seconds(timings, f"{split}.retrieval_seconds", perf_counter() - started)
    distance_path = root / split / "neighbor_distance.npy"
    id_path = root / split / "neighbor_id.npy"
    distance_store = _memmap(distance_path, distances.shape, np.float32)
    id_store = _memmap(id_path, ids.shape, np.int64)
    distance_store[:] = distances
    id_store[:] = ids
    distance_store.flush()
    id_store.flush()
    arrays[f"{split}.neighbor_distance"] = str(distance_path.relative_to(root))
    arrays[f"{split}.neighbor_id"] = str(id_path.relative_to(root))
    return distance_store, id_store


def _materialize_unique_neighbor_forecasts(
    prepared: PreparedDataset,
    forecaster: AdaptimeForecaster,
    config: ExtractionConfig,
    root: Path,
    arrays: dict[str, str],
    timings: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    timings.setdefault("offline.neighbor_forecast_seconds", 0.0)
    selected = np.empty(0, dtype=np.int64)
    for split in FIT_QUERY_SPLITS:
        split_ids = np.load(root / arrays[f"{split}.neighbor_id"], mmap_mode="r")
        selected = np.union1d(
            selected,
            np.unique(np.asarray(split_ids)[np.asarray(split_ids) >= 0]),
        )
    datastore_refs = prepared.indices("datastore")
    reader = prepared.reader(cache_items=config.arrow_cache_items)
    datastore_target = np.load(root / arrays["datastore.target"], mmap_mode="r")
    values_path = root / "datastore" / "selected_forecast.npy"
    ids_path = root / "datastore" / "selected_forecast_id.npy"
    forecasts = _memmap(
        values_path,
        (len(selected), datastore_target.shape[1], prepared.prediction_length),
        np.float32,
    )
    ids = _memmap(ids_path, selected.shape, np.int64)
    ids[:] = selected
    for start in range(0, len(selected), config.model_batch_size):
        stop = min(start + config.model_batch_size, len(selected))
        batch = reader.read(datastore_refs[selected[start:stop]])
        forecasts[start:stop] = _timed_forecast(
            forecaster,
            batch.context,
            timings,
            "offline.neighbor_forecast_seconds",
            horizon=prepared.prediction_length,
        )
    forecasts.flush()
    ids.flush()
    arrays["datastore.selected_forecast"] = str(values_path.relative_to(root))
    arrays["datastore.selected_forecast_id"] = str(ids_path.relative_to(root))
    return forecasts, selected


def _materialize_context_forecasts(
    prepared: PreparedDataset,
    split: str,
    forecaster: AdaptimeForecaster,
    config: ExtractionConfig,
    root: Path,
    arrays: dict[str, str],
    timings: dict[str, float],
) -> None:
    query_refs = prepared.indices(split)
    datastore_refs = prepared.indices("datastore")
    datastore_target = np.load(root / arrays["datastore.target"], mmap_mode="r")
    neighbor_ids = np.load(root / arrays[f"{split}.neighbor_id"], mmap_mode="r")
    rag_eligible = np.load(root / arrays[f"{split}.rag_eligible"], mmap_mode="r")
    vanilla = np.load(root / arrays[f"{split}.vanilla"], mmap_mode="r")
    reader = prepared.reader(cache_items=config.arrow_cache_items)
    channels = int(np.load(root / arrays[f"{split}.target"], mmap_mode="r").shape[1])
    stores: dict[int, np.memmap] = {}
    for k in config.context_k:
        timings.setdefault(f"{split}.context_construction_k{k}_seconds", 0.0)
        timings.setdefault(f"{split}.context_forecast_k{k}_seconds", 0.0)
        path = root / split / f"context_forecast_k{k}.npy"
        stores[k] = _memmap(
            path,
            (len(query_refs), channels, prepared.prediction_length),
            np.float32,
        )
        arrays[f"{split}.context_forecast_k{k}"] = str(path.relative_to(root))

    for start in range(0, len(query_refs), config.model_batch_size):
        stop = min(start + config.model_batch_size, len(query_refs))
        for store in stores.values():
            store[start:stop] = vanilla[start:stop]
        for k in config.context_k:
            selected_ids = np.asarray(neighbor_ids[start:stop, :k])
            eligible = np.asarray(rag_eligible[start:stop], dtype=bool) & np.all(
                selected_ids >= 0, axis=1
            )
            positions = np.flatnonzero(eligible) + start
            if not len(positions):
                continue
            selected_ids = np.asarray(neighbor_ids[positions, :k])
            query_batch = reader.read(query_refs[positions])
            flat_neighbor_batch = reader.read(datastore_refs[selected_ids.reshape(-1)])
            neighbor_context = flat_neighbor_batch.context.reshape(
                len(positions),
                k,
                channels,
                prepared.context_length,
            )
            neighbor_target = np.asarray(datastore_target[selected_ids])
            started = perf_counter()
            retrieval_context = _query_scaled_retrieval_context(
                query_batch.context,
                neighbor_context,
                neighbor_target,
            )
            _record_seconds(
                timings,
                f"{split}.context_construction_k{k}_seconds",
                perf_counter() - started,
            )
            stores[k][positions] = _timed_forecast(
                forecaster,
                query_batch.context,
                timings,
                f"{split}.context_forecast_k{k}_seconds",
                horizon=prepared.prediction_length,
                retrieval_context=retrieval_context,
            )
    for store in stores.values():
        store.flush()


def extract_adaptation_features(
    prepared_path: str | Path,
    forecaster: AdaptimeForecaster,
    config: ExtractionConfig,
    output_dir: str | Path,
) -> Path:
    """Extract datastore and train/validation quantities for a complete sweep."""

    config.validate()
    prepared = PreparedDataset(prepared_path)
    if prepared.target_mode == "multivariate" and not forecaster.supports_multivariate:
        raise ValueError(f"{forecaster.model_name} does not support multivariate extraction")
    if config.context_k and not forecaster.supports_retrieval_context:
        raise ValueError(
            f"{forecaster.model_name} cannot produce the required context forecast C"
        )
    identity = {
        "schema_version": EXTRACTION_SCHEMA,
        "timing_contract": "component_seconds",
        "prepared_signature": prepared.signature,
        "model": forecaster.model_name,
        "weights_id": forecaster.weights_id,
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
        raise FileExistsError(f"extraction directory already contains a different run: {root}")
    root.mkdir(parents=True, exist_ok=True)

    extraction_started = perf_counter()
    arrays: dict[str, str] = {}
    timings: dict[str, float] = {}
    _, datastore_representation = _materialize_source_rows(
        prepared,
        "datastore",
        forecaster,
        config,
        root,
        arrays,
        timings,
    )
    for split in FIT_QUERY_SPLITS:
        _materialize_source_rows(
            prepared,
            split,
            forecaster,
            config,
            root,
            arrays,
            timings,
            include_vanilla=True,
        )
        _materialize_neighbors(
            prepared,
            split,
            datastore_representation,
            config,
            root,
            arrays,
            timings,
        )
    unique_forecast, _ = _materialize_unique_neighbor_forecasts(
        prepared, forecaster, config, root, arrays, timings
    )
    for split in FIT_QUERY_SPLITS:
        _materialize_context_forecasts(
            prepared, split, forecaster, config, root, arrays, timings
        )
    timings["extraction_total_seconds"] = perf_counter() - extraction_started

    eligibility = {}
    for split in ("datastore", *FIT_QUERY_SPLITS):
        eligible = np.load(root / arrays[f"{split}.rag_eligible"], mmap_mode="r")
        reasons = np.load(root / arrays[f"{split}.fallback_reason"], mmap_mode="r")
        eligibility[split] = {
            "total": int(len(eligible)),
            "rag_eligible": int(np.count_nonzero(eligible)),
            "fallback": int(len(eligible) - np.count_nonzero(eligible)),
            "fallback_reasons": {
                label: int(np.count_nonzero(reasons == code))
                for code, label in FALLBACK_REASONS.items()
                if code != 0
            },
        }

    manifest: dict[str, object] = {
        **identity,
        "format": "adaptime_fit_extraction",
        "signature": signature,
        "status": "completed",
        "arrays": arrays,
        "counts": {
            split: int(len(prepared.indices(split)))
            for split in ("datastore", *FIT_QUERY_SPLITS)
        },
        "computed_neighbor_forecasts": int(len(unique_forecast)),
        "eligibility": eligibility,
        "fallback_reason_codes": {str(code): label for code, label in FALLBACK_REASONS.items()},
        "full_ridge_design": ["V", "C", "Y_1..Y_K", "N_1..N_K"],
        "timing_seconds": timings,
    }
    _atomic_json(manifest_path, manifest)
    return manifest_path


def open_extraction(path: str | Path) -> tuple[Path, dict[str, object]]:
    manifest_path = Path(path).expanduser().resolve()
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != EXTRACTION_SCHEMA
        or manifest.get("format") != "adaptime_fit_extraction"
        or manifest.get("status") != "completed"
    ):
        raise ValueError("Adaptime fit extraction is not a completed schema-1 artifact")
    return manifest_path.parent, manifest


def extract_adaptation_eval_features(
    prepared_path: str | Path,
    fit_extraction_path: str | Path,
    model_path: str | Path,
    vanilla_path: str | Path,
    forecaster: AdaptimeForecaster,
    config: ExtractionConfig,
    output_dir: str | Path,
) -> Path:
    """Extract test quantities for the frozen family-selected K values."""

    from timebench.pipeline.adaptime_training import open_adaptation_model
    from timebench.pipeline.adaptime_vanilla import open_vanilla_test_forecasts

    config.validate()
    prepared = PreparedDataset(prepared_path)
    fit_root, fit_extraction = open_extraction(fit_extraction_path)
    _, model = open_adaptation_model(model_path)
    vanilla_root, vanilla = open_vanilla_test_forecasts(vanilla_path)
    if fit_extraction["prepared_signature"] != prepared.signature:
        raise ValueError("fit extraction and prepared TIME windows do not match")
    if model["extraction_signature"] != fit_extraction["signature"]:
        raise ValueError("frozen adaptor and fit extraction do not match")
    if vanilla["prepared_signature"] != prepared.signature:
        raise ValueError("vanilla forecasts and prepared TIME windows do not match")
    if (
        fit_extraction["model"] != forecaster.model_name
        or fit_extraction["weights_id"] != forecaster.weights_id
        or vanilla["model"] != forecaster.model_name
        or vanilla["weights_id"] != forecaster.weights_id
    ):
        raise ValueError("fit, vanilla, and evaluation forecasters do not match")
    if _normalize_extraction_config(fit_extraction["config"]) != asdict(config):
        raise ValueError("fit and evaluation extraction configurations do not match")
    selected_ks = tuple(sorted(map(int, model.get("evaluation_k_values", ()))))
    max_selected_k = max(selected_ks, default=0)
    identity = {
        "schema_version": EVAL_EXTRACTION_SCHEMA,
        "timing_contract": "component_seconds",
        "prepared_signature": prepared.signature,
        "fit_extraction_signature": fit_extraction["signature"],
        "adaptation_signature": model["signature"],
        "vanilla_signature": vanilla["signature"],
        "model": forecaster.model_name,
        "weights_id": forecaster.weights_id,
        "selected_k_values": list(selected_ks),
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
        raise FileExistsError(f"evaluation extraction already differs: {root}")
    root.mkdir(parents=True, exist_ok=True)

    references = prepared.indices("test")
    total = int(len(references))
    arrays: dict[str, str] = {}
    timings: dict[str, float] = {}
    eligible_path = root / "test" / "rag_eligible.npy"
    reason_path = root / "test" / "fallback_reason.npy"
    rag_eligible = _memmap(eligible_path, (total,), bool)
    fallback_reason = _memmap(reason_path, (total,), np.uint8)
    rag_eligible[:] = False
    fallback_reason[:] = 6 if not selected_ks else 4
    arrays["test.rag_eligible"] = str(eligible_path.relative_to(root))
    arrays["test.fallback_reason"] = str(reason_path.relative_to(root))

    reused_neighbor_forecasts = 0
    computed_neighbor_forecasts = 0
    if selected_ks:
        fit_arrays = dict(fit_extraction["arrays"])
        datastore_representation = np.load(
            fit_root / fit_arrays["datastore.representation"], mmap_mode="r"
        )
        datastore_eligible = np.load(
            fit_root / fit_arrays["datastore.rag_eligible"], mmap_mode="r"
        )
        datastore_target = np.load(
            fit_root / fit_arrays["datastore.target"], mmap_mode="r"
        )
        representation_path = root / "test" / "representation.npy"
        scale_path = root / "test" / "query_scale.npy"
        finite_fraction_path = root / "test" / "context_finite_fraction.npy"
        distance_path = root / "test" / "neighbor_distance.npy"
        neighbor_path = root / "test" / "neighbor_id.npy"
        representation = _memmap(
            representation_path,
            (total, datastore_representation.shape[1]),
            np.float32,
        )
        scale = _memmap(scale_path, (total, 1), np.float32)
        finite_fraction = _memmap(finite_fraction_path, (total,), np.float32)
        distances = _memmap(distance_path, (total, max_selected_k), np.float32)
        neighbor_ids = _memmap(neighbor_path, (total, max_selected_k), np.int64)
        vanilla_values = np.load(
            vanilla_root / vanilla["arrays"]["predictions"], mmap_mode="r"
        )
        context_forecasts = {
            k: _memmap(
                root / "test" / f"context_forecast_k{k}.npy",
                vanilla_values.shape,
                np.float32,
            )
            for k in selected_ks
        }
        representation[:] = np.nan
        scale[:] = np.nan
        finite_fraction[:] = 0.0
        distances[:] = np.inf
        neighbor_ids[:] = -1
        for context_forecast in context_forecasts.values():
            context_forecast[:] = vanilla_values
        arrays.update(
            {
                "test.representation": str(representation_path.relative_to(root)),
                "test.query_scale": str(scale_path.relative_to(root)),
                "test.context_finite_fraction": str(
                    finite_fraction_path.relative_to(root)
                ),
                "test.neighbor_distance": str(distance_path.relative_to(root)),
                "test.neighbor_id": str(neighbor_path.relative_to(root)),
                **{
                    f"test.context_forecast_k{k}": str(
                        (root / "test" / f"context_forecast_k{k}.npy").relative_to(root)
                    )
                    for k in selected_ks
                },
            }
        )

        reader = prepared.reader(cache_items=config.arrow_cache_items)
        fixed_positions = np.flatnonzero(
            np.asarray(references[:, 2], dtype=np.int64) >= prepared.context_length
        )
        timings["test.representation_seconds"] = 0.0
        for start in range(0, len(fixed_positions), config.model_batch_size):
            positions = fixed_positions[start : start + config.model_batch_size]
            batch = reader.read(references[positions])
            representation[positions] = _timed_represent(
                forecaster,
                batch.context,
                config.representation,
                timings,
                "test.representation_seconds",
            )
            scale[positions] = query_scale(batch.context)
            fraction, eligible, reason = _source_eligibility(
                batch.context,
                batch.target,
                "test",
                config.minimum_query_finite_fraction,
            )
            finite_fraction[positions] = fraction
            rag_eligible[positions] = eligible
            fallback_reason[positions] = reason

        query_positions = np.flatnonzero(rag_eligible)
        datastore_positions = np.flatnonzero(datastore_eligible)
        started = perf_counter()
        if len(query_positions) and len(datastore_positions):
            retrieval_k = min(max_selected_k, len(datastore_positions))
            selected_distances, selected_ids = blockwise_topk(
                representation[query_positions],
                datastore_representation[datastore_positions],
                prepared.indices("test")[query_positions],
                prepared.indices("datastore")[datastore_positions],
                query_calendar_ticks=prepared.calendar_ticks("test")[query_positions],
                datastore_calendar_ticks=prepared.calendar_ticks("datastore")[
                    datastore_positions
                ],
                retrieval_period=int(prepared.config["retrieval_period"]),
                datastore_end_ticks_by_item=prepared.datastore_end_ticks_by_item,
                k=retrieval_k,
                stride=int(prepared.config["datastore_stride"]),
                horizon=prepared.prediction_length,
                scope=config.retrieval_scope,
                metric=config.distance_metric,
                minimum_overlap_fraction=config.minimum_overlap_fraction,
                query_block_size=config.query_block_size,
                datastore_block_size=config.datastore_block_size,
                require_complete_k=False,
            )
            distances[query_positions, :retrieval_k] = selected_distances
            valid = selected_ids >= 0
            mapped = np.full_like(selected_ids, -1)
            mapped[valid] = datastore_positions[selected_ids[valid]]
            neighbor_ids[query_positions, :retrieval_k] = mapped
            enough = np.count_nonzero(valid, axis=1) >= max_selected_k
            rejected = query_positions[~enough]
            rag_eligible[rejected] = False
            fallback_reason[rejected] = 2
        else:
            rag_eligible[query_positions] = False
            fallback_reason[query_positions] = 2
        timings["test.retrieval_seconds"] = perf_counter() - started

        eligible_neighbor_ids = np.asarray(neighbor_ids)[np.asarray(rag_eligible)]
        selected = np.unique(eligible_neighbor_ids[eligible_neighbor_ids >= 0])
        fit_forecast_ids = np.load(
            fit_root / fit_arrays["datastore.selected_forecast_id"], mmap_mode="r"
        )
        missing = np.setdiff1d(selected, np.asarray(fit_forecast_ids))
        reused_neighbor_forecasts = int(len(selected) - len(missing))
        computed_neighbor_forecasts = int(len(missing))
        missing_id_path = root / "datastore" / "selected_forecast_id.npy"
        missing_value_path = root / "datastore" / "selected_forecast.npy"
        missing_ids = _memmap(missing_id_path, missing.shape, np.int64)
        missing_values = _memmap(
            missing_value_path,
            (len(missing), datastore_target.shape[1], prepared.prediction_length),
            np.float32,
        )
        missing_ids[:] = missing
        timings["test.neighbor_forecast_seconds"] = 0.0
        datastore_references = prepared.indices("datastore")
        for start in range(0, len(missing), config.model_batch_size):
            stop = min(start + config.model_batch_size, len(missing))
            batch = reader.read(datastore_references[missing[start:stop]])
            missing_values[start:stop] = _timed_forecast(
                forecaster,
                batch.context,
                timings,
                "test.neighbor_forecast_seconds",
                horizon=prepared.prediction_length,
            )
        arrays["datastore.selected_forecast_id"] = str(
            missing_id_path.relative_to(root)
        )
        arrays["datastore.selected_forecast"] = str(
            missing_value_path.relative_to(root)
        )

        timings["test.context_construction_seconds"] = 0.0
        timings["test.context_forecast_seconds"] = 0.0
        eligible_positions = np.flatnonzero(rag_eligible)
        for start in range(0, len(eligible_positions), config.model_batch_size):
            positions = eligible_positions[start : start + config.model_batch_size]
            ids = np.asarray(neighbor_ids[positions])
            query_batch = reader.read(references[positions])
            neighbor_batch = reader.read(datastore_references[ids.reshape(-1)])
            neighbor_context = neighbor_batch.context.reshape(
                len(positions), max_selected_k, 1, prepared.context_length
            )
            neighbor_target = np.asarray(datastore_target[ids])
            for k in selected_ks:
                started = perf_counter()
                retrieval_context = _query_scaled_retrieval_context(
                    query_batch.context,
                    neighbor_context[:, :k],
                    neighbor_target[:, :k],
                )
                _record_seconds(
                    timings,
                    "test.context_construction_seconds",
                    perf_counter() - started,
                )
                context_forecasts[k][positions] = _timed_forecast(
                    forecaster,
                    query_batch.context,
                    timings,
                    "test.context_forecast_seconds",
                    horizon=prepared.prediction_length,
                    retrieval_context=retrieval_context,
                )

        for store in (
            representation,
            scale,
            finite_fraction,
            distances,
            neighbor_ids,
            *context_forecasts.values(),
            missing_ids,
            missing_values,
        ):
            store.flush()

    rag_eligible.flush()
    fallback_reason.flush()
    manifest: dict[str, object] = {
        **identity,
        "format": "adaptime_eval_extraction",
        "signature": signature,
        "status": "completed",
        "arrays": arrays,
        "counts": {"test": total},
        "eligibility": {
            "total": total,
            "rag_eligible": int(np.count_nonzero(rag_eligible)),
            "vanilla_fallback": int(total - np.count_nonzero(rag_eligible)),
            "fallback_reasons": {
                label: int(np.count_nonzero(fallback_reason == code))
                for code, label in FALLBACK_REASONS.items()
                if code != 0
            },
        },
        "neighbor_forecasts": {
            "reused_from_fit_extraction": reused_neighbor_forecasts,
            "computed_for_test": computed_neighbor_forecasts,
        },
        "fallback_reason_codes": {
            str(code): label for code, label in FALLBACK_REASONS.items()
        },
        "timing_seconds": timings,
    }
    _atomic_json(manifest_path, manifest)
    return manifest_path


def open_eval_extraction(path: str | Path) -> tuple[Path, dict[str, object]]:
    manifest_path = Path(path).expanduser().resolve()
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != EVAL_EXTRACTION_SCHEMA
        or manifest.get("format") != "adaptime_eval_extraction"
        or manifest.get("status") != "completed"
    ):
        raise ValueError("Adaptime evaluation extraction is not completed")
    return manifest_path.parent, manifest
