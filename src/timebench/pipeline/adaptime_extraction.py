"""One-pass, disk-backed extraction for Adaptime retrieval experiments."""

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
from timebench.evaluation.adaptation_data import PreparedDataset, QUERY_SPLITS
from timebench.evaluation.timing import EvaluationTimer


EXTRACTION_SCHEMA = 1


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
    first = reader.read(references[:first_stop])
    representation_key = f"{split}.representation_seconds"
    first_representation = _timed_represent(
        forecaster,
        first.context,
        config.representation,
        timings,
        representation_key,
    )
    channels = int(first.context.shape[1])
    horizon = prepared.prediction_length
    target_path = root / split / "target.npy"
    representation_path = root / split / "representation.npy"
    scale_path = root / split / "query_scale.npy"
    target = _memmap(target_path, (len(references), channels, horizon), np.float32)
    representation = _memmap(
        representation_path,
        (len(references), first_representation.shape[1]),
        np.float32,
    )
    scale = _memmap(scale_path, (len(references), channels), np.float32)
    vanilla: np.memmap | None = None
    if include_vanilla:
        vanilla_path = root / split / "vanilla.npy"
        vanilla = _memmap(
            vanilla_path,
            (len(references), channels, horizon),
            np.float32,
        )
        arrays[f"{split}.vanilla"] = str(vanilla_path.relative_to(root))
    arrays[f"{split}.target"] = str(target_path.relative_to(root))
    arrays[f"{split}.representation"] = str(representation_path.relative_to(root))
    arrays[f"{split}.query_scale"] = str(scale_path.relative_to(root))

    target[:first_stop] = first.target
    representation[:first_stop] = first_representation
    scale[:first_stop] = query_scale(first.context)
    if vanilla is not None:
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
        if vanilla is not None:
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
    if vanilla is not None:
        vanilla.flush()
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
    started = perf_counter()
    distances, ids = blockwise_topk(
        query_representation,
        datastore_representation,
        prepared.indices(split),
        prepared.indices("datastore"),
        query_calendar_ticks=prepared.calendar_ticks(split),
        datastore_calendar_ticks=prepared.calendar_ticks("datastore"),
        retrieval_period=int(prepared.config["retrieval_period"]),
        datastore_end_ticks_by_item=prepared.datastore_end_ticks_by_item,
        k=config.max_k,
        stride=int(prepared.config["datastore_stride"]),
        horizon=prepared.prediction_length,
        scope=config.retrieval_scope,
        metric=config.distance_metric,
        minimum_overlap_fraction=config.minimum_overlap_fraction,
        query_block_size=config.query_block_size,
        datastore_block_size=config.datastore_block_size,
    )
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
    selected = np.empty(0, dtype=np.int64)
    for split in QUERY_SPLITS:
        split_ids = np.load(root / arrays[f"{split}.neighbor_id"], mmap_mode="r")
        selected = np.union1d(
            selected,
            np.unique(np.asarray(split_ids).reshape(-1)),
        )
    datastore_refs = prepared.indices("datastore")
    reader = prepared.reader(cache_items=config.arrow_cache_items)
    sample = reader.read(datastore_refs[selected[:1]])
    values_path = root / "datastore" / "selected_forecast.npy"
    ids_path = root / "datastore" / "selected_forecast_id.npy"
    forecasts = _memmap(
        values_path,
        (len(selected), sample.context.shape[1], prepared.prediction_length),
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
    reader = prepared.reader(cache_items=config.arrow_cache_items)
    channels = int(np.load(root / arrays[f"{split}.target"], mmap_mode="r").shape[1])
    stores: dict[int, np.memmap] = {}
    for k in config.context_k:
        path = root / split / f"context_forecast_k{k}.npy"
        stores[k] = _memmap(
            path,
            (len(query_refs), channels, prepared.prediction_length),
            np.float32,
        )
        arrays[f"{split}.context_forecast_k{k}"] = str(path.relative_to(root))

    for start in range(0, len(query_refs), config.model_batch_size):
        stop = min(start + config.model_batch_size, len(query_refs))
        query_batch = reader.read(query_refs[start:stop])
        selected_ids = np.asarray(neighbor_ids[start:stop])
        flat_neighbor_batch = reader.read(datastore_refs[selected_ids.reshape(-1)])
        neighbor_context = flat_neighbor_batch.context.reshape(
            stop - start,
            config.max_k,
            channels,
            prepared.context_length,
        )
        neighbor_target = np.asarray(datastore_target[selected_ids])
        for k in config.context_k:
            started = perf_counter()
            retrieval_context = _query_scaled_retrieval_context(
                query_batch.context,
                neighbor_context[:, :k],
                neighbor_target[:, :k],
            )
            _record_seconds(
                timings,
                f"{split}.context_construction_k{k}_seconds",
                perf_counter() - started,
            )
            stores[k][start:stop] = _timed_forecast(
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
    """Extract every expensive quantity once for a complete ridge sweep."""

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
    for split in QUERY_SPLITS:
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
    for split in QUERY_SPLITS:
        _materialize_context_forecasts(
            prepared, split, forecaster, config, root, arrays, timings
        )
    timings["extraction_total_seconds"] = perf_counter() - extraction_started

    manifest: dict[str, object] = {
        **identity,
        "format": "adaptime_extraction",
        "signature": signature,
        "status": "completed",
        "arrays": arrays,
        "counts": {
            split: int(len(prepared.indices(split))) for split in ("datastore", *QUERY_SPLITS)
        },
        "computed_neighbor_forecasts": int(len(unique_forecast)),
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
    if manifest.get("schema_version") != EXTRACTION_SCHEMA or manifest.get("status") != "completed":
        raise ValueError("Adaptime extraction is not a completed schema-1 artifact")
    return manifest_path.parent, manifest
