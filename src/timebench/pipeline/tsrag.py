"""Source-faithful TS-RAG extraction and frozen TIME evaluation."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from timebench.adaptime.ridge import query_scale
from timebench.evaluation.timing import EvaluationTimer
from timebench.external_models.tsrag.retriever import TSRAGIndex, TSRAGRetriever
from timebench.model_loading.tsrag import LoadedTSRAG
from timebench.pipeline.adaptime_testing import _aggregate_metrics, _metric_values
from timebench.pipeline.tsrag_data import (
    TSRAG_CONTEXT_LENGTH,
    TSRAG_DATASTORE_STRIDE,
    TSRAG_NATIVE_HORIZON,
    TSRAGPreparedDataset,
)


TSRAG_SOURCE_COMMIT = "73ac807789d2e61b8a3dfc8514e3fc947fe185cc"
TSRAG_EXTRACTION_SCHEMA = 1
TSRAG_RESULT_SCHEMA = 1
TSRAG_TOP_K = 10
TSRAG_EMBEDDING_BATCH_SIZE = 512
TSRAG_EMBEDDING_DIMENSION = 768
TSRAG_METHODS = ("vanilla", "tsrag")
TSRAG_METRICS = ("mse", "mae", "mase")


@dataclass(frozen=True)
class TSRAGRuntimeConfig:
    model_batch_size: int = 256
    arrow_cache_items: int = 2

    def validate(self) -> None:
        if int(self.model_batch_size) <= 0:
            raise ValueError("model_batch_size must be positive")
        if int(self.arrow_cache_items) <= 0:
            raise ValueError("arrow_cache_items must be positive")


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


def _groups(references: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    refs = np.asarray(references, dtype=np.int64)
    result: dict[tuple[int, int], list[int]] = {}
    for position, (item, channel, _) in enumerate(refs):
        result.setdefault((int(item), int(channel)), []).append(position)
    return {
        key: np.asarray(positions, dtype=np.int64)
        for key, positions in result.items()
    }


def _representation(
    retriever: TSRAGRetriever,
    contexts: np.ndarray,
) -> np.ndarray:
    values = retriever.representation(torch.from_numpy(np.asarray(contexts)))
    array = values.detach().float().cpu().numpy()
    if array.ndim != 2 or array.shape[1] != TSRAG_EMBEDDING_DIMENSION:
        raise ValueError(
            f"TS-RAG Chronos-T5 EOS embeddings must have dimension "
            f"{TSRAG_EMBEDDING_DIMENSION}, received {array.shape}"
        )
    return np.ascontiguousarray(array, dtype=np.float32)


def _materialize_representations(
    prepared: TSRAGPreparedDataset,
    split: str,
    retriever: TSRAGRetriever,
    root: Path,
    runtime: TSRAGRuntimeConfig,
) -> tuple[np.memmap, float]:
    references = prepared.indices(split)
    reader = prepared.reader(cache_items=runtime.arrow_cache_items)
    store = _memmap(
        root / split / "representation.npy",
        (len(references), TSRAG_EMBEDDING_DIMENSION),
        np.float32,
    )
    timer = EvaluationTimer()
    timer.start()
    for start in range(0, len(references), TSRAG_EMBEDDING_BATCH_SIZE):
        stop = min(start + TSRAG_EMBEDDING_BATCH_SIZE, len(references))
        batch = reader.read(references[start:stop], target_length=1)
        store[start:stop] = _representation(retriever, batch.context)
    seconds = timer.stop()
    store.flush()
    return store, seconds


def _build_indexes(
    references: np.ndarray,
    representations: np.ndarray,
    retriever: TSRAGRetriever,
) -> tuple[dict[tuple[int, int], tuple[np.ndarray, TSRAGIndex]], float]:
    started = perf_counter()
    indexes = {
        key: (positions, retriever.build_index(np.asarray(representations[positions])))
        for key, positions in _groups(references).items()
    }
    return indexes, perf_counter() - started


def _search(
    query_references: np.ndarray,
    query_representations: np.ndarray,
    indexes: dict[tuple[int, int], tuple[np.ndarray, TSRAGIndex]],
) -> tuple[np.ndarray, np.ndarray]:
    queries = np.asarray(query_references, dtype=np.int64)
    distances = np.empty((len(queries), TSRAG_TOP_K), dtype=np.float32)
    neighbors = np.empty((len(queries), TSRAG_TOP_K), dtype=np.int64)
    for key, query_positions in _groups(queries).items():
        if key not in indexes:
            raise ValueError(f"TS-RAG datastore has no same-series index for {key}")
        datastore_positions, index = indexes[key]
        selected_distances, selected_local = index.search(
            np.asarray(query_representations[query_positions]),
            top_k=TSRAG_TOP_K,
        )
        distances[query_positions] = selected_distances
        neighbors[query_positions] = datastore_positions[selected_local]
    return distances, neighbors


def extract_tsrag_features(
    prepared_path: str | Path,
    retriever: TSRAGRetriever,
    runtime: TSRAGRuntimeConfig,
    output_dir: str | Path,
) -> Path:
    """Apply TS-RAG's stride-one EOS/FAISS extraction to the TIME test rows."""

    runtime.validate()
    prepared = TSRAGPreparedDataset(prepared_path)
    identity = {
        "schema_version": TSRAG_EXTRACTION_SCHEMA,
        "prepared_signature": prepared.signature,
        "source_commit": TSRAG_SOURCE_COMMIT,
        "context_length": TSRAG_CONTEXT_LENGTH,
        "native_prediction_length": TSRAG_NATIVE_HORIZON,
        "top_k": TSRAG_TOP_K,
        "datastore_stride": TSRAG_DATASTORE_STRIDE,
        "retrieval_scope": "same_series",
        "embedding": "chronos_t5_base_eos",
        "embedding_batch_size": TSRAG_EMBEDDING_BATCH_SIZE,
        "embedding_dimension": TSRAG_EMBEDDING_DIMENSION,
        "fetcher": "raw_same_series_512_plus_64",
        "index": "faiss.IndexFlatL2_float32",
        "selection": "top_k_plus_one_then_remove_final_result",
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
        raise FileExistsError(f"TS-RAG extraction already differs: {root}")

    extraction_started = perf_counter()
    datastore_representation, datastore_seconds = _materialize_representations(
        prepared, "datastore", retriever, root, runtime
    )
    test_representation, test_seconds = _materialize_representations(
        prepared, "test", retriever, root, runtime
    )
    indexes, index_seconds = _build_indexes(
        prepared.indices("datastore"), datastore_representation, retriever
    )
    search_started = perf_counter()
    distances, neighbors = _search(
        prepared.indices("test"), test_representation, indexes
    )
    search_seconds = perf_counter() - search_started
    distance_store = _memmap(
        root / "test" / "neighbor_distance.npy", distances.shape, np.float32
    )
    neighbor_store = _memmap(
        root / "test" / "neighbor_id.npy", neighbors.shape, np.int64
    )
    distance_store[:] = distances
    neighbor_store[:] = neighbors
    distance_store.flush()
    neighbor_store.flush()

    arrays = {
        "datastore.representation": "datastore/representation.npy",
        "test.representation": "test/representation.npy",
        "test.neighbor_distance": "test/neighbor_distance.npy",
        "test.neighbor_id": "test/neighbor_id.npy",
    }
    timings = {
        "datastore_representation_seconds": datastore_seconds,
        "datastore_index_construction_seconds": index_seconds,
        "test_representation_seconds": test_seconds,
        "test_retrieval_seconds": search_seconds,
        "extraction_total_seconds": perf_counter() - extraction_started,
    }
    _atomic_json(
        manifest_path,
        {
            **identity,
            "format": "adaptime_tsrag_extraction",
            "signature": signature,
            "status": "completed",
            "runtime_config": asdict(runtime),
            "arrays": arrays,
            "counts": {
                "datastore": int(len(prepared.indices("datastore"))),
                "test": int(len(prepared.indices("test"))),
            },
            "timing_seconds": timings,
        },
    )
    return manifest_path


def open_tsrag_extraction(path: str | Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = Path(path).expanduser().resolve()
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != TSRAG_EXTRACTION_SCHEMA
        or manifest.get("status") != "completed"
        or manifest.get("format") != "adaptime_tsrag_extraction"
    ):
        raise ValueError("TS-RAG extraction is not a completed schema-1 artifact")
    return manifest_path.parent, manifest


def _native_forecast(
    model: torch.nn.Module,
    context: np.ndarray,
    median_index: int,
    device: torch.device,
    *,
    retrieved: np.ndarray | None = None,
    distances: np.ndarray | None = None,
) -> np.ndarray:
    kwargs: dict[str, torch.Tensor] = {}
    if retrieved is not None:
        kwargs["retrieved_seq"] = torch.from_numpy(retrieved).to(device)
    if distances is not None:
        kwargs["distances"] = torch.from_numpy(distances).to(device)
    with torch.inference_mode():
        output = model(
            context=torch.from_numpy(np.asarray(context, dtype=np.float32)).to(device),
            **kwargs,
        )
    values = output.quantile_preds[:, int(median_index), :].detach().float().cpu().numpy()
    if values.shape != (len(context), TSRAG_NATIVE_HORIZON):
        raise ValueError(f"TS-RAG native forecast has unexpected shape {values.shape}")
    return np.asarray(values, dtype=np.float32)


def _fetch_neighbors(
    prepared: TSRAGPreparedDataset,
    reader: Any,
    neighbor_ids: np.ndarray,
) -> np.ndarray:
    ids = np.asarray(neighbor_ids, dtype=np.int64)
    batch = reader.read(
        prepared.indices("datastore")[ids.reshape(-1)],
        target_length=TSRAG_NATIVE_HORIZON,
    )
    sequences = np.concatenate((batch.context, batch.target), axis=-1)[:, 0]
    return sequences.reshape(len(ids), TSRAG_TOP_K, -1)


def _rollout_vanilla(
    loaded: LoadedTSRAG,
    context: np.ndarray,
    horizon: int,
    device: torch.device,
    timings: dict[str, float],
) -> np.ndarray:
    current = np.asarray(context, dtype=np.float32)
    chunks: list[np.ndarray] = []
    remaining = int(horizon)
    while remaining > 0:
        timer = EvaluationTimer()
        timer.start()
        native = _native_forecast(
            loaded.vanilla_model, current, loaded.median_index, device
        )
        timings["vanilla_model_seconds"] += timer.stop()
        take = min(remaining, TSRAG_NATIVE_HORIZON)
        chunks.append(native[:, :take])
        remaining -= take
        if remaining:
            current = np.concatenate((current, native), axis=-1)[
                :, -TSRAG_CONTEXT_LENGTH:
            ]
    return np.concatenate(chunks, axis=-1)


def _rollout_tsrag(
    loaded: LoadedTSRAG,
    retriever: TSRAGRetriever,
    prepared: TSRAGPreparedDataset,
    reader: Any,
    indexes: dict[tuple[int, int], tuple[np.ndarray, TSRAGIndex]],
    references: np.ndarray,
    context: np.ndarray,
    initial_neighbors: np.ndarray,
    initial_distances: np.ndarray,
    horizon: int,
    device: torch.device,
    timings: dict[str, float],
) -> np.ndarray:
    current = np.asarray(context, dtype=np.float32)
    chunks: list[np.ndarray] = []
    remaining = int(horizon)
    chunk = 0
    while remaining > 0:
        if chunk == 0:
            neighbor_ids = np.asarray(initial_neighbors, dtype=np.int64)
            distances = np.asarray(initial_distances, dtype=np.float32)
        else:
            timer = EvaluationTimer()
            timer.start()
            representation = _representation(retriever, current[:, None, :])
            timings["rollout_representation_seconds"] += timer.stop()
            started = perf_counter()
            distances, neighbor_ids = _search(references, representation, indexes)
            timings["rollout_retrieval_seconds"] += perf_counter() - started

        started = perf_counter()
        retrieved = _fetch_neighbors(prepared, reader, neighbor_ids)
        timings["retrieved_sequence_fetch_seconds"] += perf_counter() - started
        timer = EvaluationTimer()
        timer.start()
        native = _native_forecast(
            loaded.model,
            current,
            loaded.median_index,
            device,
            retrieved=retrieved,
            distances=distances,
        )
        timings["tsrag_model_seconds"] += timer.stop()
        take = min(remaining, TSRAG_NATIVE_HORIZON)
        chunks.append(native[:, :take])
        remaining -= take
        if remaining:
            current = np.concatenate((current, native), axis=-1)[
                :, -TSRAG_CONTEXT_LENGTH:
            ]
        chunk += 1
    return np.concatenate(chunks, axis=-1)


def evaluate_tsrag(
    prepared_path: str | Path,
    extraction_path: str | Path,
    loaded: LoadedTSRAG,
    retriever: TSRAGRetriever,
    runtime: TSRAGRuntimeConfig,
    output_dir: str | Path,
    *,
    seasonality: int,
    device: str | torch.device = "cuda",
) -> Path:
    """Evaluate native TS-RAG, rolling 64-step blocks only when H exceeds 64."""

    runtime.validate()
    prepared = TSRAGPreparedDataset(prepared_path)
    extraction_root, extraction = open_tsrag_extraction(extraction_path)
    if extraction["prepared_signature"] != prepared.signature:
        raise ValueError("TS-RAG extraction and prepared TIME windows do not match")
    horizon = prepared.prediction_length
    identity = {
        "schema_version": TSRAG_RESULT_SCHEMA,
        "prepared_signature": prepared.signature,
        "extraction_signature": extraction["signature"],
        "source_commit": TSRAG_SOURCE_COMMIT,
        "methods": list(TSRAG_METHODS),
        "metrics": list(TSRAG_METRICS),
        "rollout": (
            "native_single_call_crop" if horizon <= TSRAG_NATIVE_HORIZON
            else "autoregressive_64_step_reembed_retrieve"
        ),
    }
    signature = _canonical_hash(identity)
    root = Path(output_dir).expanduser().resolve()
    manifest_path = root / "result_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("signature") == signature and existing.get("status") == "completed":
            return manifest_path
        raise FileExistsError(f"TS-RAG result already differs: {root}")

    arrays = dict(extraction["arrays"])
    datastore_representation = np.load(
        extraction_root / arrays["datastore.representation"], mmap_mode="r"
    )
    initial_neighbors = np.load(
        extraction_root / arrays["test.neighbor_id"], mmap_mode="r"
    )
    initial_distances = np.load(
        extraction_root / arrays["test.neighbor_distance"], mmap_mode="r"
    )
    indexes, _ = _build_indexes(
        prepared.indices("datastore"), datastore_representation, retriever
    )
    test_references = prepared.indices("test")
    reader = prepared.reader(cache_items=runtime.arrow_cache_items)
    prediction_stores = {
        method: _memmap(
            root / "predictions" / f"{method}.npy",
            (len(test_references), 1, horizon),
            np.float32,
        )
        for method in TSRAG_METHODS
    }
    metric_stores = {
        (method, metric): _memmap(
            root / "metrics" / f"{method}_{metric}.npy",
            (len(test_references), 1),
            np.float32,
        )
        for method in TSRAG_METHODS
        for metric in TSRAG_METRICS
    }
    timings = {
        "vanilla_model_seconds": 0.0,
        "rollout_representation_seconds": 0.0,
        "rollout_retrieval_seconds": 0.0,
        "retrieved_sequence_fetch_seconds": 0.0,
        "tsrag_model_seconds": 0.0,
    }
    torch_device = torch.device(device)
    for start in range(0, len(test_references), runtime.model_batch_size):
        stop = min(start + runtime.model_batch_size, len(test_references))
        batch_references = np.asarray(test_references[start:stop])
        batch = reader.read(batch_references, target_length=horizon)
        context = batch.context[:, 0]
        predictions = {
            "vanilla": _rollout_vanilla(
                loaded, context, horizon, torch_device, timings
            ),
            "tsrag": _rollout_tsrag(
                loaded,
                retriever,
                prepared,
                reader,
                indexes,
                batch_references,
                context,
                initial_neighbors[start:stop],
                initial_distances[start:stop],
                horizon,
                torch_device,
                timings,
            ),
        }
        target = batch.target
        scale = query_scale(batch.context)
        for method, values in predictions.items():
            values = values[:, None, :]
            prediction_stores[method][start:stop] = values
            computed = _metric_values(
                values,
                target,
                scale,
                batch.context,
                int(seasonality),
            )
            for metric in TSRAG_METRICS:
                metric_stores[(method, metric)][start:stop] = computed[metric]

    for store in (*prediction_stores.values(), *metric_stores.values()):
        store.flush()
    summaries = {
        method: _aggregate_metrics(
            test_references,
            {
                metric: np.asarray(metric_stores[(method, metric)])
                for metric in TSRAG_METRICS
            },
        )
        for method in TSRAG_METHODS
    }
    extraction_timing = dict(extraction["timing_seconds"])
    tsrag_total = (
        float(extraction_timing["test_representation_seconds"])
        + float(extraction_timing["test_retrieval_seconds"])
        + timings["rollout_representation_seconds"]
        + timings["rollout_retrieval_seconds"]
        + timings["retrieved_sequence_fetch_seconds"]
        + timings["tsrag_model_seconds"]
    )
    method_seconds = {
        "vanilla": timings["vanilla_model_seconds"],
        "tsrag": tsrag_total,
    }
    timing = {
        "unit": "seconds",
        "test_windows": int(len(test_references)),
        "native_calls_per_window": int(np.ceil(horizon / TSRAG_NATIVE_HORIZON)),
        "methods": {
            method: {
                "total_seconds": float(seconds),
                "seconds_per_window": float(seconds) / len(test_references),
            }
            for method, seconds in method_seconds.items()
        },
        "components": {
            "initial_query_representation_seconds": float(
                extraction_timing["test_representation_seconds"]
            ),
            "initial_retrieval_seconds": float(
                extraction_timing["test_retrieval_seconds"]
            ),
            **timings,
        },
        "precomputed_extraction": {
            "datastore_representation_seconds": float(
                extraction_timing["datastore_representation_seconds"]
            ),
            "datastore_index_construction_seconds": float(
                extraction_timing["datastore_index_construction_seconds"]
            ),
            "complete_extraction_seconds": float(
                extraction_timing["extraction_total_seconds"]
            ),
        },
    }
    _atomic_json(root / "comparison_summary.json", {"methods": summaries, "timing": timing})
    _atomic_json(
        manifest_path,
        {
            **identity,
            "format": "adaptime_tsrag_time_comparison",
            "signature": signature,
            "status": "completed",
            "protocol": "frozen_tsrag_on_ridge_official_time_test_support",
            "context_length": TSRAG_CONTEXT_LENGTH,
            "prediction_length": horizon,
            "native_prediction_length": TSRAG_NATIVE_HORIZON,
            "top_k": TSRAG_TOP_K,
            "seasonality": int(seasonality),
            "parameters": {
                "trainable_during_evaluation": 0,
                "released_arm_parameters": loaded.adaptor_parameters,
            },
            "checkpoint": str(loaded.checkpoint),
            "timing": timing,
            "files": {
                "predictions": {
                    method: f"predictions/{method}.npy" for method in TSRAG_METHODS
                },
                "metrics": {
                    f"{method}.{metric}": f"metrics/{method}_{metric}.npy"
                    for method in TSRAG_METHODS
                    for metric in TSRAG_METRICS
                },
                "comparison_summary": "comparison_summary.json",
            },
        },
    )
    return manifest_path
