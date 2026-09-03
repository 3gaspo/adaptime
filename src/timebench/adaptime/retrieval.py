"""Memory-bounded exact retrieval for Adaptime window representations."""

from __future__ import annotations

import numpy as np


RETRIEVAL_SCOPES = ("all", "same_series", "other_series")
DISTANCE_METRICS = ("euclidean", "cosine")


def context_representation(context: np.ndarray, mode: str) -> np.ndarray:
    """Return raw or channel-wise instance-normalized flattened contexts."""

    values = np.asarray(context, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("contexts must have shape (batch, channels, lookback)")
    if mode == "raw":
        return np.ascontiguousarray(values.reshape(len(values), -1))
    if mode != "instance":
        raise ValueError("representation mode must be raw or instance")
    mean = values.mean(axis=-1, keepdims=True)
    scale = np.maximum(values.std(axis=-1, keepdims=True), 1e-8)
    return np.ascontiguousarray(((values - mean) / scale).reshape(len(values), -1))


def _distance(query: np.ndarray, datastore: np.ndarray, metric: str) -> np.ndarray:
    if metric == "euclidean":
        query_norm = np.einsum("ij,ij->i", query, query)[:, None]
        datastore_norm = np.einsum("ij,ij->i", datastore, datastore)[None, :]
        squared = query_norm + datastore_norm - 2.0 * (query @ datastore.T)
        return np.maximum(squared, 0.0)
    if metric == "cosine":
        query_scale = np.maximum(np.linalg.norm(query, axis=1), 1e-8)
        datastore_scale = np.maximum(np.linalg.norm(datastore, axis=1), 1e-8)
        similarity = (query @ datastore.T) / query_scale[:, None] / datastore_scale[None, :]
        return 1.0 - np.clip(similarity, -1.0, 1.0)
    raise ValueError(f"unknown distance metric {metric!r}")


def _eligible(
    query_refs: np.ndarray,
    datastore_refs: np.ndarray,
    *,
    query_ticks: np.ndarray | None,
    datastore_ticks: np.ndarray | None,
    retrieval_period: int | None,
    datastore_end_ticks_by_item: np.ndarray | None,
    scope: str,
    stride: int,
    horizon: int,
) -> np.ndarray:
    query_item = query_refs[:, 0, None]
    query_channel = query_refs[:, 1, None]
    query_origin = (
        query_refs[:, 2, None]
        if query_ticks is None
        else np.asarray(query_ticks, dtype=np.int64)[:, None]
    )
    datastore_item = datastore_refs[None, :, 0]
    datastore_channel = datastore_refs[None, :, 1]
    datastore_origin = (
        datastore_refs[None, :, 2]
        if datastore_ticks is None
        else np.asarray(datastore_ticks, dtype=np.int64)[None, :]
    )
    same_series = (query_item == datastore_item) & (query_channel == datastore_channel)
    if datastore_end_ticks_by_item is None:
        allowed = ((query_origin - datastore_origin) % int(stride) == 0) & (
            datastore_origin + int(horizon) <= query_origin
        )
    else:
        if retrieval_period is None or int(retrieval_period) <= 0:
            raise ValueError("retrieval_period is required with fixed datastore ends")
        datastore_end = np.asarray(datastore_end_ticks_by_item, dtype=np.int64)[
            datastore_refs[:, 0]
        ][None, :]
        last = np.minimum(query_origin - int(horizon), datastore_end)
        last = last - (last - query_origin) % int(retrieval_period)
        allowed = (datastore_origin <= last) & (
            (last - datastore_origin) % int(stride) == 0
        )
    if scope == "same_series":
        return allowed & same_series
    if scope == "other_series":
        return allowed & ~same_series
    if scope == "all":
        return allowed
    raise ValueError(f"retrieval scope must be one of {RETRIEVAL_SCOPES}")


def blockwise_topk(
    query: np.ndarray,
    datastore: np.ndarray,
    query_references: np.ndarray,
    datastore_references: np.ndarray,
    *,
    query_calendar_ticks: np.ndarray | None = None,
    datastore_calendar_ticks: np.ndarray | None = None,
    retrieval_period: int | None = None,
    datastore_end_ticks_by_item: np.ndarray | None = None,
    k: int,
    stride: int,
    horizon: int,
    scope: str = "all",
    metric: str = "euclidean",
    query_block_size: int = 256,
    datastore_block_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Find exact neighbors without allocating the full distance matrix.

    Returned indices address the complete datastore array. With fixed datastore
    endpoints, the latest candidate first shifts to the query's dataset-period
    residue and earlier candidates follow the configured datastore stride.
    """

    query = np.asarray(query, dtype=np.float32)
    datastore = np.asarray(datastore, dtype=np.float32)
    query_references = np.asarray(query_references, dtype=np.int64).reshape(-1, 3)
    datastore_references = np.asarray(datastore_references, dtype=np.int64).reshape(-1, 3)
    if query.ndim != 2 or datastore.ndim != 2 or query.shape[1] != datastore.shape[1]:
        raise ValueError("query and datastore representations must be aligned matrices")
    if len(query) != len(query_references) or len(datastore) != len(datastore_references):
        raise ValueError("representations and references must have matching row counts")
    if query_calendar_ticks is not None and len(query_calendar_ticks) != len(query):
        raise ValueError("query calendar ticks must match query rows")
    if datastore_calendar_ticks is not None and len(datastore_calendar_ticks) != len(datastore):
        raise ValueError("datastore calendar ticks must match datastore rows")
    if datastore_end_ticks_by_item is not None:
        max_item = int(datastore_references[:, 0].max(initial=-1))
        if len(datastore_end_ticks_by_item) <= max_item:
            raise ValueError("fixed datastore ends do not cover every datastore item")
    if int(k) <= 0 or int(k) > len(datastore):
        raise ValueError("k must be positive and no larger than the datastore")
    if int(stride) <= 0 or int(horizon) <= 0:
        raise ValueError("stride and horizon must be positive")
    if scope not in RETRIEVAL_SCOPES:
        raise ValueError(f"retrieval scope must be one of {RETRIEVAL_SCOPES}")
    if metric not in DISTANCE_METRICS:
        raise ValueError(f"distance metric must be one of {DISTANCE_METRICS}")

    neighbor_ids = np.empty((len(query), int(k)), dtype=np.int64)
    neighbor_distances = np.empty((len(query), int(k)), dtype=np.float32)
    for query_start in range(0, len(query), int(query_block_size)):
        query_stop = min(query_start + int(query_block_size), len(query))
        query_block = query[query_start:query_stop]
        query_refs = query_references[query_start:query_stop]
        best_distance = np.full((len(query_block), int(k)), np.inf, dtype=np.float32)
        best_index = np.full((len(query_block), int(k)), -1, dtype=np.int64)

        for datastore_start in range(0, len(datastore), int(datastore_block_size)):
            datastore_stop = min(datastore_start + int(datastore_block_size), len(datastore))
            candidate = datastore[datastore_start:datastore_stop]
            distance = _distance(query_block, candidate, metric).astype(np.float32, copy=False)
            allowed = _eligible(
                query_refs,
                datastore_references[datastore_start:datastore_stop],
                query_ticks=(
                    None
                    if query_calendar_ticks is None
                    else query_calendar_ticks[query_start:query_stop]
                ),
                datastore_ticks=(
                    None
                    if datastore_calendar_ticks is None
                    else datastore_calendar_ticks[datastore_start:datastore_stop]
                ),
                retrieval_period=retrieval_period,
                datastore_end_ticks_by_item=datastore_end_ticks_by_item,
                scope=scope,
                stride=stride,
                horizon=horizon,
            )
            distance[~allowed] = np.inf
            local_k = min(int(k), candidate.shape[0])
            local_position = np.argpartition(distance, local_k - 1, axis=1)[:, :local_k]
            local_distance = np.take_along_axis(distance, local_position, axis=1)
            local_index = local_position.astype(np.int64) + datastore_start

            combined_distance = np.concatenate((best_distance, local_distance), axis=1)
            combined_index = np.concatenate((best_index, local_index), axis=1)
            selected = np.argpartition(combined_distance, int(k) - 1, axis=1)[:, : int(k)]
            best_distance = np.take_along_axis(combined_distance, selected, axis=1)
            best_index = np.take_along_axis(combined_index, selected, axis=1)

        order = np.argsort(best_distance, axis=1)
        best_distance = np.take_along_axis(best_distance, order, axis=1)
        best_index = np.take_along_axis(best_index, order, axis=1)
        missing = np.flatnonzero(~np.isfinite(best_distance[:, -1]))
        if len(missing):
            first = int(query_start + missing[0])
            raise ValueError(f"query row {first} has fewer than k={k} eligible neighbors")
        if metric == "euclidean":
            np.sqrt(best_distance, out=best_distance)
        neighbor_ids[query_start:query_stop] = best_index
        neighbor_distances[query_start:query_stop] = best_distance

    return neighbor_distances, neighbor_ids
