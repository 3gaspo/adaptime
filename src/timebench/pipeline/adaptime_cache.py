"""Append-only window computations shared by Adaptime adaptation methods."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np

from timebench.adaptime.retrieval import context_representation
from timebench.evaluation.adaptation_data import PreparedDataset


WINDOW_CACHE_SCHEMA = 1


def _canonical_hash(value: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    os.replace(temporary, path)


class SharedWindowCache:
    """Persist sparse forecasts and representations by exact source window."""

    def __init__(
        self,
        root: str | Path,
        *,
        prepared: PreparedDataset,
        forecaster: Any,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.prepared = prepared
        self.forecaster = forecaster
        self.identity: dict[str, object] = {
            "schema_version": WINDOW_CACHE_SCHEMA,
            "prepared_signature": prepared.signature,
            "model": str(forecaster.model_name),
            "weights_id": str(forecaster.weights_id),
            "prediction_length": prepared.prediction_length,
        }
        self.signature = _canonical_hash(self.identity)
        self.manifest_path = self.root / "manifest.json"
        if self.manifest_path.is_file():
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if self.manifest.get("signature") != self.signature:
                raise ValueError(f"shared window cache identity differs: {self.root}")
        else:
            self.manifest = {
                **self.identity,
                "format": "adaptime_shared_window_cache",
                "signature": self.signature,
                "chunks": [],
            }
            _atomic_json(self.manifest_path, self.manifest)
        self._indices: dict[str, dict[tuple[int, ...], tuple[str, int]]] = {}
        self._loaded_values: dict[str, np.ndarray] = {}

    def _index(self, kind: str) -> dict[tuple[int, ...], tuple[str, int]]:
        if kind in self._indices:
            return self._indices[kind]
        result: dict[tuple[int, ...], tuple[str, int]] = {}
        for chunk in self.manifest["chunks"]:
            if chunk["kind"] != kind:
                continue
            references = np.load(self.root / chunk["references"], allow_pickle=False)
            for row, reference in enumerate(references):
                result[tuple(map(int, reference))] = (str(chunk["values"]), row)
        self._indices[kind] = result
        return result

    def _append(self, kind: str, references: np.ndarray, values: np.ndarray) -> None:
        chunk_number = len(self.manifest["chunks"])
        directory = self.root / "chunks"
        directory.mkdir(parents=True, exist_ok=True)
        while (directory / f"{chunk_number:06d}_references.npy").exists():
            chunk_number += 1
        references_path = directory / f"{chunk_number:06d}_references.npy"
        values_path = directory / f"{chunk_number:06d}_values.npy"
        _atomic_npy(references_path, np.asarray(references, dtype=np.int64))
        _atomic_npy(values_path, np.asarray(values, dtype=np.float32))
        relative_references = str(references_path.relative_to(self.root))
        relative_values = str(values_path.relative_to(self.root))
        chunk = {
            "kind": kind,
            "references": relative_references,
            "values": relative_values,
            "rows": int(len(references)),
        }
        self.manifest["chunks"].append(chunk)
        _atomic_json(self.manifest_path, self.manifest)
        index = self._index(kind)
        for row, reference in enumerate(references):
            index[tuple(map(int, reference))] = (relative_values, row)
        self._loaded_values[relative_values] = np.asarray(values, dtype=np.float32)

    def _values(
        self,
        kind: str,
        references: np.ndarray,
        build: Callable[[np.ndarray], np.ndarray],
    ) -> tuple[np.ndarray, float]:
        requested = np.asarray(references, dtype=np.int64)
        keys = [tuple(map(int, row)) for row in requested]
        index = self._index(kind)
        missing_positions = np.asarray(
            [position for position, key in enumerate(keys) if key not in index],
            dtype=np.int64,
        )
        compute_seconds = 0.0
        if len(missing_positions):
            started = perf_counter()
            missing_values = np.asarray(build(missing_positions), dtype=np.float32)
            compute_seconds = perf_counter() - started
            if len(missing_values) != len(missing_positions):
                raise ValueError("shared cache builder returned the wrong row count")
            self._append(kind, requested[missing_positions], missing_values)
            index = self._index(kind)
        locations: dict[str, list[tuple[int, int]]] = {}
        for output_row, key in enumerate(keys):
            relative, source_row = index[key]
            locations.setdefault(relative, []).append((output_row, source_row))
        result: np.ndarray | None = None
        for relative, rows in locations.items():
            values = self._loaded_values.get(relative)
            if values is None:
                values = np.load(self.root / relative, mmap_mode="r", allow_pickle=False)
                self._loaded_values[relative] = values
            if result is None:
                result = np.empty((len(requested), *values.shape[1:]), dtype=np.float32)
            output_rows, source_rows = zip(*rows, strict=True)
            result[np.asarray(output_rows)] = values[np.asarray(source_rows)]
        if result is None:
            raise ValueError("cannot read an empty shared-cache request")
        return result, compute_seconds

    def forecasts(
        self,
        references: np.ndarray,
        contexts: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        contexts = np.asarray(contexts, dtype=np.float32)
        refs = np.asarray(references, dtype=np.int64).reshape(-1, 3)
        if len(refs) != len(contexts):
            raise ValueError("forecast cache references and contexts do not align")
        cache_refs = np.column_stack(
            (refs, np.full(len(refs), contexts.shape[-1], dtype=np.int64))
        )

        def build(positions: np.ndarray) -> np.ndarray:
            return np.asarray(
                self.forecaster.forecast(contexts[positions]), dtype=np.float32
            )

        return self._values("forecast", cache_refs, build)

    def representations(
        self,
        references: np.ndarray,
        contexts: np.ndarray,
        mode: str,
    ) -> tuple[np.ndarray, float]:
        contexts = np.asarray(contexts, dtype=np.float32)
        refs = np.asarray(references, dtype=np.int64).reshape(-1, 3)
        if len(refs) != len(contexts):
            raise ValueError("representation cache references and contexts do not align")
        cache_refs = np.column_stack(
            (refs, np.full(len(refs), contexts.shape[-1], dtype=np.int64))
        )

        def build(positions: np.ndarray) -> np.ndarray:
            selected = contexts[positions]
            return (
                np.asarray(self.forecaster.represent(selected), dtype=np.float32)
                if mode == "model"
                else context_representation(selected, mode)
            )

        return self._values(f"representation:{mode}", cache_refs, build)

    def forecasts_for_references(
        self,
        references: np.ndarray,
        *,
        reader: Any,
    ) -> tuple[np.ndarray, float]:
        refs = np.asarray(references, dtype=np.int64).reshape(-1, 3)
        cache_refs = np.column_stack(
            (
                refs,
                np.full(len(refs), self.prepared.context_length, dtype=np.int64),
            )
        )

        def build(positions: np.ndarray) -> np.ndarray:
            contexts = reader.read(refs[positions]).context
            return np.asarray(self.forecaster.forecast(contexts), dtype=np.float32)

        return self._values("forecast", cache_refs, build)

    def representations_for_references(
        self,
        references: np.ndarray,
        *,
        reader: Any,
        mode: str,
    ) -> tuple[np.ndarray, float]:
        refs = np.asarray(references, dtype=np.int64).reshape(-1, 3)
        cache_refs = np.column_stack(
            (
                refs,
                np.full(len(refs), self.prepared.context_length, dtype=np.int64),
            )
        )

        def build(positions: np.ndarray) -> np.ndarray:
            contexts = reader.read(refs[positions]).context
            return (
                np.asarray(self.forecaster.represent(contexts), dtype=np.float32)
                if mode == "model"
                else context_representation(contexts, mode)
            )

        return self._values(f"representation:{mode}", cache_refs, build)


def shared_window_cache_root(
    artifact_root: str | Path,
    prepared: PreparedDataset,
    forecaster: Any,
) -> Path:
    identity = {
        "prepared_signature": prepared.signature,
        "model": str(forecaster.model_name),
        "weights_id": str(forecaster.weights_id),
        "prediction_length": prepared.prediction_length,
    }
    signature = _canonical_hash(identity)[:16]
    return (
        Path(artifact_root).expanduser().resolve()
        / "window_cache"
        / prepared.target_mode
        / str(prepared.config["dataset"])
        / str(prepared.config["term"])
        / signature
    )
