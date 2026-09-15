"""Coarse, timed ``run_n`` cache fragments for expensive source forecasts."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np

from timebench.evaluation.adaptation_data import PreparedDataset
from timebench.evaluation.timing import EvaluationTimer


FORECAST_CACHE_SCHEMA = 1
FORECAST_CACHE_FORMAT = "adaptime_source_forecast_cache"
DEFAULT_SHARD_ROWS = 8192
RUN_PATTERN = re.compile(r"run_(\d+)")


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    os.replace(temporary, path)


class SharedWindowCache:
    """Reuse only costly foundation forecasts across exact source windows."""

    def __init__(
        self,
        root: str | Path,
        *,
        prepared: PreparedDataset,
        forecaster: Any,
        shard_rows: int = DEFAULT_SHARD_ROWS,
    ) -> None:
        if int(shard_rows) <= 0:
            raise ValueError("forecast-cache shard_rows must be positive")
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.prepared = prepared
        self.forecaster = forecaster
        self.shard_rows = int(shard_rows)
        self.config: dict[str, object] = {
            "prepared_artifact": str(prepared.root),
            "prepared_config": dict(prepared.config),
            "model": str(forecaster.model_name),
            "weights_id": str(forecaster.weights_id),
            "prediction_length": prepared.prediction_length,
            "cached_value": "source_window_vanilla_backbone_forecast",
            "reference_columns": ["item", "channel", "origin", "context_length"],
        }
        self._index: dict[tuple[int, ...], tuple[Path, int]] = {}
        self._loaded_values: dict[Path, np.ndarray] = {}
        self._pending: dict[tuple[int, ...], np.ndarray] = {}
        self._run_dir: Path | None = None
        self._manifest: dict[str, object] | None = None
        self._closed = False
        self._timing = {
            "discovery_seconds": 0.0,
            "lookup_seconds": 0.0,
            "read_seconds": 0.0,
            "compute_seconds": 0.0,
            "write_seconds": 0.0,
            "manifest_seconds": 0.0,
        }
        self._counts = {
            "requested_rows": 0,
            "hit_rows": 0,
            "deduplicated_rows": 0,
            "computed_rows": 0,
            "written_rows": 0,
        }
        self._load_completed_runs()
        self._allocate_run()

    def __enter__(self) -> "SharedWindowCache":
        return self

    def __exit__(self, error_type, error, traceback) -> bool:
        if error_type is None:
            self.close()
        else:
            self._finish("interrupted", error)
        return False

    def _load_completed_runs(self) -> None:
        started = perf_counter()
        run_dirs = sorted(
            (
                path
                for path in self.root.iterdir()
                if path.is_dir() and RUN_PATTERN.fullmatch(path.name)
            ),
            key=lambda path: int(RUN_PATTERN.fullmatch(path.name).group(1)),
        )
        for run_dir in run_dirs:
            manifest_path = run_dir / "manifest.json"
            if not manifest_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("schema_version") != FORECAST_CACHE_SCHEMA
                or manifest.get("format") != FORECAST_CACHE_FORMAT
                or manifest.get("status") != "completed"
                or manifest.get("config") != self.config
            ):
                continue
            for shard in manifest.get("shards", []):
                references_path = run_dir / str(shard["references"])
                values_path = run_dir / str(shard["values"])
                references = np.load(references_path, allow_pickle=False)
                if len(references) != int(shard["rows"]):
                    raise ValueError(f"forecast-cache shard row count differs: {run_dir}")
                for row, reference in enumerate(references):
                    self._index[tuple(map(int, reference))] = (values_path, row)
        self._timing["discovery_seconds"] += perf_counter() - started

    def _allocate_run(self) -> None:
        if self._run_dir is not None:
            return
        indexes = [
            int(match.group(1))
            for path in self.root.iterdir()
            if path.is_dir() and (match := RUN_PATTERN.fullmatch(path.name))
        ]
        index = max(indexes, default=-1) + 1
        while True:
            run_dir = self.root / f"run_{index}"
            try:
                run_dir.mkdir()
                break
            except FileExistsError:
                index += 1
        self._run_dir = run_dir
        self._manifest = {
            "schema_version": FORECAST_CACHE_SCHEMA,
            "format": FORECAST_CACHE_FORMAT,
            "status": "running",
            "config": self.config,
            "storage": {
                "shard_rows": self.shard_rows,
                "dtype": "float32",
                "compression": "none",
            },
            "launch": {
                "launch_id": os.environ.get("TIME_LAUNCH_ID"),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            },
            "shards": [],
        }
        started = perf_counter()
        _atomic_json(run_dir / "manifest.json", self._manifest)
        self._timing["manifest_seconds"] += perf_counter() - started

    def _flush(self) -> None:
        if not self._pending:
            return
        self._allocate_run()
        assert self._run_dir is not None and self._manifest is not None
        started = perf_counter()
        directory = self._run_dir / "shards"
        directory.mkdir(exist_ok=True)
        pending_keys = list(self._pending)
        for offset in range(0, len(pending_keys), self.shard_rows):
            keys = pending_keys[offset : offset + self.shard_rows]
            references = np.asarray(keys, dtype=np.int64)
            values = np.stack([self._pending[key] for key in keys]).astype(
                np.float32, copy=False
            )
            shard_number = len(self._manifest["shards"])
            references_path = directory / f"{shard_number:06d}_references.npy"
            values_path = directory / f"{shard_number:06d}_values.npy"
            _atomic_npy(references_path, references)
            _atomic_npy(values_path, values)
            self._manifest["shards"].append(
                {
                    "references": str(references_path.relative_to(self._run_dir)),
                    "values": str(values_path.relative_to(self._run_dir)),
                    "rows": len(keys),
                    "value_shape": list(values.shape[1:]),
                }
            )
            for row, key in enumerate(keys):
                self._index[key] = (values_path, row)
            self._loaded_values[values_path] = values
            self._counts["written_rows"] += len(keys)
        self._pending.clear()
        self._timing["write_seconds"] += perf_counter() - started

    def _values(
        self,
        references: np.ndarray,
        build: Callable[[np.ndarray], np.ndarray],
    ) -> tuple[np.ndarray, float]:
        if self._closed:
            raise RuntimeError("forecast cache is already closed")
        requested = np.asarray(references, dtype=np.int64).reshape(-1, 4)
        if not len(requested):
            raise ValueError("cannot read an empty forecast-cache request")
        keys = [tuple(map(int, row)) for row in requested]
        self._counts["requested_rows"] += len(keys)

        lookup_started = perf_counter()
        missing_positions: list[int] = []
        missing_keys: set[tuple[int, ...]] = set()
        for position, key in enumerate(keys):
            if key in self._index or key in self._pending:
                self._counts["hit_rows"] += 1
                continue
            if key in missing_keys:
                self._counts["deduplicated_rows"] += 1
                continue
            missing_positions.append(position)
            missing_keys.add(key)
        self._timing["lookup_seconds"] += perf_counter() - lookup_started

        compute_seconds = 0.0
        if missing_positions:
            positions = np.asarray(missing_positions, dtype=np.int64)
            timer = EvaluationTimer()
            timer.start()
            values = np.asarray(build(positions), dtype=np.float32)
            compute_seconds = timer.stop()
            if len(values) != len(positions):
                raise ValueError("forecast-cache builder returned the wrong row count")
            for position, value in zip(positions, values, strict=True):
                self._pending[keys[int(position)]] = np.array(value, copy=True)
            self._counts["computed_rows"] += len(positions)
            self._timing["compute_seconds"] += compute_seconds
            if len(self._pending) >= self.shard_rows:
                self._flush()

        read_started = perf_counter()
        result: np.ndarray | None = None
        disk_locations: dict[Path, list[tuple[int, int]]] = {}
        for output_row, key in enumerate(keys):
            pending = self._pending.get(key)
            if pending is not None:
                if result is None:
                    result = np.empty((len(keys), *pending.shape), dtype=np.float32)
                result[output_row] = pending
            else:
                values_path, source_row = self._index[key]
                disk_locations.setdefault(values_path, []).append(
                    (output_row, source_row)
                )
        for values_path, rows in disk_locations.items():
            values = self._loaded_values.get(values_path)
            if values is None:
                values = np.load(values_path, mmap_mode="r", allow_pickle=False)
                self._loaded_values[values_path] = values
            if result is None:
                result = np.empty((len(keys), *values.shape[1:]), dtype=np.float32)
            output_rows, source_rows = zip(*rows, strict=True)
            result[np.asarray(output_rows)] = values[np.asarray(source_rows)]
        self._timing["read_seconds"] += perf_counter() - read_started
        assert result is not None
        return result, compute_seconds

    def forecasts(
        self,
        references: np.ndarray,
        contexts: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        contexts = np.asarray(contexts, dtype=np.float32)
        refs = np.asarray(references, dtype=np.int64).reshape(-1, 3)
        if len(refs) != len(contexts):
            raise ValueError("forecast-cache references and contexts do not align")
        cache_refs = np.column_stack(
            (refs, np.full(len(refs), contexts.shape[-1], dtype=np.int64))
        )

        def build(positions: np.ndarray) -> np.ndarray:
            return np.asarray(
                self.forecaster.forecast(contexts[positions]), dtype=np.float32
            )

        return self._values(cache_refs, build)

    def forecasts_for_references(
        self,
        references: np.ndarray,
        *,
        reader: Any,
    ) -> tuple[np.ndarray, float]:
        refs = np.asarray(references, dtype=np.int64).reshape(-1, 3)
        if not len(refs):
            raise ValueError("cannot forecast an empty reference collection")
        lengths = np.minimum(refs[:, 2], self.prepared.context_length)
        result: np.ndarray | None = None
        compute_seconds = 0.0
        for length in np.unique(lengths):
            positions = np.flatnonzero(lengths == length)
            contexts = reader.read(
                refs[positions], context_length=int(length)
            ).context
            values, seconds = self.forecasts(refs[positions], contexts)
            if result is None:
                result = np.empty((len(refs), *values.shape[1:]), dtype=np.float32)
            result[positions] = values
            compute_seconds += seconds
        assert result is not None
        return result, compute_seconds

    def close(self) -> None:
        if self._closed:
            return
        self._flush()
        self._finish("completed", None)
        self._closed = True

    def _finish(self, status: str, error: BaseException | None) -> None:
        if self._manifest is None or self._run_dir is None:
            return
        self._manifest["status"] = status
        self._manifest["counts"] = dict(self._counts)
        self._manifest["timing_seconds"] = {
            **self._timing,
            "measured_operation_seconds": sum(self._timing.values()),
            "measured_cache_overhead_seconds": sum(
                self._timing[name]
                for name in (
                    "discovery_seconds",
                    "lookup_seconds",
                    "read_seconds",
                    "write_seconds",
                    "manifest_seconds",
                )
            ),
            "final_manifest_write_included": False,
        }
        if error is not None:
            self._manifest["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
        started = perf_counter()
        _atomic_json(self._run_dir / "manifest.json", self._manifest)
        self._timing["manifest_seconds"] += perf_counter() - started


def shared_window_cache_root(
    artifact_root: str | Path,
    prepared: PreparedDataset,
    forecaster: Any,
) -> Path:
    """Return a readable identity root; exact cache fills live in ``run_n``."""

    return (
        Path(artifact_root).expanduser().resolve()
        / "forecast_cache"
        / str(forecaster.model_name)
        / prepared.target_mode
        / str(prepared.config["dataset"])
        / str(prepared.config["term"])
    )
