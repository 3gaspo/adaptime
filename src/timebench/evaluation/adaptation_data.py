"""Arrow-backed window planning for leakage-free Adaptime experiments."""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import datasets
import numpy as np
import pandas as pd


PREPARATION_SCHEMA = 1
QUERY_SPLITS = ("adaptation_train", "adaptation_validation", "test")
ALL_SPLITS = ("datastore", *QUERY_SPLITS)


def _canonical_hash(value: Mapping[str, object]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
    os.replace(temporary, path)


@dataclass(frozen=True)
class PreparationConfig:
    """Scientific split and window contract for one TIME dataset/term."""

    dataset: str
    term: str
    context_length: int
    prediction_length: int
    test_length: int
    adaptation_train_length: int
    adaptation_validation_length: int
    target_mode: str = "univariate"
    adaptation_stride: int | None = None
    retrieval_period: int = 1
    datastore_stride: int = 1
    datastore_length: int | None = None

    @property
    def query_stride(self) -> int:
        return int(self.adaptation_stride or self.prediction_length)

    def validate(self) -> None:
        positive = {
            "context_length": self.context_length,
            "prediction_length": self.prediction_length,
            "test_length": self.test_length,
            "adaptation_train_length": self.adaptation_train_length,
            "adaptation_validation_length": self.adaptation_validation_length,
            "retrieval_period": self.retrieval_period,
            "datastore_stride": self.datastore_stride,
            "adaptation_stride": self.query_stride,
        }
        invalid = [name for name, value in positive.items() if int(value) <= 0]
        if invalid:
            raise ValueError(f"positive preparation settings required: {', '.join(invalid)}")
        if self.target_mode not in {"univariate", "multivariate"}:
            raise ValueError("target_mode must be univariate or multivariate")
        if self.test_length < self.prediction_length:
            raise ValueError("test_length must contain at least one complete horizon")
        if self.adaptation_train_length < self.prediction_length:
            raise ValueError("adaptation_train_length must contain a complete horizon")
        if self.adaptation_validation_length < self.prediction_length:
            raise ValueError("adaptation_validation_length must contain a complete horizon")
        if self.datastore_length is not None and int(self.datastore_length) <= 0:
            raise ValueError("datastore_length must be positive when supplied")
        if self.datastore_stride % self.retrieval_period:
            raise ValueError("datastore_stride must be a multiple of retrieval_period")


@dataclass(frozen=True)
class WindowBatch:
    """One bounded batch with canonical ``(batch, channel, time)`` arrays."""

    references: np.ndarray
    context: np.ndarray
    target: np.ndarray


def _target_array(entry: Mapping[str, object]) -> np.ndarray:
    target = np.asarray(entry["target"])
    if target.ndim == 1:
        return target[None, :]
    if target.ndim == 2:
        return target
    raise ValueError(f"TIME targets must have one or two dimensions, got {target.shape}")


def _intervals(length: int, config: PreparationConfig) -> dict[str, tuple[int, int]]:
    test_start = length - config.test_length
    validation_start = test_start - config.adaptation_validation_length
    train_start = validation_start - config.adaptation_train_length
    datastore_start = (
        0
        if config.datastore_length is None
        else train_start - int(config.datastore_length)
    )
    if datastore_start < 0 or train_start <= 0:
        required = (
            config.test_length
            + config.adaptation_validation_length
            + config.adaptation_train_length
            + int(config.datastore_length or 0)
        )
        raise ValueError(
            f"series length {length} cannot provide the requested chronological "
            f"intervals (at least {required} values required)"
        )
    return {
        "datastore": (datastore_start, train_start),
        "adaptation_train": (train_start, validation_start),
        "adaptation_validation": (validation_start, test_start),
        "test": (test_start, length),
    }


def _query_origins(
    interval: tuple[int, int],
    *,
    context_length: int,
    horizon: int,
    stride: int,
) -> np.ndarray:
    start, stop = interval
    first = max(int(start), int(context_length))
    last = int(stop) - int(horizon)
    if first > last:
        return np.empty(0, dtype=np.int64)
    return np.arange(first, last + 1, int(stride), dtype=np.int64)


def _official_test_origins(
    interval: tuple[int, int],
    *,
    context_length: int,
    horizon: int,
) -> np.ndarray:
    start, stop = interval
    windows = (int(stop) - int(start)) // int(horizon)
    if int(start) < int(context_length):
        raise ValueError(
            "context_length reaches before the series start at the first official TIME test origin"
        )
    return int(start) + np.arange(windows, dtype=np.int64) * int(horizon)


def _phase_origins(
    interval: tuple[int, int],
    *,
    context_length: int,
    horizon: int,
    stride: int,
    phases: Sequence[int],
) -> np.ndarray:
    start, stop = interval
    first = max(int(start), int(context_length))
    last = int(stop) - int(horizon)
    origins: list[np.ndarray] = []
    for phase in phases:
        aligned = first + (int(phase) - first) % int(stride)
        if aligned <= last:
            origins.append(np.arange(aligned, last + 1, int(stride), dtype=np.int64))
    if not origins:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(origins))


def _append_references(
    destination: list[np.ndarray],
    calendar_ticks: list[np.ndarray],
    *,
    item: int,
    channels: int,
    origins: np.ndarray,
    target_mode: str,
    start_tick: int,
) -> None:
    origins = np.asarray(origins, dtype=np.int64)
    if target_mode == "univariate":
        repeated_origins = np.tile(origins, int(channels))
        selected_channels = np.repeat(
            np.arange(int(channels), dtype=np.int64), len(origins)
        )
    else:
        repeated_origins = origins
        selected_channels = np.full(len(origins), -1, dtype=np.int64)
    destination.append(
        np.column_stack(
            (
                np.full(len(repeated_origins), int(item), dtype=np.int64),
                selected_channels,
                repeated_origins,
            )
        )
    )
    calendar_ticks.append(int(start_tick) + repeated_origins)


def prepare_adaptation_dataset(
    hf_dataset: datasets.Dataset,
    config: PreparationConfig,
    output_dir: str | Path,
    *,
    source_path: str | Path,
) -> Path:
    """Write compact indices while leaving every time-series value in Arrow.

    The official test grid always uses TIME's non-overlapping horizon spacing.
    The datastore contains only phases needed by at least one train, validation,
    or test query, so a large period-multiple stride reduces both representation
    work and search size without materializing query-specific datastores.
    """

    config.validate()
    if len(hf_dataset) == 0:
        raise ValueError("cannot prepare an empty TIME dataset")
    root = Path(output_dir).expanduser().resolve()
    scientific = asdict(config)
    scientific["adaptation_stride"] = config.query_stride
    signature = _canonical_hash(
        {
            "schema_version": PREPARATION_SCHEMA,
            "dataset_fingerprint": str(hf_dataset._fingerprint),
            "config": scientific,
        }
    )
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("signature") == signature and all(
            (root / relative).is_file()
            for relative in dict(existing.get("arrays", {})).values()
        ):
            return manifest_path
        raise FileExistsError(f"prepared directory already contains a different dataset: {root}")

    target_shapes: list[tuple[int, int]] = []
    interval_rows: list[dict[str, tuple[int, int]]] = []
    references: dict[str, list[np.ndarray]] = {split: [] for split in ALL_SPLITS}
    calendar_ticks: dict[str, list[np.ndarray]] = {split: [] for split in ALL_SPLITS}
    start_ticks: list[int] = []
    query_period_residues: set[int] = set()
    multivariate_channels: int | None = None

    for item in range(len(hf_dataset)):
        entry = hf_dataset[item]
        target = _target_array(entry)
        channels, length = map(int, target.shape)
        start_tick = int(pd.Period(pd.Timestamp(entry["start"]), freq=entry["freq"]).ordinal)
        start_ticks.append(start_tick)
        if config.target_mode == "multivariate":
            if multivariate_channels is None:
                multivariate_channels = channels
            elif channels != multivariate_channels:
                raise ValueError("multivariate mode requires the same target dimension in every item")
        target_shapes.append((channels, length))
        intervals = _intervals(length, config)
        interval_rows.append(intervals)
        split_origins = {
            "adaptation_train": _query_origins(
                intervals["adaptation_train"],
                context_length=config.context_length,
                horizon=config.prediction_length,
                stride=config.query_stride,
            ),
            "adaptation_validation": _query_origins(
                intervals["adaptation_validation"],
                context_length=config.context_length,
                horizon=config.prediction_length,
                stride=config.query_stride,
            ),
            "test": _official_test_origins(
                intervals["test"],
                context_length=config.context_length,
                horizon=config.prediction_length,
            ),
        }
        for split, origins in split_origins.items():
            if len(origins) == 0:
                raise ValueError(f"item {item} has no complete {split} windows")
            query_period_residues.update(
                int((start_tick + value) % config.retrieval_period) for value in origins
            )
            _append_references(
                references[split],
                calendar_ticks[split],
                item=item,
                channels=channels,
                origins=origins,
                target_mode=config.target_mode,
                start_tick=start_tick,
            )

    period_residues = sorted(query_period_residues)
    datastore_end_ticks: list[int] = []
    for item, ((channels, _), intervals, start_tick) in enumerate(
        zip(target_shapes, interval_rows, start_ticks)
    ):
        last_origin = int(intervals["datastore"][1]) - config.prediction_length
        datastore_end_tick = int(start_tick + last_origin)
        datastore_end_ticks.append(datastore_end_tick)
        phases = [
            int(
                (
                    datastore_end_tick
                    - (datastore_end_tick - residue) % config.retrieval_period
                    - start_tick
                )
                % config.datastore_stride
            )
            for residue in period_residues
        ]
        origins = _phase_origins(
            intervals["datastore"],
            context_length=config.context_length,
            horizon=config.prediction_length,
            stride=config.datastore_stride,
            phases=phases,
        )
        if len(origins) == 0:
            raise ValueError(f"item {item} has no eligible datastore windows")
        _append_references(
            references["datastore"],
            calendar_ticks["datastore"],
            item=item,
            channels=channels,
            origins=origins,
            target_mode=config.target_mode,
            start_tick=start_tick,
        )

    arrays: dict[str, str] = {}
    counts: dict[str, int] = {}
    for split in ALL_SPLITS:
        values = (
            np.concatenate(references[split], axis=0)
            if references[split]
            else np.empty((0, 3), dtype=np.int64)
        )
        ticks = (
            np.concatenate(calendar_ticks[split])
            if calendar_ticks[split]
            else np.empty(0, dtype=np.int64)
        )
        relative = f"indices/{split}.npy"
        calendar_relative = f"indices/{split}_calendar_tick.npy"
        _atomic_npy(root / relative, values)
        _atomic_npy(root / calendar_relative, ticks)
        arrays[split] = relative
        arrays[f"{split}_calendar_tick"] = calendar_relative
        counts[split] = int(len(values))
    datastore_end_relative = "indices/datastore_end_tick_by_item.npy"
    _atomic_npy(
        root / datastore_end_relative,
        np.asarray(datastore_end_ticks, dtype=np.int64),
    )
    arrays["datastore_end_tick_by_item"] = datastore_end_relative

    manifest: dict[str, object] = {
        "schema_version": PREPARATION_SCHEMA,
        "format": "adaptime_prepared_windows",
        "signature": signature,
        "dataset_fingerprint": str(hf_dataset._fingerprint),
        "source_path": str(Path(source_path).expanduser().resolve()),
        "config": scientific,
        "reference_columns": ["item", "channel", "origin"],
        "calendar_tick": "global pandas Period ordinal at the window target origin",
        "channel_convention": "-1 denotes the complete multivariate target",
        "interval_convention": "target start inclusive, target stop exclusive",
        "query_period_residues": period_residues,
        "arrays": arrays,
        "counts": counts,
    }
    _atomic_json(manifest_path, manifest)
    return manifest_path


class WindowReader:
    """Read arbitrary prepared windows through a bounded Arrow-row cache."""

    def __init__(self, prepared: "PreparedDataset", cache_items: int = 2) -> None:
        if int(cache_items) <= 0:
            raise ValueError("cache_items must be positive")
        self.prepared = prepared
        self.cache_items = int(cache_items)
        self._targets: OrderedDict[int, np.ndarray] = OrderedDict()

    def _target(self, item: int) -> np.ndarray:
        item = int(item)
        if item not in self._targets:
            self._targets[item] = _target_array(self.prepared.hf_dataset[item])
            self._targets.move_to_end(item)
            while len(self._targets) > self.cache_items:
                self._targets.popitem(last=False)
        return self._targets[item]

    def read(self, references: np.ndarray) -> WindowBatch:
        refs = np.asarray(references, dtype=np.int64).reshape(-1, 3)
        context_length = self.prepared.context_length
        horizon = self.prepared.prediction_length
        contexts: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for item, channel, origin in refs:
            values = self._target(int(item))
            selected = values if int(channel) == -1 else values[int(channel) : int(channel) + 1]
            context = selected[:, int(origin) - context_length : int(origin)]
            target = selected[:, int(origin) : int(origin) + horizon]
            if context.shape[-1] != context_length or target.shape[-1] != horizon:
                raise ValueError(f"invalid prepared reference {(item, channel, origin)}")
            contexts.append(np.asarray(context))
            targets.append(np.asarray(target))
        return WindowBatch(
            references=refs,
            context=np.stack(contexts),
            target=np.stack(targets),
        )


class PreparedDataset:
    """Open one prepared manifest and lazily access its TIME Arrow source."""

    def __init__(self, path: str | Path) -> None:
        manifest_path = Path(path).expanduser().resolve()
        if manifest_path.is_dir():
            manifest_path = manifest_path / "manifest.json"
        self.root = manifest_path.parent
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != PREPARATION_SCHEMA:
            raise ValueError("unsupported Adaptime preparation schema")
        self.config = dict(self.manifest["config"])
        self._hf_dataset: datasets.Dataset | None = None

    @property
    def hf_dataset(self) -> datasets.Dataset:
        """Load Arrow only for stages that actually read source windows."""

        if self._hf_dataset is None:
            source = datasets.load_from_disk(self.manifest["source_path"])
            if str(source._fingerprint) != self.manifest["dataset_fingerprint"]:
                raise ValueError("prepared indices do not match the current Arrow dataset")
            self._hf_dataset = source
        return self._hf_dataset

    @property
    def context_length(self) -> int:
        return int(self.config["context_length"])

    @property
    def prediction_length(self) -> int:
        return int(self.config["prediction_length"])

    @property
    def target_mode(self) -> str:
        return str(self.config["target_mode"])

    @property
    def signature(self) -> str:
        return str(self.manifest["signature"])

    def indices(self, split: str) -> np.ndarray:
        if split not in ALL_SPLITS:
            raise ValueError(f"unknown prepared split {split!r}")
        return np.load(self.root / self.manifest["arrays"][split], mmap_mode="r")

    def calendar_ticks(self, split: str) -> np.ndarray:
        if split not in ALL_SPLITS:
            raise ValueError(f"unknown prepared split {split!r}")
        key = f"{split}_calendar_tick"
        return np.load(self.root / self.manifest["arrays"][key], mmap_mode="r")

    @property
    def datastore_end_ticks_by_item(self) -> np.ndarray:
        path = self.manifest["arrays"]["datastore_end_tick_by_item"]
        return np.load(self.root / path, mmap_mode="r")

    def reader(self, cache_items: int = 2) -> WindowReader:
        return WindowReader(self, cache_items=cache_items)

    def batches(
        self,
        split: str,
        batch_size: int,
        *,
        cache_items: int = 2,
    ) -> Iterator[WindowBatch]:
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        indices = self.indices(split)
        reader = self.reader(cache_items=cache_items)
        for start in range(0, len(indices), int(batch_size)):
            yield reader.read(indices[start : start + int(batch_size)])
