"""TIME window planning for the source-faithful TS-RAG comparison."""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import datasets
import numpy as np

from timebench.evaluation.adaptation_data import PreparedDataset


TSRAG_PREPARATION_SCHEMA = 1
TSRAG_CONTEXT_LENGTH = 512
TSRAG_NATIVE_HORIZON = 64
TSRAG_DATASTORE_STRIDE = 1


def _target_array(entry: Mapping[str, object]) -> np.ndarray:
    target = np.asarray(entry["target"])
    if target.ndim == 1:
        return target[None, :]
    if target.ndim == 2:
        return target
    raise ValueError(f"TIME targets must have one or two dimensions, got {target.shape}")


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


def _hash_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _references(item: int, channels: int, origins: np.ndarray) -> np.ndarray:
    origins = np.asarray(origins, dtype=np.int64)
    return np.column_stack(
        (
            np.full(len(origins) * channels, int(item), dtype=np.int64),
            np.repeat(np.arange(channels, dtype=np.int64), len(origins)),
            np.tile(origins, channels),
        )
    )


@dataclass(frozen=True)
class TSRAGWindowBatch:
    references: np.ndarray
    context: np.ndarray
    target: np.ndarray


class TSRAGWindowReader:
    """Fetch native 512+64 TS-RAG sequences directly from TIME Arrow rows."""

    def __init__(self, prepared: "TSRAGPreparedDataset", cache_items: int = 2) -> None:
        if int(cache_items) <= 0:
            raise ValueError("cache_items must be positive")
        self.prepared = prepared
        self.cache_items = int(cache_items)
        self._targets: OrderedDict[int, np.ndarray] = OrderedDict()
        self._seasonal_prefixes: OrderedDict[
            int, tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = OrderedDict()

    def _target(self, item: int) -> np.ndarray:
        item = int(item)
        if item not in self._targets:
            self._targets[item] = _target_array(self.prepared.hf_dataset[item])
            self._targets.move_to_end(item)
            while len(self._targets) > self.cache_items:
                expired, _ = self._targets.popitem(last=False)
                self._seasonal_prefixes.pop(expired, None)
        return self._targets[item]

    def _seasonal_prefix(
        self, item: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        item = int(item)
        if item not in self._seasonal_prefixes:
            values = np.asarray(self._target(item), dtype=np.float64)
            period = self.prepared.seasonality
            left = values[:, :-period]
            right = values[:, period:]
            valid = np.isfinite(left) & np.isfinite(right)
            difference = np.where(valid, right - left, 0.0)
            shape = (values.shape[0], difference.shape[1] + 1)
            absolute = np.zeros(shape, dtype=np.float64)
            squared = np.zeros(shape, dtype=np.float64)
            counts = np.zeros(shape, dtype=np.int64)
            absolute[:, 1:] = np.cumsum(np.abs(difference), axis=-1)
            squared[:, 1:] = np.cumsum(np.square(difference), axis=-1)
            counts[:, 1:] = np.cumsum(valid, axis=-1)
            self._seasonal_prefixes[item] = (absolute, squared, counts)
            self._seasonal_prefixes.move_to_end(item)
        return self._seasonal_prefixes[item]

    def seasonal_scales(self, references: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return full-prefix MASE and RMS seasonal scales per TS-RAG row."""

        absolute: list[np.ndarray] = []
        rms: list[np.ndarray] = []
        for item, channel, origin in np.asarray(references, dtype=np.int64).reshape(-1, 3):
            absolute_prefix, squared_prefix, count_prefix = self._seasonal_prefix(int(item))
            position = max(int(origin) - self.prepared.seasonality, 0)
            selected = slice(int(channel), int(channel) + 1)
            count = count_prefix[selected, position]
            absolute_mean = np.divide(
                absolute_prefix[selected, position],
                count,
                out=np.full(count.shape, np.nan, dtype=np.float64),
                where=count > 0,
            )
            squared_mean = np.divide(
                squared_prefix[selected, position],
                count,
                out=np.full(count.shape, np.nan, dtype=np.float64),
                where=count > 0,
            )
            absolute.append(np.where(absolute_mean > 0, absolute_mean, np.nan))
            rms.append(np.where(squared_mean > 0, np.sqrt(squared_mean), np.nan))
        return np.stack(absolute), np.stack(rms)

    def seasonal_naive_forecast(self, references: np.ndarray) -> np.ndarray:
        """Return deterministic seasonal-naive forecasts on identical test rows."""

        forecasts: list[np.ndarray] = []
        period = self.prepared.seasonality
        horizon = self.prepared.prediction_length
        repeats = int(np.ceil(horizon / period))
        for item, channel, origin in np.asarray(references, dtype=np.int64).reshape(-1, 3):
            values = self._target(int(item))[int(channel) : int(channel) + 1]
            season = values[:, int(origin) - period : int(origin)]
            if season.shape[-1] != period:
                raise ValueError(f"reference {(item, channel, origin)} lacks one season")
            forecasts.append(np.tile(season, (1, repeats))[..., :horizon])
        return np.stack(forecasts)

    def read(
        self,
        references: np.ndarray,
        *,
        target_length: int,
    ) -> TSRAGWindowBatch:
        refs = np.asarray(references, dtype=np.int64).reshape(-1, 3)
        contexts: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for item, channel, origin in refs:
            values = self._target(int(item))[int(channel) : int(channel) + 1]
            context = values[
                :,
                int(origin) - TSRAG_CONTEXT_LENGTH : int(origin),
            ]
            target = values[:, int(origin) : int(origin) + int(target_length)]
            if context.shape[-1] != TSRAG_CONTEXT_LENGTH:
                raise ValueError(
                    f"TS-RAG reference lacks 512 context values: "
                    f"{(int(item), int(channel), int(origin))}"
                )
            if target.shape[-1] != int(target_length):
                raise ValueError(
                    f"TS-RAG reference lacks {target_length} future values: "
                    f"{(int(item), int(channel), int(origin))}"
                )
            contexts.append(np.asarray(context, dtype=np.float32))
            targets.append(np.asarray(target, dtype=np.float32))
        return TSRAGWindowBatch(
            references=refs,
            context=np.stack(contexts),
            target=np.stack(targets),
        )


class TSRAGPreparedDataset:
    """Open TS-RAG indices while retaining the owning TIME Arrow source."""

    def __init__(self, path: str | Path) -> None:
        manifest_path = Path(path).expanduser().resolve()
        if manifest_path.is_dir():
            manifest_path = manifest_path / "manifest.json"
        self.root = manifest_path.parent
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != TSRAG_PREPARATION_SCHEMA:
            raise ValueError("unsupported TS-RAG preparation schema")
        if self.manifest.get("format") != "adaptime_tsrag_windows":
            raise ValueError("not an Adaptime TS-RAG preparation artifact")
        self.config = dict(self.manifest["config"])
        self._hf_dataset: datasets.Dataset | None = None

    @property
    def hf_dataset(self) -> datasets.Dataset:
        if self._hf_dataset is None:
            source = datasets.load_from_disk(self.manifest["source_path"])
            if str(source._fingerprint) != self.manifest["dataset_fingerprint"]:
                raise ValueError("TS-RAG indices do not match the current Arrow dataset")
            self._hf_dataset = source
        return self._hf_dataset

    @property
    def signature(self) -> str:
        return str(self.manifest["signature"])

    @property
    def prediction_length(self) -> int:
        return int(self.config["prediction_length"])

    @property
    def seasonality(self) -> int:
        return int(self.config["seasonality"])

    def indices(self, split: str) -> np.ndarray:
        if split not in {"datastore", "test"}:
            raise ValueError(f"unknown TS-RAG split {split!r}")
        return np.load(self.root / self.manifest["arrays"][split], mmap_mode="r")

    def reader(self, cache_items: int = 2) -> TSRAGWindowReader:
        return TSRAGWindowReader(self, cache_items=cache_items)


def prepare_tsrag_dataset(
    ridge_prepared_path: str | Path,
    output_dir: str | Path,
) -> Path:
    """Use ridge's raw date budget and test rows with native TS-RAG windows."""

    ridge = PreparedDataset(ridge_prepared_path)
    if ridge.target_mode != "univariate":
        raise ValueError("TS-RAG comparison requires the univariate ridge protocol")
    ridge_test = np.asarray(ridge.indices("test"), dtype=np.int64)
    if np.any(ridge_test[:, 2] < TSRAG_CONTEXT_LENGTH):
        raise ValueError("an official TIME test origin cannot supply TS-RAG's L=512")

    config = dict(ridge.config)
    source_path = Path(ridge.manifest["source_path"]).expanduser().resolve()
    source = ridge.hf_dataset
    datastore_parts: list[np.ndarray] = []
    accessible_intervals: list[dict[str, int]] = []
    datastore_length = config.get("datastore_length")
    for item in range(len(source)):
        target = _target_array(source[item])
        channels, length = map(int, target.shape)
        train_stop = (
            length
            - int(config["test_length"])
            - int(config["adaptation_validation_length"])
            - int(config["adaptation_train_length"])
        )
        datastore_start = (
            0 if datastore_length is None else train_stop - int(datastore_length)
        )
        first_origin = max(datastore_start, TSRAG_CONTEXT_LENGTH)
        last_origin = train_stop - TSRAG_NATIVE_HORIZON
        if datastore_start < 0 or first_origin > last_origin:
            raise ValueError(
                f"TIME item {item} cannot provide a TS-RAG datastore within ridge's date budget"
            )
        origins = np.arange(
            first_origin,
            last_origin + 1,
            TSRAG_DATASTORE_STRIDE,
            dtype=np.int64,
        )
        datastore_parts.append(_references(item, channels, origins))
        accessible_intervals.append(
            {
                "item": item,
                "candidate_origin_start": int(first_origin),
                "candidate_origin_stop": int(train_stop),
                "raw_start": int(first_origin - TSRAG_CONTEXT_LENGTH),
                "raw_stop": int(train_stop),
                "raw_points": int(
                    train_stop - first_origin + TSRAG_CONTEXT_LENGTH
                ),
            }
        )

    datastore = np.concatenate(datastore_parts)
    scientific = {
        "dataset": config["dataset"],
        "term": config["term"],
        "target_mode": "univariate",
        "context_length": TSRAG_CONTEXT_LENGTH,
        "prediction_length": int(config["prediction_length"]),
        "seasonality": int(config["seasonality"]),
        "native_prediction_length": TSRAG_NATIVE_HORIZON,
        "datastore_stride": TSRAG_DATASTORE_STRIDE,
        "datastore_scope": "same_series",
        "test_support": "ridge_official_time_test_references",
        "ridge_matched_accessible_date_intervals": accessible_intervals,
    }
    signature_payload = {
        "schema_version": TSRAG_PREPARATION_SCHEMA,
        "ridge_prepared_signature": ridge.signature,
        "dataset_fingerprint": str(source._fingerprint),
        "config": scientific,
        "test_reference_sha256": _hash_array(ridge_test),
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    root = Path(output_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("signature") == signature and all(
            (root / relative).is_file()
            for relative in dict(existing.get("arrays", {})).values()
        ):
            return manifest_path
        raise FileExistsError(f"TS-RAG preparation already differs: {root}")

    arrays = {"datastore": "indices/datastore.npy", "test": "indices/test.npy"}
    _atomic_npy(root / arrays["datastore"], datastore)
    _atomic_npy(root / arrays["test"], ridge_test)
    _atomic_json(
        manifest_path,
        {
            **signature_payload,
            "format": "adaptime_tsrag_windows",
            "signature": signature,
            "source_path": str(source_path),
            "arrays": arrays,
            "counts": {"datastore": int(len(datastore)), "test": int(len(ridge_test))},
        },
    )
    return manifest_path
