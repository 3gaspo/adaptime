"""TS-RAG's native window view over the shared Adaptime datastore."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from timebench.evaluation.adaptation_data import PreparedDataset


TSRAG_CONTEXT_LENGTH = 512
TSRAG_NATIVE_HORIZON = 64


def _target_array(entry: Mapping[str, object]) -> np.ndarray:
    target = np.asarray(entry["target"])
    if target.ndim == 1:
        return target[None, :]
    if target.ndim == 2:
        return target
    raise ValueError(f"TIME targets must have one or two dimensions, got {target.shape}")


@dataclass(frozen=True)
class TSRAGWindowBatch:
    references: np.ndarray
    context: np.ndarray
    target: np.ndarray


class TSRAGWindowReader:
    """Fetch native 512+64 TS-RAG sequences from shared TIME references."""

    def __init__(self, prepared: "TSRAGPreparedDataset", cache_items: int = 2) -> None:
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
            if int(channel) < 0:
                raise ValueError("TS-RAG requires univariate shared references")
            values = self._target(int(item))[int(channel) : int(channel) + 1]
            context = values[:, int(origin) - TSRAG_CONTEXT_LENGTH : int(origin)]
            target = values[:, int(origin) : int(origin) + int(target_length)]
            if context.shape[-1] != TSRAG_CONTEXT_LENGTH:
                raise ValueError(
                    f"TS-RAG reference lacks {TSRAG_CONTEXT_LENGTH} context values: "
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
    """Read TS-RAG windows from the method-neutral Adaptime data artifact."""

    def __init__(self, path: str | Path) -> None:
        self.shared = PreparedDataset(path)
        if self.shared.target_mode != "univariate":
            raise ValueError("TS-RAG requires a shared univariate Adaptime datastore")
        if self.shared.context_length < TSRAG_CONTEXT_LENGTH:
            raise ValueError(
                "the shared datastore context must cover TS-RAG's native context"
            )
        datastore_horizon = int(
            self.shared.config.get("datastore_prediction_length")
            or self.shared.prediction_length
        )
        if datastore_horizon < TSRAG_NATIVE_HORIZON:
            raise ValueError(
                "the shared datastore must retain TS-RAG's 64-step neighbor future"
            )

    @property
    def hf_dataset(self):
        return self.shared.hf_dataset

    @property
    def signature(self) -> str:
        return self.shared.signature

    @property
    def prediction_length(self) -> int:
        return self.shared.prediction_length

    @property
    def seasonality(self) -> int:
        return self.shared.seasonality

    def indices(self, split: str) -> np.ndarray:
        if split not in {"datastore", "test"}:
            raise ValueError(f"unknown TS-RAG split {split!r}")
        return self.shared.indices(split)

    def reader(self, cache_items: int = 2) -> TSRAGWindowReader:
        return TSRAGWindowReader(self, cache_items=cache_items)
