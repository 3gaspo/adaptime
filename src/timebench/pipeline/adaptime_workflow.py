"""TIME-wide orchestration for Adaptime extraction, fitting, and testing."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import datasets
import numpy as np
from gluonts.time_feature import get_seasonality, norm_freq_str
from pandas.tseries.frequencies import to_offset

from timebench.evaluation.adaptation_data import PreparationConfig, prepare_adaptation_dataset
from timebench.evaluation.data import (
    M4_PRED_LENGTH_MAP,
    PRED_LENGTH_MAP,
    Term,
    get_dataset_settings,
    load_dataset_config,
)
from timebench.evaluation.utils import get_available_terms
from timebench.model_loading import load_adaptime_forecaster
from timebench.paths import dataset_storage_root, outputs_root
from timebench.pipeline.adaptime_extraction import (
    ExtractionConfig,
    extract_adaptation_features,
)
from timebench.pipeline.adaptime_testing import (
    AdaptimeTestingConfig,
    METHODS,
    METRICS,
    evaluate_frozen_adaptation,
)
from timebench.pipeline.adaptime_training import RidgeTrainingConfig, fit_full_ridge


@dataclass(frozen=True)
class AdaptimeWorkflowConfig:
    model: str = "chronos2"
    target_mode: str = "univariate"
    max_context_length: int = 2048
    adaptation_train_length: int | None = None
    adaptation_validation_length: int | None = None
    adaptation_stride: int | None = None
    retrieval_period: int | None = None
    datastore_stride_multiple: int = 1
    datastore_length: int | None = None
    representation: str = "instance"
    distance_metric: str = "euclidean"
    retrieval_scope: str = "all"
    max_k: int = 15
    k_values: tuple[int, ...] = (1, 5, 10, 15)
    alpha_values: tuple[float, ...] = (1e-3, 1e-2, 1e-1)
    model_batch_size: int = 64
    query_block_size: int = 256
    datastore_block_size: int = 4096
    arrow_cache_items: int = 2
    ridge_chunk_size: int = 1024
    seed: int = 1
    model_path: Path | None = None
    weights_id: str | None = None
    device: str = "cuda"

    def validate(self) -> None:
        if self.target_mode != "univariate":
            raise ValueError("the current full_ridge_shared workflow is univariate")
        if int(self.max_context_length) <= 0:
            raise ValueError("max_context_length must be positive")
        if int(self.datastore_stride_multiple) <= 0:
            raise ValueError("datastore_stride_multiple must be positive")
        if max(self.k_values) > int(self.max_k):
            raise ValueError("max_k must cover every selected K")


@dataclass(frozen=True)
class AdaptimeTask:
    dataset: str
    term: str
    prepared: Path
    extraction: Path
    model: Path
    comparison: Path


def _canonical_hash(value: dict[str, object]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _prediction_length(dataset: str, term: str, configured: int | None, freq: str) -> int:
    if configured is not None:
        return int(configured)
    normalized = norm_freq_str(to_offset(freq).name)
    base = M4_PRED_LENGTH_MAP[normalized] if "m4" in dataset else PRED_LENGTH_MAP[normalized]
    return int(base * Term(term).multiplier)


def _selected_datasets(
    dataset_config: dict[str, object], selected: Iterable[str]
) -> list[str]:
    values = list(selected)
    names = list(dataset_config.get("datasets", {})) if values == ["all_datasets"] else values
    if not names:
        raise ValueError("no TIME datasets selected")
    return names


def _selected_terms(
    dataset: str,
    dataset_config: dict[str, object],
    selected: Iterable[str] | None,
) -> list[str]:
    available = list(get_available_terms(dataset, dataset_config))
    terms = available if selected is None else [term for term in selected if term in available]
    if not terms:
        raise ValueError(f"no requested terms are configured for {dataset!r}")
    return terms


def _task(
    output_root: Path,
    workflow: AdaptimeWorkflowConfig,
    dataset: str,
    term: str,
) -> AdaptimeTask:
    suffix = Path(workflow.target_mode) / dataset / term
    model_suffix = Path(workflow.model) / suffix
    return AdaptimeTask(
        dataset=dataset,
        term=term,
        prepared=output_root / "prepared" / suffix,
        extraction=output_root / "extraction" / model_suffix,
        model=output_root / "training" / model_suffix,
        comparison=output_root / "comparison" / model_suffix,
    )


def workflow_tasks(
    dataset_config: dict[str, object],
    datasets_selected: Iterable[str],
    terms_selected: Iterable[str] | None,
    output_root: Path,
    workflow: AdaptimeWorkflowConfig,
) -> list[AdaptimeTask]:
    return [
        _task(output_root, workflow, dataset, term)
        for dataset in _selected_datasets(dataset_config, datasets_selected)
        for term in _selected_terms(dataset, dataset_config, terms_selected)
    ]


def aggregate_time_comparison(
    tasks: list[AdaptimeTask],
    output_dir: Path,
) -> Path:
    """Aggregate equal-user task metrics, then terms, then TIME datasets."""

    rows: list[dict[str, object]] = []
    result_signatures: list[str] = []
    for task in tasks:
        manifest_path = task.comparison / "result_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed":
            raise ValueError(f"incomplete Adaptime result: {manifest_path}")
        summary = json.loads(
            (task.comparison / manifest["files"]["comparison_summary"]).read_text(
                encoding="utf-8"
            )
        )
        selected = dict(manifest["selected"])
        row: dict[str, object] = {
            "dataset": task.dataset,
            "term": task.term,
            "selected_k": int(selected["k"]),
            "selected_alpha": float(selected["alpha"]),
            "validation_nmse": float(selected["validation_nmse"]),
        }
        for method in METHODS:
            method_summary = dict(summary["methods"][method])
            for metric in METRICS:
                metric_summary = dict(method_summary[metric])
                row[f"{method}_{metric}_equal_user"] = float(
                    metric_summary["equal_user_mean"]
                )
                row[f"{method}_{metric}_equal_window"] = float(
                    metric_summary["equal_window_mean"]
                )
        for method in ("covariate", "adaptime"):
            row[f"{method}_mse_win_rate_vs_vanilla"] = float(
                summary["mse_win_rate_vs_vanilla"][method]
            )
        rows.append(row)
        result_signatures.append(str(manifest["signature"]))

    identity = {
        "schema_version": 1,
        "format": "adaptime_time_aggregate",
        "protocol": "equal_user_then_equal_term_then_equal_dataset",
        "result_signatures": result_signatures,
    }
    signature = _canonical_hash(identity)
    root = output_dir.expanduser().resolve()
    manifest_path = root / "time_summary_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("signature") == signature
            and existing.get("status") == "completed"
            and (root / existing["files"]["summary"]).is_file()
            and (root / existing["files"]["tasks"]).is_file()
        ):
            return manifest_path
        raise FileExistsError(f"aggregate directory contains a different run: {root}")

    dataset_rows: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        dataset_rows.setdefault(str(row["dataset"]), []).append(row)
    methods: dict[str, object] = {}
    for method in METHODS:
        metrics: dict[str, object] = {}
        for metric in METRICS:
            field = f"{method}_{metric}_equal_user"
            per_dataset = np.asarray(
                [
                    np.mean([float(row[field]) for row in dataset_rows[dataset]])
                    for dataset in sorted(dataset_rows)
                ],
                dtype=np.float64,
            )
            metrics[metric] = {
                "equal_dataset_mean": float(per_dataset.mean()),
                "equal_dataset_std": float(per_dataset.std()),
            }
        methods[method] = metrics
    wins: dict[str, object] = {}
    for method in ("covariate", "adaptime"):
        field = f"{method}_mse_win_rate_vs_vanilla"
        per_dataset = np.asarray(
            [
                np.mean([float(row[field]) for row in dataset_rows[dataset]])
                for dataset in sorted(dataset_rows)
            ],
            dtype=np.float64,
        )
        wins[method] = {
            "equal_dataset_mean": float(per_dataset.mean()),
            "equal_dataset_std": float(per_dataset.std()),
        }

    _atomic_csv(root / "time_tasks.csv", rows)
    _atomic_json(
        root / "time_summary.json",
        {
            "datasets": len(dataset_rows),
            "tasks": len(rows),
            "methods": methods,
            "mse_win_rate_vs_vanilla": wins,
        },
    )
    _atomic_json(
        manifest_path,
        {
            **identity,
            "signature": signature,
            "status": "completed",
            "files": {"summary": "time_summary.json", "tasks": "time_tasks.csv"},
        },
    )
    return manifest_path


def _positive_interval_length(
    explicit: int | None,
    settings: dict[str, object],
    name: str,
) -> int:
    value = explicit if explicit is not None else settings.get("val_length")
    if value is None or int(value) <= 0:
        raise ValueError(f"{name} needs an explicit positive length")
    return int(value)


def _prepare_and_extract_dataset(
    dataset_name: str,
    tasks: list[AdaptimeTask],
    dataset_config: dict[str, object],
    workflow: AdaptimeWorkflowConfig,
) -> None:
    source_path = dataset_storage_root() / dataset_name
    hf_dataset = datasets.load_from_disk(str(source_path))
    if len(hf_dataset) == 0:
        raise ValueError(f"empty TIME dataset: {dataset_name}")
    first = hf_dataset[0]
    freq = str(first["freq"])
    period = int(workflow.retrieval_period or get_seasonality(freq))
    shapes = [np.asarray(hf_dataset[index]["target"]).shape for index in range(len(hf_dataset))]
    min_length = min(int(shape[-1]) for shape in shapes)
    entities = sum(int(shape[0]) if len(shape) > 1 else 1 for shape in shapes)
    if workflow.retrieval_scope == "same_series":
        eligible_entities = 1
    elif workflow.retrieval_scope == "other_series":
        eligible_entities = entities - 1
    else:
        eligible_entities = entities
    if eligible_entities <= 0:
        raise ValueError(f"{dataset_name} has no eligible retrieval series")

    for task in tasks:
        settings = get_dataset_settings(dataset_name, task.term, dataset_config)
        horizon = _prediction_length(
            dataset_name, task.term, settings.get("prediction_length"), freq
        )
        train_length = _positive_interval_length(
            workflow.adaptation_train_length,
            settings,
            f"{dataset_name}/{task.term} adaptation train",
        )
        validation_length = _positive_interval_length(
            workflow.adaptation_validation_length,
            settings,
            f"{dataset_name}/{task.term} adaptation validation",
        )
        datastore_stop = (
            min_length - int(settings["test_length"]) - validation_length - train_length
        )
        datastore_stride = period * int(workflow.datastore_stride_multiple)
        origins_per_entity = math.ceil(int(workflow.max_k) / eligible_entities)
        context_length = min(
            int(workflow.max_context_length),
            datastore_stop
            - horizon
            - (period - 1)
            - (origins_per_entity - 1) * datastore_stride,
        )
        if context_length <= 0:
            raise ValueError(
                f"{dataset_name}/{task.term} has no room for a datastore window before adaptation"
            )
        prepared_manifest = prepare_adaptation_dataset(
            hf_dataset,
            PreparationConfig(
                dataset=dataset_name,
                term=task.term,
                context_length=context_length,
                prediction_length=horizon,
                test_length=int(settings["test_length"]),
                adaptation_train_length=train_length,
                adaptation_validation_length=validation_length,
                target_mode=workflow.target_mode,
                adaptation_stride=workflow.adaptation_stride,
                retrieval_period=period,
                datastore_stride=datastore_stride,
                datastore_length=workflow.datastore_length,
            ),
            task.prepared,
            source_path=source_path,
        )
        forecaster = load_adaptime_forecaster(
            workflow.model,
            horizon=horizon,
            period=period,
            model_path=workflow.model_path,
            weights_id=workflow.weights_id,
            device=workflow.device,
        )
        extraction_manifest = extract_adaptation_features(
            prepared_manifest,
            forecaster,
            ExtractionConfig(
                representation=workflow.representation,
                distance_metric=workflow.distance_metric,
                retrieval_scope=workflow.retrieval_scope,
                max_k=workflow.max_k,
                context_k=workflow.k_values,
                model_batch_size=workflow.model_batch_size,
                query_block_size=workflow.query_block_size,
                datastore_block_size=workflow.datastore_block_size,
                arrow_cache_items=workflow.arrow_cache_items,
            ),
            task.extraction,
        )
        print(extraction_manifest, flush=True)


def run_adaptation_stage(
    stage: str,
    workflow: AdaptimeWorkflowConfig,
    *,
    dataset_config_path: Path | None = None,
    datasets_selected: Iterable[str] = ("all_datasets",),
    terms_selected: Iterable[str] | None = None,
    output_root: Path | None = None,
) -> None:
    """Run one resumable stage over the selected TIME dataset/term tasks."""

    workflow.validate()
    if stage not in {"extract", "train", "test"}:
        raise ValueError("stage must be extract, train, or test")
    dataset_config = load_dataset_config(dataset_config_path)
    artifact_root = (output_root or outputs_root() / "adaptime").expanduser().resolve()
    tasks = workflow_tasks(
        dataset_config,
        datasets_selected,
        terms_selected,
        artifact_root,
        workflow,
    )
    if stage == "extract":
        for dataset_name in _selected_datasets(dataset_config, datasets_selected):
            selected_tasks = [task for task in tasks if task.dataset == dataset_name]
            _prepare_and_extract_dataset(
                dataset_name, selected_tasks, dataset_config, workflow
            )
        return

    for task in tasks:
        if stage == "train":
            manifest = fit_full_ridge(
                task.prepared,
                task.extraction,
                RidgeTrainingConfig(
                    k_values=workflow.k_values,
                    alpha_values=workflow.alpha_values,
                    chunk_size=workflow.ridge_chunk_size,
                    seed=workflow.seed,
                ),
                task.model,
            )
        else:
            manifest = evaluate_frozen_adaptation(
                task.prepared,
                task.extraction,
                task.model,
                AdaptimeTestingConfig(chunk_size=workflow.ridge_chunk_size),
                task.comparison,
            )
        print(manifest, flush=True)
    if stage == "test":
        aggregate = aggregate_time_comparison(
            tasks,
            artifact_root
            / "comparison"
            / workflow.model
            / workflow.target_mode
            / "aggregate",
        )
        print(aggregate, flush=True)
