"""TIME-wide orchestration for Adaptime extraction, fitting, and testing."""

from __future__ import annotations

import csv
import json
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
from timebench.model_loading import foundation_context_length, load_adaptime_forecaster
from timebench.paths import dataset_storage_root, outputs_root
from timebench.pipeline.adaptime_extraction import (
    ExtractionConfig,
    extract_adaptation_features,
)
from timebench.pipeline.adaptime_testing import (
    AdaptimeTestingConfig,
    METHODS,
    METRICS,
    SCORE_METHODS,
    evaluate_frozen_adaptation,
)
from timebench.pipeline.adaptime_training import RidgeTrainingConfig, fit_full_ridge
from timebench.pipeline.runs import allocate_run, select_completed_runs


@dataclass(frozen=True)
class AdaptimeWorkflowConfig:
    model: str = "chronos2"
    target_mode: str = "univariate"
    adaptation_train_length: int | None = None
    adaptation_validation_length: int | None = None
    adaptation_stride: int | None = None
    retrieval_period: int | None = None
    datastore_stride_multiple: int = 1
    datastore_length: int | None = None
    representation: str = "instance"
    distance_metric: str = "euclidean"
    retrieval_scope: str = "all"
    minimum_overlap_fraction: float = 0.8
    minimum_query_finite_fraction: float = 0.8
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
        if int(self.datastore_stride_multiple) <= 0:
            raise ValueError("datastore_stride_multiple must be positive")
        if max(self.k_values) > int(self.max_k):
            raise ValueError("max_k must cover every selected K")
        if not 0.0 < float(self.minimum_overlap_fraction) <= 1.0:
            raise ValueError("minimum_overlap_fraction must be in (0, 1]")
        if not 0.0 < float(self.minimum_query_finite_fraction) <= 1.0:
            raise ValueError("minimum_query_finite_fraction must be in (0, 1]")


@dataclass(frozen=True)
class AdaptimeTask:
    dataset: str
    term: str
    identity_root: Path


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
        identity_root=output_root / "tasks" / model_suffix,
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
    *,
    launch_id: str | None = None,
    config_policy: str = "error",
    repeat_policy: str = "selected",
) -> Path:
    """Select run manifests, then aggregate repeats, configs, terms, and datasets."""

    rows: list[dict[str, object]] = []
    input_manifests: list[str] = []
    for task in tasks:
        selected_runs = select_completed_runs(
            task.identity_root,
            launch_id=launch_id,
            config_policy=config_policy,
            repeat_policy=repeat_policy,
        )
        if not selected_runs:
            raise ValueError(
                f"no completed Adaptime run matches {task.dataset}/{task.term}"
            )
        for run_dir, run_manifest in selected_runs:
            result_path = run_dir / "comparison" / "result_manifest.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("status") != "completed":
                raise ValueError(f"incomplete Adaptime result: {result_path}")
            summary = json.loads(
                (run_dir / "comparison" / result["files"]["comparison_summary"]).read_text(
                    encoding="utf-8"
                )
            )
            selected = dict(result["selected"])
            selection = dict(run_manifest.get("selection", {}))
            row: dict[str, object] = {
                "variant": selection.get("model_label", run_manifest["identity"]["model"]),
                "dataset": task.dataset,
                "term": task.term,
                "run": run_dir.name,
                "scientific_config": json.dumps(
                    selection.get("scientific_config", {}), sort_keys=True
                ),
                "selected_k": int(selected["k"]),
                "selected_alpha": float(selected["alpha"]),
                "validation_msse": float(selected["validation_msse"]),
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
                scaled_mase = dict(method_summary["scaled_mase"])
                row[f"{method}_scaled_mase_equal_user"] = float(
                    scaled_mase["equal_user_mean"]
                )
                row[f"{method}_scaled_mase_equal_window"] = float(
                    scaled_mase["equal_window_mean"]
                )
                row[f"{method}_inference_seconds_per_window"] = float(
                    summary["timing"]["methods"][method]["seconds_per_window"]
                )
            for method in ("covariate", "adaptime"):
                row[f"{method}_scaled_mase_win_rate_vs_vanilla"] = float(
                    summary["scaled_mase_win_rate_vs_vanilla"][method]
                )
            row["rag_eligible_fraction"] = float(
                summary["rag_coverage"]["eligible_fraction"]
            )
            rows.append(row)
            input_manifests.append(str(run_dir / "manifest.json"))

    numeric_fields = [
        key
        for key in rows[0]
        if key
        not in {"variant", "dataset", "term", "run", "scientific_config"}
    ]

    def averaged(
        groups: dict[tuple[str, ...], list[dict[str, object]]],
        keys: tuple[str, ...],
    ) -> list[dict[str, object]]:
        output: list[dict[str, object]] = []
        for values in groups.values():
            row = {key: values[0][key] for key in keys}
            row.update(
                {
                    field: float(np.mean([float(value[field]) for value in values]))
                    for field in numeric_fields
                }
            )
            output.append(row)
        return output

    exact_groups: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = tuple(
            str(row[field])
            for field in ("variant", "dataset", "term", "scientific_config")
        )
        exact_groups.setdefault(key, []).append(row)
    per_config = averaged(
        exact_groups, ("variant", "dataset", "term", "scientific_config")
    )
    task_groups: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for row in per_config:
        key = tuple(str(row[field]) for field in ("variant", "dataset", "term"))
        task_groups.setdefault(key, []).append(row)
    effective_rows = averaged(task_groups, ("variant", "dataset", "term"))

    variants: dict[str, object] = {}
    for variant in sorted({str(row["variant"]) for row in effective_rows}):
        variant_rows = [row for row in effective_rows if row["variant"] == variant]
        dataset_rows: dict[str, list[dict[str, object]]] = {}
        for row in variant_rows:
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
            scaled_field = f"{method}_scaled_mase_equal_user"
            task_scaled_mase = np.asarray(
                [float(row[scaled_field]) for row in variant_rows],
                dtype=np.float64,
            )
            finite_positive = task_scaled_mase[
                np.isfinite(task_scaled_mase) & (task_scaled_mase > 0)
            ]
            metrics["scaled_mase"] = {
                "task_geometric_mean": (
                    float(np.exp(np.log(finite_positive).mean()))
                    if len(finite_positive)
                    else np.nan
                ),
                "finite_tasks": int(len(finite_positive)),
                "total_tasks": int(len(task_scaled_mase)),
            }
            timing_field = f"{method}_inference_seconds_per_window"
            per_dataset_timing = np.asarray(
                [
                    np.mean(
                        [float(row[timing_field]) for row in dataset_rows[dataset]]
                    )
                    for dataset in sorted(dataset_rows)
                ],
                dtype=np.float64,
            )
            metrics["inference_seconds_per_window"] = {
                "equal_dataset_mean": float(per_dataset_timing.mean()),
                "equal_dataset_std": float(per_dataset_timing.std()),
            }
            methods[method] = metrics
        wins: dict[str, object] = {}
        for method in ("covariate", "adaptime"):
            field = f"{method}_scaled_mase_win_rate_vs_vanilla"
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
        variants[variant] = {
            "datasets": len(dataset_rows),
            "tasks": len(variant_rows),
            "methods": methods,
            "scaled_mase_win_rate_vs_vanilla": wins,
            "rag_coverage": {
                "equal_dataset_eligible_fraction_mean": float(
                    np.mean(
                        [
                            np.mean(
                                [
                                    float(row["rag_eligible_fraction"])
                                    for row in dataset_rows[dataset]
                                ]
                            )
                            for dataset in sorted(dataset_rows)
                        ]
                    )
                )
            },
        }

    root = output_dir.expanduser().resolve()
    manifest_path = root / "time_summary_manifest.json"
    _atomic_csv(root / "time_tasks.csv", rows)
    _atomic_json(
        root / "time_summary.json",
        {"variants": variants},
    )
    _atomic_json(
        manifest_path,
        {
            "schema_version": 1,
            "format": "adaptime_time_aggregate",
            "protocol": "task_scaled_mase_then_geometric_mean_across_tasks",
            "status": "completed",
            "selection": {
                "launch_id": launch_id,
                "config_policy": config_policy,
                "repeat_policy": repeat_policy,
            },
            "input_manifests": input_manifests,
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


def _run_dataset_tasks(
    dataset_name: str,
    tasks: list[AdaptimeTask],
    dataset_config: dict[str, object],
    workflow: AdaptimeWorkflowConfig,
    dataset_config_path: Path | None,
) -> None:
    source_path = dataset_storage_root() / dataset_name
    hf_dataset = datasets.load_from_disk(str(source_path))
    if len(hf_dataset) == 0:
        raise ValueError(f"empty TIME dataset: {dataset_name}")
    first = hf_dataset[0]
    freq = str(first["freq"])
    metric_seasonality = int(get_seasonality(freq))
    period = int(workflow.retrieval_period or metric_seasonality)
    shapes = [np.asarray(hf_dataset[index]["target"]).shape for index in range(len(hf_dataset))]
    min_length = min(int(shape[-1]) for shape in shapes)

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
        context_length = foundation_context_length(workflow.model)
        if datastore_stop - context_length < horizon:
            raise ValueError(
                f"{dataset_name}/{task.term} cannot supply the foundation-model "
                f"context L={context_length} before adaptation"
            )
        preparation = PreparationConfig(
            dataset=dataset_name,
            term=task.term,
            context_length=context_length,
            prediction_length=horizon,
            test_length=int(settings["test_length"]),
            adaptation_train_length=train_length,
            adaptation_validation_length=validation_length,
            seasonality=metric_seasonality,
            target_mode=workflow.target_mode,
            adaptation_stride=workflow.adaptation_stride,
            retrieval_period=period,
            datastore_stride=datastore_stride,
            datastore_length=workflow.datastore_length,
        )
        run = allocate_run(
            task.identity_root,
            experiment="adaptime",
            identity={
                "model": workflow.model,
                "target_mode": workflow.target_mode,
                "dataset": dataset_name.rpartition("/")[0] or dataset_name,
                "frequency": freq,
                "term": task.term,
            },
            model_config={
                "method": "full_ridge_shared",
                "weights_id": workflow.weights_id,
                "representation": workflow.representation,
                "distance_metric": workflow.distance_metric,
                "retrieval_scope": workflow.retrieval_scope,
                "minimum_overlap_fraction": workflow.minimum_overlap_fraction,
                "minimum_query_finite_fraction": workflow.minimum_query_finite_fraction,
                "max_k": workflow.max_k,
                "k_values": list(workflow.k_values),
                "alpha_values": list(workflow.alpha_values),
                "seed": workflow.seed,
            },
            pipeline_config={
                "context_length": context_length,
                "prediction_length": horizon,
                "test_length": int(settings["test_length"]),
                "adaptation_train_length": train_length,
                "adaptation_validation_length": validation_length,
                "adaptation_stride": preparation.query_stride,
                "retrieval_period": period,
                "datastore_stride": datastore_stride,
                "datastore_length": workflow.datastore_length,
            },
            runtime_config={
                "model_batch_size": workflow.model_batch_size,
                "query_block_size": workflow.query_block_size,
                "datastore_block_size": workflow.datastore_block_size,
                "arrow_cache_items": workflow.arrow_cache_items,
                "ridge_chunk_size": workflow.ridge_chunk_size,
                "device": workflow.device,
                "model_path": (
                    str(workflow.model_path) if workflow.model_path is not None else None
                ),
            },
            experiment_config={
                "target_mode": workflow.target_mode,
                "methods": list(SCORE_METHODS),
                "metrics": list(METRICS),
                "performance_metric": "scaled_mase",
                "formulation": "full_ridge_shared",
            },
            provenance={
                "dataset_source": str(source_path),
                "dataset_config": (
                    str(dataset_config_path.expanduser().resolve())
                    if dataset_config_path is not None
                    else "timebench.config.datasets"
                ),
            },
        )
        if not run.should_run:
            continue

        prepared_dir = run.run_dir / "prepared"
        extraction_dir = run.run_dir / "extraction"
        training_dir = run.run_dir / "training"
        comparison_dir = run.run_dir / "comparison"
        with run:
            prepared_manifest = prepare_adaptation_dataset(
                hf_dataset,
                preparation,
                prepared_dir,
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
                    minimum_overlap_fraction=workflow.minimum_overlap_fraction,
                    minimum_query_finite_fraction=workflow.minimum_query_finite_fraction,
                    max_k=workflow.max_k,
                    context_k=workflow.k_values,
                    model_batch_size=workflow.model_batch_size,
                    query_block_size=workflow.query_block_size,
                    datastore_block_size=workflow.datastore_block_size,
                    arrow_cache_items=workflow.arrow_cache_items,
                ),
                extraction_dir,
            )
            model_manifest = fit_full_ridge(
                prepared_manifest,
                extraction_manifest,
                RidgeTrainingConfig(
                    k_values=workflow.k_values,
                    alpha_values=workflow.alpha_values,
                    chunk_size=workflow.ridge_chunk_size,
                    seed=workflow.seed,
                ),
                training_dir,
            )
            result_manifest = evaluate_frozen_adaptation(
                prepared_manifest,
                extraction_manifest,
                model_manifest,
                AdaptimeTestingConfig(
                    chunk_size=workflow.ridge_chunk_size,
                ),
                comparison_dir,
            )
            run.complete(
                [
                    "prepared/manifest.json",
                    "extraction/manifest.json",
                    "training/model_manifest.json",
                    "comparison/result_manifest.json",
                    "comparison/comparison_summary.json",
                ]
            )
        print(result_manifest, flush=True)


def run_adaptation_stage(
    stage: str,
    workflow: AdaptimeWorkflowConfig,
    *,
    dataset_config_path: Path | None = None,
    datasets_selected: Iterable[str] = ("all_datasets",),
    terms_selected: Iterable[str] | None = None,
    output_root: Path | None = None,
    config_policy: str = "error",
    repeat_policy: str = "selected",
) -> None:
    """Run each TIME dataset/term task atomically, then aggregate selected runs."""

    workflow.validate()
    if stage != "run":
        raise ValueError("stage must be run")
    dataset_config = load_dataset_config(dataset_config_path)
    artifact_root = (output_root or outputs_root() / "adaptime").expanduser().resolve()
    tasks = workflow_tasks(
        dataset_config,
        datasets_selected,
        terms_selected,
        artifact_root,
        workflow,
    )
    for dataset_name in _selected_datasets(dataset_config, datasets_selected):
        selected_tasks = [task for task in tasks if task.dataset == dataset_name]
        _run_dataset_tasks(
            dataset_name,
            selected_tasks,
            dataset_config,
            workflow,
            dataset_config_path,
        )

    launch_id = os.environ.get("TIME_LAUNCH_ID")
    aggregate = aggregate_time_comparison(
        tasks,
        artifact_root
        / "summary"
        / workflow.model
        / workflow.target_mode
        / (launch_id or "manual"),
        launch_id=launch_id,
        config_policy=config_policy,
        repeat_policy=repeat_policy,
    )
    print(aggregate, flush=True)
