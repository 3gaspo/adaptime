"""Task lifecycle and tables for the matched TS-RAG comparison."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from timebench.evaluation.adaptation_data import PreparedDataset
from timebench.evaluation.data import load_dataset_config
from timebench.evaluation.utils import get_available_terms
from timebench.external_models.tsrag.retriever import TSRAGRetriever
from timebench.model_loading.tsrag import LoadedTSRAG, load_tsrag
from timebench.paths import outputs_root, weights_root
from timebench.pipeline.runs import allocate_run, select_completed_runs
from timebench.pipeline.tsrag import (
    TSRAG_CONTEXT_LENGTH,
    TSRAG_DATASTORE_STRIDE,
    TSRAG_EMBEDDING_BATCH_SIZE,
    TSRAG_EMBEDDING_DIMENSION,
    TSRAG_NATIVE_HORIZON,
    TSRAG_SOURCE_COMMIT,
    TSRAG_TOP_K,
    TSRAGRuntimeConfig,
    evaluate_tsrag,
    extract_tsrag_features,
)
from timebench.pipeline.tsrag_data import prepare_tsrag_dataset


@dataclass(frozen=True)
class TSRAGWorkflowConfig:
    ridge_model: str = "chronos2"
    device: str = "cuda"
    model_batch_size: int = 256
    arrow_cache_items: int = 2
    chronos_bolt_path: Path | None = None
    retriever_path: Path | None = None
    checkpoint_path: Path | None = None

    def validate(self) -> None:
        if self.ridge_model != "chronos2":
            raise ValueError("the matched full_ridge_shared dependency must use chronos2")
        TSRAGRuntimeConfig(
            model_batch_size=self.model_batch_size,
            arrow_cache_items=self.arrow_cache_items,
        ).validate()


@dataclass(frozen=True)
class TSRAGTask:
    dataset: str
    term: str
    identity_root: Path
    ridge_identity_root: Path


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


def comparison_tasks(
    dataset_config: dict[str, object],
    datasets_selected: Iterable[str],
    terms_selected: Iterable[str] | None,
    output_root: Path,
    ridge_root: Path,
    ridge_model: str,
) -> list[TSRAGTask]:
    return [
        TSRAGTask(
            dataset=dataset,
            term=term,
            identity_root=(
                output_root / "tasks" / "tsrag" / "univariate" / dataset / term
            ),
            ridge_identity_root=(
                ridge_root
                / "tasks"
                / ridge_model
                / "univariate"
                / dataset
                / term
            ),
        )
        for dataset in _selected_datasets(dataset_config, datasets_selected)
        for term in _selected_terms(dataset, dataset_config, terms_selected)
    ]


def _one_completed_run(
    root: Path,
    *,
    launch_id: str | None,
    config_policy: str,
    repeat_policy: str,
    name: str,
) -> tuple[Path, dict[str, object]]:
    selected = select_completed_runs(
        root,
        launch_id=launch_id,
        config_policy=config_policy,
        repeat_policy=repeat_policy,
    )
    if len(selected) != 1:
        raise ValueError(
            f"expected one selected completed {name} run below {root}, found {len(selected)}"
        )
    return selected[0]


def _scientific_config(manifest: dict[str, object]) -> dict[str, object]:
    return {
        "model_config": manifest.get("model_config", {}),
        "pipeline_config": manifest.get("pipeline_config", {}),
        "experiment_config": manifest.get("experiment_config", {}),
    }


def _resolve_paths(
    workflow: TSRAGWorkflowConfig,
) -> tuple[Path, Path, Path]:
    root = weights_root()
    base = (workflow.chronos_bolt_path or root / "chronos-bolt-base").expanduser().resolve()
    retriever = (workflow.retriever_path or root / "chronos-t5-base").expanduser().resolve()
    checkpoint = (workflow.checkpoint_path or root / "ts-rag").expanduser().resolve()
    return base, retriever, checkpoint


def _method_row(
    *,
    dataset: str,
    term: str,
    horizon: int,
    method: str,
    backbone: str,
    context_length: int,
    summary: dict[str, object],
    timing_method: str,
    metric_method: str,
) -> dict[str, object]:
    methods = dict(summary["methods"])
    metric = dict(dict(methods[metric_method])["scaled_mase"])
    timing = dict(dict(summary["timing"])["methods"])[timing_method]
    return {
        "dataset": dataset,
        "term": term,
        "horizon": int(horizon),
        "method": method,
        "backbone": backbone,
        "context_length": int(context_length),
        "scaled_mase_equal_user": float(metric["equal_user_mean"]),
        "total_inference_seconds": float(timing["total_seconds"]),
        "test_windows": int(dict(summary["timing"])["test_windows"]),
    }


def _write_markdown(path: Path, title: str, rows: list[dict[str, object]]) -> None:
    lines = [
        f"# {title}",
        "",
        "| Dataset | Term | H | Method | Backbone | L | Scaled MASE | Total inference (s) | Windows |",
        "|---|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['term']} | {row['horizon']} | "
            f"{row['method']} | {row['backbone']} | {row['context_length']} | "
            f"{float(row['scaled_mase_equal_user']):.6f} | "
            f"{float(row['total_inference_seconds']):.3f} | {row['test_windows']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# TS-RAG comparison summary",
        "",
        "Each task MASE is divided by matching Seasonal Naive MASE, then tasks are "
        "combined with a geometric mean. Inference time is summed over every task.",
        "",
        "| Method | Backbone | Scaled MASE (GM) | Total inference (s) | Datasets | Tasks | Windows |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['backbone']} | "
            f"{float(row['scaled_mase_geometric_mean']):.6f} | "
            f"{float(row['total_inference_seconds']):.3f} | "
            f"{row['datasets']} | {row['tasks']} | {row['test_windows']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_tsrag_comparison_table(
    tasks: list[TSRAGTask],
    output_dir: Path,
    *,
    ridge_launch_id: str | None,
    tsrag_launch_id: str | None,
    config_policy: str,
    repeat_policy: str,
) -> Path:
    """Build identical-support scaled-MASE and total-time tables."""

    rows: list[dict[str, object]] = []
    input_manifests: list[str] = []
    for task in tasks:
        ridge_dir, ridge_manifest = _one_completed_run(
            task.ridge_identity_root,
            launch_id=ridge_launch_id,
            config_policy=config_policy,
            repeat_policy=repeat_policy,
            name="full_ridge_shared",
        )
        tsrag_dir, tsrag_manifest = _one_completed_run(
            task.identity_root,
            launch_id=tsrag_launch_id,
            config_policy=config_policy,
            repeat_policy=repeat_policy,
            name="TS-RAG",
        )
        ridge_result = json.loads(
            (ridge_dir / "comparison" / "result_manifest.json").read_text(encoding="utf-8")
        )
        ridge_summary = json.loads(
            (
                ridge_dir
                / "comparison"
                / ridge_result["files"]["comparison_summary"]
            ).read_text(encoding="utf-8")
        )
        tsrag_result = json.loads(
            (tsrag_dir / "comparison" / "result_manifest.json").read_text(encoding="utf-8")
        )
        tsrag_summary = json.loads(
            (
                tsrag_dir
                / "comparison"
                / tsrag_result["files"]["comparison_summary"]
            ).read_text(encoding="utf-8")
        )
        ridge_prepared = PreparedDataset(ridge_dir / "prepared" / "manifest.json")
        tsrag_prepared_manifest = json.loads(
            (tsrag_dir / "prepared" / "manifest.json").read_text(encoding="utf-8")
        )
        if tsrag_prepared_manifest["test_reference_sha256"] != _reference_hash(
            ridge_prepared.indices("test")
        ):
            raise ValueError(f"TS-RAG and ridge test rows differ for {task.dataset}/{task.term}")
        horizon = int(ridge_prepared.prediction_length)
        rows.extend(
            (
                _method_row(
                    dataset=task.dataset,
                    term=task.term,
                    horizon=horizon,
                    method="vanilla",
                    backbone="chronos_bolt",
                    context_length=TSRAG_CONTEXT_LENGTH,
                    summary=tsrag_summary,
                    timing_method="vanilla",
                    metric_method="vanilla",
                ),
                _method_row(
                    dataset=task.dataset,
                    term=task.term,
                    horizon=horizon,
                    method="tsrag",
                    backbone="chronos_bolt",
                    context_length=TSRAG_CONTEXT_LENGTH,
                    summary=tsrag_summary,
                    timing_method="tsrag",
                    metric_method="tsrag",
                ),
                _method_row(
                    dataset=task.dataset,
                    term=task.term,
                    horizon=horizon,
                    method="full_ridge_shared",
                    backbone=str(ridge_manifest["identity"]["model"]),
                    context_length=int(ridge_manifest["pipeline_config"]["context_length"]),
                    summary=ridge_summary,
                    timing_method="adaptime",
                    metric_method="adaptime",
                ),
            )
        )
        input_manifests.extend(
            (str(ridge_dir / "manifest.json"), str(tsrag_dir / "manifest.json"))
        )

    summary_rows: list[dict[str, object]] = []
    method_keys = sorted({(str(row["method"]), str(row["backbone"])) for row in rows})
    for method, backbone in method_keys:
        method_rows = [
            row
            for row in rows
            if row["method"] == method and row["backbone"] == backbone
        ]
        datasets = sorted({str(row["dataset"]) for row in method_rows})
        task_values = np.asarray(
            [float(row["scaled_mase_equal_user"]) for row in method_rows],
            dtype=np.float64,
        )
        finite_positive = task_values[np.isfinite(task_values) & (task_values > 0)]
        summary_rows.append(
            {
                "method": method,
                "backbone": backbone,
                "scaled_mase_geometric_mean": (
                    float(np.exp(np.log(finite_positive).mean()))
                    if len(finite_positive)
                    else np.nan
                ),
                "total_inference_seconds": float(
                    sum(float(row["total_inference_seconds"]) for row in method_rows)
                ),
                "datasets": len(datasets),
                "tasks": len(method_rows),
                "test_windows": sum(int(row["test_windows"]) for row in method_rows),
            }
        )

    root = output_dir.expanduser().resolve()
    detailed_csv = root / "tsrag_comparison_tasks.csv"
    detailed_md = root / "tsrag_comparison_tasks.md"
    summary_csv = root / "tsrag_comparison_summary.csv"
    summary_md = root / "tsrag_comparison_summary.md"
    _atomic_csv(detailed_csv, rows)
    _write_markdown(detailed_md, "TS-RAG comparison by TIME task", rows)
    _atomic_csv(summary_csv, summary_rows)
    _write_summary_markdown(summary_md, summary_rows)
    manifest_path = root / "report_manifest.json"
    _atomic_json(
        manifest_path,
        {
            "schema_version": 1,
            "format": "adaptime_tsrag_comparison_table",
            "status": "completed",
            "selection": {
                "ridge_launch_id": ridge_launch_id,
                "tsrag_launch_id": tsrag_launch_id,
                "config_policy": config_policy,
                "repeat_policy": repeat_policy,
            },
            "aggregation": {
                "scaled_mase": "task_mase_divided_by_matching_seasonal_naive_mase_then_geometric_mean",
                "inference_seconds": "sum_over_selected_official_time_tasks",
            },
            "input_manifests": input_manifests,
            "files": {
                "detailed_csv": detailed_csv.name,
                "detailed_markdown": detailed_md.name,
                "summary_csv": summary_csv.name,
                "summary_markdown": summary_md.name,
            },
        },
    )
    return manifest_path


def _reference_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def run_tsrag_comparison(
    workflow: TSRAGWorkflowConfig,
    *,
    dataset_config_path: Path | None = None,
    datasets_selected: Iterable[str] = ("all_datasets",),
    terms_selected: Iterable[str] | None = None,
    output_root: Path | None = None,
    ridge_output_root: Path | None = None,
    ridge_launch_id: str | None = None,
    config_policy: str = "error",
    repeat_policy: str = "selected",
) -> Path:
    """Evaluate TS-RAG against the ridge run selected on the same test rows."""

    workflow.validate()
    dataset_config = load_dataset_config(dataset_config_path)
    artifact_root = (output_root or outputs_root() / "tsrag_comparison").expanduser().resolve()
    ridge_root = (ridge_output_root or outputs_root() / "adaptime").expanduser().resolve()
    launch_id = os.environ.get("TIME_LAUNCH_ID")
    selected_ridge_launch = ridge_launch_id
    tasks = comparison_tasks(
        dataset_config,
        datasets_selected,
        terms_selected,
        artifact_root,
        ridge_root,
        workflow.ridge_model,
    )
    base_path, retriever_path, checkpoint_path = _resolve_paths(workflow)
    runtime = TSRAGRuntimeConfig(
        model_batch_size=workflow.model_batch_size,
        arrow_cache_items=workflow.arrow_cache_items,
    )
    retriever: TSRAGRetriever | None = None
    loaded: LoadedTSRAG | None = None

    for task in tasks:
        ridge_dir, ridge_manifest = _one_completed_run(
            task.ridge_identity_root,
            launch_id=selected_ridge_launch,
            config_policy=config_policy,
            repeat_policy=repeat_policy,
            name="full_ridge_shared",
        )
        ridge_prepared_path = ridge_dir / "prepared" / "manifest.json"
        ridge_prepared = PreparedDataset(ridge_prepared_path)
        allocation = allocate_run(
            task.identity_root,
            experiment="tsrag_comparison",
            identity={
                "model": "tsrag",
                "target_mode": "univariate",
                "dataset": ridge_manifest["identity"]["dataset"],
                "frequency": ridge_manifest["identity"]["frequency"],
                "term": task.term,
            },
            model_config={
                "method": "tsrag",
                "backbone": "chronos_bolt",
                "source_commit": TSRAG_SOURCE_COMMIT,
                "context_length": TSRAG_CONTEXT_LENGTH,
                "native_prediction_length": TSRAG_NATIVE_HORIZON,
                "top_k": TSRAG_TOP_K,
                "embedding": "chronos_t5_base_eos",
                "embedding_dimension": TSRAG_EMBEDDING_DIMENSION,
                "index": "faiss.IndexFlatL2_float32",
            },
            pipeline_config={
                "dependency": {
                    "full_ridge_shared": _scientific_config(ridge_manifest),
                },
                "test_support": "exact_ridge_official_time_test_references",
                "accessible_dates": "same_per_item_raw_date_union_as_full_ridge_shared",
                "datastore_scope": "same_series",
                "datastore_stride": TSRAG_DATASTORE_STRIDE,
                "embedding_batch_size": TSRAG_EMBEDDING_BATCH_SIZE,
                "retrieval_rule": "top_k_plus_one_then_remove_final_result",
                "horizon": ridge_prepared.prediction_length,
                "rollout": (
                    "native_single_call_crop"
                    if ridge_prepared.prediction_length <= TSRAG_NATIVE_HORIZON
                    else "autoregressive_64_step_reembed_retrieve"
                ),
            },
            runtime_config={
                **asdict(runtime),
                "device": workflow.device,
                "chronos_bolt_path": str(base_path),
                "retriever_path": str(retriever_path),
                "checkpoint_path": str(checkpoint_path),
            },
            experiment_config={
                "comparison": ["vanilla", "tsrag", "full_ridge_shared"],
                "metrics": ["scaled_mase"],
                "timing": "total_test_inference_seconds",
            },
            provenance={
                "ridge_manifest": str(ridge_dir / "manifest.json"),
                "upstream_repository": "https://github.com/UConn-DSIS/TS-RAG",
                "upstream_commit": TSRAG_SOURCE_COMMIT,
            },
        )
        if not allocation.should_run:
            continue
        if retriever is None:
            retriever = TSRAGRetriever(
                retriever_path,
                device_map=workflow.device,
                local_files_only=True,
            )
        if loaded is None:
            loaded = load_tsrag(
                base_path,
                checkpoint_path,
                device=workflow.device,
            )
        with allocation:
            prepared_manifest = prepare_tsrag_dataset(
                ridge_prepared_path,
                allocation.run_dir / "prepared",
            )
            extraction_manifest = extract_tsrag_features(
                prepared_manifest,
                retriever,
                runtime,
                allocation.run_dir / "extraction",
            )
            result_manifest = evaluate_tsrag(
                prepared_manifest,
                extraction_manifest,
                loaded,
                retriever,
                runtime,
                allocation.run_dir / "comparison",
                device=workflow.device,
            )
            allocation.complete(
                [
                    "prepared/manifest.json",
                    "extraction/manifest.json",
                    "comparison/result_manifest.json",
                    "comparison/comparison_summary.json",
                ]
            )
        print(result_manifest, flush=True)

    report = build_tsrag_comparison_table(
        tasks,
        artifact_root / "summary" / (launch_id or "manual"),
        ridge_launch_id=selected_ridge_launch,
        tsrag_launch_id=launch_id,
        config_policy=config_policy,
        repeat_policy=repeat_policy,
    )
    print(report, flush=True)
    return report
