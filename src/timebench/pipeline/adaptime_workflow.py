"""Independent data, extraction, fitting, prediction, and evaluation stages."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import datasets
import numpy as np
from gluonts.time_feature import get_seasonality, norm_freq_str
from pandas.tseries.frequencies import to_offset

from timebench.evaluation.adaptation import evaluate_point_predictions
from timebench.evaluation.adaptation_data import (
    PreparationConfig,
    adaptation_split_lengths,
    adaptation_stride_for_frequency,
    prepare_adaptation_dataset,
)
from timebench.evaluation.data import (
    M4_PRED_LENGTH_MAP,
    PRED_LENGTH_MAP,
    Term,
    get_dataset_settings,
    load_dataset_config,
)
from timebench.evaluation.utils import get_available_terms
from timebench.external_models.tsrag.retriever import TSRAGRetriever
from timebench.model_loading import foundation_context_length, load_adaptime_forecaster
from timebench.model_loading.tsrag import LoadedTSRAG, load_tsrag
from timebench.paths import dataset_storage_root, outputs_root, weights_root
from timebench.pipeline.adaptation_prediction import (
    ADAPTATION_METHODS,
    PredictionConfig,
    predict_adaptation_family,
)
from timebench.pipeline.adaptime_extraction import (
    ExtractionConfig,
    extract_adaptation_eval_features,
    extract_adaptation_features,
)
from timebench.pipeline.adaptime_training import RidgeTrainingConfig, fit_full_ridge
from timebench.pipeline.adaptime_vanilla import (
    VanillaConfig,
    extract_vanilla_test_forecasts,
)
from timebench.pipeline.runs import allocate_run, select_completed_runs
from timebench.pipeline.tsrag import (
    TSRAG_CONTEXT_LENGTH,
    TSRAG_EMBEDDING_DIMENSION,
    TSRAG_NATIVE_HORIZON,
    TSRAG_SOURCE_COMMIT,
    TSRAG_TOP_K,
    TSRAGRuntimeConfig,
    extract_tsrag_features,
    predict_tsrag,
)
from timebench.results.adaptation import build_adaptation_comparison


METHODS = ("ridge", "tsrag")
STAGES = (
    "prepare",
    "vanilla",
    "extract",
    "fit",
    "extract_eval",
    "predict",
    "evaluate",
    "report",
    "pipeline",
    "all",
)


@dataclass(frozen=True)
class AdaptimeWorkflowConfig:
    """One configuration namespace shared by both Adaptime wrappers."""

    model: str = "chronos2"
    target_mode: str = "univariate"
    adaptation_stride: int | None = None
    retrieval_period: int | None = None
    datastore_stride_multiple: int = 1
    max_datastore_windows: int | None = None
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
    tsrag_model_batch_size: int = 256
    tsrag_chronos_bolt_path: Path | None = None
    tsrag_retriever_path: Path | None = None
    tsrag_checkpoint_path: Path | None = None

    def validate(self, method: str) -> None:
        if self.target_mode != "univariate":
            raise ValueError("Adaptime wrappers currently use univariate TIME rows")
        if int(self.datastore_stride_multiple) <= 0:
            raise ValueError("datastore_stride_multiple must be positive")
        if self.adaptation_stride is not None and int(self.adaptation_stride) <= 1:
            raise ValueError("adaptation_stride must be greater than one when supplied")
        if self.max_datastore_windows is not None and int(self.max_datastore_windows) <= 0:
            raise ValueError("max_datastore_windows must be positive when supplied")
        if method == "ridge":
            ExtractionConfig(
                representation=self.representation,
                distance_metric=self.distance_metric,
                retrieval_scope=self.retrieval_scope,
                minimum_overlap_fraction=self.minimum_overlap_fraction,
                minimum_query_finite_fraction=self.minimum_query_finite_fraction,
                max_k=self.max_k,
                context_k=self.k_values,
                model_batch_size=self.model_batch_size,
                query_block_size=self.query_block_size,
                datastore_block_size=self.datastore_block_size,
                arrow_cache_items=self.arrow_cache_items,
            ).validate()
            self.ridge_training.validate()
        else:
            self.tsrag_runtime.validate()

    @property
    def ridge_training(self) -> RidgeTrainingConfig:
        return RidgeTrainingConfig(
            k_values=self.k_values,
            alpha_values=self.alpha_values,
            chunk_size=self.ridge_chunk_size,
            seed=self.seed,
        )

    @property
    def tsrag_runtime(self) -> TSRAGRuntimeConfig:
        return TSRAGRuntimeConfig(
            model_batch_size=self.tsrag_model_batch_size,
            arrow_cache_items=self.arrow_cache_items,
        )


@dataclass(frozen=True)
class AdaptimeTask:
    dataset: str
    term: str
    frequency: str
    source_path: Path
    preparation: PreparationConfig
    minimum_series_length: int
    dataset_fingerprint: str


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


def workflow_tasks(
    dataset_config: dict[str, object],
    datasets_selected: Iterable[str],
    terms_selected: Iterable[str] | None,
    workflow: AdaptimeWorkflowConfig,
) -> list[AdaptimeTask]:
    """Resolve the same task plan regardless of the consuming method."""

    tasks: list[AdaptimeTask] = []
    context_length = foundation_context_length(workflow.model)
    if context_length < TSRAG_CONTEXT_LENGTH:
        raise ValueError("the shared context must cover TS-RAG's native L=512")
    for dataset_name in _selected_datasets(dataset_config, datasets_selected):
        source_path = (dataset_storage_root() / dataset_name).resolve()
        source = datasets.load_from_disk(str(source_path))
        if len(source) == 0:
            raise ValueError(f"empty TIME dataset: {dataset_name}")
        frequency = str(source[0]["freq"])
        seasonality = int(get_seasonality(frequency))
        period = int(workflow.retrieval_period or seasonality)
        minimum_length = min(
            int(np.asarray(source[index]["target"]).shape[-1])
            for index in range(len(source))
        )
        for term in _selected_terms(
            dataset_name, dataset_config, terms_selected
        ):
            settings = get_dataset_settings(dataset_name, term, dataset_config)
            horizon = _prediction_length(
                dataset_name, term, settings.get("prediction_length"), frequency
            )
            stride = int(
                workflow.adaptation_stride
                or adaptation_stride_for_frequency(frequency)
            )
            train_length, validation_length, _ = adaptation_split_lengths(
                int(settings["test_length"]), horizon, stride
            )
            tasks.append(
                AdaptimeTask(
                    dataset=dataset_name,
                    term=term,
                    frequency=frequency,
                    source_path=source_path,
                    minimum_series_length=minimum_length,
                    dataset_fingerprint=str(source._fingerprint),
                    preparation=PreparationConfig(
                        dataset=dataset_name,
                        term=term,
                        context_length=context_length,
                        prediction_length=horizon,
                        test_length=int(settings["test_length"]),
                        adaptation_train_length=train_length,
                        adaptation_validation_length=validation_length,
                        seasonality=seasonality,
                        target_mode=workflow.target_mode,
                        adaptation_stride=stride,
                        retrieval_period=period,
                        datastore_stride=(
                            period * int(workflow.datastore_stride_multiple)
                        ),
                        max_datastore_windows=workflow.max_datastore_windows,
                        datastore_prediction_length=max(
                            horizon, TSRAG_NATIVE_HORIZON
                        ),
                        minimum_datastore_dates_per_variate=TSRAG_TOP_K + 1,
                    ),
                )
            )
    return tasks


def _task_identity(task: AdaptimeTask, model: str) -> dict[str, object]:
    return {
        "model": model,
        "target_mode": task.preparation.target_mode,
        "dataset": task.dataset.rpartition("/")[0] or task.dataset,
        "frequency": task.frequency,
        "term": task.term,
    }


def _preparation_config(task: AdaptimeTask) -> dict[str, object]:
    values = asdict(task.preparation)
    values["adaptation_stride"] = task.preparation.query_stride
    values["datastore_prediction_length"] = task.preparation.datastore_horizon
    values["dataset_fingerprint"] = task.dataset_fingerprint
    return values


def _extraction_config(workflow: AdaptimeWorkflowConfig) -> ExtractionConfig:
    return ExtractionConfig(
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
    )


def _stage_root(root: Path, stage: str, method: str, task: AdaptimeTask) -> Path:
    return (
        root
        / stage
        / method
        / task.preparation.target_mode
        / task.dataset
        / task.term
    )


def _spec(
    task: AdaptimeTask,
    workflow: AdaptimeWorkflowConfig,
    stage: str,
    method: str,
) -> tuple[str, dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    data_config = _preparation_config(task)
    if stage == "data":
        return (
            "adaptime_data",
            _task_identity(task, "shared_datastore"),
            {"artifact": "shared_global_datastore"},
            data_config,
            {
                "splits": ["datastore", "adaptation_train", "adaptation_validation", "test"],
                "consumers": ["full_ridge_shared", "tsrag"],
            },
        )
    if stage == "vanilla":
        return (
            "adaptime_vanilla",
            _task_identity(task, workflow.model),
            {
                "model": workflow.model,
                "weights_id": workflow.weights_id,
                "context_policy": "all_available_history_capped_at_model_limit",
            },
            {"data_config": data_config},
            {"phase": "unconditional_official_test_vanilla"},
        )
    ridge_family = method == "ridge" or method in ADAPTATION_METHODS
    if ridge_family:
        method_name = "full_ridge_shared"
        extraction_science = {
            "backbone": workflow.model,
            "weights_id": workflow.weights_id,
            **asdict(_extraction_config(workflow)),
        }
    else:
        method_name = "tsrag"
        extraction_science = {
            "source_commit": TSRAG_SOURCE_COMMIT,
            "context_length": TSRAG_CONTEXT_LENGTH,
            "native_prediction_length": TSRAG_NATIVE_HORIZON,
            "top_k": TSRAG_TOP_K,
            "embedding": "chronos_t5_base_eos",
            "embedding_dimension": TSRAG_EMBEDDING_DIMENSION,
            "retrieval_scope": "same_series",
        }
    if stage == "extractions":
        return (
            "adaptime_extraction",
            _task_identity(task, method_name),
            extraction_science,
            {"data_config": data_config},
            {"phase": "fit_grid_extraction"},
        )
    if stage == "adaptations":
        return (
            "adaptime_adaptation",
            _task_identity(task, method_name),
            {
                "method": method_name,
                **asdict(workflow.ridge_training),
                "bayes_covariate_protocol": {
                    "evidence": "paired_train_validation_window_msse_wins",
                    "tie_weight": 0.5,
                    "prior": "beta_1_1",
                },
            },
            {
                "data_config": data_config,
                "extraction": extraction_science,
                "fit_extraction_contract": "fixed_context_train_validation_only",
            },
            {"phase": "closed_form_fitting"},
        )
    if stage == "eval_extractions":
        return (
            "adaptime_eval_extraction",
            _task_identity(task, method_name),
            extraction_science,
            {
                "data_config": data_config,
                "adaptation": asdict(workflow.ridge_training),
                "vanilla_context_policy": "all_available_history_capped_at_model_limit",
            },
            {"phase": "selected_k_official_test_extraction"},
        )
    if stage == "predictions":
        model_config: dict[str, object] = {
            "method": (
                "adaptime_comparison_family" if ridge_family else method_name
            ),
            "extraction": extraction_science,
        }
        if ridge_family:
            model_config["methods"] = list(ADAPTATION_METHODS)
            model_config["adaptation"] = asdict(workflow.ridge_training)
            model_config["prediction_protocol"] = (
                "unconditional_vanilla_plus_selected_k_covariate_bayes_and_ridge"
            )
        else:
            model_config["checkpoint"] = "released_tsrag_arm"
        return (
            "adaptime_prediction",
            _task_identity(task, method_name),
            model_config,
            {"data_config": data_config},
            {"phase": "frozen_inference", "forecast_type": "point"},
        )
    if stage == "evaluations":
        evaluation_method = method if method in ADAPTATION_METHODS else method_name
        prediction_science: dict[str, object] = {
            "method": (
                "adaptime_comparison_family" if ridge_family else method_name
            ),
            "extraction": extraction_science,
        }
        if ridge_family:
            prediction_science["methods"] = list(ADAPTATION_METHODS)
            prediction_science["adaptation"] = asdict(workflow.ridge_training)
            prediction_science["prediction_protocol"] = (
                "unconditional_vanilla_plus_selected_k_covariate_bayes_and_ridge"
            )
        else:
            prediction_science["checkpoint"] = "released_tsrag_arm"
        return (
            "adaptime_evaluation",
            _task_identity(task, evaluation_method),
            {"method": evaluation_method, "forecast_type": "point"},
            {
                "data_config": data_config,
                "prediction_config": prediction_science,
                "evaluator": "timebench.evaluation.saver.save_window_predictions",
            },
            {
                "phase": "evaluation",
                "quantile_levels": [0.5],
                "metrics": ["MSE", "MAE", "RMSE", "MAPE", "sMAPE", "MASE", "ND", "CRPS"],
            },
        )
    raise ValueError(f"unknown artifact stage {stage!r}")


def _completed_run(
    root: Path,
    spec: tuple[str, dict[str, object], dict[str, object], dict[str, object], dict[str, object]],
) -> Path:
    experiment, identity, model_config, pipeline_config, experiment_config = (
        spec[0],
        *(json.loads(json.dumps(value)) for value in spec[1:]),
    )
    candidates = select_completed_runs(
        root,
        config_policy="distinct",
        repeat_policy="selected",
    )
    matches = [
        path
        for path, manifest in candidates
        if manifest.get("experiment") == experiment
        and manifest.get("identity") == identity
        and manifest.get("model_config") == model_config
        and manifest.get("pipeline_config") == pipeline_config
        and manifest.get("experiment_config") == experiment_config
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one completed exact {experiment} input below {root}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _matching_evaluation_runs(
    root: Path,
    tasks: Iterable[AdaptimeTask],
    workflow: AdaptimeWorkflowConfig,
    method: str,
) -> list[Path] | None:
    """Return exact completed evaluations, or fail closed for reuse."""

    try:
        return [
            _completed_run(root, _spec(task, workflow, "evaluations", method))
            for task in tasks
        ]
    except (FileNotFoundError, ValueError):
        return None


def _allocation(
    artifact_root: Path,
    task: AdaptimeTask,
    workflow: AdaptimeWorkflowConfig,
    stage: str,
    method: str,
    *,
    runtime_config: dict[str, object],
    provenance: dict[str, object],
):
    spec = _spec(task, workflow, stage, method)
    experiment, identity, model_config, pipeline_config, experiment_config = (
        spec[0],
        *(json.loads(json.dumps(value)) for value in spec[1:]),
    )
    return allocate_run(
        _stage_root(artifact_root, stage, method, task),
        experiment=experiment,
        identity=identity,
        model_config=model_config,
        pipeline_config=pipeline_config,
        runtime_config=runtime_config,
        experiment_config=experiment_config,
        provenance=provenance,
    )


def _data_manifest(
    artifact_root: Path, task: AdaptimeTask, workflow: AdaptimeWorkflowConfig
) -> Path:
    root = _stage_root(artifact_root, "data", "shared", task)
    run = _completed_run(root, _spec(task, workflow, "data", "shared"))
    return run / "prepared" / "manifest.json"


def _vanilla_manifest(
    artifact_root: Path, task: AdaptimeTask, workflow: AdaptimeWorkflowConfig
) -> Path:
    root = _stage_root(artifact_root, "vanilla", "shared", task)
    run = _completed_run(root, _spec(task, workflow, "vanilla", "shared"))
    return run / "vanilla" / "manifest.json"


def _artifact_manifest(
    artifact_root: Path,
    task: AdaptimeTask,
    workflow: AdaptimeWorkflowConfig,
    stage: str,
    method: str,
    relative: str,
) -> Path:
    root = _stage_root(artifact_root, stage, method, task)
    run = _completed_run(root, _spec(task, workflow, stage, method))
    return run / relative


def _run_prepare(
    artifact_root: Path, task: AdaptimeTask, workflow: AdaptimeWorkflowConfig
) -> Path:
    run = _allocation(
        artifact_root,
        task,
        workflow,
        "data",
        "shared",
        runtime_config={},
        provenance={
            "dataset_source": str(task.source_path),
            "minimum_series_length": task.minimum_series_length,
            "dataset_fingerprint": task.dataset_fingerprint,
        },
    )
    manifest = run.run_dir / "prepared" / "manifest.json"
    if not run.should_run:
        return manifest
    source = datasets.load_from_disk(str(task.source_path))
    with run:
        manifest = prepare_adaptation_dataset(
            source,
            task.preparation,
            run.run_dir / "prepared",
            source_path=task.source_path,
        )
        run.complete(["prepared/manifest.json"])
    return manifest


def _run_vanilla(
    artifact_root: Path, task: AdaptimeTask, workflow: AdaptimeWorkflowConfig
) -> Path:
    prepared = _data_manifest(artifact_root, task, workflow)
    run = _allocation(
        artifact_root,
        task,
        workflow,
        "vanilla",
        "shared",
        runtime_config={
            "device": workflow.device,
            "model_path": None if workflow.model_path is None else str(workflow.model_path),
        },
        provenance={"data_manifest": str(prepared)},
    )
    manifest = run.run_dir / "vanilla" / "manifest.json"
    if not run.should_run:
        return manifest
    forecaster = load_adaptime_forecaster(
        workflow.model,
        horizon=task.preparation.prediction_length,
        period=task.preparation.retrieval_period,
        model_path=workflow.model_path,
        weights_id=workflow.weights_id,
        device=workflow.device,
    )
    with run:
        manifest = extract_vanilla_test_forecasts(
            prepared,
            forecaster,
            VanillaConfig(
                model_batch_size=workflow.model_batch_size,
                arrow_cache_items=workflow.arrow_cache_items,
            ),
            run.run_dir / "vanilla",
        )
        run.complete(
            [
                "vanilla/manifest.json",
                "vanilla/predictions.npy",
                "vanilla/context_length.npy",
            ]
        )
    return manifest


def _run_ridge_extraction(
    artifact_root: Path, task: AdaptimeTask, workflow: AdaptimeWorkflowConfig
) -> Path:
    prepared = _data_manifest(artifact_root, task, workflow)
    run = _allocation(
        artifact_root,
        task,
        workflow,
        "extractions",
        "ridge",
        runtime_config={
            "device": workflow.device,
            "model_path": None if workflow.model_path is None else str(workflow.model_path),
        },
        provenance={"data_manifest": str(prepared)},
    )
    manifest = run.run_dir / "extraction" / "manifest.json"
    if not run.should_run:
        return manifest
    forecaster = load_adaptime_forecaster(
        workflow.model,
        horizon=task.preparation.prediction_length,
        period=task.preparation.retrieval_period,
        model_path=workflow.model_path,
        weights_id=workflow.weights_id,
        device=workflow.device,
    )
    with run:
        manifest = extract_adaptation_features(
            prepared,
            forecaster,
            _extraction_config(workflow),
            run.run_dir / "extraction",
        )
        run.complete(["extraction/manifest.json"])
    return manifest


def _tsrag_paths(workflow: AdaptimeWorkflowConfig) -> tuple[Path, Path, Path]:
    root = weights_root()
    return (
        (workflow.tsrag_chronos_bolt_path or root / "chronos-bolt-base").expanduser().resolve(),
        (workflow.tsrag_retriever_path or root / "chronos-t5-base").expanduser().resolve(),
        (workflow.tsrag_checkpoint_path or root / "ts-rag").expanduser().resolve(),
    )


def _run_tsrag_extraction(
    artifact_root: Path,
    task: AdaptimeTask,
    workflow: AdaptimeWorkflowConfig,
    retriever: TSRAGRetriever,
) -> Path:
    prepared = _data_manifest(artifact_root, task, workflow)
    _, retriever_path, _ = _tsrag_paths(workflow)
    run = _allocation(
        artifact_root,
        task,
        workflow,
        "extractions",
        "tsrag",
        runtime_config={
            **asdict(workflow.tsrag_runtime),
            "device": workflow.device,
            "retriever_path": str(retriever_path),
        },
        provenance={
            "data_manifest": str(prepared),
            "upstream_repository": "https://github.com/UConn-DSIS/TS-RAG",
            "upstream_commit": TSRAG_SOURCE_COMMIT,
        },
    )
    manifest = run.run_dir / "extraction" / "manifest.json"
    if not run.should_run:
        return manifest
    with run:
        manifest = extract_tsrag_features(
            prepared,
            retriever,
            workflow.tsrag_runtime,
            run.run_dir / "extraction",
        )
        run.complete(["extraction/manifest.json"])
    return manifest


def _run_fit(
    artifact_root: Path, task: AdaptimeTask, workflow: AdaptimeWorkflowConfig
) -> Path:
    prepared = _data_manifest(artifact_root, task, workflow)
    extraction = _artifact_manifest(
        artifact_root, task, workflow, "extractions", "ridge", "extraction/manifest.json"
    )
    run = _allocation(
        artifact_root,
        task,
        workflow,
        "adaptations",
        "ridge",
        runtime_config={"chunk_size": workflow.ridge_chunk_size},
        provenance={
            "data_manifest": str(prepared),
            "extraction_manifest": str(extraction),
        },
    )
    manifest = run.run_dir / "model" / "model_manifest.json"
    if not run.should_run:
        return manifest
    with run:
        manifest = fit_full_ridge(
            prepared,
            extraction,
            workflow.ridge_training,
            run.run_dir / "model",
        )
        required = [
            "model/model_manifest.json",
            "model/selection.json",
            "model/bayes_mixture.json",
        ]
        if "coefficients" in json.loads(manifest.read_text(encoding="utf-8"))["files"]:
            required.append("model/coefficients.npy")
        run.complete(required)
    return manifest


def _run_ridge_eval_extraction(
    artifact_root: Path, task: AdaptimeTask, workflow: AdaptimeWorkflowConfig
) -> Path:
    prepared = _data_manifest(artifact_root, task, workflow)
    vanilla = _vanilla_manifest(artifact_root, task, workflow)
    extraction = _artifact_manifest(
        artifact_root, task, workflow, "extractions", "ridge", "extraction/manifest.json"
    )
    adaptation = _artifact_manifest(
        artifact_root, task, workflow, "adaptations", "ridge", "model/model_manifest.json"
    )
    run = _allocation(
        artifact_root,
        task,
        workflow,
        "eval_extractions",
        "ridge",
        runtime_config={
            "device": workflow.device,
            "model_path": None if workflow.model_path is None else str(workflow.model_path),
        },
        provenance={
            "data_manifest": str(prepared),
            "vanilla_manifest": str(vanilla),
            "fit_extraction_manifest": str(extraction),
            "adaptation_manifest": str(adaptation),
        },
    )
    manifest = run.run_dir / "extraction" / "manifest.json"
    if not run.should_run:
        return manifest
    forecaster = load_adaptime_forecaster(
        workflow.model,
        horizon=task.preparation.prediction_length,
        period=task.preparation.retrieval_period,
        model_path=workflow.model_path,
        weights_id=workflow.weights_id,
        device=workflow.device,
    )
    with run:
        manifest = extract_adaptation_eval_features(
            prepared,
            extraction,
            adaptation,
            vanilla,
            forecaster,
            _extraction_config(workflow),
            run.run_dir / "extraction",
        )
        run.complete(["extraction/manifest.json"])
    return manifest


def _run_ridge_prediction(
    artifact_root: Path, task: AdaptimeTask, workflow: AdaptimeWorkflowConfig
) -> Path:
    prepared = _data_manifest(artifact_root, task, workflow)
    extraction = _artifact_manifest(
        artifact_root, task, workflow, "extractions", "ridge", "extraction/manifest.json"
    )
    eval_extraction = _artifact_manifest(
        artifact_root,
        task,
        workflow,
        "eval_extractions",
        "ridge",
        "extraction/manifest.json",
    )
    adaptation = _artifact_manifest(
        artifact_root, task, workflow, "adaptations", "ridge", "model/model_manifest.json"
    )
    vanilla = _vanilla_manifest(artifact_root, task, workflow)
    run = _allocation(
        artifact_root,
        task,
        workflow,
        "predictions",
        "ridge",
        runtime_config={"chunk_size": workflow.ridge_chunk_size},
        provenance={
            "data_manifest": str(prepared),
            "fit_extraction_manifest": str(extraction),
            "eval_extraction_manifest": str(eval_extraction),
            "adaptation_manifest": str(adaptation),
            "vanilla_manifest": str(vanilla),
        },
    )
    manifest = run.run_dir / "prediction" / "prediction_manifest.json"
    if not run.should_run:
        return manifest
    with run:
        manifest = predict_adaptation_family(
            prepared,
            extraction,
            adaptation,
            eval_extraction,
            vanilla,
            PredictionConfig(chunk_size=workflow.ridge_chunk_size),
            run.run_dir / "prediction",
        )
        run.complete(
            [
                "prediction/prediction_manifest.json",
                *(
                    f"prediction/{method}.npy"
                    for method in ADAPTATION_METHODS
                ),
                "prediction/rag_eligible.npy",
                "prediction/fallback_reason.npy",
            ]
        )
    return manifest


def _run_tsrag_prediction(
    artifact_root: Path,
    task: AdaptimeTask,
    workflow: AdaptimeWorkflowConfig,
    retriever: TSRAGRetriever,
    loaded: LoadedTSRAG,
) -> Path:
    prepared = _data_manifest(artifact_root, task, workflow)
    extraction = _artifact_manifest(
        artifact_root, task, workflow, "extractions", "tsrag", "extraction/manifest.json"
    )
    base_path, retriever_path, checkpoint_path = _tsrag_paths(workflow)
    run = _allocation(
        artifact_root,
        task,
        workflow,
        "predictions",
        "tsrag",
        runtime_config={
            **asdict(workflow.tsrag_runtime),
            "device": workflow.device,
            "chronos_bolt_path": str(base_path),
            "retriever_path": str(retriever_path),
            "checkpoint_path": str(checkpoint_path),
        },
        provenance={
            "data_manifest": str(prepared),
            "extraction_manifest": str(extraction),
        },
    )
    manifest = run.run_dir / "prediction" / "prediction_manifest.json"
    if not run.should_run:
        return manifest
    with run:
        manifest = predict_tsrag(
            prepared,
            extraction,
            loaded,
            retriever,
            workflow.tsrag_runtime,
            run.run_dir / "prediction",
            device=workflow.device,
        )
        run.complete(["prediction/prediction_manifest.json", "prediction/predictions.npy"])
    return manifest


def _run_evaluation(
    artifact_root: Path,
    task: AdaptimeTask,
    workflow: AdaptimeWorkflowConfig,
    method: str,
) -> Path:
    prepared = _data_manifest(artifact_root, task, workflow)
    prediction_method = "ridge" if method in ADAPTATION_METHODS else method
    prediction = _artifact_manifest(
        artifact_root,
        task,
        workflow,
        "predictions",
        prediction_method,
        "prediction/prediction_manifest.json",
    )
    run = _allocation(
        artifact_root,
        task,
        workflow,
        "evaluations",
        method,
        runtime_config={},
        provenance={
            "data_manifest": str(prepared),
            "prediction_manifest": str(prediction),
        },
    )
    if not run.should_run:
        return run.run_dir / "metrics_summary.json"
    with run:
        evaluate_point_predictions(
            prepared,
            prediction,
            run.run_dir,
            method=method if method in ADAPTATION_METHODS else None,
        )
        run.complete(["predictions.npz", "metrics.npz", "config.json", "metrics_summary.json"])
    return run.run_dir / "metrics_summary.json"


def run_adaptation_stage(
    stage: str,
    method: str,
    workflow: AdaptimeWorkflowConfig,
    *,
    dataset_config_path: Path | None = None,
    datasets_selected: Iterable[str] = ("all_datasets",),
    terms_selected: Iterable[str] | None = None,
    output_root: Path | None = None,
    ridge_results_path: Path | None = None,
    config_policy: str = "error",
    repeat_policy: str = "selected",
) -> list[Path]:
    """Run one explicit phase, or a method pipeline after shared preparation."""

    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    workflow.validate(method)
    if stage in {"vanilla", "fit", "extract_eval"} and method != "ridge":
        raise ValueError(
            "vanilla, fit, and extract_eval are Ridge phases; "
            "TS-RAG retains its native frozen pipeline"
        )
    dataset_config = load_dataset_config(dataset_config_path)
    artifact_root = (output_root or outputs_root() / "adaptime").expanduser().resolve()
    tasks = workflow_tasks(
        dataset_config, datasets_selected, terms_selected, workflow
    )
    local_ridge_root = artifact_root / "evaluations" / "full_ridge_shared"
    matched_ridge_runs = (
        _matching_evaluation_runs(
            ridge_results_path.expanduser().resolve(),
            tasks,
            workflow,
            "full_ridge_shared",
        )
        if ridge_results_path is not None
        and stage == "report"
        and method == "tsrag"
        else None
    )
    if stage == "report":
        launch_id = os.environ.get("TIME_LAUNCH_ID") or "manual"
        if method == "ridge":
            roots = {
                comparison_method: artifact_root
                / "evaluations"
                / comparison_method
                for comparison_method in ADAPTATION_METHODS
            }
        else:
            ridge_root = (
                ridge_results_path.expanduser().resolve()
                if matched_ridge_runs is not None and ridge_results_path is not None
                else local_ridge_root
            )
            if ridge_results_path is not None and matched_ridge_runs is None:
                print(
                    "Precomputed Ridge results do not exactly match every requested "
                    f"task; reporting local results from {local_ridge_root}.",
                    flush=True,
                )
            roots = {
                "full_ridge_shared": ridge_root,
                "tsrag": artifact_root / "evaluations" / "tsrag",
            }
        report = build_adaptation_comparison(
            method_results_roots=roots,
            output_dir=artifact_root / "reports" / launch_id,
            expected_tasks=((task.dataset, task.term) for task in tasks),
            config_policy=config_policy,
            repeat_policy=repeat_policy,
        )
        print(report, flush=True)
        return [report]
    stages = {
        "pipeline": (
            "vanilla",
            "extract",
            "fit",
            "extract_eval",
            "predict",
            "evaluate",
        )
        if method == "ridge"
        else ("extract", "predict", "evaluate"),
        "all": (
            "prepare",
            "vanilla",
            "extract",
            "fit",
            "extract_eval",
            "predict",
            "evaluate",
        )
        if method == "ridge"
        else ("prepare", "extract", "predict", "evaluate"),
    }.get(stage, (stage,))

    retriever: TSRAGRetriever | None = None
    loaded: LoadedTSRAG | None = None
    if method == "tsrag" and any(value in stages for value in ("extract", "predict")):
        if "prepare" not in stages:
            for task in tasks:
                _data_manifest(artifact_root, task, workflow)
                if "predict" in stages and "extract" not in stages:
                    _artifact_manifest(
                        artifact_root,
                        task,
                        workflow,
                        "extractions",
                        "tsrag",
                        "extraction/manifest.json",
                    )
        base_path, retriever_path, checkpoint_path = _tsrag_paths(workflow)
        retriever = TSRAGRetriever(
            retriever_path,
            device_map=workflow.device,
            local_files_only=True,
        )
        if "predict" in stages:
            loaded = load_tsrag(base_path, checkpoint_path, device=workflow.device)

    outputs: list[Path] = []
    for current in stages:
        for task in tasks:
            if current == "prepare":
                result = _run_prepare(artifact_root, task, workflow)
            elif current == "vanilla":
                result = _run_vanilla(artifact_root, task, workflow)
            elif current == "extract" and method == "ridge":
                result = _run_ridge_extraction(artifact_root, task, workflow)
            elif current == "extract":
                assert retriever is not None
                result = _run_tsrag_extraction(
                    artifact_root, task, workflow, retriever
                )
            elif current == "fit":
                result = _run_fit(artifact_root, task, workflow)
            elif current == "extract_eval":
                result = _run_ridge_eval_extraction(artifact_root, task, workflow)
            elif current == "predict" and method == "ridge":
                result = _run_ridge_prediction(artifact_root, task, workflow)
            elif current == "predict":
                assert retriever is not None and loaded is not None
                result = _run_tsrag_prediction(
                    artifact_root, task, workflow, retriever, loaded
                )
            elif current == "evaluate" and method == "ridge":
                for comparison_method in ADAPTATION_METHODS:
                    result = _run_evaluation(
                        artifact_root,
                        task,
                        workflow,
                        comparison_method,
                    )
                    outputs.append(result)
                    print(result, flush=True)
                continue
            else:
                result = _run_evaluation(
                    artifact_root, task, workflow, method
                )
            outputs.append(result)
            print(result, flush=True)
    return outputs
