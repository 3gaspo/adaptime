"""Benchmark independent single-window inference without test-artifact reuse."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np

from timebench.adaptime.retrieval import blockwise_topk
from timebench.adaptime.ridge import (
    PER_VARIATE_RIDGE,
    RIDGE_VARIANTS,
    full_ridge_design,
    full_ridge_predict_with_fallback,
    ridge_feature_indices,
)
from timebench.evaluation.adaptation_data import PreparedDataset
from timebench.evaluation.timing import EvaluationTimer
from timebench.external_models.tsrag.retriever import TSRAGRetriever
from timebench.model_loading.adaptime import MODEL_ALIASES, load_adaptime_forecaster
from timebench.model_loading.tsrag import load_tsrag
from timebench.paths import outputs_root, weights_root
from timebench.pipeline.adaptime_extraction import (
    ExtractionConfig,
    _forecast,
    _query_scaled_retrieval_context,
    _represent,
    _source_eligibility,
    open_extraction,
)
from timebench.pipeline.adaptime_training import open_adaptation_model
from timebench.pipeline.tsrag import (
    _build_indexes,
    _representation as tsrag_representation,
    _rollout_tsrag,
    _search as tsrag_search,
    open_tsrag_extraction,
)
from timebench.pipeline.tsrag_data import TSRAGPreparedDataset


HEADLINE_METHODS = (
    "vanilla",
    "covariate_prediction",
    "bayes_covariate_prediction",
    "full_ridge_shared",
    "tsrag",
)
ADAPTIME_METHODS = (
    "vanilla",
    "covariate_prediction",
    "bayes_covariate_prediction",
    "bayes_past_targets_prediction",
    *RIDGE_VARIANTS,
    PER_VARIATE_RIDGE,
    "selected_adaptation",
)
METHODS = (*ADAPTIME_METHODS, "tsrag")
SCHEMA_VERSION = 1


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def _run_number(path: Path) -> int:
    for parent in (path, *path.parents):
        if parent.name.startswith("run_") and parent.name[4:].isdigit():
            return int(parent.name[4:])
    return -1


def _completed_artifacts(paths: list[Path]) -> list[Path]:
    completed: list[Path] = []
    for path in paths:
        run_dir = next(
            (
                parent
                for parent in (path, *path.parents)
                if parent.name.startswith("run_") and parent.name[4:].isdigit()
            ),
            None,
        )
        if run_dir is None or not (run_dir / "manifest.json").is_file():
            continue
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") == "completed":
            completed.append(path)
    return completed


def _latest(paths: list[Path], description: str) -> Path:
    if not paths:
        raise FileNotFoundError(f"no compatible completed {description} artifact found")
    return max(paths, key=_run_number)


def _resolve_task_inputs(args: argparse.Namespace) -> None:
    """Resolve the newest signature-compatible task chain for Slurm shorthand."""

    if args.prepared is not None:
        return
    if args.artifact_root is None or args.dataset is None or args.term is None:
        raise ValueError(
            "supply --prepared and exact artifact paths, or --artifact-root, "
            "--dataset, and --term"
        )
    task = Path(args.dataset) / args.term
    root = Path(args.artifact_root).expanduser().resolve()
    prepared_paths = _completed_artifacts(
        list(
            (root / "data" / "shared" / "univariate" / task).glob(
                "run_*/prepared/manifest.json"
            )
        )
    )
    prepared_by_signature = {
        PreparedDataset(path).signature: path for path in prepared_paths
    }
    adapted = any(method not in {"vanilla", "tsrag"} for method in args.methods)
    if adapted:
        fit_paths = _completed_artifacts(
            list(
                (root / "extractions" / "ridge" / "univariate" / task).glob(
                    "run_*/extraction/manifest.json"
                )
            )
        )
        fits: dict[str, tuple[Path, dict[str, Any]]] = {}
        for path in fit_paths:
            _, manifest = open_extraction(path)
            if (
                manifest["prepared_signature"] in prepared_by_signature
                and manifest["model"] == args.model
                and (
                    args.weights_id is None
                    or manifest["weights_id"] == args.weights_id
                )
            ):
                previous = fits.get(str(manifest["signature"]))
                if previous is None or _run_number(path) > _run_number(previous[0]):
                    fits[str(manifest["signature"])] = (path, manifest)
        model_paths = _completed_artifacts(
            list(
                (root / "adaptations" / "ridge" / "univariate" / task).glob(
                    "run_*/model/model_manifest.json"
                )
            )
        )
        chains: list[tuple[Path, Path, Path, dict[str, Any]]] = []
        for model_path in model_paths:
            _, model = open_adaptation_model(model_path)
            fit = fits.get(str(model["extraction_signature"]))
            if fit is None:
                continue
            fit_path, fit_manifest = fit
            prepared_path = prepared_by_signature[fit_manifest["prepared_signature"]]
            chains.append((prepared_path, fit_path, model_path, fit_manifest))
        if not chains:
            raise FileNotFoundError(
                f"no compatible completed Ridge artifact chain below {root / task}"
            )
        prepared_path, fit_path, model_path, _ = max(
            chains,
            key=lambda chain: tuple(_run_number(path) for path in chain[:3]),
        )
        args.prepared = prepared_path
        args.fit_extraction = args.fit_extraction or fit_path
        args.adaptation_model = args.adaptation_model or model_path
    else:
        args.prepared = _latest(prepared_paths, "prepared-data")

    if "tsrag" in args.methods and args.tsrag_extraction is None:
        prepared_signature = PreparedDataset(args.prepared).signature
        candidates: list[Path] = []
        tsrag_paths = _completed_artifacts(
            list(
                (
                    root / "extractions" / "tsrag" / "univariate" / task
                ).glob("run_*/extraction/manifest.json")
            )
        )
        for path in tsrag_paths:
            _, manifest = open_tsrag_extraction(path)
            if manifest["prepared_signature"] == prepared_signature:
                candidates.append(path)
        args.tsrag_extraction = _latest(candidates, "TS-RAG extraction")


def _timed(call: Callable[[], Any], *, accelerator: bool = False) -> tuple[Any, float]:
    if accelerator:
        timer = EvaluationTimer()
        timer.start()
        value = call()
        return value, timer.stop()
    started = perf_counter()
    return call(), perf_counter() - started


def _sample_positions(references: np.ndarray, samples: int, seed: int) -> np.ndarray:
    """Sample dates without replacement while balancing item/channel series."""

    refs = np.asarray(references, dtype=np.int64).reshape(-1, 3)
    if int(samples) <= 0:
        raise ValueError("samples must be positive")
    if int(samples) > len(refs):
        raise ValueError(f"samples={samples} exceeds {len(refs)} test references")
    rng = np.random.default_rng(int(seed))
    grouped: dict[tuple[int, int], list[int]] = {}
    for position, (item, channel, _) in enumerate(refs):
        grouped.setdefault((int(item), int(channel)), []).append(position)
    remaining = {
        key: list(map(int, rng.permutation(positions)))
        for key, positions in grouped.items()
    }
    selected: list[int] = []
    while len(selected) < int(samples):
        active = [key for key, positions in remaining.items() if positions]
        for index in rng.permutation(len(active)):
            selected.append(remaining[active[int(index)]].pop())
            if len(selected) == int(samples):
                break
    return np.asarray(selected, dtype=np.int64)


def _extraction_config(manifest: dict[str, Any]) -> ExtractionConfig:
    values = dict(manifest["config"])
    values["context_k"] = tuple(map(int, values["context_k"]))
    names = {field.name for field in fields(ExtractionConfig)}
    config = ExtractionConfig(**{name: values[name] for name in names})
    config.validate()
    return config


def _selected_method(
    requested: str, model: dict[str, Any]
) -> tuple[str, int, int]:
    """Return executed method, its K, and the common eligibility-support K."""

    method = str(model["selected"]["method"]) if requested == "selected_adaptation" else requested
    selected_ks = tuple(map(int, model.get("evaluation_k_values", ())))
    support_k = max(selected_ks, default=0)
    if method == "vanilla":
        return method, 0, support_k
    selections = dict(model.get("method_selections", {}))
    if method == "covariate_prediction":
        if not selected_ks:
            return "vanilla", 0, 0
        k = int(model["selected"]["k"])
        if k == 0:
            bayes_k = int(selections["bayes_covariate_prediction"]["k"])
            k = bayes_k if bayes_k > 0 else selected_ks[0]
        return method, k, support_k
    selection = dict(selections[method])
    if selection["method"] == "vanilla":
        return "vanilla", 0, support_k
    return method, int(selection.get("k", 0)), support_k


def _load_forecaster(args: argparse.Namespace, prepared: PreparedDataset) -> tuple[Any, float]:
    return _timed(
        lambda: load_adaptime_forecaster(
            args.model,
            horizon=prepared.prediction_length,
            period=int(prepared.config["retrieval_period"]),
            model_path=args.model_path,
            weights_id=args.weights_id,
            device=args.device,
        ),
        accelerator=True,
    )


def _vanilla_forecast(
    prepared: PreparedDataset,
    forecaster: Any,
    reader: Any,
    reference: np.ndarray,
    components: dict[str, float],
) -> np.ndarray:
    context_length = min(int(reference[0, 2]), prepared.context_length)
    batch, components["input_read_seconds"] = _timed(
        lambda: reader.read(reference, context_length=context_length)
    )
    values, components["vanilla_forecast_seconds"] = _timed(
        lambda: _forecast(
            forecaster, batch.context, horizon=prepared.prediction_length
        ),
        accelerator=True,
    )
    return values


def _forecast_references_with_available_context(
    prepared: PreparedDataset,
    forecaster: Any,
    reader: Any,
    references: np.ndarray,
) -> np.ndarray:
    refs = np.asarray(references, dtype=np.int64).reshape(-1, 3)
    lengths = np.minimum(refs[:, 2], prepared.context_length)
    result: np.ndarray | None = None
    for length in np.unique(lengths):
        positions = np.flatnonzero(lengths == length)
        batch = reader.read(refs[positions], context_length=int(length))
        values = _forecast(
            forecaster,
            batch.context,
            horizon=prepared.prediction_length,
        )
        if result is None:
            result = np.empty((len(refs), *values.shape[1:]), dtype=np.float32)
        result[positions] = values
    if result is None:
        raise ValueError("cannot forecast empty references")
    return result


def _load_adaptation_state(
    args: argparse.Namespace, prepared: PreparedDataset
) -> tuple[Path, dict[str, Any], Path, dict[str, Any], ExtractionConfig]:
    fit_root, extraction = open_extraction(args.fit_extraction)
    model_root, model = open_adaptation_model(args.adaptation_model)
    if extraction["prepared_signature"] != prepared.signature:
        raise ValueError("fit extraction and prepared test references do not match")
    if model["extraction_signature"] != extraction["signature"]:
        raise ValueError("frozen adaptor and fit extraction do not match")
    return fit_root, extraction, model_root, model, _extraction_config(extraction)


def _past_target_prediction(
    prepared: PreparedDataset,
    forecaster: Any,
    reader: Any,
    reference: np.ndarray,
    vanilla: np.ndarray,
    model_root: Path,
    model: dict[str, Any],
    components: dict[str, float],
) -> tuple[np.ndarray, bool, str]:
    item = int(reference[0, 0])
    context_length = min(int(reference[0, 2]), prepared.context_length)
    if (
        context_length <= 0
        or reader.target_channels(item) < 2
        or not getattr(forecaster, "supports_past_covariates", False)
    ):
        return vanilla, False, "past_target_covariates_unavailable"
    started = perf_counter()
    batch = reader.read(reference, context_length=context_length)
    covariates = reader.read_other_variates_as_past_covariates(
        reference, context_length=context_length
    )
    components["adaptation_input_read_seconds"] = perf_counter() - started
    if not (
        np.isfinite(batch.context).all() and np.isfinite(covariates).all()
    ):
        return vanilla, False, "past_target_covariates_unavailable"
    forecast, components["past_target_forecast_seconds"] = _timed(
        lambda: _forecast(
            forecaster,
            batch.context,
            horizon=prepared.prediction_length,
            past_covariates=covariates,
        ),
        accelerator=True,
    )
    if not np.isfinite(forecast).all():
        return vanilla, False, "past_target_covariates_unavailable"
    mixture = json.loads(
        (model_root / model["files"]["bayes_past_targets_mixture"]).read_text(
            encoding="utf-8"
        )
    )
    probability = float(mixture["probability_past_targets_better"])
    prediction, components["adaptation_seconds"] = _timed(
        lambda: (1.0 - probability) * vanilla + probability * forecast
    )
    return prediction, True, "rag_eligible"


def _rag_prediction(
    prepared: PreparedDataset,
    forecaster: Any,
    reader: Any,
    reference: np.ndarray,
    vanilla: np.ndarray,
    method: str,
    k: int,
    support_k: int,
    fit_root: Path,
    extraction: dict[str, Any],
    model_root: Path,
    model: dict[str, Any],
    config: ExtractionConfig,
    reference_position: int,
    components: dict[str, float],
) -> tuple[np.ndarray, bool, str]:
    if int(reference[0, 2]) < prepared.retrieval_context_length:
        return vanilla, False, "insufficient_past_context"
    model_context_length = min(int(reference[0, 2]), prepared.context_length)
    batch, components["adaptation_input_read_seconds"] = _timed(
        lambda: reader.read(reference, context_length=model_context_length)
    )
    retrieval_batch = reader.read(
        reference, context_length=prepared.retrieval_context_length
    )
    _, query_eligible, reason = _source_eligibility(
        retrieval_batch.context,
        retrieval_batch.target,
        "test",
        config.minimum_query_finite_fraction,
    )
    if not bool(query_eligible[0]):
        return vanilla, False, {1: "insufficient_finite_query_context"}.get(
            int(reason[0]), "ineligible_query"
        )

    query_representation, components["query_representation_seconds"] = _timed(
        lambda: _represent(
            forecaster, retrieval_batch.context, config.representation
        ),
        accelerator=True,
    )
    arrays = dict(extraction["arrays"])
    datastore_representation = np.load(
        fit_root / arrays["datastore.representation"], mmap_mode="r"
    )
    datastore_eligible = np.load(
        fit_root / arrays["datastore.rag_eligible"], mmap_mode="r"
    )
    datastore_positions = np.flatnonzero(datastore_eligible)
    retrieval_k = min(int(support_k), len(datastore_positions))
    if retrieval_k <= 0:
        return vanilla, False, "insufficient_valid_neighbors"
    datastore_references = prepared.indices("datastore")
    datastore_ticks = prepared.calendar_ticks("datastore")
    query_tick = prepared.calendar_ticks("test")[int(reference_position)]

    def retrieve() -> tuple[np.ndarray, np.ndarray]:
        return blockwise_topk(
            query_representation,
            datastore_representation[datastore_positions],
            reference,
            datastore_references[datastore_positions],
            query_calendar_ticks=np.asarray([query_tick], dtype=np.int64),
            datastore_calendar_ticks=datastore_ticks[datastore_positions],
            retrieval_period=int(prepared.config["retrieval_period"]),
            datastore_end_ticks_by_item=prepared.datastore_end_ticks_by_item,
            k=retrieval_k,
            stride=int(prepared.config["datastore_stride"]),
            horizon=prepared.prediction_length,
            scope=config.retrieval_scope,
            metric=config.distance_metric,
            minimum_overlap_fraction=config.minimum_overlap_fraction,
            query_block_size=1,
            datastore_block_size=config.datastore_block_size,
            require_complete_k=False,
        )

    (distances, local_ids), components["retrieval_seconds"] = _timed(retrieve)
    del distances
    valid = local_ids[0] >= 0
    if int(np.count_nonzero(valid)) < int(support_k):
        return vanilla, False, "insufficient_valid_neighbors"
    neighbor_ids = datastore_positions[local_ids[0, :k]]
    neighbor_batch, components["neighbor_fetch_seconds"] = _timed(
        lambda: reader.read(
            datastore_references[neighbor_ids],
            context_length=prepared.retrieval_context_length,
        )
    )
    neighbor_context = neighbor_batch.context.reshape(
        1, k, batch.context.shape[1], prepared.retrieval_context_length
    )
    neighbor_target = neighbor_batch.target.reshape(
        1, k, batch.context.shape[1], prepared.prediction_length
    )
    retrieval_context, components["context_construction_seconds"] = _timed(
        lambda: _query_scaled_retrieval_context(
            retrieval_batch.context, neighbor_context, neighbor_target
        )
    )
    context_forecast, components["context_forecast_seconds"] = _timed(
        lambda: _forecast(
            forecaster,
            batch.context,
            horizon=prepared.prediction_length,
            retrieval_context=retrieval_context,
        ),
        accelerator=True,
    )
    if method == "covariate_prediction":
        return context_forecast, True, "rag_eligible"
    if method == "bayes_covariate_prediction":
        mixture = json.loads(
            (model_root / model["files"]["bayes_mixture"]).read_text(
                encoding="utf-8"
            )
        )
        probability = float(mixture["probability_covariate_better"])
        prediction, components["adaptation_seconds"] = _timed(
            lambda: (1.0 - probability) * vanilla
            + probability * context_forecast
        )
        return prediction, True, "rag_eligible"

    neighbor_forecast, components["neighbor_forecast_seconds"] = _timed(
        lambda: _forecast_references_with_available_context(
            prepared,
            forecaster,
            reader,
            datastore_references[neighbor_ids],
        ),
        accelerator=True,
    )
    neighbor_forecast = neighbor_forecast.reshape(
        1, k, batch.context.shape[1], prepared.prediction_length
    )

    def build_prediction() -> tuple[np.ndarray, bool]:
        design, _ = full_ridge_design(
            vanilla,
            context_forecast,
            neighbor_target,
            neighbor_forecast,
            np.zeros_like(vanilla),
        )
        if not np.isfinite(design).all():
            return vanilla, False
        indices = ridge_feature_indices(method, k)
        if method == PER_VARIATE_RIDGE:
            keys = np.load(
                model_root / model["files"]["per_variate_series"],
                allow_pickle=False,
            )
            values = np.load(
                model_root / model["files"]["per_variate_coefficients"],
                allow_pickle=False,
            )
            key = tuple(map(int, reference[0, :2]))
            matches = np.flatnonzero(np.all(keys == key, axis=1))
            if not len(matches):
                return vanilla, False
            coefficients = values[int(matches[0])]
        else:
            relative = model["files"]["coefficients"][method]
            coefficients = np.load(model_root / relative, allow_pickle=False)
        return (
            full_ridge_predict_with_fallback(
                vanilla,
                design[..., indices],
                coefficients,
                np.asarray([True]),
            ),
            True,
        )

    (prediction, usable), components["adaptation_seconds"] = _timed(
        build_prediction
    )
    if not usable:
        return prediction, False, "nonfinite_or_missing_adaptation_design"
    return prediction, True, "rag_eligible"


def _adaptime_trial(args: argparse.Namespace) -> dict[str, Any]:
    setup_started = perf_counter()
    prepared = PreparedDataset(args.prepared)
    reference = np.asarray(
        prepared.indices("test")[args.reference_position : args.reference_position + 1],
        dtype=np.int64,
    )
    if not len(reference):
        raise IndexError(f"test reference position {args.reference_position} is absent")
    model_root: Path | None = None
    model: dict[str, Any] | None = None
    fit_root: Path | None = None
    extraction: dict[str, Any] | None = None
    config: ExtractionConfig | None = None
    if args.worker_method != "vanilla":
        fit_root, extraction, model_root, model, config = _load_adaptation_state(
            args, prepared
        )
    artifact_load_seconds = perf_counter() - setup_started
    forecaster, model_load_seconds = _load_forecaster(args, prepared)
    if args.worker_method != "vanilla" and (
        extraction["model"] != forecaster.model_name
        or extraction["weights_id"] != forecaster.weights_id
    ):
        raise ValueError("fit extraction and benchmark forecaster do not match")

    components: dict[str, float] = {}
    inference_started = perf_counter()
    reader = prepared.reader(cache_items=1)
    vanilla = _vanilla_forecast(
        prepared, forecaster, reader, reference, components
    )
    executed_method = args.worker_method
    rag_eligible: bool | None = None
    fallback_reason = "not_applicable"
    prediction = vanilla
    if args.worker_method != "vanilla":
        assert model is not None and model_root is not None
        executed_method, k, support_k = _selected_method(args.worker_method, model)
        if executed_method == "vanilla":
            rag_eligible = False
            fallback_reason = "validation_selected_vanilla"
        elif executed_method == "bayes_past_targets_prediction":
            prediction, rag_eligible, fallback_reason = _past_target_prediction(
                prepared,
                forecaster,
                reader,
                reference,
                vanilla,
                model_root,
                model,
                components,
            )
        else:
            assert fit_root is not None and extraction is not None and config is not None
            prediction, rag_eligible, fallback_reason = _rag_prediction(
                prepared,
                forecaster,
                reader,
                reference,
                vanilla,
                executed_method,
                k,
                support_k,
                fit_root,
                extraction,
                model_root,
                model,
                config,
                args.reference_position,
                components,
            )
    online_seconds = perf_counter() - inference_started
    tick = int(prepared.calendar_ticks("test")[args.reference_position])
    return {
        "requested_method": args.worker_method,
        "executed_method": executed_method,
        "reference_position": int(args.reference_position),
        "reference": list(map(int, reference[0])),
        "calendar_tick": tick,
        "rag_eligible": rag_eligible,
        "fallback_reason": fallback_reason,
        "artifact_load_seconds": artifact_load_seconds,
        "model_load_seconds": model_load_seconds,
        "online_inference_seconds": online_seconds,
        "components": components,
        "prediction_finite": bool(np.isfinite(prediction).all()),
        "prediction_mean": float(np.nanmean(prediction)),
    }


def _tsrag_trial(args: argparse.Namespace) -> dict[str, Any]:
    setup_started = perf_counter()
    prepared = TSRAGPreparedDataset(args.prepared)
    extraction_root, extraction = open_tsrag_extraction(args.tsrag_extraction)
    if extraction["prepared_signature"] != prepared.signature:
        raise ValueError("TS-RAG extraction and prepared test references do not match")
    arrays = dict(extraction["arrays"])
    datastore_representation = np.load(
        extraction_root / arrays["datastore.representation"], mmap_mode="r"
    )
    reference = np.asarray(
        prepared.indices("test")[args.reference_position : args.reference_position + 1],
        dtype=np.int64,
    )
    if not len(reference):
        raise IndexError(f"test reference position {args.reference_position} is absent")
    artifact_load_seconds = perf_counter() - setup_started

    retriever, retriever_load_seconds = _timed(
        lambda: TSRAGRetriever(
            args.tsrag_retriever_path,
            device_map=args.device,
            local_files_only=True,
        ),
        accelerator=True,
    )
    loaded, arm_load_seconds = _timed(
        lambda: load_tsrag(
            args.tsrag_chronos_bolt_path,
            args.tsrag_checkpoint_path,
            device=args.device,
        ),
        accelerator=True,
    )
    indexes, index_seconds = _build_indexes(
        prepared.indices("datastore"), datastore_representation, retriever
    )
    components: dict[str, float] = {"index_construction_seconds": index_seconds}
    inference_started = perf_counter()
    reader = prepared.reader(cache_items=1)
    batch, components["input_read_seconds"] = _timed(
        lambda: reader.read(reference, target_length=prepared.prediction_length)
    )
    representation, components["query_representation_seconds"] = _timed(
        lambda: tsrag_representation(retriever, batch.context), accelerator=True
    )
    (distances, neighbor_ids), components["retrieval_seconds"] = _timed(
        lambda: tsrag_search(reference, representation, indexes)
    )
    rollout_timings = {
        "rollout_representation_seconds": 0.0,
        "rollout_retrieval_seconds": 0.0,
        "retrieved_sequence_fetch_seconds": 0.0,
        "tsrag_model_seconds": 0.0,
    }
    prediction = _rollout_tsrag(
        loaded,
        retriever,
        prepared,
        reader,
        indexes,
        reference,
        batch.context[:, 0],
        neighbor_ids,
        distances,
        prepared.prediction_length,
        __import__("torch").device(args.device),
        rollout_timings,
    )
    components.update(rollout_timings)
    online_seconds = perf_counter() - inference_started
    tick = int(prepared.shared.calendar_ticks("test")[args.reference_position])
    return {
        "requested_method": "tsrag",
        "executed_method": "tsrag",
        "reference_position": int(args.reference_position),
        "reference": list(map(int, reference[0])),
        "calendar_tick": tick,
        "rag_eligible": True,
        "fallback_reason": "rag_eligible",
        "artifact_load_seconds": artifact_load_seconds,
        "model_load_seconds": retriever_load_seconds + arm_load_seconds,
        "retriever_load_seconds": retriever_load_seconds,
        "arm_load_seconds": arm_load_seconds,
        "online_inference_seconds": online_seconds,
        "components": components,
        "prediction_finite": bool(np.isfinite(prediction).all()),
        "prediction_mean": float(np.nanmean(prediction)),
    }


def _worker(args: argparse.Namespace) -> None:
    started = perf_counter()
    result = _tsrag_trial(args) if args.worker_method == "tsrag" else _adaptime_trial(args)
    result["worker_seconds"] = perf_counter() - started
    _atomic_json(args.worker_result, result)


def _path_argument(command: list[str], name: str, value: Path | None) -> None:
    if value is not None:
        command.extend((name, str(value)))


def _worker_command(
    args: argparse.Namespace, method: str, position: int, result: Path
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "timebench.scripts.time_inference",
        "--prepared",
        str(args.prepared),
        "--model",
        args.model,
        "--device",
        args.device,
        "--worker",
        "--worker-method",
        method,
        "--reference-position",
        str(position),
        "--worker-result",
        str(result),
    ]
    if args.weights_id is not None:
        command.extend(("--weights-id", args.weights_id))
    _path_argument(command, "--model-path", args.model_path)
    _path_argument(command, "--fit-extraction", args.fit_extraction)
    _path_argument(command, "--adaptation-model", args.adaptation_model)
    _path_argument(command, "--tsrag-extraction", args.tsrag_extraction)
    _path_argument(command, "--tsrag-chronos-bolt-path", args.tsrag_chronos_bolt_path)
    _path_argument(command, "--tsrag-retriever-path", args.tsrag_retriever_path)
    _path_argument(command, "--tsrag-checkpoint-path", args.tsrag_checkpoint_path)
    return command


def _summary(trials: list[dict[str, Any]]) -> dict[str, Any]:
    def statistics(samples: list[float]) -> dict[str, float]:
        values = np.asarray(samples, dtype=np.float64)
        return {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }

    result: dict[str, Any] = {}
    for method in sorted({str(row["requested_method"]) for row in trials}):
        rows = [row for row in trials if row["requested_method"] == method]
        values: dict[str, Any] = {
            "trials": len(rows),
            "rag_eligible": sum(row["rag_eligible"] is True for row in rows),
            "fallbacks": sum(row["rag_eligible"] is False for row in rows),
        }
        for name in (
            "subprocess_wall_seconds",
            "worker_seconds",
            "artifact_load_seconds",
            "model_load_seconds",
            "online_inference_seconds",
        ):
            values[name] = statistics([float(row[name]) for row in rows])
        component_names = sorted(
            {name for row in rows for name in dict(row["components"])}
        )
        values["components"] = {
            name: statistics(
                [float(dict(row["components"]).get(name, 0.0)) for row in rows]
            )
            for name in component_names
        }
        result[method] = values
    return result


def _validate_parent_args(args: argparse.Namespace) -> None:
    if args.samples <= 0:
        raise ValueError("samples must be positive")
    adapted = any(method not in {"vanilla", "tsrag"} for method in args.methods)
    if adapted and (args.fit_extraction is None or args.adaptation_model is None):
        raise ValueError(
            "Adaptime methods require --fit-extraction and --adaptation-model"
        )
    if "tsrag" in args.methods:
        required = {
            "--tsrag-extraction": args.tsrag_extraction,
            "--tsrag-chronos-bolt-path": args.tsrag_chronos_bolt_path,
            "--tsrag-retriever-path": args.tsrag_retriever_path,
            "--tsrag-checkpoint-path": args.tsrag_checkpoint_path,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"TS-RAG requires {', '.join(missing)}")
    if args.output.exists():
        raise FileExistsError(f"benchmark output already exists: {args.output}")


def _parent(args: argparse.Namespace) -> None:
    _resolve_task_inputs(args)
    _validate_parent_args(args)
    prepared = PreparedDataset(args.prepared)
    print(f"[{_timestamp()}] prepared={args.prepared}", flush=True)
    if args.fit_extraction is not None:
        print(f"[{_timestamp()}] fit_extraction={args.fit_extraction}", flush=True)
    if args.adaptation_model is not None:
        print(f"[{_timestamp()}] adaptation_model={args.adaptation_model}", flush=True)
    if args.tsrag_extraction is not None:
        print(f"[{_timestamp()}] tsrag_extraction={args.tsrag_extraction}", flush=True)
    positions = _sample_positions(prepared.indices("test"), args.samples, args.seed)
    order_rng = np.random.default_rng(int(args.seed) + 1)
    base_method_order = [args.methods[int(index)] for index in order_rng.permutation(len(args.methods))]
    trials: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="adaptime_time_inference_") as temporary:
        temporary_root = Path(temporary)
        for sample_number, position in enumerate(positions, start=1):
            offset = (sample_number - 1) % len(base_method_order)
            method_order = base_method_order[offset:] + base_method_order[:offset]
            for execution_order, method in enumerate(method_order, start=1):
                print(
                    f"[{_timestamp()}] sample={sample_number}/{len(positions)} "
                    f"method={method} test_position={int(position)}",
                    flush=True,
                )
                result_path = temporary_root / f"{sample_number}_{method}.json"
                started = perf_counter()
                subprocess.run(
                    _worker_command(args, method, int(position), result_path),
                    check=True,
                )
                wall_seconds = perf_counter() - started
                row = json.loads(result_path.read_text(encoding="utf-8"))
                row["sample_number"] = sample_number
                row["execution_order"] = execution_order
                row["subprocess_wall_seconds"] = wall_seconds
                trials.append(row)
    summary = _summary(trials)
    _atomic_json(
        args.output,
        {
            "schema_version": SCHEMA_VERSION,
            "format": "adaptime_independent_inference_timing",
            "created_at": _timestamp(),
            "prepared": str(Path(args.prepared).expanduser().resolve()),
            "prepared_signature": prepared.signature,
            "model": args.model,
            "methods": list(args.methods),
            "inputs": {
                "model_path": None if args.model_path is None else str(args.model_path),
                "weights_id": args.weights_id,
                "fit_extraction": None
                if args.fit_extraction is None
                else str(args.fit_extraction),
                "adaptation_model": None
                if args.adaptation_model is None
                else str(args.adaptation_model),
                "tsrag_extraction": None
                if args.tsrag_extraction is None
                else str(args.tsrag_extraction),
                "tsrag_chronos_bolt_path": str(args.tsrag_chronos_bolt_path),
                "tsrag_retriever_path": str(args.tsrag_retriever_path),
                "tsrag_checkpoint_path": str(args.tsrag_checkpoint_path),
            },
            "sampling": {
                "seed": int(args.seed),
                "samples": int(args.samples),
                "policy": "balanced_item_channel_then_random_origin_without_replacement",
                "test_positions": list(map(int, positions)),
                "method_order": "seeded_rotation_to_reduce_execution_order_bias",
            },
            "cache_contract": {
                "process": "fresh_process_per_method_and_example",
                "test_predictions": "never_read",
                "test_representations": "never_read",
                "test_neighbors": "never_read",
                "query_work": "recomputed",
                "frozen_model_state": (
                    "checkpoints_coefficients_and_datastore_representations_read_only"
                ),
                "tsrag_index": "rebuilt_per_example",
                "operating_system_page_cache": "not_controlled",
            },
            "summary": summary,
            "trials": trials,
        },
    )
    for method in args.methods:
        values = summary[method]
        wall = values["subprocess_wall_seconds"]
        online = values["online_inference_seconds"]
        print(
            f"[{_timestamp()}] method={method} trials={values['trials']} "
            f"wall_median={wall['median']:.6f}s wall_p95={wall['p95']:.6f}s "
            f"online_median={online['median']:.6f}s "
            f"fallbacks={values['fallbacks']}",
            flush=True,
        )
    print(f"[{_timestamp()}] wrote {args.output}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure independent fresh-process inference for Adaptime and TS-RAG"
        )
    )
    parser.add_argument("--prepared", type=Path)
    parser.add_argument(
        "--artifact-root", type=Path, default=outputs_root() / "adaptime"
    )
    parser.add_argument("--dataset")
    parser.add_argument("--term")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=HEADLINE_METHODS)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--model", choices=MODEL_ALIASES, default="chronos2")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--weights-id")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fit-extraction", type=Path)
    parser.add_argument("--adaptation-model", type=Path)
    parser.add_argument("--tsrag-extraction", type=Path)
    default_weights = weights_root()
    parser.add_argument(
        "--tsrag-chronos-bolt-path",
        type=Path,
        default=default_weights / "chronos-bolt-base",
    )
    parser.add_argument(
        "--tsrag-retriever-path",
        type=Path,
        default=default_weights / "chronos-t5-base",
    )
    parser.add_argument(
        "--tsrag-checkpoint-path",
        type=Path,
        default=default_weights / "ts-rag",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-method", choices=METHODS, help=argparse.SUPPRESS)
    parser.add_argument("--reference-position", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if args.worker_method is None or args.reference_position is None or args.worker_result is None:
            parser.error("worker mode requires method, reference position, and result path")
    elif args.output is None:
        parser.error("--output is required")
    return args


def main() -> None:
    args = parse_args()
    _worker(args) if args.worker else _parent(args)


if __name__ == "__main__":
    main()
