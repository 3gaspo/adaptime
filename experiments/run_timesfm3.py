"""TimesFM-3 experiments through Google's official PyTorch evaluator."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from dotenv import load_dotenv
from gluonts.time_feature import get_seasonality
from timesfm3 import ModelConfig, TimesFM3Evaluator

from timebench.evaluation.covariates import (
    COVARIATE_MODES,
    extract_covariate_window,
    validate_covariate_channels,
    validate_covariate_mode,
)
from timebench.evaluation.data import Dataset, get_dataset_settings, load_dataset_config
from timebench.evaluation.saver import save_window_predictions
from timebench.evaluation.timing import EvaluationTimer
from timebench.evaluation.utils import get_available_terms
from timebench.paths import (
    foundation_experiment_name,
    foundation_experiment_root,
    foundation_identity_root,
    foundation_weight_path,
)
from timebench.pipeline import allocate_run, resolve_target_mode


load_dotenv()

MODEL_ALIAS = "timesfm3"
MODEL_REPOSITORY = "google/timesfm-3.0-pytorch"
MAX_CONTEXT_LENGTH = 15360
OFFICIAL_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SUPPORTS_COVARIATES = True


def _prepare_target(entry: dict, context_length: int) -> np.ndarray:
    target = np.asarray(entry["target"], dtype=np.float32)
    return target[..., -context_length:]


def _time_quantiles(
    values: np.ndarray,
    *,
    prediction_length: int,
    target_mode: str,
) -> np.ndarray:
    """Move the official trailing quantile axis into TIME's quantile-first layout."""
    quantiles = np.asarray(values, dtype=np.float32)
    if quantiles.ndim == 2:
        expected = (prediction_length, len(OFFICIAL_QUANTILES))
        if quantiles.shape != expected:
            raise ValueError(
                f"Unexpected TimesFM-3 univariate quantiles {quantiles.shape}; "
                f"expected {expected}"
            )
        return quantiles.transpose(1, 0)
    if quantiles.ndim == 3:
        if quantiles.shape[1:] != (prediction_length, len(OFFICIAL_QUANTILES)):
            raise ValueError(
                f"Unexpected TimesFM-3 multivariate quantiles {quantiles.shape}"
            )
        converted = quantiles.transpose(2, 0, 1)
        if target_mode == "univariate":
            if converted.shape[1] != 1:
                raise ValueError(
                    "TimesFM-3 returned multiple target variates for an univariate task"
                )
            return converted[:, 0, :]
        return converted
    raise ValueError(f"Unexpected TimesFM-3 quantile rank: {quantiles.ndim}")


def run_timesfm3_experiment(
    dataset_name: str,
    terms: list[str] | None = None,
    model_size: str = MODEL_ALIAS,
    output_dir: str | None = None,
    batch_size: int = 4,
    context_length: int = MAX_CONTEXT_LENGTH,
    config_path: Path | None = None,
    quantile_levels: list[float] | None = None,
    model_path: str | Path | None = None,
    covariate_mode: str = "none",
    target_mode: str = "auto",
) -> None:
    """Evaluate the released TimesFM-3 checkpoint on one TIME dataset."""
    if model_size != MODEL_ALIAS:
        raise ValueError(f"Unsupported TimesFM-3 model size: {model_size}")
    if not 1 <= context_length <= MAX_CONTEXT_LENGTH:
        raise ValueError(
            f"TimesFM-3 context length must be between 1 and {MAX_CONTEXT_LENGTH}"
        )
    quantile_levels = quantile_levels or OFFICIAL_QUANTILES.copy()
    if quantile_levels != OFFICIAL_QUANTILES:
        raise ValueError(
            f"TimesFM-3 exposes the fixed quantiles {OFFICIAL_QUANTILES}, "
            f"received {quantile_levels}"
        )
    covariate_mode = validate_covariate_mode(
        MODEL_ALIAS,
        covariate_mode,
        supports_covariates=SUPPORTS_COVARIATES,
        supported_modes=COVARIATE_MODES,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = load_dataset_config(config_path)
    if terms is None:
        terms = get_available_terms(dataset_name, config)
        if not terms:
            raise ValueError(f"No terms defined for dataset {dataset_name!r}")

    checkpoint_path = foundation_weight_path(
        MODEL_ALIAS,
        explicit=model_path,
        directory=True,
    )
    output_dir = output_dir or str(foundation_experiment_root())
    os.makedirs(output_dir, exist_ok=True)
    experiment = foundation_experiment_name()

    print(f"Dataset: {dataset_name}")
    print(f"Model: {MODEL_REPOSITORY}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Terms: {terms}")
    print(f"Covariate mode: {covariate_mode}")

    for term in terms:
        settings = get_dataset_settings(dataset_name, term, config)
        prediction_length = settings.get("prediction_length")
        test_length = settings.get("test_length")
        val_length = settings.get("val_length")

        dataset = Dataset(
            name=dataset_name,
            term=term,
            to_univariate=False,
            prediction_length=prediction_length,
            test_length=test_length,
            val_length=val_length,
        )
        if covariate_mode == "past_targets":
            if target_mode == "multivariate":
                raise ValueError(
                    "past_targets forecasts one variate at a time and requires "
                    "target_mode=auto or univariate"
                )
            resolved_target_mode = "univariate"
            dataset = Dataset(
                name=dataset_name,
                term=term,
                prediction_length=prediction_length,
                test_length=test_length,
                val_length=val_length,
                other_variates_as_covariates=True,
            )
        else:
            resolved_target_mode = resolve_target_mode(
                target_mode,
                target_dim=dataset.target_dim,
                supports_multivariate=True,
            )
            if resolved_target_mode == "univariate" and dataset.target_dim > 1:
                dataset = Dataset(
                    name=dataset_name,
                    term=term,
                    to_univariate=True,
                    prediction_length=prediction_length,
                    test_length=test_length,
                    val_length=val_length,
                )
        if covariate_mode != "none" and dataset.covariate_dim == 0:
            raise ValueError(
                f"{dataset_name} does not provide covariates for "
                f"covariate_mode={covariate_mode!r}"
            )

        season_length = get_seasonality(dataset.freq)
        covariate_channels = dataset.covariate_dim if covariate_mode != "none" else 0
        identity_root = foundation_identity_root(
            output_dir,
            MODEL_ALIAS,
            resolved_target_mode,
            dataset_name,
            term,
        )
        run = allocate_run(
            identity_root,
            experiment=experiment,
            identity={
                "model": MODEL_ALIAS,
                "target_mode": resolved_target_mode,
                "dataset": dataset_name.rpartition("/")[0] or dataset_name,
                "frequency": dataset.freq,
                "term": term,
            },
            model_config={
                "model_size": model_size,
                "context_length": context_length,
                "quantile_levels": quantile_levels,
                "upstream_repository": MODEL_REPOSITORY,
                "upstream_package": "timesfm[torch]==3.0.1",
                "evaluator": "TimesFM3Evaluator",
            },
            pipeline_config={
                "prediction_length": prediction_length,
                "test_length": test_length,
                "val_length": val_length,
                "windows": dataset.windows,
                "seasonality": season_length,
            },
            runtime_config={
                "batch_size": batch_size,
                "device": device,
                "checkpoint_path": str(checkpoint_path),
                "local_files_only": True,
            },
            experiment_config={
                "covariate_mode": covariate_mode,
                "covariate_channels": covariate_channels,
                "covariate_source": (
                    "other_target_variates"
                    if covariate_mode == "past_targets"
                    else "dataset_fields"
                    if covariate_mode == "future_included"
                    else "none"
                ),
                "covariate_time_span": (
                    "L"
                    if covariate_mode == "past_targets"
                    else "L+H"
                    if covariate_mode == "future_included"
                    else "none"
                ),
                "official_evaluator_defaults": {
                    "use_symmetric_averaging": True,
                    "make_positive": True,
                    "sort_quantiles": True,
                    "use_znorm": False,
                    "padding_mode": "none",
                },
            },
            provenance={
                "dataset_config_path": None if config_path is None else str(config_path),
            },
        )
        if not run.should_run:
            print(f"Reused completed task: {run.run_dir}")
            continue

        print(f"Initializing official TimesFM-3 evaluator ({checkpoint_path})...")
        forecaster = TimesFM3Evaluator(
            ModelConfig(
                checkpoint_path=str(checkpoint_path),
                per_core_batch_size=batch_size,
                device=device,
                local_files_only=True,
            )
        )

        eval_items = list(dataset.test_data)
        quantile_batches = []
        timer = EvaluationTimer()
        timer.start()
        for start in range(0, len(eval_items), batch_size):
            batch_items = eval_items[start : start + batch_size]
            contexts = [
                _prepare_target(input_entry, context_length)
                for input_entry, _ in batch_items
            ]
            past_only_covariates = None
            past_future_covariates = None
            if covariate_mode != "none":
                windows = [
                    extract_covariate_window(
                        input_entry,
                        label_entry,
                        context_length=context_length,
                        prediction_length=prediction_length,
                        require_future=covariate_mode == "future_included",
                    )
                    for input_entry, label_entry in batch_items
                ]
                covariate_channels = validate_covariate_channels(
                    windows,
                    expected=covariate_channels,
                )
                if covariate_mode == "past_targets":
                    past_only_covariates = [window.past for window in windows]
                else:
                    past_future_covariates = [window.full for window in windows]

            outputs = list(
                forecaster.predict_batch(
                    contexts=contexts,
                    horizon=prediction_length,
                    past_only_covariates=past_only_covariates,
                    past_future_covariates=past_future_covariates,
                    return_quantiles=True,
                    use_symmetric_averaging=True,
                    make_positive=True,
                    sort_quantiles=True,
                    use_znorm=False,
                    padding_mode="none",
                )
            )
            if len(outputs) != len(batch_items):
                raise ValueError(
                    f"TimesFM-3 returned {len(outputs)} forecasts for "
                    f"{len(batch_items)} inputs"
                )
            quantile_batches.append(
                np.stack(
                    [
                        _time_quantiles(
                            output.quantiles,
                            prediction_length=prediction_length,
                            target_mode=resolved_target_mode,
                        )
                        for output in outputs
                    ],
                    axis=0,
                )
            )
            if (start // batch_size + 1) % 10 == 0:
                print(f"Processed {min(start + batch_size, len(eval_items))}/{len(eval_items)}")

        fc_quantiles = np.concatenate(quantile_batches, axis=0)
        inference_seconds = timer.stop()
        model_hyperparams = {
            "model": MODEL_ALIAS,
            "context_length": context_length,
            "quantile_levels": quantile_levels,
            "covariate_mode": covariate_mode,
            "covariate_channels": covariate_channels or 0,
            "experiment": experiment,
            "target_mode": resolved_target_mode,
            "upstream_repository": MODEL_REPOSITORY,
        }

        with run:
            metadata = save_window_predictions(
                dataset=dataset,
                fc_quantiles=fc_quantiles,
                ds_config=f"{dataset_name}/{term}",
                output_base_dir=output_dir,
                seasonality=season_length,
                model_hyperparams=model_hyperparams,
                quantile_levels=quantile_levels,
                inference_seconds=inference_seconds,
                task_output_dir=str(run.run_dir),
            )
            run.complete(
                ["predictions.npz", "metrics.npz", "config.json", "metrics_summary.json"]
            )
        print(
            f"Completed: {metadata['num_series']} series x "
            f"{metadata['num_windows']} windows -> {run.run_dir}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TimesFM-3 experiments")
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["Global_Influenza/W"],
        help="Dataset names, all_datasets, or all_multivariate_datasets",
    )
    parser.add_argument(
        "--terms",
        nargs="+",
        choices=["short", "medium", "long"],
        default=None,
    )
    parser.add_argument("--model-size", default=MODEL_ALIAS, choices=[MODEL_ALIAS])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=MAX_CONTEXT_LENGTH)
    parser.add_argument("--config", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=OFFICIAL_QUANTILES,
    )
    parser.add_argument(
        "--covariate-mode",
        choices=COVARIATE_MODES,
        default=os.environ.get("TIME_COVARIATE_MODE", "none"),
    )
    parser.add_argument(
        "--target-mode",
        choices=("auto", "univariate", "multivariate"),
        default=os.environ.get("TIME_TARGET_MODE", "auto"),
    )
    args = parser.parse_args()
    config_path = Path(args.config) if args.config else None

    if len(args.dataset) == 1 and args.dataset[0] in {
        "all_datasets",
        "all_multivariate_datasets",
    }:
        config = load_dataset_config(config_path)
        datasets = list(config.get("datasets", {}).keys())
        if args.dataset[0] == "all_multivariate_datasets":
            datasets = [name for name in datasets if Dataset(name=name).target_dim > 1]
    else:
        datasets = args.dataset

    for index, dataset_name in enumerate(datasets, 1):
        print(f"Dataset {index}/{len(datasets)}: {dataset_name}")
        run_timesfm3_experiment(
            dataset_name=dataset_name,
            terms=args.terms,
            model_size=args.model_size,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            context_length=args.context_length,
            config_path=config_path,
            quantile_levels=args.quantiles,
            model_path=args.model_path,
            covariate_mode=args.covariate_mode,
            target_mode=args.target_mode,
        )


if __name__ == "__main__":
    main()
