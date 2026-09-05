"""
Chronos-2 model experiments for time series forecasting.

Usage:
    python experiments/chronos2.py
    python experiments/chronos2.py --model-size chronos2
    python experiments/chronos2.py --dataset "TSBench_IMOS_v2/15T" --terms short medium long
    python experiments/chronos2.py --dataset "SG_Weather/D" "SG_PM25/H"  # Multiple datasets
    python experiments/chronos2.py --dataset all_datasets  # Run all datasets from config
"""

import argparse
import os
import sys
import logging
from pathlib import Path

# Ensure timebench is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from dotenv import load_dotenv
from chronos import BaseChronosPipeline
from gluonts.time_feature import get_seasonality

from timebench.evaluation.saver import save_window_predictions
from timebench.evaluation.timing import EvaluationTimer
from timebench.evaluation.utils import get_available_terms
from timebench.evaluation.covariates import (
    COVARIATE_MODES,
    extract_covariate_window,
    validate_covariate_channels,
    validate_covariate_mode,
)
from timebench.evaluation.data import (
    Dataset,
    get_dataset_settings,
    load_dataset_config,
)
from timebench.paths import (
    foundation_experiment_name,
    foundation_experiment_root,
    foundation_identity_root,
    foundation_weight_path,
)
from timebench.pipeline import allocate_run, resolve_target_mode

# Load environment variables
load_dotenv()

logging.getLogger("chronos").setLevel(logging.ERROR)

SUPPORTS_COVARIATES = True


def run_chronos2_experiment(
    dataset_name: str,
    terms: list[str] = None,
    model_size: str = "chronos2",
    output_dir: str | None = None,
    batch_size: int = 32,
    context_length: int = 2048,
    config_path: Path | None = None,
    quantile_levels: list[float] | None = None,
    model_path: str | Path | None = None,
    covariate_mode: str = "none",
    target_mode: str = "auto",
):
    """
    Run Chronos-2 model experiments.
    """
    covariate_mode = validate_covariate_mode(
        "chronos2",
        covariate_mode,
        supports_covariates=SUPPORTS_COVARIATES,
        supported_modes=COVARIATE_MODES,
    )

    # Set CUDA device
    device_map = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset configuration
    print("Loading configuration...")
    config = load_dataset_config(config_path)

    # Auto-detect available terms from config if not specified
    if terms is None:
        terms = get_available_terms(dataset_name, config)
        if not terms:
            raise ValueError(f"No terms defined for dataset '{dataset_name}' in config")

    if quantile_levels is None:
        quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    if output_dir is None:
        output_dir = str(foundation_experiment_root(covariate_mode))

    os.makedirs(output_dir, exist_ok=True)
    experiment = foundation_experiment_name(covariate_mode)

    if model_size != "chronos2":
        raise ValueError(f"Unsupported Chronos-2 model size: {model_size}")
    checkpoint_path = foundation_weight_path(
        "chronos2",
        explicit=model_path,
        directory=True,
    )

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print("Model: amazon/chronos-2")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Terms: {terms}")
    print(f"Covariate mode: {covariate_mode}")
    print(f"{'='*60}")

    for term in terms:
        print(f"\n--- Term: {term} ---")

        # Get settings from config
        settings = get_dataset_settings(dataset_name, term, config)
        prediction_length = settings.get("prediction_length")
        test_length = settings.get("test_length")
        val_length = settings.get("val_length")

        print(f"  Config: prediction_length={prediction_length}, test_length={test_length}, val_length={val_length}")

        # Dataset Initialization
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
            output_dir, "chronos2", resolved_target_mode, dataset_name, term
        )
        run = allocate_run(
            identity_root,
            experiment=experiment,
            identity={
                "model": "chronos2",
                "target_mode": resolved_target_mode,
                "dataset": dataset_name.rpartition("/")[0] or dataset_name,
                "frequency": dataset.freq,
                "term": term,
            },
            model_config={
                "model_size": model_size,
                "context_length": context_length,
                "quantile_levels": quantile_levels,
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
                "device": device_map,
                "checkpoint_path": str(checkpoint_path),
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
                    else "L+H" if covariate_mode == "future_included" else "none"
                ),
            },
            provenance={
                "dataset_config_path": None if config_path is None else str(config_path),
            },
        )
        if not run.should_run:
            print(f"  Reused completed task: {run.run_dir}")
            continue

        # Initialize Chronos only after the dataset capability check.
        print(f"  Initializing Chronos pipeline ({checkpoint_path})...")
        pipeline = BaseChronosPipeline.from_pretrained(
            str(checkpoint_path),
            device_map=device_map,
            local_files_only=True,
        )

        # Determine split
        data_length = test_length
        num_windows = dataset.windows
        split_name = "Test split"
        eval_data = dataset.test_data

        print("  Dataset info:")
        print(f"    - Frequency: {dataset.freq}")
        print(f"    - Num series: {len(dataset.hf_dataset)}")
        print(f"    - Target dim: {dataset.target_dim}")
        print(f"    - Target mode: {resolved_target_mode}")
        print(f"    - Covariate channels: {dataset.covariate_dim}")
        print(f"    - Series length: min={dataset._min_series_length}, max={dataset._max_series_length}, avg={dataset._avg_series_length:.1f}")
        print(f"    - {split_name}: {data_length} steps")
        print(f"    - Prediction length: {dataset.prediction_length}")
        print(f"    - Windows: {num_windows}")

        timer = EvaluationTimer()
        timer.start()

        # ---------------------------------------------------------
        # 1. Running Inference (Chronos-2 Specific Logic)
        # ---------------------------------------------------------
        # Helper function to prepare a single context
        def _prepare_context(d, covariates=None):
            target = np.asarray(d["target"])

            # Manually truncate context
            seq_len = target.shape[-1]
            if seq_len > context_length:
                target = target[..., -context_length:]

            if target.ndim == 1:
                target = target[np.newaxis, :]

            target_tensor = torch.tensor(target)
            if covariates is None:
                return target_tensor
            item = {
                "target": target_tensor,
                "past_covariates": {
                    f"covariate_{channel}": torch.from_numpy(values)
                    for channel, values in enumerate(covariates.past)
                },
            }
            if covariates.future.shape[-1] > 0:
                item["future_covariates"] = {
                    f"covariate_{channel}": torch.from_numpy(values)
                    for channel, values in enumerate(covariates.future)
                }
            return item

        # Batch Inference with lazy loading
        fc_quantiles_batches = []
        eval_items = list(eval_data)
        total_items = len(eval_items)
        for start in range(0, total_items, batch_size):
            end = min(start + batch_size, total_items)
            batch_items = eval_items[start:end]
            if covariate_mode != "none":
                batch_covariates = [
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
                    batch_covariates, expected=covariate_channels
                )
            else:
                batch_covariates = [None] * len(batch_items)
            batch_contexts = [
                _prepare_context(input_entry, covariates)
                for (input_entry, _), covariates in zip(batch_items, batch_covariates)
            ]

            # Filter out verbose warnings from Chronos-2 during prediction to keep output clean
            class ContentFilterStderr:
                def __init__(self, original_stream):
                    self.original_stream = original_stream

                def write(self, data):
                    if "Quantiles to be predicted" in data and "Chronos-2" in data:
                        return
                    self.original_stream.write(data)

                def flush(self):
                    self.original_stream.flush()

            original_stderr = sys.stderr
            sys.stderr = ContentFilterStderr(original_stderr)
            try:
                with torch.no_grad():
                    batch_q, batch_m = pipeline.predict_quantiles(
                        inputs=batch_contexts,
                        prediction_length=prediction_length,
                        quantile_levels=quantile_levels,
                    )
            finally:
                sys.stderr = original_stderr

            batch_quantiles_list = []
            for q in batch_q:
                if isinstance(q, torch.Tensor):
                    if q.ndim == 3 and q.shape[-1] == len(quantile_levels):
                        # Shape: (num_variates, pred_len, num_quantiles) -> (num_quantiles, num_variates, pred_len)
                        q = q.permute(2, 0, 1)
                    q = q.cpu().float().numpy()
                # q shape: (num_quantiles, num_variates, prediction_length)
                # Add batch dimension: (1, num_quantiles, num_variates, prediction_length)
                batch_quantiles_list.append(q[np.newaxis, ...])


            # Stack into batch: (batch_size, num_quantiles, num_variates, prediction_length)
            batch_q_array = np.concatenate(batch_quantiles_list, axis=0)
            if resolved_target_mode == "univariate" and batch_q_array.ndim == 4:
                batch_q_array = batch_q_array[:, :, 0, :]
            fc_quantiles_batches.append(batch_q_array)

            # Optional progress logging
            if (start // batch_size + 1) % 10 == 0:
                print(f"    Processed {min(start + batch_size, total_items)}/{total_items}...")

        # Concatenate all batches into a single array
        # Shape: (num_total_instances, num_quantiles, num_variates, prediction_length)
        fc_quantiles = np.concatenate(fc_quantiles_batches, axis=0)
        inference_seconds = timer.stop()

        # ---------------------------------------------------------
        # 3. Saving Results
        # ---------------------------------------------------------
        ds_config = f"{dataset_name}/{term}"
        model_hyperparams = {
            "model": "chronos2",
            "context_length": context_length,
            "quantile_levels": quantile_levels,
            "covariate_mode": covariate_mode,
            "covariate_channels": covariate_channels or 0,
            "experiment": experiment,
            "target_mode": resolved_target_mode,
            "covariate_source": (
                "other_target_variates"
                if covariate_mode == "past_targets"
                else "dataset_fields" if covariate_mode == "future_included" else "none"
            ),
            "covariate_time_span": (
                "L"
                if covariate_mode == "past_targets"
                else "L+H" if covariate_mode == "future_included" else "none"
            ),
        }

        with run:
            metadata = save_window_predictions(
                dataset=dataset,
                fc_quantiles=fc_quantiles,
                ds_config=ds_config,
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

        print(f"  Completed: {metadata['num_series']} series × {metadata['num_windows']} windows")
        print(f"  Output: {run.run_dir}")

    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run Chronos experiments")
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        default=["Global_Influenza/W"],
        help=(
            "Dataset names, all_datasets, or all_multivariate_datasets "
            "for the shared D>1 comparison subset"
        ),
    )
    parser.add_argument("--terms", type=str, nargs="+", default=None,
                        choices=["short", "medium", "long"],
                        help="Terms to evaluate. If not specified, auto-detect from config.")
    parser.add_argument("--model-size", type=str, default="chronos2",
                        help="Chronos model size (use 'chronos2' for amazon/chronos-2)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Experiment result root; defaults to expe_uni or expe_covar")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for prediction")
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        help="Quantile levels to predict",
    )
    parser.add_argument("--context-length", type=int, default=8192,
                        help="Maximum context length")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to datasets.yaml config file")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Local Chronos-2 checkpoint directory")
    parser.add_argument(
        "--covariate-mode",
        choices=COVARIATE_MODES,
        default=os.environ.get("TIME_COVARIATE_MODE", "none"),
        help=(
            "Known-covariate mode: future_included uses external L+H values; "
            "past_targets uses the other target variates over L"
        ),
    )
    parser.add_argument(
        "--target-mode",
        choices=("auto", "univariate", "multivariate"),
        default=os.environ.get("TIME_TARGET_MODE", "auto"),
        help="Target representation; auto uses native multivariate input when available",
    )

    args = parser.parse_args()

    # Handle dataset list or 'all_datasets'
    config_path = Path(args.config) if args.config else None

    if len(args.dataset) == 1 and args.dataset[0] in {
        "all_datasets",
        "all_multivariate_datasets",
    }:
        # Load all datasets from config
        config = load_dataset_config(config_path)
        datasets = list(config.get("datasets", {}).keys())
        if args.dataset[0] == "all_multivariate_datasets":
            datasets = [
                name
                for name in datasets
                if Dataset(name=name).target_dim > 1
            ]
        print(f"Running all {len(datasets)} datasets from config:")
        for ds in datasets:
            print(f"  - {ds}")
    else:
        datasets = args.dataset

    # Iterate over all datasets with progress logging
    total_datasets = len(datasets)
    for idx, dataset_name in enumerate(datasets, 1):
        print(f"\n{'#'*60}")
        print(f"# Dataset {idx}/{total_datasets}: {dataset_name}")
        print(f"{'#'*60}")

        run_chronos2_experiment(
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

    print(f"\n{'#'*60}")
    print(f"# All {total_datasets} dataset(s) completed!")
    print(f"{'#'*60}")

if __name__ == "__main__":
    main()
