"""TiRex experiments for the inherited TIME benchmark."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from dotenv import load_dotenv
from gluonts.time_feature import get_seasonality
from tirex import ForecastModel, load_model

from timebench.evaluation.covariates import COVARIATE_MODES, validate_covariate_mode
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

DEFAULT_QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SUPPORTS_COVARIATES = False


def run_tirex_experiment(
    dataset_name: str,
    terms: list[str] | None = None,
    output_dir: str | None = None,
    batch_size: int = 128,
    config_path: Path | None = None,
    quantile_levels: list[float] | None = None,
    model_path: str | Path | None = None,
    covariate_mode: str = "none",
    target_mode: str = "auto",
):
    covariate_mode = validate_covariate_mode(
        "tirex", covariate_mode, supports_covariates=SUPPORTS_COVARIATES
    )
    print("Loading configuration...")
    config = load_dataset_config(config_path)
    if terms is None:
        terms = get_available_terms(dataset_name, config)
        if not terms:
            raise ValueError(f"No terms defined for dataset '{dataset_name}' in config")
    if quantile_levels is None:
        quantile_levels = DEFAULT_QUANTILE_LEVELS

    if output_dir is None:
        output_dir = str(foundation_experiment_root(covariate_mode))
    os.makedirs(output_dir, exist_ok=True)
    experiment = foundation_experiment_name(covariate_mode)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = foundation_weight_path(
        "tirex", explicit=model_path, directory=True
    )

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Model: TiRex ({checkpoint_path})")
    print(f"Terms: {terms}")
    print(f"{'='*60}")
    model: ForecastModel = load_model(str(checkpoint_path), device=device)

    for term in terms:
        print(f"\n--- Term: {term} ---")
        settings = get_dataset_settings(dataset_name, term, config)
        prediction_length = settings.get("prediction_length")
        test_length = settings.get("test_length")
        val_length = settings.get("val_length")
        print(
            "  Config: prediction_length={}, test_length={}, val_length={}".format(
                prediction_length, test_length, val_length
            )
        )

        dataset = Dataset(
            name=dataset_name,
            term=term,
            to_univariate=False,
            prediction_length=prediction_length,
            test_length=test_length,
            val_length=val_length,
        )
        resolved_target_mode = resolve_target_mode(
            target_mode,
            target_dim=dataset.target_dim,
            supports_multivariate=False,
        )
        if dataset.target_dim > 1:
            dataset = Dataset(
                name=dataset_name,
                term=term,
                to_univariate=True,
                prediction_length=prediction_length,
                test_length=test_length,
                val_length=val_length,
            )

        print("  Dataset info:")
        print(f"    - Frequency: {dataset.freq}")
        print(f"    - Num series: {len(dataset.hf_dataset)}")
        print(f"    - Target mode: {resolved_target_mode}")
        print(
            "    - Series length: min={}, max={}, avg={:.1f}".format(
                dataset._min_series_length,
                dataset._max_series_length,
                dataset._avg_series_length,
            )
        )
        print(f"    - Test split: {test_length} steps")
        print(f"    - Prediction length: {dataset.prediction_length}")
        print(f"    - Windows: {dataset.windows}")

        flat_contexts = []
        for entry in dataset.test_data.input:
            target = np.asarray(entry["target"], dtype=np.float32)
            flat_contexts.append(target if target.ndim == 1 else target.squeeze())

        timer = EvaluationTimer()
        timer.start()
        forecast_batches = []
        total_items = len(flat_contexts)
        for start in range(0, total_items, batch_size):
            end = min(start + batch_size, total_items)
            with torch.no_grad():
                quantiles, _ = model.forecast(
                    context=flat_contexts[start:end],
                    prediction_length=prediction_length,
                )
            quantiles_np = (
                quantiles.detach().cpu().float().numpy()
                if isinstance(quantiles, torch.Tensor)
                else np.asarray(quantiles)
            )
            forecast_batches.append(quantiles_np.transpose(0, 2, 1))
            if start % (batch_size * 5) == 0:
                sys.stdout.write(f"\r    Processed {end}/{total_items} items...")
                sys.stdout.flush()
        print(f"\r    Processed {total_items}/{total_items} items. Done.")
        fc_quantiles = np.concatenate(forecast_batches, axis=0).astype(
            np.float32, copy=False
        )
        inference_seconds = timer.stop()

        season_length = get_seasonality(dataset.freq)
        model_hyperparams = {
            "model": "tirex",
            "quantile_levels": quantile_levels,
            "covariate_mode": covariate_mode,
            "covariate_channels": 0,
            "experiment": experiment,
            "target_mode": resolved_target_mode,
        }
        identity_root = foundation_identity_root(
            output_dir, "tirex", resolved_target_mode, dataset_name, term
        )
        with allocate_run(
            identity_root,
            experiment=experiment,
            identity={
                "model": "tirex",
                "target_mode": resolved_target_mode,
                "dataset": dataset_name.rpartition("/")[0] or dataset_name,
                "frequency": dataset.freq,
                "term": term,
            },
            model_config={
                "checkpoint": "tirex",
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
                "device": device,
                "checkpoint_path": str(checkpoint_path),
            },
            experiment_config={
                "covariate_mode": covariate_mode,
                "covariate_channels": 0,
            },
            provenance={
                "dataset_config_path": None if config_path is None else str(config_path),
            },
        ) as run:
            save_window_predictions(
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
        print(f"  Output: {run.run_dir}")

    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run TiRex experiments")
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        default=["Global_Influenza/W"],
        help="Dataset name(s). 'all_datasets' runs the full configured set.",
    )
    parser.add_argument(
        "--terms",
        type=str,
        nargs="+",
        default=None,
        choices=["short", "medium", "long"],
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=DEFAULT_QUANTILE_LEVELS,
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Local TiRex checkpoint directory",
    )
    parser.add_argument(
        "--covariate-mode",
        choices=COVARIATE_MODES,
        default=os.environ.get("TIME_COVARIATE_MODE", "none"),
        help="TiRex accepts only none and rejects known covariates",
    )
    parser.add_argument(
        "--target-mode",
        choices=("auto", "univariate", "multivariate"),
        default=os.environ.get("TIME_TARGET_MODE", "auto"),
        help="TiRex rejects multivariate target input",
    )
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None
    if len(args.dataset) == 1 and args.dataset[0] == "all_datasets":
        config = load_dataset_config(config_path)
        datasets = list(config.get("datasets", {}).keys())
    else:
        datasets = args.dataset

    for idx, dataset_name in enumerate(datasets, 1):
        print(f"\n{'#'*60}")
        print(f"# Dataset {idx}/{len(datasets)}: {dataset_name}")
        print(f"{'#'*60}")
        run_tirex_experiment(
            dataset_name=dataset_name,
            terms=args.terms,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            config_path=config_path,
            quantile_levels=args.quantiles,
            model_path=args.model_path,
            covariate_mode=args.covariate_mode,
            target_mode=args.target_mode,
        )

    print(f"\n{'#'*60}")
    print("# All datasets completed!")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
