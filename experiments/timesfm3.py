"""
TimesFM-3 model experiments for TIME Benchmark.

Usage:
    python experiments/timesfm3.py
    python experiments/timesfm3.py --dataset "SG_Weather/D" --terms short medium long
    python experiments/timesfm3.py --dataset "SG_Weather/D" "SG_PM25/H"  # Multiple datasets
    python experiments/timesfm3.py --dataset all_datasets  # Run all datasets from config
"""

import argparse
import math
import os
import sys
from pathlib import Path
from typing import List, Optional

# Prevent current directory from shadowing the timesfm3 package
script_dir = os.path.dirname(os.path.abspath(__file__))
while script_dir in sys.path:
    sys.path.remove(script_dir)

# Ensure timebench is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from dotenv import load_dotenv
from gluonts.time_feature import get_seasonality

from timebench.evaluation.data import (
    Dataset,
    get_dataset_settings,
    load_dataset_config,
)
from timebench.evaluation.saver import save_window_predictions
from timebench.evaluation.timing import EvaluationTimer
from timebench.evaluation.utils import get_available_terms
from timebench.paths import results_root

# Load environment variables
load_dotenv()

DEFAULT_QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def get_optimal_batch_size(
    max_context_len: int,
    num_variates: int = 1,
    min_batch: int = 4,
    max_batch: int = 64,
) -> int:
    """Compute dynamic batch size scaled by total variates (capped at 32 chunk limit) and context length, rounded to nearest power of 2."""
    full_context_count_per_ts = min(32, num_variates) * min(1.0, max(max_context_len, 32) / 15360.0)
    batch_size = 64.0 / max(full_context_count_per_ts, 0.01)
    power_of_2 = int(2 ** round(math.log2(batch_size))) if batch_size > 0 else 1
    return int(np.clip(power_of_2, min_batch, max_batch))


def prepare_time_context(item: dict, max_context_length: int = 15360) -> np.ndarray:
    """Extract target array from TIME dataset entry and format as (num_variates, context_len)."""
    target = np.asarray(item["target"], dtype=np.float32)
    if target.ndim == 1:
        target = target[np.newaxis, :]
    if target.shape[-1] > max_context_length:
        target = target[:, -max_context_length:]
    return target


def run_timesfm3_experiment(
    dataset_name: str,
    terms: Optional[List[str]] = None,
    model_size: str = "base",
    checkpoint_path: str = "google/timesfm-3.0-pytorch",
    output_dir: Optional[str] = None,
    batch_size: Optional[int] = None,
    context_length: int = 15360,
    config_path: Optional[Path] = None,
    quantile_levels: Optional[List[float]] = None,
    storage_path: Optional[Path] = None,
):
    """Run TimesFM-3 model experiments on a dataset with specified terms."""
    from timesfm3 import TimesFM3Evaluator, ModelConfig

    print("Loading configuration...")
    config = load_dataset_config(config_path)

    # Auto-detect available terms from config if not specified
    if terms is None:
        terms = get_available_terms(dataset_name, config)
        if not terms:
            raise ValueError(f"No terms defined for dataset '{dataset_name}' in config")

    if quantile_levels is None:
        quantile_levels = DEFAULT_QUANTILE_LEVELS

    if output_dir is None:
        output_dir = str(results_root() / "TimesFM-3")

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Terms: {terms}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*60}")

    # Initialize TimesFM-3 Evaluator
    print(f"Initializing TimesFM-3 Evaluator...")
    forecaster = TimesFM3Evaluator(
        ModelConfig(
            checkpoint_path=checkpoint_path,
            per_core_batch_size=64,
            use_variate_attention=True,
            use_sdpa=True,
        )
    )
    print(f"TimesFM-3 loaded on: {forecaster.device}")

    for term in terms:
        print(f"\n--- Term: {term} ---")

        # Get settings from config
        settings = get_dataset_settings(dataset_name, term, config)
        prediction_length = settings.get("prediction_length")
        test_length = settings.get("test_length")
        val_length = settings.get("val_length", 0)

        print(f"  Config: prediction_length={prediction_length}, test_length={test_length}, val_length={val_length}")

        if storage_path is None:
            time_dataset_env = os.getenv("TIME_DATASET")
            if time_dataset_env:
                storage_path = Path(time_dataset_env)
            else:
                default_cache = Path.home() / ".cache" / "TIME_dataset"
                if default_cache.exists():
                    storage_path = default_cache

        # Load dataset in full native multivariate mode
        dataset = Dataset(
            name=dataset_name,
            term=term,
            to_univariate=False,
            prediction_length=prediction_length,
            test_length=test_length,
            val_length=val_length,
            storage_path=storage_path,
        )

        timer = EvaluationTimer()
        timer.start()
        eval_data = dataset.test_data
        eval_input_list = list(eval_data.input)
        total_items = len(eval_input_list)
        num_variates = dataset.target_dim
        season_length = get_seasonality(dataset.freq)

        print("  Dataset info:")
        print(f"    - Frequency: {dataset.freq}")
        print(f"    - Num series: {len(dataset.hf_dataset)}")
        print(f"    - Target dim: {num_variates} (Native Multivariate)")
        print(f"    - Test length: {test_length} steps")
        print(f"    - Windows: {dataset.windows}")

        # Compute dynamic batch size
        max_ctx_in_task = max((np.asarray(inp["target"]).shape[-1] for inp in eval_input_list), default=32)
        effective_batch_size = (
            batch_size
            if batch_size is not None
            else get_optimal_batch_size(
                max_context_len=max_ctx_in_task,
                num_variates=num_variates,
                min_batch=4,
                max_batch=64,
            )
        )
        total_batches = math.ceil(total_items / effective_batch_size)
        print(f"  Running predictions: Total items={total_items}, Batch size={effective_batch_size}, Total batches={total_batches}...")

        fc_quantiles_batches = []
        for b_start in range(0, total_items, effective_batch_size):
            b_end = min(b_start + effective_batch_size, total_items)
            batch_contexts = [
                prepare_time_context(eval_input_list[i], max_context_length=context_length)
                for i in range(b_start, b_end)
            ]

            batch_outs = list(
                forecaster.predict_batch(
                    contexts=batch_contexts,
                    horizon=prediction_length,
                    return_quantiles=True,
                    use_symmetric_averaging=True,
                    make_positive=True,
                    sort_quantiles=True,
                )
            )

            batch_q_list = []
            for out in batch_outs:
                # out.quantiles shape: (num_variates, prediction_length, num_quantiles)
                # Permute to expected TIME shape: (num_quantiles, num_variates, prediction_length)
                q = out.quantiles.transpose(2, 0, 1)
                batch_q_list.append(q[np.newaxis, ...])

            batch_q_arr = np.concatenate(batch_q_list, axis=0)
            fc_quantiles_batches.append(batch_q_arr)

            current_batch_idx = (b_start // effective_batch_size) + 1
            if current_batch_idx % 5 == 0 or current_batch_idx == total_batches:
                print(f"    Processed batch [{current_batch_idx}/{total_batches}] ({len(batch_contexts)} series)...")

        fc_quantiles = np.concatenate(fc_quantiles_batches, axis=0)
        inference_seconds = timer.stop()
        ds_config = f"{dataset_name}/{term}"

        model_hyperparams = {
            "model": "TimesFM-3",
            "context_length": context_length,
            "use_variate_attention": True,
            "quantile_levels": quantile_levels,
        }

        metadata = save_window_predictions(
            dataset=dataset,
            fc_quantiles=fc_quantiles,
            ds_config=ds_config,
            output_base_dir=output_dir,
            seasonality=season_length,
            model_hyperparams=model_hyperparams,
            quantile_levels=quantile_levels,
            inference_seconds=inference_seconds,
        )
        print(f"  Completed: {metadata['num_series']} series x {metadata['num_windows']} windows")
        print(f"  Output saved to: {metadata.get('output_dir', output_dir)}")

    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run TimesFM-3 experiments on TIME benchmark")
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        default=["ECDC_COVID/W"],
        help="Dataset name(s). 'all_datasets' for all.",
    )
    parser.add_argument(
        "--terms",
        type=str,
        nargs="+",
        default=None,
        choices=["short", "medium", "long"],
        help="Terms to evaluate. If not specified, auto-detect from config.",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="base",
        help="TimesFM-3 model size",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=os.getenv("TIMESFM3_CHECKPOINT", "google/timesfm-3.0-pytorch"),
        help="Model checkpoint path or Hugging Face repo ID",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for prediction (default: dynamic adaptive batch size)",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=15360,
        help="Maximum context length (default: 15360)",
    )
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=DEFAULT_QUANTILE_LEVELS,
        help="Quantile levels to predict",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to datasets.yaml config file",
    )

    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None

    if len(args.dataset) == 1 and args.dataset[0] == "all_datasets":
        config = load_dataset_config(config_path)
        datasets = list(config.get("datasets", {}).keys())
        print(f"Running all {len(datasets)} datasets from config:")
    else:
        datasets = args.dataset

    total_datasets = len(datasets)
    for idx, dataset_name in enumerate(datasets, 1):
        print(f"\n{'#'*60}")
        print(f"# Dataset {idx}/{total_datasets}: {dataset_name}")
        print(f"{'#'*60}")

        try:
            run_timesfm3_experiment(
                dataset_name=dataset_name,
                terms=args.terms,
                model_size=args.model_size,
                checkpoint_path=args.checkpoint_path,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                context_length=args.context_length,
                quantile_levels=args.quantiles,
                config_path=config_path,
            )
        except Exception as e:
            print(f"ERROR: Failed to run experiment for {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'#'*60}")
    print(f"# All {total_datasets} dataset(s) completed!")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
