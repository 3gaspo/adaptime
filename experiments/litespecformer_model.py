"""
LiteSpecFormer model experiments for time series forecasting.

LiteSpecFormer is a univariate (single-channel) zero-shot forecasting model.
Multivariate datasets are evaluated with channel-independent forecasting
(`to_univariate=True`). Context length must be at least 64.

Usage:
    python experiments/litespecformer_model.py
    python experiments/litespecformer_model.py --model-id FlowVortex/LiteSpecFormer-1.0-36M
    python experiments/litespecformer_model.py --dataset "SG_Weather/D" --terms short medium long
    python experiments/litespecformer_model.py --dataset "SG_Weather/D" "SG_PM25/H"
    python experiments/litespecformer_model.py --dataset all_datasets
"""

import argparse
import logging
import os
import re
import sys
import traceback
from pathlib import Path

# Ensure timebench is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from dotenv import load_dotenv
from gluonts.dataset.common import ProcessDataEntry as _ProcessDataEntry
from gluonts.time_feature import get_seasonality
from litespecformer import LiteSpecFormerPipeline

import timebench.evaluation.data as tb_data


def _normalize_gluonts_freq(freq) -> str:
    """Normalize freq for GluonTS ProcessDataEntry / pd.Period."""
    freq = str(freq).strip()
    if re.fullmatch(r"\d*T", freq):
        n = freq[:-1] or "1"
        freq = f"{n}min"
    freq = freq.replace("H", "h").replace("S", "s")
    return freq


def _normalize_seasonality_freq(freq) -> str:
    """Normalize freq for get_seasonality / pd.to_offset."""
    freq = _normalize_gluonts_freq(freq)
    if freq == "M":
        freq = "ME"
    elif freq == "Q":
        freq = "QE"
    return freq


class _CompatProcessDataEntry(_ProcessDataEntry):
    """Normalize numpy.str_ / legacy freq aliases before GluonTS processing."""

    def __init__(self, freq, one_dim_target: bool = True, use_timestamp: bool = False):
        super().__init__(
            _normalize_gluonts_freq(freq),
            one_dim_target=one_dim_target,
            use_timestamp=use_timestamp,
        )


tb_data.ProcessDataEntry = _CompatProcessDataEntry

from timebench.evaluation.data import (
    Dataset,
    get_dataset_settings,
    load_dataset_config,
)
from timebench.evaluation.saver import save_window_predictions
from timebench.evaluation.utils import clean_nan_target, get_available_terms

load_dotenv()

logging.getLogger("litespecformer").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

DEFAULT_QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MIN_CONTEXT_LENGTH = 64
DEFAULT_MODEL_ID = "FlowVortex/LiteSpecFormer-1.0-36M"


def _seasonality_from_freq(freq) -> int:
    """Return seasonal period length for MASE computation."""
    return get_seasonality(_normalize_seasonality_freq(freq))


def _load_pipeline(model_id: str, device: str) -> LiteSpecFormerPipeline:
    print(f"  Loading LiteSpecFormer pipeline ({model_id})...")
    pipeline = LiteSpecFormerPipeline.from_pretrained(
        model_id,
        dtype=torch.float32,
    )
    pipeline.model.to(device)
    pipeline.model.eval()
    return pipeline


def _resolve_context_length(
    context_length: int | None,
    pipeline: LiteSpecFormerPipeline,
) -> int:
    if context_length is None:
        context_length = pipeline.model_context_length

    if context_length < MIN_CONTEXT_LENGTH:
        raise ValueError(f"context_length must be >= {MIN_CONTEXT_LENGTH}, got {context_length}")

    if context_length > pipeline.model_context_length:
        print(
            f"  Warning: context_length {context_length} exceeds model default "
            f"{pipeline.model_context_length}; clipping to model default."
        )
        context_length = pipeline.model_context_length

    return context_length


def _prepare_context(series, context_length: int) -> torch.Tensor:
    """Prepare a univariate context tensor with left NaN padding and truncation."""
    series = clean_nan_target(np.asarray(series))
    if series.ndim == 2:
        series = series.squeeze(0)

    ts = torch.tensor(series, dtype=torch.float32)
    current_length = ts.shape[-1]

    if current_length < MIN_CONTEXT_LENGTH:
        padding_size = MIN_CONTEXT_LENGTH - current_length
        ts = torch.nn.functional.pad(ts, (padding_size, 0), mode="constant", value=float("nan"))
        current_length = ts.shape[-1]

    if current_length < context_length:
        padding_size = context_length - current_length
        ts = torch.nn.functional.pad(ts, (padding_size, 0), mode="constant", value=float("nan"))
    elif current_length > context_length:
        ts = ts[..., -context_length:]

    return ts


def _quantile_to_numpy(
    quantile_tensor,
    prediction_length: int,
    num_quantiles: int,
) -> np.ndarray:
    """Convert LiteSpecFormer output to (num_quantiles, prediction_length)."""
    q_np = (
        quantile_tensor.cpu().float().numpy()
        if isinstance(quantile_tensor, torch.Tensor)
        else quantile_tensor
    )

    if q_np.ndim == 3:
        # (n_variates, prediction_length, num_quantiles)
        if q_np.shape[-1] == num_quantiles:
            q_np = q_np[0].T
        elif q_np.shape[1] == num_quantiles:
            q_np = q_np[0]
    elif q_np.ndim == 2:
        if q_np.shape[0] == prediction_length and q_np.shape[1] == num_quantiles:
            q_np = q_np.T

    if q_np.shape != (num_quantiles, prediction_length):
        raise ValueError(
            f"Unexpected quantile shape {q_np.shape}, "
            f"expected ({num_quantiles}, {prediction_length})"
        )

    return q_np


def run_litespecformer_experiment(
    dataset_name: str,
    terms: list[str] | None = None,
    model_id: str = DEFAULT_MODEL_ID,
    output_dir: str | None = None,
    batch_size: int = 32,
    context_length: int | None = None,
    config_path: Path | None = None,
    quantile_levels: list[float] | None = None,
    pipeline: LiteSpecFormerPipeline | None = None,
):
    print("Loading configuration...")
    config = load_dataset_config(config_path)

    if terms is None:
        terms = get_available_terms(dataset_name, config)
        if not terms:
            raise ValueError(f"No terms defined for dataset '{dataset_name}' in config")

    if quantile_levels is None:
        quantile_levels = DEFAULT_QUANTILE_LEVELS

    if output_dir is None:
        output_dir = "./output/results/litespecformer"

    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Model: {model_id}")
    print(f"Terms: {terms}")
    print(f"{'='*60}")

    if pipeline is None:
        pipeline = _load_pipeline(model_id, device)

    context_length = _resolve_context_length(context_length, pipeline)

    print(f"  Device: {device}")
    print(f"  Context length: {context_length} (min={MIN_CONTEXT_LENGTH})")

    for term in terms:
        print(f"\n--- Term: {term} ---")
        settings = get_dataset_settings(dataset_name, term, config)
        prediction_length = settings.get("prediction_length")
        test_length = settings.get("test_length")
        val_length = settings.get("val_length")

        print(
            f"  Config: prediction_length={prediction_length}, "
            f"test_length={test_length}, val_length={val_length}"
        )

        # LiteSpecFormer only supports univariate forecasting
        to_univariate = (
            False
            if Dataset(name=dataset_name, term=term, to_univariate=False).target_dim == 1
            else True
        )
        dataset = Dataset(
            name=dataset_name,
            term=term,
            to_univariate=to_univariate,
            prediction_length=prediction_length,
            test_length=test_length,
            val_length=val_length,
        )

        data_length = test_length
        num_windows = dataset.windows
        split_name = "Test split"
        eval_data = dataset.test_data

        print("  Dataset info:")
        print(f"    - Frequency: {dataset.freq}")
        print(f"    - Num series: {len(dataset.hf_dataset)}")
        print(f"    - Target dim: {dataset.target_dim}")
        print(f"    - to_univariate: {to_univariate}")
        print(
            "    - Series length: "
            f"min={dataset._min_series_length}, "
            f"max={dataset._max_series_length}, "
            f"avg={dataset._avg_series_length:.1f}"
        )
        print(f"    - {split_name}: {data_length} steps")
        print(f"    - Prediction length: {dataset.prediction_length}")
        print(f"    - Windows: {num_windows}")

        season_length = _seasonality_from_freq(dataset.freq)

        print(f"  Preparing input batches from {split_name} data...")
        all_inputs = []
        for inp in eval_data.input:
            all_inputs.append(_prepare_context(inp["target"], context_length))

        num_total_instances = len(all_inputs)
        print(f"    Total instances to forecast: {num_total_instances}")

        fc_quantiles_batches = []
        print(f"  Running predictions (batch size: {batch_size})...")

        with torch.no_grad():
            for start in range(0, num_total_instances, batch_size):
                end = min(start + batch_size, num_total_instances)
                batch_contexts = all_inputs[start:end]
                # LiteSpecFormer expects a 2D tensor (n_series, history_length), not a list
                batch_tensor = torch.stack(batch_contexts)

                batch_quantiles, _ = pipeline.predict_quantiles(
                    inputs=batch_tensor,
                    prediction_length=prediction_length,
                    quantile_levels=quantile_levels,
                    batch_size=len(batch_contexts),
                    context_length=context_length,
                    limit_prediction_length=False,
                )

                batch_q_list = []
                for q in batch_quantiles:
                    q_np = _quantile_to_numpy(
                        q,
                        prediction_length=prediction_length,
                        num_quantiles=len(quantile_levels),
                    )
                    batch_q_list.append(q_np[np.newaxis, ...])

                batch_q_array = np.concatenate(batch_q_list, axis=0)
                fc_quantiles_batches.append(batch_q_array)

                if (start // batch_size + 1) % 10 == 0 or end == num_total_instances:
                    print(f"    Processed {end}/{num_total_instances}...")

        # Shape: (num_total_instances, num_quantiles, prediction_length)
        fc_quantiles = np.concatenate(fc_quantiles_batches, axis=0)

        ds_config = f"{dataset_name}/{term}"
        model_hyperparams = {
            "model": "litespecformer",
            "model_id": model_id,
            "context_length": context_length,
            "min_context_length": MIN_CONTEXT_LENGTH,
            "quantile_levels": quantile_levels,
            "to_univariate": to_univariate,
        }

        metadata = save_window_predictions(
            dataset=dataset,
            fc_quantiles=fc_quantiles,
            ds_config=ds_config,
            output_base_dir=output_dir,
            seasonality=season_length,
            model_hyperparams=model_hyperparams,
            quantile_levels=quantile_levels,
        )

        print(f"  Completed: {metadata['num_series']} series × {metadata['num_windows']} windows")
        print(f"  Output: {metadata.get('output_dir', output_dir)}")

    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run LiteSpecFormer experiments")
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        default=["Global_Influenza/W"],
        help="Dataset name(s). Can be a single dataset, multiple datasets, or 'all_datasets'",
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
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model ID or local path for LiteSpecFormer",
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
        default=32,
        help="Batch size for prediction",
    )
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=DEFAULT_QUANTILE_LEVELS,
        help="Quantile levels to predict",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=None,
        help=(
            "Maximum context length for inference. "
            f"Must be >= {MIN_CONTEXT_LENGTH}. Defaults to the model's context length."
        ),
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
        for ds in datasets:
            print(f"  - {ds}")
    else:
        datasets = args.dataset

    total_datasets = len(datasets)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = None
    context_length = args.context_length

    if total_datasets > 0:
        pipeline = _load_pipeline(args.model_id, device)
        context_length = _resolve_context_length(args.context_length, pipeline)
        print(f"Using device: {device}, context length: {context_length}")

    for idx, dataset_name in enumerate(datasets, 1):
        print(f"\n{'#'*60}")
        print(f"# Dataset {idx}/{total_datasets}: {dataset_name}")
        print(f"{'#'*60}")

        try:
            run_litespecformer_experiment(
                dataset_name=dataset_name,
                terms=args.terms,
                model_id=args.model_id,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                context_length=context_length,
                config_path=config_path,
                quantile_levels=args.quantiles,
                pipeline=pipeline,
            )
        except Exception as e:
            print(f"ERROR: Failed to run experiment for {dataset_name}: {e}")
            traceback.print_exc()
            continue

    print(f"\n{'#'*60}")
    print(f"# All {total_datasets} dataset(s) completed!")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
