"""Prepare compact Adaptime window indices for one or more TIME datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

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
from timebench.paths import dataset_storage_root, outputs_root


def _prediction_length(dataset: str, term: str, configured: int | None, freq: str) -> int:
    if configured is not None:
        return int(configured)
    normalized = norm_freq_str(to_offset(freq).name)
    base = M4_PRED_LENGTH_MAP[normalized] if "m4" in dataset else PRED_LENGTH_MAP[normalized]
    return int(base * Term(term).multiplier)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare leakage-free, Arrow-backed Adaptime window indices"
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        required=True,
        help="TIME dataset path(s), or all_datasets from the YAML configuration",
    )
    parser.add_argument("--terms", nargs="+", choices=("short", "medium", "long"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument(
        "--adaptation-train-length",
        type=int,
        help="Defaults to the dataset's TIME val_length",
    )
    parser.add_argument(
        "--adaptation-validation-length",
        type=int,
        help="Defaults to the dataset's TIME val_length",
    )
    parser.add_argument(
        "--adaptation-stride",
        type=int,
        help="Defaults to one prediction horizon; never changes official test origins",
    )
    parser.add_argument(
        "--datastore-length",
        type=int,
        help="Use this many immediately preceding values; defaults to all earlier history",
    )
    parser.add_argument(
        "--retrieval-period",
        type=int,
        help="Defaults to the cadence seasonality returned by GluonTS",
    )
    parser.add_argument(
        "--datastore-stride-multiple",
        type=int,
        default=1,
        help="Datastore stride as a positive multiple of retrieval_period",
    )
    parser.add_argument(
        "--target-mode",
        nargs="+",
        choices=("univariate", "multivariate"),
        default=("univariate", "multivariate"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=outputs_root() / "adaptime" / "prepared",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_dataset_config(args.config)
    dataset_names = (
        list(config.get("datasets", {}))
        if args.dataset == ["all_datasets"]
        else list(args.dataset)
    )
    if not dataset_names:
        raise ValueError("no datasets selected")
    if args.datastore_stride_multiple <= 0:
        raise ValueError("datastore_stride_multiple must be positive")

    for dataset_name in dataset_names:
        source_path = dataset_storage_root() / dataset_name
        hf_dataset = datasets.load_from_disk(str(source_path))
        if len(hf_dataset) == 0:
            raise ValueError(f"empty TIME dataset: {dataset_name}")
        freq = str(hf_dataset[0]["freq"])
        native_target = np.asarray(hf_dataset[0]["target"])
        native_channels = int(native_target.shape[0]) if native_target.ndim > 1 else 1
        period = int(args.retrieval_period or get_seasonality(freq))
        terms = list(args.terms or get_available_terms(dataset_name, config))
        if not terms:
            raise ValueError(f"no terms configured for {dataset_name!r}")

        for term in terms:
            settings = get_dataset_settings(dataset_name, term, config)
            prediction_length = _prediction_length(
                dataset_name,
                term,
                settings.get("prediction_length"),
                freq,
            )
            validation_length = (
                args.adaptation_validation_length
                if args.adaptation_validation_length is not None
                else settings.get("val_length")
            )
            if validation_length is None or int(validation_length) <= 0:
                raise ValueError(
                    f"{dataset_name}/{term} needs an explicit positive adaptation validation length"
                )
            train_length = (
                args.adaptation_train_length
                if args.adaptation_train_length is not None
                else settings.get("val_length")
            )
            if train_length is None or int(train_length) <= 0:
                raise ValueError(
                    f"{dataset_name}/{term} needs an explicit positive adaptation train length"
                )
            for target_mode in args.target_mode:
                if target_mode == "multivariate" and native_channels < 2:
                    print(f"Skipping {dataset_name}/{term}/multivariate: native target is univariate")
                    continue
                preparation = PreparationConfig(
                    dataset=dataset_name,
                    term=term,
                    context_length=args.context_length,
                    prediction_length=prediction_length,
                    test_length=int(settings["test_length"]),
                    adaptation_train_length=int(train_length),
                    adaptation_validation_length=int(validation_length),
                    target_mode=target_mode,
                    adaptation_stride=args.adaptation_stride,
                    retrieval_period=period,
                    datastore_stride=period * args.datastore_stride_multiple,
                    datastore_length=args.datastore_length,
                )
                output = args.output_root / target_mode / dataset_name / term
                manifest = prepare_adaptation_dataset(
                    hf_dataset,
                    preparation,
                    output,
                    source_path=source_path,
                )
                print(manifest)


if __name__ == "__main__":
    main()
