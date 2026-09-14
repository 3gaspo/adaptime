"""Prepare compact Adaptime window indices for one or more TIME datasets."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import datasets
import numpy as np
from gluonts.time_feature import get_seasonality, norm_freq_str
from pandas.tseries.frequencies import to_offset

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
from timebench.paths import dataset_storage_root, outputs_root


def _prediction_length(dataset: str, term: str, configured: int | None, freq: str) -> int:
    if configured is not None:
        return int(configured)
    normalized = norm_freq_str(to_offset(freq).name)
    base = M4_PRED_LENGTH_MAP[normalized] if "m4" in dataset else PRED_LENGTH_MAP[normalized]
    return int(base * Term(term).multiplier)


def _task_protocol(
    dataset: str, term: str, config: dict[str, object]
) -> dict[str, int]:
    task = dict(dict(config.get("adaptime_tasks", {})).get(dataset, {}))
    ranges = dict(task.pop("ranges", {}))
    return {
        name: int(value)
        for name, value in {**task, **dict(ranges.get(term, {}))}.items()
        if name
        in {
            "alignment_period",
            "datastore_stride",
            "fitting_stride",
            "retrieval_context_length",
        }
    }


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
        "--retrieval-context-length",
        type=int,
        help="Fixed retrieval lookback; task YAML then horizon/period fallback",
    )
    parser.add_argument(
        "--adaptation-stride",
        type=int,
        help="Defaults by dataset frequency; never changes official test origins",
    )
    parser.add_argument(
        "--max-datastore-windows",
        type=int,
        help="Global balanced cap; keeps the most recent complete dates per variate",
    )
    parser.add_argument(
        "--max-fitting-windows",
        type=int,
        help="Global balanced training cap; keeps the most recent strided dates per variate",
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
        "--retrieval-scope",
        choices=("all", "same_series", "other_series"),
        default="all",
    )
    parser.add_argument(
        "--fitting-window-scope",
        choices=("all", "same_series"),
        default="all",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=outputs_root() / "adaptime" / "prepared",
    )
    parser.add_argument(
        "--datastore-prediction-length",
        type=int,
        default=64,
        help="Future support shared by Ridge and TS-RAG datastore rows",
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
        terms = list(args.terms or get_available_terms(dataset_name, config))
        if not terms:
            raise ValueError(f"no terms configured for {dataset_name!r}")

        for term in terms:
            protocol = _task_protocol(dataset_name, term, config)
            period = int(
                args.retrieval_period
                or protocol.get("alignment_period")
                or get_seasonality(freq)
            )
            settings = get_dataset_settings(dataset_name, term, config)
            prediction_length = _prediction_length(
                dataset_name,
                term,
                settings.get("prediction_length"),
                freq,
            )
            adaptation_stride = int(
                args.adaptation_stride
                or protocol.get("fitting_stride")
                or adaptation_stride_for_frequency(freq)
            )
            retrieval_context_length = int(
                args.retrieval_context_length
                or protocol.get("retrieval_context_length")
                or min(
                    args.context_length,
                    period * math.ceil(prediction_length / period),
                )
            )
            if retrieval_context_length > args.context_length:
                raise ValueError(
                    f"{dataset_name}/{term} retrieval context exceeds context_length"
                )
            train_length, validation_length, _ = adaptation_split_lengths(
                int(settings["test_length"]),
                prediction_length,
                adaptation_stride,
            )
            for target_mode in args.target_mode:
                if target_mode == "multivariate" and native_channels < 2:
                    print(f"Skipping {dataset_name}/{term}/multivariate: native target is univariate")
                    continue
                preparation = PreparationConfig(
                    dataset=dataset_name,
                    term=term,
                    context_length=args.context_length,
                    retrieval_context_length=retrieval_context_length,
                    reference_context_length=max(
                        retrieval_context_length, min(args.context_length, 512)
                    ),
                    prediction_length=prediction_length,
                    test_length=int(settings["test_length"]),
                    adaptation_train_length=int(train_length),
                    adaptation_validation_length=int(validation_length),
                    seasonality=int(get_seasonality(freq)),
                    target_mode=target_mode,
                    adaptation_stride=adaptation_stride,
                    retrieval_period=period,
                    datastore_stride=int(
                        (
                            period
                            if args.retrieval_period is not None
                            else protocol.get("datastore_stride", period)
                        )
                        * args.datastore_stride_multiple
                    ),
                    max_datastore_windows=args.max_datastore_windows,
                    max_fitting_windows=args.max_fitting_windows,
                    datastore_scope=args.retrieval_scope,
                    fitting_window_scope=args.fitting_window_scope,
                    datastore_prediction_length=max(
                        prediction_length, args.datastore_prediction_length
                    ),
                    minimum_datastore_dates_per_variate=11,
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
