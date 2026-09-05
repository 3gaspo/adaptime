"""Extract reusable Adaptime retrieval and foundation-model quantities."""

from __future__ import annotations

import argparse
from pathlib import Path

from timebench.evaluation.adaptation_data import PreparedDataset
from timebench.model_loading import load_adaptime_forecaster
from timebench.pipeline.adaptime_extraction import ExtractionConfig, extract_adaptation_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Adaptime full-ridge inputs")
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=("chronos_bolt", "chronos2", "ts_icl", "seasonal_naive"),
        required=True,
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--weights-id")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--representation", choices=("raw", "instance", "model"), default="instance")
    parser.add_argument("--distance-metric", choices=("euclidean", "cosine"), default="euclidean")
    parser.add_argument("--retrieval-scope", choices=("all", "same_series", "other_series"), default="all")
    parser.add_argument("--minimum-overlap-fraction", type=float, default=0.8)
    parser.add_argument("--max-k", type=int, default=15)
    parser.add_argument("--context-k", type=int, nargs="+", default=(1, 5, 10, 15))
    parser.add_argument("--model-batch-size", type=int, default=64)
    parser.add_argument("--query-block-size", type=int, default=256)
    parser.add_argument("--datastore-block-size", type=int, default=4096)
    parser.add_argument("--arrow-cache-items", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepared = PreparedDataset(args.prepared)
    forecaster = load_adaptime_forecaster(
        args.model,
        horizon=prepared.prediction_length,
        period=int(prepared.config["retrieval_period"]),
        model_path=args.model_path,
        weights_id=args.weights_id,
        device=args.device,
    )
    manifest = extract_adaptation_features(
        args.prepared,
        forecaster,
        ExtractionConfig(
            representation=args.representation,
            distance_metric=args.distance_metric,
            retrieval_scope=args.retrieval_scope,
            minimum_overlap_fraction=args.minimum_overlap_fraction,
            max_k=args.max_k,
            context_k=tuple(sorted(set(args.context_k))),
            model_batch_size=args.model_batch_size,
            query_block_size=args.query_block_size,
            datastore_block_size=args.datastore_block_size,
            arrow_cache_items=args.arrow_cache_items,
        ),
        args.output_dir,
    )
    print(manifest)


if __name__ == "__main__":
    main()
