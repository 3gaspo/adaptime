"""Run the TIME-wide Adaptime task workflow from local or Slurm orchestration."""

from __future__ import annotations

import argparse
from pathlib import Path

from timebench.model_loading.adaptime import MODEL_ALIASES
from timebench.pipeline.adaptime_workflow import (
    AdaptimeWorkflowConfig,
    run_adaptation_stage,
)


def _csv(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Adaptime tasks over TIME")
    parser.add_argument("--stage", choices=("run",), default="run")
    parser.add_argument("--datasets", type=_csv, default=("all_datasets",))
    parser.add_argument("--terms", type=_csv)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--model", choices=MODEL_ALIASES, default="chronos2")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--weights-id")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-mode", choices=("univariate",), default="univariate")
    parser.add_argument("--max-context-length", type=int, default=2048)
    parser.add_argument("--adaptation-train-length", type=int)
    parser.add_argument("--adaptation-validation-length", type=int)
    parser.add_argument("--adaptation-stride", type=int)
    parser.add_argument("--retrieval-period", type=int)
    parser.add_argument("--datastore-stride-multiple", type=int, default=1)
    parser.add_argument("--datastore-length", type=int)
    parser.add_argument("--representation", choices=("raw", "instance", "model"), default="instance")
    parser.add_argument("--distance-metric", choices=("euclidean", "cosine"), default="euclidean")
    parser.add_argument("--retrieval-scope", choices=("all", "same_series", "other_series"), default="all")
    parser.add_argument("--minimum-overlap-fraction", type=float, default=0.8)
    parser.add_argument("--max-k", type=int, default=15)
    parser.add_argument("--k", type=int, nargs="+", default=(1, 5, 10, 15))
    parser.add_argument("--alpha", type=float, nargs="+", default=(1e-3, 1e-2, 1e-1))
    parser.add_argument("--model-batch-size", type=int, default=64)
    parser.add_argument("--query-block-size", type=int, default=256)
    parser.add_argument("--datastore-block-size", type=int, default=4096)
    parser.add_argument("--arrow-cache-items", type=int, default=2)
    parser.add_argument("--ridge-chunk-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--config-policy",
        choices=("error", "distinct", "latest", "average"),
        default="error",
    )
    parser.add_argument(
        "--repeat-policy",
        choices=("selected", "latest", "distinct", "average"),
        default="selected",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_adaptation_stage(
        args.stage,
        AdaptimeWorkflowConfig(
            model=args.model,
            target_mode=args.target_mode,
            max_context_length=args.max_context_length,
            adaptation_train_length=args.adaptation_train_length,
            adaptation_validation_length=args.adaptation_validation_length,
            adaptation_stride=args.adaptation_stride,
            retrieval_period=args.retrieval_period,
            datastore_stride_multiple=args.datastore_stride_multiple,
            datastore_length=args.datastore_length,
            representation=args.representation,
            distance_metric=args.distance_metric,
            retrieval_scope=args.retrieval_scope,
            minimum_overlap_fraction=args.minimum_overlap_fraction,
            max_k=args.max_k,
            k_values=tuple(sorted(set(args.k))),
            alpha_values=tuple(args.alpha),
            model_batch_size=args.model_batch_size,
            query_block_size=args.query_block_size,
            datastore_block_size=args.datastore_block_size,
            arrow_cache_items=args.arrow_cache_items,
            ridge_chunk_size=args.ridge_chunk_size,
            seed=args.seed,
            model_path=args.model_path,
            weights_id=args.weights_id,
            device=args.device,
        ),
        dataset_config_path=args.config,
        datasets_selected=args.datasets,
        terms_selected=args.terms,
        output_root=args.output_root,
        config_policy=args.config_policy,
        repeat_policy=args.repeat_policy,
    )


if __name__ == "__main__":
    main()
