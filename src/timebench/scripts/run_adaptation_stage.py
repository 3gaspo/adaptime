"""Run the TIME-wide Adaptime task workflow from local or Slurm orchestration."""

from __future__ import annotations

import argparse
from pathlib import Path

from timebench.model_loading.adaptime import MODEL_ALIASES
from timebench.pipeline.adaptime_workflow import (
    AdaptimeWorkflowConfig,
    run_adaptation_stage,
)
from timebench.pipeline.adaptime_rolling import ROLLING_RIDGE_METHOD


def _csv(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one independent Adaptime phase over TIME tasks"
    )
    parser.add_argument(
        "--stage",
        choices=(
            "prepare",
            "vanilla",
            "extract",
            "fit",
            "extract_eval",
            "predict",
            "evaluate",
            "report",
            "pipeline",
            "all",
        ),
        required=True,
    )
    parser.add_argument(
        "--method",
        choices=(
            "seasonal_naive",
            "vanilla",
            "ridge",
            ROLLING_RIDGE_METHOD,
            "tsrag",
            "unified",
        ),
        required=True,
    )
    parser.add_argument("--datasets", type=_csv, default=("all_datasets",))
    parser.add_argument(
        "--exclude-datasets",
        type=_csv,
        default=(),
        help="Comma-separated dataset/frequency identifiers omitted from every stage",
    )
    parser.add_argument("--terms", type=_csv)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--seasonal-results-path",
        type=Path,
        help="Completed shared Seasonal Naive evaluation root used by reports",
    )
    parser.add_argument("--model", choices=MODEL_ALIASES, default="chronos2")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--weights-id")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-mode", choices=("univariate",), default="univariate")
    parser.add_argument("--adaptation-stride", type=int)
    parser.add_argument("--retrieval-period", type=int)
    parser.add_argument("--retrieval-context-length", type=int)
    parser.add_argument("--datastore-stride-multiple", type=int, default=1)
    parser.add_argument("--max-datastore-windows", type=int)
    parser.add_argument("--max-fitting-windows", type=int)
    parser.add_argument("--representation", choices=("raw", "instance", "model"), default="instance")
    parser.add_argument("--distance-metric", choices=("euclidean", "cosine"), default="euclidean")
    parser.add_argument("--retrieval-scope", choices=("all", "same_series", "other_series"), default="all")
    parser.add_argument("--minimum-overlap-fraction", type=float, default=0.8)
    parser.add_argument("--minimum-query-finite-fraction", type=float, default=0.8)
    parser.add_argument("--max-k", type=int, default=15)
    parser.add_argument("--k", type=int, nargs="+", default=(1, 5, 10, 15))
    parser.add_argument("--alpha", type=float, nargs="+", default=(1e-3, 1e-2, 1e-1))
    parser.add_argument("--model-batch-size", type=int, default=64)
    parser.add_argument("--query-block-size", type=int, default=256)
    parser.add_argument("--datastore-block-size", type=int, default=4096)
    parser.add_argument("--arrow-cache-items", type=int, default=2)
    parser.add_argument("--ridge-chunk-size", type=int, default=1024)
    parser.add_argument("--bootstrap-replications", type=int, default=1000)
    parser.add_argument("--bootstrap-block-length", type=int)
    parser.add_argument(
        "--fitting-scope",
        nargs="+",
        choices=("all", "same_series"),
        default=("all", "same_series"),
    )
    parser.add_argument("--rolling-k", type=int, default=15)
    parser.add_argument("--rolling-alpha", type=float, default=1.0)
    parser.add_argument("--rolling-n-fitting-dates", type=int, default=100)
    parser.add_argument("--rolling-minimum-fitting-dates", type=int, default=64)
    parser.add_argument("--rolling-fitting-stride-multiple", type=int, default=1)
    parser.add_argument("--rolling-max-datastore-windows", type=int, default=10_000)
    parser.add_argument("--rolling-datastore-stride-multiple", type=int, default=1)
    parser.add_argument("--rolling-datastore-block-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--tsrag-model-batch-size", type=int, default=256)
    parser.add_argument("--tsrag-chronos-bolt-path", type=Path)
    parser.add_argument("--tsrag-retriever-path", type=Path)
    parser.add_argument("--tsrag-checkpoint-path", type=Path)
    parser.add_argument(
        "--ridge-results-path",
        type=Path,
        help=(
            "Use matching full-ridge evaluations in the separate TS-RAG report"
        ),
    )
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
        args.method,
        AdaptimeWorkflowConfig(
            model=args.model,
            target_mode=args.target_mode,
            adaptation_stride=args.adaptation_stride,
            retrieval_period=args.retrieval_period,
            retrieval_context_length=args.retrieval_context_length,
            datastore_stride_multiple=args.datastore_stride_multiple,
            max_datastore_windows=args.max_datastore_windows,
            max_fitting_windows=args.max_fitting_windows,
            representation=args.representation,
            distance_metric=args.distance_metric,
            retrieval_scope=args.retrieval_scope,
            minimum_overlap_fraction=args.minimum_overlap_fraction,
            minimum_query_finite_fraction=args.minimum_query_finite_fraction,
            max_k=args.max_k,
            k_values=tuple(sorted(set(args.k))),
            alpha_values=tuple(args.alpha),
            model_batch_size=args.model_batch_size,
            query_block_size=args.query_block_size,
            datastore_block_size=args.datastore_block_size,
            arrow_cache_items=args.arrow_cache_items,
            ridge_chunk_size=args.ridge_chunk_size,
            bootstrap_replications=args.bootstrap_replications,
            bootstrap_block_length=args.bootstrap_block_length,
            fitting_scopes=tuple(args.fitting_scope),
            rolling_k=args.rolling_k,
            rolling_alpha=args.rolling_alpha,
            rolling_n_fitting_dates=args.rolling_n_fitting_dates,
            rolling_minimum_fitting_dates=args.rolling_minimum_fitting_dates,
            rolling_fitting_stride_multiple=args.rolling_fitting_stride_multiple,
            rolling_max_datastore_windows=args.rolling_max_datastore_windows,
            rolling_datastore_stride_multiple=args.rolling_datastore_stride_multiple,
            rolling_datastore_block_size=args.rolling_datastore_block_size,
            seed=args.seed,
            model_path=args.model_path,
            weights_id=args.weights_id,
            device=args.device,
            tsrag_model_batch_size=args.tsrag_model_batch_size,
            tsrag_chronos_bolt_path=args.tsrag_chronos_bolt_path,
            tsrag_retriever_path=args.tsrag_retriever_path,
            tsrag_checkpoint_path=args.tsrag_checkpoint_path,
        ),
        dataset_config_path=args.config,
        datasets_selected=args.datasets,
        excluded_datasets=args.exclude_datasets,
        terms_selected=args.terms,
        output_root=args.output_root,
        seasonal_results_path=args.seasonal_results_path,
        ridge_results_path=args.ridge_results_path,
        config_policy=args.config_policy,
        repeat_policy=args.repeat_policy,
    )


if __name__ == "__main__":
    main()
