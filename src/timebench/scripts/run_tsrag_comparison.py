"""Run the matched TS-RAG and full-ridge TIME comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

from timebench.pipeline.tsrag_workflow import (
    TSRAGWorkflowConfig,
    run_tsrag_comparison,
)


def _csv(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate pinned TS-RAG on the official TIME support used by full ridge"
    )
    parser.add_argument("--datasets", type=_csv, default=("all_datasets",))
    parser.add_argument("--terms", type=_csv)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--ridge-output-root", type=Path)
    parser.add_argument("--ridge-launch-id")
    parser.add_argument("--chronos-bolt-path", type=Path)
    parser.add_argument("--retriever-path", type=Path)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-batch-size", type=int, default=256)
    parser.add_argument("--arrow-cache-items", type=int, default=2)
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
    run_tsrag_comparison(
        TSRAGWorkflowConfig(
            device=args.device,
            model_batch_size=args.model_batch_size,
            arrow_cache_items=args.arrow_cache_items,
            chronos_bolt_path=args.chronos_bolt_path,
            retriever_path=args.retriever_path,
            checkpoint_path=args.checkpoint_path,
        ),
        dataset_config_path=args.config,
        datasets_selected=args.datasets,
        terms_selected=args.terms,
        output_root=args.output_root,
        ridge_output_root=args.ridge_output_root,
        ridge_launch_id=args.ridge_launch_id,
        config_policy=args.config_policy,
        repeat_policy=args.repeat_policy,
    )


if __name__ == "__main__":
    main()
