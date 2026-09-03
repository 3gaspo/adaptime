"""Evaluate a frozen Adaptime model on the untouched TIME test interval."""

from __future__ import annotations

import argparse
from pathlib import Path

from timebench.pipeline.adaptime_testing import (
    AdaptimeTestingConfig,
    evaluate_frozen_adaptation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare vanilla, retrieval-covariate, and Adaptime forecasts"
    )
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--extraction", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = evaluate_frozen_adaptation(
        args.prepared,
        args.extraction,
        args.model,
        AdaptimeTestingConfig(chunk_size=args.chunk_size),
        args.output_dir,
    )
    print(manifest)


if __name__ == "__main__":
    main()
