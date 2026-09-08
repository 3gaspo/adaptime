"""Generate frozen full-ridge point forecasts without evaluating them."""

from __future__ import annotations

import argparse
from pathlib import Path

from timebench.pipeline.adaptation_prediction import (
    PredictionConfig,
    predict_frozen_ridge,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen full-ridge inference")
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--extraction", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = predict_frozen_ridge(
        args.prepared,
        args.extraction,
        args.model,
        PredictionConfig(chunk_size=args.chunk_size),
        args.output_dir,
    )
    print(manifest)


if __name__ == "__main__":
    main()
