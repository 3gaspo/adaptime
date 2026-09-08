"""Evaluate completed wrapper predictions with TIME's common evaluator."""

from __future__ import annotations

import argparse
from pathlib import Path

from timebench.evaluation.adaptation import evaluate_point_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate point predictions on the official TIME test rows"
    )
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = evaluate_point_predictions(
        args.prepared,
        args.prediction,
        args.output_dir,
    )
    print(args.output_dir / metadata["metrics_summary_file"])


if __name__ == "__main__":
    main()
