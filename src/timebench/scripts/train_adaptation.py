"""Fit and select Adaptime's disk-backed full shared ridge adaptor."""

from __future__ import annotations

import argparse
from pathlib import Path

from timebench.pipeline.adaptime_training import RidgeTrainingConfig, fit_full_ridge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Adaptime full_ridge_shared")
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--extraction", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, nargs="+", default=(1, 5, 10, 15))
    parser.add_argument("--alpha", type=float, nargs="+", default=(1e-3, 1e-2, 1e-1))
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = fit_full_ridge(
        args.prepared,
        args.extraction,
        RidgeTrainingConfig(
            k_values=tuple(sorted(set(args.k))),
            alpha_values=tuple(args.alpha),
            chunk_size=args.chunk_size,
            seed=args.seed,
        ),
        args.output_dir,
    )
    print(manifest)


if __name__ == "__main__":
    main()
