#!/usr/bin/env python3
"""Plot model MASE against selected or top-correlated dataset features."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from timebench.feature.performance import analyze_feature_performance
from timebench.paths import dataset_metadata_root, foundation_experiment_root
from timebench.pipeline import parse_config_filters


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot dataset-level MASE against TIME features."
    )
    parser.add_argument(
        "--features-root",
        type=Path,
        default=dataset_metadata_root() / "stl_features",
        help="Root containing {dataset}/{freq}/{split}_dataset.csv summaries",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=foundation_experiment_root(),
        help="One manifest-based experiment root",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=foundation_experiment_root().parent
        / "feature_analysis"
        / "manual"
        / "mase_vs_features.svg",
        help="SVG plot to write",
    )
    parser.add_argument("--split", choices=["full", "test"], default="full")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument(
        "--launch-id",
        default=None,
        help="Include only result tasks stamped with this launch ID",
    )
    parser.add_argument(
        "--target-mode",
        nargs="+",
        choices=("univariate", "multivariate"),
        default=None,
    )
    parser.add_argument("--run-config", action="append", default=[])
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
    parser.add_argument(
        "--features",
        nargs="+",
        default=None,
        help="Explicit feature columns; otherwise select the top correlated features",
    )
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    joined, correlations, selected = analyze_feature_performance(
        features_root=args.features_root,
        results_root=args.results_dir,
        output_svg=args.output,
        split=args.split,
        models=set(args.models) if args.models else None,
        launch_id=args.launch_id,
        target_modes=None if args.target_mode is None else set(args.target_mode),
        config_filters=parse_config_filters(args.run_config),
        config_policy=args.config_policy,
        repeat_policy=args.repeat_policy,
        features=args.features,
        top=args.top,
    )
    data_path = args.output.with_name(f"{args.output.stem}_data.csv")
    correlation_path = args.output.with_name(f"{args.output.stem}_correlations.csv")
    joined.to_csv(data_path, index=False)
    correlations.to_csv(correlation_path, index=False)
    print(f"Selected features: {', '.join(selected)}")
    print(f"Plot: {args.output}")
    print(f"Joined data: {data_path}")
    print(f"Correlations: {correlation_path}")


if __name__ == "__main__":
    main()
