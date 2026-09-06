"""Focused regression checks for finite-pair and scaled-MASE behavior."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS_PATH = PROJECT_ROOT / "src/timebench/evaluation/metrics.py"
SPEC = importlib.util.spec_from_file_location("timebench_metrics", METRICS_PATH)
assert SPEC is not None and SPEC.loader is not None
METRICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(METRICS)


def main() -> None:
    # Removing the internal NaN would incorrectly produce mean(|3-1|, |4-3|)=1.5.
    context = np.asarray([1.0, np.nan, 3.0, 4.0, np.nan])
    assert METRICS.seasonal_naive_scale(context, 1) == 1.0
    assert METRICS.seasonal_naive_scale(context, 1, squared=True) == 1.0

    summary = (PROJECT_ROOT / "scripts/compute_foundation_summary.py").read_text(
        encoding="utf-8"
    )
    channel = (PROJECT_ROOT / "src/slurm/run_chronos2_comparison.sh").read_text(
        encoding="utf-8"
    )
    assert "scaled_MASE" in summary
    assert "geometric_mean_over_tasks" in summary
    assert "--seasonal-naive-results-dir" in channel
    ast.parse(summary)
    print("Finite-pair MASE and Seasonal-Naive-scaled comparison contract passed.")


if __name__ == "__main__":
    main()
