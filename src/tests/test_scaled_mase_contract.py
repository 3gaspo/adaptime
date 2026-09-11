"""Focused regression checks for finite-pair and scaled-MASE behavior."""

from __future__ import annotations

import importlib.util
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS_PATH = PROJECT_ROOT / "src/timebench/evaluation/metrics.py"
SPEC = importlib.util.spec_from_file_location("timebench_metrics", METRICS_PATH)
assert SPEC is not None and SPEC.loader is not None
METRICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(METRICS)

PIPELINE = types.ModuleType("timebench.pipeline")
PIPELINE.select_completed_runs = lambda *args, **kwargs: []
sys.modules.setdefault("timebench", types.ModuleType("timebench"))
sys.modules["timebench.pipeline"] = PIPELINE
PERFORMANCE_PATH = PROJECT_ROOT / "src/timebench/feature/performance.py"
PERFORMANCE_SPEC = importlib.util.spec_from_file_location(
    "timebench_feature_performance", PERFORMANCE_PATH
)
assert PERFORMANCE_SPEC is not None and PERFORMANCE_SPEC.loader is not None
PERFORMANCE = importlib.util.module_from_spec(PERFORMANCE_SPEC)
PERFORMANCE_SPEC.loader.exec_module(PERFORMANCE)

def main() -> None:
    # Removing the internal NaN would incorrectly produce mean(|3-1|, |4-3|)=1.5.
    context = np.asarray([1.0, np.nan, 3.0, 4.0, np.nan])
    assert METRICS.seasonal_naive_scale(context, 1) == 1.0
    assert METRICS.seasonal_naive_scale(context, 1, squared=True) == 1.0

    frame = pd.DataFrame(
        {
            "dataset_id": ["a", "b", "c", "d"] * 2,
            "model": ["seasonal_naive"] * 4 + ["chronos2"] * 4,
            "feature": [1.0, 2.0, 3.0, 4.0] * 2,
            "scaled_MASE": [1.0] * 4 + [4.0, 3.0, 2.0, 1.0],
        }
    )
    with warnings.catch_warnings(record=True) as caught:
        correlations = PERFORMANCE.feature_correlations(frame, ["feature"])
    seasonal = correlations[correlations["model"] == "seasonal_naive"].iloc[0]
    learned = correlations[correlations["model"] == "chronos2"].iloc[0]
    mean = correlations[correlations["model"] == "mean_absolute"].iloc[0]
    assert np.isnan(seasonal["spearman_rho"])
    assert np.isclose(learned["spearman_rho"], -1.0)
    assert np.isclose(mean["spearman_rho"], 1.0)
    assert not caught
    print("Finite-pair, scaled-MASE, and constant-correlation contracts passed.")


if __name__ == "__main__":
    main()
