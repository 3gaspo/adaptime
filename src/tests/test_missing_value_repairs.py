"""Focused regression checks for missing-value and feature repair behavior."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from timebench.evaluation.covariates import extract_covariate_window
from timebench.feature.features import mean_seasonal_cycle_correlation
from timebench.feature.features_runner import missing_feature_ids


def main() -> None:
    covariates = np.arange(16, dtype=np.float32).reshape(2, 8)
    covariates[0, 2] = np.inf
    covariates[1, 6] = np.nan
    window = extract_covariate_window(
        {
            "target": np.arange(5, dtype=np.float32),
            "feat_dynamic_real": covariates,
        },
        {},
        context_length=5,
        prediction_length=3,
    )
    assert np.isnan(window.past[0, 2])
    assert np.isnan(window.future[1, 1])

    cycles = np.asarray(
        [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 2.0, 1.0]]
    )
    expected = np.mean(
        [
            np.corrcoef(cycles[i], cycles[j])[0, 1]
            for i in range(len(cycles))
            for j in range(i + 1, len(cycles))
        ]
    )
    actual = mean_seasonal_cycle_correlation(cycles.reshape(-1), period=3)
    assert np.isclose(actual, expected)

    with_constant_cycle = np.concatenate([cycles, np.ones((1, 3))])
    actual = mean_seasonal_cycle_correlation(
        with_constant_cycle.reshape(-1), period=3
    )
    assert np.isclose(actual, expected)
    assert np.isnan(mean_seasonal_cycle_correlation(np.ones(12), period=3))

    existing = pd.DataFrame({"unique_id": ["a", "b"]})
    assert missing_feature_ids({"a", "b"}, existing) == set()
    assert missing_feature_ids({"a", "b", "c"}, existing) == {"c"}
    assert missing_feature_ids({"a"}, existing) is None
    assert missing_feature_ids(
        {"a", "b"}, pd.DataFrame({"unique_id": ["a", "a"]})
    ) is None

    print("Missing-value and feature repair checks passed")


if __name__ == "__main__":
    main()
