"""Focused contract check for foundation-model known covariates."""

import importlib.util
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_covariates():
    path = PROJECT_ROOT / "src/timebench/evaluation/covariates.py"
    spec = importlib.util.spec_from_file_location("timebench_covariates", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _raises(error_type, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except error_type:
        return
    raise AssertionError(f"Expected {error_type.__name__}")


def main() -> None:
    covariates = _load_covariates()
    values = np.arange(16, dtype=np.float32).reshape(2, 8)
    input_entry = {
        "target": np.arange(5, dtype=np.float32),
        "feat_dynamic_real": values,
    }
    window = covariates.extract_covariate_window(
        input_entry,
        {},
        context_length=4,
        prediction_length=3,
    )
    assert window.full.shape == (2, 7)
    assert np.array_equal(window.past, values[:, 1:5])
    assert np.array_equal(window.future, values[:, 5:])

    split_window = covariates.extract_covariate_window(
        {
            "target": np.arange(5, dtype=np.float32),
            "past_feat_dynamic_real": values[:, :5],
        },
        {"past_feat_dynamic_real": values[:, 5:]},
        context_length=10,
        prediction_length=3,
    )
    assert np.array_equal(split_window.full, values)
    assert covariates.validate_covariate_channels([window, split_window]) == 2

    past_window = covariates.extract_covariate_window(
        {
            "target": np.arange(5, dtype=np.float32),
            "past_feat_dynamic_real": values[:, :5],
        },
        {},
        context_length=4,
        prediction_length=3,
        require_future=False,
    )
    assert past_window.full.shape == (2, 4)
    assert past_window.future.shape == (2, 0)

    nonfinite_values = values.copy()
    nonfinite_values[0, 2] = np.inf
    nonfinite_values[1, 6] = np.nan
    masked_window = covariates.extract_covariate_window(
        {
            "target": np.arange(5, dtype=np.float32),
            "feat_dynamic_real": nonfinite_values,
        },
        {},
        context_length=5,
        prediction_length=3,
    )
    assert np.isnan(masked_window.past[0, 2])
    assert np.isnan(masked_window.future[1, 1])

    assert (
        covariates.validate_covariate_mode(
            "chronos2", "future_included", supports_covariates=True
        )
        == "future_included"
    )
    assert (
        covariates.validate_covariate_mode(
            "chronos2",
            "past_targets",
            supports_covariates=True,
            supported_modes=covariates.COVARIATE_MODES,
        )
        == "past_targets"
    )
    _raises(
        ValueError,
        covariates.validate_covariate_mode,
        "ts_icl",
        "past_targets",
        supports_covariates=True,
    )
    _raises(
        ValueError,
        covariates.validate_covariate_mode,
        "chronos_bolt",
        "future_included",
        supports_covariates=False,
    )
    _raises(
        ValueError,
        covariates.extract_covariate_window,
        {"target": np.arange(5, dtype=np.float32)},
        {},
        context_length=5,
        prediction_length=3,
    )

    chronos2 = (PROJECT_ROOT / "experiments/chronos2.py").read_text(encoding="utf-8")
    ts_icl = (PROJECT_ROOT / "experiments/ts_icl.py").read_text(encoding="utf-8")
    bolt = (PROJECT_ROOT / "experiments/chronos_bolt.py").read_text(encoding="utf-8")
    seasonal = (PROJECT_ROOT / "experiments/seasonal_naive.py").read_text(
        encoding="utf-8"
    )
    assert 'SUPPORTS_COVARIATES = True' in chronos2
    assert '"past_covariates"' in chronos2 and '"future_covariates"' in chronos2
    assert 'SUPPORTS_COVARIATES = True' in ts_icl and 'covars=grouped_covariates' in ts_icl
    assert "supports_multivariate=False" in ts_icl
    assert 'SUPPORTS_COVARIATES = False' in bolt
    assert 'SUPPORTS_COVARIATES = False' in seasonal

    print("Covariate contract check passed")


if __name__ == "__main__":
    main()
