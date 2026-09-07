"""Focused missing-data and Slurm checkpoint contract for Adaptime."""

from __future__ import annotations

import ast
import importlib.util
import sys
import tempfile
import types
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# The local checkpoint runtime intentionally omits Hugging Face datasets. The
# preparation module needs only its deferred Dataset annotation in this test.
sys.modules.setdefault("datasets", types.ModuleType("datasets"))

from timebench.adaptime.retrieval import blockwise_topk
from timebench.adaptime.ridge import (
    FullRidgeStatistics,
    full_ridge_predict_with_fallback,
)
from timebench.evaluation.adaptation_data import (
    ADAPTATION_STRIDES,
    InsufficientAdaptationHistory,
    PreparationConfig,
    adaptation_split_lengths,
    adaptation_stride_for_frequency,
    prepare_adaptation_dataset,
)


def _raises(error_type, function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except error_type:
        return
    raise AssertionError(f"Expected {error_type.__name__}")


def _run_slurm_contract() -> None:
    path = PROJECT_ROOT / "src/tests/test_slurm_workflow.py"
    spec = importlib.util.spec_from_file_location("adaptime_slurm_contract", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.main()


def main() -> None:
    assert {
        frequency: adaptation_stride_for_frequency(frequency)
        for frequency in ("5T", "H", "B", "D", "W", "M", "Q")
    } == {
        "5T": ADAPTATION_STRIDES["intraday"],
        "H": ADAPTATION_STRIDES["hourly"],
        "B": ADAPTATION_STRIDES["business_daily"],
        "D": ADAPTATION_STRIDES["daily"],
        "W": ADAPTATION_STRIDES["weekly"],
        "M": ADAPTATION_STRIDES["monthly"],
        "Q": ADAPTATION_STRIDES["quarterly"],
    }
    train_length, validation_length, test_windows = adaptation_split_lengths(8, 4, 27)
    assert (train_length, validation_length, test_windows) == (85, 31, 2)

    class SyntheticDataset:
        _fingerprint = "adaptime-split-contract"

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> dict[str, object]:
            assert index == 0
            return {
                "target": np.arange(500, dtype=np.float32).reshape(2, 250),
                "start": "2000-01-01",
                "freq": "D",
            }

    preparation = PreparationConfig(
        dataset="synthetic",
        term="short",
        context_length=10,
        prediction_length=4,
        test_length=8,
        adaptation_train_length=train_length,
        adaptation_validation_length=validation_length,
        seasonality=7,
        adaptation_stride=27,
        retrieval_period=7,
        datastore_stride=7,
        max_datastore_windows=14,
    )
    with tempfile.TemporaryDirectory() as temporary:
        manifest_path = prepare_adaptation_dataset(
            SyntheticDataset(), preparation, temporary, source_path=temporary
        )
        manifest = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["counts"] == {
            "datastore": 14,
            "adaptation_train": 8,
            "adaptation_validation": 4,
            "test": 4,
        }
        assert manifest["datastore_dates_per_variate"] == {
            "minimum": 7,
            "maximum": 7,
            "balanced_cap": 7,
        }

    clipped = PreparationConfig(
        **{
            **preparation.__dict__,
            "context_length": 140,
            "max_datastore_windows": None,
        }
    )
    with tempfile.TemporaryDirectory() as temporary:
        _raises(
            InsufficientAdaptationHistory,
            prepare_adaptation_dataset,
            SyntheticDataset(),
            clipped,
            temporary,
            source_path=temporary,
        )

    vanilla = np.array([[[10.0, 11.0]], [[20.0, 21.0]]])
    design = np.ones((2, 1, 2, 2), dtype=np.float64)
    coefficients = np.array([2.0, -0.5])
    prediction = full_ridge_predict_with_fallback(
        vanilla, design, coefficients, np.array([True, False])
    )
    assert np.array_equal(prediction[1], vanilla[1])
    assert np.allclose(prediction[0], vanilla[0] + 1.5)

    statistics = FullRidgeStatistics(features=2)
    invalid_design = design[:1].copy()
    invalid_design[0, 0, 0, 0] = np.nan
    _raises(
        ValueError,
        statistics.update,
        invalid_design,
        np.ones((1, 1, 2)),
    )
    msse_statistics = FullRidgeStatistics(features=1)
    msse_statistics.update(
        np.ones((1, 1, 2, 1)),
        np.array([[[2.0, 4.0]]]),
        scale=np.array([[2.0]]),
    )
    assert np.isclose(msse_statistics.y_sum_squares, 5.0)

    query = np.array([[0.0, 1.0], [np.nan, 1.0]], dtype=np.float32)
    datastore = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    references = np.array([[0, 0, 10], [0, 0, 11]], dtype=np.int64)
    datastore_references = np.array([[0, 0, 0], [0, 0, 1]], dtype=np.int64)
    distances, neighbors = blockwise_topk(
        query,
        datastore,
        references,
        datastore_references,
        k=2,
        stride=1,
        horizon=1,
        minimum_overlap_fraction=0.75,
        require_complete_k=False,
    )
    assert np.isfinite(distances[0]).all() and np.all(neighbors[0] >= 0)
    assert np.isinf(distances[1]).all() and np.all(neighbors[1] == -1)

    extraction = (
        PROJECT_ROOT / "src/timebench/pipeline/adaptime_extraction.py"
    ).read_text(encoding="utf-8")
    training = (
        PROJECT_ROOT / "src/timebench/pipeline/adaptime_training.py"
    ).read_text(encoding="utf-8")
    testing = (
        PROJECT_ROOT / "src/timebench/pipeline/adaptime_testing.py"
    ).read_text(encoding="utf-8")
    model_loading = (
        PROJECT_ROOT / "src/timebench/model_loading/adaptime.py"
    ).read_text(encoding="utf-8")
    window_audit = (
        PROJECT_ROOT / "src/timebench/evaluation/window_audit.py"
    ).read_text(encoding="utf-8")
    tsrag_workflow = (
        PROJECT_ROOT / "src/timebench/pipeline/tsrag_workflow.py"
    ).read_text(encoding="utf-8")
    additional_sources = [
        (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        for relative in (
            "src/timebench/evaluation/adaptation_data.py",
            "src/timebench/pipeline/adaptime_workflow.py",
            "src/timebench/pipeline/tsrag.py",
            "src/timebench/pipeline/tsrag_data.py",
        )
    ]
    for source in (
        extraction,
        training,
        testing,
        model_loading,
        tsrag_workflow,
        *additional_sources,
    ):
        ast.parse(source)
    assert "minimum_query_finite_fraction" in extraction
    assert 'arrays[f"{split}.rag_eligible"]' in extraction
    assert "include_vanilla_fallback=True" in training
    assert 'arrays.open(f"{split}.msse_scale")' in training
    assert '"criterion": "adaptation_validation_msse"' in training
    assert "full_ridge_predict_with_fallback" in testing
    assert '"rag_coverage"' in testing
    assert '"scaled_mase"' in testing
    assert "DEFAULT_CONTEXT_PROFILES" in model_loading
    assert '"chronos2": 8192' in window_audit
    assert '"ts_icl": 4096' in window_audit
    assert '"chronos_bolt": 2048' in window_audit
    assert "ridge_prepared.context_length != TSRAG_CONTEXT_LENGTH" not in tsrag_workflow
    assert '"metrics": ["scaled_mase"]' in tsrag_workflow
    adaptation_data = additional_sources[0]
    adaptime_workflow = additional_sources[1]
    tsrag_data = additional_sources[3]
    assert '"insufficient_history_policy": "vanilla"' in adaptime_workflow
    assert "adaptation_split_lengths(" in adaptime_workflow
    assert "max_datastore_windows" in adaptation_data
    assert "TSRAG_DATASTORE_STRIDE = 1" in tsrag_data
    assert "origins[-int(retained_dates) :]" in tsrag_data

    _run_slurm_contract()
    print("Adaptime split, fallback, datastore, and Slurm contracts passed.")


if __name__ == "__main__":
    main()
