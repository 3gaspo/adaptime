"""Focused contracts for Adaptime's memory-bounded scientific core."""

import numpy as np

from timebench.adaptime.retrieval import blockwise_topk
from timebench.adaptime.ridge import (
    FullRidgeStatistics,
    full_ridge_design,
    full_ridge_predict,
)
from timebench.evaluation.adaptation_data import _official_test_origins
from timebench.pipeline.adaptime_training import RidgeTrainingConfig


def test_official_time_test_origins_remain_horizon_spaced() -> None:
    origins = _official_test_origins((80, 101), context_length=16, horizon=5)
    np.testing.assert_array_equal(origins, np.asarray([80, 85, 90, 95]))


def test_full_shared_ridge_grid_has_no_pretest_refit_switch() -> None:
    config = RidgeTrainingConfig()
    assert config.k_values == (1, 5, 10, 15)
    assert config.alpha_values == (1e-3, 1e-2, 1e-1)
    assert not hasattr(config, "refit_on_train_and_validation")


def test_blockwise_retrieval_applies_query_relative_calendar_stride() -> None:
    query = np.asarray([[0.1], [9.9]], dtype=np.float32)
    datastore = np.asarray([[0.0], [10.0], [2.0], [8.0]], dtype=np.float32)
    query_refs = np.asarray([[0, 0, 10], [1, 0, 11]], dtype=np.int64)
    datastore_refs = np.asarray(
        [[2, 0, 2], [2, 0, 3], [3, 0, 4], [3, 0, 5]], dtype=np.int64
    )
    _, indices = blockwise_topk(
        query,
        datastore,
        query_refs,
        datastore_refs,
        k=1,
        stride=2,
        horizon=1,
        query_block_size=1,
        datastore_block_size=2,
    )
    np.testing.assert_array_equal(indices[:, 0], np.asarray([0, 1]))


def test_fixed_datastore_aligns_period_before_applying_larger_stride() -> None:
    query = np.asarray([[0.0]], dtype=np.float32)
    datastore = np.asarray([[0.01], [0.05], [0.02], [0.2]], dtype=np.float32)
    query_refs = np.asarray([[0, 0, 0]], dtype=np.int64)
    datastore_refs = np.asarray(
        [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int64
    )
    _, indices = blockwise_topk(
        query,
        datastore,
        query_refs,
        datastore_refs,
        query_calendar_ticks=np.asarray([32]),
        datastore_calendar_ticks=np.asarray([10, 12, 16, 18]),
        retrieval_period=2,
        datastore_end_ticks_by_item=np.asarray([19]),
        k=1,
        stride=6,
        horizon=2,
        query_block_size=1,
        datastore_block_size=4,
    )
    # latest=min(32-2, 19)=19, then period-aligned to 18; stride-6
    # candidates are 18 and 12, so the visually closest tick-10 row is ineligible.
    np.testing.assert_array_equal(indices[:, 0], np.asarray([1]))


def test_streaming_ridge_statistics_match_one_batch() -> None:
    rng = np.random.default_rng(7)
    vanilla = rng.normal(size=(6, 2, 4))
    context = rng.normal(size=(6, 2, 4))
    neighbor_y = rng.normal(size=(6, 3, 2, 4))
    neighbor_n = rng.normal(size=(6, 3, 2, 4))
    target = rng.normal(size=(6, 2, 4))
    scale = rng.uniform(0.5, 2.0, size=(6, 2))
    design, residual = full_ridge_design(
        vanilla, context, neighbor_y, neighbor_n, target
    )

    complete = FullRidgeStatistics(design.shape[-1])
    complete.update(design, residual, scale=scale)
    streamed = FullRidgeStatistics(design.shape[-1])
    streamed.update(design[:2], residual[:2], scale=scale[:2])
    streamed.update(design[2:], residual[2:], scale=scale[2:])

    np.testing.assert_allclose(streamed.solve(0.01), complete.solve(0.01))
    np.testing.assert_allclose(
        streamed.mean_squared_error(streamed.solve(0.01)),
        complete.mean_squared_error(complete.solve(0.01)),
    )


def test_full_ridge_uses_one_coefficient_vector_over_every_horizon() -> None:
    vanilla = np.asarray([[[1.0, 2.0]]])
    context = np.asarray([[[3.0, 4.0]]])
    neighbor_y = np.asarray([[[[5.0, 6.0]]]])
    neighbor_n = np.asarray([[[[0.5, 1.0]]]])
    target = np.asarray([[[7.0, 8.0]]])
    design, _ = full_ridge_design(
        vanilla, context, neighbor_y, neighbor_n, target
    )
    coefficients = np.asarray([0.1, 0.2, 0.3, -0.4])
    expected = vanilla + np.einsum("...f,f->...", design, coefficients)
    np.testing.assert_allclose(
        full_ridge_predict(vanilla, design, coefficients), expected
    )
