"""Exact shared full-ridge formulation with streaming sufficient statistics."""

from __future__ import annotations

import warnings

import numpy as np


def full_ridge_feature_names(k: int) -> list[str]:
    if int(k) <= 0:
        raise ValueError("k must be positive")
    return [
        "V",
        "C",
        *(f"Y_{index + 1}" for index in range(int(k))),
        *(f"N_{index + 1}" for index in range(int(k))),
    ]


def full_ridge_design(
    vanilla: np.ndarray,
    context_forecast: np.ndarray,
    neighbor_target: np.ndarray,
    neighbor_forecast: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``[V, C, Y_1..Y_K, N_1..N_K]`` and residual ``Y - V``.

    Query arrays have shape ``(batch, channels, horizon)`` and neighbor arrays
    have shape ``(batch, K, channels, horizon)``. The returned feature axis is
    last, allowing the same coefficient vector to be shared over channels and
    horizons without materializing a flattened design matrix.
    """

    vanilla = np.asarray(vanilla)
    context_forecast = np.asarray(context_forecast)
    neighbor_target = np.asarray(neighbor_target)
    neighbor_forecast = np.asarray(neighbor_forecast)
    target = np.asarray(target)
    if vanilla.ndim != 3 or target.shape != vanilla.shape:
        raise ValueError("vanilla and target must share (batch, channels, horizon)")
    if context_forecast.shape != vanilla.shape:
        raise ValueError("context forecast must match the vanilla forecast")
    expected_neighbor_tail = (vanilla.shape[0], vanilla.shape[1], vanilla.shape[2])
    if neighbor_target.ndim != 4 or (
        neighbor_target.shape[0],
        neighbor_target.shape[2],
        neighbor_target.shape[3],
    ) != expected_neighbor_tail:
        raise ValueError("neighbor targets must have shape (batch, K, channels, horizon)")
    if neighbor_forecast.shape != neighbor_target.shape:
        raise ValueError("neighbor forecasts must match neighbor targets")
    design = np.concatenate(
        (
            vanilla[..., None],
            context_forecast[..., None],
            np.moveaxis(neighbor_target, 1, -1),
            np.moveaxis(neighbor_forecast, 1, -1),
        ),
        axis=-1,
    )
    return design, target - vanilla


class FullRidgeStatistics:
    """Float64 exact sufficient statistics for shared no-intercept ridge."""

    def __init__(self, features: int) -> None:
        if int(features) <= 0:
            raise ValueError("features must be positive")
        self.features = int(features)
        self.windows = 0
        self.observations = 0
        self.feature_sum_squares = np.zeros(self.features, dtype=np.float64)
        self.xtx = np.zeros((self.features, self.features), dtype=np.float64)
        self.xty = np.zeros(self.features, dtype=np.float64)
        self.y_sum_squares = 0.0

    def update(
        self,
        design: np.ndarray,
        residual: np.ndarray,
        *,
        scale: np.ndarray | None = None,
    ) -> None:
        design = np.asarray(design, dtype=np.float64)
        residual = np.asarray(residual, dtype=np.float64)
        if design.ndim < 3 or design.shape[-1] != self.features:
            raise ValueError("design must end in the configured feature dimension")
        if residual.shape != design.shape[:-1]:
            raise ValueError("residual must match every design axis except features")
        if scale is None:
            scale_values = np.ones(residual.shape[:-1], dtype=np.float64)
        else:
            scale_values = np.asarray(scale, dtype=np.float64)
            if scale_values.shape != residual.shape[:-1]:
                raise ValueError("scale must have one value per query/channel window")
            scale_values = np.maximum(scale_values, 1e-8)
        normalized_design = design / scale_values[..., None, None]
        normalized_residual = residual / scale_values[..., None]
        x = normalized_design.reshape(-1, self.features)
        y = normalized_residual.reshape(-1)
        self.feature_sum_squares += np.einsum("ij,ij->j", x, x)
        self.xtx += x.T @ x
        self.xty += x.T @ y
        self.y_sum_squares += float(y @ y)
        self.windows += int(np.prod(residual.shape[:-1]))
        self.observations += int(y.size)

    def merge(self, other: "FullRidgeStatistics") -> None:
        if other.features != self.features:
            raise ValueError("cannot merge ridge statistics with different feature counts")
        self.feature_sum_squares += other.feature_sum_squares
        self.xtx += other.xtx
        self.xty += other.xty
        self.y_sum_squares += other.y_sum_squares
        self.windows += other.windows
        self.observations += other.observations

    def solve(self, alpha: float) -> np.ndarray:
        if self.observations == 0:
            raise ValueError("cannot solve empty ridge statistics")
        if float(alpha) < 0:
            raise ValueError("ridge alpha must be non-negative")
        feature_rms = np.maximum(
            np.sqrt(self.feature_sum_squares / self.observations),
            1e-12,
        )
        matrix = self.xtx / np.outer(feature_rms, feature_rms) / self.observations
        target = self.xty / feature_rms / self.observations
        regularized = matrix + float(alpha) * np.eye(self.features, dtype=np.float64)
        try:
            standardized = np.linalg.solve(regularized, target)
        except np.linalg.LinAlgError:
            standardized = np.linalg.lstsq(regularized, target, rcond=None)[0]
        return standardized / feature_rms

    def mean_squared_error(self, coefficients: np.ndarray) -> float:
        if self.observations == 0:
            raise ValueError("cannot score empty ridge statistics")
        coefficients = np.asarray(coefficients, dtype=np.float64)
        if coefficients.shape != (self.features,):
            raise ValueError("coefficient shape does not match ridge features")
        error = (
            self.y_sum_squares
            - 2.0 * float(coefficients @ self.xty)
            + float(coefficients @ self.xtx @ coefficients)
        )
        return max(error / self.observations, 0.0)


def full_ridge_predict(
    vanilla: np.ndarray,
    design: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    vanilla = np.asarray(vanilla)
    design = np.asarray(design)
    coefficients = np.asarray(coefficients)
    if design.shape[:-1] != vanilla.shape or coefficients.shape != (design.shape[-1],):
        raise ValueError("vanilla, design, and coefficients are not aligned")
    return vanilla + np.einsum("...f,f->...", design, coefficients)


def query_scale(context: np.ndarray) -> np.ndarray:
    """Return the online-adaptation per-window/channel standard-deviation scale."""

    context = np.asarray(context, dtype=np.float64)
    if context.ndim != 3:
        raise ValueError("context must have shape (batch, channels, lookback)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.maximum(np.nanstd(context, axis=-1), 1e-8)
