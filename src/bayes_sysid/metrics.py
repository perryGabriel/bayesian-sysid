from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _as_1d_float(x: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional after reshaping.")
    return arr


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    y_true = _as_1d_float(y_true, "y_true")
    y_pred = _as_1d_float(y_pred, "y_pred")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    y_true = _as_1d_float(y_true, "y_true")
    y_pred = _as_1d_float(y_pred, "y_pred")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    return float(np.mean(np.abs(y_true - y_pred)))


def gaussian_nll(y_true: ArrayLike, mean: ArrayLike, var: ArrayLike) -> float:
    y_true = _as_1d_float(y_true, "y_true")
    mean = _as_1d_float(mean, "mean")
    var = _as_1d_float(var, "var")
    if len(y_true) != len(mean) or len(y_true) != len(var):
        raise ValueError("y_true, mean, and var must have the same length.")
    if np.any(var <= 0):
        raise ValueError("All variances must be positive.")

    return float(np.mean(0.5 * np.log(2.0 * np.pi * var) + 0.5 * ((y_true - mean) ** 2) / var))


def interval_coverage(y_true: ArrayLike, mean: ArrayLike, std: ArrayLike, z: float = 1.96) -> float:
    y_true = _as_1d_float(y_true, "y_true")
    mean = _as_1d_float(mean, "mean")
    std = _as_1d_float(std, "std")
    if len(y_true) != len(mean) or len(y_true) != len(std):
        raise ValueError("y_true, mean, and std must have the same length.")
    if np.any(std < 0):
        raise ValueError("std entries must be non-negative.")

    lo = mean - z * std
    hi = mean + z * std
    return float(np.mean((y_true >= lo) & (y_true <= hi)))
