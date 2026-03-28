from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass
class ARXRegressionData:
    Phi: np.ndarray
    y_target: np.ndarray
    max_lag: int


def _as_1d_float(x: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional after reshaping.")
    return arr


def build_arx_regression(y: ArrayLike, u: ArrayLike, na: int, nb: int) -> ARXRegressionData:
    """Construct the regression matrix for a SISO ARX model."""
    if na < 0 or nb < 0 or (na == 0 and nb == 0):
        raise ValueError("At least one of na or nb must be positive.")

    y = _as_1d_float(y, "y")
    u = _as_1d_float(u, "u")
    if len(y) != len(u):
        raise ValueError("y and u must have the same length.")

    max_lag = max(na, nb)
    if len(y) <= max_lag:
        raise ValueError("Time series is too short for the requested lags.")

    rows = []
    targets = []
    for t_idx in range(max_lag, len(y)):
        phi = []
        for i in range(1, na + 1):
            phi.append(y[t_idx - i])
        for j in range(1, nb + 1):
            phi.append(u[t_idx - j])
        rows.append(phi)
        targets.append(y[t_idx])

    Phi = np.asarray(rows, dtype=float)
    y_target = np.asarray(targets, dtype=float)
    return ARXRegressionData(Phi=Phi, y_target=y_target, max_lag=max_lag)
