from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def simulate_arx(
    a: ArrayLike,
    b: ArrayLike,
    u: ArrayLike,
    sigma: float = 0.0,
    y_init: ArrayLike | None = None,
    random_state: int | None = None,
) -> np.ndarray:
    """Simulate a SISO ARX process.

    Model:
        y_t = sum_i a_i y_{t-i} + sum_j b_j u_{t-j} + e_t

    Parameters
    ----------
    a, b : array-like
        ARX coefficients.
    u : array-like
        Input sequence.
    sigma : float
        Standard deviation of Gaussian noise.
    y_init : array-like or None
        Initial output history. If None, zeros are used.
    """
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    u = np.asarray(u, dtype=float).reshape(-1)

    na = len(a)
    nb = len(b)
    max_lag = max(na, nb)
    rng = np.random.default_rng(random_state)

    if y_init is None:
        y_hist = [0.0] * max_lag
    else:
        y_init = np.asarray(y_init, dtype=float).reshape(-1)
        if len(y_init) < max_lag:
            raise ValueError("y_init must have length at least max(na, nb).")
        y_hist = list(y_init[-max_lag:].copy())

    y_out = []
    for t in range(max_lag, len(u)):
        y_t = 0.0
        for i in range(1, na + 1):
            y_t += a[i - 1] * y_hist[-i]
        for j in range(1, nb + 1):
            y_t += b[j - 1] * u[t - j]
        if sigma > 0:
            y_t += rng.normal(0.0, sigma)
        y_hist.append(y_t)
        y_out.append(y_t)

    return np.asarray(y_out, dtype=float)
