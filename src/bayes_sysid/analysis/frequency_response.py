from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from bayes_sysid.models import BayesianARX


def arx_frequency_response(theta: ArrayLike, na: int, nb: int, w: ArrayLike) -> np.ndarray:
    """Compute H(exp(jw)) for ARX model B(z^-1)/A(z^-1)."""
    theta = np.asarray(theta, dtype=float).reshape(-1)
    if len(theta) < na + nb:
        raise ValueError("theta length must be at least na + nb.")

    a = theta[:na]
    b = theta[na : na + nb]
    w = np.asarray(w, dtype=float).reshape(-1)

    z_inv = np.exp(-1j * w)
    num = np.zeros_like(z_inv, dtype=complex)
    den = np.ones_like(z_inv, dtype=complex)

    for j in range(1, nb + 1):
        num += b[j - 1] * z_inv**j
    for i in range(1, na + 1):
        den += a[i - 1] * z_inv**i

    return num / den


def posterior_frequency_response_samples(
    model: BayesianARX,
    w: ArrayLike,
    n_samples: int = 200,
    random_state: int | None = None,
) -> np.ndarray:
    """Return array shape (n_samples, n_freq) of posterior transfer samples."""
    theta_samples = model.sample_parameters(n_samples=n_samples, random_state=random_state)
    H = [arx_frequency_response(theta, model.na, model.nb, w) for theta in theta_samples]
    return np.asarray(H)


def posterior_magnitude_envelope(
    H_samples: ArrayLike,
    q_low: float = 0.1,
    q_high: float = 0.9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    H_samples = np.asarray(H_samples)
    if H_samples.ndim != 2:
        raise ValueError("H_samples must have shape (n_samples, n_freq).")
    mag = np.abs(H_samples)
    return (
        np.quantile(mag, q_low, axis=0),
        np.quantile(mag, 0.5, axis=0),
        np.quantile(mag, q_high, axis=0),
    )
