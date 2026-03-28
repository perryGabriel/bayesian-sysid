from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def isotropic_prior_covariance(p: int, variance: float = 10.0) -> np.ndarray:
    """Create an isotropic Gaussian prior covariance matrix variance * I."""
    p = int(p)
    if p <= 0:
        raise ValueError("p must be positive.")
    if variance <= 0:
        raise ValueError("variance must be positive.")
    return np.eye(p, dtype=float) * float(variance)


def diagonal_arx_prior_covariance(
    na: int,
    nb: int,
    ar_variance: float = 10.0,
    input_variance: float = 10.0,
) -> np.ndarray:
    """Create diagonal prior covariance with separate AR and input shrinkage."""
    na = int(na)
    nb = int(nb)
    if na < 0 or nb < 0 or (na == 0 and nb == 0):
        raise ValueError("At least one of na or nb must be positive.")
    if ar_variance <= 0 or input_variance <= 0:
        raise ValueError("ar_variance and input_variance must be positive.")

    diag = np.r_[np.full(na, float(ar_variance)), np.full(nb, float(input_variance))]
    return np.diag(diag)


def scale_prior_covariance_by_regressor_variance(
    Phi: ArrayLike,
    Sigma0: ArrayLike,
    min_variance: float = 1e-8,
) -> np.ndarray:
    """Scale diagonal prior covariance by inverse regressor variance."""
    Phi_arr = np.asarray(Phi, dtype=float)
    Sigma0_arr = np.asarray(Sigma0, dtype=float)

    if Phi_arr.ndim != 2:
        raise ValueError("Phi must be a 2D array.")
    p = Phi_arr.shape[1]
    if Sigma0_arr.shape != (p, p):
        raise ValueError("Sigma0 must have shape (p, p) matching Phi columns.")
    if min_variance <= 0:
        raise ValueError("min_variance must be positive.")
    if not np.allclose(Sigma0_arr, np.diag(np.diag(Sigma0_arr))):
        raise ValueError("Sigma0 must be diagonal for variance scaling.")

    feature_var = np.var(Phi_arr, axis=0)
    inv_scale = 1.0 / np.maximum(feature_var, float(min_variance))
    scaled_diag = np.diag(Sigma0_arr) * inv_scale
    return np.diag(scaled_diag)
