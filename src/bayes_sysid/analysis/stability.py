from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from bayes_sysid.models import BayesianARX


def arx_poles(theta: ArrayLike, na: int) -> np.ndarray:
    """Return poles for ARX denominator A(z^-1)=1+a1 z^-1+...+ana z^-na."""
    theta = np.asarray(theta, dtype=float).reshape(-1)
    if na <= 0:
        raise ValueError("na must be positive.")
    if len(theta) < na:
        raise ValueError("theta must contain at least na AR coefficients.")

    a = theta[:na]
    poly = np.r_[1.0, a]
    return np.roots(poly)


def is_stable_discrete(poles: ArrayLike, tol: float = 1e-9) -> bool:
    poles = np.asarray(poles, dtype=complex).reshape(-1)
    return bool(np.all(np.abs(poles) < 1.0 - tol))


def posterior_stability_probability(
    model: BayesianARX,
    n_samples: int = 1000,
    random_state: int | None = None,
) -> float:
    """Estimate P(stable | D) by posterior sampling of ARX coefficients."""
    theta_samples = model.sample_parameters(n_samples=n_samples, random_state=random_state)
    stable = []
    for theta in theta_samples:
        poles = arx_poles(theta, model.na)
        stable.append(is_stable_discrete(poles))
    return float(np.mean(stable))
