from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from bayes_sysid.models import BayesianARX

_VALID_STABILITY_DOMAINS = ("discrete", "continuous")


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


def arx_transfer_to_continuous_poles(
    theta: ArrayLike,
    na: int,
    sample_time: float | None = None,
) -> np.ndarray:
    """
    Convert ARX transfer poles to a continuous-time approximation.

    Notes
    -----
    This conversion is not implemented yet. It is intentionally separated into a
    dedicated helper so continuous-domain stability requests fail with a clear
    message instead of silently applying a discrete-time criterion.
    """
    _ = np.asarray(theta, dtype=float)
    if na <= 0:
        raise ValueError("na must be positive.")
    raise NotImplementedError(
        "Continuous-domain pole conversion for ARX models is not implemented yet. "
        "Use domain='discrete' or implement arx_transfer_to_continuous_poles(...). "
        f"Received sample_time={sample_time!r}."
    )


def arx_poles_by_domain(
    theta: ArrayLike,
    na: int,
    domain: str = "discrete",
    sample_time: float | None = None,
) -> np.ndarray:
    """Return ARX poles in the requested domain ('discrete' or 'continuous')."""
    if domain == "discrete":
        return arx_poles(theta, na)
    if domain == "continuous":
        return arx_transfer_to_continuous_poles(theta, na=na, sample_time=sample_time)
    raise ValueError(f"domain must be one of {_VALID_STABILITY_DOMAINS}, got {domain!r}.")


def is_stable(poles: ArrayLike, tol: float = 1e-9, domain: str = "discrete") -> bool:
    """
    Return stability flag under the requested domain convention.

    - domain='discrete': stable iff |z| < 1 - tol.
    - domain='continuous': stable iff Re(s) < -tol.
    """
    poles = np.asarray(poles, dtype=complex).reshape(-1)
    if domain == "discrete":
        return bool(np.all(np.abs(poles) < 1.0 - tol))
    if domain == "continuous":
        return bool(np.all(poles.real < -tol))
    raise ValueError(f"domain must be one of {_VALID_STABILITY_DOMAINS}, got {domain!r}.")


def is_stable_discrete(poles: ArrayLike, tol: float = 1e-9) -> bool:
    """Backward-compatible discrete-time stability helper."""
    return is_stable(poles, tol=tol, domain="discrete")


def posterior_stability_probability(
    model: BayesianARX,
    n_samples: int = 1000,
    random_state: int | None = None,
    tol: float = 1e-9,
    domain: str = "discrete",
    sample_time: float | None = None,
) -> float:
    """
    Estimate P(stable | D) by posterior sampling of ARX coefficients.

    Parameters
    ----------
    domain:
        'discrete' (default) uses |z| < 1 - tol.
        'continuous' uses Re(s) < -tol after converting ARX poles to a
        continuous-time approximation.
    """
    theta_samples = model.sample_parameters(n_samples=n_samples, random_state=random_state)
    stable = []
    for theta in theta_samples:
        poles = arx_poles_by_domain(theta, model.na, domain=domain, sample_time=sample_time)
        stable.append(is_stable(poles, tol=tol, domain=domain))
    return float(np.mean(stable))
