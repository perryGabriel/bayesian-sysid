from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import solve_discrete_are

from .lqr import closed_loop_poles, lqr_gain_from_realization, sampled_bayesian_lqr_summary
from .realization import arx_to_state_space


Array = np.ndarray


def _validate_kalman_inputs(A: ArrayLike, C: ArrayLike, Qn: ArrayLike, Rn: ArrayLike) -> tuple[Array, Array, Array, Array]:
    A = np.asarray(A, dtype=float)
    C = np.asarray(C, dtype=float)
    Qn = np.asarray(Qn, dtype=float)
    Rn = np.asarray(Rn, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square with shape (n, n).")
    n = A.shape[0]

    if C.ndim == 1:
        C = C.reshape(1, -1)
    if C.ndim != 2 or C.shape[1] != n:
        raise ValueError("C must have shape (p, n).")
    p = C.shape[0]

    if Qn.shape != (n, n):
        raise ValueError("Qn must have shape (n, n).")
    if Rn.shape != (p, p):
        raise ValueError("Rn must have shape (p, p).")

    return A, C, Qn, Rn


def steady_state_kalman_gain(A: ArrayLike, C: ArrayLike, Qn: ArrayLike, Rn: ArrayLike) -> Array:
    """Compute steady-state Kalman gain via DARE on the dual system."""
    A, C, Qn, Rn = _validate_kalman_inputs(A, C, Qn, Rn)
    P = solve_discrete_are(A.T, C.T, Qn, Rn)
    S = C @ P @ C.T + Rn
    L = np.linalg.solve(S, C @ P).T
    return np.asarray(L, dtype=float)


def lqg_controller(
    A: ArrayLike,
    B: ArrayLike,
    C: ArrayLike,
    Q: ArrayLike,
    R: ArrayLike,
    Qn: ArrayLike,
    Rn: ArrayLike,
) -> tuple[Array, Array]:
    """Return separated-design gains ``(K, L)`` for discrete LQG."""
    K = lqr_gain_from_realization(A, B, Q, R)
    L = steady_state_kalman_gain(A, C, Qn, Rn)
    return K, L


def sampled_bayesian_lqg_summary(
    model,
    Q: ArrayLike,
    R: ArrayLike,
    Qn: ArrayLike,
    Rn: ArrayLike,
    n_samples: int = 200,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    random_state: int | None = None,
) -> dict[str, object]:
    """Posterior Monte Carlo summary for LQG synthesis from ARX samples."""
    theta_samples = model.sample_parameters(n_samples=n_samples, random_state=random_state)
    qs = np.asarray(quantiles, dtype=float)

    stable = np.zeros(n_samples, dtype=bool)
    control_pole_radius = np.full(n_samples, np.nan, dtype=float)
    estimator_pole_radius = np.full(n_samples, np.nan, dtype=float)
    control_gain_norms = np.full(n_samples, np.nan, dtype=float)
    estimator_gain_norms = np.full(n_samples, np.nan, dtype=float)

    for idx, theta in enumerate(theta_samples):
        a = np.asarray(theta[: model.na], dtype=float)
        b = np.asarray(theta[model.na : model.na + model.nb], dtype=float)
        A, B, C, _ = arx_to_state_space(a=a, b=b)

        try:
            K, L = lqg_controller(A, B, C, Q, R, Qn, Rn)
        except np.linalg.LinAlgError:
            continue

        control_poles = closed_loop_poles(A, B, K)
        estimator_poles = np.linalg.eigvals(A - L @ C)

        stable[idx] = bool(np.all(np.abs(control_poles) < 1.0) and np.all(np.abs(estimator_poles) < 1.0))
        control_pole_radius[idx] = float(np.max(np.abs(control_poles)))
        estimator_pole_radius[idx] = float(np.max(np.abs(estimator_poles)))
        control_gain_norms[idx] = float(np.linalg.norm(K, ord=2))
        estimator_gain_norms[idx] = float(np.linalg.norm(L, ord=2))

    return {
        "n_samples": int(n_samples),
        "stability_probability": float(np.mean(stable)),
        "quantiles": tuple(float(q) for q in qs),
        "control_pole_radius_quantiles": np.nanquantile(control_pole_radius, qs).tolist(),
        "estimator_pole_radius_quantiles": np.nanquantile(estimator_pole_radius, qs).tolist(),
        "control_gain_norm_quantiles": np.nanquantile(control_gain_norms, qs).tolist(),
        "estimator_gain_norm_quantiles": np.nanquantile(estimator_gain_norms, qs).tolist(),
    }
