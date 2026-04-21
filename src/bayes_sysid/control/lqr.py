from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import solve_discrete_are

from .realization import arx_to_state_space


Array = np.ndarray


def _validate_lqr_inputs(A: ArrayLike, B: ArrayLike, Q: ArrayLike, R: ArrayLike) -> tuple[Array, Array, Array, Array]:
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square with shape (n, n).")
    n = A.shape[0]

    if B.ndim == 1:
        B = B.reshape(-1, 1)
    if B.ndim != 2 or B.shape[0] != n:
        raise ValueError("B must have shape (n, m).")

    m = B.shape[1]
    if Q.shape != (n, n):
        raise ValueError("Q must have shape (n, n).")
    if R.shape != (m, m):
        raise ValueError("R must have shape (m, m).")

    return A, B, Q, R


def solve_discrete_lqr(A: ArrayLike, B: ArrayLike, Q: ArrayLike, R: ArrayLike) -> tuple[Array, Array]:
    """Solve the infinite-horizon discrete LQR problem.

    Returns ``(P, K)`` where ``P`` solves the DARE and
    ``u[k] = -K x[k]`` is the optimal static state feedback.
    """
    A, B, Q, R = _validate_lqr_inputs(A, B, Q, R)
    P = solve_discrete_are(A, B, Q, R)
    G = R + B.T @ P @ B
    K = np.linalg.solve(G, B.T @ P @ A)
    return np.asarray(P, dtype=float), np.asarray(K, dtype=float)


def lqr_gain_from_realization(A: ArrayLike, B: ArrayLike, Q: ArrayLike, R: ArrayLike) -> Array:
    """Compute LQR gain ``K`` for realization ``(A, B)``."""
    _, K = solve_discrete_lqr(A, B, Q, R)
    return K


def closed_loop_poles(A: ArrayLike, B: ArrayLike, K: ArrayLike) -> Array:
    """Return eigenvalues of ``A - B K``."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    K = np.asarray(K, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square with shape (n, n).")
    n = A.shape[0]

    if B.ndim == 1:
        B = B.reshape(-1, 1)
    if B.ndim != 2 or B.shape[0] != n:
        raise ValueError("B must have shape (n, m).")

    if K.ndim == 1:
        K = K.reshape(1, -1)
    if K.ndim != 2 or K.shape[1] != n or K.shape[0] != B.shape[1]:
        raise ValueError("K must have shape (m, n).")

    return np.linalg.eigvals(A - B @ K)


def sampled_bayesian_lqr_summary(
    model,
    Q: ArrayLike,
    R: ArrayLike,
    n_samples: int = 200,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    random_state: int | None = None,
) -> dict[str, object]:
    """Posterior Monte Carlo summary for LQR synthesis from ARX samples."""
    theta_samples = model.sample_parameters(n_samples=n_samples, random_state=random_state)
    qs = np.asarray(quantiles, dtype=float)

    stable = np.zeros(n_samples, dtype=bool)
    costs = np.full(n_samples, np.nan, dtype=float)
    pole_radius = np.full(n_samples, np.nan, dtype=float)
    gain_norms = np.full(n_samples, np.nan, dtype=float)

    for idx, theta in enumerate(theta_samples):
        a = np.asarray(theta[: model.na], dtype=float)
        b = np.asarray(theta[model.na : model.na + model.nb], dtype=float)
        A, B, _, _ = arx_to_state_space(a=a, b=b)
        try:
            P, K = solve_discrete_lqr(A, B, Q, R)
        except np.linalg.LinAlgError:
            continue

        poles = closed_loop_poles(A, B, K)
        stable[idx] = bool(np.all(np.abs(poles) < 1.0))
        costs[idx] = float(np.trace(P))
        pole_radius[idx] = float(np.max(np.abs(poles)))
        gain_norms[idx] = float(np.linalg.norm(K, ord=2))

    return {
        "n_samples": int(n_samples),
        "stability_probability": float(np.mean(stable)),
        "quantiles": tuple(float(q) for q in qs),
        "cost_quantiles": np.nanquantile(costs, qs).tolist(),
        "pole_radius_quantiles": np.nanquantile(pole_radius, qs).tolist(),
        "gain_norm_quantiles": np.nanquantile(gain_norms, qs).tolist(),
    }
