from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import place_poles


Array = np.ndarray


def observability_matrix(A: ArrayLike, C: ArrayLike, horizon: int | None = None) -> Array:
    """Build the finite-horizon observability matrix for a discrete-time model.

    For system ``x[k+1] = A x[k] + ...`` and ``y[k] = C x[k] + ...``, this returns
    ``[C; C A; ...; C A^(horizon-1)]``.
    """
    A = np.asarray(A, dtype=float)
    C = np.asarray(C, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square with shape (n, n).")
    if C.ndim == 1:
        C = C.reshape(1, -1)
    if C.ndim != 2 or C.shape[1] != A.shape[0]:
        raise ValueError("C must have shape (p, n) compatible with A.")

    n = A.shape[0]
    h = n if horizon is None else int(horizon)
    if h < 1:
        raise ValueError("horizon must be >= 1.")

    blocks = [C]
    CAk = C
    for _ in range(1, h):
        CAk = CAk @ A
        blocks.append(CAk)
    return np.vstack(blocks)


def is_observable(A: ArrayLike, C: ArrayLike, tol: float = 1e-9) -> bool:
    """Return ``True`` when the pair ``(A, C)`` is observable."""
    O = observability_matrix(A, C)
    s = np.linalg.svd(O, compute_uv=False)
    rank = int(np.sum(s > tol))
    return rank == np.asarray(A).shape[0]


def design_luenberger_gain(A: ArrayLike, C: ArrayLike, desired_poles: ArrayLike) -> Array:
    """Design a full-order Luenberger observer gain via pole placement.

    Solves the dual problem for ``A.T`` and ``C.T`` and returns ``L`` such that
    ``eig(A - L C)`` matches ``desired_poles`` (up to numerical tolerance).
    """
    A = np.asarray(A, dtype=float)
    C = np.asarray(C, dtype=float)
    poles = np.asarray(desired_poles, dtype=complex).reshape(-1)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square with shape (n, n).")
    if C.ndim == 1:
        C = C.reshape(1, -1)
    if C.ndim != 2 or C.shape[1] != A.shape[0]:
        raise ValueError("C must have shape (p, n) compatible with A.")
    if poles.size != A.shape[0]:
        raise ValueError("desired_poles must contain exactly n poles.")

    placed = place_poles(A.T, C.T, poles)
    return np.asarray(placed.gain_matrix.T, dtype=float)


def run_kalman_filter(
    A: ArrayLike,
    B: ArrayLike,
    C: ArrayLike,
    Q: ArrayLike,
    R: ArrayLike,
    u: ArrayLike,
    y: ArrayLike,
    x0: ArrayLike | None = None,
    P0: ArrayLike | None = None,
) -> tuple[Array, Array]:
    """Run a discrete-time Kalman filter for a linear Gaussian state-space model.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(x_filt, P_filt)`` where ``x_filt`` has shape ``(T, n)`` and ``P_filt``
        has shape ``(T, n, n)``.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)
    u = np.asarray(u, dtype=float)
    y = np.asarray(y, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square with shape (n, n).")
    n = A.shape[0]

    if B.ndim == 1:
        B = B.reshape(-1, 1)
    if B.shape[0] != n:
        raise ValueError("B must have shape (n, m).")
    m = B.shape[1]

    if C.ndim == 1:
        C = C.reshape(1, -1)
    if C.shape[1] != n:
        raise ValueError("C must have shape (p, n).")
    p = C.shape[0]

    if Q.shape != (n, n):
        raise ValueError("Q must have shape (n, n).")
    if R.shape != (p, p):
        raise ValueError("R must have shape (p, p).")

    if u.ndim == 1:
        if m != 1:
            raise ValueError("u must have shape (T, m) for multi-input systems.")
        u = u.reshape(-1, 1)
    if u.ndim != 2 or u.shape[1] != m:
        raise ValueError("u must have shape (T, m).")

    if y.ndim == 1:
        if p != 1:
            raise ValueError("y must have shape (T, p) for multi-output systems.")
        y = y.reshape(-1, 1)
    if y.ndim != 2 or y.shape[1] != p:
        raise ValueError("y must have shape (T, p).")

    T = y.shape[0]
    if u.shape[0] != T:
        raise ValueError("u and y must have the same number of time steps.")

    x = np.zeros(n, dtype=float) if x0 is None else np.asarray(x0, dtype=float).reshape(n)
    P = np.eye(n, dtype=float) if P0 is None else np.asarray(P0, dtype=float)
    if P.shape != (n, n):
        raise ValueError("P0 must have shape (n, n).")

    x_filt = np.empty((T, n), dtype=float)
    P_filt = np.empty((T, n, n), dtype=float)

    I = np.eye(n, dtype=float)
    for k in range(T):
        x_pred = A @ x + B @ u[k]
        P_pred = A @ P @ A.T + Q

        innov = y[k] - C @ x_pred
        S = C @ P_pred @ C.T + R
        K = P_pred @ C.T @ np.linalg.solve(S, np.eye(p))

        x = x_pred + K @ innov
        # Joseph form for numerical robustness.
        IKC = I - K @ C
        P = IKC @ P_pred @ IKC.T + K @ R @ K.T

        x_filt[k] = x
        P_filt[k] = P

    return x_filt, P_filt
