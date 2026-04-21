from __future__ import annotations

import numpy as np


Array = np.ndarray


def validate_realization_shapes(A: Array, B: Array, C: Array, D: Array) -> tuple[Array, Array, Array, Array]:
    """Validate and normalize state-space arrays for a SISO realization.

    Parameters
    ----------
    A, B, C, D:
        State-space matrices for the realization
            x[k+1] = A x[k] + B u[k]
            y[k]   = C x[k] + D u[k].

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        The validated arrays cast to ``float`` with shapes ``(n, n)``, ``(n, 1)``,
        ``(1, n)``, ``(1, 1)``.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    D = np.asarray(D, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square with shape (n, n).")

    n = A.shape[0]
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    if C.ndim == 1:
        C = C.reshape(1, -1)
    if D.ndim == 0:
        D = D.reshape(1, 1)

    if B.shape != (n, 1):
        raise ValueError("B must have shape (n, 1) for SISO systems.")
    if C.shape != (1, n):
        raise ValueError("C must have shape (1, n) for SISO systems.")
    if D.shape != (1, 1):
        raise ValueError("D must have shape (1, 1) for SISO systems.")

    return A, B, C, D


def arx_to_state_space(a: Array, b: Array, dt: float | None = None) -> tuple[Array, Array, Array, Array]:
    """Construct a deterministic controllable companion realization of a SISO ARX model.

    The ARX model convention is
    ``y[k] + a1*y[k-1] + ... + ana*y[k-na] = b1*u[k-1] + ... + bnb*u[k-nb]``.

    Parameters
    ----------
    a, b:
        ARX denominator and numerator coefficients (without leading 1).
    dt:
        Optional sample time. Included for API compatibility and validated when
        provided, but not embedded in the returned matrix tuple.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ``(A, B, C, D)`` in controllable companion form.
    """
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)

    if dt is not None and dt <= 0:
        raise ValueError("dt must be positive when provided.")

    na = int(a.size)
    nb = int(b.size)
    if na < 1:
        raise ValueError("a must contain at least one denominator coefficient.")
    if nb > na:
        raise ValueError("ARX transfer function must be proper (len(b) <= len(a)).")

    n = na
    b_pad = np.zeros(n, dtype=float)
    b_pad[:nb] = b

    A = np.zeros((n, n), dtype=float)
    if n > 1:
        A[:-1, 1:] = np.eye(n - 1)
    A[-1, :] = -a[::-1]

    B = np.zeros((n, 1), dtype=float)
    B[-1, 0] = 1.0

    C = b_pad[::-1].reshape(1, -1)
    D = np.zeros((1, 1), dtype=float)

    return validate_realization_shapes(A, B, C, D)


def _controllability_matrix(A: Array, B: Array) -> Array:
    n = A.shape[0]
    blocks = [B]
    AB = B
    for _ in range(1, n):
        AB = A @ AB
        blocks.append(AB)
    return np.hstack(blocks)


def _observability_matrix(A: Array, C: Array) -> Array:
    n = A.shape[0]
    blocks = [C]
    CA = C
    for _ in range(1, n):
        CA = CA @ A
        blocks.append(CA)
    return np.vstack(blocks)


def minimal_realization(
    A: Array,
    B: Array,
    C: Array,
    D: Array,
    tol: float = 1e-9,
) -> tuple[Array, Array, Array, Array, np.ndarray]:
    """Compute a numerically minimal SISO realization by trimming unreachable/unobservable modes.

    Returns a reduced-order realization and indices of original states whose basis
    vectors have non-negligible projection on the retained minimal subspace.
    """
    A, B, C, D = validate_realization_shapes(A, B, C, D)
    n = A.shape[0]

    ctrb = _controllability_matrix(A, B)
    Uc, sc, _ = np.linalg.svd(ctrb, full_matrices=False)
    rc = int(np.sum(sc > tol))
    if rc == 0:
        kept = np.array([], dtype=int)
        return np.zeros((0, 0)), np.zeros((0, 1)), np.zeros((1, 0)), D.copy(), kept

    Tc = Uc[:, :rc]
    A_c = Tc.T @ A @ Tc
    B_c = Tc.T @ B
    C_c = C @ Tc

    ob_c = _observability_matrix(A_c, C_c)
    Uo, so, _ = np.linalg.svd(ob_c.T, full_matrices=False)
    ro = int(np.sum(so > tol))
    To = Uo[:, :ro]

    A_m = To.T @ A_c @ To
    B_m = To.T @ B_c
    C_m = C_c @ To

    T_total = Tc @ To
    support = np.linalg.norm(T_total, axis=1)
    kept_states = np.flatnonzero(support > tol)

    return validate_realization_shapes(A_m, B_m, C_m, D.copy()) + (kept_states,)
