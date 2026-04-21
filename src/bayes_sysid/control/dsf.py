from __future__ import annotations

import numpy as np

Array = np.ndarray


def _validate_transfer_matrix_array(G: Array) -> Array:
    G = np.asarray(G, dtype=complex)
    if G.ndim == 2:
        G = G[None, ...]
    if G.ndim != 3:
        raise ValueError("G must have shape (p, m) or (n_freq, p, m).")
    if G.shape[1] == 0 or G.shape[2] == 0:
        raise ValueError("G must have non-zero output and input dimensions.")
    return G


def transfer_matrix_from_mimo_arx(
    a_lags: Array,
    b_lags: Array,
    w: Array,
) -> Array:
    """Build frequency-domain MIMO transfer matrices from ARX lag tensors.

    ARX convention:
    ``y[t] + sum_{k=1..na} A_k y[t-k] = sum_{k=1..nb} B_k u[t-k]``.

    Parameters
    ----------
    a_lags:
        Array of shape ``(na, p, p)``.
    b_lags:
        Array of shape ``(nb, p, m)``.
    w:
        Frequency grid in rad/sample with shape ``(n_freq,)``.

    Returns
    -------
    np.ndarray
        Complex transfer matrix samples with shape ``(n_freq, p, m)``.
    """
    a_lags = np.asarray(a_lags, dtype=float)
    b_lags = np.asarray(b_lags, dtype=float)
    w = np.asarray(w, dtype=float).reshape(-1)

    if a_lags.ndim != 3:
        raise ValueError("a_lags must have shape (na, p, p).")
    if b_lags.ndim != 3:
        raise ValueError("b_lags must have shape (nb, p, m).")
    if w.size == 0:
        raise ValueError("w must contain at least one frequency.")

    na, p, p2 = a_lags.shape
    nb, p_b, m = b_lags.shape
    if p != p2:
        raise ValueError("a_lags must be square in its last two dimensions.")
    if p_b != p:
        raise ValueError("Output dimension mismatch between a_lags and b_lags.")

    eye = np.eye(p, dtype=complex)
    G = np.empty((w.size, p, m), dtype=complex)

    for idx, wk in enumerate(w):
        z_inv = np.exp(-1j * wk)
        A_eval = eye.copy()
        B_eval = np.zeros((p, m), dtype=complex)

        for k in range(na):
            A_eval += a_lags[k] * (z_inv ** (k + 1))
        for k in range(nb):
            B_eval += b_lags[k] * (z_inv ** (k + 1))

        G[idx] = np.linalg.solve(A_eval, B_eval)

    return G


def validate_identifiability_assumptions(
    G: Array,
    rank_tol: float = 1e-8,
    diagonal_tol: float = 1e-8,
) -> dict[str, object]:
    """Heuristic DSF identifiability checks for square transfer matrices.

    Notes
    -----
    This helper is intentionally conservative and not a formal theorem. It checks
    practical preconditions commonly used by DSF reconstructions: square transfer
    map, invertibility across frequencies, and sufficiently non-degenerate direct
    channels.
    """
    G3 = _validate_transfer_matrix_array(G)
    _, p, m = G3.shape
    is_square = p == m
    full_rank = []
    min_abs_diag = []

    for Gk in G3:
        svals = np.linalg.svd(Gk, compute_uv=False)
        full_rank.append(bool(svals[-1] > rank_tol))
        min_abs_diag.append(float(np.min(np.abs(np.diag(Gk[: min(p, m), : min(p, m)])))))

    all_full_rank = bool(np.all(full_rank))
    diagonal_nonzero = bool(np.all(np.asarray(min_abs_diag) > diagonal_tol))

    return {
        "is_square": is_square,
        "full_rank_all_frequencies": all_full_rank,
        "diagonal_nonzero_all_frequencies": diagonal_nonzero,
        "passes": bool(is_square and all_full_rank and diagonal_nonzero),
        "rank_tol": rank_tol,
        "diagonal_tol": diagonal_tol,
    }


def validate_excitation_richness(
    u: Array,
    rank_tol: float = 1e-8,
    condition_number_max: float = 1e6,
) -> dict[str, object]:
    """Check whether MIMO excitation appears informative enough for DSF experiments."""
    u = np.asarray(u, dtype=float)
    if u.ndim != 2:
        raise ValueError("u must have shape (n_samples, n_inputs).")
    n_samples, n_inputs = u.shape
    if n_samples < 2:
        raise ValueError("u must contain at least two samples.")

    uc = u - u.mean(axis=0, keepdims=True)
    gram = (uc.T @ uc) / max(n_samples - 1, 1)
    svals = np.linalg.svd(gram, compute_uv=False)

    rank = int(np.sum(svals > rank_tol))
    if svals[-1] <= rank_tol:
        condition_number = np.inf
    else:
        condition_number = float(svals[0] / svals[-1])

    return {
        "n_samples": n_samples,
        "n_inputs": n_inputs,
        "rank": rank,
        "full_rank": rank == n_inputs,
        "condition_number": condition_number,
        "well_conditioned": condition_number <= condition_number_max,
        "passes": bool(rank == n_inputs and condition_number <= condition_number_max),
        "rank_tol": rank_tol,
        "condition_number_max": condition_number_max,
    }


def dsf_from_transfer_matrix(
    G: Array,
    method: str = "stable_factorization",
) -> dict[str, Array | str]:
    """Prototype DSF factorization from transfer matrix samples.

    For square ``G``, this routine returns per-frequency factors ``Q`` and ``P``
    such that approximately ``G = (I - Q)^{-1} P``. The current implementation is
    a pragmatic algebraic factorization: ``P = diag(diag(G))`` and
    ``Q = I - P @ inv(G)``.

    Limitations
    -----------
    - Prototype only; no formal identifiability proof or uniqueness guarantees.
    - The ``stable_factorization`` name reflects intended future extension to
      dynamic/stable DSF constraints, not a complete theorem-backed implementation.
    """
    if method != "stable_factorization":
        raise ValueError(f"Unsupported method '{method}'.")

    G3 = _validate_transfer_matrix_array(G)
    n_freq, p, m = G3.shape
    if p != m:
        raise ValueError("Current DSF prototype requires square transfer matrices (p == m).")

    I = np.eye(p, dtype=complex)
    Q = np.empty((n_freq, p, p), dtype=complex)
    P = np.empty((n_freq, p, p), dtype=complex)

    for k, Gk in enumerate(G3):
        diag_terms = np.diag(np.diag(Gk))
        P[k] = diag_terms
        Qk = I - diag_terms @ np.linalg.inv(Gk)
        np.fill_diagonal(Qk, 0.0)
        Q[k] = Qk

    return {"Q": Q, "P": P, "method": method}


def posterior_edge_probability(dsf_samples: Array, threshold: float) -> Array:
    """Estimate posterior edge existence probability from DSF samples.

    Parameters
    ----------
    dsf_samples:
        DSF edge samples with shape ``(n_samples, p, p)`` or
        ``(n_samples, n_freq, p, p)``.
    threshold:
        Edge is counted as present if its maximum magnitude exceeds this value.
    """
    if threshold < 0:
        raise ValueError("threshold must be non-negative.")

    samples = np.asarray(dsf_samples)
    if samples.ndim == 3:
        edge_mag = np.abs(samples)
    elif samples.ndim == 4:
        edge_mag = np.max(np.abs(samples), axis=1)
    else:
        raise ValueError("dsf_samples must have shape (n_samples,p,p) or (n_samples,n_freq,p,p).")

    edge_events = edge_mag > threshold
    probs = edge_events.mean(axis=0)
    if probs.shape[0] == probs.shape[1]:
        np.fill_diagonal(probs, 0.0)
    return probs
