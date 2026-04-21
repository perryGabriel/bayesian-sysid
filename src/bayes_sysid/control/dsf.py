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



def transfer_matrix_samples_from_mimo_posterior(
    posterior_samples: dict[str, Array],
    w: Array,
) -> Array:
    """Construct transfer-matrix posterior samples from native MIMO ARX samples.

    Parameters
    ----------
    posterior_samples:
        Dictionary containing ``a_lags`` and ``b_lags`` from
        ``BayesianMIMOARX.sample_parameters``.
    w:
        Frequency grid with shape ``(n_freq,)`` in rad/sample.

    Returns
    -------
    np.ndarray
        Complex array with shape ``(n_samples, n_freq, p, m)``.
    """
    if "a_lags" not in posterior_samples or "b_lags" not in posterior_samples:
        raise ValueError("posterior_samples must include 'a_lags' and 'b_lags'.")

    a_samples = np.asarray(posterior_samples["a_lags"], dtype=float)
    b_samples = np.asarray(posterior_samples["b_lags"], dtype=float)
    if a_samples.ndim != 4:
        raise ValueError("a_lags samples must have shape (n_samples, na, p, p).")
    if b_samples.ndim != 4:
        raise ValueError("b_lags samples must have shape (n_samples, nb, p, m).")
    if a_samples.shape[0] != b_samples.shape[0]:
        raise ValueError("a_lags and b_lags must have matching n_samples.")

    n_samples = a_samples.shape[0]
    G_samples = np.empty((n_samples, np.asarray(w).reshape(-1).size, a_samples.shape[2], b_samples.shape[3]), dtype=complex)
    for s_idx in range(n_samples):
        G_samples[s_idx] = transfer_matrix_from_mimo_arx(a_samples[s_idx], b_samples[s_idx], w)
    return G_samples

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
    return excitation_richness_score(
        u,
        rank_tol=rank_tol,
        condition_number_max=condition_number_max,
    )


def heuristic_identifiability_screening(
    G: Array,
    rank_tol: float = 1e-8,
    diagonal_tol: float = 1e-8,
) -> dict[str, object]:
    """Heuristic DSF screen (not a theorem-backed guarantee)."""
    return validate_identifiability_assumptions(
        G,
        rank_tol=rank_tol,
        diagonal_tol=diagonal_tol,
    )


def evaluate_transfer_rank_identifiability(
    G: Array,
    rank_tol: float = 1e-8,
) -> dict[str, object]:
    """Evaluate frequency-wise rank conditions for transfer-map identifiability."""
    G3 = _validate_transfer_matrix_array(G)
    n_freq, p, m = G3.shape
    min_singular_values = np.empty(n_freq, dtype=float)
    rank_per_freq = np.empty(n_freq, dtype=int)

    for k, Gk in enumerate(G3):
        svals = np.linalg.svd(Gk, compute_uv=False)
        min_singular_values[k] = float(svals[-1])
        rank_per_freq[k] = int(np.sum(svals > rank_tol))

    deficient = np.flatnonzero(rank_per_freq < min(p, m))
    return {
        "condition_name": "transfer_rank",
        "n_freq": n_freq,
        "shape": (p, m),
        "rank_tol": rank_tol,
        "rank_per_frequency": rank_per_freq,
        "min_singular_values": min_singular_values,
        "rank_deficient_indices": deficient,
        "satisfied": bool(deficient.size == 0),
    }


def evaluate_persistence_of_excitation(
    u: Array,
    max_lag: int = 5,
    rank_tol: float = 1e-8,
    condition_number_max: float = 1e6,
) -> dict[str, object]:
    """Evaluate persistence-of-excitation via block-Hankel embedding rank."""
    u = np.asarray(u, dtype=float)
    if u.ndim != 2:
        raise ValueError("u must have shape (n_samples, n_inputs).")
    n_samples, n_inputs = u.shape
    if n_samples <= max_lag:
        raise ValueError("u must have more rows than max_lag.")

    rows = n_samples - max_lag + 1
    hankel = np.zeros((rows, n_inputs * max_lag), dtype=float)
    for lag in range(max_lag):
        hankel[:, lag * n_inputs : (lag + 1) * n_inputs] = u[lag : lag + rows]

    centered = hankel - hankel.mean(axis=0, keepdims=True)
    gram = (centered.T @ centered) / max(rows - 1, 1)
    svals = np.linalg.svd(gram, compute_uv=False)
    rank = int(np.sum(svals > rank_tol))
    target_rank = n_inputs * max_lag
    condition_number = np.inf if svals[-1] <= rank_tol else float(svals[0] / svals[-1])
    return {
        "n_samples": n_samples,
        "n_inputs": n_inputs,
        "max_lag": max_lag,
        "hankel_shape": hankel.shape,
        "rank": rank,
        "target_rank": target_rank,
        "full_rank": rank == target_rank,
        "condition_number": condition_number,
        "well_conditioned": condition_number <= condition_number_max,
        "passes": bool(rank == target_rank and condition_number <= condition_number_max),
        "rank_tol": rank_tol,
        "condition_number_max": condition_number_max,
        "singular_values": svals,
    }


def evaluate_identifiability_conditions(
    G: Array,
    u: Array,
    max_lag: int = 5,
    rank_tol: float = 1e-8,
    condition_number_max: float = 1e6,
) -> dict[str, object]:
    """Guarantee-oriented identifiability evaluators with structured diagnostics."""
    rank_diag = evaluate_transfer_rank_identifiability(G, rank_tol=rank_tol)
    pe_diag = evaluate_persistence_of_excitation(
        u,
        max_lag=max_lag,
        rank_tol=rank_tol,
        condition_number_max=condition_number_max,
    )
    return {
        "rank_condition": rank_diag,
        "persistence_of_excitation": pe_diag,
        "all_conditions_satisfied": bool(rank_diag["satisfied"] and pe_diag["passes"]),
    }


def minimum_horizon_guidance(
    n_inputs: int,
    model_order: int,
    safety_factor: float = 10.0,
) -> dict[str, object]:
    """Recommend minimum data horizon for DSF/ARX-style experiment design."""
    if n_inputs < 1 or model_order < 1:
        raise ValueError("n_inputs and model_order must be positive.")
    min_horizon_rank = n_inputs * model_order + 1
    recommended_horizon = int(np.ceil(safety_factor * min_horizon_rank))
    return {
        "n_inputs": n_inputs,
        "model_order": model_order,
        "minimum_rank_horizon": min_horizon_rank,
        "recommended_horizon": recommended_horizon,
        "safety_factor": safety_factor,
    }


def excitation_richness_score(
    u: Array,
    rank_tol: float = 1e-8,
    condition_number_max: float = 1e6,
) -> dict[str, object]:
    """Score excitation richness with rank, conditioning, and balance breakdown."""
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
    condition_number = np.inf if svals[-1] <= rank_tol else float(svals[0] / svals[-1])
    variances = np.diag(gram)
    positive_variances = variances[variances > rank_tol]
    variance_balance = (
        0.0
        if positive_variances.size == 0
        else float(np.min(positive_variances) / np.max(positive_variances))
    )
    conditioning_score = 0.0 if not np.isfinite(condition_number) else float(
        min(1.0, condition_number_max / max(condition_number, 1.0))
    )
    rank_score = float(rank / n_inputs)
    total_score = float((rank_score + conditioning_score + variance_balance) / 3.0)

    return {
        "n_samples": n_samples,
        "n_inputs": n_inputs,
        "rank": rank,
        "full_rank": rank == n_inputs,
        "condition_number": condition_number,
        "well_conditioned": condition_number <= condition_number_max,
        "rank_score": rank_score,
        "conditioning_score": conditioning_score,
        "variance_balance_score": variance_balance,
        "total_score": total_score,
        "passes": bool(rank == n_inputs and condition_number <= condition_number_max),
        "rank_tol": rank_tol,
        "condition_number_max": condition_number_max,
    }


def identifiability_warning_flags(
    identifiability: dict[str, object],
    edge_probabilities: Array | None = None,
    high_probability_threshold: float = 0.9,
) -> list[str]:
    """Emit warning flags for unidentifiable or weakly identified patterns."""
    warnings: list[str] = []
    rank_condition = identifiability.get("rank_condition", {})
    pe = identifiability.get("persistence_of_excitation", {})

    deficient = np.asarray(rank_condition.get("rank_deficient_indices", np.array([], dtype=int)))
    if deficient.size > 0:
        warnings.append("transfer_rank_deficient_frequencies")
    if not bool(pe.get("passes", False)):
        warnings.append("insufficient_persistence_of_excitation")

    if edge_probabilities is not None:
        probs = np.asarray(edge_probabilities, dtype=float)
        if probs.ndim != 2 or probs.shape[0] != probs.shape[1]:
            raise ValueError("edge_probabilities must have shape (p,p).")
        for i in range(probs.shape[0]):
            for j in range(i + 1, probs.shape[1]):
                if probs[i, j] >= high_probability_threshold and probs[j, i] >= high_probability_threshold:
                    warnings.append(f"bidirectional_high_confidence_edge_pair:{i}-{j}")
    return warnings


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
    return posterior_edge_confidence_summary(dsf_samples, threshold)["posterior_probability"]


def posterior_edge_confidence_summary(
    dsf_samples: Array,
    threshold: float,
    credible_level: float = 0.95,
    calibration_truth: Array | None = None,
    calibration_bins: int = 10,
) -> dict[str, object]:
    """Posterior edge summary with uncertainty and optional calibration metrics."""
    if threshold < 0:
        raise ValueError("threshold must be non-negative.")
    if not (0.0 < credible_level < 1.0):
        raise ValueError("credible_level must be in (0, 1).")

    samples = np.asarray(dsf_samples)
    if samples.ndim == 3:
        edge_mag = np.abs(samples)
    elif samples.ndim == 4:
        edge_mag = np.max(np.abs(samples), axis=1)
    else:
        raise ValueError("dsf_samples must have shape (n_samples,p,p) or (n_samples,n_freq,p,p).")

    edge_events = edge_mag > threshold
    n_samples = edge_events.shape[0]
    probs = edge_events.mean(axis=0).astype(float)
    z = 1.96 if credible_level >= 0.95 else 1.64
    se = np.sqrt(np.maximum(probs * (1.0 - probs) / max(n_samples, 1), 0.0))
    lower = np.clip(probs - z * se, 0.0, 1.0)
    upper = np.clip(probs + z * se, 0.0, 1.0)

    if probs.shape[0] == probs.shape[1]:
        np.fill_diagonal(probs, 0.0)
        np.fill_diagonal(lower, 0.0)
        np.fill_diagonal(upper, 0.0)

    summary: dict[str, object] = {
        "n_samples": n_samples,
        "threshold": threshold,
        "credible_level": credible_level,
        "posterior_probability": probs,
        "credible_interval_lower": lower,
        "credible_interval_upper": upper,
        "uncertainty_std_error": se,
    }

    if calibration_truth is not None:
        truth = np.asarray(calibration_truth).astype(bool)
        if truth.shape != probs.shape:
            raise ValueError("calibration_truth must have the same shape as posterior probabilities.")

        flat_prob = probs.reshape(-1)
        flat_truth = truth.reshape(-1).astype(float)
        brier = float(np.mean((flat_prob - flat_truth) ** 2))

        bin_edges = np.linspace(0.0, 1.0, calibration_bins + 1)
        ece = 0.0
        bin_stats = []
        for i in range(calibration_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            if i < calibration_bins - 1:
                mask = (flat_prob >= lo) & (flat_prob < hi)
            else:
                mask = (flat_prob >= lo) & (flat_prob <= hi)
            if not np.any(mask):
                continue
            conf = float(np.mean(flat_prob[mask]))
            acc = float(np.mean(flat_truth[mask]))
            frac = float(np.mean(mask))
            ece += abs(acc - conf) * frac
            bin_stats.append({"bin": (float(lo), float(hi)), "confidence": conf, "accuracy": acc, "fraction": frac})

        summary["calibration"] = {"brier_score": brier, "ece": float(ece), "bins": bin_stats}
    return summary
