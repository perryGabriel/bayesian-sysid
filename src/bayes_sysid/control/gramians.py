from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import solve_discrete_lyapunov

from .realization import arx_to_state_space


Array = np.ndarray


def _as_2d(arr: ArrayLike, name: str) -> Array:
    x = np.asarray(arr, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2D.")
    return x


def _validate_gramian_inputs(A: ArrayLike, X: ArrayLike, name: str) -> tuple[Array, Array]:
    A = _as_2d(A, "A")
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square with shape (n, n).")
    X = _as_2d(X, name)
    if X.shape[0] != A.shape[0]:
        raise ValueError(f"{name} must have compatible state dimension with A.")
    return A, X


def controllability_gramian(A: ArrayLike, B: ArrayLike) -> Array:
    """Solve the discrete-time controllability Gramian.

    Returns ``Wc`` solving ``A Wc A.T - Wc + B B.T = 0``.
    """
    A, B = _validate_gramian_inputs(A, B, "B")
    return np.asarray(solve_discrete_lyapunov(A, B @ B.T), dtype=float)


def observability_gramian(A: ArrayLike, C: ArrayLike) -> Array:
    """Solve the discrete-time observability Gramian.

    Returns ``Wo`` solving ``A.T Wo A - Wo + C.T C = 0``.
    """
    A = _as_2d(A, "A")
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square with shape (n, n).")
    C = _as_2d(C, "C")
    if C.shape[1] != A.shape[0]:
        raise ValueError("C must have shape (p, n) compatible with A.")
    return np.asarray(solve_discrete_lyapunov(A.T, C.T @ C), dtype=float)


def _gramian_diagnostics(W: Array, tol: float = 1e-10) -> dict[str, float | bool]:
    Ws = 0.5 * (W + W.T)
    sym_rel = float(np.linalg.norm(W - W.T, ord="fro") / max(np.linalg.norm(W, ord="fro"), tol))
    eig = np.linalg.eigvalsh(Ws)
    min_eig = float(np.min(eig))
    cond = float(np.linalg.cond(Ws)) if np.linalg.norm(Ws, ord=2) > 0 else float("inf")
    return {
        "symmetry_relative_error": sym_rel,
        "is_symmetric": bool(sym_rel <= 1e-7),
        "min_eigenvalue": min_eig,
        "is_psd": bool(min_eig >= -tol),
        "condition_number": cond,
    }


def hankel_singular_values(A: ArrayLike, B: ArrayLike, C: ArrayLike) -> Array:
    """Compute discrete-time Hankel singular values from Gramian product eigenvalues."""
    Wc = controllability_gramian(A, B)
    Wo = observability_gramian(A, C)
    lam = np.linalg.eigvals(Wc @ Wo)
    hsv = np.sqrt(np.clip(np.real(lam), 0.0, np.inf))
    return np.sort(hsv)[::-1]


def posterior_hsv_summary(
    model,
    n_samples: int = 400,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    energy_levels: tuple[float, ...] = (0.9, 0.95, 0.99),
    random_state: int | None = None,
    stability_margin: float = 1e-8,
    near_mode_ratio: float = 1e-4,
) -> dict[str, object]:
    """Posterior summary of Hankel singular values from ARX parameter samples."""
    theta_samples = model.sample_parameters(n_samples=n_samples, random_state=random_state)
    qs = np.asarray(quantiles, dtype=float)

    hsv_samples: list[Array] = []
    diag_wc: list[dict[str, float | bool]] = []
    diag_wo: list[dict[str, float | bool]] = []
    warnings_list: list[str] = []

    stable = 0
    for theta in theta_samples:
        a = np.asarray(theta[: model.na], dtype=float)
        b = np.asarray(theta[model.na : model.na + model.nb], dtype=float)
        A, B, C, _ = arx_to_state_space(a=a, b=b)

        poles = np.linalg.eigvals(A)
        if not np.all(np.abs(poles) < 1.0 - stability_margin):
            continue
        stable += 1

        Wc = controllability_gramian(A, B)
        Wo = observability_gramian(A, C)
        dc = _gramian_diagnostics(Wc)
        do = _gramian_diagnostics(Wo)
        diag_wc.append(dc)
        diag_wo.append(do)

        hsv = hankel_singular_values(A, B, C)
        hsv_samples.append(hsv)

        if hsv.size > 0 and hsv[0] > 0.0:
            weak_modes = int(np.sum(hsv / hsv[0] < near_mode_ratio))
            if weak_modes > 0:
                warnings_list.append(
                    f"Detected {weak_modes} near-uncontrollable/unobservable mode(s) with HSV ratio < {near_mode_ratio:.1e}."
                )

    if stable == 0:
        warnings.warn("No stable posterior samples found; HSV summary is empty.", RuntimeWarning)
        return {
            "n_samples": int(n_samples),
            "n_stable": 0,
            "stable_fraction": 0.0,
            "quantiles": tuple(float(q) for q in qs),
            "hsv_quantiles": [],
            "mode_energy_retention": {},
            "gramian_diagnostics": {"Wc": {}, "Wo": {}},
            "warnings": ["No stable posterior samples found."],
        }

    hsv_arr = np.vstack(hsv_samples)
    hsv_q = np.quantile(hsv_arr, qs, axis=0)
    hsv_median = np.quantile(hsv_arr, 0.5, axis=0)

    total = float(np.sum(hsv_median))
    cum = np.cumsum(hsv_median) / total if total > 0 else np.zeros_like(hsv_median)
    retention = {}
    for level in energy_levels:
        idx = int(np.searchsorted(cum, level, side="left") + 1)
        retention[f"{level:.2f}"] = min(idx, hsv_median.size)

    def _aggregate(diags: list[dict[str, float | bool]]) -> dict[str, float]:
        conds = np.array([float(d["condition_number"]) for d in diags], dtype=float)
        mins = np.array([float(d["min_eigenvalue"]) for d in diags], dtype=float)
        sym = np.array([float(d["symmetry_relative_error"]) for d in diags], dtype=float)
        return {
            "median_condition_number": float(np.median(conds)),
            "q90_condition_number": float(np.quantile(conds, 0.9)),
            "min_eigenvalue_min": float(np.min(mins)),
            "symmetry_relative_error_max": float(np.max(sym)),
        }

    return {
        "n_samples": int(n_samples),
        "n_stable": int(stable),
        "stable_fraction": float(stable / n_samples),
        "quantiles": tuple(float(q) for q in qs),
        "hsv_quantiles": hsv_q.tolist(),
        "mode_energy_retention": retention,
        "gramian_diagnostics": {
            "Wc": _aggregate(diag_wc),
            "Wo": _aggregate(diag_wo),
        },
        "warnings": sorted(set(warnings_list)),
    }
