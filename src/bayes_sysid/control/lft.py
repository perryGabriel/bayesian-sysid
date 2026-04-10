from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from bayes_sysid.analysis.frequency_response import arx_frequency_response


@dataclass
class DeltaBlock:
    """Minimal SISO uncertainty block for robust-stability sampling.

    Parameters
    ----------
    gain_bound:
        Absolute bound for |Delta(jw)|.
    frequency_grid:
        Optional frequency grid used for dynamic samples.
    dynamic:
        If True, sampled realizations are frequency-dependent. Otherwise,
        realizations are static real scalars.
    """

    gain_bound: float
    frequency_grid: np.ndarray | None = None
    dynamic: bool = False

    def __post_init__(self) -> None:
        self.gain_bound = float(self.gain_bound)
        if self.gain_bound <= 0:
            raise ValueError("gain_bound must be positive.")
        if self.frequency_grid is not None:
            self.frequency_grid = np.asarray(self.frequency_grid, dtype=float).reshape(-1)
            if len(self.frequency_grid) == 0:
                raise ValueError("frequency_grid must be non-empty when provided.")

    def sample(self, rng: np.random.Generator, n_freq: int | None = None) -> np.ndarray:
        """Sample one Delta realization over frequency."""
        if self.dynamic:
            if self.frequency_grid is None and n_freq is None:
                raise ValueError("n_freq is required for dynamic Delta without frequency_grid.")
            n = len(self.frequency_grid) if self.frequency_grid is not None else int(n_freq)
            mag = self.gain_bound * rng.random(n)
            phase = rng.uniform(-np.pi, np.pi, size=n)
            return mag * np.exp(1j * phase)

        scalar = rng.uniform(-self.gain_bound, self.gain_bound)
        if n_freq is None:
            return np.asarray([scalar], dtype=complex)
        return np.full(int(n_freq), scalar, dtype=complex)


@dataclass
class RobustStabilityResult:
    probability: float
    ci_low: float
    ci_high: float
    n_theta_samples: int


def build_nominal_interconnection(
    theta: ArrayLike,
    na: int,
    nb: int,
    controller_gain: float,
    w: ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a SISO interconnection M(theta) for output-feedback uncertainty loop.

    The loop uses:
        u = K (r - y),
        y = G (u + w),
        z = y,
        w = Delta z.

    Returning partition blocks M11, M12, M21, M22 such that
        [z; y] = M [r; w].
    """
    G = arx_frequency_response(theta=theta, na=na, nb=nb, w=w)
    K = complex(controller_gain)
    den = 1.0 + K * G
    M11 = (K * G) / den
    M12 = G / den
    M21 = M11.copy()
    M22 = M12.copy()
    return M11, M12, M21, M22


def lower_lft_siso(
    M11: ArrayLike,
    M12: ArrayLike,
    M21: ArrayLike,
    M22: ArrayLike,
    delta: ArrayLike,
    regularization: float = 1e-12,
) -> np.ndarray:
    """Evaluate lower LFT F_l(M, Delta) for scalar SISO blocks."""
    m11 = np.asarray(M11, dtype=complex)
    m12 = np.asarray(M12, dtype=complex)
    m21 = np.asarray(M21, dtype=complex)
    m22 = np.asarray(M22, dtype=complex)
    d = np.asarray(delta, dtype=complex)
    inv_term = 1.0 / (1.0 - m22 * d + regularization)
    return m11 + m12 * d * inv_term * m21


def upper_lft_siso(
    M11: ArrayLike,
    M12: ArrayLike,
    M21: ArrayLike,
    M22: ArrayLike,
    delta: ArrayLike,
    regularization: float = 1e-12,
) -> np.ndarray:
    """Evaluate upper LFT F_u(M, Delta) for scalar SISO blocks."""
    m11 = np.asarray(M11, dtype=complex)
    m12 = np.asarray(M12, dtype=complex)
    m21 = np.asarray(M21, dtype=complex)
    m22 = np.asarray(M22, dtype=complex)
    d = np.asarray(delta, dtype=complex)
    inv_term = 1.0 / (1.0 - m11 * d + regularization)
    return m22 + m21 * d * inv_term * m12


def _wilson_interval(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n <= 0:
        raise ValueError("n must be positive.")
    z = 1.959963984540054  # 95% default
    if alpha != 0.05:
        from scipy.stats import norm

        z = float(norm.ppf(1 - alpha / 2.0))
    p = successes / n
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def posterior_robust_stability_confidence(
    posterior_model,
    controller_gain: float,
    delta_block: DeltaBlock,
    w: ArrayLike,
    n_theta_samples: int = 200,
    n_delta_samples: int = 200,
    random_state: int | None = None,
    margin_tol: float = 1e-3,
    alpha: float = 0.05,
) -> RobustStabilityResult:
    """Estimate P(robustly stable | D) under posterior and uncertainty sampling.

    A sampled theta is declared robustly stable when all sampled Delta realizations
    satisfy max_w |M22(theta, w) Delta(w)| < 1 - margin_tol.
    """
    w = np.asarray(w, dtype=float).reshape(-1)
    if len(w) == 0:
        raise ValueError("w must be non-empty.")
    theta_samples = posterior_model.sample_parameters(
        n_samples=n_theta_samples,
        random_state=random_state,
    )
    rng = np.random.default_rng(random_state)

    robust_count = 0
    for theta in theta_samples:
        _, _, _, m22 = build_nominal_interconnection(
            theta=theta,
            na=posterior_model.na,
            nb=posterior_model.nb,
            controller_gain=controller_gain,
            w=w,
        )

        stable_for_all_delta = True
        for _ in range(n_delta_samples):
            delta = delta_block.sample(rng=rng, n_freq=len(w))
            if np.max(np.abs(m22 * delta)) >= (1.0 - margin_tol):
                stable_for_all_delta = False
                break

        robust_count += int(stable_for_all_delta)

    p_hat = robust_count / n_theta_samples
    ci_low, ci_high = _wilson_interval(robust_count, n_theta_samples, alpha=alpha)
    return RobustStabilityResult(
        probability=float(p_hat),
        ci_low=ci_low,
        ci_high=ci_high,
        n_theta_samples=n_theta_samples,
    )
