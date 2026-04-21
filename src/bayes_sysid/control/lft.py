from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from bayes_sysid.analysis.frequency_response import arx_frequency_response


@dataclass
class ScalarUncertaintyBlock:
    """Scalar uncertainty block used in sampled robust-stability surrogates.

    Notes
    -----
    This class supports simple scalar-real/scalar-complex uncertainty models only.
    It is *not* a full structured-singular-value (mu) uncertainty object.

    Parameters
    ----------
    bound:
        Absolute bound for ``|Delta(jw)|``.
    kind:
        ``"real"`` or ``"complex"``.
    frequency_grid:
        Optional frequency grid used for dynamic samples.
    dynamic:
        If True, sampled realizations are frequency-dependent.
    """

    bound: float
    kind: str = "real"
    frequency_grid: np.ndarray | None = None
    dynamic: bool = False

    def __post_init__(self) -> None:
        self.bound = float(self.bound)
        if self.bound <= 0:
            raise ValueError("bound must be positive.")
        if self.kind not in {"real", "complex"}:
            raise ValueError("kind must be 'real' or 'complex'.")
        if self.frequency_grid is not None:
            self.frequency_grid = np.asarray(self.frequency_grid, dtype=float).reshape(-1)
            if len(self.frequency_grid) == 0:
                raise ValueError("frequency_grid must be non-empty when provided.")

    def sample(self, rng: np.random.Generator, n_freq: int | None = None) -> np.ndarray:
        """Sample one realization over frequency."""
        n = 1 if n_freq is None else int(n_freq)
        if self.dynamic:
            if self.frequency_grid is None and n_freq is None:
                raise ValueError("n_freq is required for dynamic uncertainty without frequency_grid.")
            n = len(self.frequency_grid) if self.frequency_grid is not None else int(n_freq)
            if self.kind == "real":
                return rng.uniform(-self.bound, self.bound, size=n).astype(complex)
            mag = self.bound * rng.random(n)
            phase = rng.uniform(-np.pi, np.pi, size=n)
            return mag * np.exp(1j * phase)

        if self.kind == "real":
            scalar = rng.uniform(-self.bound, self.bound)
            return np.full(n, scalar, dtype=complex)

        mag = self.bound * rng.random()
        phase = rng.uniform(-np.pi, np.pi)
        scalar = mag * np.exp(1j * phase)
        return np.full(n, scalar, dtype=complex)


@dataclass
class RepeatedScalarUncertaintyBlock:
    """Repeated scalar uncertainty ``delta * I_r`` surrogate representation.

    Notes
    -----
    This is a lightweight repeated-scalar model for sampled analyses. It does not
    construct an exact LFT object for mu-computation.
    """

    base_block: ScalarUncertaintyBlock
    repetitions: int

    def __post_init__(self) -> None:
        self.repetitions = int(self.repetitions)
        if self.repetitions <= 0:
            raise ValueError("repetitions must be positive.")

    def sample(self, rng: np.random.Generator, n_freq: int | None = None) -> np.ndarray:
        """Sample repeated diagonal entries with shape ``(repetitions, n_freq)``."""
        sample = self.base_block.sample(rng=rng, n_freq=n_freq)
        return np.tile(sample.reshape(1, -1), (self.repetitions, 1))


@dataclass
class StructuredDelta:
    """Simple block-diagonal structured uncertainty surrogate.

    The resulting structure is represented via diagonal channels only:

    ``Delta = diag(Delta_1, Delta_2, ..., Delta_k)``

    where each ``Delta_i`` is either a scalar block or repeated-scalar block.

    Limitations
    -----------
    This representation is intentionally simple for conservative sampled/upper-
    bound surrogates. It is not equivalent to exact structured singular value
    analysis and should not be interpreted as exact mu.
    """

    blocks: tuple[ScalarUncertaintyBlock | RepeatedScalarUncertaintyBlock, ...]

    def __post_init__(self) -> None:
        if len(self.blocks) == 0:
            raise ValueError("blocks must be non-empty.")

    def sample(self, rng: np.random.Generator, n_freq: int) -> np.ndarray:
        """Sample all block diagonal channels with shape ``(n_channels, n_freq)``."""
        channels: list[np.ndarray] = []
        for block in self.blocks:
            sample = block.sample(rng=rng, n_freq=n_freq)
            sample_2d = np.asarray(sample, dtype=complex)
            if sample_2d.ndim == 1:
                sample_2d = sample_2d.reshape(1, -1)
            channels.extend(list(sample_2d))
        return np.asarray(channels, dtype=complex)

    def channel_bounds(self) -> np.ndarray:
        """Return per-channel absolute bounds consistent with :meth:`sample`."""
        bounds: list[float] = []
        for block in self.blocks:
            if isinstance(block, RepeatedScalarUncertaintyBlock):
                bounds.extend([block.base_block.bound] * block.repetitions)
            else:
                bounds.append(block.bound)
        return np.asarray(bounds, dtype=float)


class DeltaBlock(ScalarUncertaintyBlock):
    """Backward-compatible scalar uncertainty block.

    Notes
    -----
    Preserved API for existing callers. Prefer :class:`ScalarUncertaintyBlock` for
    new code.
    """

    def __init__(
        self,
        gain_bound: float,
        frequency_grid: np.ndarray | None = None,
        dynamic: bool = False,
    ) -> None:
        self.gain_bound = float(gain_bound)
        super().__init__(
            bound=self.gain_bound,
            kind="complex" if dynamic else "real",
            frequency_grid=frequency_grid,
            dynamic=dynamic,
        )


@dataclass
class RobustStabilityResult:
    probability: float
    ci_low: float
    ci_high: float
    n_theta_samples: int


@dataclass
class StructuredSurrogateResult:
    """Upper-bound style surrogate for structured uncertainty robustness.

    Limitations
    -----------
    ``max_scaled_gain < 1`` is a conservative sufficient condition based on
    small-gain style channel bounds. It is an approximation and not exact mu.
    """

    max_scaled_gain: float
    approx_stable: bool
    scaled_gain_by_frequency: np.ndarray


def construct_nominal_interconnection_m_blocks(
    theta: ArrayLike,
    na: int,
    nb: int,
    controller_gain: float,
    w: ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct nominal interconnection blocks ``M11...M22``.

    Signal convention
    -----------------
    The closed-loop interconnection is

    ``u = K (r - y)``, ``y = G (u + w)``, ``z = y``, ``w = Delta z``.

    with stacked signals ``[z; y] = M [r; w]`` and partition

    ``M = [[M11, M12], [M21, M22]]``.
    """
    G = arx_frequency_response(theta=theta, na=na, nb=nb, w=w)
    K = complex(controller_gain)
    den = 1.0 + K * G
    M11 = (K * G) / den
    M12 = G / den
    M21 = M11.copy()
    M22 = M12.copy()
    return M11, M12, M21, M22


def build_nominal_interconnection(
    theta: ArrayLike,
    na: int,
    nb: int,
    controller_gain: float,
    w: ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Backward-compatible alias of :func:`construct_nominal_interconnection_m_blocks`."""
    return construct_nominal_interconnection_m_blocks(theta, na, nb, controller_gain, w)


def lower_lft_siso(
    M11: ArrayLike,
    M12: ArrayLike,
    M21: ArrayLike,
    M22: ArrayLike,
    delta: ArrayLike,
    regularization: float = 1e-12,
) -> np.ndarray:
    """Evaluate lower LFT ``F_l(M, Delta)`` for scalar SISO blocks."""
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
    """Evaluate upper LFT ``F_u(M, Delta)`` for scalar SISO blocks."""
    m11 = np.asarray(M11, dtype=complex)
    m12 = np.asarray(M12, dtype=complex)
    m21 = np.asarray(M21, dtype=complex)
    m22 = np.asarray(M22, dtype=complex)
    d = np.asarray(delta, dtype=complex)
    inv_term = 1.0 / (1.0 - m11 * d + regularization)
    return m22 + m21 * d * inv_term * m12


def structured_small_gain_surrogate(
    m22_by_channel: ArrayLike,
    structured_delta: StructuredDelta,
    margin_tol: float = 1e-3,
) -> StructuredSurrogateResult:
    """Compute conservative upper-bound surrogate metric for structured uncertainty.

    Parameters
    ----------
    m22_by_channel:
        Channelized uncertainty path with shape ``(n_channels, n_freq)``.
    structured_delta:
        Simple block-diagonal structured uncertainty representation.

    Notes
    -----
    Uses ``sum_i |M22_i(jw)| * bound_i`` as a conservative small-gain-style bound.
    This is an approximation for structured cases and is not exact mu-analysis.
    """
    m22_channels = np.asarray(m22_by_channel, dtype=complex)
    if m22_channels.ndim != 2:
        raise ValueError("m22_by_channel must be 2D with shape (n_channels, n_freq).")
    bounds = structured_delta.channel_bounds()
    if len(bounds) != m22_channels.shape[0]:
        raise ValueError("Number of channel bounds must match m22_by_channel channels.")

    scaled_gain = np.sum(np.abs(m22_channels) * bounds[:, None], axis=0)
    max_scaled = float(np.max(scaled_gain))
    return StructuredSurrogateResult(
        max_scaled_gain=max_scaled,
        approx_stable=bool(max_scaled < (1.0 - margin_tol)),
        scaled_gain_by_frequency=scaled_gain,
    )


def _wilson_interval(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n <= 0:
        raise ValueError("n must be positive.")
    z = 1.959963984540054
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
    delta_block: ScalarUncertaintyBlock,
    w: ArrayLike,
    n_theta_samples: int = 200,
    n_delta_samples: int = 200,
    random_state: int | None = None,
    margin_tol: float = 1e-3,
    alpha: float = 0.05,
) -> RobustStabilityResult:
    """Sampling-based sufficient robustness test under posterior uncertainty.

    A sampled ``theta`` is declared robustly stable when all sampled realizations
    satisfy ``max_w |M22(theta, w) Delta(w)| < 1 - margin_tol``.

    This is a sufficient, sampling-based test and does not compute exact mu.
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
        _, _, _, m22 = construct_nominal_interconnection_m_blocks(
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
