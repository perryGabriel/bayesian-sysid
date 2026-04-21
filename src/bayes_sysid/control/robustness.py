from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .lft import StructuredDelta


@dataclass
class RobustnessReport:
    """Sampled robustness diagnostics for simple structured uncertainty.

    Limitations
    -----------
    These diagnostics are empirical surrogates based on sampled uncertainty and
    simple channelization assumptions. They are not exact structured singular
    value (mu) computations.
    """

    worst_case_sampled_return_difference: float
    critical_point_encroachment_frequency: float
    block_sensitivity_ranking: list[tuple[int, float]]


def robustness_report_from_structured_samples(
    m22_by_channel: ArrayLike,
    structured_delta: StructuredDelta,
    n_samples: int = 256,
    random_state: int | None = None,
    critical_radius: float = 0.1,
) -> RobustnessReport:
    """Build sampled robustness report for block-diagonal surrogate structure.

    Parameters
    ----------
    m22_by_channel:
        Frequency response channels with shape ``(n_channels, n_freq)``.
    structured_delta:
        Structured uncertainty model.
    n_samples:
        Number of uncertainty realizations used for the empirical diagnostics.
    critical_radius:
        Encroachment threshold for ``|1 - L_delta(jw)|`` where
        ``L_delta = sum_i M22_i Delta_i``.
    """
    m22_channels = np.asarray(m22_by_channel, dtype=complex)
    if m22_channels.ndim != 2:
        raise ValueError("m22_by_channel must be 2D with shape (n_channels, n_freq).")

    n_channels, n_freq = m22_channels.shape
    bounds = structured_delta.channel_bounds()
    if n_channels != len(bounds):
        raise ValueError("StructuredDelta channel count does not match m22_by_channel.")

    rng = np.random.default_rng(random_state)

    sampled_return_differences: list[np.ndarray] = []
    encroachment_points = 0
    total_points = n_samples * n_freq

    for _ in range(n_samples):
        delta = structured_delta.sample(rng=rng, n_freq=n_freq)
        loop = np.sum(m22_channels * delta, axis=0)
        return_diff = np.abs(1.0 - loop)
        sampled_return_differences.append(return_diff)
        encroachment_points += int(np.count_nonzero(return_diff <= critical_radius))

    return_diff_stack = np.asarray(sampled_return_differences)
    worst_case = float(np.min(return_diff_stack))
    encroach_freq = float(encroachment_points / total_points)

    sensitivity_scores = np.max(np.abs(m22_channels) * bounds[:, None], axis=1)
    ranking = sorted(
        [(int(i), float(score)) for i, score in enumerate(sensitivity_scores)],
        key=lambda x: x[1],
        reverse=True,
    )

    return RobustnessReport(
        worst_case_sampled_return_difference=worst_case,
        critical_point_encroachment_frequency=encroach_freq,
        block_sensitivity_ranking=ranking,
    )
