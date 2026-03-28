from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from bayes_sysid.analysis.frequency_response import arx_frequency_response
from bayes_sysid.models import BayesianARX


@dataclass
class MarginReport:
    gain_margin: float
    phase_margin_deg: float
    gain_cross_freq: float | None
    phase_cross_freq: float | None


def _interp_x_for_y_crossing(x: np.ndarray, y: np.ndarray, target: float) -> float | None:
    for i in range(len(x) - 1):
        y0, y1 = y[i], y[i + 1]
        if (y0 - target) == 0:
            return float(x[i])
        if (y0 - target) * (y1 - target) < 0:
            a = (target - y0) / (y1 - y0)
            return float(x[i] + a * (x[i + 1] - x[i]))
    return None


def classical_margins_from_open_loop(Lw: ArrayLike, w: ArrayLike) -> MarginReport:
    """Approximate gain/phase margins from sampled open-loop response."""
    Lw = np.asarray(Lw, dtype=complex).reshape(-1)
    w = np.asarray(w, dtype=float).reshape(-1)
    if len(Lw) != len(w):
        raise ValueError("Lw and w must have same length.")

    mag = np.abs(Lw)
    phase_deg = np.unwrap(np.angle(Lw)) * 180.0 / np.pi

    w_gc = _interp_x_for_y_crossing(w, mag, 1.0)
    phase_margin = np.nan
    if w_gc is not None:
        phase_at_gc = np.interp(w_gc, w, phase_deg)
        phase_margin = 180.0 + phase_at_gc

    w_pc = _interp_x_for_y_crossing(w, phase_deg, -180.0)
    gain_margin = np.inf
    if w_pc is not None:
        mag_at_pc = np.interp(w_pc, w, mag)
        if mag_at_pc > 0:
            gain_margin = 1.0 / mag_at_pc

    return MarginReport(
        gain_margin=float(gain_margin),
        phase_margin_deg=float(phase_margin),
        gain_cross_freq=w_gc,
        phase_cross_freq=w_pc,
    )


def empirical_margin_report(
    model: BayesianARX,
    w: ArrayLike,
    n_samples: int = 200,
    random_state: int | None = None,
) -> dict:
    """Empirical robust margin summary over posterior samples."""
    theta_samples = model.sample_parameters(n_samples=n_samples, random_state=random_state)
    reports = []
    for theta in theta_samples:
        L = arx_frequency_response(theta, model.na, model.nb, w)
        reports.append(classical_margins_from_open_loop(L, w))

    gm = np.asarray([r.gain_margin for r in reports], dtype=float)
    pm = np.asarray([r.phase_margin_deg for r in reports], dtype=float)
    gm_finite = gm[np.isfinite(gm)]

    return {
        "p_gain_margin_gt_1": float(np.mean(gm > 1.0)),
        "p_phase_margin_gt_0": float(np.mean(pm > 0.0)),
        "median_gain_margin": float(np.median(gm_finite)) if len(gm_finite) else np.inf,
        "median_phase_margin_deg": float(np.nanmedian(pm)),
    }
