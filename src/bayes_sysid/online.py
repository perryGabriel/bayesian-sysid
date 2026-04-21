from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike


def _as_1d_float(x: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional after reshaping.")
    return arr


def recursive_posterior_update(
    mu: ArrayLike,
    Sigma: ArrayLike,
    phi_t: ArrayLike,
    y_t: float,
    sigma2: float,
    forgetting_factor: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Perform one-step recursive Bayesian posterior update for linear-Gaussian regression.

    Model:
        y_t = phi_t^T theta + e_t,  e_t ~ N(0, sigma2)
        theta | D_{t-1} ~ N(mu, Sigma)

    Optional forgetting applies covariance inflation before seeing y_t:
        Sigma_prior = Sigma / forgetting_factor

    Parameters
    ----------
    mu, Sigma : array-like
        Prior/posterior mean and covariance from the previous step.
    phi_t : array-like
        Regressor for the current observation.
    y_t : float
        Current observation.
    sigma2 : float
        Known observation-noise variance.
    forgetting_factor : float
        In (0, 1]. Value < 1 discounts past data (nonstationary tracking).

    Returns
    -------
    mu_new, Sigma_new, pred_mean, pred_var
        Updated posterior parameters and one-step predictive moments.
    """
    if sigma2 <= 0:
        raise ValueError("sigma2 must be positive.")
    if not (0.0 < forgetting_factor <= 1.0):
        raise ValueError("forgetting_factor must be in (0, 1].")

    mu = _as_1d_float(mu, "mu")
    Sigma = np.asarray(Sigma, dtype=float)
    phi_t = _as_1d_float(phi_t, "phi_t")

    p = mu.shape[0]
    if phi_t.shape[0] != p:
        raise ValueError("phi_t has incompatible dimension.")
    if Sigma.shape != (p, p):
        raise ValueError("Sigma has incompatible shape.")

    Sigma_prior = Sigma / forgetting_factor
    pred_mean = float(phi_t @ mu)
    pred_var = float(phi_t @ Sigma_prior @ phi_t + sigma2)

    k_t = (Sigma_prior @ phi_t) / pred_var
    innovation = float(y_t - pred_mean)
    mu_new = mu + k_t * innovation

    # Joseph-form covariance update for numerical stability.
    I = np.eye(p)
    outer = np.outer(k_t, phi_t)
    Sigma_new = (I - outer) @ Sigma_prior @ (I - outer).T + sigma2 * np.outer(k_t, k_t)
    Sigma_new = 0.5 * (Sigma_new + Sigma_new.T)

    return mu_new, Sigma_new, pred_mean, pred_var


@dataclass
class OnlineSummarySnapshot:
    step_index: int
    num_updates: int
    posterior_mean: np.ndarray
    posterior_covariance: np.ndarray
    covariance_trace: float
    forgetting_factor: float


class OnlineBayesianARX:
    """Online Bayesian ARX estimator with known observation noise.

    The class consumes one sample at a time via ``update(y_t, u_t)``. It starts
    updating once enough history is available to build an ARX regressor.
    """

    def __init__(
        self,
        na: int,
        nb: int,
        sigma2: float,
        mu0: ArrayLike | None = None,
        Sigma0: ArrayLike | None = None,
        forgetting_factor: float = 1.0,
        snapshot_stride: int = 50,
    ) -> None:
        if na < 0 or nb < 0 or (na == 0 and nb == 0):
            raise ValueError("At least one of na or nb must be positive.")
        if sigma2 <= 0:
            raise ValueError("sigma2 must be positive.")
        if not (0.0 < forgetting_factor <= 1.0):
            raise ValueError("forgetting_factor must be in (0, 1].")
        if snapshot_stride <= 0:
            raise ValueError("snapshot_stride must be positive.")

        self.na = int(na)
        self.nb = int(nb)
        self.p = self.na + self.nb
        self.max_lag = max(self.na, self.nb)
        self.sigma2 = float(sigma2)
        self.forgetting_factor = float(forgetting_factor)
        self.snapshot_stride = int(snapshot_stride)

        self.mu = np.zeros(self.p, dtype=float) if mu0 is None else np.asarray(mu0, dtype=float).reshape(self.p)
        self.Sigma = np.eye(self.p, dtype=float) * 10.0 if Sigma0 is None else np.asarray(Sigma0, dtype=float)
        if self.Sigma.shape != (self.p, self.p):
            raise ValueError("Sigma0 has the wrong shape.")

        self._y_hist: list[float] = []
        self._u_hist: list[float] = []
        self._num_steps = 0
        self.num_updates = 0
        self.snapshots: list[OnlineSummarySnapshot] = []

    def _make_regressor(self) -> np.ndarray:
        phi: list[float] = []
        for i in range(1, self.na + 1):
            phi.append(self._y_hist[-i])
        for j in range(1, self.nb + 1):
            phi.append(self._u_hist[-j])
        return np.asarray(phi, dtype=float)

    def update(self, y_t: float, u_t: float) -> dict[str, Any]:
        """Ingest one sample and apply a Bayesian update when history is sufficient."""
        result: dict[str, Any] = {
            "updated": False,
            "step_index": self._num_steps,
            "num_updates": self.num_updates,
        }

        if len(self._y_hist) >= self.na and len(self._u_hist) >= self.nb:
            phi_t = self._make_regressor()
            self.mu, self.Sigma, pred_mean, pred_var = recursive_posterior_update(
                self.mu,
                self.Sigma,
                phi_t,
                float(y_t),
                self.sigma2,
                forgetting_factor=self.forgetting_factor,
            )
            self.num_updates += 1
            result.update(
                {
                    "updated": True,
                    "phi_t": phi_t,
                    "pred_mean": pred_mean,
                    "pred_var": pred_var,
                    "innovation": float(y_t - pred_mean),
                    "num_updates": self.num_updates,
                }
            )

            if self.num_updates % self.snapshot_stride == 0:
                snap = OnlineSummarySnapshot(
                    step_index=self._num_steps,
                    num_updates=self.num_updates,
                    posterior_mean=self.mu.copy(),
                    posterior_covariance=self.Sigma.copy(),
                    covariance_trace=float(np.trace(self.Sigma)),
                    forgetting_factor=self.forgetting_factor,
                )
                self.snapshots.append(snap)
                result["snapshot"] = snap

        self._y_hist.append(float(y_t))
        self._u_hist.append(float(u_t))
        self._num_steps += 1
        return result
