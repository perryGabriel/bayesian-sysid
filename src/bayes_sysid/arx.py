from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm


@dataclass
class ARXRegressionData:
    Phi: np.ndarray
    y_target: np.ndarray
    max_lag: int


def _as_1d_float(x: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional after reshaping.")
    return arr


def build_arx_regression(y: ArrayLike, u: ArrayLike, na: int, nb: int) -> ARXRegressionData:
    """Construct the regression matrix for a SISO ARX model.

    Model:
        y_t = sum_i a_i y_{t-i} + sum_j b_j u_{t-j} + e_t

    Regressor convention:
        phi_t = [y_{t-1}, ..., y_{t-na}, u_{t-1}, ..., u_{t-nb}]

    Parameters
    ----------
    y, u : array-like
        Output and input sequences of equal length.
    na, nb : int
        Output and input lags.
    """
    if na < 0 or nb < 0 or (na == 0 and nb == 0):
        raise ValueError("At least one of na or nb must be positive.")

    y = _as_1d_float(y, "y")
    u = _as_1d_float(u, "u")
    if len(y) != len(u):
        raise ValueError("y and u must have the same length.")

    max_lag = max(na, nb)
    if len(y) <= max_lag:
        raise ValueError("Time series is too short for the requested lags.")

    rows = []
    targets = []
    for t in range(max_lag, len(y)):
        phi = []
        for i in range(1, na + 1):
            phi.append(y[t - i])
        for j in range(1, nb + 1):
            phi.append(u[t - j])
        rows.append(phi)
        targets.append(y[t])

    Phi = np.asarray(rows, dtype=float)
    y_target = np.asarray(targets, dtype=float)
    return ARXRegressionData(Phi=Phi, y_target=y_target, max_lag=max_lag)


class LeastSquaresARX:
    """Classical least-squares ARX baseline."""

    def __init__(self, na: int, nb: int) -> None:
        self.na = int(na)
        self.nb = int(nb)
        self.theta_hat: Optional[np.ndarray] = None
        self.max_lag = max(self.na, self.nb)

    def fit(self, y: ArrayLike, u: ArrayLike) -> "LeastSquaresARX":
        reg = build_arx_regression(y, u, self.na, self.nb)
        theta_hat, *_ = np.linalg.lstsq(reg.Phi, reg.y_target, rcond=None)
        self.theta_hat = theta_hat
        return self

    def _require_fit(self) -> None:
        if self.theta_hat is None:
            raise RuntimeError("Model has not been fit yet.")

    def make_regressor(self, y_hist: ArrayLike, u_hist: ArrayLike) -> np.ndarray:
        y_hist = _as_1d_float(y_hist, "y_hist")
        u_hist = _as_1d_float(u_hist, "u_hist")
        if len(y_hist) < self.na or len(u_hist) < self.nb:
            raise ValueError("Not enough history to build the regressor.")
        phi = []
        for i in range(1, self.na + 1):
            phi.append(y_hist[-i])
        for j in range(1, self.nb + 1):
            phi.append(u_hist[-j])
        return np.asarray(phi, dtype=float)

    def predict_next(self, y_hist: ArrayLike, u_hist: ArrayLike) -> float:
        self._require_fit()
        phi = self.make_regressor(y_hist, u_hist)
        return float(phi @ self.theta_hat)

    def simulate_one_step_rollout(self, y_init: ArrayLike, u: ArrayLike) -> np.ndarray:
        """Deterministic rollout using the plug-in parameter estimate."""
        self._require_fit()
        y_init = _as_1d_float(y_init, "y_init")
        u = _as_1d_float(u, "u")
        if len(y_init) < self.na:
            raise ValueError("y_init must contain at least na past outputs.")
        if len(u) < self.nb:
            raise ValueError("u must contain at least nb values.")

        y_hist = list(y_init.copy())
        preds = []
        for k in range(len(u) - self.nb + 1):
            u_hist = u[: self.nb + k]
            y_arr = np.asarray(y_hist, dtype=float)
            phi = self.make_regressor(y_arr, u_hist)
            y_next = float(phi @ self.theta_hat)
            y_hist.append(y_next)
            preds.append(y_next)
        return np.asarray(preds)


class BayesianARX:
    """Bayesian ARX with Gaussian prior and known observation variance.

    Prior:
        theta ~ N(mu0, Sigma0)

    Likelihood:
        y | theta ~ N(Phi theta, sigma2 I)

    Posterior:
        theta | D ~ N(muN, SigmaN)
    """

    def __init__(
        self,
        na: int,
        nb: int,
        sigma2: float,
        mu0: Optional[ArrayLike] = None,
        Sigma0: Optional[ArrayLike] = None,
    ) -> None:
        if sigma2 <= 0:
            raise ValueError("sigma2 must be positive.")
        self.na = int(na)
        self.nb = int(nb)
        self.p = self.na + self.nb
        self.sigma2 = float(sigma2)
        self.max_lag = max(self.na, self.nb)

        self.mu0 = np.zeros(self.p) if mu0 is None else np.asarray(mu0, dtype=float).reshape(self.p)
        self.Sigma0 = np.eye(self.p) * 10.0 if Sigma0 is None else np.asarray(Sigma0, dtype=float)
        if self.Sigma0.shape != (self.p, self.p):
            raise ValueError("Sigma0 has the wrong shape.")

        self.muN: Optional[np.ndarray] = None
        self.SigmaN: Optional[np.ndarray] = None
        self.Phi: Optional[np.ndarray] = None
        self.y_target: Optional[np.ndarray] = None

    def fit(self, y: ArrayLike, u: ArrayLike) -> "BayesianARX":
        reg = build_arx_regression(y, u, self.na, self.nb)
        Phi = reg.Phi
        y_target = reg.y_target

        Sigma0_inv = np.linalg.inv(self.Sigma0)
        SigmaN_inv = Sigma0_inv + (Phi.T @ Phi) / self.sigma2
        SigmaN = np.linalg.inv(SigmaN_inv)
        muN = SigmaN @ (Sigma0_inv @ self.mu0 + (Phi.T @ y_target) / self.sigma2)

        self.Phi = Phi
        self.y_target = y_target
        self.muN = muN
        self.SigmaN = SigmaN
        return self

    def _require_fit(self) -> None:
        if self.muN is None or self.SigmaN is None:
            raise RuntimeError("Model has not been fit yet.")

    def make_regressor(self, y_hist: ArrayLike, u_hist: ArrayLike) -> np.ndarray:
        y_hist = _as_1d_float(y_hist, "y_hist")
        u_hist = _as_1d_float(u_hist, "u_hist")
        if len(y_hist) < self.na or len(u_hist) < self.nb:
            raise ValueError("Not enough history to build the regressor.")

        phi = []
        for i in range(1, self.na + 1):
            phi.append(y_hist[-i])
        for j in range(1, self.nb + 1):
            phi.append(u_hist[-j])
        return np.asarray(phi, dtype=float)

    def posterior_mean_prediction(self, y_hist: ArrayLike, u_hist: ArrayLike) -> float:
        self._require_fit()
        phi = self.make_regressor(y_hist, u_hist)
        return float(phi @ self.muN)

    def predict_next_distribution(self, y_hist: ArrayLike, u_hist: ArrayLike) -> tuple[float, float]:
        """Return posterior predictive mean and variance for next output."""
        self._require_fit()
        phi = self.make_regressor(y_hist, u_hist)
        mean = float(phi @ self.muN)
        var = float(phi @ self.SigmaN @ phi + self.sigma2)
        return mean, var

    def predictive_density_grid(
        self,
        y_hist: ArrayLike,
        u_hist: ArrayLike,
        grid: ArrayLike,
    ) -> np.ndarray:
        mean, var = self.predict_next_distribution(y_hist, u_hist)
        x = np.asarray(grid, dtype=float)
        return norm.pdf(x, loc=mean, scale=np.sqrt(var))

    def sample_parameters(self, n_samples: int, random_state: Optional[int] = None) -> np.ndarray:
        self._require_fit()
        rng = np.random.default_rng(random_state)
        return rng.multivariate_normal(self.muN, self.SigmaN, size=n_samples)

    def sample_predictive_next(
        self,
        y_hist: ArrayLike,
        u_hist: ArrayLike,
        n_samples: int,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        self._require_fit()
        rng = np.random.default_rng(random_state)
        phi = self.make_regressor(y_hist, u_hist)
        theta_samps = self.sample_parameters(n_samples, random_state=random_state)
        means = theta_samps @ phi
        return means + rng.normal(0.0, np.sqrt(self.sigma2), size=n_samples)

    def rollout_posterior_samples(
        self,
        y_init: ArrayLike,
        u_future: ArrayLike,
        n_parameter_samples: int = 200,
        include_process_noise: bool = True,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        """Sample future trajectories by drawing theta from the posterior.

        Returns an array of shape (n_parameter_samples, horizon).
        """
        self._require_fit()
        rng = np.random.default_rng(random_state)
        y_init = _as_1d_float(y_init, "y_init")
        u_future = _as_1d_float(u_future, "u_future")

        if len(y_init) < self.na:
            raise ValueError("y_init must contain at least na values.")
        if len(u_future) < self.nb:
            raise ValueError("u_future must contain at least nb values.")

        horizon = len(u_future) - self.nb + 1
        thetas = self.sample_parameters(n_parameter_samples, random_state=random_state)
        paths = np.zeros((n_parameter_samples, horizon), dtype=float)

        for s, theta in enumerate(thetas):
            y_hist = list(y_init.copy())
            for k in range(horizon):
                phi = []
                for i in range(1, self.na + 1):
                    phi.append(y_hist[-i])
                for j in range(1, self.nb + 1):
                    phi.append(u_future[k + self.nb - j])
                phi = np.asarray(phi, dtype=float)
                y_next = float(phi @ theta)
                if include_process_noise:
                    y_next += rng.normal(0.0, np.sqrt(self.sigma2))
                y_hist.append(y_next)
                paths[s, k] = y_next
        return paths
