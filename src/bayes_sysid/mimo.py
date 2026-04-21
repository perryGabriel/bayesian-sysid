from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike


@dataclass
class MIMORegressionData:
    Phi: np.ndarray
    Y_target: np.ndarray
    max_lag: int


def _as_2d_float(x: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array.")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must have non-zero rows and columns.")
    return arr


def _as_vector_sigma2(sigma2: ArrayLike, n_outputs: int) -> np.ndarray:
    sigma2_arr = np.asarray(sigma2, dtype=float)
    if sigma2_arr.ndim == 0:
        sigma2_vec = np.full(n_outputs, float(sigma2_arr), dtype=float)
    else:
        sigma2_vec = sigma2_arr.reshape(-1)
        if sigma2_vec.size != n_outputs:
            raise ValueError("sigma2 must be scalar or length n_outputs.")
    if np.any(sigma2_vec <= 0):
        raise ValueError("sigma2 entries must be positive.")
    return sigma2_vec


def build_mimo_regression(y: ArrayLike, u: ArrayLike, na: int, nb: int) -> MIMORegressionData:
    """Build block regressor matrix for MIMO ARX.

    Model:
        y_t = sum_{i=1..na} A_i y_{t-i} + sum_{j=1..nb} B_j u_{t-j} + e_t

    Regressor rows:
        phi_t = [y_{t-1}, ..., y_{t-na}, u_{t-1}, ..., u_{t-nb}]  # flattened blocks
    """
    if na < 0 or nb < 0 or (na == 0 and nb == 0):
        raise ValueError("At least one of na or nb must be positive.")

    y_arr = _as_2d_float(y, "y")
    u_arr = _as_2d_float(u, "u")
    if y_arr.shape[0] != u_arr.shape[0]:
        raise ValueError("y and u must have the same number of samples.")

    n_samples, n_outputs = y_arr.shape
    _, n_inputs = u_arr.shape
    max_lag = max(na, nb)
    if n_samples <= max_lag:
        raise ValueError("Time series is too short for the requested lags.")

    rows = []
    targets = []
    for t_idx in range(max_lag, n_samples):
        blocks = []
        for i in range(1, na + 1):
            blocks.append(y_arr[t_idx - i])
        for j in range(1, nb + 1):
            blocks.append(u_arr[t_idx - j])
        phi_t = np.concatenate(blocks) if blocks else np.empty((0,), dtype=float)
        rows.append(phi_t)
        targets.append(y_arr[t_idx])

    Phi = np.asarray(rows, dtype=float).reshape(n_samples - max_lag, na * n_outputs + nb * n_inputs)
    Y_target = np.asarray(targets, dtype=float).reshape(n_samples - max_lag, n_outputs)
    return MIMORegressionData(Phi=Phi, Y_target=Y_target, max_lag=max_lag)


class BayesianMIMOARX:
    """Bayesian MIMO ARX with independent-output posterior baseline.

    Supports an optional coupled-output predictive noise extension via empirical
    residual covariance when ``coupled_output_covariance=True``.
    """

    def __init__(
        self,
        na: int,
        nb: int,
        sigma2: ArrayLike,
        mu0: Optional[ArrayLike] = None,
        Sigma0: Optional[ArrayLike] = None,
        coupled_output_covariance: bool = False,
    ) -> None:
        self.na = int(na)
        self.nb = int(nb)
        if self.na < 0 or self.nb < 0 or (self.na == 0 and self.nb == 0):
            raise ValueError("At least one of na or nb must be positive.")
        self.max_lag = max(self.na, self.nb)
        self.coupled_output_covariance = bool(coupled_output_covariance)

        self._sigma2_input = np.asarray(sigma2, dtype=float)
        self.mu0_input = None if mu0 is None else np.asarray(mu0, dtype=float)
        self.Sigma0_input = None if Sigma0 is None else np.asarray(Sigma0, dtype=float)

        self.n_outputs: Optional[int] = None
        self.n_inputs: Optional[int] = None
        self.p: Optional[int] = None

        self.muN: Optional[np.ndarray] = None  # (n_outputs, p)
        self.SigmaN: Optional[np.ndarray] = None  # (n_outputs, p, p)
        self.sigma2_vec: Optional[np.ndarray] = None
        self.output_noise_covariance: Optional[np.ndarray] = None

    def _require_fit(self) -> None:
        if self.muN is None or self.SigmaN is None or self.sigma2_vec is None:
            raise RuntimeError("Model has not been fit yet.")

    def _make_regressor(self, y_hist: np.ndarray, u_hist: np.ndarray) -> np.ndarray:
        if y_hist.shape[0] < self.na or u_hist.shape[0] < self.nb:
            raise ValueError("Not enough history to build regressor for requested lags.")
        blocks = []
        for i in range(1, self.na + 1):
            blocks.append(y_hist[-i])
        for j in range(1, self.nb + 1):
            blocks.append(u_hist[-j])
        return np.concatenate(blocks) if blocks else np.empty((0,), dtype=float)

    def fit(self, y: ArrayLike, u: ArrayLike) -> "BayesianMIMOARX":
        y_arr = _as_2d_float(y, "y")
        u_arr = _as_2d_float(u, "u")
        if y_arr.shape[0] != u_arr.shape[0]:
            raise ValueError("y and u must have the same number of samples.")

        reg = build_mimo_regression(y_arr, u_arr, self.na, self.nb)
        Phi = reg.Phi
        Y_target = reg.Y_target

        n_outputs = Y_target.shape[1]
        n_inputs = u_arr.shape[1]
        p = Phi.shape[1]
        sigma2_vec = _as_vector_sigma2(self._sigma2_input, n_outputs)

        mu0 = np.zeros(p, dtype=float) if self.mu0_input is None else self.mu0_input.reshape(p)
        Sigma0 = np.eye(p, dtype=float) * 10.0 if self.Sigma0_input is None else self.Sigma0_input
        if Sigma0.shape != (p, p):
            raise ValueError("Sigma0 has wrong shape; expected (n_regressors, n_regressors).")

        Sigma0_inv = np.linalg.inv(Sigma0)

        muN = np.zeros((n_outputs, p), dtype=float)
        SigmaN = np.zeros((n_outputs, p, p), dtype=float)
        for out_idx in range(n_outputs):
            s2 = sigma2_vec[out_idx]
            SigmaN_inv = Sigma0_inv + (Phi.T @ Phi) / s2
            SigmaN_out = np.linalg.inv(SigmaN_inv)
            muN_out = SigmaN_out @ (Sigma0_inv @ mu0 + (Phi.T @ Y_target[:, out_idx]) / s2)
            muN[out_idx] = muN_out
            SigmaN[out_idx] = SigmaN_out

        residuals = Y_target - Phi @ muN.T
        if self.coupled_output_covariance:
            dof = max(residuals.shape[0] - 1, 1)
            output_cov = (residuals.T @ residuals) / dof
            output_cov = output_cov + 1e-12 * np.eye(n_outputs)
        else:
            output_cov = np.diag(sigma2_vec)

        self.n_outputs = n_outputs
        self.n_inputs = n_inputs
        self.p = p
        self.sigma2_vec = sigma2_vec
        self.muN = muN
        self.SigmaN = SigmaN
        self.output_noise_covariance = output_cov
        return self

    def sample_parameters(self, n_samples: int, random_state: Optional[int] = None) -> dict[str, np.ndarray]:
        self._require_fit()
        rng = np.random.default_rng(random_state)

        theta = np.empty((n_samples, self.n_outputs, self.p), dtype=float)
        for out_idx in range(self.n_outputs):
            theta[:, out_idx, :] = rng.multivariate_normal(
                self.muN[out_idx], self.SigmaN[out_idx], size=n_samples
            )

        split_y = self.na * self.n_outputs
        a_flat = theta[:, :, :split_y]
        b_flat = theta[:, :, split_y:]

        a_lags = np.transpose(a_flat.reshape(n_samples, self.n_outputs, self.na, self.n_outputs), (0, 2, 1, 3))
        b_lags = np.transpose(b_flat.reshape(n_samples, self.n_outputs, self.nb, self.n_inputs), (0, 2, 1, 3))

        return {"theta": theta, "a_lags": a_lags, "b_lags": b_lags}

    def predict_next_distribution(self, y_hist: ArrayLike, u_hist: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        self._require_fit()
        y_hist_arr = _as_2d_float(y_hist, "y_hist")
        u_hist_arr = _as_2d_float(u_hist, "u_hist")
        if y_hist_arr.shape[1] != self.n_outputs:
            raise ValueError("y_hist has wrong number of output channels.")
        if u_hist_arr.shape[1] != self.n_inputs:
            raise ValueError("u_hist has wrong number of input channels.")

        phi = self._make_regressor(y_hist_arr, u_hist_arr)
        mean = self.muN @ phi

        param_cov = np.zeros((self.n_outputs, self.n_outputs), dtype=float)
        for out_idx in range(self.n_outputs):
            param_cov[out_idx, out_idx] = float(phi @ self.SigmaN[out_idx] @ phi)

        cov = param_cov + self.output_noise_covariance
        return mean, cov

    def rollout_posterior_samples(
        self,
        y_init: ArrayLike,
        u_future: ArrayLike,
        n_parameter_samples: int = 200,
        include_process_noise: bool = True,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        self._require_fit()
        y_init_arr = _as_2d_float(y_init, "y_init")
        u_future_arr = _as_2d_float(u_future, "u_future")
        if y_init_arr.shape[1] != self.n_outputs:
            raise ValueError("y_init has wrong output dimension.")
        if u_future_arr.shape[1] != self.n_inputs:
            raise ValueError("u_future has wrong input dimension.")
        if y_init_arr.shape[0] < self.na:
            raise ValueError("y_init must contain at least na rows.")
        if u_future_arr.shape[0] < self.nb:
            raise ValueError("u_future must contain at least nb rows.")

        horizon = u_future_arr.shape[0] - self.nb + 1
        posterior = self.sample_parameters(n_parameter_samples, random_state=random_state)
        theta_samples = posterior["theta"]
        paths = np.zeros((n_parameter_samples, horizon, self.n_outputs), dtype=float)
        rng = np.random.default_rng(random_state)

        for s in range(n_parameter_samples):
            y_hist = y_init_arr.copy()
            for k in range(horizon):
                u_hist = u_future_arr[: self.nb + k]
                phi = self._make_regressor(y_hist, u_hist)
                y_next = theta_samples[s] @ phi
                if include_process_noise:
                    y_next = y_next + rng.multivariate_normal(
                        np.zeros(self.n_outputs), self.output_noise_covariance
                    )
                y_hist = np.vstack([y_hist, y_next])
                paths[s, k] = y_next
        return paths
