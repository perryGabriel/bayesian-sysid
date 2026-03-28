from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from bayes_sysid.models import BayesianARX


ControllerType = Literal["static", "pid"]


def _control_input(
    mode: ControllerType,
    e_k: float,
    e_prev: float,
    i_state: float,
    params: dict,
) -> tuple[float, float]:
    if mode == "static":
        k = float(params.get("k", 1.0))
        return k * e_k, i_state

    kp = float(params.get("kp", 1.0))
    ki = float(params.get("ki", 0.0))
    kd = float(params.get("kd", 0.0))
    i_new = i_state + e_k
    u = kp * e_k + ki * i_new + kd * (e_k - e_prev)
    return u, i_new


def simulate_closed_loop_arx(
    theta: ArrayLike,
    na: int,
    nb: int,
    r: ArrayLike,
    controller: ControllerType = "static",
    controller_params: dict | None = None,
    noise_std: float = 0.0,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate unity-feedback closed loop for ARX plant."""
    theta = np.asarray(theta, dtype=float).reshape(-1)
    if len(theta) < na + nb:
        raise ValueError("theta length must be at least na + nb.")

    r = np.asarray(r, dtype=float).reshape(-1)
    if len(r) <= max(na, nb):
        raise ValueError("reference length must exceed max(na, nb).")

    params = {} if controller_params is None else dict(controller_params)
    rng = np.random.default_rng(random_state)

    y_hist = [0.0] * max(na, nb)
    u_hist = [0.0] * nb

    y_out = []
    u_out = []
    i_state = 0.0
    e_prev = 0.0

    for k in range(max(na, nb), len(r)):
        e_k = float(r[k] - y_hist[-1])
        u_k, i_state = _control_input(controller, e_k, e_prev, i_state, params)
        e_prev = e_k

        if "u_min" in params:
            u_k = max(float(params["u_min"]), u_k)
        if "u_max" in params:
            u_k = min(float(params["u_max"]), u_k)

        u_hist.append(u_k)
        a = theta[:na]
        b = theta[na : na + nb]

        y_k = 0.0
        for i in range(1, na + 1):
            y_k += a[i - 1] * y_hist[-i]
        for j in range(1, nb + 1):
            y_k += b[j - 1] * u_hist[-j]
        if noise_std > 0:
            y_k += rng.normal(0.0, noise_std)

        y_hist.append(float(y_k))
        y_out.append(float(y_k))
        u_out.append(float(u_k))

    return np.asarray(y_out), np.asarray(u_out)


def monte_carlo_closed_loop_paths(
    model: BayesianARX,
    r: ArrayLike,
    n_parameter_samples: int = 200,
    controller: ControllerType = "static",
    controller_params: dict | None = None,
    include_process_noise: bool = True,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate closed-loop paths across posterior parameter samples."""
    noise_std = np.sqrt(model.sigma2) if include_process_noise else 0.0
    theta_samples = model.sample_parameters(n_parameter_samples, random_state=random_state)

    y_paths = []
    u_paths = []
    for idx, theta in enumerate(theta_samples):
        y, u = simulate_closed_loop_arx(
            theta=theta,
            na=model.na,
            nb=model.nb,
            r=r,
            controller=controller,
            controller_params=controller_params,
            noise_std=noise_std,
            random_state=None if random_state is None else random_state + idx,
        )
        y_paths.append(y)
        u_paths.append(u)

    return np.asarray(y_paths), np.asarray(u_paths)
