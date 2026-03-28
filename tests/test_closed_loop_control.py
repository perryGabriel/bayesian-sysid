import numpy as np
import pytest

from bayes_sysid import BayesianARX, simulate_arx
from bayes_sysid.control.closed_loop import monte_carlo_closed_loop_paths, simulate_closed_loop_arx


def test_simulate_closed_loop_arx_shapes_static_and_pid():
    theta = np.array([0.4, -0.1, 0.9, 0.2])
    r = np.ones(40)

    y_s, u_s = simulate_closed_loop_arx(theta, na=2, nb=2, r=r, controller="static", controller_params={"k": 0.5})
    y_p, u_p = simulate_closed_loop_arx(
        theta,
        na=2,
        nb=2,
        r=r,
        controller="pid",
        controller_params={"kp": 0.8, "ki": 0.05, "kd": 0.01},
    )

    assert y_s.shape == u_s.shape
    assert y_p.shape == u_p.shape
    assert len(y_s) == len(r) - 2


def test_closed_loop_validations_and_monte_carlo_paths():
    theta = np.array([0.4, -0.1, 0.9, 0.2])
    with pytest.raises(ValueError):
        simulate_closed_loop_arx(theta[:2], na=2, nb=2, r=np.ones(20))
    with pytest.raises(ValueError):
        simulate_closed_loop_arx(theta, na=2, nb=2, r=np.ones(2))

    rng = np.random.default_rng(3)
    u = rng.normal(size=180)
    y = simulate_arx([0.4, -0.1], [0.9, 0.2], u, sigma=0.05, random_state=4)
    model = BayesianARX(na=2, nb=2, sigma2=0.05**2).fit(y, u[2:])

    y_paths, u_paths = monte_carlo_closed_loop_paths(
        model,
        r=np.ones(60),
        n_parameter_samples=25,
        controller="static",
        controller_params={"k": 0.5},
        include_process_noise=False,
        random_state=5,
    )
    assert y_paths.shape[0] == 25
    assert u_paths.shape == y_paths.shape
