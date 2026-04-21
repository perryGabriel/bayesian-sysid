import numpy as np
import pytest

from bayes_sysid import BayesianMIMOARX, build_mimo_regression
from bayes_sysid.control.dsf import transfer_matrix_samples_from_mimo_posterior


def simulate_mimo_arx(A1: np.ndarray, B1: np.ndarray, u: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_samples = u.shape[0]
    n_outputs = A1.shape[0]
    y = np.zeros((n_samples, n_outputs), dtype=float)
    for t in range(1, n_samples):
        y[t] = A1 @ y[t - 1] + B1 @ u[t - 1] + rng.normal(0.0, sigma, size=n_outputs)
    return y


def test_build_mimo_regression_dimensions_and_block_order():
    y = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ]
    )
    u = np.array(
        [
            [10.0, 11.0],
            [12.0, 13.0],
            [14.0, 15.0],
            [16.0, 17.0],
        ]
    )
    reg = build_mimo_regression(y, u, na=1, nb=1)
    assert reg.Phi.shape == (3, 4)
    assert reg.Y_target.shape == (3, 2)
    np.testing.assert_allclose(reg.Phi[0], np.array([1.0, 2.0, 10.0, 11.0]))
    np.testing.assert_allclose(reg.Y_target[0], np.array([3.0, 4.0]))


def test_bayesian_mimo_arx_recovers_low_order_2x2_system():
    A1 = np.array([[0.45, -0.12], [0.08, 0.35]])
    B1 = np.array([[0.9, 0.15], [-0.05, 0.7]])
    rng = np.random.default_rng(22)
    u = rng.normal(size=(1500, 2))
    y = simulate_mimo_arx(A1, B1, u, sigma=0.02, seed=23)

    model = BayesianMIMOARX(na=1, nb=1, sigma2=0.02**2).fit(y, u)
    posterior = model.sample_parameters(1, random_state=3)

    A_hat = posterior["a_lags"][0, 0]
    B_hat = posterior["b_lags"][0, 0]

    assert np.max(np.abs(A_hat - A1)) < 0.08
    assert np.max(np.abs(B_hat - B1)) < 0.08


def test_reproducibility_and_native_dsf_posterior_transfer_samples():
    A1 = np.array([[0.4, 0.02], [-0.03, 0.28]])
    B1 = np.array([[0.6, 0.1], [0.0, 0.5]])
    rng = np.random.default_rng(5)
    u = rng.normal(size=(500, 2))
    y = simulate_mimo_arx(A1, B1, u, sigma=0.03, seed=9)

    model = BayesianMIMOARX(na=1, nb=1, sigma2=0.03**2, coupled_output_covariance=True).fit(y, u)

    samples_a = model.sample_parameters(15, random_state=11)
    samples_b = model.sample_parameters(15, random_state=11)
    np.testing.assert_allclose(samples_a["theta"], samples_b["theta"])

    y_init = y[-1:]
    u_future = rng.normal(size=(20, 2))
    rollout_a = model.rollout_posterior_samples(y_init, u_future, n_parameter_samples=12, random_state=77)
    rollout_b = model.rollout_posterior_samples(y_init, u_future, n_parameter_samples=12, random_state=77)
    np.testing.assert_allclose(rollout_a, rollout_b)

    w = np.linspace(0.1, np.pi - 0.1, 16)
    G_samples = transfer_matrix_samples_from_mimo_posterior(samples_a, w)
    assert G_samples.shape == (15, 16, 2, 2)


def test_failure_modes_have_clear_messages():
    model = BayesianMIMOARX(na=1, nb=1, sigma2=0.1)
    with pytest.raises(RuntimeError, match="Model has not been fit yet"):
        model.sample_parameters(2)

    y = np.zeros((10, 2))
    u_bad = np.zeros((9, 2))
    with pytest.raises(ValueError, match="same number of samples"):
        model.fit(y, u_bad)

    model.fit(np.zeros((10, 2)), np.zeros((10, 2)))
    with pytest.raises(ValueError, match="wrong number of output channels"):
        model.predict_next_distribution(np.zeros((2, 1)), np.zeros((2, 2)))
