import numpy as np
import pytest

from bayes_sysid import (
    BayesianARX,
    BayesianARXUnknownNoise,
    LeastSquaresARX,
    build_arx_regression,
    diagonal_arx_prior_covariance,
    isotropic_prior_covariance,
    rolling_order_search,
    scale_prior_covariance_by_regressor_variance,
    simulate_arx,
)


def make_data(seed: int = 0, n: int = 140):
    rng = np.random.default_rng(seed)
    a_true = np.array([0.45, -0.2])
    b_true = np.array([0.8, 0.15])
    u = rng.normal(size=n)
    y = simulate_arx(a_true, b_true, u, sigma=0.07, random_state=seed + 1)
    return y, u[2:]


def test_build_arx_regression_shapes_and_values():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    u = np.array([10.0, 11.0, 12.0, 13.0])
    reg = build_arx_regression(y, u, na=2, nb=1)
    assert reg.max_lag == 2
    assert reg.Phi.shape == (2, 3)
    np.testing.assert_allclose(reg.Phi[0], np.array([2.0, 1.0, 11.0]))
    np.testing.assert_allclose(reg.y_target, np.array([3.0, 4.0]))


def test_build_arx_regression_validation_errors():
    with pytest.raises(ValueError):
        build_arx_regression([1, 2], [1, 2], na=0, nb=0)
    with pytest.raises(ValueError):
        build_arx_regression([1, 2, 3], [1, 2], na=1, nb=1)
    with pytest.raises(ValueError):
        build_arx_regression([1, 2], [1, 2], na=2, nb=0)


def test_least_squares_requires_fit_and_predicts():
    model = LeastSquaresARX(na=1, nb=1)
    with pytest.raises(RuntimeError):
        model.predict_next([0.0], [0.0])

    y, u = make_data(seed=4)
    model.fit(y, u)
    pred = model.predict_next(y[-1:], u[-1:])
    assert np.isfinite(pred)


def test_least_squares_rollout_and_validation():
    y, u = make_data(seed=5)
    model = LeastSquaresARX(na=2, nb=2).fit(y, u)

    with pytest.raises(ValueError):
        model.simulate_one_step_rollout(y_init=y[-1:], u=u[-2:])
    with pytest.raises(ValueError):
        model.simulate_one_step_rollout(y_init=y[-2:], u=u[-1:])

    u_future = np.array([0.2, -0.1, 0.5, 0.3])
    rollout = model.simulate_one_step_rollout(y_init=y[-2:], u=u_future)
    assert rollout.shape == (3,)


def test_bayesian_arx_core_api_and_checks():
    y, u = make_data(seed=6)
    model = BayesianARX(na=2, nb=2, sigma2=0.07**2).fit(y, u)

    mean = model.posterior_mean_prediction(y[-2:], u[-2:])
    dmean, dvar = model.predict_next_distribution(y[-2:], u[-2:])
    grid = np.linspace(dmean - 1.0, dmean + 1.0, 9)
    dens = model.predictive_density_grid(y[-2:], u[-2:], grid)
    assert np.isfinite(mean)
    assert np.isfinite(dmean)
    assert dvar > 0
    assert dens.shape == grid.shape

    theta = model.sample_parameters(10, random_state=2)
    y_next = model.sample_predictive_next(y[-2:], u[-2:], n_samples=50, random_state=3)
    paths = model.rollout_posterior_samples(
        y_init=y[-2:],
        u_future=np.array([0.3, -0.2, 0.1, 0.6, -0.1]),
        n_parameter_samples=20,
        include_process_noise=False,
        random_state=9,
    )
    assert theta.shape == (10, 4)
    assert y_next.shape == (50,)
    assert paths.shape == (20, 4)


def test_bayesian_arx_input_and_prior_validation():
    with pytest.raises(ValueError):
        BayesianARX(na=1, nb=1, sigma2=0.0)
    with pytest.raises(ValueError):
        BayesianARX(na=1, nb=1, sigma2=0.1, Sigma0=np.eye(3))

    model = BayesianARX(na=1, nb=1, sigma2=0.1)
    with pytest.raises(RuntimeError):
        model.predict_next_distribution([0.0], [0.0])

    y, u = make_data(seed=7)
    model.fit(y, u)
    with pytest.raises(ValueError):
        model.make_regressor([], [])
    with pytest.raises(ValueError):
        model.rollout_posterior_samples(y_init=np.array([]), u_future=np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        model.rollout_posterior_samples(y_init=y[-2:], u_future=np.array([]))


def test_unknown_noise_model_fit_predict_and_sampling():
    y, u = make_data(seed=8)
    model = BayesianARXUnknownNoise(na=2, nb=2).fit(y, u)

    assert model.alphaN > model.alpha0
    assert model.betaN > 0
    assert model.sigma2_posterior_mean > 0

    mean, scale2, dof = model.predict_next_distribution(y[-2:], u[-2:])
    assert np.isfinite(mean)
    assert scale2 > 0
    assert dof > 0

    dens = model.predictive_density_grid(y[-2:], u[-2:], np.linspace(mean - 1, mean + 1, 5))
    assert dens.shape == (5,)
    assert np.all(dens >= 0)

    sigma2 = model.sample_sigma2(12, random_state=1)
    theta = model.sample_parameters(12, random_state=1)
    assert sigma2.shape == (12,)
    assert np.all(sigma2 > 0)
    assert theta.shape == (12, 4)


def test_unknown_noise_validation_and_require_fit():
    with pytest.raises(ValueError):
        BayesianARXUnknownNoise(na=1, nb=1, alpha0=0, beta0=1)
    with pytest.raises(ValueError):
        BayesianARXUnknownNoise(na=1, nb=1, Lambda0=np.eye(3))

    model = BayesianARXUnknownNoise(na=1, nb=1)
    with pytest.raises(RuntimeError):
        _ = model.sigma2_posterior_mean
    with pytest.raises(RuntimeError):
        model.sample_sigma2(3)

    model_alpha = BayesianARXUnknownNoise(na=1, nb=1, alpha0=0.5, beta0=1.0)
    y = np.array([0.0, 1.0, 2.0])
    u = np.array([0.0, 1.0, 2.0])
    model_alpha.fit(y, u)
    model_alpha.alphaN = 1.0
    with pytest.raises(RuntimeError):
        _ = model_alpha.sigma2_posterior_mean


def test_rolling_order_search_happy_path_and_validation():
    rng = np.random.default_rng(42)
    u = rng.normal(size=220)
    y = simulate_arx([0.55], [1.2], u, sigma=0.02, random_state=43)
    u_obs = u[1:]

    result = rolling_order_search(
        y,
        u_obs,
        na_candidates=[1, 2, 3],
        nb_candidates=[1, 2, 3],
        train_fraction=0.6,
        sigma2=0.02**2,
        metric="mse",
    )
    assert result.na in {1, 2, 3}
    assert result.nb in {1, 2, 3}
    assert np.isfinite(result.score)

    result_nll = rolling_order_search(
        y,
        u_obs,
        na_candidates=[1, 2],
        nb_candidates=[1, 2],
        train_fraction=0.6,
        sigma2=0.02**2,
        metric="nll",
    )
    assert np.isfinite(result_nll.score)

    with pytest.raises(ValueError):
        rolling_order_search(y, u_obs[:-1], [1], [1])
    with pytest.raises(ValueError):
        rolling_order_search(y, u_obs, [], [1])
    with pytest.raises(ValueError):
        rolling_order_search(y, u_obs, [1], [1], train_fraction=1.0)
    with pytest.raises(ValueError):
        rolling_order_search(y, u_obs, [1], [1], metric="bad")
    with pytest.raises(ValueError):
        rolling_order_search(y, u_obs, [50], [50], train_fraction=0.2)


def test_prior_configuration_helpers():
    iso = isotropic_prior_covariance(4, variance=3.0)
    np.testing.assert_allclose(np.diag(iso), np.array([3.0, 3.0, 3.0, 3.0]))

    diag = diagonal_arx_prior_covariance(na=2, nb=3, ar_variance=4.0, input_variance=1.5)
    np.testing.assert_allclose(np.diag(diag), np.array([4.0, 4.0, 1.5, 1.5, 1.5]))

    Phi = np.array(
        [
            [1.0, 2.0, -1.0],
            [2.0, 1.0, -1.5],
            [3.0, 4.0, -2.0],
            [4.0, 3.0, -0.5],
        ]
    )
    base = np.diag([10.0, 5.0, 2.0])
    scaled = scale_prior_covariance_by_regressor_variance(Phi, base)
    var = np.var(Phi, axis=0)
    expected = np.diag(np.diag(base) / np.maximum(var, 1e-8))
    np.testing.assert_allclose(scaled, expected)


def test_prior_configuration_helpers_validation():
    with pytest.raises(ValueError):
        isotropic_prior_covariance(0, variance=1.0)
    with pytest.raises(ValueError):
        isotropic_prior_covariance(2, variance=0.0)

    with pytest.raises(ValueError):
        diagonal_arx_prior_covariance(na=0, nb=0)
    with pytest.raises(ValueError):
        diagonal_arx_prior_covariance(na=1, nb=1, ar_variance=-1.0)

    Phi = np.ones((5, 2))
    with pytest.raises(ValueError):
        scale_prior_covariance_by_regressor_variance(Phi.reshape(-1), np.eye(2))
    with pytest.raises(ValueError):
        scale_prior_covariance_by_regressor_variance(Phi, np.eye(3))
    with pytest.raises(ValueError):
        scale_prior_covariance_by_regressor_variance(Phi, np.array([[1.0, 0.1], [0.1, 1.0]]))
    with pytest.raises(ValueError):
        scale_prior_covariance_by_regressor_variance(Phi, np.eye(2), min_variance=0.0)
