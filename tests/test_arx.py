import numpy as np

from bayes_sysid import BayesianARX, LeastSquaresARX, simulate_arx


def test_shapes_and_fit():
    rng = np.random.default_rng(0)
    a_true = np.array([0.4, -0.1])
    b_true = np.array([0.8, 0.2])
    u = rng.normal(size=120)
    y = simulate_arx(a_true, b_true, u, sigma=0.05, random_state=1)
    u_obs = u[2:]

    ls = LeastSquaresARX(na=2, nb=2).fit(y, u_obs)
    bayes = BayesianARX(na=2, nb=2, sigma2=0.05**2).fit(y, u_obs)

    assert ls.theta_hat.shape == (4,)
    assert bayes.muN.shape == (4,)
    assert bayes.SigmaN.shape == (4, 4)


def test_predictive_variance_positive():
    rng = np.random.default_rng(4)
    u = rng.normal(size=80)
    y = simulate_arx([0.3], [1.0], u, sigma=0.1, random_state=2)
    u_obs = u[1:]

    model = BayesianARX(na=1, nb=1, sigma2=0.01).fit(y, u_obs)
    mean, var = model.predict_next_distribution(y[-1:], u_obs[-1:])
    assert np.isfinite(mean)
    assert var > 0.0
