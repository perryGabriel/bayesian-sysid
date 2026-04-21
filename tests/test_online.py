import numpy as np

from bayes_sysid import BayesianARX, OnlineBayesianARX, build_arx_regression, recursive_posterior_update, simulate_arx


def _make_dataset(seed: int = 123, n: int = 420):
    rng = np.random.default_rng(seed)
    u = rng.normal(size=n)
    y = simulate_arx([0.62, -0.17], [0.95, 0.25], u, sigma=0.08, random_state=seed + 1)
    return y, u[2:]


def test_recursive_update_matches_batch_posterior_no_forgetting():
    na, nb = 2, 2
    sigma2 = 0.08**2
    y, u = _make_dataset()

    batch = BayesianARX(na=na, nb=nb, sigma2=sigma2).fit(y, u)
    online = OnlineBayesianARX(na=na, nb=nb, sigma2=sigma2, snapshot_stride=80)

    for y_t, u_t in zip(y, u):
        online.update(y_t, u_t)

    np.testing.assert_allclose(online.mu, batch.muN, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(online.Sigma, batch.SigmaN, rtol=1e-10, atol=1e-10)


def test_order_invariant_posterior_for_fixed_regression_rows_no_forgetting():
    na, nb = 2, 2
    sigma2 = 0.08**2
    y, u = _make_dataset(seed=202)
    reg = build_arx_regression(y, u, na=na, nb=nb)

    p = na + nb
    mu_ref = np.zeros(p)
    Sigma_ref = np.eye(p) * 10.0
    for phi_t, y_t in zip(reg.Phi, reg.y_target):
        mu_ref, Sigma_ref, *_ = recursive_posterior_update(mu_ref, Sigma_ref, phi_t, y_t, sigma2=sigma2)

    rng = np.random.default_rng(7)
    perm = rng.permutation(reg.Phi.shape[0])
    mu_perm = np.zeros(p)
    Sigma_perm = np.eye(p) * 10.0
    for idx in perm:
        mu_perm, Sigma_perm, *_ = recursive_posterior_update(
            mu_perm,
            Sigma_perm,
            reg.Phi[idx],
            reg.y_target[idx],
            sigma2=sigma2,
        )

    np.testing.assert_allclose(mu_perm, mu_ref, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(Sigma_perm, Sigma_ref, rtol=1e-10, atol=1e-10)


def test_online_long_horizon_numerical_stability():
    rng = np.random.default_rng(99)
    n = 8_000
    u = rng.normal(size=n)
    y = simulate_arx([0.4, -0.12], [0.9, 0.1], u, sigma=0.05, random_state=100)
    u_obs = u[2:]

    model = OnlineBayesianARX(
        na=2,
        nb=2,
        sigma2=0.05**2,
        forgetting_factor=0.995,
        snapshot_stride=250,
    )
    for y_t, u_t in zip(y, u_obs):
        model.update(y_t, u_t)

    assert np.all(np.isfinite(model.mu))
    assert np.all(np.isfinite(model.Sigma))
    np.testing.assert_allclose(model.Sigma, model.Sigma.T, atol=1e-10)

    eigvals = np.linalg.eigvalsh(model.Sigma)
    assert np.all(eigvals > 0.0)
    assert len(model.snapshots) >= model.num_updates // model.snapshot_stride
