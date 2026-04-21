import numpy as np

from bayes_sysid import BayesianARX, simulate_arx
from bayes_sysid.control.gramians import (
    controllability_gramian,
    hankel_singular_values,
    observability_gramian,
    posterior_hsv_summary,
)


def test_small_system_gramians_are_symmetric_psd_and_solve_lyapunov():
    A = np.array([[0.7, 0.1], [0.0, 0.5]])
    B = np.array([[1.0], [0.3]])
    C = np.array([[1.0, -0.2]])

    Wc = controllability_gramian(A, B)
    Wo = observability_gramian(A, C)

    np.testing.assert_allclose(Wc, Wc.T, atol=1e-10)
    np.testing.assert_allclose(Wo, Wo.T, atol=1e-10)

    assert np.min(np.linalg.eigvalsh(Wc)) >= -1e-10
    assert np.min(np.linalg.eigvalsh(Wo)) >= -1e-10

    np.testing.assert_allclose(A @ Wc @ A.T - Wc + B @ B.T, np.zeros_like(Wc), atol=1e-9)
    np.testing.assert_allclose(A.T @ Wo @ A - Wo + C.T @ C, np.zeros_like(Wo), atol=1e-9)


def test_hankel_singular_values_are_sorted_monotone_nonnegative():
    A = np.array([[0.75, 0.0], [0.1, 0.45]])
    B = np.array([[1.0], [0.2]])
    C = np.array([[1.0, 0.5]])

    hsv = hankel_singular_values(A, B, C)

    assert np.all(hsv >= -1e-12)
    assert np.all(np.diff(hsv) <= 1e-12)


def test_posterior_hsv_summary_is_robust_on_synthetic_data():
    rng = np.random.default_rng(22)
    a_true = np.array([0.5, -0.2])
    b_true = np.array([0.8, 0.15])
    sigma = 0.08

    u = rng.normal(size=320)
    y = simulate_arx(a_true, b_true, u, sigma=sigma, random_state=33)
    model = BayesianARX(na=2, nb=2, sigma2=sigma**2).fit(y, u[2:])

    summary = posterior_hsv_summary(model, n_samples=200, random_state=7)

    assert summary["n_samples"] == 200
    assert summary["n_stable"] > 100
    assert summary["stable_fraction"] > 0.5

    hsv_q = np.asarray(summary["hsv_quantiles"], dtype=float)
    assert hsv_q.shape == (3, model.na)
    assert np.all(np.diff(hsv_q[1]) <= 1e-12)

    retention = summary["mode_energy_retention"]
    assert int(retention["0.90"]) <= int(retention["0.95"]) <= int(retention["0.99"])

    wc_diag = summary["gramian_diagnostics"]["Wc"]
    wo_diag = summary["gramian_diagnostics"]["Wo"]
    assert wc_diag["median_condition_number"] > 0
    assert wo_diag["median_condition_number"] > 0
