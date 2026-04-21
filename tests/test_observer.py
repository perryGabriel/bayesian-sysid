import numpy as np

from bayes_sysid.control.observer import (
    design_luenberger_gain,
    is_observable,
    observability_matrix,
    run_kalman_filter,
)


def test_observability_detects_observable_and_unobservable_pairs():
    A_obs = np.array([[1.0, 1.0], [0.0, 1.0]])
    C_obs = np.array([[1.0, 0.0]])

    O = observability_matrix(A_obs, C_obs)
    assert O.shape == (2, 2)
    assert np.linalg.matrix_rank(O) == 2
    assert is_observable(A_obs, C_obs)

    A_unobs = np.diag([0.9, 0.5])
    C_unobs = np.array([[1.0, 0.0]])

    O_unobs = observability_matrix(A_unobs, C_unobs)
    assert O_unobs.shape == (2, 2)
    assert np.linalg.matrix_rank(O_unobs) == 1
    assert not is_observable(A_unobs, C_unobs)


def test_luenberger_gain_pole_placement_matches_requested_eigenvalues():
    A = np.array([[1.1, 0.2], [0.0, 0.8]])
    C = np.array([[1.0, 0.0]])
    desired_poles = np.array([0.2, 0.3])

    L = design_luenberger_gain(A, C, desired_poles)
    eigvals = np.linalg.eigvals(A - L @ C)

    np.testing.assert_allclose(np.sort_complex(eigvals), np.sort_complex(desired_poles), atol=1e-8, rtol=1e-8)


def test_kalman_filter_improves_state_rmse_over_open_loop_prediction():
    rng = np.random.default_rng(7)

    A = np.array([[1.0, 0.1], [0.0, 0.95]])
    B = np.array([[0.0], [0.1]])
    C = np.array([[1.0, 0.0]])

    Q = np.diag([1e-3, 3e-3])
    R = np.array([[2e-2]])

    T = 250
    u = rng.normal(0.0, 0.8, size=(T, 1))

    x_true = np.zeros((T, 2), dtype=float)
    y = np.zeros((T, 1), dtype=float)
    xk = np.array([0.4, -0.2], dtype=float)

    for k in range(T):
        wk = rng.multivariate_normal(mean=np.zeros(2), cov=Q)
        vk = rng.multivariate_normal(mean=np.zeros(1), cov=R)
        xk = A @ xk + B @ u[k] + wk
        yk = C @ xk + vk
        x_true[k] = xk
        y[k] = yk

    x_filt, _ = run_kalman_filter(A=A, B=B, C=C, Q=Q, R=R, u=u, y=y)

    x_open = np.zeros_like(x_true)
    xk_open = np.zeros(2, dtype=float)
    for k in range(T):
        xk_open = A @ xk_open + B @ u[k]
        x_open[k] = xk_open

    rmse_kf = np.sqrt(np.mean((x_filt - x_true) ** 2))
    rmse_open = np.sqrt(np.mean((x_open - x_true) ** 2))

    assert rmse_kf < rmse_open
    assert rmse_kf <= 0.85 * rmse_open
