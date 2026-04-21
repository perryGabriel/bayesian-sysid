import numpy as np

from bayes_sysid.control.lqr import (
    closed_loop_poles,
    lqr_gain_from_realization,
    sampled_bayesian_lqr_summary,
    solve_discrete_lqr,
)


class _PosteriorModel:
    def __init__(self, na: int, nb: int, theta_mean: np.ndarray, scale: float = 0.02) -> None:
        self.na = na
        self.nb = nb
        self.theta_mean = np.asarray(theta_mean, dtype=float)
        self.scale = float(scale)

    def sample_parameters(self, n_samples: int, random_state: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(random_state)
        return self.theta_mean + self.scale * rng.standard_normal((n_samples, self.na + self.nb))


def test_lqr_shape_validation_raises_for_bad_Q_shape():
    A = np.array([[1.0, 0.1], [0.0, 0.95]])
    B = np.array([[0.0], [1.0]])
    Q_bad = np.eye(3)
    R = np.array([[1.0]])

    with np.testing.assert_raises(ValueError):
        _ = solve_discrete_lqr(A, B, Q_bad, R)


def test_lqr_small_system_stabilizes_open_loop_unstable_mode():
    A = np.array([[1.1]])
    B = np.array([[1.0]])
    Q = np.array([[1.0]])
    R = np.array([[1.0]])

    P, K = solve_discrete_lqr(A, B, Q, R)
    poles = closed_loop_poles(A, B, K)

    assert P.shape == (1, 1)
    assert K.shape == (1, 1)
    assert np.abs(poles[0]) < 1.0


def test_lqr_gain_function_matches_solver_gain():
    A = np.array([[0.9, 0.1], [0.0, 0.8]])
    B = np.array([[0.0], [1.0]])
    Q = np.eye(2)
    R = np.array([[0.5]])

    _, K_ref = solve_discrete_lqr(A, B, Q, R)
    K = lqr_gain_from_realization(A, B, Q, R)
    np.testing.assert_allclose(K, K_ref, atol=1e-12)


def test_sampled_bayesian_lqr_summary_keys_and_reproducibility():
    model = _PosteriorModel(na=2, nb=2, theta_mean=np.array([-0.3, 0.08, 0.9, 0.1]))
    Q = np.eye(2)
    R = np.array([[1.0]])

    s1 = sampled_bayesian_lqr_summary(model, Q=Q, R=R, n_samples=60, random_state=4)
    s2 = sampled_bayesian_lqr_summary(model, Q=Q, R=R, n_samples=60, random_state=4)

    expected_keys = {
        "n_samples",
        "stability_probability",
        "quantiles",
        "cost_quantiles",
        "pole_radius_quantiles",
        "gain_norm_quantiles",
    }
    assert expected_keys.issubset(s1.keys())
    assert s1 == s2
    assert 0.0 <= s1["stability_probability"] <= 1.0
