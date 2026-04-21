import numpy as np

from bayes_sysid.control.lqg import (
    lqg_controller,
    sampled_bayesian_lqg_summary,
    steady_state_kalman_gain,
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


def test_kalman_shape_validation_raises_for_bad_Rn_shape():
    A = np.array([[0.9, 0.0], [0.1, 0.8]])
    C = np.array([[1.0, 0.0]])
    Qn = np.eye(2) * 0.1
    Rn_bad = np.eye(2)

    with np.testing.assert_raises(ValueError):
        _ = steady_state_kalman_gain(A, C, Qn, Rn_bad)


def test_small_system_lqg_gains_stabilize_control_and_estimation_dynamics():
    A = np.array([[1.05]])
    B = np.array([[1.0]])
    C = np.array([[1.0]])
    Q = np.array([[1.0]])
    R = np.array([[1.0]])
    Qn = np.array([[0.1]])
    Rn = np.array([[0.2]])

    K, L = lqg_controller(A, B, C, Q, R, Qn, Rn)

    assert K.shape == (1, 1)
    assert L.shape == (1, 1)
    assert np.abs(np.linalg.eigvals(A - B @ K)[0]) < 1.0
    assert np.abs(np.linalg.eigvals(A - L @ C)[0]) < 1.0


def test_sampled_bayesian_lqg_summary_keys_and_reproducibility():
    model = _PosteriorModel(na=2, nb=2, theta_mean=np.array([-0.25, 0.05, 0.8, 0.15]))
    Q = np.eye(2)
    R = np.array([[1.0]])
    Qn = np.eye(2) * 0.1
    Rn = np.array([[0.2]])

    s1 = sampled_bayesian_lqg_summary(model, Q=Q, R=R, Qn=Qn, Rn=Rn, n_samples=50, random_state=5)
    s2 = sampled_bayesian_lqg_summary(model, Q=Q, R=R, Qn=Qn, Rn=Rn, n_samples=50, random_state=5)

    expected_keys = {
        "n_samples",
        "stability_probability",
        "quantiles",
        "control_pole_radius_quantiles",
        "estimator_pole_radius_quantiles",
        "control_gain_norm_quantiles",
        "estimator_gain_norm_quantiles",
    }
    assert expected_keys.issubset(s1.keys())
    assert s1 == s2
    assert 0.0 <= s1["stability_probability"] <= 1.0
