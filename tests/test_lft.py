import numpy as np

from bayes_sysid.control.lft import (
    DeltaBlock,
    build_nominal_interconnection,
    lower_lft_siso,
    posterior_robust_stability_confidence,
    upper_lft_siso,
)


class _DeterministicPosterior:
    def __init__(self, theta: np.ndarray, na: int, nb: int):
        self.theta = np.asarray(theta, dtype=float)
        self.na = na
        self.nb = nb

    def sample_parameters(self, n_samples: int, random_state: int | None = None) -> np.ndarray:
        _ = random_state
        return np.tile(self.theta, (n_samples, 1))


def test_lower_upper_lft_scalar_reduce_to_closed_forms():
    m11 = np.array([0.2 + 0.1j])
    m12 = np.array([0.4])
    m21 = np.array([0.3])
    m22 = np.array([0.1 - 0.05j])
    delta = np.array([0.2])

    fl = lower_lft_siso(m11, m12, m21, m22, delta)
    fu = upper_lft_siso(m11, m12, m21, m22, delta)

    expected_fl = m11 + (m12 * delta * m21) / (1 - m22 * delta + 1e-12)
    expected_fu = m22 + (m21 * delta * m12) / (1 - m11 * delta + 1e-12)

    assert np.allclose(fl, expected_fl)
    assert np.allclose(fu, expected_fu)


def test_posterior_robust_stability_confidence_stable_case_high_probability():
    # Stable first-order plant: G(z)=0.2 z^-1 / (1 - 0.2 z^-1), closed-loop uncertainty gain stays < 1
    model = _DeterministicPosterior(theta=np.array([-0.2, 0.2]), na=1, nb=1)
    w = np.linspace(1e-3, np.pi, 512)
    delta = DeltaBlock(gain_bound=1.0, dynamic=False)

    result = posterior_robust_stability_confidence(
        posterior_model=model,
        controller_gain=0.5,
        delta_block=delta,
        w=w,
        n_theta_samples=40,
        n_delta_samples=60,
        random_state=1,
    )

    assert result.probability > 0.9
    assert 0 <= result.ci_low <= result.ci_high <= 1


def test_posterior_robust_stability_confidence_unstable_case_low_probability():
    # Unstable robust loop under |Delta|<=1 because |M22| > 1 at low freq.
    model = _DeterministicPosterior(theta=np.array([-0.9, 0.95]), na=1, nb=1)
    w = np.linspace(1e-3, np.pi, 512)
    delta = DeltaBlock(gain_bound=1.0, dynamic=False)

    m11, m12, m21, m22 = build_nominal_interconnection(model.theta, model.na, model.nb, 0.2, w)
    assert np.max(np.abs(m22)) > 1.0
    assert m11.shape == m12.shape == m21.shape == m22.shape

    result = posterior_robust_stability_confidence(
        posterior_model=model,
        controller_gain=0.2,
        delta_block=delta,
        w=w,
        n_theta_samples=40,
        n_delta_samples=80,
        random_state=2,
    )

    assert result.probability < 0.2
    assert 0 <= result.ci_low <= result.ci_high <= 1
