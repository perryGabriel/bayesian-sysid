import numpy as np

from bayes_sysid.control.lft import (
    DeltaBlock,
    RepeatedScalarUncertaintyBlock,
    ScalarUncertaintyBlock,
    StructuredDelta,
    build_nominal_interconnection,
    lower_lft_siso,
    posterior_robust_stability_confidence,
    structured_small_gain_surrogate,
    upper_lft_siso,
)
from bayes_sysid.control.robustness import robustness_report_from_structured_samples


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


def test_nominal_interconnection_shape_contracts_and_surrogate_shapes():
    model = _DeterministicPosterior(theta=np.array([-0.2, 0.2]), na=1, nb=1)
    w = np.linspace(1e-3, np.pi, 128)
    m11, m12, m21, m22 = build_nominal_interconnection(model.theta, model.na, model.nb, 0.5, w)

    assert m11.shape == m12.shape == m21.shape == m22.shape == w.shape

    structured = StructuredDelta(
        blocks=(
            ScalarUncertaintyBlock(bound=0.8, kind="real", dynamic=False),
            RepeatedScalarUncertaintyBlock(
                base_block=ScalarUncertaintyBlock(bound=0.3, kind="complex", dynamic=False),
                repetitions=2,
            ),
        )
    )
    m22_channels = np.vstack([0.6 * m22, 0.25 * m22, 0.15 * m22])
    surrogate = structured_small_gain_surrogate(m22_channels, structured)

    assert m22_channels.shape == (3, len(w))
    assert surrogate.scaled_gain_by_frequency.shape == (len(w),)


def test_posterior_robust_stability_confidence_stable_case_high_probability():
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


def test_posterior_robust_stability_confidence_seed_is_deterministic():
    model = _DeterministicPosterior(theta=np.array([-0.45, 0.5]), na=1, nb=1)
    w = np.linspace(1e-3, np.pi, 256)
    delta = DeltaBlock(gain_bound=1.0, dynamic=True, frequency_grid=w)

    r1 = posterior_robust_stability_confidence(
        posterior_model=model,
        controller_gain=0.3,
        delta_block=delta,
        w=w,
        n_theta_samples=10,
        n_delta_samples=20,
        random_state=123,
    )
    r2 = posterior_robust_stability_confidence(
        posterior_model=model,
        controller_gain=0.3,
        delta_block=delta,
        w=w,
        n_theta_samples=10,
        n_delta_samples=20,
        random_state=123,
    )

    assert r1 == r2


def test_uncertainty_bound_monotonicity_is_conservative_for_sampling_and_surrogate():
    model = _DeterministicPosterior(theta=np.array([-0.75, 0.8]), na=1, nb=1)
    w = np.linspace(1e-3, np.pi, 256)
    _, _, _, m22 = build_nominal_interconnection(model.theta, model.na, model.nb, 0.3, w)

    low = DeltaBlock(gain_bound=0.2, dynamic=True, frequency_grid=w)
    high = DeltaBlock(gain_bound=0.9, dynamic=True, frequency_grid=w)

    low_res = posterior_robust_stability_confidence(
        posterior_model=model,
        controller_gain=0.3,
        delta_block=low,
        w=w,
        n_theta_samples=16,
        n_delta_samples=30,
        random_state=7,
    )
    high_res = posterior_robust_stability_confidence(
        posterior_model=model,
        controller_gain=0.3,
        delta_block=high,
        w=w,
        n_theta_samples=16,
        n_delta_samples=30,
        random_state=7,
    )

    assert low_res.probability >= high_res.probability

    m22_channels = np.vstack([m22])
    low_struct = StructuredDelta(blocks=(ScalarUncertaintyBlock(bound=0.2, kind="complex", dynamic=False),))
    high_struct = StructuredDelta(blocks=(ScalarUncertaintyBlock(bound=0.9, kind="complex", dynamic=False),))
    low_surr = structured_small_gain_surrogate(m22_channels, low_struct)
    high_surr = structured_small_gain_surrogate(m22_channels, high_struct)

    assert low_surr.max_scaled_gain <= high_surr.max_scaled_gain


def test_robustness_reporting_outputs_expected_fields_and_sorted_ranking():
    w = np.linspace(1e-3, np.pi, 64)
    m22 = 0.4 * np.exp(-1j * w)
    m22_channels = np.vstack([0.8 * m22, 0.2 * m22])
    structured = StructuredDelta(
        blocks=(
            ScalarUncertaintyBlock(bound=0.9, kind="complex", dynamic=False),
            ScalarUncertaintyBlock(bound=0.3, kind="complex", dynamic=False),
        )
    )

    report = robustness_report_from_structured_samples(
        m22_by_channel=m22_channels,
        structured_delta=structured,
        n_samples=100,
        random_state=9,
    )

    assert 0 <= report.critical_point_encroachment_frequency <= 1
    assert report.worst_case_sampled_return_difference >= 0
    assert report.block_sensitivity_ranking[0][1] >= report.block_sensitivity_ranking[1][1]
