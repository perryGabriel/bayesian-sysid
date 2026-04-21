import numpy as np

from bayes_sysid.control.dsf import (
    dsf_from_transfer_matrix,
    evaluate_identifiability_conditions,
    identifiability_warning_flags,
    minimum_horizon_guidance,
    posterior_edge_confidence_summary,
    posterior_edge_probability,
    transfer_matrix_from_mimo_arx,
    validate_excitation_richness,
    validate_identifiability_assumptions,
)


def _build_2x2_ground_truth_transfer(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z_inv = np.exp(-1j * w)
    p1 = 0.8 * z_inv / (1 - 0.25 * z_inv)
    p2 = 0.6 * z_inv / (1 - 0.15 * z_inv)
    q12 = 0.35 * z_inv / (1 - 0.2 * z_inv)

    Q = np.zeros((w.size, 2, 2), dtype=complex)
    P = np.zeros((w.size, 2, 2), dtype=complex)
    G = np.zeros((w.size, 2, 2), dtype=complex)

    eye = np.eye(2, dtype=complex)
    for k in range(w.size):
        Q[k, 0, 1] = q12[k]  # edge y2 -> y1 is present
        P[k, 0, 0] = p1[k]
        P[k, 1, 1] = p2[k]
        G[k] = np.linalg.inv(eye - Q[k]) @ P[k]

    return G, Q


def test_transfer_matrix_from_mimo_arx_matches_expected_first_order_formula():
    w = np.linspace(0.05, np.pi - 0.05, 64)

    # y[t] + A1 y[t-1] = B1 u[t-1] with one directed coupling y2 -> y1.
    A1 = np.array([[[-0.25, -0.20], [0.00, -0.15]]])
    B1 = np.array([[[0.8, 0.0], [0.0, 0.6]]])

    G = transfer_matrix_from_mimo_arx(A1, B1, w)

    z_inv = np.exp(-1j * w)
    for k, zi in enumerate(z_inv):
        A_eval = np.eye(2) + A1[0] * zi
        B_eval = B1[0] * zi
        np.testing.assert_allclose(G[k], np.linalg.solve(A_eval, B_eval), atol=1e-12, rtol=1e-12)


def test_dsf_from_transfer_matrix_recovers_known_sparse_edge_pattern():
    w = np.linspace(0.08, np.pi - 0.08, 128)
    G, Q_true = _build_2x2_ground_truth_transfer(w)

    dsf = dsf_from_transfer_matrix(G, method="stable_factorization")
    Q_hat = dsf["Q"]

    max_present = np.max(np.abs(Q_hat[:, 0, 1] - Q_true[:, 0, 1]))
    max_absent = np.max(np.abs(Q_hat[:, 1, 0]))

    assert max_present < 1e-10
    assert max_absent < 1e-12


def test_posterior_edge_probability_highlights_true_sparse_edge():
    rng = np.random.default_rng(7)
    w = np.linspace(0.1, np.pi - 0.1, 80)
    _, Q_true = _build_2x2_ground_truth_transfer(w)

    n_samples = 250
    noise_scale = 0.01
    samples = np.repeat(Q_true[None, ...], n_samples, axis=0)
    samples = samples + noise_scale * (
        rng.standard_normal(samples.shape) + 1j * rng.standard_normal(samples.shape)
    )

    probs = posterior_edge_probability(samples, threshold=0.05)

    assert probs[0, 1] > 0.95
    assert probs[1, 0] < 0.1
    assert probs[0, 0] == 0.0
    assert probs[1, 1] == 0.0


def test_validation_helpers_report_identifiability_and_excitation_health():
    w = np.linspace(0.1, np.pi - 0.1, 32)
    G, _ = _build_2x2_ground_truth_transfer(w)

    ident = validate_identifiability_assumptions(G)
    assert ident["passes"] is True

    rng = np.random.default_rng(3)
    u_rich = rng.normal(size=(400, 2))
    excitation = validate_excitation_richness(u_rich)
    assert excitation["passes"] is True

    u_poor = np.column_stack([u_rich[:, 0], 2.0 * u_rich[:, 0]])
    poor = validate_excitation_richness(u_poor)
    assert poor["passes"] is False


def test_identifiability_conditions_distinguish_identifiable_vs_non_identifiable():
    w = np.linspace(0.1, np.pi - 0.1, 48)
    G, _ = _build_2x2_ground_truth_transfer(w)
    rng = np.random.default_rng(17)

    u_rich = rng.normal(size=(800, 2))
    good = evaluate_identifiability_conditions(G, u_rich, max_lag=2)
    assert good["all_conditions_satisfied"] is True

    G_bad = G.copy()
    G_bad[:, :, 1] = G_bad[:, :, 0]
    u_bad = np.column_stack([u_rich[:, 0], u_rich[:, 0]])
    bad = evaluate_identifiability_conditions(G_bad, u_bad, max_lag=2)
    assert bad["all_conditions_satisfied"] is False
    assert bad["rank_condition"]["satisfied"] is False
    assert bad["persistence_of_excitation"]["passes"] is False


def test_false_edge_suppression_under_posterior_summary():
    rng = np.random.default_rng(5)
    w = np.linspace(0.1, np.pi - 0.1, 64)
    _, Q_true = _build_2x2_ground_truth_transfer(w)

    n_samples = 500
    samples = np.repeat(Q_true[None, ...], n_samples, axis=0)
    samples = samples + 0.005 * (
        rng.standard_normal(samples.shape) + 1j * rng.standard_normal(samples.shape)
    )
    summary = posterior_edge_confidence_summary(samples, threshold=0.05)
    probs = summary["posterior_probability"]
    upper = summary["credible_interval_upper"]

    assert probs[0, 1] > 0.95
    assert probs[1, 0] < 0.08
    assert upper[1, 0] < 0.15


def test_edge_probability_sensitivity_to_snr_and_trajectory_length():
    rng = np.random.default_rng(23)
    w = np.linspace(0.1, np.pi - 0.1, 48)
    _, Q_true = _build_2x2_ground_truth_transfer(w)

    truth = np.array([[False, True], [False, False]])

    def sample_summary(n_samples: int, noise_scale: float):
        samples = np.repeat(Q_true[None, ...], n_samples, axis=0)
        samples = samples + noise_scale * (
            rng.standard_normal(samples.shape) + 1j * rng.standard_normal(samples.shape)
        )
        return posterior_edge_confidence_summary(
            samples,
            threshold=0.08,
            calibration_truth=truth,
        )

    low_snr_short = sample_summary(n_samples=80, noise_scale=0.05)
    high_snr_long = sample_summary(n_samples=400, noise_scale=0.01)

    p_low = low_snr_short["posterior_probability"]
    p_high = high_snr_long["posterior_probability"]

    assert p_high[1, 0] < p_low[1, 0]
    assert high_snr_long["calibration"]["brier_score"] < low_snr_short["calibration"]["brier_score"]


def test_experiment_design_helpers_and_warning_flags():
    guidance = minimum_horizon_guidance(n_inputs=2, model_order=3, safety_factor=8.0)
    assert guidance["minimum_rank_horizon"] == 7
    assert guidance["recommended_horizon"] == 56

    ident = {
        "rank_condition": {"rank_deficient_indices": np.array([1, 2]), "satisfied": False},
        "persistence_of_excitation": {"passes": False},
    }
    edge_probs = np.array([[0.0, 0.96], [0.94, 0.0]])
    warnings = identifiability_warning_flags(ident, edge_probabilities=edge_probs)
    assert "transfer_rank_deficient_frequencies" in warnings
    assert "insufficient_persistence_of_excitation" in warnings
    assert any(flag.startswith("bidirectional_high_confidence_edge_pair") for flag in warnings)
