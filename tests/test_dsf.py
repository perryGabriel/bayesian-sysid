import numpy as np

from bayes_sysid.control.dsf import (
    dsf_from_transfer_matrix,
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
