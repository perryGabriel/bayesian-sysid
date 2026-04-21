import numpy as np

from bayes_sysid.control.realization import arx_to_state_space, minimal_realization


def _arx_freq_response(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray:
    z_inv = np.exp(-1j * w)
    num = np.zeros_like(z_inv, dtype=complex)
    den = np.ones_like(z_inv, dtype=complex)
    for j, bj in enumerate(b, start=1):
        num += bj * (z_inv**j)
    for i, ai in enumerate(a, start=1):
        den += ai * (z_inv**i)
    return num / den


def _ss_freq_response(A, B, C, D, w: np.ndarray) -> np.ndarray:
    I = np.eye(A.shape[0])
    out = np.empty_like(w, dtype=complex)
    for k, wk in enumerate(w):
        z = np.exp(1j * wk)
        out[k] = (C @ np.linalg.solve(z * I - A, B) + D)[0, 0]
    return out


def _controllability_matrix(A, B):
    n = A.shape[0]
    mats = [B]
    AB = B
    for _ in range(1, n):
        AB = A @ AB
        mats.append(AB)
    return np.hstack(mats)


def test_arx_realization_matches_frequency_response_dense_grid():
    a = np.array([0.45, -0.22, 0.06])
    b = np.array([0.12, 0.05, -0.01])

    A, B, C, D = arx_to_state_space(a=a, b=b)

    w = np.linspace(1e-3, np.pi - 1e-3, 512)
    H_arx = _arx_freq_response(a, b, w)
    H_ss = _ss_freq_response(A, B, C, D, w)

    np.testing.assert_allclose(H_ss, H_arx, rtol=1e-10, atol=1e-10)


def test_companion_form_is_fully_controllable_for_canonical_example():
    a = np.array([0.3, -0.1, 0.04, 0.01])
    b = np.array([0.2, -0.05, 0.02, 0.01])

    A, B, _, _ = arx_to_state_space(a=a, b=b)
    ctrb = _controllability_matrix(A, B)

    assert np.linalg.matrix_rank(ctrb) == A.shape[0]


def test_minimal_realization_removes_unreachable_and_unobservable_states():
    A = np.diag([0.8, 0.5, -0.1])
    B = np.array([[1.0], [0.0], [0.2]])
    C = np.array([[1.0, 0.0, 0.0]])
    D = np.array([[0.0]])

    A_m, B_m, C_m, D_m, kept = minimal_realization(A, B, C, D, tol=1e-12)

    assert A_m.shape == (1, 1)
    np.testing.assert_allclose(A_m, np.array([[0.8]]), atol=1e-12)
    np.testing.assert_allclose(B_m, np.array([[1.0]]), atol=1e-12)
    np.testing.assert_allclose(C_m, np.array([[1.0]]), atol=1e-12)
    np.testing.assert_allclose(D_m, D, atol=1e-12)
    np.testing.assert_array_equal(kept, np.array([0]))
