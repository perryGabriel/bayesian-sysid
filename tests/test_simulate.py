import numpy as np
import pytest

from bayes_sysid import simulate_arx


def test_simulate_arx_basic_shape_and_determinism():
    u = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y1 = simulate_arx(a=[0.2], b=[1.0], u=u, sigma=0.1, random_state=3)
    y2 = simulate_arx(a=[0.2], b=[1.0], u=u, sigma=0.1, random_state=3)
    assert y1.shape == (4,)
    np.testing.assert_allclose(y1, y2)


def test_simulate_arx_no_noise_matches_manual_recursion():
    a = np.array([0.5])
    b = np.array([1.0])
    u = np.array([0.0, 2.0, -1.0, 3.0, 1.5])
    y = simulate_arx(a=a, b=b, u=u, sigma=0.0)

    expected = []
    y_prev = 0.0
    for t in range(1, len(u)):
        y_t = 0.5 * y_prev + 1.0 * u[t - 1]
        expected.append(y_t)
        y_prev = y_t
    np.testing.assert_allclose(y, np.asarray(expected))


def test_simulate_arx_validates_y_init_length():
    with pytest.raises(ValueError):
        simulate_arx(a=[0.1, 0.2], b=[1.0], u=np.ones(5), y_init=[0.0])
