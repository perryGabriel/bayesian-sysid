import numpy as np
import pytest

from bayes_sysid import BayesianARX, simulate_arx
from bayes_sysid.analysis.frequency_response import (
    arx_frequency_response,
    posterior_frequency_response_samples,
    posterior_magnitude_envelope,
)


def test_arx_frequency_response_shape_and_dc_gain():
    theta = np.array([0.0, 1.0])  # na=1, nb=1
    w = np.array([0.0, np.pi / 2])
    H = arx_frequency_response(theta, na=1, nb=1, w=w)
    assert H.shape == w.shape
    assert np.isclose(H[0].real, 1.0)


def test_posterior_frequency_response_and_envelope():
    rng = np.random.default_rng(1)
    u = rng.normal(size=160)
    y = simulate_arx([0.4, -0.1], [0.7, 0.2], u, sigma=0.05, random_state=2)
    model = BayesianARX(na=2, nb=2, sigma2=0.05**2).fit(y, u[2:])

    w = np.linspace(0.05, np.pi, 50)
    Hs = posterior_frequency_response_samples(model, w, n_samples=40, random_state=3)
    q10, q50, q90 = posterior_magnitude_envelope(Hs)

    assert Hs.shape == (40, 50)
    assert q10.shape == q50.shape == q90.shape == (50,)
    assert np.all(q10 <= q50)
    assert np.all(q50 <= q90)


def test_frequency_response_validation():
    with pytest.raises(ValueError):
        arx_frequency_response(np.array([1.0]), na=1, nb=1, w=np.array([1.0]))
    with pytest.raises(ValueError):
        posterior_magnitude_envelope(np.ones(3))
