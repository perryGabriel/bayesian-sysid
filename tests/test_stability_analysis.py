import numpy as np

from bayes_sysid import BayesianARX, simulate_arx
from bayes_sysid.analysis.stability import arx_poles, is_stable_discrete, posterior_stability_probability


def test_arx_poles_and_stability_flags_known_cases():
    stable_theta = np.array([-0.4, 0.0, 1.0])
    unstable_theta = np.array([-1.2, 0.0, 1.0])

    stable_poles = arx_poles(stable_theta, na=1)
    unstable_poles = arx_poles(unstable_theta, na=1)

    assert is_stable_discrete(stable_poles)
    assert not is_stable_discrete(unstable_poles)


def test_posterior_stability_probability_is_bounded():
    rng = np.random.default_rng(10)
    u = rng.normal(size=150)
    y = simulate_arx([0.4, -0.1], [0.8, 0.2], u, sigma=0.05, random_state=9)
    model = BayesianARX(na=2, nb=2, sigma2=0.05**2).fit(y, u[2:])

    p = posterior_stability_probability(model, n_samples=100, random_state=0)
    assert 0.0 <= p <= 1.0
