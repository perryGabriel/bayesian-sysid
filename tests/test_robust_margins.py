import numpy as np

from bayes_sysid import BayesianARX, simulate_arx
from bayes_sysid.analysis.frequency_response import arx_frequency_response
from bayes_sysid.control.margins import classical_margins_from_open_loop, empirical_margin_report


def test_classical_margin_report_shapes_and_fields():
    w = np.logspace(-2, np.log10(np.pi), 200)
    theta = np.array([0.3, 0.8])  # na=1, nb=1
    L = arx_frequency_response(theta, na=1, nb=1, w=w)

    rep = classical_margins_from_open_loop(L, w)
    assert rep.gain_margin > 0
    assert np.isfinite(rep.phase_margin_deg) or np.isnan(rep.phase_margin_deg)


def test_empirical_margin_report_probability_bounds():
    rng = np.random.default_rng(2)
    u = rng.normal(size=200)
    y = simulate_arx([0.4, -0.1], [0.8, 0.2], u, sigma=0.05, random_state=7)
    model = BayesianARX(na=2, nb=2, sigma2=0.05**2).fit(y, u[2:])

    w = np.logspace(-2, np.log10(np.pi), 150)
    out = empirical_margin_report(model, w, n_samples=60, random_state=0)

    assert 0.0 <= out["p_gain_margin_gt_1"] <= 1.0
    assert 0.0 <= out["p_phase_margin_gt_0"] <= 1.0
    assert out["median_gain_margin"] > 0
