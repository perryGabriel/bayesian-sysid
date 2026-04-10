import numpy as np

from bayes_sysid import BayesianARX, simulate_arx
from bayes_sysid.control.tuning import estimate_closed_loop_stability_probability, tune_controller_probabilistic


def _fit_demo_model() -> BayesianARX:
    rng = np.random.default_rng(12)
    u = rng.normal(size=220)
    y = simulate_arx([0.45, -0.12], [0.8, 0.15], u, sigma=0.06, random_state=7)
    return BayesianARX(na=2, nb=2, sigma2=0.06**2).fit(y, u[2:])


def test_estimate_closed_loop_stability_probability_outputs_valid_statistics():
    model = _fit_demo_model()
    estimate = estimate_closed_loop_stability_probability(
        model=model,
        r=np.ones(80),
        controller="static",
        controller_params={"k": 0.4},
        n_parameter_samples=50,
        output_bound=5.0,
        input_bound=3.0,
        robustness_delta=0.1,
        include_process_noise=False,
        random_state=0,
    )

    assert 0.0 <= estimate.probability <= 1.0
    assert estimate.std_error >= 0.0
    assert 0.0 <= estimate.ci95_low <= estimate.ci95_high <= 1.0
    assert estimate.n_samples == 50


def test_tune_controller_probabilistic_returns_best_report():
    model = _fit_demo_model()
    report = tune_controller_probabilistic(
        model=model,
        r=np.ones(80),
        controller="pid",
        param_bounds={"kp": (0.0, 1.5), "ki": (0.0, 0.3), "kd": (0.0, 0.2)},
        n_iterations=10,
        n_parameter_samples=40,
        output_bound=5.0,
        input_bound=3.0,
        robustness_delta=0.2,
        include_process_noise=False,
        random_state=10,
    )

    assert set(report.best_params) == {"kp", "ki", "kd"}
    assert report.n_evaluations == 10
    assert report.n_parameter_samples == 40
    assert 0.0 <= report.best_stability_probability <= 1.0
    lo, hi = report.best_probability_ci95
    assert 0.0 <= lo <= hi <= 1.0
