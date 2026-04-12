from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bayes_sysid.models import BayesianARX
from bayes_sysid import simulate_arx
from bayes_sysid.analysis.frequency_response import (
    arx_frequency_response,
    posterior_frequency_response_samples,
    posterior_magnitude_envelope,
)
from bayes_sysid.analysis.stability import arx_poles, posterior_stability_probability
from bayes_sysid.control.closed_loop import monte_carlo_closed_loop_paths
from bayes_sysid.control.margins import classical_margins_from_open_loop, empirical_margin_report
from bayes_sysid.control.tuning import tune_controller_probabilistic


def main() -> None:
    out_dir = Path("examples/artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(11)
    a_true = np.array([0.55, -0.18])
    b_true = np.array([0.9, 0.25])
    sigma = 0.15

    u = rng.normal(size=300)
    y = simulate_arx(a_true, b_true, u, sigma=sigma, random_state=2)
    u_obs = u[2:]

    model = BayesianARX(na=2, nb=2, sigma2=sigma**2).fit(y, u_obs)
    stability_domain = "discrete"

    p_stable = posterior_stability_probability(
        model,
        n_samples=1000,
        random_state=0,
        domain=stability_domain,
    )
    print(f"Posterior stability probability: {p_stable:.3f}")

    theta_samples = model.sample_parameters(300, random_state=1)
    poles = np.array([arx_poles(theta, model.na) for theta in theta_samples])

    plt.figure(figsize=(5, 5))
    th = np.linspace(0, 2 * np.pi, 400)
    if stability_domain == "discrete":
        plt.plot(np.cos(th), np.sin(th), "k--", label="|z| = 1")
    plt.scatter(poles.real.flatten(), poles.imag.flatten(), s=8, alpha=0.4, label="posterior poles")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(f"Posterior pole cloud ({stability_domain} domain)")
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.legend()
    plt.tight_layout()
    pole_path = out_dir / "stability_pole_cloud.png"
    plt.savefig(pole_path, dpi=160)
    plt.show()

    w = np.logspace(-2, np.log10(np.pi), 300)
    H_samples = posterior_frequency_response_samples(model, w, n_samples=300, random_state=3)
    q10, q50, q90 = posterior_magnitude_envelope(H_samples)

    plt.figure(figsize=(7, 4))
    plt.semilogx(w, q50, label="posterior median |H(e^jw)|")
    plt.fill_between(w, q10, q90, alpha=0.3, label="10-90% band")
    plt.xlabel("rad/sample")
    plt.ylabel("magnitude")
    plt.title("Posterior frequency-response envelope")
    plt.legend()
    plt.tight_layout()
    fr_path = out_dir / "frequency_response_envelope.png"
    plt.savefig(fr_path, dpi=160)
    plt.show()

    L_nom = arx_frequency_response(model.muN, model.na, model.nb, w)
    nominal_margins = classical_margins_from_open_loop(L_nom, w)
    empirical = empirical_margin_report(model, w, n_samples=200, random_state=4)
    print("Nominal margins:", nominal_margins)
    print("Empirical margin summary:", empirical)

    r = np.ones(120)

    tuning_report = tune_controller_probabilistic(
        model=model,
        r=r,
        controller="pid",
        param_bounds={"kp": (0.0, 2.0), "ki": (0.0, 0.4), "kd": (0.0, 0.3)},
        n_iterations=60,
        n_parameter_samples=120,
        output_bound=4.0,
        input_bound=3.0,
        robustness_delta=0.25,
        include_process_noise=True,
        random_state=8,
    )
    print("Tuning report:", tuning_report)

    y_paths, _ = monte_carlo_closed_loop_paths(
        model,
        r,
        n_parameter_samples=200,
        controller="static",
        controller_params={"k": 0.5, "u_min": -2.0, "u_max": 2.0},
        include_process_noise=True,
        random_state=6,
    )
    q05 = np.quantile(y_paths, 0.05, axis=0)
    q50 = np.quantile(y_paths, 0.50, axis=0)
    q95 = np.quantile(y_paths, 0.95, axis=0)
    t = np.arange(1, len(q50) + 1)

    plt.figure(figsize=(7, 4))
    plt.fill_between(t, q05, q95, alpha=0.25, label="5-95% closed-loop band")
    plt.plot(t, q50, label="median output")
    plt.plot(t, r[2:], "k--", label="reference")
    plt.xlabel("step")
    plt.ylabel("output")
    plt.title("Closed-loop Monte Carlo under posterior uncertainty")
    plt.legend()
    plt.tight_layout()
    cl_path = out_dir / "closed_loop_monte_carlo.png"
    plt.savefig(cl_path, dpi=160)
    plt.show()

    print("Saved figures:")
    print(" -", pole_path)
    print(" -", fr_path)
    print(" -", cl_path)


if __name__ == "__main__":
    main()
