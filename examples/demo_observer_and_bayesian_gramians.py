from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from bayes_sysid import BayesianARX, simulate_arx
from bayes_sysid.control.gramians import posterior_hsv_summary
from bayes_sysid.control.observer import design_luenberger_gain, run_kalman_filter
from bayes_sysid.control.realization import arx_to_state_space


def _is_discrete_stable(A: np.ndarray, tol: float = 1e-9) -> bool:
    return np.all(np.abs(np.linalg.eigvals(A)) < 1.0 - tol)


def main() -> None:
    out_dir = Path("examples/artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(21)

    # Simulate data and fit Bayesian ARX.
    a_true = np.array([0.55, -0.2])
    b_true = np.array([0.95, 0.2])
    sigma = 0.12

    u_id = rng.normal(size=360)
    y_id = simulate_arx(a_true, b_true, u_id, sigma=sigma, random_state=12)
    model = BayesianARX(na=2, nb=2, sigma2=sigma**2).fit(y_id, u_id[2:])

    # Posterior realization/Gramian ensemble via control.gramians APIs.
    n_samples = 500
    hsv_summary = posterior_hsv_summary(model, n_samples=n_samples, random_state=4)

    theta_samples = model.sample_parameters(n_samples=n_samples, random_state=4)
    stable_poles: list[np.ndarray] = []
    for theta in theta_samples:
        A, _, _, _ = arx_to_state_space(theta[: model.na], theta[model.na : model.na + model.nb])
        if _is_discrete_stable(A):
            stable_poles.append(np.linalg.eigvals(A))

    n_stable = int(hsv_summary["n_stable"])
    print(f"Stable posterior realizations: {n_stable}/{n_samples}")

    # Pole-placement + Kalman demonstration on posterior mean realization.
    A_mu, B_mu, C_mu, _ = arx_to_state_space(model.muN[: model.na], model.muN[model.na : model.na + model.nb])

    desired_poles = np.array([0.2, 0.3])
    L = design_luenberger_gain(A_mu, C_mu, desired_poles)
    eig_obs = np.linalg.eigvals(A_mu - L @ C_mu)

    T = 220
    u = rng.normal(0.0, 0.8, size=(T, 1))
    Q = np.diag([2e-3, 2e-3])
    R = np.array([[3e-2]])

    x_true = np.zeros((T, A_mu.shape[0]))
    y = np.zeros((T, C_mu.shape[0]))
    xk = np.zeros(A_mu.shape[0])

    for k in range(T):
        wk = rng.multivariate_normal(np.zeros(A_mu.shape[0]), Q)
        vk = rng.multivariate_normal(np.zeros(C_mu.shape[0]), R)
        xk = A_mu @ xk + B_mu @ u[k] + wk
        x_true[k] = xk
        y[k] = C_mu @ xk + vk

    x_kf, _ = run_kalman_filter(A_mu, B_mu, C_mu, Q, R, u, y)

    x_ol = np.zeros_like(x_true)
    x_ol_k = np.zeros(A_mu.shape[0])
    for k in range(T):
        x_ol_k = A_mu @ x_ol_k + B_mu @ u[k]
        x_ol[k] = x_ol_k

    rmse_kf = float(np.sqrt(np.mean((x_kf - x_true) ** 2)))
    rmse_open = float(np.sqrt(np.mean((x_ol - x_true) ** 2)))

    # Figure 1: posterior pole cloud.
    plt.figure(figsize=(5, 5))
    th = np.linspace(0, 2 * np.pi, 400)
    plt.plot(np.cos(th), np.sin(th), "k--", label="|z|=1")
    if stable_poles:
        poles = np.asarray(stable_poles)
        plt.scatter(poles.real.flatten(), poles.imag.flatten(), s=8, alpha=0.35, label="stable posterior poles")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.title("Bayesian realization pole cloud")
    plt.legend(fontsize=8)
    plt.tight_layout()
    pole_path = out_dir / "observer_demo_pole_cloud.png"
    plt.savefig(pole_path, dpi=160)
    plt.close()

    # Figure 2: Bayesian HSV quantiles from posterior summary API.
    hsv_q = np.asarray(hsv_summary["hsv_quantiles"], dtype=float)
    modes = np.arange(1, hsv_q.shape[1] + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(modes, hsv_q[1], marker="o", label="median HSV")
    plt.fill_between(modes, hsv_q[0], hsv_q[2], alpha=0.25, label="10-90% band")
    plt.xlabel("mode index")
    plt.ylabel("HSV")
    plt.title("Posterior Hankel singular value quantiles")
    plt.legend()
    plt.tight_layout()
    gramian_path = out_dir / "observer_demo_gramian_traces.png"
    plt.savefig(gramian_path, dpi=160)
    plt.close()

    # Figure 3: KF vs open-loop estimate.
    plt.figure(figsize=(8, 4.5))
    plt.plot(x_true[:, 0], color="black", linewidth=1.8, label="true x[0]")
    plt.plot(x_ol[:, 0], linestyle="--", label="open-loop predictor x[0]")
    plt.plot(x_kf[:, 0], linestyle=":", linewidth=2.0, label="Kalman estimate x[0]")
    plt.title("State-estimation baseline: Kalman vs open-loop")
    plt.xlabel("time step")
    plt.ylabel("state")
    plt.legend()
    plt.tight_layout()
    est_path = out_dir / "observer_demo_kalman_vs_open_loop.png"
    plt.savefig(est_path, dpi=160)
    plt.close()

    # Table outputs.
    summary_rows = [
        ("stable_sample_ratio", float(hsv_summary["stable_fraction"])),
        ("hsv_mode1_median", float(hsv_q[1, 0])),
        ("hsv_mode2_median", float(hsv_q[1, 1] if hsv_q.shape[1] > 1 else hsv_q[1, 0])),
        ("hsv_mode1_q10", float(hsv_q[0, 0])),
        ("hsv_mode1_q90", float(hsv_q[2, 0])),
        ("Wc_cond_median", float(hsv_summary["gramian_diagnostics"]["Wc"]["median_condition_number"])),
        ("Wo_cond_median", float(hsv_summary["gramian_diagnostics"]["Wo"]["median_condition_number"])),
        ("observer_pole_1_real", float(np.real(eig_obs[0]))),
        ("observer_pole_2_real", float(np.real(eig_obs[1]))),
        ("rmse_open_loop", rmse_open),
        ("rmse_kalman", rmse_kf),
        ("rmse_improvement_percent", 100.0 * (rmse_open - rmse_kf) / rmse_open),
    ]

    table_path = out_dir / "observer_demo_summary.csv"
    with table_path.open("w", encoding="utf-8") as f:
        f.write("metric,value\n")
        for k, v in summary_rows:
            f.write(f"{k},{v:.8f}\n")

    eq_path = out_dir / "observer_demo_equations.md"
    with eq_path.open("w", encoding="utf-8") as f:
        f.write("# Observer/Bayesian Gramian equations\n\n")
        f.write("Posterior realization samples: $(A^{(s)}, B^{(s)}, C^{(s)}, D^{(s)})$.\n\n")
        f.write("$W_c^{(s)} = A^{(s)} W_c^{(s)} A^{(s)\\top} + B^{(s)} B^{(s)\\top}$.\n\n")
        f.write("$W_o^{(s)} = A^{(s)\\top} W_o^{(s)} A^{(s)} + C^{(s)\\top} C^{(s)}$.\n\n")
        f.write("Kalman update: $\\hat{x}_{k|k} = \\hat{x}_{k|k-1} + K_k(y_k - C\\hat{x}_{k|k-1})$.\n")

    print("Observer/Bayesian Gramian demo outputs:")
    print(" -", pole_path)
    print(" -", gramian_path)
    print(" -", est_path)
    print(" -", table_path)
    print(" -", eq_path)
    print(f"Observer poles achieved: {np.sort_complex(eig_obs)}")
    print(f"RMSE open-loop={rmse_open:.4f}, Kalman={rmse_kf:.4f}")


if __name__ == "__main__":
    main()
