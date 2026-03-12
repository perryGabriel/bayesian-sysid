from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from bayes_sysid import BayesianARX, LeastSquaresARX, simulate_arx


def main() -> None:
    rng = np.random.default_rng(7)

    # True stable ARX(2,2) system
    a_true = np.array([0.55, -0.18])
    b_true = np.array([0.90, 0.25])
    sigma = 0.20
    sigma2 = sigma**2

    T = 260
    u = rng.normal(size=T)
    y = simulate_arx(a_true, b_true, u, sigma=sigma, random_state=3)

    # Because simulate_arx returns outputs beginning at max_lag,
    # align the input to the observed y sequence length.
    max_lag = 2
    u_obs = u[max_lag:]

    # Fit models
    ls_model = LeastSquaresARX(na=2, nb=2).fit(y, u_obs)
    bayes_model = BayesianARX(na=2, nb=2, sigma2=sigma2).fit(y, u_obs)

    print("True theta      :", np.r_[a_true, b_true])
    print("Least-squares   :", ls_model.theta_hat)
    print("Posterior mean  :", bayes_model.muN)
    print("Posterior std    :", np.sqrt(np.diag(bayes_model.SigmaN)))

    # Hold out one next-step target using final histories.
    y_hist = y[-2:]
    u_hist = u_obs[-2:]

    ls_pred = ls_model.predict_next(y_hist, u_hist)
    post_mean, post_var = bayes_model.predict_next_distribution(y_hist, u_hist)
    post_std = np.sqrt(post_var)

    print(f"\nPlug-in LS next prediction      : {ls_pred:.4f}")
    print(f"Bayesian posterior mean         : {post_mean:.4f}")
    print(f"Bayesian predictive std         : {post_std:.4f}")

    # Predictive density plot for the next output.
    grid = np.linspace(post_mean - 4 * post_std, post_mean + 4 * post_std, 400)
    density = bayes_model.predictive_density_grid(y_hist, u_hist, grid)

    plt.figure(figsize=(7, 4))
    plt.plot(grid, density)
    plt.axvline(ls_pred, linestyle="--", label="LS plug-in prediction")
    plt.axvline(post_mean, linestyle=":", label="Bayesian predictive mean")
    plt.title("Posterior predictive density for next output")
    plt.xlabel("next output")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Multi-step posterior trajectory samples
    horizon = 40
    u_future = rng.normal(size=horizon + 2)
    y_init = y[-2:]
    paths = bayes_model.rollout_posterior_samples(
        y_init=y_init,
        u_future=u_future,
        n_parameter_samples=300,
        include_process_noise=True,
        random_state=10,
    )
    q10 = np.quantile(paths, 0.10, axis=0)
    q50 = np.quantile(paths, 0.50, axis=0)
    q90 = np.quantile(paths, 0.90, axis=0)

    ls_rollout = ls_model.simulate_one_step_rollout(y_init=y_init, u=u_future)
    t = np.arange(1, len(q50) + 1)

    plt.figure(figsize=(7, 4))
    plt.fill_between(t, q10, q90, alpha=0.3, label="10%-90% posterior band")
    plt.plot(t, q50, label="posterior median trajectory")
    plt.plot(t, ls_rollout, linestyle="--", label="LS plug-in rollout")
    plt.title("Posterior-sampled rollout under parameter uncertainty")
    plt.xlabel("prediction step")
    plt.ylabel("output")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
