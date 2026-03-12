from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bayes_sysid import (
    BayesianARX,
    BayesianARXUnknownNoise,
    LeastSquaresARX,
    gaussian_nll,
    interval_coverage,
    mae,
    rmse,
    rolling_order_search,
    simulate_arx,
)


def _recursive_rollout_bayes_mean(
    model: BayesianARX,
    y_init: np.ndarray,
    u_future: np.ndarray,
) -> np.ndarray:
    """Recursive (free-run) rollout using posterior mean parameters."""
    y_hist = list(np.asarray(y_init, dtype=float).copy())
    preds = []
    horizon = len(u_future) - model.nb + 1
    for k in range(horizon):
        phi_u = u_future[: model.nb + k]
        y_next = model.posterior_mean_prediction(np.asarray(y_hist), phi_u)
        y_hist.append(y_next)
        preds.append(y_next)
    return np.asarray(preds)


def main() -> None:
    rng = np.random.default_rng(7)
    out_dir = Path("examples/artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

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
    bayes_unknown = BayesianARXUnknownNoise(na=2, nb=2).fit(y, u_obs)

    print("True theta      :", np.r_[a_true, b_true])
    print("Least-squares   :", ls_model.theta_hat)
    print("Posterior mean  :", bayes_model.muN)
    print("Posterior std   :", np.sqrt(np.diag(bayes_model.SigmaN)))
    print("Unknown-noise posterior sigma2 mean:", f"{bayes_unknown.sigma2_posterior_mean:.5f}")

    # Rolling one-step predictions on all in-sample points (after lag).
    y_true = []
    y_ls = []
    y_bm = []
    y_bvar = []
    for t_idx in range(max_lag, len(y)):
        y_hist = y[:t_idx]
        u_hist = u_obs[:t_idx]
        y_true.append(y[t_idx])
        y_ls.append(ls_model.predict_next(y_hist, u_hist))
        m, v = bayes_model.predict_next_distribution(y_hist, u_hist)
        y_bm.append(m)
        y_bvar.append(v)

    y_true = np.asarray(y_true)
    y_ls = np.asarray(y_ls)
    y_bm = np.asarray(y_bm)
    y_bvar = np.asarray(y_bvar)

    print("\n=== In-sample one-step metrics ===")
    print(f"LS   RMSE: {rmse(y_true, y_ls):.4f}  MAE: {mae(y_true, y_ls):.4f}")
    print(
        "Bayes RMSE: "
        f"{rmse(y_true, y_bm):.4f}  MAE: {mae(y_true, y_bm):.4f}  "
        f"NLL: {gaussian_nll(y_true, y_bm, y_bvar):.4f}  "
        f"95% cov: {interval_coverage(y_true, y_bm, np.sqrt(y_bvar), z=1.96):.3f}"
    )

    order_result = rolling_order_search(
        y=y,
        u=u_obs,
        na_candidates=[1, 2, 3],
        nb_candidates=[1, 2, 3],
        train_fraction=0.7,
        sigma2=sigma2,
        metric="nll",
    )
    print(
        "Selected orders from rolling validation: "
        f"na={order_result.na}, nb={order_result.nb}, score={order_result.score:.4f}"
    )

    # One-step predictive distribution for an actually observed target at time t.
    t_star = len(y) - 1
    y_hist = y[:t_star]
    u_hist = u_obs[:t_star]
    y_target = float(y[t_star])

    ls_pred = ls_model.predict_next(y_hist, u_hist)
    post_mean, post_var = bayes_model.predict_next_distribution(y_hist, u_hist)
    post_std = np.sqrt(post_var)

    # "True model mean" for this regressor (without observation noise)
    phi_true = np.r_[y_hist[-2], y_hist[-1], u_hist[-2], u_hist[-1]]
    true_conditional_mean = float(phi_true @ np.r_[a_true, b_true])

    print(f"\nObserved next output            : {y_target:.4f}")
    print(f"True conditional mean (noiseless): {true_conditional_mean:.4f}")
    print(f"Plug-in LS one-step prediction  : {ls_pred:.4f}")
    print(f"Bayesian predictive mean        : {post_mean:.4f}")
    print(f"Bayesian predictive std         : {post_std:.4f}")

    # Predictive density plot for one next-step target with annotations.
    grid = np.linspace(post_mean - 4 * post_std, post_mean + 4 * post_std, 500)
    density = bayes_model.predictive_density_grid(y_hist, u_hist, grid)

    plt.figure(figsize=(8, 4.5))
    plt.plot(grid, density, label="Bayesian predictive density")
    plt.axvline(y_target, color="black", linestyle="-", label="Observed next output")
    plt.axvline(true_conditional_mean, color="tab:green", linestyle="-.", label="True conditional mean")
    plt.axvline(ls_pred, color="tab:orange", linestyle="--", label="LS one-step prediction")
    plt.axvline(post_mean, color="tab:blue", linestyle=":", label="Bayesian predictive mean")
    plt.fill_between(
        grid,
        0.0,
        density,
        where=(grid >= post_mean - post_std) & (grid <= post_mean + post_std),
        alpha=0.2,
        label="~68% predictive mass region",
    )
    plt.title("One-step predictive distribution (why LS/Bayes/true can all be consistent)")
    plt.xlabel("next output")
    plt.ylabel("density")
    plt.legend(fontsize=8)
    plt.tight_layout()
    pred_plot_path = out_dir / "predictive_density_annotated.png"
    plt.savefig(pred_plot_path, dpi=160)
    plt.show()
    print(f"Saved figure: {pred_plot_path}")

    # Compare one-step-ahead (teacher forcing) vs recursive (free-run) rollouts.
    horizon = 40
    start = len(y) - horizon - 1

    y_true_segment = y[start + 1 : start + 1 + horizon]
    u_segment = u_obs[start - 1 : start - 1 + horizon + 2]  # length horizon + nb
    y_init = y[start - 1 : start + 1]  # length na

    # One-step-ahead uses true history each step.
    ls_one_step = []
    bayes_one_step = []
    for k in range(horizon):
        t_idx = start + 1 + k
        ls_one_step.append(ls_model.predict_next(y[:t_idx], u_obs[:t_idx]))
        bayes_one_step.append(bayes_model.posterior_mean_prediction(y[:t_idx], u_obs[:t_idx]))
    ls_one_step = np.asarray(ls_one_step)
    bayes_one_step = np.asarray(bayes_one_step)

    # Recursive/free-run uses the model's own previous predictions.
    ls_recursive = ls_model.simulate_one_step_rollout(y_init=y_init, u=u_segment)
    bayes_recursive = _recursive_rollout_bayes_mean(bayes_model, y_init=y_init, u_future=u_segment)

    t_axis = np.arange(1, horizon + 1)
    plt.figure(figsize=(9, 5))
    plt.plot(t_axis, y_true_segment, color="black", linewidth=2, label="Observed trajectory")
    plt.plot(t_axis, ls_one_step, color="tab:orange", linestyle="--", label="LS one-step-ahead")
    plt.plot(t_axis, ls_recursive, color="tab:orange", linestyle="-", alpha=0.6, label="LS recursive")
    plt.plot(t_axis, bayes_one_step, color="tab:blue", linestyle="--", label="Bayes one-step-ahead")
    plt.plot(t_axis, bayes_recursive, color="tab:blue", linestyle="-", alpha=0.7, label="Bayes recursive")
    plt.title("One-step-ahead (teacher forcing) vs recursive (free-run) predictions")
    plt.xlabel("prediction step")
    plt.ylabel("output")
    plt.legend(fontsize=8)
    plt.tight_layout()
    rollout_compare_path = out_dir / "rollout_teacher_forcing_vs_recursive.png"
    plt.savefig(rollout_compare_path, dpi=160)
    plt.show()
    print(f"Saved figure: {rollout_compare_path}")

    # Multi-step posterior trajectory samples (parameter + process uncertainty)
    paths = bayes_model.rollout_posterior_samples(
        y_init=y_init,
        u_future=u_segment,
        n_parameter_samples=300,
        include_process_noise=True,
        random_state=10,
    )
    q10 = np.quantile(paths, 0.10, axis=0)
    q50 = np.quantile(paths, 0.50, axis=0)
    q90 = np.quantile(paths, 0.90, axis=0)

    plt.figure(figsize=(8, 4.5))
    plt.fill_between(t_axis, q10, q90, alpha=0.3, label="10%-90% posterior band")
    plt.plot(t_axis, q50, label="Posterior median trajectory")
    plt.plot(t_axis, y_true_segment, color="black", linewidth=1.5, label="Observed trajectory")
    plt.title("Posterior trajectory uncertainty (free-run)")
    plt.xlabel("prediction step")
    plt.ylabel("output")
    plt.legend(fontsize=8)
    plt.tight_layout()
    posterior_band_path = out_dir / "posterior_trajectory_band.png"
    plt.savefig(posterior_band_path, dpi=160)
    plt.show()
    print(f"Saved figure: {posterior_band_path}")


if __name__ == "__main__":
    main()
