from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bayes_sysid import OnlineBayesianARX


def simulate_drifting_arx(u: np.ndarray, sigma: float, random_state: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """ARX(2,2) with slowly drifting first AR coefficient."""
    rng = np.random.default_rng(random_state)
    na, nb = 2, 2
    max_lag = 2
    y_hist = [0.0, 0.0]
    y = []
    theta_trace = []

    for t in range(max_lag, len(u)):
        drift = 0.45 + 0.25 * np.sin(2 * np.pi * (t - max_lag) / 320.0)
        theta_t = np.array([drift, -0.18, 0.9, 0.25])
        phi_t = np.array([y_hist[-1], y_hist[-2], u[t - 1], u[t - 2]])
        y_t = float(phi_t @ theta_t + rng.normal(0.0, sigma))
        y_hist.append(y_t)
        y.append(y_t)
        theta_trace.append(theta_t)

    return np.asarray(y), np.asarray(theta_trace)


def main() -> None:
    rng = np.random.default_rng(12)
    out_dir = Path("examples/artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    T = 1200
    sigma = 0.09
    sigma2 = sigma**2
    u_full = rng.normal(size=T)
    y, theta_true = simulate_drifting_arx(u_full, sigma=sigma, random_state=13)
    u = u_full[2:]  # align with y

    slow = OnlineBayesianARX(na=2, nb=2, sigma2=sigma2, forgetting_factor=1.0, snapshot_stride=75)
    fast = OnlineBayesianARX(na=2, nb=2, sigma2=sigma2, forgetting_factor=0.985, snapshot_stride=75)

    mu_slow = []
    mu_fast = []
    for y_t, u_t in zip(y, u):
        slow.update(y_t, u_t)
        fast.update(y_t, u_t)
        mu_slow.append(slow.mu.copy())
        mu_fast.append(fast.mu.copy())

    mu_slow = np.asarray(mu_slow)
    mu_fast = np.asarray(mu_fast)

    print("Final posterior mean (no forgetting):", slow.mu)
    print("Final posterior mean (forgetting=0.985):", fast.mu)
    print("Stored snapshots (no forgetting):", len(slow.snapshots))
    print("Stored snapshots (forgetting):", len(fast.snapshots))

    t = np.arange(len(y))
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    ax[0].plot(t, theta_true[:, 0], color="black", linewidth=2, label="True a1(t)")
    ax[0].plot(t, mu_slow[:, 0], linestyle="--", label="Online mean a1, λ=1.0")
    ax[0].plot(t, mu_fast[:, 0], linestyle="-.", label="Online mean a1, λ=0.985")
    ax[0].set_ylabel("coefficient value")
    ax[0].set_title("Drift tracking with online Bayesian ARX")
    ax[0].legend(fontsize=8)

    ax[1].plot(t, y, color="tab:gray", linewidth=1.2, label="Observed output y")
    ax[1].set_xlabel("time index")
    ax[1].set_ylabel("output")
    ax[1].legend(fontsize=8)

    fig.tight_layout()
    fig_path = out_dir / "online_arx_drift_tracking.png"
    fig.savefig(fig_path, dpi=160)
    plt.show()
    print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    main()
