from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bayes_sysid import simulate_arx
from bayes_sysid.analysis.frequency_response import posterior_frequency_response_samples
from bayes_sysid.models import BayesianARX


def _closed_nyquist_curve(H_pos_freq: np.ndarray) -> np.ndarray:
    """Build an approximate full Nyquist contour from positive-frequency response."""
    if H_pos_freq.ndim != 1:
        raise ValueError("H_pos_freq must be a 1D complex array.")
    mirrored = np.conj(H_pos_freq[-2:0:-1])
    return np.concatenate([H_pos_freq, mirrored])


def _winding_number_around_minus_one(H_closed: np.ndarray) -> int:
    """Estimate integer winding number of Nyquist curve around -1+0j."""
    shifted = H_closed + 1.0
    angles = np.unwrap(np.angle(shifted))
    total_turn = (angles[-1] - angles[0]) / (2 * np.pi)
    return int(np.rint(total_turn))


def main() -> None:
    out_dir = Path("examples/artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(20)

    # A stable data-generating process.
    a_true = np.array([0.60, -0.22])
    b_true = np.array([0.45, 0.12])
    sigma = 0.14

    u = rng.normal(size=220)
    y = simulate_arx(a_true, b_true, u, sigma=sigma, random_state=21)
    u_obs = u[2:]

    model = BayesianARX(na=2, nb=2, sigma2=sigma**2).fit(y, u_obs)

    # Positive frequencies for the discrete-time Nyquist trace.
    w = np.linspace(0.0, np.pi, 450)
    H_samples = posterior_frequency_response_samples(model, w, n_samples=500, random_state=22)

    closed_samples = np.asarray([_closed_nyquist_curve(H) for H in H_samples])
    windings = np.asarray([_winding_number_around_minus_one(Hc) for Hc in closed_samples])

    # "Near-critical" means close to -1, where margin can be small.
    critical_radius = 0.50
    warn_prob_threshold = 0.005
    near_critical_prob = np.mean(np.abs(H_samples + 1.0) <= critical_radius, axis=0)

    print("Posterior Nyquist winding summary around -1:")
    unique, counts = np.unique(windings, return_counts=True)
    for wnum, count in zip(unique, counts, strict=True):
        print(f"  winding {wnum:+d}: {count}/{len(windings)} ({count / len(windings):.1%})")

    p_nonzero = float(np.mean(windings != 0))
    p_near = float(np.max(near_critical_prob))
    print(f"Probability of nonzero winding count: {p_nonzero:.3f}")
    print(f"Maximum pointwise near-critical probability (|L+1| <= {critical_radius}): {p_near:.3f}")

    # Build polar statistics in coordinates centered at -1: z = L + 1.
    shifted = H_samples + 1.0
    angle = np.angle(shifted)
    radius = np.abs(shifted)

    angle_mean = np.angle(np.mean(np.exp(1j * angle), axis=0))
    r_lo = np.quantile(radius, 0.025, axis=0)
    r_md = np.quantile(radius, 0.50, axis=0)
    r_hi = np.quantile(radius, 0.975, axis=0)

    order = np.argsort(angle_mean)
    th = angle_mean[order]

    fig = plt.figure(figsize=(12, 5))

    # Left: classic Nyquist view in the complex plane.
    ax1 = fig.add_subplot(1, 2, 1)
    for i in range(120):
        Hc = closed_samples[i]
        ax1.plot(Hc.real, Hc.imag, color="tab:blue", alpha=0.08, linewidth=1.0)

    H_med = np.median(H_samples, axis=0)
    H_med_c = _closed_nyquist_curve(H_med)
    ax1.plot(H_med_c.real, H_med_c.imag, color="tab:orange", linewidth=2.0, label="posterior median")
    ax1.scatter([-1.0], [0.0], color="red", marker=".", s=8, label="-1 point")
    ax1.set_title("Posterior Nyquist cloud")
    ax1.set_xlabel("Re")
    ax1.set_ylabel("Im")
    ax1.axhline(0.0, color="black", alpha=0.2, linewidth=0.8)
    ax1.axvline(0.0, color="black", alpha=0.2, linewidth=0.8)
    ax1.set_aspect("equal", adjustable="box")

    # set the figure bounds to be +- 50% around posterior mean region, with some margin
    x_min = np.min(H_med_c.real)*1.5
    x_max = np.max(H_med_c.real)*1.5
    y_min = np.min(H_med_c.imag)*1.5
    y_max = np.max(H_med_c.imag)*1.5
    # round up the shorter side to make the plot 1:2 aspect ratio, which is more standard for Nyquist plots
    x_range = x_max - x_min
    y_range = y_max - y_min
    if x_range < 2 * y_range:
        x_center = (x_min + x_max) / 2
        x_min = x_center - y_range
        x_max = x_center + y_range
    else:
        y_center = (y_min + y_max) / 2
        y_min = y_center - x_range / 2
        y_max = y_center + x_range / 2
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)  


    ax1.legend(fontsize=8)

    # Right: polar uncertainty band after translating by +1 (center at -1).
    ax2 = fig.add_subplot(1, 2, 2, projection="polar")
    ax2.fill_between(th, np.log10(r_lo[order]), np.log10(r_hi[order]), color="tab:purple", alpha=0.3, label="95% radial band")
    ax2.plot(th, np.log10(r_md[order]), color="tab:purple", linewidth=2.0, label="median radius")

    warn_mask = near_critical_prob[order] >= warn_prob_threshold
    if np.any(warn_mask):
        ax2.scatter(
            th[warn_mask],
            np.log10(r_md[order][warn_mask]),
            c=near_critical_prob[order][warn_mask],
            cmap="Reds",
            s=18,
            label=f"P(|L+1| <= {critical_radius}) >= {warn_prob_threshold:.3f}",
        )

    ax2.set_title("Polar band of L(e^{jw}) + 1")
    ax2.set_rlabel_position(20)
    ax2.legend(loc="lower left", bbox_to_anchor=(1.02, 0.0), fontsize=8)

    fig.suptitle(
        "Nyquist uncertainty under posterior ARX parameters\n"
        f"P(nonzero winding)={p_nonzero:.2f}, max near-critical prob={p_near:.2f}",
        fontsize=20,
    )
    fig.tight_layout()

    out_path = out_dir / "nyquist_posterior_band.png"
    fig.savefig(out_path, dpi=170)
    plt.show()

    print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    main()
