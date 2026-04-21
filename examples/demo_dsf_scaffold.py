"""Demonstrate the prototype DSF scaffold on a 2x2 synthetic network."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bayes_sysid.control.dsf import (
    dsf_from_transfer_matrix,
    posterior_edge_probability,
    validate_excitation_richness,
    validate_identifiability_assumptions,
)


def build_synthetic_transfer(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z_inv = np.exp(-1j * w)
    p1 = 0.8 * z_inv / (1 - 0.25 * z_inv)
    p2 = 0.6 * z_inv / (1 - 0.15 * z_inv)
    q12 = 0.35 * z_inv / (1 - 0.2 * z_inv)

    Q = np.zeros((w.size, 2, 2), dtype=complex)
    P = np.zeros((w.size, 2, 2), dtype=complex)
    G = np.zeros((w.size, 2, 2), dtype=complex)

    eye = np.eye(2, dtype=complex)
    for k in range(w.size):
        Q[k, 0, 1] = q12[k]  # edge y2 -> y1
        P[k, 0, 0] = p1[k]
        P[k, 1, 1] = p2[k]
        G[k] = np.linalg.inv(eye - Q[k]) @ P[k]

    return G, Q


def main() -> None:
    out_dir = Path("examples/artifacts/dsf")
    out_dir.mkdir(parents=True, exist_ok=True)

    w = np.linspace(0.05, np.pi - 0.05, 256)
    G, Q_true = build_synthetic_transfer(w)

    dsf = dsf_from_transfer_matrix(G)
    Q_hat = dsf["Q"]

    rng = np.random.default_rng(42)
    n_samples = 300
    dsf_samples = np.repeat(Q_hat[None, ...], n_samples, axis=0)
    dsf_samples += 0.012 * (rng.standard_normal(dsf_samples.shape) + 1j * rng.standard_normal(dsf_samples.shape))
    edge_probs = posterior_edge_probability(dsf_samples, threshold=0.05)

    ident = validate_identifiability_assumptions(G)
    u = rng.normal(size=(500, 2))
    excitation = validate_excitation_richness(u)

    np.savetxt(out_dir / "edge_probability_table.csv", edge_probs, delimiter=",", fmt="%.6f")

    with (out_dir / "equations.md").open("w", encoding="utf-8") as f:
        f.write("# DSF prototype equations\n\n")
        f.write("- Transfer matrix from ARX lag-polynomials:\n")
        f.write("  $G(z) = A(z)^{-1}B(z)$, where $A(z)=I+\\sum_{k=1}^{n_a}A_k z^{-k}$ and $B(z)=\\sum_{k=1}^{n_b}B_k z^{-k}$.\n")
        f.write("- Prototype DSF factorization:\n")
        f.write("  $G(z)\\approx (I-Q(z))^{-1}P(z)$ with $P(z)=\\mathrm{diag}(G(z))$ and $Q(z)=I-P(z)G(z)^{-1}$.\n")

    with (out_dir / "validation_report.txt").open("w", encoding="utf-8") as f:
        f.write("Identifiability checks:\n")
        for k, v in ident.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nExcitation richness checks:\n")
        for k, v in excitation.items():
            f.write(f"  {k}: {v}\n")

    # Save frequency-wise edge magnitudes as CSV (binary artifacts are intentionally avoided).
    np.savetxt(
        out_dir / "edge_magnitude_recovery.csv",
        np.column_stack([w, np.abs(Q_true[:, 0, 1]), np.abs(Q_hat[:, 0, 1]), np.abs(Q_hat[:, 1, 0])]),
        delimiter=",",
        header="omega_rad_per_sample,abs_q_true_1_from_2,abs_q_hat_1_from_2,abs_q_hat_2_from_1",
        comments="",
        fmt="%.8f",
    )

    print(f"Saved DSF artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
