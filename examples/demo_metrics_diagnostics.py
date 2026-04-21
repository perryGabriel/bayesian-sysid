from __future__ import annotations

from pathlib import Path

import numpy as np

from bayes_sysid import (
    build_predictive_diagnostics_report,
    export_report_csv,
    sharpness_calibration_tradeoff_table,
)


def main() -> None:
    rng = np.random.default_rng(2026)
    out_dir = Path("examples/artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    n = 4000
    mean = rng.normal(loc=0.0, scale=0.4, size=n)
    std = np.exp(rng.normal(loc=-0.3, scale=0.15, size=n))
    y_true = mean + std * rng.normal(size=n)

    nominal_levels = np.array([0.5, 0.8, 0.9, 0.95])
    nll_by_order = {
        (1, 1): rng.normal(loc=1.10, scale=0.08, size=24),
        (2, 2): rng.normal(loc=0.97, scale=0.06, size=24),
        (3, 3): rng.normal(loc=0.99, scale=0.07, size=24),
    }

    report = build_predictive_diagnostics_report(
        y_true=y_true,
        mean=mean,
        std=std,
        nominal_levels=nominal_levels,
        nll_by_order=nll_by_order,
    )

    calibration_csv, rolling_nll_csv = export_report_csv(
        report,
        calibration_csv_path=out_dir / "predictive_calibration_table.csv",
        rolling_nll_csv_path=out_dir / "rolling_origin_nll_by_order.csv",
    )

    print("=== Predictive interval coverage vs nominal ===")
    for row in report.calibration_rows:
        print(
            f"level={row.nominal_level:.2f} | empirical={row.empirical_coverage:.3f} "
            f"| error={row.calibration_error:+.3f} | width={row.mean_interval_width:.3f}"
        )

    print("\n=== Sharpness vs calibration tradeoff ===")
    for row in sharpness_calibration_tradeoff_table(y_true, mean, std, nominal_levels):
        print(
            f"level={row['nominal_level']:.2f} | abs_error={row['abs_calibration_error']:.3f} "
            f"| width={row['mean_interval_width']:.3f}"
        )

    print("\n=== Rolling-origin aggregated NLL diagnostics ===")
    for row in report.rolling_nll_rows:
        print(
            f"(na, nb)=({row.na}, {row.nb}) | windows={row.n_windows} | "
            f"mean={row.mean_nll:.3f} | std={row.std_nll:.3f}"
        )

    print(f"\nWrote: {calibration_csv}")
    print(f"Wrote: {rolling_nll_csv}")


if __name__ == "__main__":
    main()
