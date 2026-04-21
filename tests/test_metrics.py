import numpy as np
import pytest

from bayes_sysid import (
    build_predictive_diagnostics_report,
    export_report_csv,
    gaussian_nll,
    interval_coverage,
    mae,
    predictive_interval_coverage_vs_nominal,
    rmse,
    rolling_origin_nll_diagnostics,
    sharpness_calibration_tradeoff_table,
    z_from_nominal_level,
)


def test_rmse_and_mae_values():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 1.0, 5.0])
    assert np.isclose(rmse(y_true, y_pred), np.sqrt(5.0 / 3.0))
    assert np.isclose(mae(y_true, y_pred), 1.0)


def test_gaussian_nll_and_interval_coverage_values():
    y = np.array([0.0, 1.0, 2.0])
    mean = np.array([0.0, 1.0, 2.0])
    var = np.array([1.0, 1.0, 1.0])
    nll = gaussian_nll(y, mean, var)
    assert np.isclose(nll, 0.5 * np.log(2.0 * np.pi))

    cov = interval_coverage(y, mean, np.array([0.1, 0.1, 0.1]), z=0.0)
    assert np.isclose(cov, 1.0)


def test_calibration_identity_on_simulated_gaussian_data():
    rng = np.random.default_rng(123)
    n = 30_000
    mean = rng.normal(loc=0.2, scale=0.5, size=n)
    std = np.exp(rng.normal(loc=-0.2, scale=0.2, size=n))
    y = mean + std * rng.normal(size=n)

    nominal_levels = np.array([0.50, 0.80, 0.90, 0.95])
    rows = predictive_interval_coverage_vs_nominal(y, mean, std, nominal_levels)

    for row in rows:
        assert abs(row.empirical_coverage - row.nominal_level) < 0.02
        assert row.mean_interval_width > 0.0
        assert np.isclose(row.z_value, z_from_nominal_level(row.nominal_level))

    tradeoff_rows = sharpness_calibration_tradeoff_table(y, mean, std, nominal_levels)
    assert [r["nominal_level"] for r in tradeoff_rows] == list(nominal_levels)
    assert tradeoff_rows[0]["mean_interval_width"] < tradeoff_rows[-1]["mean_interval_width"]


def test_rolling_origin_nll_diagnostics_report_and_csv_export(tmp_path):
    nll_by_order = {
        (1, 1): [1.0, 1.2, 1.1],
        (2, 2): [0.8, 0.9, 1.0],
    }
    rows = rolling_origin_nll_diagnostics(nll_by_order)
    assert len(rows) == 2
    assert rows[0].na == 1 and rows[0].nb == 1
    assert np.isclose(rows[1].mean_nll, 0.9)

    y = np.array([0.0, 0.1, -0.1, 0.05])
    mean = np.array([0.0, 0.1, -0.1, 0.05])
    std = np.ones_like(y) * 0.2
    report = build_predictive_diagnostics_report(
        y_true=y,
        mean=mean,
        std=std,
        nominal_levels=np.array([0.8, 0.95]),
        nll_by_order=nll_by_order,
    )
    assert len(report.calibration_rows) == 2
    assert len(report.rolling_nll_rows) == 2

    calibration_csv, rolling_csv = export_report_csv(
        report,
        calibration_csv_path=tmp_path / "calibration.csv",
        rolling_nll_csv_path=tmp_path / "rolling_nll.csv",
    )
    assert calibration_csv.exists()
    assert rolling_csv.exists()
    assert "nominal_level" in calibration_csv.read_text(encoding="utf-8")
    assert "mean_nll" in rolling_csv.read_text(encoding="utf-8")


def test_metrics_validation_errors():
    with pytest.raises(ValueError):
        rmse([1], [1, 2])
    with pytest.raises(ValueError):
        mae([1], [1, 2])
    with pytest.raises(ValueError):
        gaussian_nll([1], [1], [0])
    with pytest.raises(ValueError):
        gaussian_nll([1], [1, 2], [1, 2])
    with pytest.raises(ValueError):
        interval_coverage([1], [1], [-1])
    with pytest.raises(ValueError):
        interval_coverage([1], [1, 2], [1, 2])
    with pytest.raises(ValueError):
        z_from_nominal_level(0.0)
    with pytest.raises(ValueError):
        rolling_origin_nll_diagnostics({(1, 1): []})
