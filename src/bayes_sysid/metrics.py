from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
from statistics import NormalDist
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike


_STANDARD_NORMAL = NormalDist(mu=0.0, sigma=1.0)


def _as_1d_float(x: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional after reshaping.")
    return arr


def _validate_triplet(y_true: ArrayLike, mean: ArrayLike, std: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true_arr = _as_1d_float(y_true, "y_true")
    mean_arr = _as_1d_float(mean, "mean")
    std_arr = _as_1d_float(std, "std")
    if len(y_true_arr) != len(mean_arr) or len(y_true_arr) != len(std_arr):
        raise ValueError("y_true, mean, and std must have the same length.")
    if np.any(std_arr < 0):
        raise ValueError("std entries must be non-negative.")
    return y_true_arr, mean_arr, std_arr


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    y_true = _as_1d_float(y_true, "y_true")
    y_pred = _as_1d_float(y_pred, "y_pred")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    y_true = _as_1d_float(y_true, "y_true")
    y_pred = _as_1d_float(y_pred, "y_pred")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    return float(np.mean(np.abs(y_true - y_pred)))


def gaussian_nll(y_true: ArrayLike, mean: ArrayLike, var: ArrayLike) -> float:
    y_true = _as_1d_float(y_true, "y_true")
    mean = _as_1d_float(mean, "mean")
    var = _as_1d_float(var, "var")
    if len(y_true) != len(mean) or len(y_true) != len(var):
        raise ValueError("y_true, mean, and var must have the same length.")
    if np.any(var <= 0):
        raise ValueError("All variances must be positive.")

    return float(np.mean(0.5 * np.log(2.0 * np.pi * var) + 0.5 * ((y_true - mean) ** 2) / var))


def interval_coverage(y_true: ArrayLike, mean: ArrayLike, std: ArrayLike, z: float = 1.96) -> float:
    y_true, mean, std = _validate_triplet(y_true, mean, std)
    lo = mean - z * std
    hi = mean + z * std
    return float(np.mean((y_true >= lo) & (y_true <= hi)))


def z_from_nominal_level(level: float) -> float:
    if not 0.0 < level < 1.0:
        raise ValueError("level must satisfy 0 < level < 1.")
    return float(_STANDARD_NORMAL.inv_cdf(0.5 + level / 2.0))


@dataclass(frozen=True)
class CalibrationRow:
    nominal_level: float
    z_value: float
    empirical_coverage: float
    calibration_error: float
    mean_interval_width: float


@dataclass(frozen=True)
class RollingOriginNLLRow:
    na: int
    nb: int
    n_windows: int
    mean_nll: float
    std_nll: float
    min_nll: float
    max_nll: float


@dataclass(frozen=True)
class PredictiveDiagnosticsReport:
    calibration_rows: list[CalibrationRow]
    rolling_nll_rows: list[RollingOriginNLLRow]

    def calibration_records(self) -> list[dict[str, float]]:
        return [
            {
                "nominal_level": row.nominal_level,
                "z_value": row.z_value,
                "empirical_coverage": row.empirical_coverage,
                "calibration_error": row.calibration_error,
                "mean_interval_width": row.mean_interval_width,
            }
            for row in self.calibration_rows
        ]

    def rolling_nll_records(self) -> list[dict[str, float | int]]:
        return [
            {
                "na": row.na,
                "nb": row.nb,
                "n_windows": row.n_windows,
                "mean_nll": row.mean_nll,
                "std_nll": row.std_nll,
                "min_nll": row.min_nll,
                "max_nll": row.max_nll,
            }
            for row in self.rolling_nll_rows
        ]


def predictive_interval_coverage_vs_nominal(
    y_true: ArrayLike,
    mean: ArrayLike,
    std: ArrayLike,
    nominal_levels: ArrayLike,
) -> list[CalibrationRow]:
    y_true_arr, mean_arr, std_arr = _validate_triplet(y_true, mean, std)
    levels = _as_1d_float(nominal_levels, "nominal_levels")
    rows: list[CalibrationRow] = []
    for level in levels:
        z = z_from_nominal_level(float(level))
        empirical = interval_coverage(y_true_arr, mean_arr, std_arr, z=z)
        rows.append(
            CalibrationRow(
                nominal_level=float(level),
                z_value=z,
                empirical_coverage=empirical,
                calibration_error=empirical - float(level),
                mean_interval_width=float(np.mean(2.0 * z * std_arr)),
            )
        )
    return rows


def sharpness_calibration_tradeoff_table(
    y_true: ArrayLike,
    mean: ArrayLike,
    std: ArrayLike,
    nominal_levels: ArrayLike,
) -> list[dict[str, float]]:
    rows = predictive_interval_coverage_vs_nominal(y_true, mean, std, nominal_levels)
    return [
        {
            "nominal_level": row.nominal_level,
            "empirical_coverage": row.empirical_coverage,
            "abs_calibration_error": abs(row.calibration_error),
            "mean_interval_width": row.mean_interval_width,
        }
        for row in rows
    ]


def rolling_origin_nll_diagnostics(
    nll_by_order: Mapping[tuple[int, int], ArrayLike],
) -> list[RollingOriginNLLRow]:
    rows: list[RollingOriginNLLRow] = []
    for (na, nb), values in sorted(nll_by_order.items()):
        nll_values = _as_1d_float(values, f"nll_by_order[{(na, nb)}]")
        if len(nll_values) == 0:
            raise ValueError(f"nll list for order {(na, nb)} must be non-empty.")
        rows.append(
            RollingOriginNLLRow(
                na=int(na),
                nb=int(nb),
                n_windows=int(len(nll_values)),
                mean_nll=float(np.mean(nll_values)),
                std_nll=float(np.std(nll_values, ddof=0)),
                min_nll=float(np.min(nll_values)),
                max_nll=float(np.max(nll_values)),
            )
        )
    return rows


def build_predictive_diagnostics_report(
    y_true: ArrayLike,
    mean: ArrayLike,
    std: ArrayLike,
    nominal_levels: ArrayLike,
    nll_by_order: Mapping[tuple[int, int], ArrayLike],
) -> PredictiveDiagnosticsReport:
    return PredictiveDiagnosticsReport(
        calibration_rows=predictive_interval_coverage_vs_nominal(y_true, mean, std, nominal_levels),
        rolling_nll_rows=rolling_origin_nll_diagnostics(nll_by_order),
    )


def export_report_csv(
    report: PredictiveDiagnosticsReport,
    calibration_csv_path: str | Path,
    rolling_nll_csv_path: str | Path,
) -> tuple[Path, Path]:
    calibration_path = Path(calibration_csv_path)
    rolling_nll_path = Path(rolling_nll_csv_path)
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    rolling_nll_path.parent.mkdir(parents=True, exist_ok=True)

    calibration_records = report.calibration_records()
    with calibration_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["nominal_level", "z_value", "empirical_coverage", "calibration_error", "mean_interval_width"],
        )
        writer.writeheader()
        writer.writerows(calibration_records)

    rolling_records = report.rolling_nll_records()
    with rolling_nll_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["na", "nb", "n_windows", "mean_nll", "std_nll", "min_nll", "max_nll"],
        )
        writer.writeheader()
        writer.writerows(rolling_records)

    return calibration_path, rolling_nll_path
