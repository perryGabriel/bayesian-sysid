import numpy as np
import pytest

from bayes_sysid import gaussian_nll, interval_coverage, mae, rmse


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
