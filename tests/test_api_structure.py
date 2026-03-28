import bayes_sysid


def test_public_api_exports_present():
    expected = {
        "ARXRegressionData",
        "BayesianARX",
        "BayesianARXUnknownNoise",
        "LeastSquaresARX",
        "OrderSearchResult",
        "build_arx_regression",
        "diagonal_arx_prior_covariance",
        "isotropic_prior_covariance",
        "rolling_order_search",
        "scale_prior_covariance_by_regressor_variance",
        "gaussian_nll",
        "interval_coverage",
        "mae",
        "rmse",
        "simulate_arx",
    }
    exported = set(bayes_sysid.__all__)
    assert expected.issubset(exported)


def test_backward_compatible_arx_module_reexports():
    from bayes_sysid import arx

    assert hasattr(arx, "BayesianARX")
    assert hasattr(arx, "BayesianARXUnknownNoise")
    assert hasattr(arx, "rolling_order_search")
    assert hasattr(arx, "build_arx_regression")
