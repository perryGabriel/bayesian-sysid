"""Backward-compatible ARX API facade.

This module re-exports classes/functions that now live in focused modules:
- regression.py
- models.py
- priors.py
- selection.py
"""

from .models import BayesianARX, BayesianARXUnknownNoise, LeastSquaresARX
from .priors import (
    diagonal_arx_prior_covariance,
    isotropic_prior_covariance,
    scale_prior_covariance_by_regressor_variance,
)
from .regression import ARXRegressionData, build_arx_regression
from .selection import OrderSearchResult, rolling_order_search

__all__ = [
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
]
