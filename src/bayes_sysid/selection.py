from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .models import BayesianARX
from .regression import _as_1d_float


@dataclass
class OrderSearchResult:
    na: int
    nb: int
    score: float


def rolling_order_search(
    y: ArrayLike,
    u: ArrayLike,
    na_candidates: ArrayLike,
    nb_candidates: ArrayLike,
    *,
    train_fraction: float = 0.7,
    sigma2: float = 1.0,
    metric: str = "nll",
) -> OrderSearchResult:
    """Select ARX orders via rolling-origin one-step validation."""
    y_arr = _as_1d_float(y, "y")
    u_arr = _as_1d_float(u, "u")
    if len(y_arr) != len(u_arr):
        raise ValueError("y and u must have the same length.")
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must be in (0, 1).")

    na_list = [int(v) for v in np.asarray(na_candidates).reshape(-1)]
    nb_list = [int(v) for v in np.asarray(nb_candidates).reshape(-1)]
    if len(na_list) == 0 or len(nb_list) == 0:
        raise ValueError("Candidate lists must be non-empty.")

    split = int(np.floor(train_fraction * len(y_arr)))
    metric = metric.lower()
    if metric not in {"nll", "mse"}:
        raise ValueError("metric must be 'nll' or 'mse'.")

    best = OrderSearchResult(na=na_list[0], nb=nb_list[0], score=np.inf)
    for na in na_list:
        for nb in nb_list:
            max_lag = max(na, nb)
            if split <= max_lag or split >= len(y_arr):
                continue

            losses = []
            for t_idx in range(split, len(y_arr)):
                y_train = y_arr[:t_idx]
                u_train = u_arr[:t_idx]
                try:
                    model = BayesianARX(na=na, nb=nb, sigma2=sigma2).fit(y_train, u_train)
                except ValueError:
                    continue
                y_hist = y_arr[:t_idx]
                u_hist = u_arr[:t_idx]
                mean, var = model.predict_next_distribution(y_hist, u_hist)
                resid = y_arr[t_idx] - mean
                if metric == "mse":
                    losses.append(resid * resid)
                else:
                    losses.append(0.5 * np.log(2.0 * np.pi * var) + 0.5 * (resid * resid) / var)

            if len(losses) == 0:
                continue
            score = float(np.mean(losses))
            if score < best.score:
                best = OrderSearchResult(na=na, nb=nb, score=score)

    if not np.isfinite(best.score):
        raise ValueError("No valid (na, nb) candidate produced a score.")
    return best
