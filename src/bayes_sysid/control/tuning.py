from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bayes_sysid.control.closed_loop import ControllerType, monte_carlo_closed_loop_paths
from bayes_sysid.models import BayesianARX


@dataclass
class StabilityEstimate:
    probability: float
    std_error: float
    ci95_low: float
    ci95_high: float
    n_stable: int
    n_samples: int


@dataclass
class TuningReport:
    best_params: dict[str, float]
    best_stability_probability: float
    best_probability_std_error: float
    best_probability_ci95: tuple[float, float]
    n_evaluations: int
    n_parameter_samples: int
    controller: ControllerType
    robustness_delta: float


def _default_param_bounds(controller: ControllerType) -> dict[str, tuple[float, float]]:
    if controller == "static":
        return {"k": (-2.0, 2.0)}
    if controller == "pid":
        return {
            "kp": (0.0, 3.0),
            "ki": (0.0, 1.0),
            "kd": (0.0, 1.0),
        }
    raise ValueError(f"Unsupported controller type: {controller!r}.")


def _sample_uniform_params(
    rng: np.random.Generator,
    param_bounds: dict[str, tuple[float, float]],
) -> dict[str, float]:
    sample: dict[str, float] = {}
    for name, (low, high) in param_bounds.items():
        if low > high:
            raise ValueError(f"Invalid bounds for {name!r}: ({low}, {high}).")
        sample[name] = float(rng.uniform(low, high))
    return sample


def _stability_flags(
    y_paths: np.ndarray,
    u_paths: np.ndarray,
    output_bound: float,
    input_bound: float | None,
    robustness_delta: float,
) -> np.ndarray:
    if output_bound <= 0:
        raise ValueError("output_bound must be positive.")
    if robustness_delta < 0:
        raise ValueError("robustness_delta must be non-negative.")

    y_limit = output_bound - robustness_delta
    if y_limit <= 0:
        raise ValueError("robustness_delta is too large for output_bound.")

    stable = np.isfinite(y_paths).all(axis=1) & (np.max(np.abs(y_paths), axis=1) <= y_limit)

    if input_bound is not None:
        if input_bound <= 0:
            raise ValueError("input_bound must be positive when provided.")
        u_limit = input_bound - robustness_delta
        if u_limit <= 0:
            raise ValueError("robustness_delta is too large for input_bound.")
        stable &= np.isfinite(u_paths).all(axis=1) & (np.max(np.abs(u_paths), axis=1) <= u_limit)

    return stable


def estimate_closed_loop_stability_probability(
    model: BayesianARX,
    r: np.ndarray,
    controller: ControllerType,
    controller_params: dict[str, float],
    n_parameter_samples: int = 200,
    output_bound: float = 10.0,
    input_bound: float | None = None,
    robustness_delta: float = 0.0,
    include_process_noise: bool = True,
    random_state: int | None = None,
) -> StabilityEstimate:
    """Estimate P(stable closed-loop | D, controller_params) from Monte Carlo rollouts."""
    y_paths, u_paths = monte_carlo_closed_loop_paths(
        model=model,
        r=r,
        n_parameter_samples=n_parameter_samples,
        controller=controller,
        controller_params=controller_params,
        include_process_noise=include_process_noise,
        random_state=random_state,
    )
    stable = _stability_flags(y_paths, u_paths, output_bound, input_bound, robustness_delta)
    n_stable = int(stable.sum())
    n_samples = int(stable.size)
    p_hat = float(n_stable / n_samples)
    std_error = float(np.sqrt(max(p_hat * (1.0 - p_hat), 0.0) / n_samples))
    z = 1.96
    ci_low = max(0.0, p_hat - z * std_error)
    ci_high = min(1.0, p_hat + z * std_error)
    return StabilityEstimate(
        probability=p_hat,
        std_error=std_error,
        ci95_low=float(ci_low),
        ci95_high=float(ci_high),
        n_stable=n_stable,
        n_samples=n_samples,
    )


def tune_controller_probabilistic(
    model: BayesianARX,
    r: np.ndarray,
    controller: ControllerType = "pid",
    param_bounds: dict[str, tuple[float, float]] | None = None,
    n_iterations: int = 100,
    n_parameter_samples: int = 200,
    output_bound: float = 10.0,
    input_bound: float | None = None,
    robustness_delta: float = 0.0,
    include_process_noise: bool = True,
    random_state: int | None = None,
) -> TuningReport:
    """
    Derivative-free random-search tuning for robust probabilistic closed-loop stability.

    Objective maximized:
        P(stable closed-loop | D, controller_params, robustness_delta)
    """
    if n_iterations <= 0:
        raise ValueError("n_iterations must be positive.")
    if n_parameter_samples <= 1:
        raise ValueError("n_parameter_samples must be at least 2.")

    bounds = _default_param_bounds(controller) if param_bounds is None else dict(param_bounds)
    rng = np.random.default_rng(random_state)

    best_params: dict[str, float] | None = None
    best_estimate: StabilityEstimate | None = None

    for _ in range(n_iterations):
        params = _sample_uniform_params(rng, bounds)
        candidate_seed = None if random_state is None else int(rng.integers(0, 2**31 - 1))
        estimate = estimate_closed_loop_stability_probability(
            model=model,
            r=r,
            controller=controller,
            controller_params=params,
            n_parameter_samples=n_parameter_samples,
            output_bound=output_bound,
            input_bound=input_bound,
            robustness_delta=robustness_delta,
            include_process_noise=include_process_noise,
            random_state=candidate_seed,
        )
        if best_estimate is None or estimate.probability > best_estimate.probability:
            best_params = params
            best_estimate = estimate

    assert best_params is not None and best_estimate is not None
    return TuningReport(
        best_params=best_params,
        best_stability_probability=best_estimate.probability,
        best_probability_std_error=best_estimate.std_error,
        best_probability_ci95=(best_estimate.ci95_low, best_estimate.ci95_high),
        n_evaluations=n_iterations,
        n_parameter_samples=n_parameter_samples,
        controller=controller,
        robustness_delta=robustness_delta,
    )
