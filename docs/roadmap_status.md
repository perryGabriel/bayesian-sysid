# Roadmap Status Matrix

**Owner:** Bayes SysID maintainers  
**Last validated on:** 2026-04-21

This file is the single source of truth for feature status across code, tests, and demos.

## Status matrix

| Feature | Module | Test coverage | Demo | Status | Next steps |
|---|---|---|---|---|---|
| Bayesian ARX (known noise) | `bayes_sysid.arx.BayesianARX` | `tests/test_arx.py`, `tests/test_simulate.py` | `examples/demo_arx.py` | Implemented | Add richer benchmark datasets and calibration stress tests. |
| Bayesian ARX (unknown noise, Student-t predictive) | `bayes_sysid.arx.BayesianARXUnknownNoise` | `tests/test_arx.py` | `examples/demo_arx.py` | Implemented | Add posterior predictive checks for outlier-heavy regimes. |
| Rolling order search and diagnostics | `bayes_sysid.arx.rolling_order_search`, `bayes_sysid.metrics.rolling_origin_nll_diagnostics` | `tests/test_arx.py`, `tests/test_metrics.py` | `examples/demo_metrics_diagnostics.py` | Implemented | Add larger `(na, nb)` search policies and runtime guardrails. |
| Online recursive Bayesian ARX updates | `bayes_sysid.online.OnlineBayesianARX`, `recursive_posterior_update` | `tests/test_online.py` | `examples/demo_online_arx.py` | Implemented | Add forgetting-factor and change-point options. |
| MIMO Bayesian ARX regression | `bayes_sysid.mimo.BayesianMIMOARX` | `tests/test_mimo.py` | `examples/demo_dsf_scaffold.py` | Implemented | Add higher-order/high-dimensional examples and conditioning diagnostics. |
| Stability and pole-cloud uncertainty | `bayes_sysid.analysis.stability` | `tests/test_stability_analysis.py` | `examples/demo_stability_and_robustness.py` | Implemented | Add dataset-level stability confidence intervals in reports. |
| Frequency-response and Nyquist uncertainty envelopes | `bayes_sysid.analysis.frequency_response` | `tests/test_frequency_response.py` | `examples/demo_stability_and_robustness.py`, `examples/demo_posterior_nyquist_band.py` | Implemented | Add automatic frequency-grid adaptation and uncertainty summaries near critical points. |
| Realization + minimal realization helpers | `bayes_sysid.control.realization` | `tests/test_realization.py` | `examples/demo_realization.ipynb` | Implemented | Add conditioning warnings for near-canceling pole/zero structures. |
| Observer utilities and Kalman filter baseline | `bayes_sysid.control.observer` | `tests/test_observer.py` | `examples/demo_observer_and_bayesian_gramians.py` | Implemented | Add innovation diagnostics and finite-horizon smoothing companion. |
| Bayesian Gramian ensemble + HSV summaries | `bayes_sysid.control.gramians` | `tests/test_gramians.py` | `examples/demo_observer_and_bayesian_gramians.py` | Implemented | Add balanced truncation suggestions and report export. |
| Closed-loop Monte Carlo + probabilistic tuning | `bayes_sysid.control.closed_loop`, `bayes_sysid.control.tuning` | `tests/test_closed_loop_control.py`, `tests/test_controller_tuning.py` | `examples/demo_stability_and_robustness.py` | Implemented | Add controller families beyond static/PID and tighter budgeted search strategies. |
| LQR/LQG synthesis from realizations | `bayes_sysid.control.lqr`, `bayes_sysid.control.lqg` | `tests/test_lqr.py`, `tests/test_lqg.py` | `examples/demo_observer_and_bayesian_gramians.py` | Implemented | Add integrated posterior risk summary plots for controller/observer co-design. |
| Robust margins and structured surrogate robustness | `bayes_sysid.control.margins`, `bayes_sysid.control.lft`, `bayes_sysid.control.robustness` | `tests/test_robust_margins.py`, `tests/test_lft.py` | `examples/demo_stability_and_robustness.py` | Implemented (surrogate) | Keep terminology explicit: empirical/classical margins and structured small-gain surrogates are not full μ-analysis. |
| DSF reconstruction workflow | `bayes_sysid.control.dsf` | `tests/test_dsf.py` | `examples/demo_dsf_scaffold.py` | Prototype implemented (not theorem-backed) | Add theorem-backed identifiability guarantees and uncertainty-aware MIMO experiment design. |

## Public API claims (checked by CI)

<!-- DOCS_API_CLAIMS_START -->
- `ARXRegressionData`
- `BayesianARX`
- `BayesianARXUnknownNoise`
- `LeastSquaresARX`
- `OrderSearchResult`
- `build_arx_regression`
- `BayesianMIMOARX`
- `build_mimo_regression`
- `rolling_order_search`
- `OnlineBayesianARX`
- `recursive_posterior_update`
- `build_predictive_diagnostics_report`
- `simulate_arx`
<!-- DOCS_API_CLAIMS_END -->
