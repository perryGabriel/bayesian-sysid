# Repository Audit and Expansion Plan

## Current functionality

This repository is a compact, pedagogical package for Bayesian system identification with
single-input single-output (SISO) ARX models.

### Implemented core features

- **Regression matrix builder (`build_arx_regression`)**
  - Converts aligned output/input sequences into an ARX design matrix and targets.
  - Supports flexible lag orders `na`, `nb` and validates input lengths.

- **Classical baseline (`LeastSquaresARX`)**
  - Fits ARX parameters via ordinary least squares.
  - Supports one-step prediction and deterministic multi-step rollout.

- **Bayesian estimator (`BayesianARX`)**
  - Uses a Gaussian prior on parameters and Gaussian likelihood with known noise variance `sigma2`.
  - Computes closed-form posterior mean and covariance.
  - Exposes posterior-mean prediction and full posterior predictive mean/variance.
  - Provides predictive density evaluation over a grid.
  - Supports parameter sampling and predictive simulation from posterior draws.
  - Supports multi-step trajectory rollout under parameter uncertainty.

- **Simulation utility (`simulate_arx`)**
  - Simulates SISO ARX data from known coefficients and optional Gaussian process noise.

- **Example script (`examples/demo_arx.py`)**
  - End-to-end workflow: simulate data, fit LS and Bayesian ARX, compare predictions,
    and visualize predictive density + trajectory uncertainty bands.

- **Tests**
  - Broad unit-test coverage now spans regression construction, Bayesian/LS APIs, unknown-noise model behavior,
    prior helpers, simulation checks, API compatibility checks, stability/frequency analysis, closed-loop utilities,
    and preliminary margin tools.

- **Controls analysis add-ons (first draft implemented)**
  - Stability utilities with posterior stability probability.
  - Posterior frequency-response envelopes.
  - Closed-loop Monte Carlo under static/PID feedback.
  - Preliminary nominal/empirical gain/phase margin summaries.

### Packaging and usability status

- Packaged with setuptools and editable-install friendly (`pip install -e .`).
- Public API exports are minimal and clear in `bayes_sysid.__init__`.
- Scope is intentionally educational and compact, not yet production-oriented.

## Gaps and limitations

1. **Only SISO ARX**: no multi-input/multi-output support yet.
2. **No explicit state-space bridge** for full observer/LQG workflows.
3. **Robust margins are preliminary** (classical approximations; no full structured \(M-\Delta\) pipeline).
4. **No online/sequential updates**: fitting remains batch-only.
5. **No constrained/stability-enforcing priors** yet (only unconstrained priors and posterior diagnostics).

## Expansion roadmap

### Phase 1: Reliability and DX (quick wins)

- Add richer tests:
  - edge-case validation (invalid lags, mismatched lengths, insufficient history),
  - posterior consistency checks against analytical identities,
  - deterministic behavior checks for random seeds.
- Add metrics utilities:
  - one-step RMSE/MAE/NLL,
  - calibration diagnostics for predictive intervals.
- Improve docs:
  - API docs with shape conventions,
  - “input/output alignment” guide to avoid off-by-lag mistakes.

### Phase 2: Bayesian model completeness

- Add **unknown noise variance** with conjugate prior:
  - Normal-Inverse-Gamma posterior,
  - Student-t posterior predictive.
- Add prior configuration helpers:
  - isotropic ridge prior,
  - diagonal AR-vs-input shrinkage,
  - prior scaling by regressor variance.

### Phase 3: Modeling breadth

- Add exogenous terms and variants:
  - ARMAX-like structure (with caution about latent noise dynamics),
  - optional bias/intercept term,
  - automatic feature standardization.
- Add **MIMO support**:
  - block regressors,
  - independent-output and coupled-output posterior options.

### Phase 4: Selection and automation

- Implement model order selection utilities:
  - grid search over `(na, nb)`,
  - criteria: marginal likelihood (Bayesian), AIC/BIC, validation NLL.
- Add cross-validation helpers for time series (rolling-origin splits).

### Phase 5: Advanced uncertainty and deployment

- Online/sequential Bayesian updates for streaming identification.
- Constrained priors for stability preferences.
- Optional robust likelihoods (e.g., Student-t noise) for outlier resilience.
- Export/persistence helpers and reproducible experiment configs.

## Status update

The original immediate tasks from this plan are now implemented:

1. Test expansion with broad functionality coverage.
2. Metrics module and demo integration.
3. Normal-Inverse-Gamma unknown-noise Bayesian ARX.
4. Rolling order-selection utility.

Additionally, first-draft controls add-ons (stability/frequency/closed-loop/margins) have been added.
