# Repository Audit and Expansion Plan

**Owner:** Bayes SysID maintainers  
**Last validated on:** 2026-04-21

## Current functionality snapshot

The repository now supports a full identification-to-controls workflow, centered on Bayesian ARX and expanded with MIMO, online updates, and control add-ons.

### Implemented features

- SISO Bayesian ARX (`BayesianARX`) and unknown-noise variant (`BayesianARXUnknownNoise`).
- LS baseline (`LeastSquaresARX`) and simulation (`simulate_arx`).
- Rolling order search and predictive diagnostics (`rolling_order_search`, calibration and NLL utilities).
- Online recursive Bayesian updates (`OnlineBayesianARX`).
- MIMO Bayesian ARX (`BayesianMIMOARX`).
- Stability, frequency-response, and Nyquist uncertainty analysis.
- Realization, observer, LQR/LQG, and Bayesian Gramian/HSV summaries.
- Closed-loop Monte Carlo, probabilistic controller tuning, classical/empirical robust margin summaries, and structured-surrogate robustness via LFT helpers.
- DSF prototype workflow for MIMO transfer factorization and edge-probability diagnostics (prototype only, not theorem-backed).

See `docs/roadmap_status.md` for module-level status, tests, demos, and next steps.

## Remaining gaps (current)

1. **DSF identifiability theory gap**: current DSF pipeline is useful for exploration but does not provide theorem-backed guarantees.
2. **Robustness theory gap**: current robust outputs are classical/empirical margins and structured small-gain surrogates, not full structured singular value (μ) analysis.
3. **Benchmark depth gap**: coverage is broad but still mostly synthetic and pedagogical; broader benchmark suites are needed.
4. **Experiment design gap**: more guidance is needed for MIMO excitation design under noisy/partial observability.

## Expansion roadmap

### Phase 1: Hardening and reproducibility

- Expand benchmark datasets and scripted experiment reports.
- Add richer CI checks for docs/code consistency and artifact indexing.
- Add numerical conditioning diagnostics in realization/Gramian routines.

### Phase 2: Theorem-backed structure learning

- Lift DSF prototype into theorem-backed identifiability workflow.
- Add experiment-design constraints (excitation richness, horizon sizing, sensor placement sensitivity).

### Phase 3: Robust-control depth

- Add optional integration path for full μ-analysis toolchains.
- Separate “surrogate robust indicators” vs “certified robust guarantees” in reporting templates.

### Phase 4: Scaled Bayesian controls workflow

- Add larger MIMO examples and partial-observability studies.
- Add posterior-risk summaries across LQR/LQG + robustness metrics in one report artifact.

## Immediate next tasks

1. Deliver theorem-backed DSF identifiability tests and documentation.
2. Add automated summary report combining stability probability, Gramian/HSV spread, and closed-loop risk.
3. Expand examples to include noisy partially observed MIMO networks.
4. Add optional interoperability notes for external μ-analysis tools while keeping internal terminology precise.
