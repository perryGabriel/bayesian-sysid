# Controls-Focused Add-ons Roadmap (from ARX to publishable research)

## Context

Current repository capabilities are strongest for Bayesian SISO ARX identification and uncertainty-aware prediction.
A controls audience will appreciate progress toward closed-loop guarantees, robustness claims, and structural
inference beyond one-step forecasting metrics.

This document answers:
1. What can be done **without explicit state-space models**?
2. What minimum extensions are required for each add-on?
3. Which conference communities are likely to value each direction?

---

## Implementation status (current repo snapshot)

Implemented as first drafts:

- `analysis/stability.py`: `arx_poles`, `is_stable(domain=...)`, `posterior_stability_probability(domain=...)`.
- `analysis/frequency_response.py`: ARX frequency response + posterior uncertainty envelopes.
- `control/closed_loop.py`: closed-loop simulation + posterior Monte Carlo under static/PID feedback.
- `control/margins.py`: preliminary classical margin extraction + empirical posterior summaries.
- `examples/demo_stability_and_robustness.py`: end-to-end controls-oriented demonstration and plotting.

This means Sprint 1 and Sprint 2 now have baseline implementations, and the roadmap below should be read as
"strengthen + formalize" rather than "start from scratch."

Not implemented yet (important gap for next phases):

- ARX/state-space bridge utilities (realization, minimality, and model-order cleanup).
- Controllability/observability Grammians and Hankel singular value analysis.
- LQR/LQG synthesis and observer/Kalman design tools built on posterior samples.
- DSF reconstruction and MIMO identifiability diagnostics.

---

## 1) Stability analysis from ARX/non-state-space models

### What is possible now (no explicit state-space realization)

For SISO ARX with

$$
y_t + a_1 y_{t-1}+\cdots+a_{n_a}y_{t-n_a}=b_1u_{t-1}+\cdots+b_{n_b}u_{t-n_b}+e_t,
$$

internal stability of the noise-free autonomous part is governed by roots of

$$
A(z^{-1}) = 1 + a_1 z^{-1}+\cdots+a_{n_a}z^{-n_a}.
$$

Discrete-time stability can be tested directly by checking if poles of
$A(z^{-1})^{-1}$ lie strictly inside the unit disk.

For continuous-time analyses, stability is based on pole real parts:
$\Re(s) < -\text{tol}$ after mapping discrete ARX poles to a continuous approximation.

With Bayesian posteriors, this can become a **probabilistic stability** statement:
- sample $\theta$ from posterior,
- compute poles per sample,
- estimate $\Pr(\text{stable}\mid \mathcal D)$.

This is immediately useful and interpretable for controls users.

### Minimal implementation steps

- Add utility: `arx_poles(theta, na)`, `arx_poles_by_domain(...)`, and `is_stable(poles, domain=...)`.
- Add Bayesian stability probability: `posterior_stability_probability(..., domain=...)`.
- Add plots: pole cloud in complex plane + a domain-appropriate boundary (unit circle for discrete-time).

### Novel angle

Most ARX workflows report fit metrics only. A strong controls contribution is:
- uncertainty-aware stability certification,
- and closed-loop risk metrics derived from posterior pole clouds.

### Likely conference fit

- **CDC / ACC**: strong fit if tied to guarantees or risk bounds.
- **IFAC World Congress**: good fit for practical identification + stability.

---

## 2) LQR/LQG controller construction

### What is possible without state-space?

Classical LQR/LQG is naturally state-space-based, so **not directly** from ARX coefficients alone.
However, you can derive realizations from transfer models (ARX -> realization) with caveats:
- realization may be non-minimal/noisy,
- uncertainty propagation is harder but still possible by Monte Carlo.

### Minimal next steps required

- Add ARX -> state-space realization routine (controllable canonical form for SISO).
- Add model reduction/minimality checks.
- Add discrete-time Riccati solver wrappers (`scipy.linalg.solve_discrete_are`).
- For Bayesian version: sample state-space models from posterior ARX coefficients and produce
  distribution over gains and closed-loop poles.

### Codex prompt (ready to implement after state-space bridge)

```text
Implement Sprint 3, task 1: add a deterministic ARX->state-space realization module for SISO models.
Requirements:
- New module: src/bayes_sysid/control/realization.py
- API:
  - arx_to_state_space(a: np.ndarray, b: np.ndarray, dt: float | None = None) -> tuple[A,B,C,D]
  - minimal_realization(A,B,C,D, tol=1e-9) -> tuple[A,B,C,D, kept_states]
  - validate_realization_shapes(...)
- Use controllable companion-form realization for proper ARX transfer functions.
- Add tests:
  - realization reproduces ARX transfer response on dense frequency grid;
  - controllability rank is full before reduction for canonical examples;
  - minimal_realization removes uncontrollable/unobservable modes in synthetic case.
- Add docstrings and one usage snippet in docs/controls_addons_research_plan.md.
```

### Novel angle

- **Bayesian LQR under identification uncertainty** from ARX posterior samples.
- Evaluate chance of instability / performance regret over posterior draw ensemble.

### Likely conference fit

- **ACC/CDC**: good for robust/adaptive control framing.
- **L4DC**: very strong if decision-theoretic uncertainty-aware synthesis is emphasized.

---

## 3) Doyle-style robust margin / feedback uncertainty (Delta construction)

### What is possible without state-space?

Some robust margins can be approximated from transfer models/frequency response directly
(e.g., gain/phase margins, disk margins approximations), but full structured uncertainty
($M-\Delta$) workflows are much easier with state-space and robust-control toolchains.

### Minimal next steps required

- Add frequency-response generation from ARX posterior samples.
- Define uncertainty envelope from posterior (e.g., multiplicative uncertainty weight).
- For full $\mu$-analysis style results, introduce a robust-control backend
  (likely Python wrappers or MATLAB bridge if strict functionality needed).
- Formalize feedback interconnection and uncertainty block assumptions.

### Novel angle

- Data-driven uncertainty blocks inferred from Bayesian identification posterior,
  then mapped into robust margin claims.

### Likely conference fit

- **CDC** robust control sessions.
- **European Control Conference (ECC)** if theoretical guarantees are clean.

---

## 4) Kalman filtering or Luenberger observer design

### What is possible without state-space?

Observers require a state-space model (or equivalent latent representation).
Without one, only output-predictor style filtering can be done (ARX predictor updates)
but that is not a full observer design framework.

### Minimal next steps required

- Add explicit state-space realization.
- Add observability checks and observer gain design.
- For stochastic setting: process/measurement covariance estimation and Kalman recursion.
- Compare certainty-equivalent observer/controller to posterior-sampled alternatives.

### Codex prompt (ready to implement in Sprint 3 after realization utilities land)

```text
Implement observer baseline tooling in src/bayes_sysid/control/observer.py.
Requirements:
- APIs:
  - observability_matrix(A, C, horizon=None)
  - is_observable(A, C, tol=1e-9)
  - design_luenberger_gain(A, C, desired_poles)
  - run_kalman_filter(A, B, C, Q, R, u, y, x0=None, P0=None)
- Add tests for:
  - observable/unobservable toy systems;
  - pole placement sanity (eigenvalues(A - L*C));
  - Kalman state estimate RMSE improvement vs open-loop predictor on noisy simulation.
- Keep interfaces NumPy-first and consistent with existing control modules.
```

### Novel angle

- Bayesian observer gain distributions from posterior-identified models.
- Risk-aware observer tuning based on posterior predictive error decomposition.

### Likely conference fit

- **ACC/CDC**, also **IFAC SYSID** if framed as identification-to-estimation pipeline.

---

## 5) Dynamical Structure Function (DSF) reconstruction (Gonçalves/Warnick style)

### What is possible without state-space?

DSF sits between transfer-function and state-space representations, targeting
network interactions among measured variables. In strict SISO ARX, DSF content is limited.
For meaningful DSF reconstruction, at least MIMO ARX/ARMAX-style identification is needed.

### Minimal next steps required

- Add MIMO identification support and excitation design guidance.
- Build transfer matrix estimates with uncertainty.
- Implement DSF factorization/reconstruction constraints from measured outputs.
- Add identifiability checks and experiment-design requirements.

### Codex prompt (ready to implement in Sprint 4 once MIMO backbone exists)

```text
Implement initial DSF scaffold under src/bayes_sysid/control/dsf.py.
Requirements:
- Provide placeholders + working utilities for:
  - transfer_matrix_from_mimo_arx(...)
  - dsf_from_transfer_matrix(G, method="stable_factorization")
  - posterior_edge_probability(dsf_samples, threshold)
- Add validation helpers for identifiability assumptions and excitation richness.
- Add tests with a 2x2 synthetic network where ground-truth edge sparsity is known.
- Document current limitations clearly (prototype; no full identifiability theorem yet).
```

---

## 6) Controllability/observability Grammians and Hankel singular values

### What is possible without state-space?

Not in a principled way. Grammians and Hankel singular values are state-space objects, so this idea
requires an explicit realization pipeline first (ARX -> state-space).

### Minimal next steps required

- Add a stable discrete-time Lyapunov solver wrapper for:
  - controllability Grammian $W_c$ from $A W_c A^\top - W_c + B B^\top = 0$,
  - observability Grammian $W_o$ from $A^\top W_o A - W_o + C^\top C = 0$.
- Add rank/conditioning diagnostics:
  - numerical controllability/observability ranks,
  - condition numbers and near-uncontrollable mode flags.
- Add Hankel singular value computation:
  - compute singular values of $W_c W_o$ (or balanced factor forms),
  - expose cumulative-energy truncation suggestions for reduced-order models.
- Add posterior uncertainty workflow:
  - sample ARX/posterior models -> realizations -> Grammians/HSVs,
  - summarize distribution of HSV spectra and probability that each mode is important.

### Novel angle

- Bayesian uncertainty over balanced truncation quantities (not just point-estimate model reduction).
- Risk-aware reduced-order model selection based on posterior mode-energy probabilities.

### Likely conference fit

- **ACC/CDC** model reduction + robust/adaptive sessions.
- **IFAC SYSID** for identification-to-reduction pipelines with uncertainty quantification.

### Codex prompt (ready to implement in Sprint 3 after realization utilities land)

```text
Implement Grammians + Hankel singular values module: src/bayes_sysid/control/grammians.py
Requirements:
- APIs:
  - controllability_grammian(A, B)
  - observability_grammian(A, C)
  - hankel_singular_values(A, B, C, sort_desc=True)
  - balanced_truncation_energy(hsv, energy=0.99) -> retained_order
  - posterior_hsv_summary(arx_posterior_samples, na, nb, dt=None)
- Use scipy.linalg.solve_discrete_lyapunov where appropriate.
- Add robust input validation and numerical warnings for unstable A.
- Tests:
  - known stable system with analytic/benchmark values;
  - invariance of HSVs to similarity transform (within tolerance);
  - monotone cumulative-energy behavior.
- Add example script showing posterior HSV bands and suggested reduced orders.
```

### Novel angle

- Bayesian DSF with uncertainty quantification over edges/interaction strengths.
- Closed-loop DSF reconstruction under realistic experimental constraints.

### Likely conference fit

- **IFAC SYSID**, **CDC networks**, **NetControl**-adjacent communities.

---

## What can be done immediately (no state-space yet)

1. ✅ **Probabilistic stability module** for Bayesian ARX posterior.
2. ✅ **Frequency-domain uncertainty envelopes** from posterior samples.
3. ✅ **Closed-loop Monte Carlo with simple fixed controllers** (PID/static output feedback)
   to quantify robustness to identified-model uncertainty.
4. ✅ **Preliminary robust margins** in transfer-function form (gain/phase/disk-style approximations).

These produce controls-relevant figures quickly while postponing full state-space machinery.

---

## What requires state-space and should be Phase 3+

- Full LQR/LQG synthesis with principled uncertainty propagation.
- Kalman/Luenberger observer design with observability guarantees.
- Mature structured-uncertainty robust analysis ($M-\Delta$, $\mu$-like analyses).
- DSF reconstruction at meaningful scale (MIMO + identifiability conditions).

---

## Suggested research programs (novel and publishable)

### Program A: Bayesian stability certificates for ARX identification

- Problem: estimate $\Pr(\text{stable}\mid\mathcal D)$ and confidence intervals.
- Deliverables: bounds, diagnostics, and benchmark comparisons with LS confidence ellipsoids.
- Venue: ACC/CDC.

### Program B: Data-driven robust margins from posterior uncertainty

- Problem: infer uncertainty weights from posterior predictive spread and map to margin claims.
- Deliverables: uncertainty-to-margin pipeline and closed-loop stress tests.
- Venue: CDC/ECC robust control tracks.

### Program C: Bayesian DSF under limited excitation and partial observability

- Problem: reconstruct interaction structure with quantified uncertainty.
- Deliverables: posterior edge probabilities, identifiability diagnostics, experiment design rules.
- Venue: IFAC SYSID, CDC networked systems.

---

## Concrete backlog (recommended order)

### Sprint 1 (high impact, low dependency)

- ✅ Added `stability.py` utilities (pole test, posterior stability probability).
- ✅ Added stability plots/demo script (`examples/demo_stability_and_robustness.py`).
- ✅ Added tests for known stable/unstable ARX examples.

### Sprint 2

- ✅ Added frequency response utilities and posterior uncertainty bands.
- ✅ Added closed-loop simulation against fixed controllers (PID/static gain).
- ✅ Added margin-report helper (gain/phase + empirical Monte Carlo robustness).

### Sprint 3 (state-space bridge)

- [ ] Add ARX -> state-space realization.
- [ ] Add LQR controller synthesis and observer/Kalman baseline tools.
- [ ] Add controllability/observability Grammians and Hankel singular value analysis.
- [ ] Add Bayesian gain distribution reporting.

#### Codex prompts for Sprint 3

- **Prompt A (state-space bridge):** use the "LQR/LQG controller construction" prompt in Section 2.
- **Prompt B (observer tooling):** use the "Kalman filtering or Luenberger observer design" prompt in Section 4.
- **Prompt C (Grammians/HSV):** use the Section 6 prompt.

### Sprint 4 (advanced research)

- [ ] Add MIMO identification and DSF reconstruction prototypes.
- [ ] Add identifiability diagnostics and experiment-design module.

#### Codex prompts for Sprint 4

- **Prompt D (DSF prototype):** use the Section 5 prompt.

---

## Evaluation metrics controls reviewers will care about

- Stability probability and calibration.
- Closed-loop pole distributions and instability probability.
- Tracking/energy tradeoffs under uncertainty.
- Robustness margins versus uncertainty set definitions.
- Reproducibility of inferred structure (for DSF) across datasets/experiments.

---

## Bottom line

Without state-space, the most credible near-term controls contributions are:
- uncertainty-aware **stability analysis**,
- frequency/robustness characterization from posterior uncertainty,
- and closed-loop risk quantification under identified-model uncertainty.

LQR/LQG, observer design, and mature DSF workflows become strongest once the repo includes
an explicit state-space bridge and MIMO support.
