# Bayesian ARX System Identification

A research-oriented Python project for **Bayesian linear system identification** with
**ARX (AutoRegressive with eXogenous input)** models, plus uncertainty-aware tools
for stability, realizations, observers, and closed-loop analysis.

---

### YouTube Presentation

Click [here](https://youtu.be/ctGgKKvs0lw) to view a presentation of this project.

## 1) Project objective

The main objective is to identify low-order dynamical models from input/output data,
while preserving and using uncertainty in parameter estimates. In contrast to plain
least squares, Bayesian ARX gives a posterior distribution over parameters, enabling:

- predictive intervals (not just point predictions),
- uncertainty-aware stability analysis,
- posterior-sampled frequency-response envelopes,
- Monte Carlo closed-loop performance and robustness checks,
- Bayesian state-space realization and observer studies from posterior samples.

---

## 2) Mathematical model

For ARX orders `na` and `nb`, we model

$$
y_t = \phi_t^\top \theta + e_t,\qquad e_t \sim \mathcal{N}(0,\sigma^2)
$$

with regressor

$$
\phi_t = [y_{t-1},\dots,y_{t-na},u_{t-1},\dots,u_{t-nb}]^\top.
$$

Prior:

$$
\theta \sim \mathcal{N}(\mu_0,\Sigma_0).
$$

Given stacked data `(Φ, y)`, the Gaussian posterior is

$$
\Sigma_N^{-1} = \Sigma_0^{-1} + \frac{1}{\sigma^2}\Phi^\top\Phi,
\qquad
\mu_N = \Sigma_N\left(\Sigma_0^{-1}\mu_0 + \frac{1}{\sigma^2}\Phi^\top y\right).
$$

For a new regressor $\phi_*$, posterior predictive is

$$
y_*\mid\mathcal{D},\phi_* \sim
\mathcal{N}\left(\phi_*^\top\mu_N,\,\phi_*^\top\Sigma_N\phi_*+\sigma^2\right).
$$

The repository also includes an **unknown-noise** Bayesian ARX variant with a
Student-t predictive distribution.

---

## 3) Bayesian state-space and Bayesian Gramians

For each posterior parameter sample $\theta^{(s)}$, we construct a realization
$(A^{(s)}, B^{(s)}, C^{(s)}, D^{(s)})$ via companion-form ARX realization.
This induces a **posterior distribution over state-space models**, i.e. a Bayesian
state-space family.

For stable sampled realizations, we compute discrete-time Gramians

$$
W_c^{(s)} = A^{(s)}W_c^{(s)}A^{(s)\top} + B^{(s)}B^{(s)\top},
$$
$$
W_o^{(s)} = A^{(s)\top}W_o^{(s)}A^{(s)} + C^{(s)\top}C^{(s)}.
$$

Across posterior samples, the set
$\{W_c^{(s)}, W_o^{(s)}\}_{s=1}^S$ is a **Bayesian Gramian ensemble**. In this
project, “Bayesian Gramians” means these Gramians are random objects induced by
posterior uncertainty in ARX parameters.

Observer tooling now includes:

- `observability_matrix(A, C, horizon=None)`
- `is_observable(A, C, tol=1e-9)`
- `design_luenberger_gain(A, C, desired_poles)`
- `run_kalman_filter(A, B, C, Q, R, u, y, x0=None, P0=None)`

---

## 4) Key features

- Bayesian and least-squares ARX fitting APIs.
- Known-noise and unknown-noise Bayesian variants.
- Prior construction helpers (isotropic, diagonal, scale-aware).
- Rolling-origin ARX order selection.
- Posterior-sampled stability probability and pole-cloud analysis.
- Frequency-response sampling and uncertainty envelopes.
- State-space realization and minimal realization helpers.
- Observer baselines: observability, Luenberger gain, and Kalman filtering.
- Closed-loop Monte Carlo simulation under posterior parameter samples.
- Nominal and empirical (posterior) gain/phase margin summaries.
- Prototype DSF utilities for 2x2 MIMO network structure exploration (edge probabilities + diagnostics).

Stability conventions:

- `domain="discrete"` (default): stable if every pole satisfies `|z| < 1 - tol`.
- `domain="continuous"`: stable if every pole satisfies `Re(s) < -tol`.

---

## 5) Repository structure

```text
.
├── README.md
├── pyproject.toml
├── docs/
├── examples/
│   ├── demo_arx.py
│   ├── demo_stability_and_robustness.py
│   ├── demo_posterior_nyquist_band.py
│   ├── demo_uncertainty_insufficient_information.py
│   ├── demo_observer_and_bayesian_gramians.py
│   └── artifacts/
├── src/bayes_sysid/
│   ├── analysis/
│   └── control/
│       ├── realization.py
│       ├── observer.py
│       ├── closed_loop.py
│       ├── margins.py
│       ├── lft.py
│       ├── tuning.py
│       └── dsf.py
└── tests/
```

---

## 6) Installation

From repository root:

```bash
pip install -e .
pip install numpy scipy matplotlib pytest
```

---

## 7) Quick start

### Minimal API example

```python
from bayes_sysid import BayesianARX

model = BayesianARX(na=2, nb=2, sigma2=0.05).fit(y, u)
mean, var = model.predict_next_distribution(y_hist, u_hist)
```

### Run demos

```bash
python examples/demo_arx.py
python examples/demo_stability_and_robustness.py
python examples/demo_posterior_nyquist_band.py
python examples/demo_uncertainty_insufficient_information.py
python examples/demo_observer_and_bayesian_gramians.py
python examples/demo_dsf_scaffold.py
```

The new observer/Gramian demo writes figures and tables to `examples/artifacts/`.

---

## 8) Validation and tests

Run unit tests:

```bash
pytest -q
```

The test suite covers ARX fitting behavior, predictive utilities, API structure,
stability/frequency-response analysis, realizations, observer helpers, and
closed-loop/robustness utilities.

---

## 9) Citation

If this repository is useful in your work, you can cite:

```bibtex
@misc{bayesian_arx_sysid_2026,
  title        = {Bayesian ARX System Identification},
  author       = {Gabriel M. Perry},
  year         = {2026},
  version      = {0.1.0},
  url          = {https://github.com/perryGabriel/bayesian-sysid}
}
```

---

## 10) Limitations and next steps

Current scope is still primarily SISO linear ARX with uncertainty propagation into
selected control analyses. Natural extensions:

- theorem-backed MIMO/DSF identifiability guarantees (current DSF scaffold is prototype-only),
- robust synthesis with posterior-aware uncertainty blocks,
- online/sequential Bayesian updates for adaptive loops.
