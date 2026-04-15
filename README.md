# Bayesian ARX System Identification

A research-oriented Python project for **Bayesian linear system identification** with
**ARX (AutoRegressive with eXogenous input)** models, plus uncertainty-aware tools
for stability and closed-loop control analysis.

This repository is designed for class projects and rapid experimentation: it provides
simple APIs, interpretable Bayesian math, and reproducible demos that visualize what
changes when we propagate posterior uncertainty into downstream control tasks.

---

## 1) Project objective

The main objective is to identify low-order dynamical models from input/output data,
while preserving and using uncertainty in parameter estimates. In contrast to plain
least squares, Bayesian ARX gives a posterior distribution over parameters, enabling:

- predictive intervals (not just point predictions),
- uncertainty-aware stability analysis,
- posterior-sampled frequency-response envelopes,
- Monte Carlo closed-loop performance and robustness checks.

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

Predictive variance combines:

- observation noise: $\sigma^2$,
- parameter uncertainty: $\phi_*^\top\Sigma_N\phi_*$.

The repository also includes an **unknown-noise** Bayesian ARX variant with a
Student-t predictive distribution.

---

## 3) Key features

- Bayesian and least-squares ARX fitting APIs.
- Known-noise and unknown-noise Bayesian variants.
- Prior construction helpers (isotropic, diagonal, scale-aware).
- Rolling-origin ARX order selection.
- Posterior-sampled stability probability and pole-cloud analysis.
- Frequency-response sampling and uncertainty envelopes.
- Closed-loop Monte Carlo simulation under posterior parameter samples.
- Nominal and empirical (posterior) gain/phase margin summaries.

Stability conventions:

- `domain="discrete"` (default): stable if every pole satisfies `|z| < 1 - tol`.
- `domain="continuous"`: stable if every pole satisfies `Re(s) < -tol`.

---

## 4) Repository structure

```text
.
├── README.md
├── pyproject.toml
├── docs/
│   ├── controls_addons_research_plan.md
│   ├── repo_audit_and_expansion_plan.md
│   └── final_report.md
├── examples/
│   ├── demo_arx.py
│   ├── demo_stability_and_robustness.py
│   ├── demo_posterior_nyquist_band.py
│   ├── demo_uncertainty_insufficient_information.py
│   └── artifacts/
│       ├── posterior_trajectory_band.png
│       ├── predictive_density_annotated.png
│       ├── stability_pole_cloud.png
│       ├── frequency_response_envelope.png
│       └── closed_loop_monte_carlo.png
│       ├── closed_loop_monte_carlo.png
│       └── nyquist_posterior_band.png
├── src/bayes_sysid/
│   ├── __init__.py
│   ├── arx.py
│   ├── models.py
│   ├── regression.py
│   ├── priors.py
│   ├── selection.py
│   ├── metrics.py
│   ├── simulate.py
│   ├── analysis/
│   │   ├── stability.py
│   │   └── frequency_response.py
│   └── control/
│       ├── closed_loop.py
│       ├── margins.py
│       ├── lft.py
│       └── tuning.py
└── tests/
```

---

## 5) Installation

From repository root:
@@ -150,50 +152,51 @@ pip install numpy scipy matplotlib pytest

---

## 6) Quick start

### Minimal API example

```python
from bayes_sysid import BayesianARX

# Build + fit
model = BayesianARX(na=2, nb=2, sigma2=0.05)
model.fit(y, u)

# One-step predictive distribution
mean, var = model.predict_next_distribution(y_hist, u_hist)
print("predictive mean:", mean)
print("predictive variance:", var)
```

### Run demos

```bash
python examples/demo_arx.py
python examples/demo_stability_and_robustness.py
python examples/demo_posterior_nyquist_band.py
python examples/demo_uncertainty_insufficient_information.py
```

---

## 7) Validation and tests

Run unit tests:

```bash
pytest -q
```

The test suite covers ARX fitting behavior, predictive utilities, API structure,
stability/frequency-response analysis, and closed-loop/robustness helpers.

---

## 8) Final report for course rubric

A course-style final report aligned with the grading rubric (background,
Bayesian-method details, results interpretation, and clarity) is provided at:

- `docs/final_report.md`

It is formatted for straightforward conversion to PDF (e.g., Pandoc) using 12pt
font, 1-inch margins, single spacing, and target length 4–5 pages.

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

Current scope is SISO linear ARX with uncertainty propagation into selected control
analyses. Natural extensions:

- explicit state-space realization and observer design,
- MIMO identification,
- richer structured uncertainty and robust synthesis,
- online/sequential Bayesian updates for adaptive control loops.
