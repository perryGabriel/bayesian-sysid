# Bayesian ARX System Identification

A small research-oriented Python repo for **Bayesian linear system identification** using an **ARX model**.

The basic idea is:

- model the next output as a linear function of past outputs and inputs,
- place a Gaussian prior on the ARX parameter vector,
- update that prior with data to obtain a Gaussian posterior,
- use the posterior to produce either
  - a **posterior predictive distribution** for the next output, or
  - a **plug-in prediction** using the posterior mean parameters.

This is the Bayesian analogue of least-squares ARX identification.

## Model

For orders `na` and `nb`, we use

$$
y_t = \phi_t^\top \theta + e_t, \qquad e_t \sim \mathcal N(0, \sigma^2)
$$

with regressor

$$
\phi_t = [y_{t-1},\dots,y_{t-na},u_{t-1},\dots,u_{t-nb}]^\top.
$$

The prior is

$$
\theta \sim \mathcal N(\mu_0, \Sigma_0).
$$

Given data, the posterior is Gaussian:

$$
\Sigma_N^{-1} = \Sigma_0^{-1} + \frac{1}{\sigma^2} \Phi^\top \Phi,
\qquad
\mu_N = \Sigma_N\left(\Sigma_0^{-1}\mu_0 + \frac{1}{\sigma^2}\Phi^\top y\right).
$$

For a new regressor $\phi_{*}$, the posterior predictive distribution is

$$
y_{\ast} \mid \mathcal D, \phi_{\ast} \sim
\mathcal N\left(
\phi_{\ast}^{\top} \mu_N,
\phi_{\ast}^{\top} \Sigma_N \phi_{\ast} + \sigma^2
\right).
$$

That predictive variance contains both:

- **noise variance** $\sigma_2$, and
- **parameter uncertainty** $\phi_{\ast}^T \sum_N \phi_{*}$.

## Repo structure

```text
bayesian_arx_repo/
├── README.md
├── pyproject.toml
├── examples/
│   └── demo_arx.py
├── src/
│   └── bayes_sysid/
│       ├── __init__.py
│       ├── arx.py          # backward-compatible facade
│       ├── regression.py   # ARX regressor construction
│       ├── models.py       # LS/Bayesian ARX estimators
│       ├── selection.py    # rolling order search
│       ├── priors.py       # prior helper utilities
│       ├── metrics.py
│       ├── simulate.py
│       ├── analysis/
│       │   ├── stability.py
│       │   └── frequency_response.py
│       └── control/
│           ├── closed_loop.py
│           └── margins.py
└── tests/
    ├── test_arx.py
    ├── test_metrics.py
    ├── test_simulate.py
    ├── test_api_structure.py
    ├── test_stability_analysis.py
    ├── test_frequency_response.py
    ├── test_closed_loop_control.py
    └── test_robust_margins.py
```

## Installation

From the repo root:

```bash
pip install -e .
```

Or install dependencies manually:

```bash
pip install numpy scipy matplotlib
```

## Quick demos

```bash
python examples/demo_arx.py
python examples/demo_stability_and_robustness.py
```

For uncertainty demos in undersampled / low-data regimes, open:

```bash
jupyter notebook examples/demo_uncertainty_insufficient_information.ipynb
```

`demo_arx.py` focuses on identification/prediction comparisons.

`demo_stability_and_robustness.py` adds controls-oriented outputs:

- posterior stability probability and pole-cloud visualization,
- posterior frequency-response uncertainty envelope,
- nominal + empirical robust margin summary,
- closed-loop Monte Carlo response bands under posterior uncertainty.


## Current capabilities

In addition to Bayesian/LS ARX fitting, the repo now includes:

- unknown-noise Bayesian ARX (`BayesianARXUnknownNoise`) with Student-t predictive distributions,
- prior helper utilities (isotropic, AR-vs-input diagonal, regressor-variance scaling),
- rolling-origin ARX order search,
- analysis tools for ARX stability and posterior stability probability,
- posterior frequency-response sampling and uncertainty envelopes,
- closed-loop Monte Carlo simulation with static/PID controllers,
- preliminary gain/phase margin summaries (nominal + empirical posterior).

## Main class

```python
from bayes_sysid import BayesianARX

model = BayesianARX(na=2, nb=2, sigma2=0.05)
model.fit(y, u)

mean, var = model.predict_next_distribution(y_hist, u_hist)
print(mean, var)
```


## Citation

If this repository is useful in your work, you can cite it with the following BibTeX entry:

```bibtex
@software{bayesian_arx_sysid_2026,
  title        = {Bayesian ARX System Identification},
  author       = {{Bayesian ARX SysID Contributors}},
  year         = {2026},
  version      = {0.1.0},
  note         = {Python package},
  url          = {https://github.com/<owner>/bayesian-sysid}
}
```

> Replace `<owner>` in the URL with your GitHub organization or username.

## Notes

This repo is intentionally pedagogical, but now includes a first controls-analysis stack
(stability/frequency/closed-loop/margins) on top of Bayesian ARX.

Natural next extensions include:

- explicit state-space realization and LQR/LQG/observer pipelines,
- structured robustness analysis with richer uncertainty blocks,
- MIMO identification and dynamical structure function reconstruction,
- online/sequential Bayesian updates and adaptive control loops.
