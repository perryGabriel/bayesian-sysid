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

\[
y_t = \phi_t^\top \theta + e_t, \qquad e_t \sim \mathcal N(0, \sigma^2)
\]

with regressor

\[
\phi_t = [y_{t-1},\dots,y_{t-na},u_{t-1},\dots,u_{t-nb}]^\top.
\]

The prior is

\[
\theta \sim \mathcal N(\mu_0, \Sigma_0).
\]

Given data, the posterior is Gaussian:

\[
\Sigma_N^{-1} = \Sigma_0^{-1} + \frac{1}{\sigma^2} \Phi^\top \Phi,
\qquad
\mu_N = \Sigma_N\left(\Sigma_0^{-1}\mu_0 + \frac{1}{\sigma^2}\Phi^\top y\right).
\]

For a new regressor `phi_*`, the posterior predictive distribution is

\[
y_* \mid \mathcal D, \phi_* \sim
\mathcal N\left(
\phi_*^\top \mu_N,
\phi_*^\top \Sigma_N \phi_* + \sigma^2
\right).
\]

That predictive variance contains both:

- **noise variance** `sigma2`, and
- **parameter uncertainty** `phi_*^T Sigma_N phi_*`.

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
│       ├── arx.py
│       └── simulate.py
└── tests/
    └── test_arx.py
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

## Quick demo

```bash
python examples/demo_arx.py
```

This demo will:

- simulate data from a stable ARX process,
- fit both least-squares and Bayesian ARX models,
- compare one-step predictions,
- plot the posterior predictive density for the next output,
- plot Monte Carlo trajectory bands from posterior samples.

## Main class

```python
from bayes_sysid import BayesianARX

model = BayesianARX(na=2, nb=2, sigma2=0.05)
model.fit(y, u)

mean, var = model.predict_next_distribution(y_hist, u_hist)
print(mean, var)
```

## Notes

This repo is intentionally small and pedagogical. Natural extensions include:

- unknown noise variance with inverse-gamma prior,
- full MIMO/state-space Bayesian identification,
- stability-constrained priors,
- Bayesian model order selection,
- impulse-response priors / Gaussian-process system identification.
