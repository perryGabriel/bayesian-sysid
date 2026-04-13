# Final Report: Bayesian ARX System Identification with Uncertainty-Aware Analysis

**Course:** CS 677  
**Project repository:** `bayesian-sysid`  
**Suggested export settings for submission PDF:** 12pt font, 1-inch margins, single spacing, 4–5 pages.

---

## Abstract

This project studies Bayesian system identification for linear discrete-time dynamical systems using ARX models. The central goal is to move beyond point estimation and preserve uncertainty through the entire modeling pipeline: prediction, stability assessment, frequency-domain interpretation, and closed-loop performance analysis. We implement a compact Python toolkit that supports Bayesian and least-squares ARX fitting, posterior predictive inference, prior construction helpers, model-order search, and control-oriented post-processing. Experiments and visualizations demonstrate that Bayesian uncertainty is especially valuable in low-data regimes, where deterministic estimates often appear overconfident. We show that posterior distributions provide interpretable risk information, such as stability probability and closed-loop response spread, which is essential for responsible control design.

---

## 1. Problem statement, background, and motivation

### 1.1 Problem statement

Given observed input/output trajectories from an unknown dynamical system, we want to identify a model that predicts future outputs and is useful for control analysis. The specific modeling family is ARX:

$$
y_{t} = \sum_{i=1}^{n_a} a_i y_{t-i} + \sum_{j=1}^{n_b} b_j u_{t-j} + e_{t}.
$$

Classical least-squares ARX estimation returns a single parameter vector $\hat\theta$. In many realistic settings (limited data, noisy measurements, weak input excitation), this can yield unstable or misleading downstream conclusions because uncertainty is ignored.

### 1.2 Background

System identification has a long history in control, where parametric linear models like ARX are valued for interpretability and computational simplicity. In Bayesian inference, uncertain quantities are represented as random variables with prior distributions updated by data. For linear-Gaussian regression, conjugacy gives a closed-form Gaussian posterior over coefficients. This makes Bayesian ARX practical: we can retain tractability while obtaining calibrated uncertainty estimates.

Related ideas in Bayesian regression and probabilistic forecasting emphasize predictive distributions rather than single predictions. In control contexts, uncertainty-aware modeling aligns with robust design goals: we care not only about nominal behavior but about variability and tail risks.

### 1.3 Motivation

This project is worth studying for three reasons:

1. **Practical reliability:** Engineers often tune controllers on nominal models only; parameter uncertainty can invalidate margins and performance claims.
2. **Interpretability:** ARX remains easy to understand, and Bayesian posteriors produce direct confidence bands and probabilities.
3. **Pedagogical value:** The model links class topics (Bayesian inference, linear algebra, predictive uncertainty) to control-relevant outputs (poles, Bode-like envelopes, closed-loop trajectories).

---

## 2. Methodology: what we implemented and why it is Bayesian

### 2.1 Core Bayesian ARX model

Let $\phi_{t}$ denote the regressor of lagged outputs and inputs. We define

$$
y_{t} = \phi_{t}^\top\theta + e_{t}, \quad e_{t}\sim \mathcal{N}(0,\sigma^2),
$$

with prior

$$
\theta\sim\mathcal{N}(\mu_0,\Sigma_0).
$$

For stacked data matrix $\Phi$ and target vector $y$, conjugate updating yields

$$
\Sigma_N^{-1}=\Sigma_0^{-1}+\frac{1}{\sigma^2}\Phi^\top\Phi,
\qquad
\mu_N=\Sigma_N\left(\Sigma_0^{-1}\mu_0+\frac{1}{\sigma^2}\Phi^\top y\right).
$$

The one-step predictive distribution at $\phi_{\ast}$ is

$$
y_{\ast}\mid\mathcal D,\phi_{\ast}\sim \mathcal N\left(\phi_{\ast}^\top\mu_N,\;\phi_{\ast}^\top\Sigma_N\phi_{\ast}+\sigma^2\right).
$$

This is explicitly Bayesian because uncertainty in $\theta$ is maintained and integrated into predictions.

### 2.2 Unknown-noise extension

We also include a variant where noise variance is not fixed in advance. Marginalizing this uncertainty leads to heavier-tailed predictive behavior (Student-t), improving robustness when data are scarce or outliers exist.

### 2.3 Prior design utilities

We implemented helper utilities to construct priors that are:

- isotropic (uniform scale across parameters),
- diagonal with different scales for autoregressive vs input terms,
- normalized with regressor-scale considerations.

These utilities are important in Bayesian modeling because priors encode inductive bias and regularization. In low-data settings, prior quality strongly influences posterior quality.

### 2.4 Model order selection

We added rolling-origin order search for `(na, nb)` to reduce overfitting and better emulate forecasting usage. Instead of evaluating only in-sample fit, rolling evaluation checks generalization under temporal ordering.

### 2.5 Control-oriented uncertainty propagation

To connect identification to control analysis, we implemented:

- **posterior stability probability** (fraction of posterior samples yielding stable poles),
- **pole-cloud visualization** in the complex plane,
- **frequency-response envelopes** from posterior samples,
- **closed-loop Monte Carlo response bands** under sampled models and fixed controllers,
- **nominal + empirical robustness margin summaries**.

These components answer the practical question: *How does identification uncertainty change engineering conclusions?*

---

## 3. Experimental setup and results

### 3.1 Data and scenarios

We evaluate through provided demos and tests, focusing on synthetic linear systems where ground-truth behavior is controlled. This lets us isolate effects of sample size and noise. We compare:

- least-squares ARX (point estimate),
- Bayesian ARX with posterior predictive uncertainty.

### 3.2 Prediction quality and uncertainty

In moderate-to-high data regimes, both methods can fit trajectories similarly in mean behavior. The key difference appears in predictive uncertainty:

- Bayesian predictive variance shrinks with informative data.
- Under low data, intervals widen appropriately and communicate uncertainty.
- Point-estimate methods may look accurate on average while underreporting risk.

The notebook artifact `predictive_density_annotated.png` illustrates this distinction: posterior predictive density communicates plausible output ranges rather than a single line.

### 3.3 Stability and poles

Posterior sampling yields a cloud of plausible poles. Instead of binary “stable/unstable” claims from one estimate, we obtain a probability of stability. Results show:

- datasets with stronger excitation typically produce tighter pole clouds and higher stability confidence,
- undersampled cases create broader clouds crossing critical boundaries.

This is useful for decision-making: a model with 55% stability probability should be treated differently from one with 99%, even if their posterior means are both stable.

### 3.4 Frequency-domain interpretation

Sampling posterior parameters and plotting frequency responses produces an envelope rather than one transfer curve. This reveals where uncertainty is concentrated (often higher frequencies or weakly excited regions). Engineers can then design controllers conservatively in bands with larger spread.

### 3.5 Closed-loop Monte Carlo analysis

When the identified model feeds into closed-loop simulation (static or PID-like controllers), posterior uncertainty induces a distribution over transient responses. We observe:

- nominal trajectories can hide occasional high-overshoot outcomes,
- uncertainty bands provide clearer risk communication,
- robust margin summaries can differ substantially from nominal-only margin values.

Together, these outcomes motivate uncertainty-aware controller assessment.

---

## 4. Discussion and interpretation

### 4.1 What the results mean

The strongest conclusion is that Bayesian ARX provides a *qualitatively better decision signal* than point identification when uncertainty matters. Even when mean prediction errors look close, downstream control quantities (stability, margins, transient behavior) can have wide uncertainty.

This suggests a practical workflow:

1. fit Bayesian ARX,
2. inspect posterior predictive calibration,
3. evaluate stability probability and frequency-response spread,
4. run closed-loop Monte Carlo before final tuning.

### 4.2 Connection to Bayesian inference

This project directly applies Bayesian principles:

- prior specification (regularization and beliefs),
- likelihood from ARX residual model,
- posterior computation in closed form (or conjugate extension),
- posterior predictive integration for uncertainty-aware forecasting.

The control analyses are not separate from Bayes—they are derived from posterior samples, so every conclusion is conditioned on uncertainty in identified parameters.

### 4.3 Strengths and limitations

**Strengths**

- Fast, interpretable linear model class.
- Closed-form updates for core model.
- Meaningful uncertainty outputs aligned with control needs.

**Limitations**

- ARX is linear and may miss nonlinear dynamics.
- Current scope is mostly SISO and pedagogical.
- Robustness summaries are preliminary compared to full structured robust control toolchains.

### 4.4 Future work

- MIMO Bayesian identification and structure learning.
- Explicit state-space Bayesian models and observer synthesis.
- Sequential/online Bayesian updates for adaptive control.
- Better uncertainty calibration diagnostics and benchmark datasets.

---

## 5. Conclusion

We built and documented a coherent Bayesian ARX identification pipeline that extends naturally into uncertainty-aware control analysis. The project demonstrates that Bayesian inference is not just a statistical add-on: it materially changes engineering interpretation by quantifying confidence in predictions, stability, and closed-loop behavior. For CS 677-level practice, this offers a clear bridge from course Bayesian foundations to actionable system-identification workflows.

---

## References

1. T. Söderström and P. Stoica, *System Identification*, Prentice Hall, 1989.  
2. L. Ljung, *System Identification: Theory for the User*, 2nd ed., Prentice Hall, 1999.  
3. C. M. Bishop, *Pattern Recognition and Machine Learning*, Springer, 2006.  
4. K. P. Murphy, *Machine Learning: A Probabilistic Perspective*, MIT Press, 2012.  
5. G. C. Goodwin and K. S. Sin, *Adaptive Filtering Prediction and Control*, Dover, 2009.

---

## Appendix: Suggested export command

If you want a submission-ready PDF that matches the rubric constraints, one option is:

```bash
pandoc docs/final_report.md -o docs/final_report.pdf \
  -V geometry:margin=1in -V fontsize=12pt -V linestretch=1
```

(If your local Pandoc/LaTeX setup differs, use equivalent options to enforce 12pt, 1-inch margins, single spacing.)
