# Observer/Bayesian Gramian equations

**Owner:** Bayes SysID maintainers  
**Last validated on:** 2026-04-21

Posterior realization samples: $(A^{(s)}, B^{(s)}, C^{(s)}, D^{(s)})$.

$W_c^{(s)} = A^{(s)} W_c^{(s)} A^{(s)\top} + B^{(s)} B^{(s)\top}$.

$W_o^{(s)} = A^{(s)\top} W_o^{(s)} A^{(s)} + C^{(s)\top} C^{(s)}$.

Kalman update: $\hat{x}_{k|k} = \hat{x}_{k|k-1} + K_k(y_k - C\hat{x}_{k|k-1})$.
