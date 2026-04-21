# DSF prototype equations

**Owner:** Bayes SysID maintainers  
**Last validated on:** 2026-04-21

- Transfer matrix from ARX lag-polynomials:
  $G(z) = A(z)^{-1}B(z)$, where $A(z)=I+\sum_{k=1}^{n_a}A_k z^{-k}$ and $B(z)=\sum_{k=1}^{n_b}B_k z^{-k}$.
- Prototype DSF factorization:
  $G(z)\approx (I-Q(z))^{-1}P(z)$ with $P(z)=\mathrm{diag}(G(z))$ and $Q(z)=I-P(z)G(z)^{-1}$.
