# Core Concepts (Pedagogical)

AF partitions and ordered K₀
- ReLU gates define polyhedral partitions that refine with depth.
- Incidences B_k encode parent→child relations; masses τ_k aggregate sample weights per cell.
- Finite snapshots form a Bratteli diagram; the inductive limit is AF and K₀ tracks refinement statistics.

CP maps and Stinespring
- Nonlinearities/averaging act completely positively on observables.
- Helix builds V from (B_k, τ_k): V[p,j] ≈ √(τ_k[j] / τ_{k-1}[p]) when B[p,j]=1.
- Diagnose unitality (Φ(I)=I), PSD preservation, and coisometry (V V* ≈ I).

Flows and crossed products
- Residual/invertible blocks act as *‑automorphisms.
- Ulam PF discretization approximates the Perron–Frobenius operator.
- Spectral gap 1 − |λ₂| serves as a practical mixing proxy.

Stabilization and Morita
- Channel/width changes correspond to matrix stabilization.
- K‑invariants and traces are robust to many width/basis changes.

Data structures and performance
- Sparse parents (parent_of) compress refinements with O(n) memory and fast aggregation.
- Convert on demand with B_from_parents / parents_from_B.

