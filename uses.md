Helix Use Cases (Intuitive & Pedagogical)

1) Understand how a ReLU net carves space
- Goal: visualize and quantify region growth across depth.
- How: run the CLI or API to extract partitions; plot or print `region_counts(B_list)`.
- Why: rapid growth can indicate over‑complexity; stable counts suggest smoother decision geometry.

2) Check mass transport consistency layer‑by‑layer
- Goal: verify that empirical masses at depth k−1 match aggregated masses from depth k.
- How: compute `mass_consistency_errors(B_list, tau_list)`; smaller is better.
- Why: flags sampling artifacts or extraction issues; useful when using non‑uniform sample weights.

3) Build a CP map from observed refinements
- Goal: model nonlinear/channel mixing with a unital CP map on observables.
- How: `V = build_V_from_incidence(B_k, τ_{k-1}, τ_k)` then `sanity_check_ucp(V)`.
- Why: unitality/PSD/coisometry proxies surface numerical stability and structural issues.

4) Probe mixing of invertible/residual blocks
- Goal: estimate a practical mixing proxy via Ulam PF.
- How: define a map `F` (e.g., residual block), run `ulam_pf(F, (lo,hi), bins, samples)`, then `spectral_gap(P)`.
- Why: larger gaps (1−|λ₂|) typically indicate better mixing; helpful for flow‑like architectures.

5) Sparse structure for scale
- Goal: handle large refinements efficiently.
- How: prefer `parent_of_list` over dense B; convert on demand with `B_from_parents`.
- Why: O(n) memory, fast aggregation, simple persistence.

6) Training‑time diagnostics
- Goal: track geometry as you train.
- How: after each epoch or checkpoint, extract (on a fixed probe set) and log region counts, mass L1, CP checks.
- Why: catch regressions early (e.g., exploding regions, poor mass consistency, CP violations).

7) Toy K‑theory experiments
- Goal: explore stationary systems’ algebraic fingerprints.
- How: for a fixed B, inspect `k_invariants_from_B(B)` (Smith normal form of I − Bᵗ).
- Why: learn how torsion/rank/nullity change under architectural motifs.

8) Interactive exploration with the TUI
- Goal: quickly iterate on parameters and see results.
- How: `pip install -e .[tui]` then `helix tui`; tweak samples/noise/ulam bins; export JSON.
- Why: lowers friction for demos, teaching, and ablation studies.

Tips
- Use a fixed probe set X for apples‑to‑apples comparisons across epochs/architectures.
- Start with low `--ulam-bins` and increase as needed.
- When τ has zeros, Helix’s V construction masks divisions; expect coisometry to be an approximate proxy.
