# API Surface

Import from `helix` (re-exports):

Partitions
- extract_partitions(model, X[, sample_weights]) → AFExtraction
- AFExtraction: B_list, tau_list, parts, n_list, parent_of_list
- PartitionLevel: cell_of, cells, signatures

CP maps
- build_V_from_incidence(B, τ_prev, τ_cur) → V
- cp_embed_apply(V, X) → V* X V
- sanity_check_ucp(V, trials=6) → {unital_err_fro, coisometry_err_fro, psd_min_eig_violation}

Ulam PF
- ulam_pf(F, (lo, hi), bins_per_dim, samples_per_cell=1) → (P, axes)
- spectral_gap(P) → float

Diagnostics
- region_counts(B_list) → List[int]
- mass_consistency_errors(B_list, τ_list) → List[float]
- cumulative_anisotropy(B_list) → np.ndarray

Sparse helpers
- parents_from_B(B) → parents
- B_from_parents(parents, n_prev) → B

K‑theory
- smith_normal_form_Z(M) → dict (U, S_diag, V, torsion, rank, nullity, nullspace_Q)
- k_invariants_from_B(B) → smith data for I − B^T

