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
- ulam_pf(F, (lo, hi), bins_per_dim, samples_per_cell=1, show_progress=False) → (P, axes)
- spectral_gap(P, show_progress=False) → float

Tensor Train Ulam (High-dimensional)
- tensor_ulam_pf(F, box, bins_per_dim, max_rank, tolerance) → (TensorTrainOperator, centers)
- spectral_gap_tt(P_tt, num_eigenvalues=5) → float
- enhanced_ulam_pf(F, box, bins_per_dim, use_tensor_train=True, max_rank=10) → (P, centers, gap)

Diagnostics
- region_counts(B_list) → List[int]
- mass_consistency_errors(B_list, τ_list) → List[float]
- cumulative_anisotropy(B_list) → np.ndarray

Sparse helpers
- parents_from_B(B) → parents
- B_from_parents(parents, n_prev) → B

K‑theory
- smith_normal_form_Z(M, show_progress=False) → dict (U, S_diag, V, torsion, rank, nullity, nullspace_Q)
- k_invariants_from_B(B, show_progress=False) → smith data for I − B^T
- k_invariants_hodge(B, tolerance=1e-10, show_progress=False) → enhanced K-theory via Hodge decomposition

Topology
- compute_persistent_homology(points, maxdim=2, sample_cap=1024, show_progress=False) → PersistentHomologySummary
- PersistentHomologySummary: betti_numbers, average_lifetimes, max_lifetimes, finite_pairs, computed, backend

Vision Transformer Analysis
- extract_vit_partitions(model, X, target_layers=None) → ViTPartitionExtraction
- ViTAdapter: register_attention_hooks(), extract_attention_partitions()
- AttentionPartition: compute_attention_entropy(), compute_gauge_field_strength(), to_af_partition_metrics()

Environment API
- extract_af_metrics(model, X, compute_ph=False, compute_capacity=False, compute_spectral_gap=False, compute_k_theory=False, show_progress=False) → AFMetrics
- AFMetrics: levels, global_capacity, global_persistent_homology
- AFLevelMetrics: depth, B, tau_prev, tau_cur, n_regions, mass_error, cp_diagnostics, persistent_homology, capacity_metrics, spectral_gap, k_theory_invariants

