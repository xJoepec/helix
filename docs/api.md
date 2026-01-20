# API Surface

## Re-exported from `helix`

Partitions
- extract_partitions(model, X[, sample_weights]) -> AFExtraction
- AFExtraction: B_list, tau_list, parts, n_list, parent_of_list
- PartitionLevel: cell_of, cells, signatures

CP maps
- build_V_from_incidence(B, tau_prev, tau_cur) -> V
- cp_embed_apply(V, X) -> V* X V
- sanity_check_ucp(V, trials=6) -> {unital_err_fro, coisometry_err_fro, psd_min_eig_violation}

Ulam PF
- ulam_pf(F, (lo, hi), bins_per_dim, samples_per_cell=1, show_progress=False, use_vectorized=True) -> (P, axes)
- spectral_gap(P, k=5, show_progress=False, use_sparse=True) -> float

Diagnostics
- region_counts(B_list) -> List[int]
- mass_consistency_errors(B_list, tau_list) -> List[float]
- cumulative_anisotropy(B_list) -> np.ndarray

Sparse helpers
- parents_from_B(B) -> parents
- B_from_parents(parents, n_prev) -> B

K-theory
- smith_normal_form_Z(M, show_progress=False) -> dict (U, S_diag, V, torsion, rank, nullity, nullspace_Q)
- k_invariants_from_B(B, show_progress=False) -> smith data for I - B^T

Topology
- compute_persistent_homology(points, maxdim=2, sample_cap=1024, show_progress=False) -> PersistentHomologySummary
- PersistentHomologySummary: betti_numbers, average_lifetimes, max_lifetimes, finite_pairs, computed, backend, notes

Capacity
- compute_capacity_loss(model, singular_value_threshold=1e-3) -> CapacityLossMetrics
- CapacityLossMetrics: layer_scores, mean_loss, max_loss, nontrainable_layers, total_layers, computed, notes

Environment API
- extract_af_metrics(
    model, X, sample_weights=None, mass_tol=1e-10,
    compute_ph=False, compute_capacity=False, compute_spectral_gap=False, compute_k_theory=False,
    ph_maxdim=2, ph_sample_cap=500, show_progress=False
  ) -> AFMetrics
- AFMetrics: levels, extraction
- AFLevelMetrics: depth, B, tau_prev, tau, mass_error, trace_residual_linf, wasted_regions,
  n_regions, combinatorial_entropy, cp_diagnostics, persistent_homology, capacity_metrics,
  spectral_gap, k_theory_invariants
- af_feature_vector(level) -> np.ndarray

## Module-level APIs (not re-exported)

Tensor-train Ulam (`helix.ulam_tensor`)
- TensorTrainCore, TensorTrainOperator, tt_cross_approximation
- tensor_ulam_pf(F, box, bins_per_dim=20, max_rank=10, tolerance=1e-6) -> (TensorTrainOperator, centers)
- spectral_gap_tt(P_tt, num_eigenvalues=5) -> float
- enhanced_ulam_pf(F, box, bins_per_dim=20, use_tensor_train=True, max_rank=10, **kwargs)
  -> (P, centers, gap)

Vision Transformer adapter (`helix.architectures.vit_adapter`)
- ViTAdapter(...)
- AttentionPartition, ViTPartitionExtraction
- extract_vit_partitions(model, X, sample_weights=None, **adapter_kwargs) -> ViTPartitionExtraction
- interpret_attention_as_gauge_field(partition) -> dict

K-theory helpers (`helix.ktheory`)
- k_invariants_hodge(B, tolerance=1e-10, show_progress=False) -> dict
- compute_persistence_k_theory(B_sequence, tolerance=1e-10) -> dict
