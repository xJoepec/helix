"""Stable helper APIs used by external training environments.

The goal of this module is to provide light-weight, numpy-first data structures
that encapsulate the outputs of the Helix extraction pipeline. Environments can
import these helpers without touching internal dataclasses or mutating
implementation details inside :mod:`helix.partitions`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from .capacity import CapacityLossMetrics, compute_capacity_loss
from .cp import build_V_from_incidence, sanity_check_ucp
from .ktheory import k_invariants_from_B
from .partitions import AFExtraction, extract_partitions
from .topology import PersistentHomologySummary, compute_persistent_homology
from .ulam import spectral_gap

ArrayLike = np.ndarray


@dataclass(frozen=True)
class AFCPDiagnostics:
    """Compact CP-map health indicators for an AF level."""

    unital_err_fro: float
    coisometry_err_fro: float
    psd_min_eig_violation: float


@dataclass(frozen=True)
class AFLevelMetrics:
    """Summary statistics for a single partition depth.

    Attributes
    ----------
    depth:
        Depth index (1-based).
    B:
        Incidence matrix :math:`B_k` with shape ``(n_{k-1}, n_k)``.
    tau_prev:
        Mass vector :math:`\tau_{k-1}` (``(n_{k-1},)``). ``[1.0]`` when ``depth == 1``.
    tau:
        Mass vector :math:`\tau_k` (``(n_k,)``).
    mass_error:
        L1 residual ``||tau_prev - B @ tau||_1``.
    trace_residual_linf:
        ``L_\\infty`` residual ``||tau_prev - B @ tau||_\\infty``.
    wasted_regions:
        Number of zero (or near-zero) mass regions at this depth.
    n_regions:
        Cardinality of the partition at depth ``k``.
    combinatorial_entropy:
        ``(1 / depth) * log(n_regions)``; ``0.0`` when ``n_regions == 0``.
    cp_diagnostics:
        Optional CP-map diagnostics derived from :math:`B_k` and masses.
    persistent_homology:
        Optional persistent homology summary of the partition structure.
    capacity_metrics:
        Optional capacity loss metrics for trainable parameters.
    spectral_gap:
        Optional spectral gap from Ulam-Perron-Frobenius analysis.
    k_theory_invariants:
        Optional K-theory invariants (torsion, rank, nullity) from I - B^T.
    """

    depth: int
    B: ArrayLike
    tau_prev: ArrayLike
    tau: ArrayLike
    mass_error: float
    trace_residual_linf: float
    wasted_regions: int
    n_regions: int
    combinatorial_entropy: float
    cp_diagnostics: Optional[AFCPDiagnostics]
    persistent_homology: Optional[PersistentHomologySummary] = None
    capacity_metrics: Optional[CapacityLossMetrics] = None
    spectral_gap: Optional[float] = None
    k_theory_invariants: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class AFMetrics:
    """Container aggregating metrics for every depth discovered."""

    levels: Sequence[AFLevelMetrics]
    extraction: AFExtraction

    def __iter__(self) -> Iterable[AFLevelMetrics]:  # pragma: no cover - convenience wrapper
        return iter(self.levels)

    def __len__(self) -> int:  # pragma: no cover - convenience wrapper
        return len(self.levels)


def extract_af_metrics(
    model: Any,
    X: np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
    *,
    mass_tol: float = 1e-10,
    compute_ph: bool = False,
    compute_capacity: bool = False,
    compute_spectral_gap: bool = False,
    compute_k_theory: bool = False,
    ph_maxdim: int = 2,
    ph_sample_cap: int = 500,
    show_progress: bool = False,
) -> AFMetrics:
    """Run the Helix AF extraction and return per-depth metrics.

    Parameters
    ----------
    model:
        PyTorch module (typed loosely to avoid strict torch dependency at import).
    X:
        Input samples ``(N, d)`` stored as ``np.float32``/``np.float64``.
    sample_weights:
        Optional non-negative weights that sum to 1.
    mass_tol:
        Threshold used to count zero-mass regions.
    compute_ph:
        Whether to compute persistent homology for each level.
    compute_capacity:
        Whether to compute capacity loss metrics.
    compute_spectral_gap:
        Whether to compute spectral gaps via Ulam discretization.
    compute_k_theory:
        Whether to compute K-theory invariants.
    ph_maxdim:
        Maximum homology dimension for PH computation.
    ph_sample_cap:
        Maximum samples for PH computation.
    show_progress:
        Whether to display progress bars during long computations.
    """

    extraction = extract_partitions(model, X, sample_weights=sample_weights)

    # Compute global capacity metrics once if requested
    global_capacity_metrics = None
    if compute_capacity:
        try:
            global_capacity_metrics = compute_capacity_loss(model)
        except Exception:
            global_capacity_metrics = None

    levels: list[AFLevelMetrics] = []
    for idx, (B, tau) in enumerate(zip(extraction.B_list, extraction.tau_list), start=1):
        if idx == 1:
            tau_prev = np.array([1.0], dtype=np.float64)
        else:
            tau_prev = np.array(extraction.tau_list[idx - 2], dtype=np.float64, copy=True)
        B_copy = np.array(B, dtype=np.float64, copy=True)
        tau_copy = np.array(tau, dtype=np.float64, copy=True)
        residual = tau_prev - B_copy @ tau_copy if B_copy.size else tau_prev.copy()
        residual = residual.astype(np.float64, copy=False)
        mass_error_l1 = float(np.abs(residual).sum())
        trace_residual_linf = float(np.max(np.abs(residual))) if residual.size else 0.0
        wasted = int(np.count_nonzero(tau_copy <= mass_tol))
        n_regions = int(B_copy.shape[1])
        comb_entropy = 0.0
        if n_regions > 0:
            comb_entropy = float(np.log(n_regions) / idx)
        cp_diag: Optional[AFCPDiagnostics] = None
        if n_regions > 0:
            try:
                V = build_V_from_incidence(B_copy, tau_prev, tau_copy)
                stats = sanity_check_ucp(V, trials=4)
                cp_diag = AFCPDiagnostics(
                    unital_err_fro=float(stats.get("unital_err_fro", float("nan"))),
                    coisometry_err_fro=float(stats.get("coisometry_err_fro", float("nan"))),
                    psd_min_eig_violation=float(
                        stats.get("psd_min_eig_violation", float("nan"))
                    ),
                )
            except Exception:
                cp_diag = None

        # Compute persistent homology if requested
        ph_summary = None
        if compute_ph and n_regions > 2:
            try:
                # Extract representative points from AF partition cells
                representative_points = _extract_representative_points(
                    extraction.parts[idx - 1], X, ph_sample_cap
                )
                if len(representative_points) > 2:
                    ph_summary = compute_persistent_homology(
                        representative_points, maxdim=ph_maxdim, sample_cap=ph_sample_cap,
                        show_progress=show_progress
                    )
            except Exception:
                ph_summary = None

        # Compute spectral gap if requested
        spectral_gap_value = None
        if compute_spectral_gap and n_regions > 1:
            try:
                from .ulam import ulam_pf
                # Create a simple box around the data for Ulam discretization
                data_min = np.min(X, axis=0)
                data_max = np.max(X, axis=0)
                box = (data_min, data_max)

                # Simple identity map for testing spectral properties
                def identity_map(x):
                    return x

                # Use relatively small grid to avoid memory issues
                bins = min(10, int(np.sqrt(n_regions)))
                P, _ = ulam_pf(
                    identity_map,
                    box,
                    bins_per_dim=bins,
                    samples_per_cell=1,
                    show_progress=show_progress,
                )
                spectral_gap_value = spectral_gap(P, show_progress=show_progress)
            except Exception:
                spectral_gap_value = None

        # Compute K-theory invariants if requested
        k_theory_invariants = None
        if compute_k_theory and n_regions > 0:
            try:
                k_theory_invariants = k_invariants_from_B(
                    B_copy.astype(int),
                    show_progress=show_progress,
                )
            except Exception:
                k_theory_invariants = None

        level = AFLevelMetrics(
            depth=idx,
            B=B_copy,
            tau_prev=tau_prev,
            tau=tau_copy,
            mass_error=mass_error_l1,
            trace_residual_linf=trace_residual_linf,
            wasted_regions=wasted,
            n_regions=n_regions,
            combinatorial_entropy=comb_entropy,
            cp_diagnostics=cp_diag,
            persistent_homology=ph_summary,
            capacity_metrics=global_capacity_metrics,
            spectral_gap=spectral_gap_value,
            k_theory_invariants=k_theory_invariants,
        )
        levels.append(level)

    return AFMetrics(levels=tuple(levels), extraction=extraction)


def _extract_representative_points(
    partition_level,
    X: np.ndarray,
    max_points: int = 500
) -> list[list[float]]:
    """Extract representative points from AF partition cells for persistent homology.

    This function creates a point cloud suitable for persistent homology computation
    by sampling representatives from each cell in the AF partition.
    """
    representative_points = []

    # Get cell information
    cells = partition_level.cells

    for cell_indices in cells:
        if len(cell_indices) == 0:
            continue

        # Sample points from this cell
        cell_data = X[cell_indices]

        if len(cell_data) == 1:
            # Single point, add it directly
            representative_points.append(cell_data[0].tolist())
        else:
            # Multiple points, use centroid as representative
            centroid = np.mean(cell_data, axis=0)
            representative_points.append(centroid.tolist())

            # Also add a few actual points for topological richness
            if len(cell_data) > 3:
                # Add boundary points (min/max along first dimension)
                sorted_indices = np.argsort(cell_data[:, 0])
                representative_points.append(cell_data[sorted_indices[0]].tolist())
                representative_points.append(cell_data[sorted_indices[-1]].tolist())

    # Limit total points to avoid memory issues
    if len(representative_points) > max_points:
        # Random subsample
        indices = np.random.choice(len(representative_points), max_points, replace=False)
        representative_points = [representative_points[i] for i in indices]

    return representative_points


def af_feature_vector(level: AFLevelMetrics) -> np.ndarray:
    """Convert :class:`AFLevelMetrics` into a compact numeric feature vector."""

    def _safe(seq: Sequence[float], idx: int, default: float = 0.0) -> float:
        return float(seq[idx]) if len(seq) > idx else float(default)

    # Core features
    features = [
        float(level.depth),
        float(level.n_regions),
        float(level.mass_error),
        float(level.trace_residual_linf),
        float(level.wasted_regions),
        float(level.combinatorial_entropy),
    ]

    # CP map diagnostics
    features.extend([
        float(level.cp_diagnostics.unital_err_fro if level.cp_diagnostics else 0.0),
        float(level.cp_diagnostics.coisometry_err_fro if level.cp_diagnostics else 0.0),
        float(level.cp_diagnostics.psd_min_eig_violation if level.cp_diagnostics else 0.0),
    ])

    # Persistent homology features
    if level.persistent_homology and level.persistent_homology.computed:
        ph = level.persistent_homology
        betti = ph.betti_numbers
        avg = ph.average_lifetimes
        maxima = ph.max_lifetimes
        features.extend(
            [
                _safe(betti, 0),
                _safe(betti, 1),
                _safe(betti, 2),
                _safe(avg, 1),
                _safe(maxima, 1),
            ]
        )
    else:
        features.extend([0.0, 0.0, 0.0, 0.0, 0.0])

    # Capacity metrics
    if level.capacity_metrics and level.capacity_metrics.computed:
        features.extend([
            float(level.capacity_metrics.mean_loss),
            float(level.capacity_metrics.max_loss),
        ])
    else:
        features.extend([0.0, 0.0])

    # Spectral gap
    features.append(float(level.spectral_gap if level.spectral_gap is not None else 0.0))

    # K-theory invariants
    if level.k_theory_invariants:
        features.extend([
            float(level.k_theory_invariants.get('rank', 0)),
            float(level.k_theory_invariants.get('nullity', 0)),
            float(len(level.k_theory_invariants.get('torsion', []))),
        ])
    else:
        features.extend([0.0, 0.0, 0.0])

    return np.array(features, dtype=np.float64)


__all__ = [
    "AFCPDiagnostics",
    "AFLevelMetrics",
    "AFMetrics",
    "af_feature_vector",
    "extract_af_metrics",
]
