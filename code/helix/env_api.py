from __future__ import annotations

"""Stable helper APIs used by external training environments.

The goal of this module is to provide light-weight, numpy-first data structures
that encapsulate the outputs of the Helix extraction pipeline. Environments can
import these helpers without touching internal dataclasses or mutating
implementation details inside :mod:`helix.partitions`.
"""

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from .diagnostics import mass_consistency_errors
from .partitions import AFExtraction, extract_partitions

ArrayLike = np.ndarray


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
    wasted_regions:
        Number of zero (or near-zero) mass regions at this depth.
    n_regions:
        Cardinality of the partition at depth ``k``.
    combinatorial_entropy:
        ``(1 / depth) * log(n_regions)``; ``0.0`` when ``n_regions == 0``.
    """

    depth: int
    B: ArrayLike
    tau_prev: ArrayLike
    tau: ArrayLike
    mass_error: float
    wasted_regions: int
    n_regions: int
    combinatorial_entropy: float


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
    """

    extraction = extract_partitions(model, X, sample_weights=sample_weights)
    mass_errors = mass_consistency_errors(extraction.B_list, extraction.tau_list)

    levels: list[AFLevelMetrics] = []
    for idx, (B, tau, mass_err) in enumerate(
        zip(extraction.B_list, extraction.tau_list, mass_errors), start=1
    ):
        if idx == 1:
            tau_prev = np.array([1.0], dtype=np.float64)
        else:
            tau_prev = np.array(extraction.tau_list[idx - 2], dtype=np.float64, copy=True)
        B_copy = np.array(B, dtype=np.float64, copy=True)
        tau_copy = np.array(tau, dtype=np.float64, copy=True)
        wasted = int(np.count_nonzero(tau_copy <= mass_tol))
        n_regions = int(B_copy.shape[1])
        comb_entropy = 0.0
        if n_regions > 0:
            comb_entropy = float(np.log(n_regions) / idx)
        level = AFLevelMetrics(
            depth=idx,
            B=B_copy,
            tau_prev=tau_prev,
            tau=tau_copy,
            mass_error=float(mass_err),
            wasted_regions=wasted,
            n_regions=n_regions,
            combinatorial_entropy=comb_entropy,
        )
        levels.append(level)

    return AFMetrics(levels=tuple(levels), extraction=extraction)


def af_feature_vector(level: AFLevelMetrics) -> np.ndarray:
    """Convert :class:`AFLevelMetrics` into a compact numeric feature vector."""

    return np.array(
        [
            float(level.depth),
            float(level.n_regions),
            float(level.mass_error),
            float(level.wasted_regions),
            float(level.combinatorial_entropy),
        ],
        dtype=np.float64,
    )


__all__ = [
    "AFLevelMetrics",
    "AFMetrics",
    "af_feature_vector",
    "extract_af_metrics",
]
