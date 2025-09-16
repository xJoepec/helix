from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np


def ulam_pf(
    F: Callable[[np.ndarray], np.ndarray],
    box: Tuple[np.ndarray, np.ndarray],
    bins_per_dim: int = 20,
    *,
    samples_per_cell: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Ulam–Perron–Frobenius discretization with barycentric mass splitting.

    - Builds a uniform grid over the box with `bins_per_dim` per axis.
    - For each source cell, samples `samples_per_cell` points (center if 1),
      pushes through `F`, and deposits mass into up to 2^d neighbor bins
      via multilinear (barycentric) weights. Rows are normalized at the end.

    Returns (P, centers_axes) where P is row-stochastic.
    """
    lo, hi = [np.asarray(v, dtype=np.float64) for v in box]
    d = lo.shape[0]
    edges = [np.linspace(lo[i], hi[i], bins_per_dim + 1) for i in range(d)]
    centers_axes = [0.5 * (edges[i][:-1] + edges[i][1:]) for i in range(d)]
    centers = np.array(np.meshgrid(*centers_axes, indexing="ij"))
    centers = centers.reshape(d, -1).T  # [M, d]
    M = centers.shape[0]

    # Grid cell widths per-dim (uniform)
    widths = (hi - lo) / float(bins_per_dim)
    if rng is None:
        rng = np.random.default_rng(0)

    # Prepare sampling points and their source rows
    q = int(max(1, samples_per_cell))
    if q == 1:
        samples = centers
        src_rows = np.arange(M, dtype=np.int64)
    else:
        # Jitter within each cell: uniform in [-0.5, 0.5] per dim scaled by widths
        jitter = rng.uniform(-0.5, 0.5, size=(M, q, d)) * widths
        samples = (centers[:, None, :] + jitter).reshape(-1, d)
        # Clip to box to avoid leaving domain
        samples = np.minimum(np.maximum(samples, lo), hi)
        src_rows = np.repeat(np.arange(M, dtype=np.int64), q)

    # Push samples through F and clamp to box
    Y = F(samples)
    Y = np.minimum(np.maximum(Y, lo), hi)

    # Accumulate into P with barycentric splitting
    P = np.zeros((M, M), dtype=np.float64)
    # Map multi-index (idx[0],...,idx[d-1]) to flat index consistent with
    # np.meshgrid(..., indexing="ij") followed by C-order flattening.
    # Axis 0 varies slowest, so use reversed powers.
    multipliers = np.array([bins_per_dim ** (d - 1 - i) for i in range(d)], dtype=np.int64)

    def deposit_point(y: np.ndarray, row: int):
        # Compute fractional bin coordinates per dimension
        # u in [0, bins], base index b=floor(u) clipped to [0, bins-1], r in [0,1)
        b = np.floor((y - lo) / widths).astype(np.int64)
        # Handle boundary at hi: y==hi -> put at last bin with r=0
        b = np.clip(b, 0, bins_per_dim - 1)
        r = ((y - lo) / widths) - b
        r = np.clip(r, 0.0, 1.0)
        r = np.where(b == (bins_per_dim - 1), 0.0, r)

        # Enumerate 2^d neighbors
        for mask in range(1 << d):
            idx = b.copy()
            w = 1.0
            valid = True
            for i in range(d):
                if (mask >> i) & 1:
                    idx[i] += 1
                    if idx[i] >= bins_per_dim:
                        valid = False
                        break
                    w *= r[i]
                else:
                    w *= (1.0 - r[i])
            if not valid or w == 0.0:
                continue
            tgt = int((idx * multipliers).sum())
            P[row, tgt] += w

    for y, row in zip(Y, src_rows):
        deposit_point(y, int(row))

    # Normalize rows (account for numeric drift and multi-sampling)
    if q > 1:
        P /= float(q)
    row_sums = P.sum(axis=1, keepdims=True)
    # Avoid division by zero for empty rows (shouldn't happen with our sampling)
    row_sums[row_sums == 0.0] = 1.0
    P /= row_sums

    centers_axes = [a.copy() for a in centers_axes]
    return P, centers_axes


def spectral_gap(P: np.ndarray, k: int = 5) -> float:
    """Return 1 - |λ2(P)| (magnitude of second-largest eigenvalue of P^T)."""
    eigs = np.linalg.eigvals(P.T)
    eigs = np.sort(np.abs(eigs))[::-1]
    if eigs.size < 2:
        return 0.0
    return float(1.0 - eigs[1])
