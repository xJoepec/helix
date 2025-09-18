from __future__ import annotations

from typing import List

import numpy as np


def region_counts(B_list: List[np.ndarray]) -> List[int]:
    """Return region counts n_k from incidence list (n_{k-1} x n_k)."""
    return [B.shape[1] for B in B_list]


def mass_consistency_errors(B_list: List[np.ndarray], tau_list: List[np.ndarray]) -> List[float]:
    """Compute L1 errors of τ_{k-1} - B_k τ_k for k >= 1, with τ_0 := [1.0]."""
    errs: List[float] = []
    for k, (B, tau_k) in enumerate(zip(B_list, tau_list), start=1):
        tau_prev = np.array([1.0]) if k == 1 else tau_list[k - 2]
        lhs = tau_prev
        rhs = B @ tau_k
        errs.append(float(np.linalg.norm(lhs - rhs, 1)))
    return errs


def cumulative_anisotropy(B_list: List[np.ndarray]) -> np.ndarray:
    """Column sums of cumulative product B_1 ... B_k as an anisotropy proxy.

    Returns vector of column sums for the deepest cumulative product.
    """
    Bcum = None
    for B in B_list:
        Bcum = B if Bcum is None else Bcum @ B
    if Bcum is None:
        return np.array([], dtype=np.int64)
    return Bcum.sum(axis=0)
