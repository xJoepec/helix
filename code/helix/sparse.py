from __future__ import annotations

from typing import List

import numpy as np


def parents_from_B(B: np.ndarray) -> np.ndarray:
    """Infer parent pointers from a 0/1 incidence matrix B (n_prev x n_cur).

    For columns with a single 1, return the row index as the parent.
    Columns with all zeros get parent -1 (no parent). If a column has
    multiple ones, the argmax row is chosen as a best-effort fallback.
    """
    if B.ndim != 2:
        raise ValueError("B must be a 2D array")
    n_prev, n_cur = B.shape
    B = B.astype(np.int8, copy=False)
    col_sums = B.sum(axis=0)
    parents = np.full(n_cur, -1, dtype=np.int64)
    if n_cur == 0:
        return parents
    # argmax fallback handles tie-breaking deterministically
    argmax_rows = B.argmax(axis=0)
    has_parent = col_sums > 0
    parents[has_parent] = argmax_rows[has_parent]
    return parents


def B_from_parents(parents: np.ndarray, n_prev: int) -> np.ndarray:
    """Build a dense 0/1 incidence matrix B from parent pointers.

    parents[j] in {0..n_prev-1} points to the parent row of column j.
    parents[j] < 0 leaves column j all zeros.
    """
    parents = np.asarray(parents, dtype=np.int64)
    n_cur = int(parents.shape[0])
    B = np.zeros((n_prev, n_cur), dtype=np.int32)
    if n_cur == 0:
        return B
    valid = parents >= 0
    cols = np.nonzero(valid)[0]
    rows = parents[valid]
    B[rows, cols] = 1
    return B


def parents_memory_bytes(parents_list: List[np.ndarray]) -> int:
    """Total memory footprint in bytes for a list of parent arrays."""
    return sum(p.nbytes for p in parents_list)
