from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .sparse_ops import stable_mass_computation

# Import platform utilities for optimization
try:
    from .platform_utils import NUMBA_AVAILABLE
except ImportError:
    NUMBA_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
except Exception as _e:  # pragma: no cover
    torch = None
    nn = None


@dataclass
class PartitionLevel:
    # mapping from sample index -> cell id
    cell_of: np.ndarray  # shape: [N]
    # list of arrays of sample indices in each cell
    cells: List[np.ndarray]
    # sign pattern representatives for this depth (tuple of {0,1})
    signatures: List[Tuple[int, ...]]


@dataclass
class AFExtraction:
    # incidence matrices B_k (n_{k-1} x n_k), entries 0/1
    B_list: List[np.ndarray]
    # masses tau_k in simplex (length n_k)
    tau_list: List[np.ndarray]
    # empirical partition info per depth
    parts: List[PartitionLevel]
    # region counts n_k
    n_list: List[int]
    # sparse parent pointers per depth: parent_of_list[k-1] has length n_k
    parent_of_list: List[np.ndarray]


# Numba JIT compilation if available for signature generation
if NUMBA_AVAILABLE:
    try:
        from numba import jit

        @jit(nopython=True, cache=True)
        def _build_cumulative_signatures_numba(gates_list, N):
            """Numba-optimized cumulative signature building."""
            n_layers = len(gates_list)
            if n_layers == 0:
                return np.zeros((N, 0), dtype=np.uint8)

            # Calculate total width for cumulative signatures
            total_width = sum(g.shape[1] for g in gates_list)

            # Pre-allocate array for cumulative signatures
            cum_sigs = np.zeros((N, total_width), dtype=np.uint8)

            # Fill cumulative signatures
            offset = 0
            for g in gates_list:
                width = g.shape[1]
                cum_sigs[:, offset:offset+width] = g
                offset += width

            return cum_sigs

        USE_NUMBA_SIGS = True
    except ImportError:
        USE_NUMBA_SIGS = False
else:
    USE_NUMBA_SIGS = False


def _build_cumulative_signatures_vectorized(
    gates: np.ndarray,
    cum_prev: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Vectorized cumulative signature building using NumPy.

    Args:
        gates: Current layer gates (N x width)
        cum_prev: Previous cumulative signatures (N x prev_width) or None

    Returns:
        Cumulative signatures as (N x total_width) array
    """
    if cum_prev is None:
        # First layer - just return gates
        return gates.astype(np.uint8)
    else:
        # Concatenate with previous cumulative signatures
        return np.concatenate([cum_prev, gates.astype(np.uint8)], axis=1)


def _bucket_by_signature_vectorized(sigs: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Vectorized bucketing by signature using NumPy unique.

    Args:
        sigs: Signatures array (N x width)

    Returns:
        cells: List of arrays containing sample indices per unique signature
        unique_sigs: List of unique signatures (as arrays)
    """
    N = sigs.shape[0]

    # Convert signatures to unique identifiers
    # For small signatures, we can pack them into a single integer
    if sigs.shape[1] <= 64:  # Can fit in uint64
        # Pack bits into integers for fast comparison
        packed = np.zeros(N, dtype=np.uint64)
        for i in range(min(sigs.shape[1], 64)):
            packed |= sigs[:, i].astype(np.uint64) << i

        # Find unique packed signatures
        unique_packed, inverse_indices = np.unique(packed, return_inverse=True)

        # Group sample indices by signature
        cells = []
        unique_sigs = []
        for j in range(len(unique_packed)):
            mask = (inverse_indices == j)
            cells.append(np.where(mask)[0].astype(np.int32))

            # Reconstruct signature from packed value
            sig = np.zeros(sigs.shape[1], dtype=np.uint8)
            val = unique_packed[j]
            for i in range(sigs.shape[1]):
                sig[i] = (val >> i) & 1
            unique_sigs.append(sig)
    else:
        # For larger signatures, use row-wise unique
        # Convert each row to a tuple for hashing
        sig_tuples = [tuple(row) for row in sigs]

        # Use dictionary for bucketing (more memory but handles any size)
        buckets = {}
        for i, sig in enumerate(sig_tuples):
            if sig not in buckets:
                buckets[sig] = []
            buckets[sig].append(i)

        # Sort signatures for consistency
        sorted_sigs = sorted(buckets.keys(), key=lambda t: (len(t), t))

        cells = [np.array(buckets[sig], dtype=np.int32) for sig in sorted_sigs]
        unique_sigs = [np.array(sig, dtype=np.uint8) for sig in sorted_sigs]

    return cells, unique_sigs


def _relu_gates_from_module(mod: "nn.Module", x: "torch.Tensor") -> List[np.ndarray]:
    """Capture ReLU gate indicators as compact numpy arrays without autograd ties.

    Returns a list of uint8 numpy arrays (N x width) per ReLU layer.
    This avoids retaining tensors/graphs and mitigates GPU memory retention.
    """
    gates: List[np.ndarray] = []

    def hook(module, inp, out):
        # Convert immediately to cpu numpy and copy to avoid referencing freed storage
        z = inp[0].detach()
        g = (z > 0).to(torch.uint8).cpu().numpy()
        gates.append(g.copy())

    handles = []
    for m in mod.modules():
        if isinstance(m, nn.ReLU):
            handles.append(m.register_forward_hook(hook))

    with torch.no_grad():
        _ = mod(x)

    for h in handles:
        h.remove()
    return gates


def extract_partitions(
    model: "nn.Module",
    X: np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
    use_vectorized: bool = True,
) -> AFExtraction:
    """Empirically extract ReLU partitions and incidence from a PyTorch model.

    Optimized version with vectorized signature generation and bucketing.

    Args:
        model: PyTorch model with ReLU activations
        X: Input samples (N x input_dim)
        sample_weights: Optional sample weights
        use_vectorized: Whether to use optimized implementations (default True)

    Returns:
        AFExtraction with B_k, tau_k, and per-depth region info.
    """
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required for partition extraction.")

    model.eval()
    X_t = torch.from_numpy(X.astype(np.float32))
    gates = _relu_gates_from_module(model, X_t)  # per ReLU layer (numpy uint8 arrays)
    N = X.shape[0]
    if sample_weights is None:
        w = np.ones(N, dtype=np.float64) / N
    else:
        w = sample_weights.astype(np.float64)
        w = w / w.sum()

    parts: List[PartitionLevel] = []
    parent_cell_of: Optional[np.ndarray] = None
    B_list: List[np.ndarray] = []
    tau_list: List[np.ndarray] = []
    n_list: List[int] = []
    parent_of_list: List[np.ndarray] = []

    # Track cumulative signatures
    if use_vectorized:
        cum_array: Optional[np.ndarray] = None
    else:
        cum: List[Tuple[int, ...]] = [tuple() for _ in range(N)]

    for k, G in enumerate(gates, start=1):
        # Build signatures - either vectorized or original
        if use_vectorized:
            # Build cumulative signatures using vectorized operations
            cum_array = _build_cumulative_signatures_vectorized(G, cum_array)

            # Bucket by signature using vectorized approach
            cells, unique_sigs = _bucket_by_signature_vectorized(cum_array)

            # Convert unique_sigs to tuples for compatibility
            signatures = [tuple(sig.tolist()) for sig in unique_sigs]
        else:
            # Original implementation
            G_np = G
            for i in range(N):
                cum[i] = cum[i] + tuple(G_np[i].tolist())
            sigs = cum
            buckets: Dict[Tuple[int, ...], List[int]] = {}
            for i, s in enumerate(sigs):
                if s not in buckets:
                    buckets[s] = []
                buckets[s].append(i)
            signatures = sorted(buckets.keys(), key=lambda t: (len(t), t))
            cells = [np.array(buckets[s], dtype=np.int32) for s in signatures]

        # Build cell_of mapping
        cell_of = np.empty(N, dtype=np.int32)
        for j, idxs in enumerate(cells):
            cell_of[idxs] = j

        # Use numerically stable mass computation to ensure conservation to machine precision
        tau = stable_mass_computation(cells, w)
        tau_list.append(tau)
        n_list.append(len(cells))

        if k == 1:
            B = np.zeros((1, len(cells)), dtype=np.int32)
            for j, idxs in enumerate(cells):
                if len(idxs) > 0:
                    B[0, j] = 1
            parent_cell_of = np.zeros(N, dtype=np.int32)
            parent_of = np.zeros(len(cells), dtype=np.int32)
        else:
            assert parent_cell_of is not None
            n_prev = len(parts[-1].cells)
            B = np.zeros((n_prev, len(cells)), dtype=np.int32)
            parent_of = np.full(len(cells), -1, dtype=np.int32)
            for j, idxs in enumerate(cells):
                if len(idxs) == 0:
                    continue
                p = parent_cell_of[idxs[0]]
                # Refinement safety: all samples in a child cell must share the same parent
                if not np.all(parent_cell_of[idxs] == p):
                    raise ValueError(
                        "Partition refinement violated: mixed parents within a child cell"
                    )
                B[p, j] = 1
                parent_of[j] = p
        B_list.append(B)
        parent_of_list.append(parent_of)

        parts.append(PartitionLevel(cell_of=cell_of, cells=cells, signatures=signatures))
        parent_cell_of = cell_of

    return AFExtraction(
        B_list=B_list,
        tau_list=tau_list,
        parts=parts,
        n_list=n_list,
        parent_of_list=parent_of_list,
    )
