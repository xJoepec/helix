from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

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
) -> AFExtraction:
    """Empirically extract ReLU partitions and incidence from a PyTorch model.

    Returns B_k, tau_k, and per-depth region info.
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

    # cumulative signatures ensure refinement (each child has a unique parent)
    cum: List[Tuple[int, ...]] = [tuple() for _ in range(N)]

    for k, G in enumerate(gates, start=1):
        # G is already a numpy array (uint8)
        G_np = G
        for i in range(N):
            cum[i] = cum[i] + tuple(G_np[i].tolist())
        sigs = cum  # cumulative gate signatures up to depth k
        buckets: Dict[Tuple[int, ...], List[int]] = {}
        for i, s in enumerate(sigs):
            if s not in buckets:
                buckets[s] = []
            buckets[s].append(i)
        signatures = sorted(buckets.keys(), key=lambda t: (len(t), t))
        cells = [np.array(buckets[s], dtype=np.int32) for s in signatures]
        cell_of = np.empty(N, dtype=np.int32)
        for j, idxs in enumerate(cells):
            cell_of[idxs] = j
        tau = np.array([w[idxs].sum() for idxs in cells], dtype=np.float64)
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
