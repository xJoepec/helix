"""
Helix Standalone — single-file toolkit for AF partitions, CP maps, Ulam PF, and demo CLI.

This file consolidates the core functionality of the Helix project into one module:

1) Sparse helpers (parents_from_B, B_from_parents)
2) Partitions (extract_partitions, PartitionLevel, AFExtraction)
3) CP maps (build_V_from_incidence, cp_embed_apply, sanity_check_ucp)
4) Ulam PF (ulam_pf, spectral_gap)
5) K-theory (smith_normal_form_Z, k_invariants_from_B)
6) Diagnostics (region_counts, mass_consistency_errors, cumulative_anisotropy)
7) Plotting (plot_* functions; lazy matplotlib import)
8) Serialization (to_jsonable, json_dumps)
9) Demo + CLI (MLP, make_moons, run_demo, main)

Optional dependencies:
- torch (partitions + demo), sympy (K-theory), matplotlib (plots)
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# =====================
# Sparse helper module
# =====================


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
    parents = np.full(n_cur, -1, dtype=np.int64)
    if n_cur == 0:
        return parents
    argmax_rows = B.argmax(axis=0)
    has_parent = B.sum(axis=0) > 0
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


# =====================
# Partitions module
# =====================

try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore


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


# =====================
# CP map module
# =====================


def build_V_from_incidence(B: np.ndarray, tau_prev: np.ndarray, tau_cur: np.ndarray) -> np.ndarray:
    """Construct a numerically-stable V from incidence and masses.

    Intended structure: each column j has a single parent row p with B[p,j]=1.
    We set V[p,j] = sqrt(tau_cur[j] / tau_prev[p]) for valid masses, else 0.

    Shapes:
      - B: (n_prev x n_cur), entries 0/1, ideally one 1 per column
      - tau_prev: (n_prev,)
      - tau_cur: (n_cur,)
    Returns V: (n_prev x n_cur)
    """
    B = B.astype(np.int8, copy=False)
    tau_prev = tau_prev.astype(np.float64, copy=False)
    tau_cur = tau_cur.astype(np.float64, copy=False)
    n_prev, n_cur = B.shape
    V = np.zeros((n_prev, n_cur), dtype=np.float64)

    parents = B.argmax(axis=0)

    for j in range(n_cur):
        p = int(parents[j])
        if B[p, j] == 0:
            continue
        tp = tau_prev[p]
        tj = tau_cur[j]
        if tp <= 0.0 or tj <= 0.0:
            continue
        V[p, j] = np.sqrt(tj / tp)

    return V


def cp_embed_apply(V: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Heisenberg picture: Φ(X) = V^* X V.

    V: (n_prev x n_cur); X: (n_prev x n_prev) -> returns (n_cur x n_cur)
    """
    return V.T.conj() @ X @ V


def sanity_check_ucp(
    V: np.ndarray, trials: int = 6, rng: Optional[np.random.Generator] = None
) -> Dict[str, float]:
    """Sanity checks: unitality Φ(I)=I and PSD preservation for random PSD inputs.

    Returns dict with Frobenius error of unitality and maximum PSD violation.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n_prev, n_cur = V.shape
    I_prev = np.eye(n_prev)
    I_cur = np.eye(n_cur)
    Phi_I = cp_embed_apply(V, I_prev)
    unital_err = np.linalg.norm(Phi_I - I_cur, ord="fro")
    coiso_err = np.linalg.norm(V @ V.T.conj() - I_prev, ord="fro")

    max_psd_violation = 0.0
    for _ in range(trials):
        Z = rng.standard_normal((n_prev, n_prev)) + 1j * rng.standard_normal((n_prev, n_prev))
        X = Z.conj().T @ Z
        Y = cp_embed_apply(V, X)
        lam_min = np.linalg.eigvalsh((Y + Y.conj().T) / 2.0).min().real
        max_psd_violation = max(max_psd_violation, max(0.0, -lam_min))

    return {
        "unital_err_fro": float(unital_err),
        "coisometry_err_fro": float(coiso_err),
        "psd_min_eig_violation": float(max_psd_violation),
    }


# =====================
# Ulam PF module
# =====================


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

    widths = (hi - lo) / float(bins_per_dim)
    if rng is None:
        rng = np.random.default_rng(0)

    q = int(max(1, samples_per_cell))
    if q == 1:
        samples = centers
        src_rows = np.arange(M, dtype=np.int64)
    else:
        jitter = rng.uniform(-0.5, 0.5, size=(M, q, d)) * widths
        samples = (centers[:, None, :] + jitter).reshape(-1, d)
        samples = np.minimum(np.maximum(samples, lo), hi)
        src_rows = np.repeat(np.arange(M, dtype=np.int64), q)

    Y = F(samples)
    Y = np.minimum(np.maximum(Y, lo), hi)

    P = np.zeros((M, M), dtype=np.float64)
    multipliers = np.array([bins_per_dim ** (d - 1 - i) for i in range(d)], dtype=np.int64)

    def deposit_point(y: np.ndarray, row: int):
        b = np.floor((y - lo) / widths).astype(np.int64)
        b = np.clip(b, 0, bins_per_dim - 1)
        r = ((y - lo) / widths) - b
        r = np.clip(r, 0.0, 1.0)
        r = np.where(b == (bins_per_dim - 1), 0.0, r)

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
                    w *= 1.0 - r[i]
            if not valid or w == 0.0:
                continue
            tgt = int((idx * multipliers).sum())
            P[row, tgt] += w

    for y, row in zip(Y, src_rows):
        deposit_point(y, int(row))

    if q > 1:
        P /= float(q)
    row_sums = P.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    P /= row_sums

    centers_axes = [a.copy() for a in centers_axes]
    return P, centers_axes


def spectral_gap(P: np.ndarray) -> float:
    """Return 1 - |λ2(P)| (magnitude of second-largest eigenvalue of P^T)."""
    eigs = np.linalg.eigvals(P.T)
    eigs = np.sort(np.abs(eigs))[::-1]
    if eigs.size < 2:
        return 0.0
    return float(1.0 - eigs[1])


# =====================
# K-theory module (optional: sympy)
# =====================


def smith_normal_form_Z(M: np.ndarray) -> Dict[str, object]:
    """Compute Smith normal form over Z and derive basic invariants.

    Returns dict with U, S_diag, V, torsion, rank, nullity, and nullspace_Q.
    """
    try:
        import sympy as sp  # type: ignore
    except Exception:  # pragma: no cover
        raise RuntimeError("SymPy required for Smith Normal Form. pip install sympy")

    Mz = sp.Matrix(M.astype(int).tolist())
    U, S, V = Mz.smith_normal_form()  # U*M*V = S
    diag = [int(S[i, i]) for i in range(min(S.shape))]
    torsion = [d for d in diag if d not in (0, 1)]
    rank = sum(1 for d in diag if d != 0)
    nullity = M.shape[1] - rank
    null_q = Mz.nullspace()
    null_basis = [np.array(v, dtype=object).astype(np.float64).flatten() for v in null_q]

    return {
        "U": np.array(U, dtype=object),
        "S_diag": diag,
        "V": np.array(V, dtype=object),
        "torsion": torsion,
        "rank": rank,
        "nullity": nullity,
        "nullspace_Q": null_basis,
    }


def k_invariants_from_B(B: np.ndarray) -> Dict[str, object]:
    """Given a stationary incidence matrix B, compute K0/K1 toy invariants.

    Returns smith data for M = I - B^T.
    """
    n = B.shape[1]
    M = np.eye(n, dtype=int) - B.T.astype(int)
    return smith_normal_form_Z(M)


# =====================
# Diagnostics module
# =====================


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


# =====================
# Plotting module (lazy matplotlib import)
# =====================


def _import_matplotlib():
    plt = importlib.import_module("matplotlib.pyplot")
    return plt


def plot_region_counts(n_list: Sequence[int], *, title: str = "Region counts vs depth"):
    plt = _import_matplotlib()
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(n_list) + 1), n_list, marker="o")
    ax.set_xlabel("Depth k")
    ax.set_ylabel("Number of regions n_k")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig, ax


def plot_mass_consistency(
    errs: Sequence[float], *, title: str = r"Mass consistency $\|\tau_{k-1}-B_k\tau_k\|_1$"
):
    plt = _import_matplotlib()
    fig, ax = plt.subplots()
    markerline, stemlines, baseline = ax.stem(np.arange(1, len(errs) + 1), errs)
    ax.set_xlabel("Depth k")
    ax.set_ylabel("L1 error")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig, ax


def plot_cp_errors(cp_stats: List[Dict[str, float]], *, title: str = "CP diagnostics by depth"):
    plt = _import_matplotlib()
    ks = np.arange(1, len(cp_stats) + 1)
    unital = [s.get("unital_err_fro", np.nan) for s in cp_stats]
    coiso = [s.get("coisometry_err_fro", np.nan) for s in cp_stats]
    psd = [s.get("psd_min_eig_violation", np.nan) for s in cp_stats]
    fig, ax = plt.subplots()
    ax.plot(ks, unital, marker="o", label="unital ‖Φ(I)-I‖_F")
    ax.plot(ks, coiso, marker="s", label="coisometry ‖V V* - I‖_F")
    ax.plot(ks, psd, marker="^", label="PSD min eig violation")
    ax.set_yscale("log")
    ax.set_xlabel("Depth k")
    ax.set_ylabel("Error (log scale)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    return fig, ax


def plot_ulam_spectrum(
    P: np.ndarray, top: int = 10, *, title: str = "Ulam PF eigenvalue magnitudes"
):
    plt = _import_matplotlib()
    vals = np.sort(np.abs(np.linalg.eigvals(P.T)))[::-1][:top]
    fig, ax = plt.subplots()
    ax.bar(np.arange(1, len(vals) + 1), vals)
    ax.set_xlabel("Index")
    ax.set_ylabel("|λ|")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    return fig, ax


# =====================
# Serialization helpers
# =====================


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy types and arrays to JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, set):
        return [to_jsonable(v) for v in obj]
    return obj


def json_dumps(obj: Any, **kwargs: Any) -> str:
    import json

    return json.dumps(to_jsonable(obj), **kwargs)


# =====================
# Demo/CLI components
# =====================


class MLP(nn.Module):  # type: ignore[misc]
    def __init__(self, d_in: int = 2, widths: Tuple[int, int] = (16, 16), d_out: int = 2):
        super().__init__()
        layers: List[nn.Module] = []
        last = d_in
        for w in widths:
            layers += [nn.Linear(last, w), nn.ReLU(inplace=False)]
            last = w
        layers += [nn.Linear(last, d_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # type: ignore[override]
        return self.net(x)


def make_moons(n: int = 2000, noise: float = 0.08, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, math.pi, n // 2)
    x1 = np.c_[np.cos(theta), np.sin(theta)]
    x2 = np.c_[1 - np.cos(theta), 1 - np.sin(theta)] + np.array([0.1, -0.2])
    X = np.vstack([x1, x2]).astype(np.float32)
    X += noise * rng.standard_normal(X.shape).astype(np.float32)
    y = np.r_[np.zeros(n // 2, dtype=np.int64), np.ones(n // 2, dtype=np.int64)]
    return X, y


def run_demo(args: Any) -> int:
    if torch is None or nn is None:
        raise RuntimeError("PyTorch required for the demo. pip install torch")

    X, y = make_moons(n=args.samples, noise=args.noise, seed=args.seed)
    model = MLP(d_in=2, widths=(args.width, args.width), d_out=2)

    if not args.no_train:
        import torch.nn.functional as F  # type: ignore

        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        X_t = torch.from_numpy(X)
        y_t = torch.from_numpy(y)
        for _ in range(args.epochs):
            opt.zero_grad()
            logits = model(X_t)
            loss = F.cross_entropy(logits, y_t)
            loss.backward()
            opt.step()

    af = extract_partitions(model, X)
    n_list = region_counts(af.B_list)
    errs = mass_consistency_errors(af.B_list, af.tau_list)

    cp_stats = []
    for k, B in enumerate(af.B_list, start=1):
        tau_prev = np.array([1.0]) if k == 1 else af.tau_list[k - 2]
        tau_cur = af.tau_list[k - 1]
        V = build_V_from_incidence(B, tau_prev, tau_cur)
        cp_stats.append(sanity_check_ucp(V, trials=6))

    # Ulam on identity map (fast, robust). Adjust box for visualization.
    lo = np.array([-2.0, -1.5])
    hi = np.array([3.0, 2.5])
    P, _ = ulam_pf(
        lambda z: z,
        (lo, hi),
        bins_per_dim=args.ulam_bins,
        samples_per_cell=args.ulam_samples_per_cell,
    )
    gap = spectral_gap(P)

    print("[AF] region counts:", n_list)
    print("[AF] mass consistency L1 errors:", errs)
    for k, s in enumerate(cp_stats, start=1):
        print(
            f"[CP] depth {k}: unital={s['unital_err_fro']:.2e}, "
            f"coiso={s['coisometry_err_fro']:.2e}, psd_vio={s['psd_min_eig_violation']:.2e}"
        )
    print("[ULAM] spectral gap:", f"{gap:.4f}")

    if args.plot:
        fig1, _ = plot_region_counts(n_list)
        fig2, _ = plot_mass_consistency(errs)
        fig3, _ = plot_cp_errors(cp_stats)
        fig4, _ = plot_ulam_spectrum(P)
        import matplotlib.pyplot as plt  # type: ignore

        if args.save_prefix:
            fig1.savefig(f"{args.save_prefix}_regions.png", dpi=150, bbox_inches="tight")
            fig2.savefig(f"{args.save_prefix}_mass_consistency.png", dpi=150, bbox_inches="tight")
            fig3.savefig(f"{args.save_prefix}_cp.png", dpi=150, bbox_inches="tight")
            fig4.savefig(f"{args.save_prefix}_ulam.png", dpi=150, bbox_inches="tight")
        if not args.no_show:
            plt.show()

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Helix Standalone: run diagnostics and plots")
    p.add_argument("--samples", type=int, default=4000)
    p.add_argument("--noise", type=float, default=0.07)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--width", type=int, default=16)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--no-train", action="store_true")
    p.add_argument("--ulam-bins", type=int, default=25, help="Ulam bins per dimension")
    p.add_argument(
        "--ulam-samples-per-cell",
        type=int,
        default=1,
        help="Number of random samples per grid cell for Ulam (barycentric)",
    )
    p.add_argument("--plot", action="store_true")
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--save-prefix", type=str, default="")
    args = p.parse_args(argv)
    return run_demo(args)


if __name__ == "__main__":
    raise SystemExit(main())
