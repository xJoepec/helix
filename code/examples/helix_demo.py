from __future__ import annotations

import math
from typing import Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as _e:  # pragma: no cover
    torch = None
    nn = None
    F = None

from helix import (
    extract_partitions,
    build_V_from_incidence,
    sanity_check_ucp,
    ulam_pf,
    spectral_gap,
    region_counts,
    mass_consistency_errors,
    cumulative_anisotropy,
)


class MLP(nn.Module):
    def __init__(self, d_in=2, widths=(16, 16), d_out=2):
        super().__init__()
        layers = []
        last = d_in
        for w in widths:
            layers += [nn.Linear(last, w), nn.ReLU(inplace=False)]
            last = w
        layers += [nn.Linear(last, d_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def make_moons(n=2000, noise=0.08, seed=0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, math.pi, n // 2)
    x1 = np.c_[np.cos(theta), np.sin(theta)]
    x2 = np.c_[1 - np.cos(theta), 1 - np.sin(theta)] + np.array([0.1, -0.2])
    X = np.vstack([x1, x2]).astype(np.float32)
    X += noise * rng.standard_normal(X.shape).astype(np.float32)
    y = np.r_[np.zeros(n // 2, dtype=np.int64), np.ones(n // 2, dtype=np.int64)]
    return X, y


def main():
    if torch is None:
        raise RuntimeError("PyTorch required for the demo.")

    X, y = make_moons(n=4000, noise=0.07, seed=1)
    model = MLP(d_in=2, widths=(16, 16), d_out=2)

    # quick warmup to get nontrivial gates
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)
    for _ in range(100):
        opt.zero_grad()
        logits = model(X_t)
        loss = F.cross_entropy(logits, y_t)
        loss.backward()
        opt.step()

    # AF extraction
    af = extract_partitions(model, X)
    print("[AF] region counts:", region_counts(af.B_list))
    print("[AF] mass consistency L1 errors:", mass_consistency_errors(af.B_list, af.tau_list))

    # CP sanity
    for k, B in enumerate(af.B_list, start=1):
        tau_prev = np.array([1.0]) if k == 1 else af.tau_list[k - 2]
        tau_cur = af.tau_list[k - 1]
        V = build_V_from_incidence(B, tau_prev, tau_cur)
        stats = sanity_check_ucp(V, trials=6)
        print(
            f"[CP] depth {k}: unital={stats['unital_err_fro']:.2e}, coiso={stats['coisometry_err_fro']:.2e}, psd_vio={stats['psd_min_eig_violation']:.2e}"
        )

    # Ulam PF on a tiny residual block built from scratch
    h2 = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 2))
    with torch.no_grad():
        for p in h2.parameters():
            p.mul_(0.1)

    def F_block(x_np: np.ndarray, eps: float = 0.3) -> np.ndarray:
        x_t = torch.from_numpy(x_np.astype(np.float32))
        y = x_t + eps * h2(x_t)
        return y.detach().numpy().astype(np.float64)

    lo = np.array([-2.0, -1.5])
    hi = np.array([3.0, 2.5])
    P, _ = ulam_pf(lambda z: F_block(z, eps=0.4), (lo, hi), bins_per_dim=25)
    print("[ULAM] spectral gap:", f"{spectral_gap(P):.4f}")

    # Anisotropy proxy
    col_mass = cumulative_anisotropy(af.B_list)
    if col_mass.size:
        print(
            f"[ANISO] cumulative column sums: min={col_mass.min()}, median={np.median(col_mass)}, max={col_mass.max()}"
        )


if __name__ == "__main__":
    main()


