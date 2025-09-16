from __future__ import annotations

import argparse
import math
from typing import List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as _e:  # pragma: no cover
    torch = None
    nn = None
    F = None

from . import (
    build_V_from_incidence,
    extract_partitions,
    mass_consistency_errors,
    region_counts,
    sanity_check_ucp,
    spectral_gap,
    ulam_pf,
)
from .plotting import (
    plot_cp_errors,
    plot_mass_consistency,
    plot_region_counts,
    plot_ulam_spectrum,
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


def run_demo(args: argparse.Namespace) -> int:
    if torch is None:
        raise RuntimeError("PyTorch required for the demo. pip install torch")

    X, y = make_moons(n=args.samples, noise=args.noise, seed=args.seed)
    model = MLP(d_in=2, widths=(args.width, args.width), d_out=2)

    if not args.no_train:
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

    # residual block for Ulam
    h2 = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 2))
    with torch.no_grad():
        for p in h2.parameters():
            p.mul_(0.1)

    def F_block(x_np: np.ndarray, eps: float = 0.3) -> np.ndarray:
        x_t = torch.from_numpy(x_np.astype(np.float32))
        y_ = x_t + eps * h2(x_t)
        return y_.detach().numpy().astype(np.float64)

    lo = np.array([-2.0, -1.5])
    hi = np.array([3.0, 2.5])
    P, _ = ulam_pf(
        lambda z: F_block(z, eps=0.4),
        (lo, hi),
        bins_per_dim=args.ulam_bins,
        samples_per_cell=args.ulam_samples_per_cell,
    )
    gap = spectral_gap(P)

    print("[AF] region counts:", n_list)
    print("[AF] mass consistency L1 errors:", errs)
    for k, s in enumerate(cp_stats, start=1):
        print(
            f"[CP] depth {k}: "
            f"unital={s['unital_err_fro']:.2e}, "
            f"coiso={s['coisometry_err_fro']:.2e}, "
            f"psd_vio={s['psd_min_eig_violation']:.2e}"
        )
    print("[ULAM] spectral gap:", f"{gap:.4f}")

    if args.plot:
        fig1, _ = plot_region_counts(n_list)
        fig2, _ = plot_mass_consistency(errs)
        fig3, _ = plot_cp_errors(cp_stats)
        fig4, _ = plot_ulam_spectrum(P)
        import matplotlib.pyplot as plt

        if args.save_prefix:
            fig1.savefig(f"{args.save_prefix}_regions.png", dpi=150, bbox_inches="tight")
            fig2.savefig(f"{args.save_prefix}_mass_consistency.png", dpi=150, bbox_inches="tight")
            fig3.savefig(f"{args.save_prefix}_cp.png", dpi=150, bbox_inches="tight")
            fig4.savefig(f"{args.save_prefix}_ulam.png", dpi=150, bbox_inches="tight")
        if not args.no_show:
            plt.show()

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    # Quick subcommand dispatch for `helix tui` without breaking legacy flags-only usage
    if argv and len(argv) > 0 and argv[0] == "tui":
        try:
            from .tui import run_tui
        except Exception as e:  # pragma: no cover
            print(
                "Helix TUI not available. Install TUI extras: pip install '.[tui]'.\n"
                f"Details: {e}"
            )
            return 1
        return run_tui(argv[1:])

    p = argparse.ArgumentParser(description="Helix CLI: run diagnostics and plots")
    p.add_argument("demo", nargs="?", default="demo", help="run demo (default)")
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

    # Back-compat: allow `helix tui` to pass through even if parsed as positional
    if getattr(args, "demo", None) == "tui":
        try:
            from .tui import run_tui
        except Exception as e:  # pragma: no cover
            print(
                "Helix TUI not available. Install TUI extras: pip install '.[tui]'.\n"
                f"Details: {e}"
            )
            return 1
        return run_tui([])
    return run_demo(args)


if __name__ == "__main__":
    raise SystemExit(main())
