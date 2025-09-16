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


def _seed_torch(seed: int) -> None:
    try:
        torch.manual_seed(seed)
        if hasattr(torch, "cuda") and torch.cuda.is_available():  # pragma: no cover
            torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(True)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:  # pragma: no cover
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False     # type: ignore[attr-defined]
        except Exception:
            pass
    except Exception:
        pass


def _load_array(path: str) -> np.ndarray:
    p = str(path)
    lower = p.lower()
    if lower.endswith(".npy"):
        return np.load(p)
    if lower.endswith(".npz"):
        npz = np.load(p)
        # Prefer common keys
        for k in ("X", "x", "data", "array"):
            if k in npz:
                return np.array(npz[k])
        # Fallback to first array
        for k in npz.files:
            return np.array(npz[k])
        raise ValueError(f"No arrays found in npz: {p}")
    if lower.endswith(".csv") or lower.endswith(".txt"):
        return np.loadtxt(p, delimiter=",")
    raise ValueError(f"Unsupported array file type: {p}")


def _dynamic_import_builder(module_path: str, func_name: str = "build_model"):
    import importlib.util
    import types

    spec = importlib.util.spec_from_file_location("helix_user_model", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)
    fn = getattr(mod, func_name, None)
    if not callable(fn):
        raise AttributeError(f"Function {func_name} not found in {module_path}")
    return fn


def run_demo(args: argparse.Namespace) -> int:
    if torch is None:
        raise RuntimeError("PyTorch required for the demo. pip install torch")

    # Deterministic seeding for reproducibility
    _seed_torch(args.seed)

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


def run_analyze(args: argparse.Namespace) -> int:
    if torch is None:
        raise RuntimeError("PyTorch required. pip install torch")
    _seed_torch(args.seed)

    # Load data or synthesize
    if args.data_x:
        X = _load_array(args.data_x)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X = X.astype(np.float32)
    else:
        X, _ = make_moons(n=args.samples, noise=args.noise, seed=args.seed)

    y = None
    if args.data_y:
        y_arr = _load_array(args.data_y)
        y = y_arr.astype(np.int64).reshape(-1)
        if y.shape[0] != X.shape[0]:
            raise ValueError("data_y length must match number of rows in data_x")

    # Build or import model
    if args.model_module:
        builder = _dynamic_import_builder(args.model_module, args.model_func)
        kwargs = {}
        if args.model_kwargs:
            import json as _json

            try:
                kwargs = _json.loads(args.model_kwargs)
            except Exception as e:
                raise ValueError(f"Invalid JSON for --model-kwargs: {e}")
        model = builder(**kwargs)
        if not isinstance(model, nn.Module):
            raise TypeError("Builder must return a torch.nn.Module")
    else:
        d_in = int(X.shape[1]) if X.ndim == 2 else 2
        widths = (args.width, args.width)
        model = MLP(d_in=d_in, widths=widths, d_out=args.d_out)

    # Optional weights
    if args.weights:
        state = torch.load(args.weights, map_location="cpu")
        try:
            model.load_state_dict(state)
        except Exception:
            # allow non-strict if user state dict has extras
            model.load_state_dict(state, strict=False)

    # Optional training if labels provided and not disabled
    if (not args.no_train) and (y is not None):
        X_t = torch.from_numpy(X)
        y_t = torch.from_numpy(y)
        try:
            opt = torch.optim.Adam(model.parameters(), lr=1e-2)
            for _ in range(args.epochs):
                opt.zero_grad()
                logits = model(X_t)
                loss = F.cross_entropy(logits, y_t)
                loss.backward()
                opt.step()
        except Exception as e:
            print(f"[warn] Training skipped due to error: {e}")

    # Metrics
    af = extract_partitions(model, X)
    n_list = region_counts(af.B_list)
    errs = mass_consistency_errors(af.B_list, af.tau_list)

    cp_stats = []
    for k, B in enumerate(af.B_list, start=1):
        tau_prev = np.array([1.0]) if k == 1 else af.tau_list[k - 2]
        tau_cur = af.tau_list[k - 1]
        V = build_V_from_incidence(B, tau_prev, tau_cur)
        cp_stats.append(sanity_check_ucp(V, trials=6))

    # Optional Ulam
    gap = None
    if not args.no_ulam:
        d = int(X.shape[1]) if X.ndim == 2 else 2
        # Default box; for d>2, broadcast typical 2D extents
        lo = np.full((d,), -2.0)
        hi = np.full((d,), 2.5)
        if d >= 2:
            lo[:2] = np.array([-2.0, -1.5])
            hi[:2] = np.array([3.0, 2.5])
        # Use a simple residual MLP block unless user disables
        h2 = nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Linear(16, d))
        with torch.no_grad():
            for p in h2.parameters():
                p.mul_(0.1)

        def F_block(x_np: np.ndarray, eps: float = 0.3) -> np.ndarray:
            x_t = torch.from_numpy(x_np.astype(np.float32))
            y_ = x_t + eps * h2(x_t)
            return y_.detach().numpy().astype(np.float64)

        P, _ = ulam_pf(
            lambda z: F_block(z, eps=0.4),
            (lo, hi),
            bins_per_dim=args.ulam_bins,
            samples_per_cell=args.ulam_samples_per_cell,
        )
        gap = spectral_gap(P)

    # Output
    print("[AF] region counts:", n_list)
    print("[AF] mass consistency L1 errors:", errs)
    for k, s in enumerate(cp_stats, start=1):
        print(
            f"[CP] depth {k}: "
            f"unital={s['unital_err_fro']:.2e}, "
            f"coiso={s['coisometry_err_fro']:.2e}, "
            f"psd_vio={s['psd_min_eig_violation']:.2e}"
        )
    if gap is not None:
        print("[ULAM] spectral gap:", f"{gap:.4f}")
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
    # New: custom analysis subcommand
    if argv and len(argv) > 0 and argv[0] == "analyze":
        pz = argparse.ArgumentParser(description="Helix CLI: analyze a custom model/dataset")
        pz.add_argument("--model-module", type=str, default="", help="Path to Python file that defines a model builder")
        pz.add_argument("--model-func", type=str, default="build_model", help="Builder function name in the module")
        pz.add_argument("--model-kwargs", type=str, default="", help="JSON dict of kwargs to pass to the builder")
        pz.add_argument("--weights", type=str, default="", help="Optional path to a state_dict .pt/.pth file")
        pz.add_argument("--data-x", type=str, default="", help="Path to features array (.npy/.npz/.csv)")
        pz.add_argument("--data-y", type=str, default="", help="Optional path to labels array for training")
        pz.add_argument("--d-out", type=int, default=2, help="Output width if using built-in MLP")
        pz.add_argument("--width", type=int, default=16, help="Hidden width for built-in MLP if used")
        pz.add_argument("--epochs", type=int, default=50)
        pz.add_argument("--no-train", action="store_true")
        pz.add_argument("--ulam-bins", type=int, default=25)
        pz.add_argument("--ulam-samples-per-cell", type=int, default=1)
        pz.add_argument("--no-ulam", action="store_true")
        pz.add_argument("--seed", type=int, default=1)
        az = pz.parse_args(argv[1:])
        return run_analyze(az)

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
