from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _import_matplotlib():
    import importlib

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


def plot_mass_consistency(errs: Sequence[float], *, title: str = r"Mass consistency $\|\tau_{k-1}-B_k\tau_k\|_1$"):
    plt = _import_matplotlib()
    fig, ax = plt.subplots()
    # Compatibility across Matplotlib versions
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


def plot_ulam_spectrum(P: np.ndarray, top: int = 10, *, title: str = "Ulam PF eigenvalue magnitudes"):
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


