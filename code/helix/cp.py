from __future__ import annotations

from typing import Dict, Optional

import numpy as np


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

    # Identify a parent for each column; prefer unique 1, fall back to argmax
    parents = B.argmax(axis=0)

    # Compute scaling safely; zero masses lead to zero columns/rows automatically
    for j in range(n_cur):
        p = int(parents[j])
        if B[p, j] == 0:
            continue  # no valid parent in this column
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
    # Coisometry test for mass-weighted variant would check V V^* = I_prev; here plain V
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
