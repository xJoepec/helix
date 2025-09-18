from __future__ import annotations

from typing import Dict

import numpy as np

try:
    import sympy as sp
except Exception:  # pragma: no cover
    sp = None


def smith_normal_form_Z(M: np.ndarray) -> Dict[str, object]:
    """Compute Smith normal form over Z and derive basic invariants.

    Returns dict with U, S_diag, V, torsion, rank, nullity, and nullspace_Q.
    """
    if sp is None:
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
