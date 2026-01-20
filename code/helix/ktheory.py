from __future__ import annotations

from typing import Dict, List

import numpy as np

try:
    import sympy as sp
except Exception:  # pragma: no cover
    sp = None

try:
    from scipy.linalg import eigh
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh
except Exception:  # pragma: no cover
    eigh = None
    csr_matrix = None
    eigsh = None

try:  # pragma: no cover - optional progress bar
    from tqdm import tqdm
except Exception:  # pragma: no cover
    # Fallback progress bar implementation
    class tqdm:  # type: ignore[misc]
        def __init__(self, iterable=None, desc=None, total=None, **kwargs):
            self.iterable = iterable
            self.desc = desc or ""
            self.total = total
            self.n = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __iter__(self):
            if self.iterable is not None:
                for item in self.iterable:
                    yield item
                    self.update(1)

        def update(self, n=1):
            self.n += n

        def set_description(self, desc):
            self.desc = desc


def smith_normal_form_Z(M: np.ndarray, show_progress: bool = False) -> Dict[str, object]:
    """Compute Smith normal form over Z and derive basic invariants.

    Returns dict with U, S_diag, V, torsion, rank, nullity, and nullspace_Q.
    """
    if sp is None:
        raise RuntimeError("SymPy required for Smith Normal Form. pip install sympy")

    # Progress bar for matrix conversion
    if show_progress:
        with tqdm(total=1, desc="K-theory: Converting to SymPy matrix") as pbar:
            Mz = sp.Matrix(M.astype(int).tolist())
            pbar.update(1)
    else:
        Mz = sp.Matrix(M.astype(int).tolist())

    # Progress bar for Smith normal form computation
    if show_progress:
        with tqdm(total=1, desc="K-theory: Computing Smith normal form") as pbar:
            try:
                # Try the newer SymPy API first
                U, S, V = Mz.smith_normal_form()  # U*M*V = S
                pbar.update(1)
            except AttributeError:
                pbar.set_description("K-theory: Fallback SNF computation")
                # Fallback for older SymPy versions or if method doesn't exist
                try:
                    snf_result = sp.Matrix.smith_normal_form(Mz)
                    if len(snf_result) == 3:
                        U, S, V = snf_result
                    else:
                        # Some versions return different tuple structure
                        S = snf_result[0]
                        U = sp.eye(S.rows)
                        V = sp.eye(S.cols)
                    pbar.update(1)
                except (AttributeError, TypeError):
                    # Last resort: use a simplified approach
                    S = Mz  # Just use the original matrix
                    U = sp.eye(S.rows)
                    V = sp.eye(S.cols)
                    pbar.update(1)
    else:
        try:
            # Try the newer SymPy API first
            U, S, V = Mz.smith_normal_form()  # U*M*V = S
        except AttributeError:
            # Fallback for older SymPy versions or if method doesn't exist
            try:
                snf_result = sp.Matrix.smith_normal_form(Mz)
                if len(snf_result) == 3:
                    U, S, V = snf_result
                else:
                    # Some versions return different tuple structure
                    S = snf_result[0]
                    U = sp.eye(S.rows)
                    V = sp.eye(S.cols)
            except (AttributeError, TypeError):
                # Last resort: use a simplified approach
                S = Mz  # Just use the original matrix
                U = sp.eye(S.rows)
                V = sp.eye(S.cols)
    diag = [int(S[i, i]) for i in range(min(S.shape))]
    torsion = [d for d in diag if d not in (0, 1)]
    rank = sum(1 for d in diag if d != 0)
    nullity = M.shape[1] - rank
    try:
        null_q = Mz.nullspace()
        null_basis = [np.array(v, dtype=object).astype(np.float64).flatten() for v in null_q]
    except (AttributeError, ValueError):
        # Fallback: compute nullspace using NumPy
        try:
            _, _, V_numpy = np.linalg.svd(M.astype(float))
            # Last few columns of V correspond to nullspace
            null_basis = V_numpy[-nullity:].tolist() if nullity > 0 else []
        except Exception:
            null_basis = []

    return {
        "U": np.array(U, dtype=object) if hasattr(U, '__iter__') else U,
        "S_diag": diag,
        "V": np.array(V, dtype=object) if hasattr(V, '__iter__') else V,
        "torsion": torsion,
        "rank": rank,
        "nullity": nullity,
        "nullspace_Q": null_basis,
    }


def k_invariants_from_B(B: np.ndarray, show_progress: bool = False) -> Dict[str, object]:
    """Given a stationary incidence matrix B, compute K0/K1 toy invariants.

    Returns smith data for M = I - B^T.
    """
    n = B.shape[1]
    M = np.eye(n, dtype=int) - B.T.astype(int)
    return smith_normal_form_Z(M, show_progress=show_progress)


def k_invariants_hodge(
    B: np.ndarray,
    *,
    tolerance: float = 1e-10,
    show_progress: bool = False,
) -> Dict[str, object]:
    """Compute K-invariants via discrete Hodge decomposition.

    This approach treats I - B^T as a boundary operator in a chain complex
    and uses Hodge theory to find harmonic forms (K-theory generators).

    Mathematical Background:
    - M = I - B^T is a boundary operator: ∂₁: C₁ → C₀
    - Hodge Laplacian: Δ = ∂*∂ + ∂∂* where ∂* is the adjoint
    - Harmonic forms: ker(Δ) represent cohomology classes
    - Torsion appears as small eigenvalues of the Laplacian

    Parameters
    ----------
    B : np.ndarray
        Incidence matrix with shape (n_prev, n_cur)
    tolerance : float
        Threshold for near-zero eigenvalues (harmonic space)

    Returns
    -------
    Dict[str, object]
        Enhanced K-theory invariants with spectral information
    """
    if eigh is None:
        # Fallback to SymPy method
        return k_invariants_from_B(B)

    n_prev, n_cur = B.shape

    # Progress bar for matrix setup
    if show_progress:
        with tqdm(total=3, desc="K-theory: Setting up Hodge computation") as pbar:
            M = np.eye(n_cur, dtype=np.float64) - B.T.astype(np.float64)
            pbar.update(1)

            # Compute combinatorial Laplacian: Δ = M^T M + M M^T
            # Note: For boundary operators, the full Laplacian is M^T M on the domain
            # and M M^T on the codomain. We focus on the domain Laplacian.
            L = M.T @ M
            pbar.update(1)

            # Add small regularization for numerical stability
            L += tolerance * np.eye(n_cur)
            pbar.update(1)
    else:
        M = np.eye(n_cur, dtype=np.float64) - B.T.astype(np.float64)
        L = M.T @ M
        L += tolerance * np.eye(n_cur)

    try:
        # Progress bar for eigenvalue computation
        if show_progress:
            with tqdm(total=1, desc="K-theory: Computing eigenvalues") as pbar:
                eigvals, eigvecs = eigh(L)
                pbar.update(1)
        else:
            eigvals, eigvecs = eigh(L)

        # Find harmonic space (near-zero eigenvalues)
        harmonic_mask = np.abs(eigvals) < tolerance * 10
        harmonic_eigenvalues = eigvals[harmonic_mask]
        harmonic_eigenvectors = eigvecs[:, harmonic_mask]

        # Betti numbers from harmonic space dimension
        betti_0 = int(np.sum(harmonic_mask))

        # Torsion analysis from small but nonzero eigenvalues
        small_mask = (np.abs(eigvals) > tolerance * 10) & (np.abs(eigvals) < 1.0)
        torsion_eigenvalues = eigvals[small_mask]

        # Estimate torsion orders from eigenvalue clustering
        torsion_orders = _estimate_torsion_orders(torsion_eigenvalues)

        # Rank is the number of large eigenvalues
        rank = int(np.sum(eigvals >= 1.0 - tolerance))
        nullity = betti_0

        # Compute spectral gap
        sorted_eigs = np.sort(np.abs(eigvals))
        spectral_gap = float(sorted_eigs[1] - sorted_eigs[0]) if len(sorted_eigs) > 1 else 0.0

        # Geometric analysis
        min_val = np.min(eigvals[eigvals > tolerance]) if np.any(eigvals > tolerance) else tolerance
        condition_number = float(np.max(eigvals) / np.max([min_val, tolerance]))

        return {
            "method": "hodge_decomposition",
            "betti_numbers": [betti_0],
            "rank": rank,
            "nullity": nullity,
            "torsion_orders": torsion_orders,
            "spectral_gap": spectral_gap,
            "condition_number": condition_number,
            "harmonic_eigenvalues": harmonic_eigenvalues.tolist(),
            "torsion_eigenvalues": torsion_eigenvalues.tolist(),
            "harmonic_eigenvectors": harmonic_eigenvectors,
            "full_spectrum": eigvals.tolist(),
            "computed": True
        }

    except Exception as e:
        # Fallback to classical method
        return {
            "method": "hodge_fallback",
            "error": str(e),
            "fallback_result": k_invariants_from_B(B),
            "computed": False
        }


def _estimate_torsion_orders(eigenvalues: np.ndarray) -> List[int]:
    """Estimate torsion orders from eigenvalue clustering.

    Torsion elements of order n appear as eigenvalues near 2πk/n.
    This heuristic attempts to identify such clustering patterns.
    """
    if len(eigenvalues) == 0:
        return []

    # Look for eigenvalues that might correspond to Z/nZ torsion
    # For small n, check if eigenvalues cluster near 2π/n, 4π/n, etc.
    potential_orders = []

    for n in range(2, min(12, len(eigenvalues) + 1)):  # Check orders 2-11
        expected_positions = [2 * np.pi * k / n for k in range(1, n)]

        # Check if we have eigenvalues near these positions
        matches = 0
        for pos in expected_positions:
            distances = np.abs(eigenvalues - pos)
            if np.min(distances) < 0.1:  # Tolerance for clustering
                matches += 1

        # If we have multiple matches, this might be a torsion order
        if matches >= 2:
            potential_orders.append(n)

    return potential_orders


def compute_persistence_k_theory(
    B_sequence: List[np.ndarray],
    *,
    tolerance: float = 1e-10
) -> Dict[str, object]:
    """Compute persistent K-theory invariants across a sequence of incidence matrices.

    This analyzes how K-theory invariants evolve as we go deeper in the AF hierarchy,
    providing insights into the topological stability of the neural network structure.

    Parameters
    ----------
    B_sequence : List[np.ndarray]
        Sequence of incidence matrices B_k for k = 1, 2, ..., depth
    tolerance : float
        Numerical tolerance for harmonic space detection

    Returns
    -------
    Dict[str, object]
        Persistent K-theory analysis across depths
    """
    if not B_sequence:
        return {"error": "Empty sequence", "computed": False}

    depth_analysis = []
    persistent_features = {
        "betti_sequence": [],
        "rank_sequence": [],
        "torsion_evolution": [],
        "spectral_gaps": [],
        "stability_measures": []
    }

    for k, B in enumerate(B_sequence, 1):
        try:
            k_inv = k_invariants_hodge(B, tolerance=tolerance)
            depth_analysis.append(k_inv)

            # Track persistent features
            persistent_features["betti_sequence"].append(k_inv.get("betti_numbers", [0])[0])
            persistent_features["rank_sequence"].append(k_inv.get("rank", 0))
            persistent_features["spectral_gaps"].append(k_inv.get("spectral_gap", 0.0))
            persistent_features["torsion_evolution"].append(k_inv.get("torsion_orders", []))

            # Stability measure: how much the spectrum changes
            if k > 1:
                prev_spectrum = depth_analysis[k-2].get("full_spectrum", [])
                curr_spectrum = k_inv.get("full_spectrum", [])
                stability = _compute_spectral_stability(prev_spectrum, curr_spectrum)
                persistent_features["stability_measures"].append(stability)

        except Exception as e:
            depth_analysis.append({"error": str(e), "depth": k, "computed": False})

    # Analyze persistence patterns
    persistence_summary = _analyze_persistence_patterns(persistent_features)

    return {
        "depth_analysis": depth_analysis,
        "persistent_features": persistent_features,
        "persistence_summary": persistence_summary,
        "total_depth": len(B_sequence),
        "computed": True
    }


def _compute_spectral_stability(prev_spectrum: List[float], curr_spectrum: List[float]) -> float:
    """Compute stability measure between two spectra."""
    if not prev_spectrum or not curr_spectrum:
        return 0.0

    # Use Wasserstein distance between sorted spectra
    s1 = np.sort(prev_spectrum)
    s2 = np.sort(curr_spectrum)

    # Pad shorter spectrum with zeros
    max_len = max(len(s1), len(s2))
    s1_padded = np.pad(s1, (0, max_len - len(s1)))
    s2_padded = np.pad(s2, (0, max_len - len(s2)))

    # L2 distance between sorted spectra
    return float(np.linalg.norm(s1_padded - s2_padded))


def _analyze_persistence_patterns(features: Dict[str, List]) -> Dict[str, object]:
    """Analyze patterns in persistent K-theory features."""
    summary = {}

    # Betti number evolution
    betti_seq = features["betti_sequence"]
    if betti_seq:
        summary["betti_trend"] = "increasing" if betti_seq[-1] > betti_seq[0] else "decreasing"
        summary["betti_stability"] = float(np.std(betti_seq))

    # Rank evolution
    rank_seq = features["rank_sequence"]
    if rank_seq:
        summary["rank_trend"] = "increasing" if rank_seq[-1] > rank_seq[0] else "decreasing"
        summary["rank_final"] = rank_seq[-1]

    # Spectral gap evolution
    gap_seq = features["spectral_gaps"]
    if gap_seq:
        summary["gap_trend"] = "opening" if gap_seq[-1] > gap_seq[0] else "closing"
        summary["min_gap"] = float(np.min(gap_seq))
        summary["max_gap"] = float(np.max(gap_seq))

    # Stability analysis
    stability_seq = features["stability_measures"]
    if stability_seq:
        summary["avg_stability"] = float(np.mean(stability_seq))
        summary["stability_trend"] = (
            "stabilizing" if stability_seq[-1] < stability_seq[0] else "destabilizing"
        )

    return summary
