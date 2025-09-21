from __future__ import annotations

"""Topology utilities for persistent homology style diagnostics."""

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - optional topology backend
    from ripser import ripser  # type: ignore
except Exception:  # pragma: no cover
    ripser = None  # type: ignore[assignment]

try:  # pragma: no cover - optional scipy helpers
    from scipy.spatial.distance import pdist, squareform
except Exception:  # pragma: no cover
    pdist = None  # type: ignore[assignment]
    squareform = None  # type: ignore[assignment]

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


@dataclass(frozen=True)
class PersistentHomologySummary:
    """Light-weight summary of persistent homology statistics."""

    betti_numbers: Tuple[int, ...]
    average_lifetimes: Tuple[float, ...]
    max_lifetimes: Tuple[float, ...]
    finite_pairs: Tuple[int, ...]
    computed: bool
    backend: str
    notes: Tuple[str, ...] = ()


def _clean_diagram(diagram: np.ndarray) -> np.ndarray:
    if diagram.size == 0:
        return diagram.reshape(0, 2)
    mask = np.isfinite(diagram[:, 1])
    return diagram[mask]


def compute_persistent_homology(
    points: Sequence[Sequence[float]],
    *,
    maxdim: int = 2,
    sample_cap: int = 1024,
    show_progress: bool = False,
) -> PersistentHomologySummary:
    """Compute persistent homology summary for the provided point cloud.

    Parameters
    ----------
    points:
        Iterable of coordinate vectors (will be converted to ``np.ndarray``).
    maxdim:
        Maximum homology dimension to compute (default: ``2``).
    sample_cap:
        Randomly sub-sample the cloud when more than ``sample_cap`` points are
        provided to keep complexity manageable.
    show_progress:
        Whether to display progress bar during computation (default: ``False``).
    """

    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError("points must have shape (n, d) with n > 0")

    n_points = array.shape[0]

    # Progress bar for sampling phase
    if show_progress and n_points > sample_cap:
        with tqdm(total=3, desc="PH: Sampling points") as pbar:
            pbar.update(1)
            rng = np.random.default_rng(0)
            pbar.update(1)
            idx = rng.choice(n_points, size=sample_cap, replace=False)
            pbar.update(1)
            array = array[idx]
    elif n_points > sample_cap:
        rng = np.random.default_rng(0)
        idx = rng.choice(n_points, size=sample_cap, replace=False)
        array = array[idx]

    backend_notes: list[str] = []
    if ripser is not None:
        # Progress bar for main computation
        if show_progress:
            with tqdm(total=1, desc="PH: Computing persistence diagrams") as pbar:
                result = ripser(array, maxdim=maxdim)
                pbar.update(1)
        else:
            result = ripser(array, maxdim=maxdim)

        diagrams: Iterable[np.ndarray] = result.get("dgms", [])
        betti: list[int] = []
        avg_life: list[float] = []
        max_life: list[float] = []
        finite_counts: list[int] = []

        # Progress bar for diagram processing
        diagrams_list = list(diagrams)
        if show_progress:
            diagrams_iter = tqdm(diagrams_list, desc="PH: Processing diagrams")
        else:
            diagrams_iter = diagrams_list

        for diagram in diagrams_iter:
            clean = _clean_diagram(np.asarray(diagram))
            finite_counts.append(int(clean.shape[0]))
            betti.append(int(clean.shape[0]))
            if clean.size == 0:
                avg_life.append(0.0)
                max_life.append(0.0)
            else:
                lifetimes = clean[:, 1] - clean[:, 0]
                avg_life.append(float(np.mean(lifetimes)))
                max_life.append(float(np.max(lifetimes)))
        while len(betti) <= maxdim:
            betti.append(0)
            avg_life.append(0.0)
            max_life.append(0.0)
            finite_counts.append(0)
        return PersistentHomologySummary(
            betti_numbers=tuple(betti[: maxdim + 1]),
            average_lifetimes=tuple(avg_life[: maxdim + 1]),
            max_lifetimes=tuple(max_life[: maxdim + 1]),
            finite_pairs=tuple(finite_counts[: maxdim + 1]),
            computed=True,
            backend="ripser",
            notes=tuple(backend_notes),
        )

    # Fallback: provide coarse connectivity estimation without full PH.
    if pdist is None or squareform is None:
        backend_notes.append("Persistent homology requires the ripser package; falling back to heuristics.")
        return PersistentHomologySummary(
            betti_numbers=(1,),
            average_lifetimes=(0.0,),
            max_lifetimes=(0.0,),
            finite_pairs=(array.shape[0],),
            computed=False,
            backend="heuristic",
            notes=tuple(backend_notes),
        )

    # Progress bar for distance computation in fallback mode
    if show_progress:
        with tqdm(total=2, desc="PH: Computing distance matrix") as pbar:
            dist_condensed = pdist(array)
            pbar.update(1)
            distances = squareform(dist_condensed)
            pbar.update(1)
    else:
        distances = squareform(pdist(array))

    if distances.shape[0] <= 1:
        return PersistentHomologySummary(
            betti_numbers=(1,),
            average_lifetimes=(0.0,),
            max_lifetimes=(0.0,),
            finite_pairs=(array.shape[0],),
            computed=False,
            backend="heuristic",
            notes=("Single point cloud; topology trivially connected.",),
        )

    upper = distances[np.triu_indices(distances.shape[0], k=1)]
    nonzero = upper[upper > 0]
    if nonzero.size == 0:
        threshold = 0.0
    else:
        threshold = float(np.median(nonzero))
    betti0 = int(_count_components(distances, threshold))
    backend_notes.append(
        "Ripser not available; estimated Betti-0 via connectivity at median edge length."
    )
    return PersistentHomologySummary(
        betti_numbers=(betti0,),
        average_lifetimes=(0.0,),
        max_lifetimes=(0.0,),
        finite_pairs=(array.shape[0],),
        computed=False,
        backend="heuristic",
        notes=tuple(backend_notes),
    )


def _count_components(distances: np.ndarray, threshold: float) -> int:
    n = distances.shape[0]
    visited = np.zeros(n, dtype=bool)
    components = 0
    for start in range(n):
        if visited[start]:
            continue
        components += 1
        stack = [start]
        visited[start] = True
        while stack:
            idx = stack.pop()
            neighbors = np.where((distances[idx] <= threshold) & (np.arange(n) != idx))[0]
            for nb in neighbors:
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(int(nb))
    return components


__all__ = ["PersistentHomologySummary", "compute_persistent_homology"]
