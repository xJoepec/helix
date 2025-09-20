from __future__ import annotations

import numpy as np
import pytest

from helix.topology import compute_persistent_homology, PersistentHomologySummary


def _make_circle(n: int = 64) -> np.ndarray:
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = np.cos(theta)
    y = np.sin(theta)
    return np.stack([x, y], axis=1).astype(np.float64)


def test_persistent_homology_summary_shape() -> None:
    points = _make_circle(128)
    summary = compute_persistent_homology(points, maxdim=1)
    assert isinstance(summary, PersistentHomologySummary)
    assert len(summary.betti_numbers) >= 1
    assert summary.betti_numbers[0] >= 1
    assert summary.backend


def test_persistent_homology_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from helix import topology

    points = _make_circle(48)
    monkeypatch.setattr(topology, "ripser", None, raising=False)
    monkeypatch.setattr(topology, "pdist", None, raising=False)
    monkeypatch.setattr(topology, "squareform", None, raising=False)
    summary = topology.compute_persistent_homology(points)
    assert not summary.computed
    assert summary.betti_numbers == (1,)
