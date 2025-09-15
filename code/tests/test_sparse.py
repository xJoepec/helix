from __future__ import annotations

import unittest

import numpy as np

import os, sys

# Ensure the 'code' directory (package root for 'helix') is on sys.path
PKG_ROOT = os.path.dirname(os.path.dirname(__file__))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from helix.sparse import parents_from_B, B_from_parents
from helix.partitions import extract_partitions
from helix.ulam import ulam_pf


class TestSparseHelpers(unittest.TestCase):
    def test_roundtrip_parents_B(self):
        parents = np.array([0, 0, 2, -1, 1], dtype=np.int64)
        n_prev = 3
        B = B_from_parents(parents, n_prev)
        self.assertEqual(B.shape, (3, 5))
        p2 = parents_from_B(B)
        self.assertTrue(np.array_equal(p2, parents))

    def test_ulam_row_stochastic(self):
        # Identity map on a 2D box
        F = lambda x: x
        lo = np.array([-1.0, -1.0])
        hi = np.array([1.0, 1.0])
        P, _ = ulam_pf(F, (lo, hi), bins_per_dim=5, samples_per_cell=3)
        rowsums = P.sum(axis=1)
        self.assertTrue(np.allclose(rowsums, 1.0))


class TestExtractPartitionsSparse(unittest.TestCase):
    def test_parent_of_matches_B(self):
        try:
            import torch
            import torch.nn as nn
        except Exception:
            self.skipTest("torch not available")
            return

        # Build a tiny deterministic network: Linear -> ReLU -> Linear
        model = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1))
        with torch.no_grad():
            # First layer as identity to create predictable gates
            lin: nn.Linear = model[0]
            lin.weight[:] = torch.eye(2)
            lin.bias.zero_()

        X = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], dtype=np.float32)
        af = extract_partitions(model, X)

        # Only one ReLU, so exactly one B and one parent_of
        self.assertEqual(len(af.B_list), 1)
        self.assertEqual(len(af.parent_of_list), 1)
        B = af.B_list[0]
        parents = af.parent_of_list[0]
        # For k=1, the unique parent is row 0 for all columns
        self.assertTrue(np.all(parents == 0))
        # Check consistency with dense B
        parents2 = parents_from_B(B)
        self.assertTrue(np.array_equal(parents2, parents))


if __name__ == "__main__":
    unittest.main()
