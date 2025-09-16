from __future__ import annotations

import os
import sys
import unittest

import numpy as np

# Ensure the 'code' directory (package root for 'helix') is on sys.path
PKG_ROOT = os.path.dirname(os.path.dirname(__file__))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)


class TestCLIAndCP(unittest.TestCase):
    def test_cli_help_smoke(self):
        # Import locally to avoid import-time side effects
        from helix import cli

        with self.assertRaises(SystemExit):
            cli.main(["--help"])  # argparse help exits

    def test_cp_identity_stats(self):
        from helix.cp import build_V_from_incidence, sanity_check_ucp

        # Trivial identity incidence and equal masses -> V = I
        n = 3
        B = np.eye(n, dtype=np.int32)
        tau = np.full(n, 1.0 / n, dtype=np.float64)
        V = build_V_from_incidence(B, tau, tau)
        stats = sanity_check_ucp(V, trials=3)
        self.assertLess(stats["unital_err_fro"], 1e-12)
        self.assertLess(stats["coisometry_err_fro"], 1e-12)
        self.assertLess(stats["psd_min_eig_violation"], 1e-12)


if __name__ == "__main__":
    unittest.main()

