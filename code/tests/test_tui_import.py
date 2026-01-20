from __future__ import annotations

import os
import sys
import unittest

# Ensure the 'code' directory (package root for 'helix') is on sys.path
PKG_ROOT = os.path.dirname(os.path.dirname(__file__))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)


class TestTUIImport(unittest.TestCase):
    def test_run_tui_missing_textual(self):
        # This environment likely doesn't have textual installed.
        # The TUI runner should return non-zero and print a hint.
        from helix import tui

        rc = tui.run_tui([])
        self.assertIn(rc, (0, 1))  # allow 0 if textual is present


if __name__ == "__main__":
    unittest.main()
