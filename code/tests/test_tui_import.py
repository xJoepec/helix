from __future__ import annotations

import unittest
from unittest.mock import patch


class TestTUIImport(unittest.TestCase):
    def test_run_tui_missing_textual(self):
        from helix import tui

        with (
            patch("builtins.print") as mock_print,
            patch.object(tui, "_ensure_tui_class", side_effect=ImportError("textual")),
        ):
            rc = tui.run_tui([])

        self.assertEqual(rc, 1)
        mock_print.assert_called_once_with(
            "Textual is not installed. Install Helix TUI extras: pip install '.[tui]'"
        )


if __name__ == "__main__":
    unittest.main()
