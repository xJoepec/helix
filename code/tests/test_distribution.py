from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_windows_launcher_runs_cli_help_outside_repository(tmp_path: Path) -> None:
    result = subprocess.run(
        ["cmd.exe", "/d", "/c", str(ROOT / "helix.cmd"), "--help"],
        cwd=tmp_path,
        capture_output=True,
        check=False,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
