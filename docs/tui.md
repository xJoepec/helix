# TUI Guide

Install
```
python3 -m pip install -e .[tui]
```

Run
```
helix tui
# or
helix-tui
```

Features
- Background computation with a progress bar
- Cancel and rerun support
- Config persistence via `platformdirs` (e.g., `~/.config/helix/config.toml`)
- JSON/CSV export (`helix_metrics_YYYYMMDD_HHMMSS.json`, per-table CSVs)
- Keyboard shortcuts: r (Run), c (Cancel), s (Save), d (Reset), e (Export JSON), x (Export CSV), : (Command), h (Help), q (Quit)

What you see
- Region counts, mass L1 errors, CP checks (unital/coisometry/PSD), Ulam spectral gap
