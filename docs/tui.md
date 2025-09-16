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
- Cancel support
- Config persistence via `platformdirs` (e.g., `~/.config/helix/config.toml`)
- JSON export `helix_metrics_YYYYMMDD_HHMMSS.json`
- Keyboard shortcuts: r (Run), s (Save), d (Reset), e (Export), q (Quit)

What you see
- Region counts, mass L1 errors, CP checks (unital/coisometry/PSD), Ulam spectral info

