# Contributing

Setup
- Python 3.10+
- Create a venv and install editable: `pip install -e .[dev]` (ruff/pytest)
- Optional extras: `torch`, `sympy`, `matplotlib`

Tests & lint
```
pytest -q
ruff check .
```

Project layout
- Library under `code/helix`; CLI/TUI integrated in the package.
- Examples under `code/examples`.
- Lightweight environments under `environments/`.
- Docs under `docs/`.

Coding guidelines
- Keep changes minimal and focused; preserve public APIs.
- Prefer numerically stable constructions; handle zero-mass cases.
- Avoid heavy deps in core unless optional.

Releasing
- Ensure README and docs are up to date.
- Tag versions and publish via standard Python packaging workflows.

