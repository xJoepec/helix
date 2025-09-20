# CLAUDE.md

Guidance for Claude Code when working in this repository.

Project Snapshot
- Helix provides operator‑algebraic diagnostics for deep networks: AF partitions (ReLU), CP maps, and Ulam–Perron–Frobenius flow analysis. Numerically stable, pedagogical, and practical.

Where To Start
- Read: `README.md`, then browse `docs/overview.md`, `docs/getting-started.md`, and `docs/concepts.md`.
- API surface: `docs/api.md`. CLI/TUI: `docs/cli.md`, `docs/tui.md`. Troubleshooting in `docs/troubleshooting.md`.

Repo Layout (authoritative)
- Library: `code/helix/` (partitions, cp, ulam, diagnostics, ktheory, sparse, plotting, cli, tui, env_api)
- Examples: `code/examples/helix_demo.py`
- Tests: `code/tests/`
- Environments: `environments/{helixenv,bixbench,hle}` (helixenv has Verifiers integration)
- Docs: `docs/`
- ASCII animations: `ascii-animations/` (CLI art frames)
- Local CLI runner: `./helix`

Key Mathematical Components (quick map)
- AF/Bratteli: ReLU refinement via incidence matrices `B_k`; masses `τ_k`. See `code/helix/partitions.py`.
- CP maps: build `V` from `(B_k, τ)` and check unitality/PSD/coisometry. See `code/helix/cp.py`.
- Ulam PF: discretize flows with barycentric splitting; inspect spectral gaps. See `code/helix/ulam.py`.
- Diagnostics: region counts, mass L1, anisotropy proxy. See `code/helix/diagnostics.py`.
- Sparse helpers: parent pointers ↔ dense incidence. See `code/helix/sparse.py`.
- K‑theory (toy): Smith normal form utilities. See `code/helix/ktheory.py`.
- Environment API: stable helpers for external environments; `AFMetrics`, `AFLevelMetrics`. See `code/helix/env_api.py`.

Coding Guidelines (important)
- Be surgical: avoid renaming/moving modules unless asked; preserve public APIs.
- Keep numerics stable: handle zero masses; avoid accidental divide‑by‑zero; prefer vectorized NumPy.
- Keep core lean: treat `torch`, `sympy`, `matplotlib`, `textual` as optional extras.
- Environment stability: `env_api.py` is an external contract for environments; maintain backwards compatibility.
- Style/tests: run `ruff check .` and `pytest -q` before proposing large changes.
- Performance: prefer parent pointers to dense `B` where possible; avoid quadratic memory.
- Docs: update `docs/` and in‑package docstrings when adding features; keep examples runnable.

Common Tasks Playbook
- Add a diagnostic: place helpers in `code/helix/diagnostics.py`, plot in `code/helix/plotting.py`, wire into CLI `code/helix/cli.py` and TUI if user‑facing.
- Extend partitions: modify `code/helix/partitions.py` and ensure sparse helpers still round‑trip; add tests in `code/tests/`.
- Create/modify environments: work in `environments/helixenv/`, use `env_api.py` helpers for stable extraction; integrate with Verifiers.
- Add ASCII animations: place frame files in `ascii-animations/`, integrate via CLI module for interactive banner/menu.
- New environment item(s): update under `environments/*` and document in `docs/environments.md`.

Run & Verify
- CLI: `./helix --no-train --ulam-bins 25 --ulam-samples-per-cell 4 --plot --no-show --save-prefix helix_out`
- Interactive menu: `./helix` (launches ASCII-animated menu)
- Environment: `helix helixenv --samples 1024 --noise 0.05 --width 24 --epochs 80`
- TUI: `pip install -e .[tui] && helix tui`
- Demo: `python code/examples/helix_demo.py`
- Tests/Lint: `pytest -q` and `ruff check .`

Environment Integration
- **helixenv**: Verifiers/Prime-compatible environment for AF partition diagnostics; `helix/af_partition:v0`.
- **AFPartitionEnv**: MCQ prompts with deterministic scoring; LLM judge optional via API key.
- **Integration**: Use `environments.helixenv.load_environment()` or CLI `helix helixenv`.
- **Arguments**: samples, noise, width, epochs, mass_weight, wasted_weight, enable_llm_judge, etc.
- **Output**: Multi-level AF metrics rendered as classification tasks (stable/capacity-wasted/collapsed).

Notes
- This codebase is already implemented; focus on incremental improvements and pedagogy.
- Do not delete important top‑level files (e.g., `uses.md`) or restructure code without explicit instruction.
