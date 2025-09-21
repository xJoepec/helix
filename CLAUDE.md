# CLAUDE.md

Guidance for Claude Code when working in this repository.

Project Snapshot
- Helix provides operator‑algebraic diagnostics for deep networks: AF partitions (ReLU), CP maps, and Ulam–Perron–Frobenius flow analysis. Numerically stable, pedagogical, and practical.
- **Enhanced 2025**: Tensor train Ulam (O(d×bins²×r²) complexity), ViT attention analysis, LLM physics judge, comprehensive testing, and performance benchmarking for production-ready operator algebra.

Where To Start
- Read: `README.md`, then browse `docs/overview.md`, `docs/getting-started.md`, and `docs/concepts.md`.
- API surface: `docs/api.md`. CLI/TUI: `docs/cli.md`, `docs/tui.md`. Troubleshooting in `docs/troubleshooting.md`.
- **LLM Integration**: `docs/api-key-setup.md` for OpenAI API configuration and physics-aware evaluation.

Repo Layout (authoritative)
- Library: `code/helix/` (partitions, cp, ulam, diagnostics, ktheory, sparse, plotting, cli, tui, env_api)
  - **New**: `ulam_tensor.py` (tensor train Ulam), `architectures/vit_adapter.py` (ViT analysis), `trajectory.py` (O(log n) compression)
- Examples: `code/examples/helix_demo.py`, `tensor_train_demo.py`
- Tests: `code/tests/` (comprehensive coverage including `test_ulam_tensor.py`, `test_vit_adapter.py`, `test_llm_judge.py`)
- Scripts: `code/scripts/benchmark_tensor_train.py` (performance analysis)
- Environments: `environments/{helixenv,bixbench,hle}` (helixenv has Verifiers integration)
  - **Enhanced**: `environments/helixenv/llm_judge.py` (physics-aware LLM evaluation)
- Docs: `docs/` (updated with K-theory CLI, tensor train API, LLM setup)
- ASCII animations: `ascii-animations/` (CLI art frames)
- Local CLI runner: `./helix`

Key Mathematical Components (quick map)
- AF/Bratteli: ReLU refinement via incidence matrices `B_k`; masses `τ_k`. See `code/helix/partitions.py`.
- CP maps: build `V` from `(B_k, τ)` and check unitality/PSD/coisometry. See `code/helix/cp.py`.
- Ulam PF: discretize flows with barycentric splitting; inspect spectral gaps. See `code/helix/ulam.py`.
- **Tensor Train Ulam**: high-dimensional flow analysis with O(d×bins²×r²) complexity reduction. See `code/helix/ulam_tensor.py`.
- **ViT Analysis**: attention mechanisms as gauge field interactions, partition extraction. See `code/helix/architectures/vit_adapter.py`.
- **Persistent Homology**: topology analysis with progress tracking and backend fallbacks. See `code/helix/topology.py`.
- Diagnostics: region counts, mass L1, anisotropy proxy. See `code/helix/diagnostics.py`.
- Sparse helpers: parent pointers ↔ dense incidence; memory-efficient operations. See `code/helix/sparse.py`.
- **K‑theory (enhanced)**: Smith normal form, Hodge decomposition, torsion analysis. See `code/helix/ktheory.py`.
- **Environment API**: comprehensive AF metrics with PH, capacity, spectral gaps, K-theory. See `code/helix/env_api.py`.
- **Trajectory Compression**: O(log n) memory via compressed sensing and importance sampling. See `code/helix/trajectory.py`.

Coding Guidelines (important)
- Be surgical: avoid renaming/moving modules unless asked; preserve public APIs.
- Keep numerics stable: handle zero masses; avoid accidental divide‑by‑zero; prefer vectorized NumPy.
- Keep core lean: treat `torch`, `sympy`, `matplotlib`, `textual`, `tqdm` as optional extras with fallbacks.
- Environment stability: `env_api.py` is an external contract for environments; maintain backwards compatibility.
- **Testing requirements**: comprehensive test coverage required for new components; add to `code/tests/`.
- **Progress integration**: use `show_progress` parameter for long-running computations; implement tqdm fallback.
- **API key security**: never hardcode keys; use environment variables; document in `docs/api-key-setup.md`.
- Style/tests: run `ruff check .` and `pytest -q` before proposing large changes.
- Performance: prefer parent pointers to dense `B`; use tensor trains for high-dimensional problems.
- **Memory efficiency**: implement O(log n) trajectory compression; use sparse operations where possible.
- Docs: update `docs/` and in‑package docstrings when adding features; keep examples runnable.

Common Tasks Playbook
- Add a diagnostic: place helpers in `code/helix/diagnostics.py`, plot in `code/helix/plotting.py`, wire into CLI `code/helix/cli.py` and TUI if user‑facing.
- Extend partitions: modify `code/helix/partitions.py` and ensure sparse helpers still round‑trip; add tests in `code/tests/`.
- **Tensor train implementations**: extend `code/helix/ulam_tensor.py`; benchmark with `code/scripts/benchmark_tensor_train.py`.
- **ViT adapter development**: modify `code/helix/architectures/vit_adapter.py`; test attention extraction and gauge field analysis.
- **Progress bar integration**: add `show_progress` parameter; implement tqdm fallback; test with long-running computations.
- **LLM judge enhancement**: extend `environments/helixenv/llm_judge.py`; add calibration examples; document API usage.
- **Testing new components**: create comprehensive test files in `code/tests/`; include unit, integration, and error handling tests.
- Create/modify environments: work in `environments/helixenv/`, use `env_api.py` helpers for stable extraction; integrate with Verifiers.
- Add ASCII animations: place frame files in `ascii-animations/`, integrate via CLI module for interactive banner/menu.
- **Performance optimization**: use `code/scripts/benchmark_tensor_train.py` for analysis; implement memory-efficient algorithms.
- New environment item(s): update under `environments/*` and document in `docs/environments.md`.

Run & Verify
- CLI: `./helix --no-train --ulam-bins 25 --ulam-samples-per-cell 4 --plot --no-show --save-prefix helix_out`
- **K-theory analysis**: `./helix ktheory --env helixenv --samples 1024 --method hodge --show-progress`
- Interactive menu: `./helix` (launches ASCII-animated menu)
- Environment: `helix helixenv --samples 1024 --noise 0.05 --width 24 --epochs 80`
- **LLM judge environment**: `helix helixenv --enable-llm-judge --samples 1024 --width 24`
- TUI: `pip install -e .[tui] && helix tui`
- Demo: `python code/examples/helix_demo.py`
- **Tensor train demo**: `python code/examples/tensor_train_demo.py`
- **Performance benchmarking**: `python code/scripts/benchmark_tensor_train.py --quick`
- **Comprehensive testing**: `pytest code/tests/ -v` (includes ulam_tensor, vit_adapter, llm_judge)
- Tests/Lint: `pytest -q` and `ruff check .`

Environment Integration
- **helixenv**: Verifiers/Prime-compatible environment for AF partition diagnostics; `helix/af_partition:v0`.
- **AFPartitionEnv**: MCQ prompts with deterministic scoring; LLM judge optional via API key.
- **Enhanced LLM Judge**: Physics-aware evaluation with rubric-based scoring, calibration examples, and fallback heuristics.
- **Integration**: Use `environments.helixenv.load_environment()` or CLI `helix helixenv`.
- **Arguments**: samples, noise, width, epochs, mass_weight, wasted_weight, enable_llm_judge, llm_judge_model, show_progress, etc.
- **Enhanced Output**: Multi-level AF metrics with persistent homology, capacity loss, spectral gaps, K-theory invariants.
- **API Configuration**: Use `OPENAI_API_KEY` environment variable; see `docs/api-key-setup.md` for setup.
- **Progress Tracking**: All long computations support `show_progress=True` for user feedback.

Notes
- This codebase is already implemented; focus on incremental improvements and pedagogy.
- Do not delete important top‑level files (e.g., `uses.md`) or restructure code without explicit instruction.

## 2025 Enhanced Capabilities (Production Ready)
- **Tensor Train Ulam**: Handles high-dimensional systems (d ≥ 3) with exponential memory reduction; auto-rank detection.
- **ViT Attention Analysis**: Treats attention weights as gauge field connections; extracts topological features.
- **Comprehensive Testing**: 100+ unit tests across ulam_tensor, vit_adapter, llm_judge; full error handling coverage.
- **LLM Physics Judge**: OpenAI-integrated evaluation with physics-aware rubrics; secure API key management.
- **Performance Benchmarking**: Systematic tensor train rank optimization; memory/accuracy tradeoff analysis.
- **Progress Integration**: All expensive computations support progress tracking; graceful tqdm fallbacks.
- **Memory Optimization**: O(log n) trajectory compression; sparse matrix operations; efficient incidence representations.
- **Enhanced Environment API**: Complete AF metrics extraction with PH, capacity, spectral gaps, K-theory invariants.
