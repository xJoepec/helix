# Temporal Helix + Keyless Unsloth Implementation Plan

**Goal:** Ship a tested, read-only `helix grok-watch` MVP for keyless Unsloth telemetry, checkpoint
discovery, Qwen diagnostics, and temporal phase-distance scoring.

**Architecture:** Independent modules isolate HTTP, filesystem, tensor, and statistical responsibilities.
The existing CLI performs only orchestration and rendering.

**Tech Stack:** Python 3.9+, standard library HTTP/JSON, NumPy, PyTorch, argparse, pytest, Ruff, uv.

**Spec:** `docs/superpowers/specs/2026-08-31-temporal-helix-unsloth-design.md`

## Global Constraints

- Loopback-only, keyless, read-only Studio integration.
- No model load while training; no mutation endpoints.
- TDD for every production behavior.
- New files Ruff-clean; existing lint/test failures remain out of scope.

### Task 1: Typed Studio client

**Files:** create `code/helix/integrations/{__init__,unsloth}.py`; test
`code/tests/test_unsloth_client.py`.

- [ ] RED: tests for loopback validation, query encoding, fixture parsing, HTTP errors, and no auth header.
- [ ] GREEN: immutable schemas plus dependency-injected `urllib.request` JSON transport.
- [ ] Verify scoped pytest/Ruff; commit `feat: add keyless Unsloth Studio client`.

### Task 2: Checkpoint discovery

**Files:** create `code/helix/integrations/checkpoints.py`; test
`code/tests/test_checkpoint_watcher.py`.

- [ ] RED: complete index, missing/moving shards, traversal, ordering, and fingerprint tests.
- [ ] GREEN: `CheckpointRef` and `CheckpointWatcher.discover(Path)` with injected stat sampling.
- [ ] Verify scoped pytest/Ruff; commit `feat: discover complete training checkpoints`.

### Task 3: Qwen3 diagnostics

**Files:** create `code/helix/architectures/qwen3_adapter.py`; modify architecture exports; test
`code/tests/test_qwen3_adapter.py`.

- [ ] RED: exact `(alpha/rank) * B @ A`, effective rank, spectral entropy, residual covariance,
  anisotropy, and SwiGLU quantile tests; add CUDA parity guard.
- [ ] GREEN: vectorized no-grad diagnostic functions returning compact immutable results.
- [ ] Verify CPU tests, Studio-CUDA parity, and Ruff; commit `feat: add Qwen3 structural diagnostics`.

### Task 4: Temporal phase detector

**Files:** create `code/helix/temporal.py`; test `code/tests/test_temporal.py`.

- [ ] RED: schema mismatch, feature intersection, insufficient baseline, singular covariance, normal and
  coordinated-drift tests.
- [ ] GREEN: baseline standardization, diagonal shrinkage, ridge, pseudoinverse, and deterministic score.
- [ ] Verify scoped pytest/Ruff; commit `feat: add temporal phase-distance detector`.

### Task 5: `grok-watch` CLI

**Files:** create `code/helix/grok_watch.py`; modify `code/helix/cli.py`, `README.md`; create
`docs/grok-watch.md`, `code/tests/test_grok_watch.py`.

- [ ] RED: fixture-server tests for JSON output, latest/explicit run, safety state, malformed/unreachable API.
- [ ] GREEN: `run_grok_watch(argv)` and early `grok-watch` dispatch in `helix.cli.main`.
- [ ] Document keyless setup, loopback restriction, and no concurrent inference.
- [ ] Verify all new tests, targeted Ruff, CLI help, and live read-only Studio smoke; commit
  `feat: add Temporal Helix grok-watch`.

## Final verification

- [ ] Run all five new test files together.
- [ ] Run targeted Ruff on every changed Python file.
- [ ] Run `helix grok-watch --help` and live `--once --json`.
- [ ] Request task reviews and final whole-branch review.
