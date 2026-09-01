# Temporal Helix Telemetry Hardening Implementation Plan

**Goal:** Make Temporal Helix checkpoint telemetry stable, run-bound, authentication-aware, provenance-safe, computationally efficient, and reproducibly validated without claiming calibrated grokking detection.

**Architecture:** Keep Studio access read-only and loopback-restricted. Normalize one selected run before collecting status and metrics, pass only stable checkpoint manifests into later analysis, and make phase observations carry an immutable identity that the detector validates. Keep Qwen diagnostics factorized in LoRA rank space and validate the complete repository through one checked-in Python command.

**Tech Stack:** Python 3.11+, NumPy, PyTorch, standard-library HTTP/JSON, argparse, pytest, Ruff, setuptools, uv.

**Spec:** `docs/superpowers/specs/2026-09-01-temporal-telemetry-hardening-design.md`

## Global Constraints

- Studio access is GET-only; no training, inference, export, stop, or shutdown mutation may be added.
- Loopback hosts only: `127.0.0.1`, `localhost`, and `::1`.
- Keyless loopback reads are the default; bearer tokens are accepted only in memory through `--studio-token` or `UNSLOTH_STUDIO_TOKEN`.
- Python support is `>=3.11,<3.14`.
- A phase score is invalid when run, probe-set, schema, source, or checkpoint provenance is missing or internally mismatched.
- Qwen LoRA diagnostics must not materialize an `out_features × in_features` delta.
- The merge gate is `uv run python scripts/validate_pr.py`.

---

### Task 1: Add stable, complete checkpoint manifests

**Files:**
- Modify: `code/helix/integrations/checkpoints.py`
- Test: `code/tests/test_checkpoint_watcher.py`
- Test: `code/tests/test_grok_watch.py`

**Interfaces:**
- `CheckpointRef` gains `stable: bool` and `stability_reason: str` while retaining `step`, `path`, `shard_paths`, and `fingerprint`.
- `CheckpointWatcher(min_age_seconds: float = 2.0).discover(output_dir: Path, *, stable_only: bool = False) -> list[CheckpointRef]` returns complete references and can expose recent-but-pending references for reporting.
- A checkpoint is complete only when its trainer state parses, its optional `global_step` matches the directory step, and all indexed/model/adapter files are regular files within the checkpoint directory.
- The fingerprint hashes every required file’s relative name, byte size, and `st_mtime_ns`; it does not read multi-gigabyte weight contents.

- [ ] **Step 1: Write failing tests** for a trainer-step mismatch, a recently modified complete checkpoint, a stable checkpoint with old mtimes, and `build_snapshot` separating stable from pending checkpoints.
- [ ] **Step 2: Run the focused tests** with `uv run pytest code/tests/test_checkpoint_watcher.py code/tests/test_grok_watch.py -q`; confirm the new assertions fail for the current presence-only watcher.
- [ ] **Step 3: Implement manifest validation and age-based stability** without mutating files. Preserve rejection of traversal paths, external symlinks, partial indexes, and partial adapters.
- [ ] **Step 4: Run the focused tests and Ruff** with `uv run pytest code/tests/test_checkpoint_watcher.py code/tests/test_grok_watch.py -q` and `uv run ruff check code/helix/integrations/checkpoints.py code/helix/grok_watch.py code/tests/test_checkpoint_watcher.py code/tests/test_grok_watch.py`.
- [ ] **Step 5: Commit** with `git commit -m "fix: require stable complete checkpoint manifests"`.

### Task 2: Bind run, status, metrics, and checkpoint paths

**Files:**
- Modify: `code/helix/grok_watch.py`
- Modify: `code/helix/integrations/unsloth.py`
- Test: `code/tests/test_grok_watch.py`
- Test: `code/tests/test_unsloth_client.py`

**Interfaces:**
- Add a normalized association object in the snapshot with selected run ID, status job ID, metrics job ID, and a state such as `bound`, `historical`, `fallback`, or `mismatch`.
- Resolve status before implicit run selection. If `status.job_id` exists, fetch that run directly; `--run-id` always wins and uses direct lookup.
- Consume live status fields only when `status.job_id == selected_run.id`.
- Accept live metrics only when `metrics.job_id == selected_run.id`; on mismatch or missing identity, use the selected run’s persisted history and mark the fallback explicitly.
- Discover checkpoints only under the selected run’s `output_dir`; never use a global output directory or current-job path.

- [ ] **Step 1: Write failing tests** for status pointing at run B while run A is explicitly selected, implicit selection following `status.job_id`, missing/mismatched metrics job IDs, and checkpoint paths belonging only to the selected run.
- [ ] **Step 2: Run the focused tests** and verify the current implementation leaks global status or accepts unverified metrics.
- [ ] **Step 3: Implement the run resolver and association state**; make human and JSON CLI output include the association result and return nonzero for a true mismatch.
- [ ] **Step 4: Run focused tests, the complete suite, and Ruff** with `uv run pytest code/tests/test_grok_watch.py code/tests/test_unsloth_client.py -q`, `uv run pytest -q`, and targeted Ruff.
- [ ] **Step 5: Commit** with `git commit -m "fix: bind Studio telemetry to selected runs"`.

### Task 3: Support and document current Studio authentication

**Files:**
- Modify: `code/helix/integrations/unsloth.py`
- Modify: `code/helix/grok_watch.py`
- Modify: `code/tests/test_unsloth_client.py`
- Modify: `code/tests/test_grok_watch.py`
- Modify: `docs/grok-watch.md`
- Modify: `README.md`

**Interfaces:**
- `StudioClient(base_url, *, token: str | None = None, auth_mode: str = "auto", token_env: str = "UNSLOTH_STUDIO_TOKEN", transport: JsonTransport | None = None)` supports `auto`, `keyless`, and `bearer`.
- `auto` uses bearer when an explicit token or `UNSLOTH_STUDIO_TOKEN` exists and otherwise sends no authorization header; `keyless` always sends no bearer; `bearer` requires a token.
- Add CLI flags `--studio-auth {auto,keyless,bearer}` and `--studio-token`; never include token values in exceptions, URLs, JSON, or logs.
- Keep URL host validation before any request and preserve GET-only client methods.

- [ ] **Step 1: Write failing transport tests** asserting keyless headers contain no `Authorization`, bearer headers contain exactly `Bearer <token>`, environment-token lookup works, and bearer mode without a token fails without exposing credentials.
- [ ] **Step 2: Run the focused tests** and confirm the current client always behaves as keyless and the CLI has no auth controls.
- [ ] **Step 3: Implement auth-mode normalization and CLI wiring** with a keyword-only token path.
- [ ] **Step 4: Document the observed current behavior: OpenAPI advertises `HTTPBearer`, current loopback Studio accepts keyless reads, and users can opt into bearer with the documented environment variable or flag.
- [ ] **Step 5: Run focused tests and Ruff**, then commit with `git commit -m "feat: support Studio keyless and bearer auth"`.

### Task 4: Enforce phase-observation provenance

**Files:**
- Modify: `code/helix/temporal.py`
- Modify: `code/tests/test_temporal.py`
- Modify: `docs/grok-watch.md`

**Interfaces:**
- Add immutable `ObservationProvenance(source, run_id, checkpoint_id, checkpoint_fingerprint, probe_set_id, schema_version)`.
- `FeatureObservation` requires a matching `provenance: ObservationProvenance`; construction rejects inconsistent duplicated IDs/schema.
- `TemporalPhaseDetector.fit` requires at least two finite observations with one run, probe set, source, and schema identity; it stores that identity.
- `score` returns `PhaseScore(nan, "insufficient_data", feature_names)` for missing or mismatched provenance, including an empty or internally inconsistent checkpoint fingerprint.

- [ ] **Step 1: Write failing tests** for missing provenance, duplicated-field inconsistency, cross-run baseline, cross-probe baseline, source/schema mismatch, and a valid same-run cross-checkpoint score.
- [ ] **Step 2: Run `uv run pytest code/tests/test_temporal.py -q`** and confirm the current detector scores observations without identity checks.
- [ ] **Step 3: Implement provenance dataclasses, fit validation, and score rejection** while retaining finite-feature and covariance safeguards.
- [ ] **Step 4: Run the focused tests and Ruff**, then update docs to state that phase scores are not produced from mixed or unverified artifacts.
- [ ] **Step 5: Commit** with `git commit -m "fix: enforce temporal observation provenance"`.

### Task 5: Replace dense Qwen diagnostics with factorized metrics

**Files:**
- Modify: `code/helix/architectures/qwen3_adapter.py`
- Modify: `code/tests/test_qwen3_adapter.py`
- Modify: `code/helix/architectures/__init__.py` only if exports change

**Interfaces:**
- `lora_delta_diagnostics(lora_a, lora_b, *, alpha, rank)` computes update singular values from `A Aᵀ` and `Bᵀ B` rank-space Gram matrices and returns the existing `LoraDeltaDiagnostics` fields.
- `residual_diagnostics(hidden_states, *, max_components: int = 32)` computes exact trace through centered sums and estimates the leading covariance spectrum without allocating a hidden-dimension covariance matrix.
- Both functions validate rank/dimensions and remain device-agnostic.

- [ ] **Step 1: Write failing tests** comparing small factorized LoRA results to a dense reference and exercising large feature dimensions with small rank; add a residual test that verifies finite output with more hidden dimensions than retained components.
- [ ] **Step 2: Run `uv run pytest code/tests/test_qwen3_adapter.py -q`** and confirm the current implementation materializes `B @ A`.
- [ ] **Step 3: Implement the rank-space Gram spectrum and matrix-free residual spectrum**; do not change the public result dataclasses.
- [ ] **Step 4: Run the focused tests, Ruff, and a small allocation smoke** with `uv run pytest code/tests/test_qwen3_adapter.py -q` and `uv run ruff check code/helix/architectures/qwen3_adapter.py code/tests/test_qwen3_adapter.py`.
- [ ] **Step 5: Commit** with `git commit -m "perf: factorize Qwen structural diagnostics"`.

### Task 6: Add reproducible PR validation

**Files:**
- Create: `scripts/validate_pr.py`
- Create: `scripts/__init__.py`
- Modify: `README.md`
- Modify: `docs/grok-watch.md`
- Modify: PR body through `gh pr edit`

**Interfaces:**
- `uv run python scripts/validate_pr.py` runs the full suite with the current interpreter, whole-repository Ruff, root wheel build, standalone `environments/helixenv` wheel build, and package-content assertions.
- Build output goes to a temporary directory; the validator reports the actual `N passed` count parsed from pytest output and returns nonzero on any failed command or missing wheel member.
- The validator performs no Studio calls by default. Live validation remains the explicit read-only command `uv run helix grok-watch --studio-url http://127.0.0.1:8888 --once --json`.

- [ ] **Step 1: Write a failing validator smoke test** by running the new command before the script exists and record the missing-file failure.
- [ ] **Step 2: Implement the command runner and wheel-content checks** using `subprocess`, `tempfile.TemporaryDirectory`, and `zipfile`; do not hard-code a test count.
- [ ] **Step 3: Run the validator** and confirm it prints the observed test count and exits zero.
- [ ] **Step 4: Update README, grok-watch docs, and PR wording** to call this a telemetry/phase-detection foundation and explicitly reserve calibrated grok-watch claims for a future behavioral-probe layer.
- [ ] **Step 5: Commit** with `git commit -m "test: add reproducible Temporal Helix validation"`.

### Task 7: Integration review and final verification

**Files:**
- Review all files changed by Tasks 1–6.

- [ ] **Step 1: Run `uv run python scripts/validate_pr.py` from the repository root.**
- [ ] **Step 2: Run the read-only live Studio smoke and verify the association, checkpoint stable/pending counts, hardware, and safety fields.**
- [ ] **Step 3: Run `rg -ni "codex|co-authored-by|generated by|chatgpt|ai-generated" .` and require no attribution markers.**
- [ ] **Step 4: Inspect `git diff --check`, `git status --short`, commit authors, and `gh pr view 9`.**
- [ ] **Step 5: Request two parallel final reviews, resolve all important findings, push the branch, and report exact validation output without claiming more than the evidence supports.**
