# Temporal Helix Telemetry Hardening

**Status:** Approved design

## Goal

Harden Temporal Helix telemetry so checkpoint observations are stable, run-scoped,
authenticated according to the current Unsloth Studio contract, provenance-safe,
and cheap enough for Qwen-scale adapters. Make the repository validation command
reproducible and stop short of claiming calibrated grokking detection.

## Current boundaries

- Unsloth Studio exposes `GET /api/train/status`, `GET /api/train/runs`,
  `GET /api/train/metrics?expected_job_id=...`, and `GET /api/inference/status`.
- The current loopback Studio accepts keyless reads, while its OpenAPI schema
  advertises an `HTTPBearer` security scheme.
- A run-list row includes `id`, `status`, model/dataset names, `total_steps`,
  and `output_dir`; direct run lookup wraps that row under `run` and returns
  sibling `config` and `metrics` objects. Metrics include `job_id`,
  step/loss/LR histories, and gradient history.
- QLoRA checkpoints use `adapter_model.safetensors` plus `trainer_state.json`;
  full models may use one file or an indexed set of shards.

## Design

### 1. Stable checkpoint manifests

`CheckpointWatcher` will validate a checkpoint entirely beneath its selected run
directory. It will require a valid trainer state and a complete model/adapter
artifact set, reject temporary or path-escaping files, and verify that the saved
global step agrees with `checkpoint-<step>` when the trainer state provides it.

Each result will carry a manifest fingerprint based on relative file names,
sizes, and nanosecond modification times. A checkpoint whose required files are
younger than the configured settling interval is marked pending; callers can
also compare fingerprints across polls. Changing or too-recent artifacts are
not fed into phase analysis. The watcher will never mutate checkpoints.

### 2. Explicit run binding

Snapshot construction will resolve one selected run before reading telemetry.
An implicit selection follows the current `status.job_id` when present; an
explicit `--run-id` always uses direct lookup. Live status fields are consumed
only when their `job_id` equals the selected run. Metrics are accepted only when
their returned `job_id` matches; otherwise persisted run history is used and the
snapshot records an association mismatch instead of silently mixing runs.

### 3. Current authentication modes

`StudioClient` will support two read-only modes:

- `keyless` (default for loopback): send no bearer header, matching the current
  local Studio behavior.
- `bearer`: use an explicit token or `UNSLOTH_STUDIO_TOKEN` and send
  `Authorization: Bearer ...`.

The CLI will expose `--studio-token` without printing or persisting its value.
All modes retain strict loopback URL validation; non-loopback URLs remain
rejected. Documentation will distinguish observed keyless behavior from the
OpenAPI-declared bearer contract.

### 4. Provenance-enforced phase observations

Feature observations will include a structured provenance record containing the
run ID, checkpoint ID and fingerprint, probe-set ID, schema version, and feature
source. A detector baseline must contain one consistent run/probe/schema/source
identity. Scoring rejects missing or mismatched provenance as
`insufficient_data`; it never produces a phase distance for an untrusted mix of
runs, checkpoints, probes, or feature schemas.

### 5. Factorized Qwen diagnostics

LoRA diagnostics will compute the nonzero singular spectrum of the low-rank
update from rank-space Gram matrices. Frobenius norm, effective rank, and
spectral entropy will therefore use `O(r²)` memory and `O(r³)` spectral work,
without materializing the `out_features × in_features` delta. Small dense
reference tests will establish numerical equivalence.

### 6. Reproducible validation

`scripts/validate_pr.py` will run, in a temporary output directory, the complete
pytest suite, whole-repository Ruff, root and standalone environment wheel
builds, and wheel-content assertions. It will be the only validation command
named by the PR claim. Live Studio validation remains an explicit read-only
smoke command and is not required for offline reproducibility.

## Error and safety behavior

- No training, inference load/unload, export, stop, or shutdown operation is
  added to the client.
- HTTP/authentication failures are concise and redact URLs/query values where
  appropriate.
- Missing, changing, or mismatched telemetry is represented explicitly rather
  than filled with current global state.
- The CLI and validation script return nonzero on invalid association,
  incomplete artifacts, failed tests, lint errors, or build errors.

## Acceptance criteria

1. Complete and changing checkpoint fixtures are distinguished deterministically.
2. A selected historical run cannot inherit status or metrics from another run.
3. Both keyless and bearer requests are tested without exposing bearer values.
4. Cross-run, cross-probe, cross-schema, and missing-provenance observations are
   rejected by the detector.
5. Qwen diagnostics do not allocate a dense delta proportional to model matrix
   dimensions and match a dense reference on small matrices.
6. `uv run python scripts/validate_pr.py` passes from a clean checkout and its
   output reports the actual test count.
7. Documentation and PR wording describe a telemetry/phase-detection
   foundation, not a validated grokking detector.
