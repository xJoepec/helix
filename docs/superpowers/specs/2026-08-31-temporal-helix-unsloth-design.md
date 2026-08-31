# Temporal Helix + Keyless Unsloth Design

## Objective

Build a read-only `helix grok-watch` integration for Unsloth Studio on `127.0.0.1:8888`.
Unsloth owns run lifecycle and GPU state; Helix observes telemetry, discovers complete checkpoints,
computes versioned structural features, and reports candidate multivariate change points.

## Safety contract

- Accept loopback Studio URLs only.
- Send no authorization header and persist no credentials.
- Expose no start, stop, reset, shutdown, inference-load, or export mutation in the MVP.
- While training is active, collect telemetry only; never load a second model.
- Analyze only complete, stable checkpoint shard sets.
- Never label a Helix-only event as confirmed grokking; behavioral evidence is independent.

## Components

1. `StudioClient`: typed reads for runs, status, metrics, hardware, inference, and export state.
2. `CheckpointWatcher`: complete-checkpoint detection and deterministic fingerprints.
3. `Qwen3Diagnostics`: LoRA delta spectra, residual geometry, and SwiGLU gate statistics.
4. `TemporalPhaseDetector`: shrinkage-covariance Mahalanobis phase distance.
5. `GrokWatchService`: joins Studio telemetry and checkpoints for one-shot JSON/human CLI output.

## Data contracts

```python
class StudioClient:
    def list_runs(self, limit: int = 20, offset: int = 0) -> list[RunSummary]: ...
    def get_run(self, run_id: str) -> RunDetail: ...
    def get_status(self) -> TrainStatus: ...
    def get_metrics(self, job_id: str) -> TrainMetrics: ...
    def get_hardware(self) -> HardwareSnapshot: ...
    def get_inference_status(self) -> InferenceStatus: ...

class TemporalPhaseDetector:
    def fit(self, observations: Sequence[FeatureObservation]) -> None: ...
    def score(self, observation: FeatureObservation) -> PhaseScore: ...
```

Every observation carries run/checkpoint/step/probe identifiers, schema and adapter versions, timestamp,
and numeric features. Missing features stay missing. The detector uses
`(1-lambda)S + lambda*diag(S) + ridge*I` and reports only `insufficient_baseline`, `normal`, or
`candidate_transition`.

## CLI

```text
helix grok-watch --studio-url http://127.0.0.1:8888 --once
helix grok-watch --studio-url http://localhost:8888 --run-id job_... --json
```

Continuous polling, autonomous control, full checkpoint loading, behavioral confirmation, and external
Prime-RL rewards are follow-on subsystems, not MVP behavior.

## Acceptance criteria

- Remote/non-loopback URLs are rejected before transport use.
- Recorded Studio fixtures parse into immutable typed objects.
- Partial, moving, or traversal-based checkpoint shards are rejected.
- Temporal scoring remains finite with singular covariance.
- Qwen diagnostics agree between CPU and CUDA within tolerance when CUDA is available.
- `grok-watch --once --json` emits valid JSON against a fixture server and live keyless Studio.
- New files are Ruff-clean and scoped tests pass despite documented pre-existing repository failures.

