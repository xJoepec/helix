# Grok Watch

`helix grok-watch` reads a local Unsloth Studio instance on loopback and emits a one-shot training
snapshot. It is deliberately read-only: it cannot start, stop, reset, load, unload, export, or shut down
Studio.

```bash
helix grok-watch --studio-url http://127.0.0.1:8888 --once
helix grok-watch --studio-url http://localhost:8888 --run-id job_... --json
UNSLOTH_STUDIO_TOKEN=... helix grok-watch --studio-auth bearer --once
```

Only `127.0.0.1`, `localhost`, and `::1` are accepted. When training is active, the output marks
`inference_allowed` false. Do not run model inference alongside training on a shared GPU.

Current loopback Studio installations have been observed accepting keyless GET telemetry requests, even
though the Studio OpenAPI specification advertises `HTTPBearer`. `--studio-auth auto` (the default) uses
keyless requests when no token is available, and uses bearer authentication when `UNSLOTH_STUDIO_TOKEN` or
`--studio-token` supplies one. Use `--studio-auth keyless` to force the observed keyless behavior, or
`--studio-auth bearer` to require a token. The token is sent only in the `Authorization` header, remains in
memory, and is never placed in URLs or telemetry output.

The command currently reports Studio telemetry and safety state. Temporal phase scoring requires complete,
internally consistent provenance for the source, run ID, checkpoint ID and fingerprint, probe set ID, and
schema version. A detector baseline and its scored observation must share source, run, probe-set, and schema
identity; later checkpoints from that same run remain scoreable when they carry their own provenance.

Temporal checkpoint features and behavioral confirmation are separate APIs. Neither a telemetry snapshot nor a
phase-distance candidate is evidence of grokking.

## Reproducible validation and claims

Run the checked-in offline validator before making a merge claim:

```bash
uv run python scripts/validate_pr.py
```

It runs the full test suite, whole-repository Ruff, both wheel builds, and wheel-content checks. It does not
contact Studio. If live read-only telemetry validation is needed, run it explicitly:

```bash
uv run helix grok-watch --studio-url http://127.0.0.1:8888 --once --json
```

Current work is telemetry/phase-detection groundwork. A calibrated grok-watch claim is reserved for future
behavioral probes.
