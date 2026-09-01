# Grok Watch

`helix grok-watch` reads a keyless Unsloth Studio instance on loopback and emits a one-shot training
snapshot. It is deliberately read-only: it cannot start, stop, reset, load, unload, export, or shut down
Studio.

```bash
helix grok-watch --studio-url http://127.0.0.1:8888 --once
helix grok-watch --studio-url http://localhost:8888 --run-id job_... --json
```

Only `127.0.0.1`, `localhost`, and `::1` are accepted. When training is active, the output marks
`inference_allowed` false. Do not run model inference alongside training on a shared GPU.

The command currently reports Studio telemetry and safety state. Temporal checkpoint features and
behavioral confirmation are separate APIs; a telemetry snapshot is not evidence of grokking.
