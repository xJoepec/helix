# Helix Overview

Helix provides operator‑algebraic diagnostics for deep networks:
- ReLU partitions across depth form AF inductive systems via incidence matrices B_k and masses τ_k.
- Nonlinear/channel‑mixing operations are modeled with unital CP maps Φ(X)=V* X V derived from (B_k, τ_k).
- Invertible/residual blocks are studied via Ulam–Perron–Frobenius discretizations and their spectral gaps.

Why this matters
- Interpretable, stable readouts: region growth, mass transport, positivity/unitality checks, mixing proxies.
- Scales from toy MLPs to modern stacks; numerically stable constructions; sparse memory footprint.

Core modules
- partitions: empirical extraction of ReLU partitions, incidences, masses.
- cp: build stable V and sanity‑check the induced CP map.
- ulam: Ulam PF with barycentric splitting; spectral analysis.
- ktheory: Smith normal form for toy K‑invariants of stationary systems.
- diagnostics: convenience metrics for training‑time monitoring.

Next steps
- Read docs/getting-started.md and docs/concepts.md.
- Use the CLI/TUI for quick exploration; then integrate the API in your experiments.

