Helix — Operator‑Algebraic Diagnostics for Deep Nets

Helix turns the “deep nets as manifold sculptors” picture into computable diagnostics using operator‑algebraic tools. It exposes stable invariants and practical readouts that scale from toy MLPs to modern architectures.

What Helix gives you
- AF partitions (ReLU): Extract refining partitions across depth, build incidence B_k, masses τ_k, and dimension‑group style summaries.
- CP embeddings (nonlinearity): Construct numerically stable unital CP maps Φ(X)=V* X V from incidence and masses; sanity‑check unitality/PSD and coisometry.
- Flow diagnostics (invertible blocks): Ulam–Perron–Frobenius discretization with barycentric mass splitting and multi‑samples per cell; read spectral gaps.
- Sparse structure: Parent‑pointer representation of refinements for O(n) memory and fast mass aggregation.
- CLI, TUI and plots: One‑command run to generate region, mass‑consistency, CP, and Ulam spectra plots; optional interactive TUI.

Install
- Python 3.10+
- Recommended: virtual environment

```
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -e .
# Optional extras
python3 -m pip install torch sympy matplotlib
```

Quickstart
- CLI (non‑interactive)
```
./helix \
  --no-train \
  --ulam-bins 25 \
  --ulam-samples-per-cell 4 \
  --plot --no-show \
  --save-prefix helix_out
```
Generates: `helix_out_regions.png`, `helix_out_mass_consistency.png`, `helix_out_cp.png`, `helix_out_ulam.png`.

- TUI (interactive)
```
python3 -m pip install -e .[tui]
helix tui  # or: helix-tui
```

- Programmatic API
```python
import numpy as np
import torch, torch.nn as nn
from helix import (
    extract_partitions, build_V_from_incidence, sanity_check_ucp,
    ulam_pf, spectral_gap, region_counts, mass_consistency_errors,
)

model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 2))
X = np.random.randn(2000, 2).astype(np.float32)

af = extract_partitions(model, X)
print("regions:", region_counts(af.B_list))
print("mass L1:", mass_consistency_errors(af.B_list, af.tau_list))

cp_stats = []
for k, B in enumerate(af.B_list, start=1):
    tau_prev = np.array([1.0]) if k == 1 else af.tau_list[k-2]
    tau_cur = af.tau_list[k-1]
    V = build_V_from_incidence(B, tau_prev, tau_cur)
    cp_stats.append(sanity_check_ucp(V))
print(cp_stats)

F = lambda x: x  # identity toy map
lo, hi = np.array([-1.0, -1.0]), np.array([1.0, 1.0])
P, _ = ulam_pf(F, (lo, hi), bins_per_dim=20, samples_per_cell=4)
print("spectral gap:", spectral_gap(P))
```

Project Layout
- code/helix: core library modules (partitions, cp, ulam, ktheory, diagnostics, sparse, plotting, TUI/CLI wrappers)
- code/examples: runnable demo
- environments/: verifiers/eval environments (`helixenv`, `bixbench`, `hle`)
- docs/: user and developer documentation (overview, concepts, API, CLI/TUI, troubleshooting)
- code/tests: unit tests (direct invocation supported; see docs)
- helix: local CLI helper to run the package from the repo without installing

Documentation
- Start here: docs/overview.md
- Concepts: docs/concepts.md
- Applications: docs/applications.md
- Getting started: docs/getting-started.md
- CLI: docs/cli.md, TUI: docs/tui.md
- API surface: docs/api.md
- Troubleshooting: docs/troubleshooting.md
- Environments: docs/environments.md
- Roadmap: docs/roadmap.md

License
MIT
